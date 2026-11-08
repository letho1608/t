# std/cudnn.t — GPU 2D Convolution (NCHW) qua NVIDIA cuDNN + CPU Fallback êm (M6)
#
# Nạp động thư viện cuDNN (`cudnn64_9.dll` / `cudnn64_8.dll` / `libcudnn.so.9` / `libcudnn.so.8`)
# và CUDA Runtime (`cudart`) qua FFI để thực thi phép tích chập 2 chiều `conv2d(x, w, bias, stride, pad)`
# trực tiếp trên GPU NVIDIA (`cudnnConvolutionForward`, định dạng `CUDNN_TENSOR_NCHW`, `CUDNN_DATA_DOUBLE`).
#
# Khi máy không có GPU NVIDIA hoặc không cài cuDNN: tự động kích hoạt Graceful CPU Fallback
# về nhân C native `x.conv2d(w, bias, stride, pad)` trong `vm/tensor_impl.h` mà không ném lỗi.

let ffi = load("std/ffi")
let _sys = load("__sys")

fn _cudart_names():
    let names = []
    let plat = _sys.platform()
    if plat == "windows" or plat == "win32":
        names.append("cudart64_13.dll")
        names.append("cudart64_12.dll")
        names.append("cudart64_11.dll")
        names.append("cudart64.dll")
    else:
        names.append("libcudart.so.13")
        names.append("libcudart.so.12")
        names.append("libcudart.so.11")
        names.append("libcudart.so")
    return names

fn _cudnn_names():
    let names = []
    let plat = _sys.platform()
    if plat == "windows" or plat == "win32":
        names.append("cudnn_cnn64_9.dll")
        names.append("cudnn64_9.dll")
        names.append("cudnn64_8.dll")
        names.append("cudnn64_7.dll")
        names.append("cudnn64.dll")
    else:
        names.append("libcudnn_cnn.so.9")
        names.append("libcudnn.so.9")
        names.append("libcudnn.so.8")
        names.append("libcudnn.so.7")
        names.append("libcudnn.so")
    return names

fn _open_first(names):
    for n in names:
        let lib = ffi.open(n)
        if lib != nil:
            return lib
    return nil

struct CudnnCtx:
    cudart
    dnn
    fn_malloc
    fn_free
    fn_memcpy
    fn_create
    fn_create_tdesc
    fn_set_tdesc4d
    fn_destroy_tdesc
    fn_create_fdesc
    fn_set_fdesc4d
    fn_destroy_fdesc
    fn_create_cdesc
    fn_set_cdesc2d
    fn_destroy_cdesc
    fn_conv_fwd
    handle

fn _init_cudnn():
    let cudart = _open_first(_cudart_names())
    let dnn = _open_first(_cudnn_names())
    if cudart == nil or dnn == nil:
        return nil
    try:
        let fn_malloc = cudart.get_fn("cudaMalloc", "i", "pl")
        let fn_free = cudart.get_fn("cudaFree", "i", "p")
        let fn_memcpy = cudart.get_fn("cudaMemcpy", "i", "pli")
        let fn_get_count = cudart.get_fn("cudaGetDeviceCount", "i", "p")
        let fn_create = dnn.get_fn("cudnnCreate", "i", "p")
        let fn_create_tdesc = dnn.get_fn("cudnnCreateTensorDescriptor", "i", "p")
        let fn_set_tdesc4d = dnn.get_fn("cudnnSetTensor4dDescriptor", "i", "piiiiii")
        let fn_destroy_tdesc = dnn.get_fn("cudnnDestroyTensorDescriptor", "i", "p")
        let fn_create_fdesc = dnn.get_fn("cudnnCreateFilterDescriptor", "i", "p")
        let fn_set_fdesc4d = dnn.get_fn("cudnnSetFilter4dDescriptor", "i", "piiiiii")
        let fn_destroy_fdesc = dnn.get_fn("cudnnDestroyFilterDescriptor", "i", "p")
        let fn_create_cdesc = dnn.get_fn("cudnnCreateConvolutionDescriptor", "i", "p")
        let fn_set_cdesc2d = dnn.get_fn("cudnnSetConvolution2dDescriptor", "i", "piiiiiiii")
        let fn_destroy_cdesc = dnn.get_fn("cudnnDestroyConvolutionDescriptor", "i", "p")
        let fn_conv_fwd = dnn.get_fn("cudnnConvolutionForward", "i", "ppppppipppp")

        if fn_malloc == nil or fn_free == nil or fn_memcpy == nil or fn_get_count == nil:
            return nil
        if fn_create == nil or fn_create_tdesc == nil or fn_set_tdesc4d == nil or fn_conv_fwd == nil:
            return nil

        let cnt_buf = ffi.buffer(4)
        let rc = fn_get_count.call(cnt_buf.ptr)
        let cnt = cnt_buf.read_i32()
        cnt_buf.free()
        if rc != 0 or cnt <= 0:
            return nil

        let h_buf = ffi.buffer(8)
        let rc2 = fn_create.call(h_buf.ptr)
        let h = h_buf.read_ptr()
        h_buf.free()
        if rc2 != 0 or h == 0:
            return nil

        return CudnnCtx(
            cudart, dnn, fn_malloc, fn_free, fn_memcpy, fn_create,
            fn_create_tdesc, fn_set_tdesc4d, fn_destroy_tdesc,
            fn_create_fdesc, fn_set_fdesc4d, fn_destroy_fdesc,
            fn_create_cdesc, fn_set_cdesc2d, fn_destroy_cdesc,
            fn_conv_fwd, h
        )
    except:
        return nil

let _ctx = _init_cudnn()

fn is_available():
    return _ctx != nil

fn backend():
    if _ctx == nil:
        return "cpu"
    return "cudnn"

fn _add_bias_nchw(out, bias, n, cout, hout, wout):
    if bias == nil:
        return out
    let pout = out.data_ptr()
    let pb = bias.data_ptr()
    let hw = hout * wout
    mut b = 0
    while b < n:
        mut c = 0
        while c < cout:
            let bval = ffi.read_f64(pb + c * 8)
            let base = ((b * cout + c) * hw) * 8
            mut idx = 0
            while idx < hw:
                let cur = ffi.read_f64(pout + base + idx * 8)
                ffi.write_f64(pout + base + idx * 8, cur + bval)
                idx = idx + 1
            c = c + 1
        b = b + 1
    return out

fn conv2d(x, w, bias = nil, stride = 1, pad = 0):
    if _ctx == nil:
        return x.conv2d(w, bias, stride, pad)
    if x.ndim != 4 or w.ndim != 4:
        return x.conv2d(w, bias, stride, pad)

    let ctx = _ctx
    let n = x.shape[0]
    let cin = x.shape[1]
    let h = x.shape[2]
    let wi = x.shape[3]
    let cout = w.shape[0]
    let kh = w.shape[2]
    let kw = w.shape[3]
    let hout = int((h + 2 * pad - kh) / stride) + 1
    let wout = int((wi + 2 * pad - kw) / stride) + 1
    if hout <= 0 or wout <= 0:
        return x.conv2d(w, bias, stride, pad)

    mut dx = 0
    mut dw = 0
    mut dy = 0
    mut xdesc = 0
    mut wdesc = 0
    mut ydesc = 0
    mut cdesc = 0
    try:
        let bx = n * cin * h * wi * 8
        let bw = cout * cin * kh * kw * 8
        let by = n * cout * hout * wout * 8

        let pbuf = ffi.buffer(8)
        if ctx.fn_malloc.call(pbuf.ptr, bx) != 0:
            pbuf.free()
            return x.conv2d(w, bias, stride, pad)
        dx = pbuf.read_ptr()
        if ctx.fn_malloc.call(pbuf.ptr, bw) != 0:
            pbuf.free()
            ctx.fn_free.call(dx)
            return x.conv2d(w, bias, stride, pad)
        dw = pbuf.read_ptr()
        if ctx.fn_malloc.call(pbuf.ptr, by) != 0:
            pbuf.free()
            ctx.fn_free.call(dx)
            ctx.fn_free.call(dw)
            return x.conv2d(w, bias, stride, pad)
        dy = pbuf.read_ptr()

        ctx.fn_memcpy.call(dx, x.data_ptr(), bx, 1)
        ctx.fn_memcpy.call(dw, w.data_ptr(), bw, 1)

        ctx.fn_create_tdesc.call(pbuf.ptr)
        xdesc = pbuf.read_ptr()
        ctx.fn_create_tdesc.call(pbuf.ptr)
        ydesc = pbuf.read_ptr()
        ctx.fn_create_fdesc.call(pbuf.ptr)
        wdesc = pbuf.read_ptr()
        ctx.fn_create_cdesc.call(pbuf.ptr)
        cdesc = pbuf.read_ptr()
        pbuf.free()

        # CUDNN_TENSOR_NCHW = 0, CUDNN_DATA_DOUBLE = 1, CUDNN_CROSS_CORRELATION = 1
        ctx.fn_set_tdesc4d.call(xdesc, 0, 1, n, cin, h, wi)
        ctx.fn_set_fdesc4d.call(wdesc, 1, 0, cout, cin, kh, kw)
        ctx.fn_set_cdesc2d.call(cdesc, pad, pad, stride, stride, 1, 1, 1, 1)
        ctx.fn_set_tdesc4d.call(ydesc, 0, 1, n, cout, hout, wout)

        let alpha = ffi.buffer(8)
        alpha.write_f64(1.0)
        let beta = ffi.buffer(8)
        beta.write_f64(0.0)
        # CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM = 0 (không cần workspace)
        let rc = ctx.fn_conv_fwd.call(ctx.handle, alpha.ptr, xdesc, dx, wdesc, dw, cdesc, 0, nil, 0, beta.ptr, ydesc, dy)
        alpha.free()
        beta.free()

        if rc != 0:
            ctx.fn_destroy_tdesc.call(xdesc)
            ctx.fn_destroy_tdesc.call(ydesc)
            ctx.fn_destroy_fdesc.call(wdesc)
            ctx.fn_destroy_cdesc.call(cdesc)
            ctx.fn_free.call(dx)
            ctx.fn_free.call(dw)
            ctx.fn_free.call(dy)
            return x.conv2d(w, bias, stride, pad)

        let out = zeros([n, cout, hout, wout])
        ctx.fn_memcpy.call(out.data_ptr(), dy, by, 2)

        ctx.fn_destroy_tdesc.call(xdesc)
        ctx.fn_destroy_tdesc.call(ydesc)
        ctx.fn_destroy_fdesc.call(wdesc)
        ctx.fn_destroy_cdesc.call(cdesc)
        ctx.fn_free.call(dx)
        ctx.fn_free.call(dw)
        ctx.fn_free.call(dy)

        if bias != nil:
            return _add_bias_nchw(out, bias, n, cout, hout, wout)
        return out
    except:
        return x.conv2d(w, bias, stride, pad)
