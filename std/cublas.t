# std/cublas.t — GPU Tensor Offloading qua cuBLAS & cuDNN (M6 Hoàn chỉnh) + CPU Fallback êm
#
# Tính năng:
#   1. `gemm(a, b)` / `matmul(a, b)`: GEMM chuẩn double-precision (`cublasDgemm_v2`).
#   2. `gemm_ex(a, b, mode)` / `gemm_fp16(a, b)` / `gemm_bf16(a, b)` / `gemm_f32(a, b)`:
#      Hỗ trợ NVIDIA Tensor Cores (`cublasSetMathMode` + `cublasGemmEx` 19 đối số qua FFI)
#      với các chế độ `"fp16"` (`CUDA_R_16F`), `"bf16"` (`CUDA_R_16BF`), `"fp32"` (`CUDA_R_32F`),
#      `"fp64"` (`CUDA_R_64F`) và tích lũy `CUBLAS_COMPUTE_32F_FAST_16F` / `CUBLAS_COMPUTE_32F`.
#   3. `conv2d(x, w, bias = nil, stride = 1, pad = 0)`: Tích chập 2D NCHW qua cuDNN (`std/cudnn.t`)
#      hoặc nhân C native `x.conv2d(w, bias, stride, pad)`.
#   4. Máy không có GPU NVIDIA / driver / cuBLAS / cuDNN: tự động kích hoạt Graceful CPU
#      Fallback (mô phỏng chuẩn xác bước lượng tử hóa FP16 / BF16 / FP32 IEEE-754 trước khi nhân `@`).

let ffi = load("std/ffi")
let cudnn = load("std/cudnn")
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

fn _cublas_names():
    let names = []
    let plat = _sys.platform()
    if plat == "windows" or plat == "win32":
        names.append("cublas64_13.dll")
        names.append("cublas64_12.dll")
        names.append("cublas64_11.dll")
        names.append("cublas64.dll")
    else:
        names.append("libcublas.so.13")
        names.append("libcublas.so.12")
        names.append("libcublas.so.11")
        names.append("libcublas.so")
    return names

fn _open_first(names):
    for n in names:
        let lib = ffi.open(n)
        if lib != nil:
            return lib
    return nil

struct CublasCtx:
    cudart
    blas
    fn_malloc
    fn_free
    fn_memcpy
    fn_get_count
    fn_create
    fn_dgemm
    fn_gemm_ex
    fn_set_math
    handle
    dev_count

fn _init_cublas():
    let cudart = _open_first(_cudart_names())
    let blas = _open_first(_cublas_names())
    if cudart == nil or blas == nil:
        return nil
    try:
        let fn_malloc = cudart.get_fn("cudaMalloc", "i", "pl")
        let fn_free = cudart.get_fn("cudaFree", "i", "p")
        let fn_memcpy = cudart.get_fn("cudaMemcpy", "i", "pli")
        let fn_get_count = cudart.get_fn("cudaGetDeviceCount", "i", "pi")
        let fn_create = blas.get_fn("cublasCreate_v2", "i", "p")
        let fn_dgemm = blas.get_fn("cublasDgemm_v2", "i", "piiiiippipippi")
        let fn_gemm_ex = blas.get_fn("cublasGemmEx", "i", "piiiiippipiiipipiii")
        let fn_set_math = blas.get_fn("cublasSetMathMode", "i", "pi")
        if fn_malloc == nil or fn_free == nil or fn_memcpy == nil:
            return nil
        if fn_get_count == nil or fn_create == nil or fn_dgemm == nil:
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

        if fn_set_math != nil:
            # CUBLAS_TENSOR_OP_MATH = 1
            fn_set_math.call(h, 1)

        return CublasCtx(
            cudart, blas, fn_malloc, fn_free, fn_memcpy,
            fn_get_count, fn_create, fn_dgemm, fn_gemm_ex, fn_set_math,
            h, cnt
        )
    except:
        return nil

let _ctx = _init_cublas()

fn is_available():
    return _ctx != nil

fn backend():
    if _ctx == nil:
        return "cpu"
    return "cublas"

fn device_count():
    if _ctx == nil:
        return 0
    return _ctx.dev_count

fn _free_all(ctx, devs):
    for p in devs:
        if p != nil:
            ctx.fn_free.call(p)

fn _alloc(ctx, devs, nbytes):
    let buf = ffi.buffer(8)
    let rc = ctx.fn_malloc.call(buf.ptr, nbytes)
    let p = buf.read_ptr()
    buf.free()
    if rc != 0 or p == 0:
        return nil
    devs.append(p)
    return p

fn gemm(a, b):
    let ctx = _ctx
    if ctx == nil:
        return a @ b
    if a.ndim != 2 or b.ndim != 2:
        return a @ b
    let m = a.shape[0]
    let k = a.shape[1]
    let n = b.shape[1]
    if k != b.shape[0] or m <= 0 or n <= 0 or k <= 0:
        return a @ b

    let devs = []
    try:
        let bytes_a = m * k * 8
        let bytes_b = k * n * 8
        let bytes_c = m * n * 8

        let da = _alloc(ctx, devs, bytes_a)
        let db = _alloc(ctx, devs, bytes_b)
        let dc = _alloc(ctx, devs, bytes_c)
        let dalpha = _alloc(ctx, devs, 8)
        let dbeta = _alloc(ctx, devs, 8)
        if da == nil or db == nil or dc == nil or dalpha == nil or dbeta == nil:
            _free_all(ctx, devs)
            return a @ b

        let ha = ffi.buffer(8)
        ha.write_f64(1.0)
        let hb = ffi.buffer(8)
        hb.write_f64(0.0)
        let r1 = ctx.fn_memcpy.call(dalpha, ha.ptr, 8, 1)
        let r2 = ctx.fn_memcpy.call(dbeta, hb.ptr, 8, 1)
        let r3 = ctx.fn_memcpy.call(da, a.data_ptr(), bytes_a, 1)
        let r4 = ctx.fn_memcpy.call(db, b.data_ptr(), bytes_b, 1)
        ha.free()
        hb.free()
        if r1 != 0 or r2 != 0 or r3 != 0 or r4 != 0:
            _free_all(ctx, devs)
            return a @ b

        # C^T (N x M) = B^T @ A^T trên buffer row-major
        let rc = ctx.fn_dgemm.call(ctx.handle, 0, 0, n, m, k, dalpha, db, n, da, k, dbeta, dc, n)
        if rc != 0:
            _free_all(ctx, devs)
            return a @ b

        let c = zeros([m, n])
        let r5 = ctx.fn_memcpy.call(c.data_ptr(), dc, bytes_c, 2)
        if r5 != 0:
            _free_all(ctx, devs)
            return a @ b

        _free_all(ctx, devs)
        return c
    except:
        _free_all(ctx, devs)
        return a @ b

fn _cpu_quant_gemm(a, b, mode):
    if mode == "fp64":
        return a @ b
    if mode == "fp16":
        let qa = _sys.dequant_f16(_sys.quant_f16(a), a.shape)
        let qb = _sys.dequant_f16(_sys.quant_f16(b), b.shape)
        let c = qa @ qb
        return _sys.dequant_f32(_sys.quant_f32(c), c.shape)
    if mode == "bf16":
        let qa = _sys.dequant_bf16(_sys.quant_bf16(a), a.shape)
        let qb = _sys.dequant_bf16(_sys.quant_bf16(b), b.shape)
        let c = qa @ qb
        return _sys.dequant_f32(_sys.quant_f32(c), c.shape)
    if mode == "fp32" or mode == "f32":
        let qa = _sys.dequant_f32(_sys.quant_f32(a), a.shape)
        let qb = _sys.dequant_f32(_sys.quant_f32(b), b.shape)
        let c = qa @ qb
        return _sys.dequant_f32(_sys.quant_f32(c), c.shape)
    return a @ b

fn gemm_ex(a, b, mode = "fp16"):
    if mode == "fp64":
        return gemm(a, b)
    let ctx = _ctx
    if ctx == nil or ctx.fn_gemm_ex == nil:
        return _cpu_quant_gemm(a, b, mode)
    if a.ndim != 2 or b.ndim != 2:
        return _cpu_quant_gemm(a, b, mode)
    let m = a.shape[0]
    let k = a.shape[1]
    let n = b.shape[1]
    if k != b.shape[0] or m <= 0 or n <= 0 or k <= 0:
        return _cpu_quant_gemm(a, b, mode)

    let devs = []
    try:
        # CUDA_R_32F = 0, CUDA_R_16F = 2, CUDA_R_16BF = 14
        # CUBLAS_COMPUTE_32F = 68, CUBLAS_GEMM_DEFAULT_TENSOR_OP = 99
        mut dtype_ab = 2
        mut elem_sz = 2
        mut qa_list = nil
        mut qb_list = nil
        if mode == "fp16":
            dtype_ab = 2
            elem_sz = 2
            qa_list = _sys.quant_f16(a)
            qb_list = _sys.quant_f16(b)
        elif mode == "bf16":
            dtype_ab = 14
            elem_sz = 2
            qa_list = _sys.quant_bf16(a)
            qb_list = _sys.quant_bf16(b)
        else:
            dtype_ab = 0
            elem_sz = 4

        let mk = m * k
        let kn = k * n
        let mn = m * n
        let bytes_a = mk * elem_sz
        let bytes_b = kn * elem_sz
        let bytes_c = mn * 4

        let da = _alloc(ctx, devs, bytes_a)
        let db = _alloc(ctx, devs, bytes_b)
        let dc = _alloc(ctx, devs, bytes_c)
        let dalpha = _alloc(ctx, devs, 4)
        let dbeta = _alloc(ctx, devs, 4)
        if da == nil or db == nil or dc == nil or dalpha == nil or dbeta == nil:
            _free_all(ctx, devs)
            return _cpu_quant_gemm(a, b, mode)

        let ha = ffi.buffer(bytes_a)
        let hb = ffi.buffer(bytes_b)
        if elem_sz == 2:
            mut i = 0
            while i < mk:
                ffi.write_u16(ha.ptr + i * 2, qa_list[i])
                i = i + 1
            i = 0
            while i < kn:
                ffi.write_u16(hb.ptr + i * 2, qb_list[i])
                i = i + 1
        else:
            let pa = a.data_ptr()
            let pb = b.data_ptr()
            mut i = 0
            while i < mk:
                ffi.write_f32(ha.ptr + i * 4, ffi.read_f64(pa + i * 8))
                i = i + 1
            i = 0
            while i < kn:
                ffi.write_f32(hb.ptr + i * 4, ffi.read_f64(pb + i * 8))
                i = i + 1

        let halpha = ffi.buffer(4)
        ffi.write_f32(halpha.ptr, 1.0)
        let hbeta = ffi.buffer(4)
        ffi.write_f32(hbeta.ptr, 0.0)

        ctx.fn_memcpy.call(dalpha, halpha.ptr, 4, 1)
        ctx.fn_memcpy.call(dbeta, hbeta.ptr, 4, 1)
        ctx.fn_memcpy.call(da, ha.ptr, bytes_a, 1)
        ctx.fn_memcpy.call(db, hb.ptr, bytes_b, 1)
        ha.free()
        hb.free()
        halpha.free()
        hbeta.free()

        # cublasGemmEx(handle, N, N, n, m, k, alpha, db, dtype_ab, n, da, dtype_ab, k, beta, dc, CUDA_R_32F=0, n, CUBLAS_COMPUTE_32F=68, CUBLAS_GEMM_DEFAULT_TENSOR_OP=99)
        let rc = ctx.fn_gemm_ex.call(
            ctx.handle, 0, 0, n, m, k,
            dalpha, db, dtype_ab, n,
            da, dtype_ab, k,
            dbeta, dc, 0, n,
            68, 99
        )
        if rc != 0:
            _free_all(ctx, devs)
            return _cpu_quant_gemm(a, b, mode)

        let hc = ffi.buffer(bytes_c)
        let r5 = ctx.fn_memcpy.call(hc.ptr, dc, bytes_c, 2)
        if r5 != 0:
            hc.free()
            _free_all(ctx, devs)
            return _cpu_quant_gemm(a, b, mode)

        let c = zeros([m, n])
        let pc = c.data_ptr()
        mut idx = 0
        while idx < mn:
            ffi.write_f64(pc + idx * 8, ffi.read_f32(hc.ptr + idx * 4))
            idx = idx + 1
        hc.free()
        _free_all(ctx, devs)
        return c
    except:
        _free_all(ctx, devs)
        return _cpu_quant_gemm(a, b, mode)

fn gemm_fp16(a, b):
    return gemm_ex(a, b, "fp16")

fn gemm_bf16(a, b):
    return gemm_ex(a, b, "bf16")

fn gemm_f32(a, b):
    return gemm_ex(a, b, "fp32")

fn conv2d(x, w, bias = nil, stride = 1, pad = 0):
    return cudnn.conv2d(x, w, bias, stride, pad)

fn matmul(a, b):
    return gemm(a, b)
