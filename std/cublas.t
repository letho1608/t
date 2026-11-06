# std/cublas.t — GPU Tensor Offloading qua cuBLAS (M6) + CPU Fallback êm
#
# Dùng std/ffi.t (FFI lên tới 16 đối số) + tensor.data_ptr() để đưa buffer
# row-major sang GPU NVIDIA, tính GEMM double-precision bằng cuBLAS
# (cublasDgemm_v2), rồi copy kết quả về tensor host.
#
# Row-major vs column-major: tensor T là row-major, cuBLAS tính column-major.
# Với A (M x K), B (K x N): cùng byte buffer, cứ xem như ma trận column-major
# thì C^T (N x M) = B^T @ A^T, tức gọi
#   cublasDgemm_v2(handle, N, N, N, M, K, alpha, B, N, A, K, beta, C, N)
# với B/A/C là con trỏ device của đúng buffer row-major.
#
# KHÔNG có GPU / driver / lib cuBLAS / cudart, hoặc bất kỳ bước nào lỗi:
# tự động rơi về phép @ CPU (TVM GEMM đa luồng, xem TODO T2) — êm, không raise.
#
# CẢNH BÁO UNSAFE: module này dùng FFI + con trỏ thô (xem std/ffi.t).
# Write kết quả về tensor host qua data_ptr() chỉ an toàn vì C vừa được tạo
# (chưa có view nào dùng chung). Không dùng data_ptr() cho tensor có view.

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

        return CublasCtx(cudart, blas, fn_malloc, fn_free, fn_memcpy, fn_get_count, fn_create, fn_dgemm, h, cnt)
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

        # C^T (N x M) = B^T @ A^T trên buffer row-major (xem header)
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

fn matmul(a, b):
    return gemm(a, b)
