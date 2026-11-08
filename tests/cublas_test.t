# tests/cublas_test.t — Kiểm thử module std/cublas, std/cudnn & Tensor Cores (M6 Hoàn chỉnh)
# Máy không có GPU NVIDIA: kiểm tra CPU fallback êm + đúng (FP64 khớp @ từng bit,
# FP16/BF16/FP32 mô phỏng chuẩn IEEE-754, conv2d chạy nhân C native + autograd backward).
# Máy có GPU NVIDIA: kiểm tra cublasDgemm_v2, cublasGemmEx (Tensor Cores FP16/BF16), và cuDNN conv2d.

let cublas = load("std/cublas")
let cudnn = load("std/cudnn")
let quant = load("std/quant")
let ag = load("std/autograd")
let ffi = load("std/ffi")

fn test_mem_primitives():
    let p = ffi.alloc(32)
    assert p != 0
    ffi.write_f64(p, 3.141592653589793)
    assert ffi.read_f64(p) == 3.141592653589793
    ffi.write_f64(p, -2.5)
    assert ffi.read_f64(p) == -2.5

    ffi.write_f32(p + 8, 1.5)
    assert ffi.read_f32(p + 8) == 1.5
    ffi.write_u16(p + 12, 0x3C00)
    assert ffi.read_u16(p + 12) == 0x3C00

    ffi.free(p)
    print("mem_primitives (f64/f32/u16) round-trip OK")

fn test_data_ptr():
    let t = tensor([[1.0, 2.0], [3.0, 4.0]])
    let p = t.data_ptr()
    assert type(p) == "int" and p != 0
    assert ffi.read_f64(p) == 1.0
    assert ffi.read_f64(p + 8) == 2.0
    assert ffi.read_f64(p + 16) == 3.0
    assert ffi.read_f64(p + 24) == 4.0
    print("data_ptr OK")

fn test_gemm_and_tensor_cores():
    let m = 6
    let k = 8
    let n = 5
    mut A = zeros([m, k])
    mut B = zeros([k, n])
    for i in range(m):
        for q in range(k):
            A[i, q] = ((i * 5 + q * 3) % 17) * 0.1 - 0.8
    for q in range(k):
        for j in range(n):
            B[q, j] = ((q * 7 + j * 2) % 13) * 0.05 - 0.3
    let ref = A @ B
    let got = cublas.gemm(A, B)
    assert got.shape == [m, n]
    let d = (got - ref).abs().max()
    assert d < 0.000000001, "cuBLAS GEMM lech @: " + str(d)
    let got2 = cublas.matmul(A, B)
    assert (got2 - ref).abs().max() < 0.000000001

    # Kiểm thử Tensor Core FP16 / BF16 / FP32 (cublasGemmEx + quantize_bf16)
    let q_bf16 = quant.quantize_bf16(A)
    let dq_bf16 = quant.dequantize_bf16(q_bf16, A.shape)
    assert (dq_bf16 - A).abs().max() < 0.01

    let got_f32 = cublas.gemm_f32(A, B)
    let got_fp16 = cublas.gemm_fp16(A, B)
    let got_bf16 = cublas.gemm_bf16(A, B)
    assert got_f32.shape == [m, n]
    assert got_fp16.shape == [m, n]
    assert got_bf16.shape == [m, n]
    let d_f32 = (got_f32 - ref).abs().max()
    let d_fp16 = (got_fp16 - ref).abs().max()
    let d_bf16 = (got_bf16 - ref).abs().max()
    assert d_f32 < 0.00001, "gemm_f32 diff too large: " + str(d_f32)
    assert d_fp16 < 0.005, "gemm_fp16 diff too large: " + str(d_fp16)
    assert d_bf16 < 0.02, "gemm_bf16 diff too large: " + str(d_bf16)
    print("gemm & Tensor Cores (FP64/FP32/FP16/BF16) OK (fp16 diff=" + str(d_fp16) + ", bf16 diff=" + str(d_bf16) + ")")

fn test_conv2d_and_autograd():
    # Kiểm thử conv2d NCHW: X [1, 2, 4, 4], W [2, 2, 3, 3], bias [2], stride=1, pad=1 -> Y [1, 2, 4, 4]
    mut x = zeros([1, 2, 4, 4])
    mut w = zeros([2, 2, 3, 3])
    let b = tensor([0.25, -0.5])
    let px = x.data_ptr()
    let pw = w.data_ptr()
    for i in range(32):
        ffi.write_f64(px + i * 8, ((i * 7 + 3) % 11) * 0.1 - 0.5)
    for i in range(36):
        ffi.write_f64(pw + i * 8, ((i * 5 + 1) % 9) * 0.05 - 0.2)

    let y_native = x.conv2d(w, b, 1, 1)
    let y_cudnn = cudnn.conv2d(x, w, b, 1, 1)
    let y_cublas = cublas.conv2d(x, w, b, 1, 1)
    assert y_native.shape == [1, 2, 4, 4]
    assert (y_cudnn - y_native).abs().max() < 0.000000001
    assert (y_cublas - y_native).abs().max() < 0.000000001

    # Kiểm thử đạo hàm tự động autograd.conv2d so với sai phân hữu hạn (finite difference)
    let vx = ag.param(x)
    let vw = ag.param(w)
    let vb = ag.param(b)
    let vy = ag.conv2d(vx, vw, vb, 1, 1)
    let loss = ag.sumall(ag.mul(vy, vy))
    ag.backward(loss)

    let eps = 0.00001
    let x_plus = x + 0.0
    let x_minus = x + 0.0
    ffi.write_f64(x_plus.data_ptr() + 5 * 8, ffi.read_f64(x.data_ptr() + 5 * 8) + eps)
    ffi.write_f64(x_minus.data_ptr() + 5 * 8, ffi.read_f64(x.data_ptr() + 5 * 8) - eps)
    let yp = x_plus.conv2d(w, b, 1, 1)
    let ym = x_minus.conv2d(w, b, 1, 1)
    let lp = (yp * yp).sum()
    let lm = (ym * ym).sum()
    let num_gx = (lp - lm) / (2.0 * eps)
    let ana_gx = ffi.read_f64(vx.grad.data_ptr() + 5 * 8)
    assert abs(num_gx - ana_gx) < 0.00001, "conv2d grad_x mismatch: " + str(num_gx) + " vs " + str(ana_gx)
    print("conv2d (cuDNN/native + autograd backward) OK")

fn test_fallback_or_gpu():
    let avail = cublas.is_available()
    print("cuBLAS available:", avail, "backend:", cublas.backend(), "cuDNN backend:", cudnn.backend())
    assert type(avail) == "bool"
    if avail:
        assert cublas.backend() == "cublas"
        assert cublas.device_count() > 0
    else:
        assert cublas.backend() == "cpu"
        assert cublas.device_count() == 0
        print("No cuBLAS/GPU: Graceful CPU Fallback active")
    test_gemm_and_tensor_cores()
    test_conv2d_and_autograd()

test_mem_primitives()
test_data_ptr()
test_fallback_or_gpu()
print("All cublas tests passed successfully!")
