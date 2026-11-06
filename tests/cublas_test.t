# tests/cublas_test.t — Kiểm thử module std/cublas (M6)
# Máy không có GPU NVIDIA: chỉ kiểm tra CPU fallback êm + đúng (khớp @ từng bit).
# Máy có GPU: kiểm tra GEMM cuBLAS khớp @ trong 1e-9.

let cublas = load("std/cublas")
let ffi = load("std/ffi")

fn test_mem_f64():
    let p = ffi.alloc(16)
    assert p != 0
    ffi.write_f64(p, 3.141592653589793)
    assert ffi.read_f64(p) == 3.141592653589793
    ffi.write_f64(p, -2.5)
    assert ffi.read_f64(p) == -2.5
    ffi.free(p)
    print("mem_f64 round-trip OK")

fn test_data_ptr():
    let t = tensor([[1.0, 2.0], [3.0, 4.0]])
    let p = t.data_ptr()
    assert type(p) == "int" and p != 0
    assert ffi.read_f64(p) == 1.0
    assert ffi.read_f64(p + 8) == 2.0
    assert ffi.read_f64(p + 16) == 3.0
    assert ffi.read_f64(p + 24) == 4.0
    print("data_ptr OK")

fn test_gemm():
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
    print("gemm OK (max diff " + str(d) + ")")

fn test_fallback_or_gpu():
    let avail = cublas.is_available()
    print("cuBLAS available:", avail, "backend:", cublas.backend())
    assert type(avail) == "bool"
    if avail:
        assert cublas.backend() == "cublas"
        assert cublas.device_count() > 0
    else:
        assert cublas.backend() == "cpu"
        assert cublas.device_count() == 0
        print("No cuBLAS/GPU: Graceful CPU Fallback active")
    test_gemm()

test_mem_f64()
test_data_ptr()
test_fallback_or_gpu()
print("All cublas tests passed successfully!")
