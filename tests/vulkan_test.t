# tests/vulkan_test.t — Kiểm thử module std/vulkan (M7)
# Máy không có Vulkan loader/ICD: chỉ kiểm tra fallback CPU êm + đúng (khớp @).
# Máy có Vulkan: kiểm tra liệt kê device + gemm vẫn đúng (hiện CPU fallback).

let vulkan = load("std/vulkan")

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
    let got = vulkan.gemm(A, B)
    assert got.shape == [m, n]
    assert (got - ref).abs().max() < 0.000000001
    let got2 = vulkan.matmul(A, B)
    assert (got2 - ref).abs().max() < 0.000000001
    print("gemm fallback OK")

fn test_vulkan():
    let avail = vulkan.is_available()
    print("Vulkan available:", avail, "backend:", vulkan.backend())
    assert type(avail) == "bool"
    assert vulkan.backend() == "cpu"
    if avail:
        assert vulkan.device_count() > 0
        assert vulkan.loader_version() != nil
        for d in vulkan.devices():
            print("  device:", d["type"], "API", d["api_version"])
            assert d["type"] != nil and d["api_version"] != nil
    else:
        print("No Vulkan loader/ICD: Graceful CPU Fallback active")
        assert vulkan.device_count() == 0
        assert vulkan.devices() == []
    test_gemm()

test_vulkan()
print("All vulkan tests passed successfully!")
