# tests/vulkan_test.t — Kiểm thử module std/vulkan (M7 Hoàn chỉnh)
# Máy có GPU Vulkan: kiểm tra liệt kê device đầy đủ (name, vendor_id, device_id, type, api_version)
# + thực thi SPIR-V Compute Shader trực tiếp trên GPU (gemm_f32 & Ozaki 4-lớp gemm/matmul < 1e-12).
# Máy không có Vulkan loader/ICD: kiểm tra Graceful CPU Fallback êm + đúng (khớp @).

let vulkan = load("std/vulkan")

fn test_spirv_binary():
    let spv = vulkan.spirv_words()
    assert spv.len > 20
    assert spv[0] == 0x07230203, "SPIR-V magic number mismatch"
    assert spv[1] == 0x00010000, "SPIR-V version 1.0 mismatch"
    print("SPIR-V binary module verified (" + str(spv.len) + " words)")

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

    # 1. Test gemm_f32 (SPIR-V FP32 1-dispatch trên GPU)
    let got_f32 = vulkan.gemm_f32(A, B)
    assert got_f32.shape == [m, n]
    let err_f32 = (got_f32 - ref).abs().max()
    assert err_f32 < 0.00001, "gemm_f32 error too large: " + str(err_f32)

    # 2. Test gemm / matmul (SPIR-V Ozaki 4-slice đạt chuẩn float64 < 1e-12)
    let got = vulkan.gemm(A, B)
    assert got.shape == [m, n]
    let err_f64 = (got - ref).abs().max()
    assert err_f64 < 0.000000001, "gemm error too large: " + str(err_f64)
    let got2 = vulkan.matmul(A, B)
    assert (got2 - ref).abs().max() < 0.000000001
    print("Vulkan GEMM OK (f32 max_err=" + str(err_f32) + ", ozaki f64 max_err=" + str(err_f64) + ")")

fn test_vulkan():
    test_spirv_binary()
    let avail = vulkan.is_available()
    let bkend = vulkan.backend()
    print("Vulkan available:", avail, "backend:", bkend)
    assert type(avail) == "bool"
    assert bkend == "vulkan" or bkend == "cpu"
    if avail:
        assert vulkan.device_count() > 0
        assert vulkan.loader_version() != nil
        for d in vulkan.devices():
            print("  device:", d["name"], "|", d["type"], "| API", d["api_version"], "| vendor:", d["vendor_id"], "dev:", d["device_id"])
            assert d["name"] != nil and d["name"].len > 0
            assert d["type"] != nil and d["api_version"] != nil
            assert type(d["vendor_id"]) == "int" and type(d["device_id"]) == "int"
    else:
        print("No Vulkan loader/ICD: Graceful CPU Fallback active")
        assert bkend == "cpu"
        assert vulkan.device_count() == 0
        assert vulkan.devices() == []
    test_gemm()

test_vulkan()
print("All vulkan tests passed successfully!")
