# tests/cuda_test.t — Kiểm thử module std/cuda

let cuda = load("std/cuda")

fn test_cuda():
    print("=== Test CUDA Subsystem ===")
    let avail = cuda.is_available()
    print("CUDA Available:", avail)
    assert type(avail) == "bool"

    if avail:
        let count = cuda.device_count()
        print("Device count:", count)
        assert count > 0
        let ver = cuda.driver_version()
        print("Driver version:", ver)
        assert ver != nil
        for i in range(count):
            let dev_name = cuda.device_name(i)
            let vram = cuda.device_total_mem(i)
            print("Device", i, ":", dev_name, "Total VRAM (MB):", int(vram / (1024 * 1024)))
    else:
        print("No CUDA hardware detected: Graceful CPU Fallback active")
        assert cuda.device_count() == 0
        assert cuda.device_name(0) == nil
        assert cuda.driver_version() == nil

    print("CUDA Test: PASSED")

test_cuda()
print("All cuda tests passed successfully!")
