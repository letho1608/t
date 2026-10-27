# tests/ffi_test.t — Kiểm thử C Foreign Function Interface (FFI)

import ffi
import os

print("=== Kiểm thử C FFI ===")

let plat = os.platform()
print("Platform:", plat)

if plat == "win32":
    let lib = ffi.open("kernel32.dll")
    assert lib != nil
    print("Mở kernel32.dll thành công, id:", lib.id)

    # 1. Gọi hàm không đối số trả về int32 (GetCurrentProcessId)
    let fn_pid = lib.get_fn("GetCurrentProcessId", "i", "")
    assert fn_pid != nil
    let pid = fn_pid.call()
    print("GetCurrentProcessId() =", pid)
    assert pid > 0

    # 2. Gọi hàm không đối số trả về int64 (GetTickCount64)
    let fn_ticks = lib.get_fn("GetTickCount64", "l", "")
    assert fn_ticks != nil
    let t1 = fn_ticks.call()
    print("GetTickCount64() =", t1)
    assert t1 > 0

    lib.close()
    print("Đóng kernel32.dll thành công")
else:
    let lib = ffi.open("libc.so.6")
    if lib != nil:
        let fn_pid = lib.get_fn("getpid", "i", "")
        if fn_pid != nil:
            let pid = fn_pid.call()
            print("getpid() =", pid)
            assert pid > 0
        lib.close()

print("=== FFI TEST PASSED ===")
