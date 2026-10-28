# std/cuda.t — CUDA Driver Interface & GPU Acceleration Support cho T

let ffi = load("std/ffi")
let _sys = load("__sys")

fn _load_cuda_lib():
    let candidates = []
    let plat = _sys.platform()
    if plat == "windows" or plat == "win32":
        candidates.append("nvcuda.dll")
    else:
        candidates.append("libcuda.so.1")
        candidates.append("libcuda.so")

    for c in candidates:
        let lib = ffi.open(c)
        if lib != nil:
            return lib
    return nil

let _lib = _load_cuda_lib()

struct CudaDriver:
    lib
    fn_init
    fn_driver_version
    fn_device_count
    fn_device_get
    fn_device_name
    fn_device_total_mem

fn _init_cuda():
    if _lib == nil:
        return nil
    let cuInit = _lib.get_fn("cuInit", "i", "i")
    if cuInit == nil:
        return nil
    let rc = cuInit.call(0)
    if rc != 0:
        return nil

    return CudaDriver(
        _lib,
        cuInit,
        _lib.get_fn("cuDriverGetVersion", "i", "p"),
        _lib.get_fn("cuDeviceGetCount", "i", "p"),
        _lib.get_fn("cuDeviceGet", "i", "pi"),
        _lib.get_fn("cuDeviceGetName", "i", "pii"),
        _lib.get_fn("cuDeviceTotalMem_v2", "i", "pi")
    )

let _drv = _init_cuda()

fn is_available():
    return _drv != nil

fn device_count():
    if _drv == nil:
        return 0
    let buf = ffi.buffer(4)
    let rc = _drv.fn_device_count.call(buf.ptr)
    if rc != 0:
        buf.free()
        return 0
    let count = buf.read_i32()
    buf.free()
    return count

fn driver_version():
    if _drv == nil:
        return nil
    let buf = ffi.buffer(4)
    let rc = _drv.fn_driver_version.call(buf.ptr)
    if rc != 0:
        buf.free()
        return nil
    let ver = buf.read_i32()
    buf.free()
    let major = int(ver / 1000)
    let minor = int((ver % 1000) / 10)
    return str(major) + "." + str(minor)

fn device_name(device_id = 0):
    if _drv == nil:
        return nil
    let dev_buf = ffi.buffer(4)
    let rc = _drv.fn_device_get.call(dev_buf.ptr, device_id)
    if rc != 0:
        dev_buf.free()
        return nil
    let dev = dev_buf.read_i32()
    dev_buf.free()

    let name_buf = ffi.alloc(256)
    let rc2 = _drv.fn_device_name.call(name_buf, 256, dev)
    if rc2 != 0:
        ffi.free(name_buf)
        return nil
    let name = ffi.read_str(name_buf)
    ffi.free(name_buf)
    return name

fn device_total_mem(device_id = 0):
    if _drv == nil or _drv.fn_device_total_mem == nil:
        return 0
    let dev_buf = ffi.buffer(4)
    let rc = _drv.fn_device_get.call(dev_buf.ptr, device_id)
    if rc != 0:
        dev_buf.free()
        return 0
    let dev = dev_buf.read_i32()
    dev_buf.free()

    let mem_buf = ffi.buffer(8)
    let rc2 = _drv.fn_device_total_mem.call(mem_buf.ptr, dev)
    if rc2 != 0:
        mem_buf.free()
        return 0
    let total = mem_buf.read_ptr()
    mem_buf.free()
    return total
