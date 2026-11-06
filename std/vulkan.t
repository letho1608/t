# std/vulkan.t — Cross-Platform GPU qua Vulkan Loader (M7, enumeration skeleton)
#
# Probe loader Vulkan (`vulkan-1.dll` / `libvulkan.so.1`), tạo instance,
# liệt kê physical devices — cho mọi hãng (AMD Radeon, Intel Arc, Apple, ...)
# như M5 (`std/cuda.t`) làm với NVIDIA, nhưng qua Vulkan Loader API.
#
# Chưa CÓ (cần GPU thật + SPIR-V toolchain để verify):
#   - Compute shader dispatch (GEMM qua SPIR-V) -> gemm() hiện fallback CPU.
#   - Tên thiết bị: offset của deviceName trong VkPhysicalDeviceProperties
#     nằm sau VkPhysicalDeviceLimits (kích thước tuỳ config) nên KHÔNG đọc
#     bừa; chỉ báo apiVersion (offset 0) và deviceType (offset 16).
#   - Memory properties (VkPhysicalDeviceMemoryProperties) cũng rất lớn, bỏ.
#
# Thiếu loader/ICD/driver, hoặc bất kỳ bước nào lỗi: êm -> is_available()=false,
# gemm() rơi về `@` CPU. CẢNH BÁO UNSAFE: dùng FFI + con trỏ thô (xem std/ffi.t).
# Giả định 64-bit (offset struct theo x86-64 / AArch64 tiêu chuẩn).

let ffi = load("std/ffi")
let _sys = load("__sys")

fn _load_vulkan_lib():
    let candidates = []
    let plat = _sys.platform()
    if plat == "windows" or plat == "win32":
        candidates.append("vulkan-1.dll")
    else:
        candidates.append("libvulkan.so.1")
        candidates.append("libvulkan.so")
        candidates.append("libvulkan.dylib")

    for c in candidates:
        let lib = ffi.open(c)
        if lib != nil:
            return lib
    return nil

let _lib = _load_vulkan_lib()

struct VulkanCtx:
    lib
    fn_enum_instance_version
    fn_create_instance
    fn_enumerate_devices
    fn_get_props
    fn_destroy_instance
    instance
    loader_version
    devices

fn _fmt_version(v):
    let major = int((v / 4194304) % 128)
    let minor = int((v / 4096) % 256)
    let patch = int(v % 4096)
    return str(major) + "." + str(minor) + "." + str(patch)

fn _type_name(code):
    if code == 1:
        return "INTEGRATED_GPU"
    if code == 2:
        return "DISCRETE_GPU"
    if code == 3:
        return "VIRTUAL_GPU"
    if code == 4:
        return "CPU"
    return "OTHER"

fn _init_vulkan():
    if _lib == nil:
        return nil
    try:
        let fn_ver = _lib.get_fn("vkEnumerateInstanceVersion", "i", "p")
        mut lver = "1.0.0"
        if fn_ver != nil:
            let vb = ffi.buffer(4)
            let rcv = fn_ver.call(vb.ptr)
            let lv = vb.read_i32()
            vb.free()
            if rcv != 0:
                return nil
            lver = _fmt_version(lv)

        let fn_create = _lib.get_fn("vkCreateInstance", "i", "ppp")
        let fn_enum = _lib.get_fn("vkEnumeratePhysicalDevices", "i", "ppp")
        let fn_props = _lib.get_fn("vkGetPhysicalDeviceProperties", "v", "pp")
        let fn_destroy = _lib.get_fn("vkDestroyInstance", "v", "pp")
        if fn_create == nil or fn_enum == nil or fn_props == nil or fn_destroy == nil:
            return nil

        # VkInstanceCreateInfo 72 byte (64-bit): buffer calloc đã zero =
        # struct hợp lệ (sType=0=VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO).
        let ci = ffi.buffer(80)
        let ibuf = ffi.buffer(8)
        let rc = fn_create.call(ci.ptr, nil, ibuf.ptr)
        let inst = ibuf.read_ptr()
        ci.free()
        ibuf.free()
        if rc != 0 or inst == 0:
            return nil

        let cbuf = ffi.buffer(4)
        let rc2 = fn_enum.call(inst, cbuf.ptr, 0)
        let cnt = cbuf.read_i32()
        cbuf.free()
        if rc2 != 0 or cnt < 0:
            return nil

        let devs = []
        if cnt > 0:
            let abuf = ffi.buffer(cnt * 8)
            let cbuf2 = ffi.buffer(4)
            ffi.write_i32(cbuf2.ptr, cnt)
            let rc3 = fn_enum.call(inst, cbuf2.ptr, abuf.ptr)
            cbuf2.free()
            if rc3 == 0:
                for i in range(cnt):
                    let h = ffi.read_ptr(abuf.ptr + i * 8)
                    if h != 0:
                        devs.append(_describe(fn_props, h))
            abuf.free()
        return VulkanCtx(_lib, fn_ver, fn_create, fn_enum, fn_props, fn_destroy, inst, lver, devs)
    except:
        return nil

fn _describe(fn_props, h):
    # vkGetPhysicalDeviceProperties trả void (buffer được điền trực tiếp)
    let pb = ffi.buffer(8192)
    fn_props.call(h, pb.ptr)
    let ver = ffi.read_i32(pb.ptr + 0)
    let t = ffi.read_i32(pb.ptr + 16)
    pb.free()
    return {"type": _type_name(t), "api_version": _fmt_version(ver)}

let _ctx = _init_vulkan()

fn is_available():
    return _ctx != nil and _ctx.devices.len > 0

fn loader_version():
    if _ctx == nil:
        return nil
    return _ctx.loader_version

fn device_count():
    if _ctx == nil:
        return 0
    return _ctx.devices.len

fn devices():
    if _ctx == nil:
        return []
    return _ctx.devices

fn backend():
    return "cpu"

fn gemm(a, b):
    return a @ b

fn matmul(a, b):
    return gemm(a, b)
