# std/ffi.t — C Foreign Function Interface (FFI) cho T

let _sys = load("__sys")

struct Function:
    addr
    retty
    argtypes

    fn call(self, *args):
        return _sys.ffi_call(self.addr, self.retty, self.argtypes, args)

struct Library:
    id

    fn sym(self, name):
        return _sys.ffi_sym(self.id, name)

    fn get_fn(self, name, retty = "v", argtypes = ""):
        let s = self.sym(name)
        if s == 0:
            return nil
        return Function(s, retty, argtypes)

    fn close(self):
        return _sys.ffi_close(self.id)

fn open(path):
    let id = _sys.ffi_open(path)
    if id < 0:
        return nil
    return Library(id)

# Thao tác bộ nhớ C thô (Raw memory buffers & pointer dereference)
fn alloc(size):
    return _sys.mem_alloc(size)

fn free(ptr):
    return _sys.mem_free(ptr)

fn read_ptr(ptr):
    return _sys.mem_read_ptr(ptr)

fn write_ptr(ptr, val):
    return _sys.mem_write_ptr(ptr, val)

fn read_i32(ptr):
    return _sys.mem_read_i32(ptr)

fn write_i32(ptr, val):
    return _sys.mem_write_i32(ptr, val)

fn read_str(ptr):
    return _sys.mem_read_str(ptr)

struct Buffer:
    ptr
    size

    fn read_ptr(self):
        return _sys.mem_read_ptr(self.ptr)

    fn write_ptr(self, val):
        return _sys.mem_write_ptr(self.ptr, val)

    fn read_i32(self):
        return _sys.mem_read_i32(self.ptr)

    fn write_i32(self, val):
        return _sys.mem_write_i32(self.ptr, val)

    fn read_str(self):
        return _sys.mem_read_str(self.ptr)

    fn free(self):
        if self.ptr != 0:
            _sys.mem_free(self.ptr)
            self.ptr = 0

fn buffer(size):
    return Buffer(_sys.mem_alloc(size), size)
