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
