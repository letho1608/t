# std/cstruct.t — #[repr(C)] Zero-Copy Contiguous Struct Array & SIMD Stride Operations (v7.0 S1)

let _sys = load("__sys")

fn type_size_align(ty):
    if ty == "f64" or ty == "float64" or ty == "double" or ty == "i64" or ty == "int64" or ty == "ptr":
        return [8, 8]
    if ty == "f32" or ty == "float32" or ty == "i32" or ty == "int32":
        return [4, 4]
    if ty == "u16" or ty == "uint16" or ty == "i16" or ty == "f16" or ty == "bf16":
        return [2, 2]
    raise "cstruct: kieu khong ho tro '" + str(ty) + "'"

fn align_up(offset, align):
    rem = offset % align
    return offset if rem == 0 else offset + (align - rem)

struct CStructArray:
    ptr
    count
    stride
    offsets
    types
    owns

    fn ptr_at(self, idx: int) -> int:
        return self.ptr + idx * self.stride

    fn get_f64(self, idx: int, field: str) -> float:
        off = self.offsets[field]
        return _sys.mem_read_f64(self.ptr + idx * self.stride + off)

    fn set_f64(self, idx: int, field: str, val):
        off = self.offsets[field]
        return _sys.mem_write_f64(self.ptr + idx * self.stride + off, float(val))

    fn get_f32(self, idx: int, field: str) -> float:
        off = self.offsets[field]
        return _sys.mem_read_f32(self.ptr + idx * self.stride + off)

    fn set_f32(self, idx: int, field: str, val):
        off = self.offsets[field]
        return _sys.mem_write_f32(self.ptr + idx * self.stride + off, float(val))

    fn get_i32(self, idx: int, field: str) -> int:
        off = self.offsets[field]
        return _sys.mem_read_i32(self.ptr + idx * self.stride + off)

    fn set_i32(self, idx: int, field: str, val: int):
        off = self.offsets[field]
        return _sys.mem_write_i32(self.ptr + idx * self.stride + off, val)

    fn get_u16(self, idx: int, field: str) -> int:
        off = self.offsets[field]
        return _sys.mem_read_u16(self.ptr + idx * self.stride + off)

    fn set_u16(self, idx: int, field: str, val: int):
        off = self.offsets[field]
        return _sys.mem_write_u16(self.ptr + idx * self.stride + off, val)

    fn sum_f64(self, field: str) -> float:
        return _sys.cstruct_sum_f64(self.ptr, self.count, self.stride, self.offsets[field])

    fn axpy_f64(self, dst_field: str, src_field: str, scale: float):
        return _sys.cstruct_axpy_f64(self.ptr, self.count, self.stride, self.offsets[dst_field], self.offsets[src_field], scale)

    fn free(self):
        if self.owns and self.ptr != 0:
            _sys.mem_free(self.ptr)
            self.ptr = 0
            self.owns = false

    fn close(self):
        self.free()

    fn __enter__(self):
        return self

    fn __exit__(self):
        self.free()

struct CStructSchema:
    name
    stride
    max_align
    offsets
    types

    fn offset_of(self, field: str) -> int:
        return self.offsets[field]

    fn alloc_array(self, count: int):
        nbytes = count * self.stride
        p = _sys.mem_alloc(nbytes if nbytes > 0 else 8)
        return CStructArray(p, count, self.stride, self.offsets, self.types, true)

    fn from_ptr(self, ptr: int, count: int, owns = false):
        return CStructArray(ptr, count, self.stride, self.offsets, self.types, owns)

fn define(name: str, field_specs):
    mut off = 0
    mut max_al = 1
    mut offsets = {}
    mut types = {}
    for spec in field_specs:
        fname = spec[0]
        fty = spec[1]
        sa = type_size_align(fty)
        sz = sa[0]
        al = sa[1]
        if al > max_al:
            max_al = al
        off = align_up(off, al)
        offsets[fname] = off
        types[fname] = fty
        off = off + sz
    stride = align_up(off, max_al)
    return CStructSchema(name, stride, max_al, offsets, types)
