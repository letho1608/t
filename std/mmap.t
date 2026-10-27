# std/mmap.t — Memory-Mapped I/O và Zero-Copy Tensor Storage cho T

let _sys = load("__sys")

struct MMap:
    id

    fn size(self):
        return _sys.mmap_size(self.id)

    fn close(self):
        return _sys.mmap_close(self.id)

    fn read_bytes(self, offset, length):
        return _sys.mmap_read_bytes(self.id, offset, length)

    fn read_u8(self, offset):
        return _sys.mmap_read_u8(self.id, offset)

    fn read_u16(self, offset):
        return _sys.mmap_read_u16(self.id, offset)

    fn read_u32(self, offset):
        return _sys.mmap_read_u32(self.id, offset)

    fn read_u64(self, offset):
        return _sys.mmap_read_u64(self.id, offset)

    fn read_i32(self, offset):
        return _sys.mmap_read_i32(self.id, offset)

    fn read_i64(self, offset):
        return _sys.mmap_read_i64(self.id, offset)

    fn read_f32(self, offset):
        return _sys.mmap_read_f32(self.id, offset)

    fn read_f64(self, offset):
        return _sys.mmap_read_f64(self.id, offset)

    fn tensor_f32(self, offset, shape):
        return _sys.mmap_tensor_f32(self.id, offset, shape)

    fn tensor_f16(self, offset, shape):
        return _sys.mmap_tensor_f16(self.id, offset, shape)

    fn tensor_q4_0(self, offset, shape):
        return _sys.mmap_tensor_q4_0(self.id, offset, shape)

    fn tensor_q8_0(self, offset, shape):
        return _sys.mmap_tensor_q8_0(self.id, offset, shape)

fn open(path):
    let id = _sys.mmap_open(path)
    return MMap(id)
