# std/gguf.t — Parser và Writer cho định dạng mô hình nhị phân GGUF v2/v3

import mmap
import quant

let GGUF_MAGIC = 0x46554747  # "GGUF"
let GGUF_VERSION = 3

# Metadata value types
let GGUF_TYPE_UINT8   = 0
let GGUF_TYPE_INT8    = 1
let GGUF_TYPE_UINT16  = 2
let GGUF_TYPE_INT16   = 3
let GGUF_TYPE_UINT32  = 4
let GGUF_TYPE_INT32   = 5
let GGUF_TYPE_FLOAT32 = 6
let GGUF_TYPE_BOOL    = 7
let GGUF_TYPE_STRING  = 8
let GGUF_TYPE_ARRAY   = 9
let GGUF_TYPE_UINT64  = 10
let GGUF_TYPE_INT64   = 11
let GGUF_TYPE_FLOAT64 = 12

# GGML Tensor types
let GGML_TYPE_F32  = 0
let GGML_TYPE_F16  = 1
let GGML_TYPE_Q4_0 = 2
let GGML_TYPE_Q8_0 = 8

struct GGUFTensorInfo:
    name
    shape
    dtype
    offset
    file_offset

struct GGUFReader:
    mm
    version
    tensor_count
    metadata
    tensors
    data_offset

    fn get_tensor(self, name):
        if not self.tensors.has(name):
            return nil
        let info = self.tensors[name]
        let ttype = info.dtype
        let off = info.file_offset
        let sh = info.shape
        if ttype == GGML_TYPE_F32:
            return self.mm.tensor_f32(off, sh)
        if ttype == GGML_TYPE_F16:
            return self.mm.tensor_f16(off, sh)
        if ttype == GGML_TYPE_Q4_0:
            return self.mm.tensor_q4_0(off, sh)
        if ttype == GGML_TYPE_Q8_0:
            return self.mm.tensor_q8_0(off, sh)
        return nil

    fn close(self):
        self.mm.close()

fn _read_str(mm, pos_ref):
    let p = pos_ref[0]
    let slen = mm.read_u64(p)
    let s = mm.read_bytes(p + 8, slen)
    pos_ref[0] = p + 8 + slen
    return s

fn _read_val(mm, vtype, pos_ref):
    let p = pos_ref[0]
    if vtype == GGUF_TYPE_UINT8:
        pos_ref[0] = p + 1
        return mm.read_u8(p)
    if vtype == GGUF_TYPE_INT8:
        pos_ref[0] = p + 1
        let v = mm.read_u8(p)
        return v if v < 128 else v - 256
    if vtype == GGUF_TYPE_UINT16:
        pos_ref[0] = p + 2
        return mm.read_u16(p)
    if vtype == GGUF_TYPE_INT16:
        pos_ref[0] = p + 2
        let v = mm.read_u16(p)
        return v if v < 32768 else v - 65536
    if vtype == GGUF_TYPE_UINT32:
        pos_ref[0] = p + 4
        return mm.read_u32(p)
    if vtype == GGUF_TYPE_INT32:
        pos_ref[0] = p + 4
        return mm.read_i32(p)
    if vtype == GGUF_TYPE_FLOAT32:
        pos_ref[0] = p + 4
        return mm.read_f32(p)
    if vtype == GGUF_TYPE_BOOL:
        pos_ref[0] = p + 1
        return mm.read_u8(p) != 0
    if vtype == GGUF_TYPE_STRING:
        return _read_str(mm, pos_ref)
    if vtype == GGUF_TYPE_UINT64 or vtype == GGUF_TYPE_INT64:
        pos_ref[0] = p + 8
        return mm.read_i64(p)
    if vtype == GGUF_TYPE_FLOAT64:
        pos_ref[0] = p + 8
        return mm.read_f64(p)
    if vtype == GGUF_TYPE_ARRAY:
        let itype = mm.read_u32(p)
        let count = mm.read_u64(p + 4)
        pos_ref[0] = p + 12
        mut arr = []
        for i in range(count):
            arr.append(_read_val(mm, itype, pos_ref))
        return arr
    pos_ref[0] = p + 1
    return nil

fn open_gguf(path):
    let mm = mmap.open(path)
    let magic = mm.read_u32(0)
    if magic != GGUF_MAGIC:
        mm.close()
        return nil
    let ver = mm.read_u32(4)
    let tensor_count = mm.read_u64(8)
    let metadata_kv_count = mm.read_u64(16)

    mut pos_ref = [24]
    mut meta = {}
    for i in range(metadata_kv_count):
        let key = _read_str(mm, pos_ref)
        let vtype = mm.read_u32(pos_ref[0])
        pos_ref[0] += 4
        let val = _read_val(mm, vtype, pos_ref)
        meta[key] = val

    mut tensors = {}
    for i in range(tensor_count):
        let name = _read_str(mm, pos_ref)
        let ndim = mm.read_u32(pos_ref[0])
        pos_ref[0] += 4
        mut shape = []
        for d in range(ndim):
            shape.append(mm.read_u64(pos_ref[0]))
            pos_ref[0] += 8
        let ttype = mm.read_u32(pos_ref[0])
        pos_ref[0] += 4
        let off = mm.read_u64(pos_ref[0])
        pos_ref[0] += 8
        tensors[name] = GGUFTensorInfo(name, shape, ttype, off, 0)

    let alignment = meta.get("general.alignment", 32)
    let cur_pos = pos_ref[0]
    let rem = cur_pos % alignment
    let data_offset = cur_pos if rem == 0 else cur_pos + (alignment - rem)

    for k in tensors:
        let info = tensors[k]
        info.file_offset = data_offset + info.offset

    return GGUFReader(mm, ver, tensor_count, meta, tensors, data_offset)

# --- GGUF Writer Helper ---

fn _pack_u8(v):
    return chr(v & 0xff)

fn _pack_u16(v):
    return chr(v & 0xff) + chr((v >> 8) & 0xff)

fn _pack_u32(v):
    return chr(v & 0xff) + chr((v >> 8) & 0xff) + chr((v >> 16) & 0xff) + chr((v >> 24) & 0xff)

fn _pack_u64(v):
    let lo = v & 0xffffffff
    let hi = (v >> 32) & 0xffffffff
    return _pack_u32(lo) + _pack_u32(hi)

fn _pack_str(s):
    return _pack_u64(len(s)) + s

fn write_gguf(path, metadata, tensors, alignment = 32):
    # tensors: list of [name, tensor, ttype]
    mut buf = ""
    # Header
    buf += _pack_u32(GGUF_MAGIC)
    buf += _pack_u32(GGUF_VERSION)
    buf += _pack_u64(len(tensors))
    buf += _pack_u64(len(metadata))

    # Metadata
    for k in metadata:
        buf += _pack_str(k)
        let v = metadata[k]
        if type(v) == "str":
            buf += _pack_u32(GGUF_TYPE_STRING)
            buf += _pack_str(v)
        elif type(v) == "int":
            buf += _pack_u32(GGUF_TYPE_INT64)
            buf += _pack_u64(v)
        elif type(v) == "float":
            buf += _pack_u32(GGUF_TYPE_FLOAT64)
            # pack float64: write through quant or sys helper
            let fb = quant.quantize_f16(tensor([v]))
            buf += _pack_u64(0)
        elif type(v) == "bool":
            buf += _pack_u32(GGUF_TYPE_BOOL)
            buf += _pack_u8(1 if v else 0)
        elif type(v) == "list":
            buf += _pack_u32(GGUF_TYPE_ARRAY)
            # Assume list of strings if string elements, else int
            if len(v) > 0 and type(v[0]) == "str":
                buf += _pack_u32(GGUF_TYPE_STRING)
                buf += _pack_u64(len(v))
                for item in v:
                    buf += _pack_str(item)
            else:
                buf += _pack_u32(GGUF_TYPE_INT64)
                buf += _pack_u64(len(v))
                for item in v:
                    buf += _pack_u64(item)

    # Tensor descriptors
    mut tensor_bytes = []
    mut cur_off = 0
    for item in tensors:
        let tname = item[0]
        let ten = item[1]
        let ttype = item[2]
        buf += _pack_str(tname)
        let sh = ten.shape
        buf += _pack_u32(len(sh))
        for d in sh:
            buf += _pack_u64(d)
        buf += _pack_u32(ttype)
        buf += _pack_u64(cur_off)

        # Quantize or serialize data
        mut bdata = ""
        if ttype == GGML_TYPE_F32:
            bdata = quant.quantize_f16(ten)
        elif ttype == GGML_TYPE_F16:
            bdata = quant.quantize_f16(ten)
        elif ttype == GGML_TYPE_Q4_0:
            bdata = quant.quantize_q4_0(ten)
        elif ttype == GGML_TYPE_Q8_0:
            bdata = quant.quantize_q8_0(ten)
        tensor_bytes.append(bdata)
        cur_off += len(bdata)
        # Pad each tensor offset to 32 bytes if needed
        let trem = cur_off % alignment
        if trem != 0:
            let pad = alignment - trem
            cur_off += pad

    # Pad header buffer to alignment
    let hrem = len(buf) % alignment
    if hrem != 0:
        for i in range(alignment - hrem):
            buf += chr(0)

    # Write file
    let f = open(path, "wb")
    write(f, buf)
    for i in range(len(tensor_bytes)):
        let b = tensor_bytes[i]
        write(f, b)
        let trem = len(b) % alignment
        if trem != 0:
            mut pstr = ""
            for p in range(alignment - trem):
                pstr += chr(0)
            write(f, pstr)
    close(f)
    return true
