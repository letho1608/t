# tests/mmap_test.t — Kiểm thử Memory-Mapped I/O và Zero-Copy Tensor Storage

import mmap
import quant
import os

print("=== 1. Tạo file nhị phân kiểm thử ===")
let test_path = "tests/_tmp_mmap_test.bin"
mut f = open(test_path, "wb")

# Ghi dữ liệu vô hướng (scalar data)
# byte 0: u8 (42)
# byte 1..2: u16 (1337)
# byte 3..6: u32 (100000)
# byte 7..14: u64 (9876543210)
# byte 15..18: f32 (3.14159)
# byte 19..26: f64 (2.718281828459)
mut s = ""
s += chr(42)                                       # u8
s += chr(1337 & 0xff) + chr((1337 >> 8) & 0xff)   # u16
s += chr(100000 & 0xff) + chr((100000 >> 8) & 0xff) + chr((100000 >> 16) & 0xff) + chr((100000 >> 24) & 0xff)
# u64
let u64_lo = 9876543210 & 0xffffffff
let u64_hi = (9876543210 >> 32) & 0xffffffff
s += chr(u64_lo & 0xff) + chr((u64_lo >> 8) & 0xff) + chr((u64_lo >> 16) & 0xff) + chr((u64_lo >> 24) & 0xff)
s += chr(u64_hi & 0xff) + chr((u64_hi >> 8) & 0xff) + chr((u64_hi >> 16) & 0xff) + chr((u64_hi >> 24) & 0xff)
write(f, s)

# Tensors
let t_data = arange(32) * 0.5
let f16_bytes = quant.quantize_f16(t_data)
let q4_bytes = quant.quantize_q4_0(t_data)
let q8_bytes = quant.quantize_q8_0(t_data)
write(f, f16_bytes)
write(f, q4_bytes)
write(f, q8_bytes)
close(f)

print("=== 2. Mở file mmap và kiểm tra đọc vô hướng ===")
let mm = mmap.open(test_path)
assert mm.size() == len(s) + len(f16_bytes) + len(q4_bytes) + len(q8_bytes)

let v_u8 = mm.read_u8(0)
assert v_u8 == 42
let v_u16 = mm.read_u16(1)
assert v_u16 == 1337
let v_u32 = mm.read_u32(3)
assert v_u32 == 100000
let v_u64 = mm.read_u64(7)
assert v_u64 == 9876543210
print("Scalar reads OK: u8=42, u16=1337, u32=100000, u64=9876543210")

print("=== 3. Đọc Tensor qua mmap ===")
let off_t = len(s)
let t_f16 = mm.tensor_f16(off_t, 32)
let diff_f16 = (t_data - t_f16).abs().max()
assert diff_f16 < 0.002
print("Tensor FP16 mmap load OK (diff < 0.002)")

let off_q4 = off_t + len(f16_bytes)
let t_q4 = mm.tensor_q4_0(off_q4, 32)
let diff_q4 = (t_data - t_q4).abs().max()
assert diff_q4 < 1.0
print("Tensor Q4_0 mmap load OK (diff < 1.0)")

let off_q8 = off_q4 + len(q4_bytes)
let t_q8 = mm.tensor_q8_0(off_q8, 32)
let diff_q8 = (t_data - t_q8).abs().max()
assert diff_q8 < 0.1
print("Tensor Q8_0 mmap load OK (diff < 0.1)")

mm.close()
os.remove(test_path)
print("=== MMAP TEST PASSED ===")
