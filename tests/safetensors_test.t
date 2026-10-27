# tests/safetensors_test.t — Kiểm thử Bộ đọc định dạng Hugging Face Safetensors

import safetensors
import quant
import os

print("=== 1. Tạo file Safetensors kiểm thử ===")
let path = "tests/_tmp_model.safetensors"

# 16 floats (64 bytes)
let t_orig = arange(16).reshape([4, 4]) * 0.25

# Pack 16 floats into binary:
# We can quantize to FP16 or use raw floats. Let's write 64 bytes using FP16 dequant or write 16 floats
let header_json = '{"w_fp16": {"dtype": "F16", "shape": [4, 4], "data_offsets": [0, 32]}}'
let header_bytes_len = len(header_json)

# Header length as uint64 little endian
let lo = header_bytes_len & 0xffffffff
let hi = (header_bytes_len >> 32) & 0xffffffff
mut hdr_len_str = chr(lo & 0xff) + chr((lo >> 8) & 0xff) + chr((lo >> 16) & 0xff) + chr((lo >> 24) & 0xff)
hdr_len_str += chr(hi & 0xff) + chr((hi >> 8) & 0xff) + chr((hi >> 16) & 0xff) + chr((hi >> 24) & 0xff)

let f16_data = quant.quantize_f16(t_orig)

let f = open(path, "wb")
write(f, hdr_len_str)
write(f, header_json)
write(f, f16_data)
close(f)
print("Tạo file safetensors thành công")

print("=== 2. Mở và tải trọng số từ Safetensors ===")
let reader = safetensors.open_safetensors(path)
assert reader != nil
let names = reader.tensor_names()
print("Danh sách tensor:", names)
assert len(names) == 1 and names[0] == "w_fp16"

let t_loaded = reader.get_tensor("w_fp16")
assert t_loaded != nil
assert t_loaded.shape[0] == 4 and t_loaded.shape[1] == 4
let diff = (t_orig - t_loaded).abs().max()
print("Max diff FP16:", diff)
assert diff < 0.002

reader.close()
os.remove(path)
print("=== SAFETENSORS TEST PASSED ===")
