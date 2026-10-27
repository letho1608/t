# tests/gguf_test.t — Kiểm thử Bộ đọc và ghi mô hình GGUF v2/v3

import gguf
import os

print("=== 1. Tạo file mô hình GGUF tổng hợp ===")
let path = "tests/_tmp_model.gguf"

let meta = {
    "general.architecture": "llama",
    "llama.context_length": 2048,
    "llama.embedding_length": 64,
    "llama.block_count": 2,
    "llama.feed_forward_length": 128,
    "llama.attention.head_count": 4,
    "llama.attention.head_count_kv": 2
}

# Tạo các trọng số kiểm thử
let t_embd = arange(32 * 64).reshape([32, 64]) * 0.01
let t_q = arange(64 * 64).reshape([64, 64]) * 0.02
let t_k = arange(32 * 64).reshape([32, 64]) * 0.03
let t_norm = arange(64) * 0.1

let tensor_list = [
    ["token_embd.weight", t_embd, gguf.GGML_TYPE_F16],
    ["blk.0.attn_q.weight", t_q, gguf.GGML_TYPE_Q4_0],
    ["blk.0.attn_k.weight", t_k, gguf.GGML_TYPE_Q8_0],
    ["output_norm.weight", t_norm, gguf.GGML_TYPE_F16]
]

let ok_write = gguf.write_gguf(path, meta, tensor_list)
assert ok_write
print("Ghi GGUF thành công:", path)

print("=== 2. Mở và phân tích GGUF qua mmap ===")
let reader = gguf.open_gguf(path)
assert reader != nil
print("GGUF version:", reader.version)
print("Số tensor:", reader.tensor_count)
assert reader.tensor_count == 4

# Kiểm tra metadata
print("Kiểm tra metadata:")
assert reader.metadata["general.architecture"] == "llama"
assert reader.metadata["llama.context_length"] == 2048
assert reader.metadata["llama.embedding_length"] == 64
assert reader.metadata["llama.block_count"] == 2
print("  general.architecture =", reader.metadata["general.architecture"])
print("  llama.context_length =", reader.metadata["llama.context_length"])

# Kiểm tra tải trọng số trực tiếp qua mmap
print("Kiểm tra tải trọng số qua mmap:")
let embd_loaded = reader.get_tensor("token_embd.weight")
assert embd_loaded != nil
assert embd_loaded.shape[0] == 32 and embd_loaded.shape[1] == 64
let diff_embd = (t_embd - embd_loaded).abs().max()
print("  token_embd.weight FP16 diff:", diff_embd)
assert diff_embd < 0.03

let q_loaded = reader.get_tensor("blk.0.attn_q.weight")
assert q_loaded != nil
assert q_loaded.shape[0] == 64 and q_loaded.shape[1] == 64
let diff_q = (t_q - q_loaded).abs().max()
print("  blk.0.attn_q.weight Q4_0 diff:", diff_q)
assert diff_q < 1.0

let k_loaded = reader.get_tensor("blk.0.attn_k.weight")
assert k_loaded != nil
assert k_loaded.shape[0] == 32 and k_loaded.shape[1] == 64
let diff_k = (t_k - k_loaded).abs().max()
print("  blk.0.attn_k.weight Q8_0 diff:", diff_k)
assert diff_k < 0.3

let norm_loaded = reader.get_tensor("output_norm.weight")
assert norm_loaded != nil
assert norm_loaded.shape[0] == 64
let diff_norm = (t_norm - norm_loaded).abs().max()
print("  output_norm.weight FP16 diff:", diff_norm)
assert diff_norm < 0.01

reader.close()
os.remove(path)
print("=== GGUF TEST PASSED ===")
