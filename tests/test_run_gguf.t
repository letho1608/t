# tests/test_run_gguf.t — Kiểm thử trực tiếp tools/run_gguf.t
import gguf
import os

let model_path = "tests/_test_cli_model.gguf"

let vocab_list = ["<unk>", "<s>", "</s>", "Xin", " chào", " bạn", " tôi", " là", " T", " AI"]
mut meta = {
    "general.architecture": "llama",
    "llama.embedding_length": 16,
    "llama.block_count": 2,
    "llama.attention.head_count": 4,
    "llama.attention.head_count_kv": 2,
    "llama.feed_forward_length": 32,
    "tokenizer.ggml.tokens": vocab_list,
    "tokenizer.ggml.bos_token_id": 1,
    "tokenizer.ggml.eos_token_id": 2
}

let dim = 16
let n_kv_heads = 2
let head_dim = 4
let kv_dim = 8
let ffn_hidden = 32

let tensor_list = [
    ["token_embd.weight", ones([len(vocab_list), dim]) * 0.1, gguf.GGML_TYPE_F16],
    ["output_norm.weight", ones(dim), gguf.GGML_TYPE_F16],
    ["output.weight", ones([dim, len(vocab_list)]) * 0.1, gguf.GGML_TYPE_F16],
    # Layer 0
    ["blk.0.attn_norm.weight", ones(dim), gguf.GGML_TYPE_F16],
    ["blk.0.attn_q.weight", ones([dim, dim]) * 0.05, gguf.GGML_TYPE_Q4_0],
    ["blk.0.attn_k.weight", ones([dim, kv_dim]) * 0.05, gguf.GGML_TYPE_Q8_0],
    ["blk.0.attn_v.weight", ones([dim, kv_dim]) * 0.05, gguf.GGML_TYPE_Q8_0],
    ["blk.0.attn_output.weight", ones([dim, dim]) * 0.05, gguf.GGML_TYPE_Q4_0],
    ["blk.0.ffn_norm.weight", ones(dim), gguf.GGML_TYPE_F16],
    ["blk.0.ffn_gate.weight", ones([dim, ffn_hidden]) * 0.05, gguf.GGML_TYPE_Q4_0],
    ["blk.0.ffn_up.weight", ones([dim, ffn_hidden]) * 0.05, gguf.GGML_TYPE_Q4_0],
    ["blk.0.ffn_down.weight", ones([ffn_hidden, dim]) * 0.05, gguf.GGML_TYPE_Q4_0],
    # Layer 1
    ["blk.1.attn_norm.weight", ones(dim), gguf.GGML_TYPE_F16],
    ["blk.1.attn_q.weight", ones([dim, dim]) * 0.05, gguf.GGML_TYPE_Q4_0],
    ["blk.1.attn_k.weight", ones([dim, kv_dim]) * 0.05, gguf.GGML_TYPE_Q8_0],
    ["blk.1.attn_v.weight", ones([dim, kv_dim]) * 0.05, gguf.GGML_TYPE_Q8_0],
    ["blk.1.attn_output.weight", ones([dim, dim]) * 0.05, gguf.GGML_TYPE_Q4_0],
    ["blk.1.ffn_norm.weight", ones(dim), gguf.GGML_TYPE_F16],
    ["blk.1.ffn_gate.weight", ones([dim, ffn_hidden]) * 0.05, gguf.GGML_TYPE_Q4_0],
    ["blk.1.ffn_up.weight", ones([dim, ffn_hidden]) * 0.05, gguf.GGML_TYPE_Q4_0],
    ["blk.1.ffn_down.weight", ones([ffn_hidden, dim]) * 0.05, gguf.GGML_TYPE_Q4_0]
]

gguf.write_gguf(model_path, meta, tensor_list)
print("Đã tạo model GGUF:", model_path)

let trun = os.exepath()
let r = os.run([trun, "tools/run_gguf.t", model_path, "Xin chào"])
print("CLI Runner Output:")
print(r["out"])
assert r["rc"] == 0
assert "Nạp mô hình thành công" in r["out"]
assert "Hoàn tất trong" in r["out"]

os.remove(model_path)
print("=== RUN GGUF TEST PASSED ===")
