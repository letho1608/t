# examples/llm_inference.t — Minh họa Động cơ Suy luận Mô hình Ngôn ngữ Lớn (LLM) trên T
# Chạy kiến trúc Transformer hiện đại (LLaMA/Mistral/Qwen): RMSNorm, RoPE, KV-Cache, GQA, SwiGLU

import transformer
import quant
import time
import math

print("=================================================================")
print("   T Programming Language — LLM & Transformer Inference Engine   ")
print("=================================================================")

# 1. Khởi tạo từ điển Vocab và Tokenizer
let vocab = {
    "<bos>": 0,
    "<eos>": 1,
    "T": 2,
    " là": 3,
    " ngôn": 4,
    " ngữ": 5,
    " lập": 6,
    " trình": 7,
    " hiệu": 8,
    " năng": 9,
    " cao": 10,
    " cho": 11,
    " AI": 12,
    " và": 13,
    " hệ": 14,
    " thống": 15,
    " lớn": 16,
    ".": 17,
    " C": 18,
    " JIT": 19,
    " tensor": 20
}

mut inv_vocab = []
for i in range(len(vocab)):
    inv_vocab.append("")
for k in vocab:
    inv_vocab[vocab[k]] = k

let tok = transformer.Tokenizer(vocab, inv_vocab, 0, 1)

# 2. Cấu hình mô hình Transformer (Micro-LLaMA)
let vocab_size = len(inv_vocab)
let model_dim = 32
let n_layers = 3
let n_heads = 4
let n_kv_heads = 2                    # GQA (Grouped-Query Attention)
let head_dim = int(model_dim / n_heads) # 8
let kv_dim = int(n_kv_heads * head_dim) # 16
let ffn_hidden = 64

print(f"Cấu hình mô hình:")
print(f"  - Vocab size: {vocab_size}")
print(f"  - Hidden dim: {model_dim}")
print(f"  - Số lớp (Layers): {n_layers}")
print(f"  - Query heads: {n_heads}, KV heads: {n_kv_heads} (GQA group: {int(n_heads / n_kv_heads)})")
print(f"  - Head dim: {head_dim}")
print(f"  - SwiGLU FFN hidden: {ffn_hidden}")

# 3. Khởi tạo trọng số (Embedding, Attention Q/K/V/O, SwiGLU Gate/Up/Down, RMSNorm)
let tok_embd = ones([vocab_size, model_dim]) * 0.15

mut layers = []
for i in range(n_layers):
    let layer = transformer.TransformerLayer(
        i, n_heads, n_kv_heads, head_dim, model_dim,
        ones(model_dim),                           # attn_norm_w
        ones([model_dim, model_dim]) * 0.08,        # wq
        ones([model_dim, kv_dim]) * 0.08,           # wk
        ones([model_dim, kv_dim]) * 0.08,           # wv
        ones([model_dim, model_dim]) * 0.08,        # wo
        ones(model_dim),                           # ffn_norm_w
        ones([model_dim, ffn_hidden]) * 0.08,       # w_gate
        ones([model_dim, ffn_hidden]) * 0.08,       # w_up
        ones([ffn_hidden, model_dim]) * 0.08        # w_down
    )
    layers.append(layer)

let model = transformer.TransformerModel(
    vocab_size, model_dim, n_layers, n_heads, n_kv_heads,
    head_dim, ffn_hidden, tok_embd, layers,
    ones(model_dim),                          # final norm_w
    ones([model_dim, vocab_size]) * 0.1       # output lm_head
)

# 4. Chạy suy luận và sinh văn bản tự hồi quy (Autoregressive Inference)
let prompt = "T là ngôn ngữ"
print(f"\nPrompt đầu vào: \"{prompt}\"")
print("Bắt đầu sinh token thời gian thực (Streaming Generation):")
print("---------------------------------------------------------")

mut generated_count = 0
let t_start = time.clock()

# Hàm callback nhận token theo luồng
fn stream_print(token_str):
    mut count_ref = generated_count + 1
    # In trực tiếp token ra console không đệm
    load("__sys").stdout_write(token_str)

let output = model.generate(tok, prompt, 12, 0.8, 20, 0.9, stream_print)
let t_end = time.clock()
let elapsed = t_end - t_start

print("\n---------------------------------------------------------")
let speed = 12.0 / elapsed if elapsed > 0 else 0.0
let speed_rounded = round(speed * 10) / 10.0
let elapsed_ms = round(elapsed * 10000) / 10.0

print(f"Tổng kết:")
print(f"  - Thời gian suy luận: {elapsed_ms} ms")
print(f"  - Tốc độ sinh từ: {speed_rounded} tokens/giây")
print(f"  - Động cơ: T JIT x86-64 + KV-Cache + GQA + SwiGLU")
print("=========================================================")
