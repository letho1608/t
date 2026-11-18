# examples/llm_inference.t — Demonstration of LLM & Transformer Inference in T
# Runs modern Transformer architectures (LLaMA/Mistral/Qwen): RMSNorm, RoPE, KV-Cache, GQA, SwiGLU

import transformer
import quant
import time
import math

print("=================================================================")
print("   T Programming Language — LLM & Transformer Inference Engine   ")
print("=================================================================")

# 1. Initialize Vocabulary and Tokenizer
let vocab = {
    "<bos>": 0,
    "<eos>": 1,
    "T": 2,
    " is": 3,
    " a": 4,
    " high": 5,
    " performance": 6,
    " programming": 7,
    " language": 8,
    " for": 9,
    " modern": 10,
    " AI": 11,
    " systems": 12,
    " and": 13,
    " large": 14,
    " scale": 15,
    " computing": 16,
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

# 2. Configure Transformer Model (Micro-LLaMA)
let vocab_size = len(inv_vocab)
let model_dim = 32
let n_layers = 3
let n_heads = 4
let n_kv_heads = 2                    # GQA (Grouped-Query Attention)
let head_dim = int(model_dim / n_heads) # 8
let kv_dim = int(n_kv_heads * head_dim) # 16
let ffn_hidden = 64

print(f"Model Configuration:")
print(f"  - Vocab size: {vocab_size}")
print(f"  - Hidden dim: {model_dim}")
print(f"  - Layers: {n_layers}")
print(f"  - Query heads: {n_heads}, KV heads: {n_kv_heads} (GQA group: {int(n_heads / n_kv_heads)})")
print(f"  - Head dim: {head_dim}")
print(f"  - SwiGLU FFN hidden: {ffn_hidden}")

# 3. Initialize Model Weights (Embedding, Attention Q/K/V/O, SwiGLU Gate/Up/Down, RMSNorm)
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

# 4. Run Autoregressive Inference
let prompt = "T is a"
print(f"\nInput prompt: \"{prompt}\"")
print("Streaming autoregressive token generation:")
print("---------------------------------------------------------")

mut generated_count = 0
let t_start = time.clock()

# Streaming token print callback
fn stream_print(token_str):
    mut count_ref = generated_count + 1
    load("__sys").stdout_write(token_str)

let output = model.generate(tok, prompt, 12, 0.8, 20, 0.9, stream_print)
let t_end = time.clock()
let elapsed = t_end - t_start

print("\n---------------------------------------------------------")
let speed = 12.0 / elapsed if elapsed > 0 else 0.0
let speed_rounded = round(speed * 10) / 10.0
let elapsed_ms = round(elapsed * 10000) / 10.0

print(f"Inference Summary:")
print(f"  - Latency: {elapsed_ms} ms")
print(f"  - Throughput: {speed_rounded} tokens/sec")
print(f"  - Engine: T JIT x86-64 + KV-Cache + GQA + SwiGLU")
print("=========================================================")
