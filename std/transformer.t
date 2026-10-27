# std/transformer.t — Industrial-Grade Transformer Inference Engine cho T
# Kiến trúc tương thích LLaMA, Mistral, Qwen, TinyLlama (RMSNorm, RoPE, KV-Cache, GQA, SwiGLU)

import math
import random
import gguf

# =========================================================================
# 1. Các khối toán học cốt lõi (Core Math Blocks)
# =========================================================================

fn rms_norm(x, weight, eps = 1e-5):
    # x: [dim], weight: [dim]
    let mean_sq = (x * x).mean()
    let scale = 1.0 / math.sqrt(mean_sq + eps)
    return (x * scale) * weight

fn rope_apply(v, pos, head_dim, base = 10000.0):
    # v: [head_dim], áp dụng Rotary Position Embeddings
    mut out = v.copy()
    let half = int(head_dim / 2)
    for i in range(half):
        let theta = 1.0 / math.pow(base, (2.0 * i) / head_dim)
        let alpha = pos * theta
        let cos_a = math.cos(alpha)
        let sin_a = math.sin(alpha)
        let x0 = v[i * 2]
        let x1 = v[i * 2 + 1]
        out[i * 2]     = x0 * cos_a - x1 * sin_a
        out[i * 2 + 1] = x0 * sin_a + x1 * cos_a
    return out

fn swiglu(x, w_gate, w_up, w_down):
    # SwiGLU FFN: (silu(x @ w_gate) * (x @ w_up)) @ w_down
    let gate = x @ w_gate
    let up = x @ w_up
    let silu_gate = gate * gate.sigmoid()
    let h = silu_gate * up
    return h @ w_down

# =========================================================================
# 2. KV-Cache & Grouped-Query Attention (GQA)
# =========================================================================

struct KVCache:
    n_layers
    max_seq_len
    k_cache  # [layer][pos] = k tensor [n_kv_heads, head_dim]
    v_cache  # [layer][pos] = v tensor [n_kv_heads, head_dim]
    cur_pos

fn create_kv_cache(n_layers, max_seq_len = 2048):
    mut k_c = []
    mut v_c = []
    for i in range(n_layers):
        k_c.append([])
        v_c.append([])
    return KVCache(n_layers, max_seq_len, k_c, v_c, 0)

fn attention_gqa(q, k_new, v_new, layer_idx, pos, kv_cache, n_heads, n_kv_heads, head_dim):
    # q: [n_heads, head_dim]
    # k_new: [n_kv_heads, head_dim]
    # v_new: [n_kv_heads, head_dim]
    # Append to cache
    kv_cache.k_cache[layer_idx].append(k_new)
    kv_cache.v_cache[layer_idx].append(v_new)

    let k_history = kv_cache.k_cache[layer_idx]
    let v_history = kv_cache.v_cache[layer_idx]
    let seq_len = len(k_history)
    let group_size = int(n_heads / n_kv_heads)
    let scale = 1.0 / math.sqrt(head_dim)

    mut head_contexts = []
    for h in range(n_heads):
        let kv_h = int(h / group_size)
        let q_h = q[h]

        # Calculate attention scores for this head across all positions
        mut scores = []
        for p in range(seq_len):
            let k_p = k_history[p][kv_h]
            let score = (q_h @ k_p) * scale
            scores.append(score)

        let scores_t = tensor(scores)
        let weights = scores_t.softmax()

        # Compute weighted sum of values
        mut context = zeros(head_dim)
        for p in range(seq_len):
            let w = weights[p]
            let v_p = v_history[p][kv_h]
            context = context + v_p * w
        head_contexts.append(context)

    # Flatten head contexts: [n_heads * head_dim]
    mut flat = []
    for h in range(n_heads):
        let c = head_contexts[h]
        for d in range(head_dim):
            flat.append(c[d])
    return tensor(flat)

# =========================================================================
# 3. Transformer Layer
# =========================================================================

struct TransformerLayer:
    idx
    n_heads
    n_kv_heads
    head_dim
    dim
    # Weights
    attn_norm_w
    wq
    wk
    wv
    wo
    ffn_norm_w
    w_gate
    w_up
    w_down

    fn forward(self, x, pos, kv_cache):
        # 1. Input RMSNorm
        let norm_x = rms_norm(x, self.attn_norm_w)

        # 2. Q, K, V Projections
        let q_flat = norm_x @ self.wq
        let k_flat = norm_x @ self.wk
        let v_flat = norm_x @ self.wv

        # Reshape into heads
        mut q = q_flat.reshape([self.n_heads, self.head_dim])
        mut k = k_flat.reshape([self.n_kv_heads, self.head_dim])
        let v = v_flat.reshape([self.n_kv_heads, self.head_dim])

        # Apply RoPE
        for h in range(self.n_heads):
            q[h] = rope_apply(q[h], pos, self.head_dim)
        for h in range(self.n_kv_heads):
            k[h] = rope_apply(k[h], pos, self.head_dim)

        # 3. Attention
        let attn_out = attention_gqa(q, k, v, self.idx, pos, kv_cache,
                                     self.n_heads, self.n_kv_heads, self.head_dim)
        let projected_attn = attn_out @ self.wo

        # Residual 1
        let h1 = x + projected_attn

        # 4. FFN RMSNorm + SwiGLU
        let norm_h1 = rms_norm(h1, self.ffn_norm_w)
        let ffn_out = swiglu(norm_h1, self.w_gate, self.w_up, self.w_down)

        # Residual 2
        return h1 + ffn_out

# =========================================================================
# 4. Sampler & Tokenizer
# =========================================================================

struct Tokenizer:
    vocab      # token str -> id
    inv_vocab  # list or map id -> token str
    bos_id
    eos_id

    fn encode(self, text):
        mut tokens = []
        if self.bos_id != nil:
            tokens.append(self.bos_id)
        # Simple whitespace & subword tokenize
        let words = text.split(" ")
        for i in range(len(words)):
            let w = words[i]
            let piece = " " + w if i > 0 else w
            if self.vocab.has(piece):
                tokens.append(self.vocab[piece])
            elif self.vocab.has(w):
                tokens.append(self.vocab[w])
            else:
                # byte / char fallback
                for c in w:
                    if self.vocab.has(c):
                        tokens.append(self.vocab[c])
        return tokens

    fn decode(self, token_ids):
        mut parts = []
        for tid in token_ids:
            if tid == self.bos_id or tid == self.eos_id:
                continue
            if tid < len(self.inv_vocab):
                parts.append(self.inv_vocab[tid])
        return "".join(parts)

fn sample_logits(logits, temperature = 0.7, top_k = 40, top_p = 0.9):
    if temperature <= 0.001:
        # Greedy argmax
        return logits.argmax()

    # Scale by temperature
    let scaled = logits / temperature
    let probs = scaled.softmax()
    let flat_probs = probs.tolist()

    # Top-K cutoff
    let n = len(flat_probs)
    mut indexed = []
    for i in range(n):
        indexed.append([flat_probs[i], i])

    # Sort descending
    let sorted_indexed = sort(indexed, fn(x): 0.0 - x[0])

    let k_lim = top_k if top_k < n and top_k > 0 else n
    mut cumsum = 0.0
    mut selected = []
    mut sum_p = 0.0
    for i in range(k_lim):
        let p = sorted_indexed[i][0]
        let idx = sorted_indexed[i][1]
        cumsum += p
        selected.append([p, idx])
        sum_p += p
        if cumsum >= top_p:
            break

    # Sample from selected
    let r = random.random() * sum_p
    mut acc = 0.0
    for item in selected:
        acc += item[0]
        if acc >= r:
            return item[1]
    return selected[0][1]

# =========================================================================
# 5. Full Transformer Model & Generation Engine
# =========================================================================

struct TransformerModel:
    vocab_size
    dim
    n_layers
    n_heads
    n_kv_heads
    head_dim
    hidden_dim
    # Weights
    tok_embeddings
    layers
    norm_w
    output_w

    fn forward_token(self, token_id, pos, kv_cache):
        # 1. Embedding lookup
        mut x = self.tok_embeddings[token_id].copy()

        # 2. Forward through all layers
        for layer in self.layers:
            x = layer.forward(x, pos, kv_cache)

        # 3. Final RMSNorm
        let norm_x = rms_norm(x, self.norm_w)

        # 4. LM Head projection -> logits [vocab_size]
        let logits = norm_x @ self.output_w
        return logits

    fn generate(self, tokenizer, prompt_text, max_new_tokens = 30,
                 temperature = 0.7, top_k = 40, top_p = 0.9, on_token = nil):
        let prompt_tokens = tokenizer.encode(prompt_text)
        let kv_cache = create_kv_cache(self.n_layers)

        # Ingest prompt
        mut pos = 0
        mut logits = nil
        for tid in prompt_tokens:
            logits = self.forward_token(tid, pos, kv_cache)
            pos += 1

        mut generated = []
        for step in range(max_new_tokens):
            let next_token = sample_logits(logits, temperature, top_k, top_p)
            if next_token == tokenizer.eos_id:
                break
            generated.append(next_token)
            if on_token != nil:
                let token_str = tokenizer.inv_vocab[next_token]
                on_token(token_str)
            logits = self.forward_token(next_token, pos, kv_cache)
            pos += 1

        return tokenizer.decode(generated)

# =========================================================================
# 6. Tải mô hình trực tiếp từ file GGUF
# =========================================================================

fn match_matmul_weight(w, in_dim):
    if w == nil:
        return nil
    if w.ndim == 2 and w.shape[0] != in_dim and w.shape[1] == in_dim:
        return w.T
    return w

fn load_model_from_gguf(gguf_path):
    let reader = gguf.open_gguf(gguf_path)
    if reader == nil:
        return nil
    let arch = reader.metadata.get("general.architecture", "llama")
    let dim = reader.metadata.get(arch + ".embedding_length", 0)
    let n_layers = reader.metadata.get(arch + ".block_count", 0)
    let n_heads = reader.metadata.get(arch + ".attention.head_count", 0)
    let n_kv_heads = reader.metadata.get(arch + ".attention.head_count_kv", n_heads)
    let ffn_hidden = reader.metadata.get(arch + ".feed_forward_length", 0)
    let head_dim = int(dim / n_heads) if n_heads > 0 else 0

    # Xây dựng Tokenizer từ metadata
    let tokens = reader.metadata.get("tokenizer.ggml.tokens", [])
    let bos_id = reader.metadata.get("tokenizer.ggml.bos_token_id", 1)
    let eos_id = reader.metadata.get("tokenizer.ggml.eos_token_id", 2)
    mut vocab = {}
    for i in range(len(tokens)):
        vocab[tokens[i]] = i
    let tokenizer = Tokenizer(vocab, tokens, bos_id, eos_id)

    # Nạp các tensor trọng số
    let tok_embd = reader.get_tensor("token_embd.weight")
    let norm_w = reader.get_tensor("output_norm.weight")
    mut output_w = reader.get_tensor("output.weight")
    if output_w == nil and tok_embd != nil:
        output_w = tok_embd.T

    output_w = match_matmul_weight(output_w, dim)

    mut layers = []
    for i in range(n_layers):
        let prefix = "blk." + str(i) + "."
        let attn_norm_w = reader.get_tensor(prefix + "attn_norm.weight")
        let wq = match_matmul_weight(reader.get_tensor(prefix + "attn_q.weight"), dim)
        let wk = match_matmul_weight(reader.get_tensor(prefix + "attn_k.weight"), dim)
        let wv = match_matmul_weight(reader.get_tensor(prefix + "attn_v.weight"), dim)
        let wo = match_matmul_weight(reader.get_tensor(prefix + "attn_output.weight"), dim)
        let ffn_norm_w = reader.get_tensor(prefix + "ffn_norm.weight")
        let w_gate = match_matmul_weight(reader.get_tensor(prefix + "ffn_gate.weight"), dim)
        let w_up = match_matmul_weight(reader.get_tensor(prefix + "ffn_up.weight"), dim)
        let w_down = match_matmul_weight(reader.get_tensor(prefix + "ffn_down.weight"), ffn_hidden)

        let layer = TransformerLayer(
            i, n_heads, n_kv_heads, head_dim, dim,
            attn_norm_w, wq, wk, wv, wo,
            ffn_norm_w, w_gate, w_up, w_down
        )
        layers.append(layer)

    let vocab_size = len(tokens) if len(tokens) > 0 else (tok_embd.shape[0] if tok_embd != nil else 0)
    let model = TransformerModel(
        vocab_size, dim, n_layers, n_heads, n_kv_heads,
        head_dim, ffn_hidden, tok_embd, layers, norm_w, output_w
    )
    return [model, tokenizer, reader]
