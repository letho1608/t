# tests/transformer_test.t — Kiểm thử Transformer Inference Engine

import transformer
import math

print("=== 1. Kiểm thử RMSNorm ===")
let x = tensor([1.0, 2.0, 3.0, 4.0])
let w_norm = ones(4)
let norm_x = transformer.rms_norm(x, w_norm)
# RMS = sqrt(mean(x^2)) = sqrt((1 + 4 + 9 + 16) / 4) = sqrt(7.5) = 2.7386
let expected_rms = math.sqrt(7.5)
let diff_norm = (norm_x - (x / expected_rms)).abs().max()
print("RMSNorm max diff:", diff_norm)
assert diff_norm < 1e-5

print("=== 2. Kiểm thử RoPE (Rotary Position Embeddings) ===")
let v = tensor([1.0, 0.0, 0.0, 1.0]) # dim 4, 2 heads of 2 or 1 head of 4
let v_pos0 = transformer.rope_apply(v, 0, 4)
# pos 0 -> góc xoay = 0 -> không đổi
let diff_pos0 = (v - v_pos0).abs().max()
assert diff_pos0 < 1e-5

# RoPE bảo toàn độ dài vector (norm2)
let v_pos5 = transformer.rope_apply(v, 5, 4)
let norm_before = (v * v).sum()
let norm_after = (v_pos5 * v_pos5).sum()
print("RoPE norm2 trước/sau:", norm_before, norm_after)
mut diff_norm_rot = norm_before - norm_after
if diff_norm_rot < 0:
    diff_norm_rot = -diff_norm_rot
assert diff_norm_rot < 1e-5

print("=== 3. Kiểm thử SwiGLU FFN ===")
let dim = 8
let hidden_dim = 16
let x_ffn = ones(dim)
let w_gate = eye(dim).reshape([dim, dim]) # dùng ma trận đơn vị mở rộng
let w_up = ones([dim, hidden_dim]) * 0.1
let w_down = ones([hidden_dim, dim]) * 0.1
let ffn_out = transformer.swiglu(x_ffn, w_up, w_up, w_down)
assert ffn_out.shape[0] == dim
print("SwiGLU output shape:", ffn_out.shape)

print("=== 4. Kiểm thử Tokenizer ===")
let vocab = {
    "<bos>": 0,
    "<eos>": 1,
    "Xin": 2,
    " chào": 3,
    " T": 4,
    " ngôn": 5,
    " ngữ": 6
}
let inv_vocab = ["<bos>", "<eos>", "Xin", " chào", " T", " ngôn", " ngữ"]
let tok = transformer.Tokenizer(vocab, inv_vocab, 0, 1)

let ids = tok.encode("Xin chào T")
print("Encoded tokens:", ids)
assert ids[0] == 0 and ids[1] == 2 and ids[2] == 3 and ids[3] == 4

let decoded = tok.decode(ids)
print("Decoded text:", decoded)
assert decoded == "Xin chào T"

print("=== 5. Kiểm thử Mô hình Transformer & Sinh từ (Autoregressive Generation) ===")
let vocab_size = len(inv_vocab)
let model_dim = 16
let n_layers = 2
let n_heads = 4
let n_kv_heads = 2 # Grouped-Query Attention (GQA)
let head_dim = int(model_dim / n_heads) # 4
let kv_dim = int(n_kv_heads * head_dim) # 8
let ffn_hidden = 32

# Khởi tạo trọng số ngẫu nhiên có kiểm soát
let tok_embd = ones([vocab_size, model_dim]) * 0.2

mut layers = []
for i in range(n_layers):
    let layer = transformer.TransformerLayer(
        i, n_heads, n_kv_heads, head_dim, model_dim,
        ones(model_dim),                          # attn_norm_w
        ones([model_dim, model_dim]) * 0.05,       # wq
        ones([model_dim, kv_dim]) * 0.05,          # wk
        ones([model_dim, kv_dim]) * 0.05,          # wv
        ones([model_dim, model_dim]) * 0.05,       # wo
        ones(model_dim),                          # ffn_norm_w
        ones([model_dim, ffn_hidden]) * 0.05,      # w_gate
        ones([model_dim, ffn_hidden]) * 0.05,      # w_up
        ones([ffn_hidden, model_dim]) * 0.05       # w_down
    )
    layers.append(layer)

let model = transformer.TransformerModel(
    vocab_size, model_dim, n_layers, n_heads, n_kv_heads,
    head_dim, ffn_hidden, tok_embd, layers,
    ones(model_dim),                         # norm_w
    ones([model_dim, vocab_size]) * 0.1      # output_w (lm_head)
)

print("Khởi tạo TransformerModel hoàn tất. Bắt đầu sinh văn bản dạng luồng:")
mut streamed_tokens = []
fn on_token_callback(token_str):
    streamed_tokens.append(token_str)

let generated_text = model.generate(tok, "Xin chào", 5, 0.0, 40, 0.9, on_token_callback)
print("Văn bản sinh ra:", generated_text)
print("Các token luồng nhận được:", streamed_tokens)
assert len(streamed_tokens) > 0

print("=== TRANSFORMER TEST PASSED ===")
