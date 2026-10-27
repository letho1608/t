# tools/run_gguf.t — Trình thực thi mô hình LLM GGUF trực tiếp từ dòng lệnh
import transformer
import os
import time

let args = argv()

if len(args) < 1:
    print("==================================================================")
    print("   T GGUF Model Runner — Thực thi LLM Độc lập không cần Python   ")
    print("==================================================================")
    print("Cách dùng:")
    print("  trun tools/run_gguf.t <duong_dan_model.gguf> [\"câu prompt\"]")
    print("")
    print("Ví dụ:")
    print("  trun tools/run_gguf.t TinyLlama-1.1B-Chat-v1.0.Q4_0.gguf \"Xin chào\"")
    print("  trun tools/run_gguf.t qwen2.5-0.5b-instruct-q4_0.gguf \"Lập trình là gì?\"")
    print("==================================================================")
    os.exit(0)

let model_path = args[0]
let prompt = args[1] if len(args) > 1 else "Xin chào"

if not os.exists(model_path):
    print(f"Lỗi: Không tìm thấy tệp mô hình tại '{model_path}'")
    os.exit(1)

print(f"[*] Đang nạp mô hình qua Memory-Mapped I/O: {model_path}...")
let t0 = time.clock()
let loaded = transformer.load_model_from_gguf(model_path)
if loaded == nil:
    print("Lỗi: Không thể phân tích định dạng GGUF của tệp.")
    os.exit(1)

let model = loaded[0]
let tok = loaded[1]
let reader = loaded[2]
let t_load = time.clock() - t0
let load_ms = round(t_load * 10000) / 10.0

print(f"[*] Nạp mô hình thành công trong {load_ms} ms!")
print(f"    - Kiến trúc: {reader.metadata.get('general.architecture', 'llama')}")
print(f"    - Kích thước ẩn (dim): {model.dim}")
print(f"    - Số lớp (layers): {model.n_layers}")
print(f"    - Số Query heads: {model.n_heads}, KV heads: {model.n_kv_heads}")
print(f"    - Kích thước từ vựng (vocab): {model.vocab_size}")

print(f"\n[*] Prompt: \"{prompt}\"")
print("[*] Đang sinh phản hồi (Streaming Output):")
print("------------------------------------------------------------------")

let _sys = load("__sys")
fn on_token(token_str):
    _sys.stdout_write(token_str)

let t_gen_start = time.clock()
let generated_text = model.generate(tok, prompt, 64, 0.7, 40, 0.9, on_token)
let t_gen_end = time.clock()
let gen_elapsed = t_gen_end - t_gen_start
let tok_count = len(tok.encode(generated_text))
let speed = tok_count / gen_elapsed if gen_elapsed > 0 else 0.0
let speed_rounded = round(speed * 10) / 10.0

print("\n------------------------------------------------------------------")
print(f"[*] Hoàn tất trong {round(gen_elapsed * 1000) / 1000.0}s ({speed_rounded} tokens/giây)")

reader.close()
