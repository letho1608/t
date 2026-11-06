# Ngôn ngữ lập trình T — README (v5.1: 100% tự thân thuần `.t`, VM C + x86-64 JIT, Tensor n-D & Autograd)

Ngôn ngữ lập trình **T**: cú pháp thụt lề hiện đại (Python/Go-like), **Tensor n-chiều là kiểu nguyên thủy**, **AI-first**, hoàn toàn độc lập không phụ thuộc Python:

| Thành phần | Lệnh | Ghi chú |
|---|---|---|
| Chạy trực tiếp file `.t` / `-c` | `trun prog.t` / `trun -c "print(1+2)"` | VM tự biên dịch ngầm qua `tcomp.tb` và thực thi với JIT |
| Chạy bytecode `.tb` | `trun prog.tb` | Khởi động tức thời + x86-64 JIT + REPL (`trun`) |
| Compiler tự thân thuần T | `trun tcomp.tb prog.t prog.tb` | `self/tcomp.t` tự biên dịch chính nó tới fixed point byte-identical |
| Evaluator tự thân thuần T | `trun tools/tdriver.t prog.t` | `self/eval.t` chạy trên VM C khớp từng byte |

## Build + Kiểm thử + Benchmark

```powershell
vm\build.bat                    # Windows (MSVC): build trun.exe + tự host tcomp.tb + trepl.tb
sh vm/build.sh                  # Linux/macOS (cc): build ./trun + tự host tcomp.tb + trepl.tb

.\trun.exe tools\run_all.t      # Chạy TOÀN BỘ 50 bộ test (7 subsystem + 43 tests/*.t) -> ALL-GREEN
.\trun.exe bench\bench.t        # Benchmark T (VM + JIT) đối chiếu trực tiếp với Python
```

## Giới hạn đã biết & cảnh báo an toàn

- **Đơn luồng cho interpreter**: chính VM thực thi đơn luồng tại một thời điểm. `std/thread.t` là worker/channel riêng biệt, **không chia sẻ trạng thái VM**. Ngoại lệ: phép nhân ma trận lớn (`@`) đã **đa luồng + chặn cache** ở native (`TVM_GEMM_THREADS`, xem TODO mục T2). Đa luồng đầy đủ nằm trong lộ trình (TODO J7... xem §6).
- **JIT chỉ x86-64**: `vm/jit_x64.c` sinh mã cho Win64 + System V; JIT ARM64 chỉ ở mức skeleton (`vm/jit_arm64.c` emitter + kiểm chứng byte qua `__sys.arm64_selftest`, chưa nối vào đường nóng — xem J7 trong TODO). Nguồn không được JIT vẫn chạy qua interpreter (deopt); trên ARM64 VM tự vô hiệu hoá JIT x64 để không phát nhầm mã.
- **`for` khai báo sau hàm**: closure không capture biến `for` khai báo sau điểm định nghĩa hàm (`tests/hoist.t` ghi nhận hành vi này).
- **`==`/`!=` trên tensor** so sánh cả tensor và trả `bool`; dùng `.eq()`/`.ne()` để so sánh từng phần tử (`<`, `>`, `<=`, `>=` là so sánh từng phần tử).
- **`std/ffi.t` + primitive `mem_*` là unsafe**: đọc/ghi bộ nhớ thô theo con trỏ — ghi sai địa chỉ có thể làm hỏng dữ liệu hoặc crash tiến trình. Chỉ dùng khi đã tối ưu phần còn lại.
- Một số thông điệp lỗi hiếm (vd. unhashable dict key) có thể khác văn bản giữa các runtime, dù output chính vẫn giống nhau.

## Tính năng ngôn ngữ (v5.1)

```python
let x = 10                      # bất biến (gán lại -> lỗi "x immutable (use mut)")
mut n = 0                       # khả biến
n += 5                          # += -= *= /= %= &= |= ^= <<= >>=
y = "lớn" if x > 3 else "nhỏ"   # biểu thức điều kiện

# Toán tử bitwise 64-bit (&, |, ^, ~, <<, >>) được JIT biên dịch thẳng sang mã máy x86-64
mask = (0b1100 & 0b1010) | (~0 & 3) ^ (1 << 4)

fn scale(a, factor = 2, bias = 0):   # tham số mặc định (default parameters)
    return a * factor + bias

fn log_event(level, *items):         # tham số biến thiên (*args)
    print(f"[{level}]", " ".join([str(x) for x in items]))

try:                                 # try / except / finally
    res = 10 / 0
except div0:
    print("bắt lỗi chia 0")
finally:
    print("dọn dẹp tài nguyên")

for i, v in enumerate(["a", "b"]):   # nhiều biến lặp
    print(i, v)

switch v:                       # rẽ nhiều nhánh
    case 1, 2:
        print("nhỏ")
    default:
        print("khác")

struct Point:                   # bản ghi (record) + method
    x, y
    fn norm2(self):
        return self.x * self.x + self.y * self.y
    fn move(self, dx):
        self.x += dx
        return self

p = Point(3, 4)
print(p.norm2(), p.move(1).x)

sq = fn(x): x * x               # lambda + list/dict comprehension
print([x * x for x in range(5) if x % 2 == 0])
```

## Thư viện chuẩn (`std/*.t`)

- `import math`: hằng số `pi`, `e`, `tau`, `inf`, `nan`; hàm `sin`, `cos`, `tan`, `asin`, `acos`, `atan`, `atan2`, `sinh`, `cosh`, `tanh`, `exp`, `log`, `log2`, `log10`, `sqrt`, `pow`, `abs`, `floor`, `ceil`, `round`, `trunc`, `hypot`, `deg`, `rad`, `gcd`, `lcm`, `factorial`, `comb`, `perm`, `isclose`, `isnan`, `isinf`, `clamp`, `lerp`.
- `import json`: `json.loads(s)`, `json.dumps(v, indent = 0)` chuẩn RFC 8259 (hỗ trợ đầy đủ escape Unicode `\uXXXX`, lồng ghép dict/list/tensor).
- `import os`: `platform`, `exepath`, `getcwd`, `chdir`, `abspath`, `getenv`, `setenv`, `exists`, `isfile`, `isdir`, `mtime`, `mkdir`, `makedirs`, `remove`, `rmtree`, `listdir`, `mktemp`, `read_file`, `write_file`, `copy_file`, `sha256`, `sha256_file`, `run` (khởi chạy tiến trình con trực tiếp qua OS API không qua shell, chống CWE-78 command injection), `zip_pack`, `zip_unpack` (chống CWE-22 zip-slip), `exit`.
- `import path`: `sep`, `join`, `basename`, `dirname`, `ext`, `stem`, `normpath`, `to_slash`, `isabs`.
- `import time`: `clock`, `time`, `sleep`, `measure`.
- `import random`: PRNG 64-bit có hạt giống (`seed`, `next_int`, `random`, `uniform`, `randint`, `randrange`, `choice`, `shuffle`, `sample`, `gauss`, `normal`).
- `import re`: Bộ phân tích và khớp biểu thức chính quy (`compile`, `match`, `search`, `test`, `fullmatch`, `findall`, `sub`, `split`, nhóm bắt `(...)`, tập ký tự `[...]`, định lượng `{m,n}`, escape `\d`, `\w`, `\s`, `\b`).
- `import csv`: Xử lý CSV chuẩn RFC 4180 (`parse_row`, `format_row`, `loads`, `dumps`, `loads_dicts`, `dumps_dicts`, `read_csv`, `write_csv`).
- `import http`: Tiện ích URL & HTTP client (`urlencode`, `urldecode`, `parse_query`, `build_query`, `parse_url`, `request`, `get`, `post`).
- `import autograd as ag`: Reverse-mode automatic differentiation trên Tensor n-chiều (`param`, `const`, `add`, `sub`, `mul`, `div`, `matmul`, `relu`, `tanh`, `sigmoid`, `exp`, `log`, `sumall`, `mean`, `softmax`, `gelu`, `rms_norm`, `layer_norm`, `swiglu` — backward của 5 phép cuối dùng kernel C gộp `softmax_bwd`/`gelu_bwd`/`rmsnorm_bwd`/`layernorm_bwd`/`swiglu_bwd`, `backward`, `sgd`).
- `import ai`: Tiện ích học máy (`mse`, `grad`, `linear`, `relu`, `gelu`, `softmax`, `dot`, `zeros`, `ones`).
- `import sqlite`: Client SQLite thuần T nối thẳng driver OS sẵn có (`winsqlite3.dll` / `libsqlite3.so`, 0 phụ thuộc cài đặt), transaction, prepared statement, bind tham số `?`.
- `import ffi`: Nạp `.dll`/`.so`, gọi hàm C (tới 16 đối số đa kiểu) và primitive bộ nhớ thô (`mem_alloc`, `mem_read_ptr`, `mem_write_i32`, `mem_read_f64`, `mem_write_f64`, `Buffer`) — **unsafe**, xem mục Giới hạn.
- `import mmap`: Ánh xạ file lớn vào RAM zero-copy, nạp thẳng tensor float/quant từ đĩa.
- `import gguf` / `import safetensors`: Đọc định dạng mô hình GGUF v2/v3 và Hugging Face Safetensors (metadata, catalog tensor, offset) qua mmap.
- `import quant`: Lượng tử hóa & giải lượng tử hóa FP16, Q8_0, Q4_0 (giảm ~85% dung lượng mã).
- `import transformer`: Suy luận LLaMA/Mistral/Qwen trên CPU (RMSNorm, RoPE, KV-Cache, GQA, SwiGLU, BPE tokenizer, sampler top-k/top-p/temperature).
- `import cuda`: Truy vấn driver CUDA qua FFI (thiết bị, VRAM, phiên bản) với fallback CPU êm khi không có GPU.
- `import cublas`: GEMM trên GPU NVIDIA qua cuBLAS (`gemm`/`matmul` double-precision, qua FFI 16 đối số + `tensor.data_ptr()`), không có GPU/lib thì tự fallback về `@` CPU.
- `import vulkan`: Probe Vulkan Loader + liệt kê physical devices mọi hãng (API, `device_count`, `devices`) qua FFI; compute shader SPIR-V chưa có nên `gemm` fallback CPU êm.
- `import base64`: Mã hóa/giải mã Base64 RFC 4648 (padding `=`/`==` chuẩn).
- `import thread`: Worker threads + kênh MPMC cách ly bộ nhớ (xem mục Giới hạn: không chia sẻ VM).

## Các chương trình mẫu (`examples/`)

- `trun examples/hello_tour.t`: Tổng quan toàn bộ cú pháp hiện đại của T (struct, method, bitwise, *args, try/except/finally, comprehension).
- `trun examples/neural_xor.t`: Huấn luyện mạng nơ-ron 2 lớp MLP giải bài toán phi tuyến XOR bằng autograd & SGD.
- `trun examples/data_pipeline.t`: Pipeline sinh dữ liệu CSV, lọc bằng Regex, tính toán thống kê và xuất báo cáo JSON.
- `trun examples/raytracer_ascii.t`: Trình dò tia 3D chiếu sáng kết xuất quả cầu ASCII art trực tiếp trên terminal.

## Hỗ trợ IDE & VS Code (`editors/vscode/`)

Đã đóng gói tiện ích mở rộng cho VS Code:
- Định vị cú pháp TextMate Grammar chuẩn (`.tmLanguage.json`).
- Tự động đóng ngoặc, thụt lề thông minh 4 spaces.
- Tích hợp Language Server chuẩn (`tools/tlsp.t`).

## JIT x86-64 (`vm/jit_x64.c`) — Hoàn tất toàn bộ mục tiêu G1–G12

- Biên dịch hàm sang mã máy x86-64 (Windows Win64 ABI & POSIX System V ABI) + **OSR** cho vòng lặp trong hàm và vòng lặp top-level.
- Hỗ trợ: `int`/`float`/`bool`/`nil`, biến cục bộ, `if/while/for range`, đệ quy & gọi hàm JIT chéo, đọc/ghi phần tử `list` có guard, tạo & đọc/ghi field `struct` (`SETATTR` cả ở OSR lẫn entry G8), hàm đệ quy trả `struct` (G9), cờ sở hữu động trên stack frame native cho biến nâng cấp `borrowed -> owned` (G10), `dict` (`{}`, literal `{"k": v}`, `d[k]`, `d[k] = v`, `d.get`, `d.has`, `d.remove`, `if d`) và chuỗi sở hữu được (`str()`, nối chuỗi, `s[i]`, truyền chuỗi mới vào hàm T, G7/G11/G12).
- Bảo mật VM (`vm/tvm.c`): Bộ kiểm tra bytecode (`verify_bytecode`) kiểm tra tính hợp lệ của opcode, độ dài lệnh, chỉ số global, bảng `LN01`/`EX01` trước khi nạp; khởi tạo tiến trình con không qua shell; kiểm tra đường dẫn chống path traversal khi giải nén gói.

## Bộ công cụ thuần T (`tools/*.t`)

| Công cụ | Lệnh |
|---|---|
| REPL | `trun` (nạp `trepl.tb` biên dịch từ `tools/repl.t`) |
| Debugger | `trun -d prog.tb` (`s` step, `b pc`, `c`, `p` stack, `l` locals, `g`, `d` disasm) |
| Formatter | `trun tools/ffmt.t [--check] prog.t` |
| LSP Server | `trun tools/tlsp.t` (hover, definition, references, rename, completion, symbols, diagnostics) |
| Package Manager | `trun tools/tpkg.t init\|install\|run\|test\|pack\|publish\|search\|release` |
| Test Runner | `trun tools/run_all.t` |
| Benchmark | `trun bench/bench.t` |
