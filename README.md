# Ngôn ngữ lập trình T — README (v7.0: 100% tự thân thuần `.t`, VM C + JIT x86-64/ARM64 + AOT Native, Multi-Dtype Tensor, GPU Vulkan/CUDA, Autograd & Rust-Grade Optimizations)

Ngôn ngữ lập trình **T**: cú pháp thụt lề hiện đại (Python/Go-like), **Tensor n-chiều đa kiểu là kiểu nguyên thủy**, **AI-first**, tích hợp **3 cơ chế tối ưu chuẩn Rust (`J9/AOT`, `L12` `Result`/`?`/Borrow-Check, `S1` `#[repr(C)]`/Lock-Free Concurrency)**:

| Thành phần | Lệnh | Ghi chú |
|---|---|---|
| Chạy trực tiếp file `.t` / `-c` | `trun prog.t` / `trun -c "print(1+2)"` | VM tự biên dịch ngầm qua `tcomp.tb` và thực thi với JIT |
| Chạy bytecode `.tb` | `trun prog.tb` | Khởi động tức thời + JIT x86-64 / ARM64 + REPL (`trun`) |
| Đóng gói file thực thi độc lập | `trun --build-exe prog.t app.exe` | Gộp VM + bytecode thành 1 file `.exe`/ELF chạy độc lập (`tpkg build --exe`) |
| Trình biên dịch AOT Native | `trun tools/taot.t --emit-c prog.t` | Dịch sang C99 / Rust hoặc thư viện mã máy gốc `.dll`/`.so` AVX2 (`std/aot.t`) |
| Kiểm tra kiểu & Borrow tĩnh | `trun tools/tcheck.t prog.t` | Kiểm tra tĩnh kiểu, Use-After-Move/Close & Exhaustiveness (`std/typecheck.t`) |
| Compiler tự thân thuần T | `trun tcomp.tb prog.t prog.tb` | `self/tcomp.t` tự biên dịch chính nó tới fixed point byte-identical |
| Evaluator tự thân thuần T | `trun tools/tdriver.t prog.t` | `self/eval.t` chạy trên VM C khớp từng byte |

## Build + Kiểm thử + Benchmark

```powershell
vm\build.bat                    # Windows (MSVC): build trun.exe + tự host tcomp.tb + trepl.tb
sh vm/build.sh                  # Linux/macOS (cc): build ./trun + tự host tcomp.tb + trepl.tb

.\trun.exe tools\run_all.t      # Chạy TOÀN BỘ 52 bộ test (7 subsystem + 45 tests/*.t) -> ALL-GREEN
.\trun.exe bench\bench.t        # Benchmark T (VM + JIT) đối chiếu trực tiếp với Python
.\trun.exe bench\rust_vs_t.t    # Benchmark T (v7.0) đối đầu trực tiếp với Rust/Native (-O2)
```

## Giới hạn đã biết & cảnh báo an toàn

- **Đơn luồng cho interpreter**: chính VM thực thi đơn luồng tại một thời điểm. `std/thread.t` là worker/channel riêng biệt, **không chia sẻ trạng thái VM**. Ngoại lệ: phép nhân ma trận lớn (`@`) đã **đa luồng + chặn cache** ở native (`TVM_GEMM_THREADS`, xem TODO mục T2).
- **JIT đa kiến trúc (x86-64 & ARM64)**: `vm/jit_x64.c` sinh mã cho Win64 + System V x86-64; `vm/jit_arm64.c` biên dịch Bytecode → AArch64 (Method JIT + OSR + Deopt + bộ giả lập thực thi mã máy AArch64 để kiểm chứng chéo trên x86-64). Nguồn không được JIT vẫn chạy qua interpreter (deopt).
- **`for` khai báo sau hàm**: closure không capture biến `for` khai báo sau điểm định nghĩa hàm (`tests/hoist.t` ghi nhận hành vi này).
- **`==`/`!=` trên tensor** so sánh cả tensor và trả `bool`; dùng `.eq()`/`.ne()` để so sánh từng phần tử (`<`, `>`, `<=`, `>=` là so sánh từng phần tử).
- **`std/ffi.t` + primitive `mem_*` là unsafe**: đọc/ghi bộ nhớ thô theo con trỏ — ghi sai địa chỉ có thể làm hỏng dữ liệu hoặc crash tiến trình. Chỉ dùng khi đã tối ưu phần còn lại.
- Một số thông điệp lỗi hiếm (vd. unhashable dict key) có thể khác văn bản giữa các runtime, dù output chính vẫn giống nhau.

## Tính năng ngôn ngữ (v5.2)

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
- `import autograd as ag`: Reverse-mode automatic differentiation trên Tensor n-chiều (`param`, `const`, `add`, `sub`, `mul`, `div`, `matmul`, `relu`, `tanh`, `sigmoid`, `exp`, `log`, `sumall`, `mean`, `softmax`, `gelu`, `rms_norm`, `layer_norm`, `swiglu`, `conv2d` — backward của 6 phép cuối dùng kernel C gộp `softmax_bwd`/`gelu_bwd`/`rmsnorm_bwd`/`layernorm_bwd`/`swiglu_bwd`/`conv2d_bwd`, `backward`, `sgd`).
- `import ai`: Tiện ích học máy (`mse`, `grad`, `linear`, `conv2d`, `relu`, `gelu`, `softmax`, `dot`, `zeros`, `ones`).
- `import sqlite`: Client SQLite thuần T nối thẳng driver OS sẵn có (`winsqlite3.dll` / `libsqlite3.so`, 0 phụ thuộc cài đặt), transaction, prepared statement, bind tham số `?`.
- `import ffi`: Nạp `.dll`/`.so`, gọi hàm C (tới 20 đối số đa kiểu) và primitive bộ nhớ thô (`mem_alloc`, `mem_read_ptr`, `mem_write_i32`, `mem_read_f32`, `mem_write_f32`, `mem_read_u16`, `mem_write_u16`, `mem_read_f64`, `mem_write_f64`, `Buffer`) — **unsafe**, xem mục Giới hạn.
- `import mmap`: Ánh xạ file lớn vào RAM zero-copy, nạp thẳng tensor float/quant từ đĩa.
- `import gguf` / `import safetensors`: Đọc định dạng mô hình GGUF v2/v3 và Hugging Face Safetensors (metadata, catalog tensor, offset) qua mmap.
- `import quant`: Lượng tử hóa & giải lượng tử hóa FP16, BF16, FP32, Q8_0, Q4_0 (giảm ~85% dung lượng mã).
- `import transformer`: Suy luận LLaMA/Mistral/Qwen trên CPU (RMSNorm, RoPE, KV-Cache, GQA, SwiGLU, BPE tokenizer, sampler top-k/top-p/temperature).
- `import cuda`: Truy vấn driver CUDA qua FFI (thiết bị, VRAM, phiên bản) với fallback CPU êm khi không có GPU.
- `import cublas` / `import cudnn`: GEMM trên GPU NVIDIA qua cuBLAS (`gemm`/`matmul` double-precision `cublasDgemm_v2`, Tensor Cores `gemm_fp16`/`gemm_bf16`/`gemm_f32`/`gemm_ex` qua `cublasGemmEx` 19 đối số) và tích chập 2D NCHW `conv2d` qua cuDNN (`cudnnConvolutionForward`), không có GPU/lib thì tự fallback về nhân C native / `@` CPU.
- `import vulkan`: Khởi tạo Vulkan Loader + liệt kê đầy đủ physical devices (`name`, `vendor_id`, `device_id`, `type`, `api_version`) + biên dịch và thực thi trực tiếp Compute Shader nhị phân SPIR-V 1.0 trên GPU (`gemm_f32` và `gemm`/`matmul` với thuật toán phân rã mantissa Ozaki 4-lớp đạt chuẩn float64 `< 1e-13`). Máy không có Vulkan tự fallback CPU êm.
- `import base64`: Mã hóa/giải mã Base64 RFC 4648 (padding `=`/`==` chuẩn).
- `import thread`: Worker threads + kênh MPMC cách ly bộ nhớ (xem mục Giới hạn: không chia sẻ VM).
- `import net`: TCP/UDP Sockets native (`tcp_listen`, `tcp_connect`, `udp_bind`, `TcpListener`, `TcpStream`, `UdpSocket`) và `HttpServer` nhúng.
- `import async`: Lập lịch bất đồng bộ hợp tác (`EventLoop`, `Future`, `spawn`, `gather`, `sleep`, `run`) kết hợp cùng `async fn` / `await`.
- `import typecheck`: Bộ kiểm tra kiểu tĩnh (`check_source`, `check_file`) cho các chú thích kiểu tùy chọn (`let x: int`, `fn f(a: int) -> str:`).

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

## JIT x86-64 (`vm/jit_x64.c`) & ARM64 (`vm/jit_arm64.c`) — Hoàn tất toàn bộ mục tiêu G1–G14 & J7–J8

- Biên dịch hàm sang mã máy x86-64 (Windows Win64 ABI & POSIX System V ABI) và ARM64 (AArch64 AAPCS64) + **OSR** cho vòng lặp trong hàm và vòng lặp top-level.
- Hỗ trợ: `int`/`float`/`bool`/`nil`, biến cục bộ, `if/while/for range`, đệ quy & gọi hàm JIT chéo, đọc/ghi phần tử `list` có guard, tạo & đọc/ghi field `struct` (`SETATTR` cả ở OSR lẫn entry G8), hàm đệ quy trả `struct` (G9), cờ sở hữu động trên stack frame native cho biến nâng cấp `borrowed -> owned` (G10), `dict` (`{}`, literal `{"k": v}`, `d[k]`, `d[k] = v`, `d.get`, `d.has`, `d.remove`, `if d`) và chuỗi sở hữu được (`str()`, nối chuỗi, `s[i]`, truyền chuỗi mới vào hàm T, G7/G11/G12).
- Trên ARM64 (`vm/jit_arm64.c`): Trình biên dịch Bytecode → AArch64 hoàn chỉnh với thông dịch trừu tượng suy diễn kiểu, 4 bản đặc hoá / hàm, số học `int64` kiểm tràn (`adds`/`subs` + `b.vs` deopt, floor-div/mod), số học `float64`, toán tử bitwise 64-bit, `list` (`OP_IDX`/`OP_SETIDX`/`OP_LEN`), `struct` (`OP_ATTR`/`OP_SETATTR`/`NAT___NEW`), `dict` (`OP_NEW_DICT`/`OP_IDX`/`OP_SETIDX`/`get`/`has`/`remove`), `str` (`OP_PUSH_STR`/`NAT_STR`/concat/eq/idx/len), đệ quy `bl`, OSR + Deopt, kèm bộ giả lập thực thi mã máy AArch64 (`a64_sim_exec`) kiểm chứng chéo ngay trên x86-64.
- Bảo mật VM (`vm/tvm.c`): Bộ kiểm tra bytecode (`verify_bytecode`) kiểm tra tính hợp lệ của opcode, độ dài lệnh, chỉ số global, bảng `LN01`/`EX01` trước khi nạp; khởi tạo tiến trình con không qua shell; kiểm tra đường dẫn chống path traversal khi giải nén gói.

## Bộ công cụ thuần T (`tools/*.t`)

| Công cụ | Lệnh |
|---|---|
| REPL | `trun` (nạp `trepl.tb` biên dịch từ `tools/repl.t`) |
| Debugger | `trun -d prog.tb` (`s` step, `b pc`, `c`, `p` stack, `l` locals, `g`, `d` disasm) |
| Type Checker | `trun tools/tcheck.t prog.t` |
| Formatter | `trun tools/ffmt.t [--check] prog.t` |
| LSP Server | `trun tools/tlsp.t` (hover, definition, references, rename, completion, symbols, diagnostics) |
| Package Manager | `trun tools/tpkg.t init\|install\|run\|test\|build [--exe]\|pack\|publish\|search\|release` |
| Test Runner | `trun tools/run_all.t` |
| Benchmark | `trun bench/bench.t` |
