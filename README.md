<div align="center">

# ⚡ Ngôn ngữ lập trình T (T Language)

**Ngôn ngữ lập trình hiện đại, tự thân 100%, AI-first với Tensor nguyên thủy, JIT kép x86-64 / ARM64, AOT Native & Tối ưu hiệu năng chuẩn Rust.**

[![CI Status](https://github.com/letho1608/t/actions/workflows/ci.yml/badge.svg)](https://github.com/letho1608/t/actions/workflows/ci.yml)
[![Version](https://img.shields.io/badge/version-v7.1-blue.svg)](https://github.com/letho1608/t)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Arch](https://img.shields.io/badge/arch-x86--64%20%7C%20ARM64-brightgreen.svg)](#)
[![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20Linux%20%7C%20macOS-lightgrey.svg)](#)
[![Tests](https://img.shields.io/badge/tests-52%2F52%20passing-success.svg)](#)
[![Self-Hosted](https://img.shields.io/badge/self--hosted-100%25%20pure%20.t-orange.svg)](#)

[Tính năng](#-tính-năng-nổi-bật) • [Cài đặt & Build](#-cài-đặt--xây-dựng) • [Cú pháp & Mẫu](#-cú-pháp--ví-dụ-mẫu) • [Tối ưu chuẩn Rust](#-tối-ưu-hiệu-năng-chuẩn-rust) • [Thư viện chuẩn](#-thư-viện-chuẩn-std) • [Hệ sinh thái](#-bộ-công-cụ-nhà-phát-triển) • [Bảo mật](#-bảo-mật--an-toàn)

</div>

---

## 📖 Giới thiệu

**T** là ngôn ngữ lập trình đa dụng thế hệ mới, kết hợp sự thanh lịch, cú pháp thụt lề trực quan như Python/Go với hiệu năng cực đại tiệm cận C/Rust:

- **100% Tự thân (Self-hosting)**: Compiler (`self/tcomp.t`), REPL (`tools/repl.t`), LSP Server (`tools/tlsp.t`), Formatter (`tools/ffmt.t`), Package Manager (`tools/tpkg.t`), Test Suite (`tools/run_all.t`) đều được viết thuần bằng chính ngôn ngữ **T** và tự biên dịch ra bytecode `.tb`.
- **AI-First & Tensor nguyên thủy**: Tensor đa chiều đa kiểu dữ liệu (`f64`, `f32`, `f16`, `bf16`, `i32`, `i8`) là kiểu primitive trực tiếp trong ngôn ngữ, tích hợp sẵn Autograd vi phân tự động (reverse-mode).
- **Tăng tốc phần cứng GPU**: Tích hợp trực tiếp Compute Shader Vulkan (SPIR-V 1.0) và NVIDIA cuBLAS / cuDNN (Tensor Cores FP16/BF16/FP32).
- **Runtime hiệu năng cao**: C VM tối ưu cache, JIT Compiler kép (x86-64 Win64/SysV ABI và ARM64 AAPCS64) với OSR (On-Stack Replacement) & Deoptimization linh hoạt.
- **Trình biên dịch AOT Native**: Dịch mã nguồn `.t` sang C99 / Rust hoặc sinh thư viện gốc `.dll`/`.so` AVX2 độc lập (`std/aot.t`).
- **An toàn & Song song chuẩn Rust**: Hệ thống kiểu `Result`/`Option` cùng toán tử `?`, bộ kiểm tra tĩnh Borrow/Move Checker (`std/typecheck.t`), struct mảng phẳng `#[repr(C)]` (`std/cstruct.t`), Rayon-style Native Thread Pool và cấu trúc dữ liệu không khóa (`std/thread.t`).
- **Đóng gói nhị phân độc lập**: Khả năng tạo 1 file thực thi nhị phân duy nhất (`.exe` / ELF) tự mang runtime và bytecode (`trun --build-exe`).

---

## 🚀 Bắt đầu nhanh

| Thao tác | Câu lệnh | Mô tả |
|---|---|---|
| **Chạy file nguồn `.t`** | `trun app.t` | Tự động biên dịch ngầm qua `tcomp.tb` và thực thi với JIT |
| **Thực thi biểu thức ngắn** | `trun -c "print(1 + 2 * 3)"` | Chạy nhanh một dòng lệnh |
| **Chạy bytecode `.tb`** | `trun app.tb` | Khởi động tức thì (Zero compile time) + JIT compilation |
| **Khởi chạy REPL** | `trun` | Mở môi trường dòng lệnh tương tác thông minh |
| **Biên dịch thủ công** | `trun tcomp.tb app.t app.tb` | Dùng compiler tự thân dịch mã sang file bytecode `.tb` |
| **Đóng gói File Thực thi** | `trun --build-exe app.t app.exe` | Đóng gói VM C + bytecode thành 1 file nhị phân duy nhất |
| **Biên dịch AOT Native** | `trun tools/taot.t --emit-c app.t` | Dịch nguồn T sang C99 hoặc Rust tối ưu hóa cao |
| **Kiểm tra kiểu & Borrow** | `trun tools/tcheck.t app.t` | Phân tích tĩnh, phát hiện Use-After-Move/Close & Exhaustiveness |

---

## 🛠️ Cài đặt & Xây dựng

### Yêu cầu hệ thống
- **Windows**: MSVC (`cl.exe` từ Visual Studio hoặc Build Tools)
- **Linux / macOS**: Trình biên dịch C (`gcc` hoặc `clang`)

### Build từ mã nguồn

```powershell
# Trên Windows (Visual Studio Developer Command Prompt)
vm\build.bat

# Trên Linux / macOS
sh vm/build.sh
```

Quá trình build sẽ:
1. Biên dịch VM C (`vm/tvm.c`, `vm/jit_x64.c`, `vm/jit_arm64.c`) thành file thực thi `trun` (`trun.exe`).
2. Tự động dùng `trun` để tự biên dịch compiler thuần T (`self/tcomp.t` -> `tcomp.tb`) và REPL (`tools/repl.t` -> `trepl.tb`).

### Kiểm thử & Benchmark

```powershell
# Chạy toàn bộ 52 bộ kiểm thử hệ thống (ALL-GREEN)
.\trun.exe tools\run_all.t

# Benchmark hiệu năng đối chiếu với Python
.\trun.exe bench\bench.t

# Benchmark so sánh trực diện với Rust/Native (-O2)
.\trun.exe bench\rust_vs_t.t
```

---

## ✨ Tính năng nổi bật

### 1. Cú pháp thụt lề thanh lịch & Mạnh mẽ
Ngôn ngữ sử dụng quy tắc phân cấp khối theo thụt lề (indentation-based), loại bỏ hoàn toàn các dấu ngoặc nhọn `{}` rườm rà.

```python
# Khai báo biến bất biến (let) và khả biến (mut)
let pi = 3.14159
mut counter = 0
counter += 1

# Biểu thức điều kiện 1 dòng
status = "chẵn" if counter % 2 == 0 else "lẻ"

# Toán tử bitwise 64-bit (JIT phát trực tiếp mã máy x86-64 / ARM64)
mask = (0b1100 & 0b1010) | (~0 & 3) ^ (1 << 4)

# Tham số mặc định & tham số biến thiên (*args)
fn log_info(module, *messages, prefix = "INFO"):
    print(f"[{prefix}][{module}]", " ".join([str(m) for m in messages]))

log_info("NET", "Đã kết nối", "cổng", 8080)
```

### 2. Hướng đối tượng & Structs với Method
```python
struct Point:
    x, y
    fn norm2(self):
        return self.x * self.x + self.y * self.y
    fn move(self, dx, dy):
        self.x += dx
        self.y += dy
        return self

p = Point(3, 4)
print(p.norm2())      # 25
p.move(1, 2)
print(p.x, p.y)       # 4, 6
```

### 3. Tensor Đa Chiều Đa Kiểu & Tự Động Vi Phân (Autograd)
Tensor trong T là kiểu nguyên thủy (primitive) hỗ trợ nhiều dtype: `f64`, `f32`, `f16`, `bf16`, `i32`, `i8` với toán tử nhân ma trận `@` vector hóa SIMD đa luồng.

```python
import autograd as ag
import ai

# Tạo tensor tham số có theo dõi đạo hàm
w = ag.param(tensor([[0.5, -0.2], [0.1, 0.8]]))
b = ag.param(tensor([[0.0, 0.0]]))
x = ag.const(tensor([[1.0, 2.0]]))

# Lan truyền tiến (Forward Pass)
z = ag.add(ag.matmul(x, w), b)
a = ag.gelu(z)

# Lan truyền ngược (Reverse-Mode Backward Pass)
ag.backward(a)

print("Đạo hàm dw:", w.grad)
```

### 4. Xử lý lỗi linh hoạt: `try/except` & Kiểu `Result<T, E>` / `?`
T hỗ trợ cả cơ chế bắt ngoại lệ truyền thống lẫn xử lý lỗi không tốn chi phí (zero-cost) theo chuẩn Rust:

```python
import result as r

fn divide(a, b):
    if b == 0:
        return r.Err("không thể chia cho 0")
    return r.Ok(a / b)

fn calculate():
    # Toán tử ? tự động giải nén Ok hoặc return sớm nếu là Err
    val = divide(10, 2)?
    return r.Ok(val * 3)

res = calculate()
if res.is_ok():
    print("Kết quả:", res.unwrap())
```

---

## ⚡ Tối ưu hiệu năng chuẩn Rust

T v7.1 tích hợp các cơ chế tối ưu chuyên sâu giúp hiệu năng mã T đạt mức ngang ngửa Rust thuần:

| Cơ chế | Module | Mô tả kỹ thuật |
|---|---|---|
| **LSD Radix Sort** | `std/vec.t` | Thuật toán sắp xếp 8-Pass $O(N)$ cho `VecI64`, nhanh hơn `qsort -O2` gấp **>6×**. |
| **Unboxed Vectors** | `std/vec.t` | `VecI64`, `VecF64` unboxed 8-byte liên tục trong bộ nhớ kèm Zero-Cost Iterator Fusion (`fused_map_filter_sum`). |
| **Fibonacci IntMap** | `std/fastmap.t` | Bảng băm mở unboxed dùng nhân băm Fibonacci ($2^{64}/\phi$), tối ưu L1/L2 cache. |
| **SIMD String Kernels** | `std/fastmap.t` | Quét chuỗi 8-way unrolled zero-copy (`str_count_byte`, `str_fnv1a_64`, `str_find_fast`). |
| **Rayon Thread Pool** | `std/thread.t` | Thread pool gốc C zero-spawn đa nhân cho tác vụ nặng (`par_collatz`, `par_reduce_i64`, `par_map_f64`). |
| **Flat Struct Array** | `std/cstruct.t` | Mảng cấu trúc phẳng `#[repr(C)]` zero-copy với L1 cache tiling và vector hóa SIMD. |
| **Concurrency Lock-Free** | `std/thread.t` | Nguyên hàm `AtomicInt`, `Mutex`, `SharedTensor` chia sẻ bộ nhớ zero-copy an toàn giữa các worker. |

```python
import vec
import thread

# 1. Sắp xếp siêu tốc với 8-Pass LSD Radix Sort
mut v = vec.VecI64()
for x in [54, 12, 999, 1, 450, 23]:
    v.push(x)
v.radix_sort()
print("Đã sắp xếp:", v.to_list())

# 2. Xử lý song song đa luồng kiểu Rayon (P1)
total = thread.par_reduce_i64(1, 1000000, 4)
print("Tổng song song:", total)
```

---

## 📚 Thư viện chuẩn (`std/`)

Thư viện chuẩn của T phong phú, sẵn sàng phục vụ từ phát triển hệ thống, mạng, tính toán khoa học cho tới AI hiện đại:

| Thư viện | Tên gọi | Chức năng chính |
|---|---|---|
| `math` | Toán học cơ bản | Đầy đủ hàm lượng giác, siêu việt, hằng số `pi`, `tau`, `e`, `isclose`, `clamp`, `gcd`, `lcm`. |
| `json` | Xử lý JSON | `loads`, `dumps` chuẩn RFC 8259 hỗ trợ Unicode `\uXXXX`, lồng ghép dict, list, tensor. |
| `os` | Tương tác HĐH | Thao tác file, thư mục, môi trường, SHA-256, chạy tiến trình con an toàn chống injection. |
| `path` | Đường dẫn | Phân tích và chuẩn hóa đường dẫn đa nền tảng (`join`, `basename`, `dirname`, `ext`). |
| `time` | Thời gian | Đo thời gian độ chính xác cao (`clock`, `measure`), lấy epoch time, sleep. |
| `random` | Số ngẫu nhiên | PRNG 64-bit có seed (`randint`, `uniform`, `shuffle`, `choice`, `gauss`). |
| `re` | Regular Expression | Regex engine hoàn chỉnh (`match`, `search`, `findall`, `sub`, `split`, capture group). |
| `csv` | Tệp CSV | Đọc/ghi CSV chuẩn RFC 4180 (`loads`, `dumps`, `read_csv`, `write_csv`). |
| `http` | HTTP & URL | URL encoding/query parser và HTTP Client (`get`, `post`, `request`). |
| `net` | Mạng Socket | Native TCP/UDP sockets (`TcpListener`, `TcpStream`, `UdpSocket`) và `HttpServer` nhúng. |
| `async` | Bất đồng bộ | EventLoop, Future, `spawn`, `gather`, hỗ trợ `async fn` / `await` mượt mà. |
| `autograd` | Vi phân tự động | Reverse-mode automatic differentiation với các kernel C gộp tối ưu cho Deep Learning. |
| `ai` | Machine Learning | Tiện ích học máy: `linear`, `conv2d`, `relu`, `gelu`, `softmax`, `mse`. |
| `transformer` | Suy luận LLM | Kiến trúc Transformer chạy CPU: RoPE, KV-Cache, GQA, SwiGLU, BPE tokenizer cho LLaMA/Mistral. |
| `vulkan` | GPU Vulkan | Khởi tạo Vulkan Loader, chạy Compute Shader SPIR-V trực tiếp trên mọi GPU (AMD/Intel/NVIDIA). |
| `cublas`/`cudnn` | GPU NVIDIA | Tăng tốc nhân ma trận và tích chập 2D bằng NVIDIA Tensor Cores qua FFI. |
| `sqlite` | Cơ sở dữ liệu | Driver SQLite thuần T nối trực tiếp thư viện OS (`winsqlite3.dll`/`libsqlite3.so`), zero dependency. |
| `mmap` | Ánh xạ bộ nhớ | Memory mapping file lớn zero-copy, đọc thẳng tensor float/quant từ ổ đĩa. |
| `gguf`/`safetensors`| Định dạng Model | Parser định dạng mô hình GGUF v2/v3 và HuggingFace Safetensors. |
| `quant` | Lượng tử hóa | Chuyển đổi và giải lượng tử FP16, BF16, FP32, Q8_0, Q4_0. |
| `ffi` | Foreign Function | Gọi trực tiếp hàm thư viện C (`.dll` / `.so`) và thao tác con trỏ bộ nhớ thô. |
| `thread` | Đa luồng & Lock-Free | Worker thread, MPMC channel, Rayon thread pool, `AtomicInt`, `SharedTensor`. |
| `typecheck` | Kiểm tra tĩnh | Type annotations checker, Borrow & Use-After-Close/Move checker. |

---

## 🧰 Bộ công cụ nhà phát triển

Toàn bộ công cụ hỗ trợ phát triển đều được viết bằng T và tích hợp sẵn trong repo:

| Công cụ | Lệnh thực thi | Mô tả |
|---|---|---|
| **REPL** | `trun` | Môi trường tương tác dòng lệnh đa năng |
| **Debugger** | `trun -d app.tb` | Gỡ lỗi từng bước (`s`), đặt breakpoint (`b pc`), xem stack/locals (`p`, `l`) |
| **Type Checker** | `trun tools/tcheck.t app.t` | Kiểm tra tĩnh kiểu, rà soát lỗi sở hữu con trỏ và exhaustive matching |
| **Formatter** | `trun tools/ffmt.t app.t` | Định dạng mã nguồn tự động, chuẩn hóa thụt lề 4 khoảng trắng |
| **LSP Server** | `trun tools/tlsp.t` | Hỗ trợ Language Server Protocol cho VS Code (autocomplete, hover, jump to definition, rename) |
| **Package Manager**| `trun tools/tpkg.t` | Quản lý gói thư viện: `init`, `install`, `pack`, `publish`, `run` |
| **Test Runner** | `trun tools/run_all.t` | Bộ chạy toàn bộ 52 test suite tự động |

### Tiện ích mở rộng VS Code (`editors/vscode/`)
Đã cấu hình sẵn extension cho Visual Studio Code tại thư mục `editors/vscode/`:
- Tô màu cú pháp TextMate Grammar chuẩn (`.tmLanguage.json`).
- Nhận diện file `.t` và định dạng thụt lề thông minh.
- Kết nối trực tiếp với Language Server `tools/tlsp.t`.

---

## 🔒 Bảo mật & An toàn

1. **Bytecode Verification (`vm/tvm.c`)**: Trước khi thực thi bất kỳ tệp `.tb` nào, VM kiểm tra toàn vẹn cấu trúc bytecode: định danh magic number, phạm vi opcode hợp lệ, chỉ số global, bảng dòng `LN01` và bảng ngoại lệ `EX01`.
2. **Chống Command Injection (CWE-78)**: Khởi chạy tiến trình con trực tiếp qua API hệ điều hành (`CreateProcessA` trên Windows, `fork`/`execvp` trên POSIX) với danh sách đối số độc lập, không thông qua shell.
3. **Chống Zip-Slip (CWE-22)**: Trình giải nén gói `.tpkg` kiểm tra chặt chẽ tên tệp, ngăn chặn tuyệt đối việc ghi đè file hệ thống qua đường dẫn tương đối `..` hoặc đường dẫn tuyệt đối.

---

## 📁 Cấu trúc thư mục

```text
├── .github/workflows/     # CI/CD pipeline (kiểm tra an toàn gitleaks, trivy và build test Windows/Linux)
├── bench/                 # Bộ kịch bản benchmark so sánh với Python và Rust
├── editors/vscode/        # Tiện ích mở rộng cho Visual Studio Code (Syntax & LSP)
├── examples/              # Các chương trình mẫu (Deep learning, Raytracer, Data pipeline)
├── self/                  # Trình biên dịch compiler & evaluator tự thân viết thuần bằng T
├── std/                   # Thư viện chuẩn (25+ modules: toán học, mạng, thread, GPU, tensor...)
├── tests/                 # 52 bộ kiểm thử tự động toàn diện cho VM, JIT và stdlib
├── tools/                 # Bộ công cụ nhà phát triển: REPL, LSP, Formatter, Package Manager, AOT
├── vm/                    # Máy ảo C siêu nhẹ, JIT x86-64, JIT ARM64 và kịch bản build
├── tcomp.tb               # Compiler thuần T đã tự bootstrap thành bytecode
├── LICENSE                # Giấy phép nguồn mở MIT
└── README.md              # Tài liệu dự án
```

---

## 📄 Giấy phép (License)

Dự án được phân phối dưới giấy phép mã nguồn mở **[MIT License](LICENSE)**.
