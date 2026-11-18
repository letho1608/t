<div align="center">

# ⚡ T Programming Language

**A modern, 100% self-hosted, AI-first programming language with native tensors, dual x86-64/ARM64 JIT, AOT compilation, and Rust-grade performance.**

[![CI](https://github.com/letho1608/t/actions/workflows/ci.yml/badge.svg)](https://github.com/letho1608/t/actions/workflows/ci.yml)
[![Version](https://img.shields.io/badge/version-v7.1-blue.svg)](https://github.com/letho1608/t)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Arch](https://img.shields.io/badge/arch-x86--64%20%7C%20ARM64-brightgreen.svg)](#)
[![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20Linux%20%7C%20macOS-lightgrey.svg)](#)
[![Tests](https://img.shields.io/badge/tests-52%2F52%20ALL--GREEN-success.svg)](#)
[![Self-Hosted](https://img.shields.io/badge/self--hosted-100%25%20pure%20.t-orange.svg)](#)

[Features](#-features) • [Quick Start](#-quick-start) • [Install & Build](#-install--build) • [Syntax](#-language-syntax) • [Standard Library](#-standard-library) • [Tooling](#-developer-tooling) • [Security](#-security)

</div>

---

## Overview

**T** is a general-purpose programming language combining Python/Go-style indented syntax with near-C/Rust performance. It is designed to be AI-first, with multi-dimensional tensors as a first-class primitive type.

- **100% Self-Hosted** — The compiler (`self/tcomp.t`), REPL (`tools/repl.t`), LSP server (`tools/tlsp.t`), formatter (`tools/ffmt.t`), package manager (`tools/tpkg.t`), and test runner (`tools/run_all.t`) are all written in T and bootstrap themselves.
- **AI-First & Native Tensors** — Multi-dimensional, multi-dtype tensors (`f64`, `f32`, `f16`, `bf16`, `i32`, `i8`) are a language primitive with built-in reverse-mode automatic differentiation.
- **Dual JIT Compiler** — Generates native code for x86-64 (Win64 ABI + System V) and ARM64 (AArch64 AAPCS64), with On-Stack Replacement (OSR) and deoptimization.
- **AOT Native Compiler** — Transpiles T source to optimized C99 or Rust, or produces standalone `.dll`/`.so` libraries with AVX2 (`std/aot.t`).
- **GPU Acceleration** — Vulkan Compute Shaders (SPIR-V 1.0) and NVIDIA cuBLAS/cuDNN (Tensor Cores FP16/BF16/FP32) supported natively.
- **Rust-Grade Safety** — `Result<T,E>` / `Option<T>` / `?` operator, static Borrow/Move checker (`std/typecheck.t`), `#[repr(C)]` flat struct arrays (`std/cstruct.t`), and lock-free concurrency primitives.
- **Single-Binary Packaging** — Produces a fully self-contained executable (`.exe` / ELF) embedding the VM and bytecode (`trun --build-exe`).

---

## 🚀 Quick Start

| Action | Command | Description |
|---|---|---|
| Run a source file | `trun app.t` | Transparently compiles via `tcomp.tb` and executes with JIT |
| Evaluate a snippet | `trun -c "print(1 + 2)"` | One-shot expression evaluation |
| Run bytecode | `trun app.tb` | Zero compile-time startup + JIT |
| Interactive REPL | `trun` | Smart interactive shell |
| Compile to bytecode | `trun tcomp.tb app.t app.tb` | Explicit compilation via the self-hosted compiler |
| Package as binary | `trun --build-exe app.t app.exe` | Bundle VM + bytecode into one standalone binary |
| AOT compile | `trun tools/taot.t --emit-c app.t` | Transpile to optimized C99 or Rust |
| Static type check | `trun tools/tcheck.t app.t` | Borrow check, Use-After-Move/Close, exhaustiveness |

---

## 🛠️ Install & Build

### Prerequisites

| Platform | Requirement |
|---|---|
| Windows | MSVC (`cl.exe`) from Visual Studio or Build Tools |
| Linux / macOS | GCC or Clang |

### Build from Source

```bash
# Windows (run inside a Visual Studio Developer Command Prompt)
vm\build.bat

# Linux / macOS
sh vm/build.sh
```

The build script will:
1. Compile the C VM (`vm/tvm.c`, `vm/jit_x64.c`, `vm/jit_arm64.c`) into `trun` / `trun.exe`.
2. Use the resulting `trun` to self-host the compiler (`self/tcomp.t` → `tcomp.tb`) and REPL (`tools/repl.t` → `trepl.tb`).

### Run Tests & Benchmarks

```bash
# Run the full 52-suite test harness
./trun tools/run_all.t        # Linux / macOS
.\trun.exe tools\run_all.t    # Windows

# Benchmark against Python
.\trun.exe bench\bench.t

# Benchmark against Rust / native -O2
.\trun.exe bench\rust_vs_t.t
```

---

## ✨ Features

### 1. Clean Indentation-Based Syntax

```python
# Immutable (let) and mutable (mut) bindings
let pi = 3.14159
mut counter = 0
counter += 1

# Inline conditional expression
label = "even" if counter % 2 == 0 else "odd"

# 64-bit bitwise operators — JIT-compiled directly to x86-64 / ARM64 machine code
mask = (0b1100 & 0b1010) | (~0 & 3) ^ (1 << 4)

# Default parameters and variadic *args
fn log(module, *messages, level = "INFO"):
    print(f"[{level}][{module}]", " ".join([str(m) for m in messages]))

log("NET", "Connected", "port", 8080)
```

### 2. Structs with Methods

```python
struct Vec2:
    x, y

    fn length(self):
        import math
        return math.hypot(float(self.x), float(self.y))

    fn add(self, other):
        return Vec2(self.x + other.x, self.y + other.y)

a = Vec2(3, 4)
b = Vec2(1, 2)
print(a.length())          # 5.0
print(a.add(b).x)          # 4
```

### 3. Native Tensors & Autograd

Tensors are a first-class primitive supporting multiple dtypes (`f64`, `f32`, `f16`, `bf16`, `i32`, `i8`) with vectorised SIMD matrix multiplication (`@`) and multi-threading.

```python
import autograd as ag

# Differentiable parameters
w = ag.param(tensor([[0.5, -0.2], [0.1, 0.8]]))
b = ag.param(tensor([[0.0, 0.0]]))
x = ag.const(tensor([[1.0, 2.0]]))

# Forward pass
z = ag.add(ag.matmul(x, w), b)
out = ag.gelu(z)

# Reverse-mode backward pass
ag.backward(out)
print("dw:", w.grad)
```

### 4. Error Handling — `try/except` and `Result<T, E>`

```python
import result as r

fn divide(a, b):
    if b == 0:
        return r.Err("division by zero")
    return r.Ok(a / b)

fn pipeline():
    val = divide(10, 2)?   # unwrap Ok or propagate Err
    return r.Ok(val * 3)

res = pipeline()
if res.is_ok():
    print("Result:", res.unwrap())   # 15.0
```

### 5. Pattern Matching & Traits

```python
switch shape:
    case Circle(r):
        print("area:", 3.14 * r * r)
    case Rect(w, h):
        print("area:", w * h)
    default:
        print("unknown shape")
```

### 6. Async / Await

```python
import async

async fn fetch(url):
    # non-blocking I/O
    return await http_get(url)

async fn main():
    results = await async.gather([fetch(u) for u in urls])
    print(results)

async.run(main())
```

---

## ⚡ Rust-Grade Performance Optimisations (v7.1)

| Feature | Module | Detail |
|---|---|---|
| **8-Pass LSD Radix Sort** | `std/vec.t` | $O(N)$ integer sort; >6× faster than `qsort -O2` |
| **Unboxed Vectors** | `std/vec.t` | Contiguous `VecI64` / `VecF64` with zero-cost iterator fusion |
| **Fibonacci IntMap** | `std/fastmap.t` | Open-addressing hash table using multiplicative Fibonacci hashing; L1/L2 cache-friendly |
| **SIMD String Kernels** | `std/fastmap.t` | 8-way unrolled zero-copy scan (`str_count_byte`, `str_fnv1a_64`, `str_find_fast`) |
| **Rayon-Style Thread Pool** | `std/thread.t` | Zero-spawn native C thread pool for CPU-heavy tasks (`par_reduce_i64`, `par_map_f64`) |
| **`#[repr(C)]` Flat Struct Arrays** | `std/cstruct.t` | Zero-copy C-layout arrays with L1 cache tiling and SIMD `axpy_f64` |
| **Lock-Free Concurrency** | `std/thread.t` | `AtomicInt`, `Mutex`, `SharedTensor` zero-copy sharing across worker threads |

```python
import vec
import thread

# 1. 8-Pass LSD Radix Sort
mut v = vec.VecI64()
for x in [54, 12, 999, 1, 450, 23]:
    v.push(x)
v.radix_sort()
print(v.to_list())   # [1, 12, 23, 54, 450, 999]

# 2. Rayon-style parallel reduction across 4 threads
total = thread.par_reduce_i64(1, 1_000_000, 4)
print(total)         # 500000500000
```

---

## 📦 Standard Library

| Module | Description |
|---|---|
| `math` | Full trigonometric, exponential, and special functions; `pi`, `tau`, `e`, `clamp`, `gcd`, `lcm` |
| `json` | RFC 8259 parser/serialiser with full Unicode (`\uXXXX`) and nested dict/list/tensor support |
| `os` | File I/O, directories, environment, SHA-256, safe subprocess spawning (CWE-78 safe) |
| `path` | Cross-platform path manipulation (`join`, `basename`, `dirname`, `ext`, `normpath`) |
| `time` | High-resolution clock, epoch time, `sleep`, `measure` |
| `random` | Seeded 64-bit PRNG (`randint`, `uniform`, `shuffle`, `choice`, `gauss`) |
| `re` | Full regex engine (`match`, `search`, `findall`, `sub`, `split`, capture groups, `{m,n}`) |
| `csv` | RFC 4180 CSV parser/serialiser (`loads`, `dumps`, `read_csv`, `write_csv`) |
| `http` | URL encoding/parsing and HTTP client (`get`, `post`, `request`) |
| `net` | Native TCP/UDP sockets (`TcpListener`, `TcpStream`, `UdpSocket`) and embedded `HttpServer` |
| `async` | Cooperative async runtime: `EventLoop`, `Future`, `spawn`, `gather`, `async fn` / `await` |
| `thread` | Worker threads, MPMC channels, Rayon thread pool, `AtomicInt`, `SharedTensor` |
| `autograd` | Reverse-mode AD with fused C kernels for DL ops (`relu`, `gelu`, `softmax`, `conv2d`, …) |
| `ai` | ML utilities: `linear`, `conv2d`, `relu`, `gelu`, `softmax`, `mse`, `grad` |
| `transformer` | CPU-side LLM inference: RoPE, KV-Cache, GQA, SwiGLU, BPE tokeniser (LLaMA/Mistral/Qwen) |
| `vulkan` | Vulkan Loader init, device enumeration, and direct SPIR-V 1.0 Compute Shader dispatch |
| `cublas` / `cudnn` | NVIDIA Tensor Core GEMM and cuDNN Conv2D via FFI; silently falls back to CPU |
| `sqlite` | Native SQLite client via OS library (`winsqlite3.dll` / `libsqlite3.so`); zero install |
| `mmap` | Zero-copy memory-mapped file I/O for large tensors |
| `gguf` / `safetensors` | GGUF v2/v3 and HuggingFace Safetensors model format parsers |
| `quant` | FP16, BF16, Q8_0, Q4_0 quantisation and dequantisation |
| `ffi` | Load `.dll` / `.so` and call C functions with raw pointer primitives (**unsafe**) |
| `typecheck` | Static type checker with Borrow/Move/Use-After-Close analysis |
| `result` | `Result<T,E>`, `Option<T>`, `move(x)`, zero-cost `?` operator |
| `cstruct` | `#[repr(C)]` flat struct arrays: `define`, `alloc_array`, `axpy_f64` |
| `aot` | Instant JIT (`aot.jit`) and AOT transpiler to C99 / Rust with pointer/array support |
| `vec` | Unboxed `VecI64` / `VecF64`, zero-cost iterator fusion, 8-pass LSD Radix Sort |
| `fastmap` | Fibonacci `IntMap`, 8-way SIMD string kernels |
| `base64` | RFC 4648 Base64 encode/decode |

---

## 🧰 Developer Tooling

All tools are written in T and ship with the repository:

| Tool | Command | Description |
|---|---|---|
| **REPL** | `trun` | Interactive shell with history and multi-line input |
| **Debugger** | `trun -d app.tb` | Step (`s`), breakpoints (`b <pc>`), inspect stack/locals (`p`, `l`), disassemble (`d`) |
| **Type Checker** | `trun tools/tcheck.t app.t` | Static types, borrow analysis, exhaustiveness checking |
| **Formatter** | `trun tools/ffmt.t [--check] app.t` | Canonical 4-space formatting; idempotent |
| **LSP Server** | `trun tools/tlsp.t` | Full LSP: hover, go-to-definition, references, rename, completion, diagnostics |
| **Package Manager** | `trun tools/tpkg.t <cmd>` | `init`, `install`, `run`, `test`, `build [--exe]`, `pack`, `publish`, `search`, `release` |
| **Test Runner** | `trun tools/run_all.t` | Runs all 52 test suites and reports a single pass/fail summary |
| **Benchmark** | `trun bench/bench.t` | Head-to-head benchmark against Python and Rust |

### VS Code Extension (`editors/vscode/`)

- Syntax highlighting via TextMate Grammar (`.tmLanguage.json`).
- Smart indentation and bracket matching.
- Integrated Language Server (`tools/tlsp.t`).

---

## 🔒 Security

1. **Bytecode Verification** — Before executing any `.tb` file, the VM validates the magic number, opcode ranges, operand bounds, global index table, and line/exception tables (`LN01` / `EX01`).
2. **No Shell Injection (CWE-78)** — Child processes are spawned directly via OS APIs (`CreateProcessA` on Windows, `fork`/`execvp` on POSIX) with an explicit argument array — never through a shell string.
3. **Zip-Slip Prevention (CWE-22)** — The `.tpkg` extractor rejects any archive entry whose resolved path would escape the target directory (absolute paths, `..` components, Windows drive prefixes).

---

## 📁 Repository Layout

```text
.github/workflows/   CI pipeline — Gitleaks, Trivy, build & test on Linux + Windows
bench/               Benchmark scripts vs Python and Rust
editors/vscode/      VS Code extension (syntax + LSP integration)
examples/            Sample programs: neural net XOR, ASCII raytracer, data pipeline
self/                Self-hosted compiler (tcomp.t) and evaluator (eval.t)
std/                 Standard library (25+ modules)
tests/               52 automated test suites covering VM, JIT, stdlib, and tooling
tools/               Developer tools: REPL, LSP, formatter, package manager, AOT
vm/                  C VM, x86-64 JIT, ARM64 JIT, build scripts
tcomp.tb             Pre-bootstrapped compiler bytecode (no Python required)
LICENSE              MIT License
README.md            This file
```

---

## 📄 License

Distributed under the [MIT License](LICENSE).
