<div align="center">

# T Programming Language

**A self-hosted, AI-first programming language with native multi-dtype tensors, dual x86-64 and ARM64 JIT compilers, AOT transpilation, and Rust-grade safety guarantees.**

[![CI](https://github.com/letho1608/t/actions/workflows/ci.yml/badge.svg)](https://github.com/letho1608/t/actions/workflows/ci.yml)
[![Version](https://img.shields.io/badge/version-v7.1-blue.svg)](https://github.com/letho1608/t)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Arch](https://img.shields.io/badge/arch-x86--64%20%7C%20ARM64-brightgreen.svg)](#)
[![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20Linux%20%7C%20macOS-lightgrey.svg)](#)
[![Tests](https://img.shields.io/badge/tests-52%2F52%20ALL--GREEN-success.svg)](#)
[![Self-Hosted](https://img.shields.io/badge/self--hosted-100%25%20pure%20.t-orange.svg)](#)

[Overview](#overview) • [Quick Start](#quick-start) • [Installation and Building](#installation-and-building) • [Language Syntax and Features](#language-syntax-and-features) • [Performance Optimizations](#performance-optimizations) • [Standard Library](#standard-library) • [Developer Tooling](#developer-tooling) • [Security Architecture](#security-architecture) • [Community and Contributing](#community-and-contributing) • [License](#license)

</div>

---

## Overview

The T programming language combines indentation-based syntax with execution performance approaching C and Rust. The language treats multi-dimensional tensors as first-class primitives and integrates reverse-mode automatic differentiation directly into the standard runtime.

- **Self-Hosting Toolchain**: The entire development environment is implemented in T itself. The reference compiler (`self/tcomp.t`), interactive REPL (`tools/repl.t`), language server (`tools/tlsp.t`), code formatter (`tools/ffmt.t`), package manager (`tools/tpkg.t`), and test harness (`tools/run_all.t`) are all written in pure T and bootstrap without external language runtimes.
- **Native Tensor Primitives**: The runtime supports contiguous multi-dimensional tensors across diverse numeric datatypes (`f64`, `f32`, `f16`, `bf16`, `i32`, `i8`). Tensors support broadcast arithmetic and multi-threaded SIMD matrix multiplication.
- **Dual-Tier JIT Compilation**: The virtual machine includes Method JIT and On-Stack Replacement (OSR) engines targeting x86-64 (Windows Win64 and System V ABIs) and ARM64 (AArch64 AAPCS64). Code paths that fail type guards automatically deoptimize back to the bytecode interpreter.
- **Ahead-of-Time (AOT) Native Compilation**: The `std/aot.t` subsystem translates T source code into standalone C99 or Rust source files, and can compile native dynamic libraries (`.dll` or `.so`) with AVX2 instruction support.
- **Hardware GPU Acceleration**: The runtime dispatches Vulkan Compute Shaders (SPIR-V 1.0) directly through the Vulkan Loader and drives NVIDIA Tensor Core operations via cuBLAS and cuDNN bindings.
- **Static Safety and Concurrency**: The language provides algebraic data types (`Result<T, E>` and `Option<T>`) paired with an error-propagation operator (`?`). A static borrow checker verifies variable lifetimes and prevents use-after-move errors. Concurrency primitives include lock-free atomics and flat contiguous struct arrays (`#[repr(C)]`).
- **Single-Binary Distribution**: Developers can bundle the C virtual machine runtime and compiled bytecode into a single standalone executable file (`trun --build-exe`).

---

## Quick Start

The table below summarizes common command-line invocations for running, compiling, and analyzing T programs:

| Action | Command | Description |
|---|---|---|
| Run a source file | `trun app.t` | The runtime compiles the source file through `tcomp.tb` and executes the resulting bytecode with the JIT compiler. |
| Evaluate an expression | `trun -c "print(1 + 2)"` | The runtime parses and evaluates the given expression string immediately. |
| Execute bytecode | `trun app.tb` | The virtual machine executes precompiled bytecode directly with zero compilation overhead. |
| Start the REPL | `trun` | The runtime launches an interactive read-eval-print loop with multi-line input support. |
| Compile source to bytecode | `trun tcomp.tb app.t app.tb` | The self-hosted compiler compiles a source script into an optimized bytecode binary. |
| Bundle a standalone binary | `trun --build-exe app.t app.exe` | The tool concatenates the VM runtime and bytecode into a self-contained executable file. |
| Generate native C source | `trun tools/taot.t --emit-c app.t` | The AOT compiler transpiles the T source code into standard C99 source code. |
| Run static analysis | `trun tools/tcheck.t app.t` | The static checker validates type annotations and verifies memory ownership rules. |

---

## Installation and Building

### System Requirements

The project requires a standard C compiler and the Git version control system:
- On Windows, the build scripts require the Microsoft Visual C++ compiler (`cl.exe`), which is included with Visual Studio 2019 or later and the standalone Build Tools for Visual Studio.
- On Linux and macOS, the build scripts require either GCC (version 11 or later) or Clang (version 13 or later).

### Building from Source

To clone the repository and build the runtime binary from source, run the following commands:

```bash
# Clone the repository
git clone https://github.com/letho1608/t.git
cd t

# On Windows (execute within a Visual Studio Developer Command Prompt)
vm\build.bat

# On Linux or macOS
sh vm/build.sh
```

The build scripts perform the following steps:
1. The host C compiler compiles the virtual machine files (`vm/tvm.c`, `vm/jit_x64.c`, `vm/jit_arm64.c`, `vm/tdbg.c`, `vm/trun.c`) into the `trun` executable binary.
2. The newly produced `trun` binary bootstraps the self-hosted compiler from `self/tcli.t` into `tcomp.tb`.
3. The bootstrapped compiler compiles `tools/repl.t` into `trepl.tb`.

### Verification and Benchmarks

To verify that the build succeeds and all subsystem tests pass, run the test runner:

```bash
# Execute the comprehensive 52-suite test harness
.\trun.exe tools\run_all.t      # On Windows
./trun tools/run_all.t          # On Linux or macOS

# Run performance benchmarks against Python
.\trun.exe bench\bench.t

# Run head-to-head performance benchmarks against native Rust compiled with -O2
.\trun.exe bench\rust_vs_t.t
```

---

## Language Syntax and Features

### Variable Declarations and Control Flow

T enforces block structure through 4-space indentation. Variables are immutable by default and must be explicitly marked mutable:

```python
# Variables declared with 'let' cannot be reassigned
let pi = 3.14159

# Variables declared with 'mut' can be modified
mut counter = 0
counter += 1

# Conditional expressions can be written inline
let status = "even" if counter % 2 == 0 else "odd"

# The runtime compiles 64-bit bitwise expressions directly into native instructions
let mask = (0b1100 & 0b1010) | (~0 & 3) ^ (1 << 4)

# Functions support default arguments and variable-length argument lists
fn log_message(subsystem, *items, level = "INFO"):
    print(f"[{level}][{subsystem}]", " ".join([str(item) for item in items]))

log_message("NETWORK", "Connection established on port", 8080)
```

### Structures and Methods

Structures group fields together and support method declarations:

```python
struct Vector2:
    x, y

    fn magnitude(self):
        import math
        return math.hypot(float(self.x), float(self.y))

    fn add(self, other):
        return Vector2(self.x + other.x, self.y + other.y)

let first = Vector2(3, 4)
let second = Vector2(1, 2)
let combined = first.add(second)

print(first.magnitude())     # Output: 5.0
print(combined.x)            # Output: 4
```

### First-Class Tensors and Reverse-Mode Automatic Differentiation

The runtime provides multi-dimensional tensors with automatic differentiation for neural network operations:

```python
import autograd as ag

# Initialize parameters with gradient tracking enabled
let weights = ag.param(tensor([[0.5, -0.2], [0.1, 0.8]]))
let bias = ag.param(tensor([[0.0, 0.0]]))
let inputs = ag.const(tensor([[1.0, 2.0]]))

# Compute the forward pass through linear projection and GELU activation
let linear = ag.add(ag.matmul(inputs, weights), bias)
let activations = ag.gelu(linear)

# Execute the backward pass to compute gradients across the computation graph
ag.backward(activations)

print("Weight gradients:", weights.grad)
```

### Algebraic Data Types and Error Handling

T supports traditional exception handling through `try/except/finally` as well as zero-cost `Result` types:

```python
import result as res

fn divide(numerator, denominator):
    if denominator == 0:
        return res.Err("division by zero")
    return res.Ok(numerator / denominator)

fn calculate():
    # The question mark operator extracts an Ok value or returns early upon an Err
    let value = divide(10, 2)?
    return res.Ok(value * 3)

let outcome = calculate()
if outcome.is_ok():
    print("Calculated result:", outcome.unwrap())
```

---

## Performance Optimizations

The T runtime includes specialized data structures and algorithms designed to achieve memory density and execution speeds comparable to compiled systems languages:

| Optimization Mechanism | Module | Technical Implementation |
|---|---|---|
| **8-Pass LSD Radix Sort** | `std/vec.t` | The module implements an $O(N)$ Least Significant Digit radix sort for 64-bit integer vectors, sorting data faster than standard C library quicksort routines. |
| **Unboxed Numeric Vectors** | `std/vec.t` | The `VecI64` and `VecF64` structures store unboxed numeric values in contiguous memory buffers and perform iterator fusion on chained map and filter operations. |
| **Multiplicative Fibonacci IntMap** | `std/fastmap.t` | The hash table maps integer keys to values using multiplicative Fibonacci hashing, minimizing cache misses during probe sequences. |
| **Vectorized SIMD String Operations** | `std/fastmap.t` | The runtime executes byte-scanning routines with eight-way unrolled loops, accelerating string searching and hash calculations. |
| **Native Worker Thread Pool** | `std/thread.t` | The thread pool schedules parallel work items across native CPU threads without creating new operating system threads on each invocation. |
| **Flat Memory Layouts** | `std/cstruct.t` | Structures marked with C-compatible memory layouts store records contiguously in memory, enabling cache-tiled arithmetic operations. |
| **Lock-Free Concurrency Primitives** | `std/thread.t` | The runtime provides atomic integer counters, mutual exclusion locks, and shared memory tensor buffers for thread-safe cross-worker coordination. |

---

## Standard Library

The standard library ships with the core runtime and provides modular functionality without external dependencies:

| Module | Description |
|---|---|
| `math` | The module implements elementary arithmetic, trigonometric, hyperbolic, and rounding functions alongside mathematical constants. |
| `json` | The module implements an RFC 8259-compliant JSON parser and serializer with Unicode escape decoding. |
| `os` | The module exposes file system management functions, environment variable access, and process execution primitives without invoking an intermediate shell. |
| `path` | The module normalizes and manipulates file system path components across Windows, Linux, and macOS platforms. |
| `time` | The module provides monotonic time measurement, calendar timestamps, and thread sleep utilities. |
| `random` | The module provides a seedable 64-bit pseudo-random number generator supporting uniform, integer, and Gaussian distributions. |
| `re` | The module implements a regular expression engine supporting character classes, quantifier bounds, and captured subexpressions. |
| `csv` | The module parses and formats comma-separated value tables in accordance with RFC 4180. |
| `http` | The module handles URL encoding and provides client methods for dispatching HTTP requests. |
| `net` | The module exposes BSD-style TCP and UDP socket primitives and includes a lightweight embedded HTTP server. |
| `async` | The module provides a cooperative asynchronous event loop managing futures and coroutines defined with `async fn` and `await`. |
| `thread` | The module manages worker threads, message-passing channels, atomic counters, and parallel reductions. |
| `autograd` | The module implements reverse-mode automatic differentiation with fused C kernels for standard neural network layers. |
| `ai` | The module provides machine learning helpers including loss computations, initialization routines, and layer evaluations. |
| `transformer` | The module implements Transformer model inference supporting rotary embeddings, grouped-query attention, and SwiGLU feed-forward networks. |
| `vulkan` | The module initializes the Vulkan API loader and executes SPIR-V 1.0 compute shaders directly on the GPU. |
| `cublas` / `cudnn` | The modules dispatch dense matrix multiplications and two-dimensional convolutions to NVIDIA Tensor Cores via foreign function interfaces. |
| `sqlite` | The module establishes database connections directly to native operating system SQLite shared libraries without third-party drivers. |
| `mmap` | The module maps large files into process virtual memory, facilitating zero-copy tensor loading from disk. |
| `gguf` / `safetensors` | The modules parse metadata and tensor catalogs from GGUF and Hugging Face Safetensors model archives. |
| `quant` | The module converts floating-point tensor buffers into quantized representations including FP16, BF16, Q8_0, and Q4_0. |
| `ffi` | The module loads external dynamic libraries and calls exported C functions using explicit foreign function calling conventions. |
| `typecheck` | The module performs static semantic checks to identify type mismatches, variable moves, and unhandled branches. |
| `result` | The module defines `Result` and `Option` container types to facilitate explicit error propagation without exceptions. |
| `cstruct` | The module defines flat struct schemas and allocates contiguous memory arrays for C-interoperable data processing. |
| `aot` | The module compiles T abstract syntax trees into C99 source files, Rust modules, or native shared libraries. |
| `vec` | The module provides unboxed 64-bit integer and floating-point vectors alongside an $O(N)$ LSD radix sort implementation. |
| `fastmap` | The module provides open-addressing integer hash tables and vectorized byte-scanning routines. |
| `base64` | The module encodes and decodes binary data using RFC 4648 Base64 character representations. |

---

## Developer Tooling

The repository includes a complete developer toolchain written entirely in T:

| Tool | Invocation | Description |
|---|---|---|
| **REPL** | `trun` | The tool launches an interactive shell supporting command history, multi-line expressions, and live evaluation. |
| **Debugger** | `trun -d app.tb` | The interactive debugger supports step execution, breakpoint placement, stack inspection, and disassembly display. |
| **Type Checker** | `trun tools/tcheck.t app.t` | The static analyzer checks type annotations and verifies ownership invariants across variable lifetimes. |
| **Code Formatter** | `trun tools/ffmt.t app.t` | The formatter applies standard 4-space indentation and canonical whitespace formatting idempotently. |
| **Language Server** | `trun tools/tlsp.t` | The server implements the Language Server Protocol to provide hover documentation, definition navigation, and diagnostics to editor clients. |
| **Package Manager** | `trun tools/tpkg.t <command>` | The package tool initializes project manifests, resolves package dependencies, and packages modules for release. |
| **Test Runner** | `trun tools/run_all.t` | The test driver runs all 52 subsystem test suites and outputs a consolidated status summary. |
| **Benchmark Suite** | `trun bench/bench.t` | The benchmark script runs comparative workload evaluations against Python and compiled Rust binaries. |

### Visual Studio Code Extension

The `editors/vscode/` directory contains an extension package for Visual Studio Code:
- The extension applies TextMate grammar definitions to highlight keywords, numeric literals, and string templates.
- The configuration specifies automatic bracket closure and indentation rules for colon-terminated blocks.
- The client integrates with the language server (`tools/tlsp.t`) to provide real-time editor feedback.

---

## Security Architecture

The runtime incorporates multiple defensive safeguards to protect host environments:

1. **Bytecode Structure Verification**:
   Before executing any bytecode file (`.tb`), the virtual machine examines header signatures, confirms that opcode values fall within defined ranges, and ensures that jump offsets and table references do not exceed allocated boundaries.

2. **Mitigation of Command Injection (CWE-78)**:
   The `os.run` function creates child processes by calling underlying operating system APIs (`CreateProcessA` on Windows and `fork`/`execvp` on POSIX systems) with discrete parameter arrays. The runtime does not invoke an intermediate command shell.

3. **Mitigation of Path Traversal (CWE-22)**:
   The package extraction logic inspects archive file paths and rejects entries that contain parent directory references (`..`), absolute root markers, or drive letter prefixes.

4. **Automated Security Scanning**:
   Continuous integration pipelines run automated secret scans with Gitleaks and filesystem vulnerability inspections with Trivy on every commit pushed to the repository.

---

## Repository Layout

```text
.github/workflows/   Continuous integration configuration files for Linux and Windows builds
bench/               Benchmark test scripts evaluating performance against Python and Rust
editors/vscode/      Visual Studio Code editor extension files and TextMate grammar rules
examples/            Demonstration programs including neural network training and data pipelines
self/                Self-hosted compiler implementation and reference tree-walking evaluator
std/                 Standard library modules providing core operating system and numeric services
tests/               Automated test suites covering virtual machine features, compiler passes, and libraries
tools/               Development utilities including the REPL, debugger, formatter, and package manager
vm/                  Virtual machine implementation in C, x86-64 JIT, ARM64 JIT, and build scripts
tcomp.tb             Pre-bootstrapped compiler bytecode required for self-hosting initial builds
LICENSE              MIT license declaration
README.md            Project overview and documentation
```

---

## Community and Contributing

We welcome contributions from the community. Please review the following documents before submitting contributions:
- The [Contributing Guide](CONTRIBUTING.md) describes environment setup, coding conventions, testing requirements, and pull request procedures.
- The [Security Policy](SECURITY.md) outlines instructions for reporting potential security vulnerabilities responsibly.
- The [Changelog](CHANGELOG.md) tracks feature additions, architectural updates, and bug fixes across all released versions.

---

## License

The T programming language project is distributed under the terms of the [MIT License](LICENSE).
