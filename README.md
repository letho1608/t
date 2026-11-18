<div align="center">

# The T Programming Language

**A self-hosted, AI-first programming language with native multi-dtype tensors, dual x86-64 and ARM64 JIT compilation, native AOT transpilation, and Rust-grade safety guarantees.**

[![CI](https://github.com/letho1608/t/actions/workflows/ci.yml/badge.svg)](https://github.com/letho1608/t/actions/workflows/ci.yml)
[![Version](https://img.shields.io/badge/version-v7.1-blue.svg)](https://github.com/letho1608/t)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Arch](https://img.shields.io/badge/arch-x86--64%20%7C%20ARM64-brightgreen.svg)](#)
[![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20Linux%20%7C%20macOS-lightgrey.svg)](#)
[![Tests](https://img.shields.io/badge/tests-52%2F52%20ALL--GREEN-success.svg)](#)
[![Self-Hosted](https://img.shields.io/badge/self--hosted-100%25%20pure%20.t-orange.svg)](#)

[Quick Start](#quick-start) • [Installation and Building](#installation-and-building) • [Showcase](#showcase) • [Architecture](#architecture) • [Examples](#examples) • [Contributing](CONTRIBUTING.md) • [License](#license)

</div>

---

## Overview

The T programming language combines indentation-based syntax with execution speeds approaching C and Rust. Designed with an AI-first philosophy, the runtime treats multi-dimensional tensors as first-class language primitives and embeds reverse-mode automatic differentiation directly into the standard library.

The entire ecosystem is self-hosted: the reference compiler, interactive REPL, language server, code formatter, package manager, and test harness are written entirely in T and bootstrap without external language dependencies.

---

## Showcase

The snippet below demonstrates first-class tensors, multi-threaded matrix multiplication (`@`), and automatic differentiation in T:

```python
import autograd as ag

# Initialize model parameters with gradient tracking enabled
let weights = ag.param(tensor([[0.5, -0.2], [0.1, 0.8]]))
let bias    = ag.param(tensor([[0.0, 0.0]]))
let inputs  = ag.const(tensor([[1.0, 2.0]]))

# Compute the forward pass through projection and GELU activation
let linear      = ag.add(ag.matmul(inputs, weights), bias)
let activations = ag.gelu(linear)

# Execute the backward pass to calculate gradients across the graph
ag.backward(activations)

print("Weight gradients:", weights.grad)
```

For more language samples—including structs, async networking, Rayon-style parallel reductions, and LLM Transformer inference—explore the [`examples/`](examples/) directory.

---

## Quick Start

The table below outlines common command-line operations:

| Task | Command | Description |
|---|---|---|
| Run a source file | `trun app.t` | The runtime compiles the script via `tcomp.tb` and executes it using the JIT compiler. |
| Evaluate a one-liner | `trun -c "print(1 + 2)"` | The runtime evaluates the expression in a temporary execution context. |
| Execute bytecode | `trun app.tb` | The virtual machine executes compiled bytecode immediately with zero compilation startup cost. |
| Start the REPL | `trun` | The runtime starts an interactive read-eval-print loop with multi-line editing support. |
| Compile to bytecode | `trun tcomp.tb app.t app.tb` | The self-hosted compiler transforms source code into an optimized bytecode binary. |
| Build standalone executable | `trun --build-exe app.t app.exe` | The tool bundles the virtual machine and bytecode into a single standalone binary. |
| Transpile to C99 | `trun tools/taot.t --emit-c app.t` | The AOT compiler converts T source code into optimized standard C99 source files. |
| Static analysis | `trun tools/tcheck.t app.t` | The type checker verifies type annotations and checks memory ownership invariants. |

---

## Installation and Building

### System Requirements

The project requires a standard C compiler and Git:
- **Windows**: The Microsoft Visual C++ compiler (`cl.exe`), available via Visual Studio 2019+ or the Build Tools for Visual Studio.
- **Linux and macOS**: GCC (version 11+) or Clang (version 13+).

### Building from Source

To clone the repository and compile the runtime from source:

```bash
# Clone the repository
git clone https://github.com/letho1608/t.git
cd t

# On Windows (inside a Visual Studio Developer Command Prompt)
vm\build.bat

# On Linux or macOS
sh vm/build.sh
```

The build script compiles the C virtual machine (`trun` / `trun.exe`), uses it to bootstrap the compiler (`self/tcomp.t` → `tcomp.tb`), and builds the REPL tool (`tools/repl.t` → `trepl.tb`).

### Verification

To run the complete test suite:

```bash
# Windows
.\trun.exe tools\run_all.t

# Linux / macOS
./trun tools/run_all.t
```

All 52 subsystem test suites must report `[PASS]` and conclude with an `ALL-GREEN` status.

---

## Architecture

The T runtime architecture consists of several modular subsystems:

- **100% Self-Hosted Compiler**: The compiler frontend and bytecode generator (`self/tcomp.t`) are implemented in pure T and produce deterministic bytecode output across architectures.
- **Dual-Tier JIT Engine**: The virtual machine incorporates Method JIT and On-Stack Replacement (OSR) compilers for x86-64 (Win64 and System V ABIs) and ARM64 (AArch64 AAPCS64). Execution paths deoptimize smoothly to the interpreter upon guard failure.
- **Hardware Acceleration**: The runtime dispatches SPIR-V 1.0 compute shaders through the Vulkan Loader and drives NVIDIA Tensor Core GEMM operations via cuBLAS and cuDNN foreign function bindings.
- **Rust-Grade Safety & Concurrency**: The language provides algebraic data types (`Result<T, E>` and `Option<T>`) with postfix error propagation (`?`), a static borrow and move checker (`std/typecheck.t`), contiguous C-layout structs (`#[repr(C)]`), and lock-free thread coordination primitives.
- **High-Performance Primitives**: The standard library includes 8-pass $O(N)$ LSD Radix Sort, unboxed contiguous vectors (`VecI64`, `VecF64`), multiplicative Fibonacci hash maps (`IntMap`), and a zero-allocation native thread pool (`std/thread.t`).

---

## Examples

The [`examples/`](examples/) directory contains complete, runnable programs demonstrating language capabilities:

- [`hello_tour.t`](examples/hello_tour.t): A tour of language syntax, covering variables, structs, bitwise operations, default arguments, and exception handling.
- [`neural_xor.t`](examples/neural_xor.t): Training a two-layer neural network on the nonlinear XOR problem using autograd and SGD.
- [`data_pipeline.t`](examples/data_pipeline.t): A data processing pipeline performing CSV parsing, regex filtering, statistical aggregation, and JSON export.
- [`raytracer_ascii.t`](examples/raytracer_ascii.t): A 3D raytracer rendering a shaded sphere in ASCII art to the terminal.
- [`llm_inference.t`](examples/llm_inference.t): Autoregressive Transformer inference featuring RMSNorm, RoPE, KV-Cache, GQA, and SwiGLU.

To run any example:

```bash
trun examples/hello_tour.t
trun examples/neural_xor.t
trun examples/llm_inference.t
```

---

## Developer Tooling

The repository ships with an integrated development toolchain written in T:

- **Interactive REPL** (`trun`): A command-line shell supporting expression evaluation and multi-line definitions.
- **Interactive Debugger** (`trun -d app.tb`): A step-level debugger supporting breakpoints, stack inspection, and disassembly display.
- **Static Type & Borrow Checker** (`tools/tcheck.t`): An analyzer checking type annotations and enforcing lifetime invariants.
- **Code Formatter** (`tools/ffmt.t`): An idempotent source formatter enforcing canonical 4-space indentation.
- **Language Server** (`tools/tlsp.t`): An LSP implementation providing diagnostics, autocomplete, definition lookup, and symbol renaming.
- **Package Manager** (`tools/tpkg.t`): A dependency manager supporting project initialization, packaging, and testing.
- **VS Code Extension** (`editors/vscode/`): Syntax highlighting and LSP integration for Visual Studio Code.

---

## Repository Layout

```text
.github/workflows/   Continuous integration workflows (Linux and Windows)
bench/               Performance benchmark scripts versus Python and Rust
editors/vscode/      Visual Studio Code extension and TextMate grammar
examples/            Runnable example programs (neural nets, raytracing, pipelines)
self/                Self-hosted compiler and tree-walking evaluator
std/                 Standard library modules (math, os, net, async, tensors, GPU)
tests/               Automated test suites covering runtime, JIT, and stdlib
tools/               Developer tools (REPL, LSP server, formatter, package manager)
vm/                  Virtual machine source in C, JIT compilers, and build scripts
tcomp.tb             Pre-bootstrapped compiler bytecode for initial self-hosting
LICENSE              MIT license declaration
README.md            Project overview and documentation
```

---

## Community and Contributing

We welcome contributions from developers worldwide. Please consult the following documents before contributing:

- [Contributing Guide](CONTRIBUTING.md): Build procedures, coding conventions, and pull request guidelines.
- [Security Policy](SECURITY.md): Instructions for reporting security vulnerabilities.
- [Changelog](CHANGELOG.md): Historical record of changes across versions.

---

## License

The T programming language project is distributed under the terms of the [MIT License](LICENSE).
