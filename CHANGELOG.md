# Changelog

All notable changes to the T programming language project are documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and the project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [7.1.0] - 2026-11-08

### Added
- **Unboxed Numeric Vectors (`std/vec.t`)**: The runtime introduces contiguous 8-byte `VecI64` and `VecF64` buffers that support iterator fusion across chained mapping and filtering operations.
- **LSD Radix Sort (`std/vec.t`)**: The module implements an in-place 8-pass least significant digit radix sort that accelerates 64-bit integer sorting by more than six times compared to standard quicksort.
- **Fibonacci Open-Addressing HashMap (`std/fastmap.t`)**: The module introduces `IntMap`, which applies multiplicative Fibonacci hashing ($2^{64} / \phi$) to reduce hash collisions and optimize CPU cache utilization.
- **Vectorized String Kernels (`std/fastmap.t`)**: The runtime incorporates eight-way unrolled scanning routines (`str_count_byte`, `str_fnv1a_64`, `str_find_fast`) to accelerate string searching, byte counting, and hash calculation.
- **Native Thread Pool (`std/thread.t`)**: The module provides a zero-allocation native worker thread pool that executes parallel map and reduce operations across CPU cores.
- **AOT Pointer and Array Arithmetic (`std/aot.t`)**: The native transpiler now supports raw pointer dereferencing and indexed array access (`i64_ptr`, `f64_ptr`, `arr[i]`).

---

## [7.0.0] - 2026-11-01

### Added
- **Algebraic Data Types and Error Propagation (`std/result.t`)**: The language now supports `Result<T, E>` and `Option<T>` constructs alongside the postfix `?` operator for explicit error propagation.
- **Flat Memory Struct Arrays (`std/cstruct.t`)**: The module allows developers to declare C-compatible `#[repr(C)]` contiguous arrays that leverage L1 cache tiling and SIMD `axpy_f64` instructions.
- **Static Borrow Checker (`std/typecheck.t`)**: The static analysis pass now detects use-after-move errors, resource leaks, and unhandled match patterns prior to runtime execution.
- **Lock-Free Concurrency Primitives (`std/thread.t`)**: The runtime introduces atomic integer counters, mutual exclusion locks, and shared-memory tensor buffers.

---

## [6.0.0] - 2026-10-25

### Added
- **Syntactic Extensions**: The compiler now supports generator functions via `yield`, context managers via `with`, optional static type annotations, and operator overloading protocols (`__add__`, `__sub__`, `__mul__`, `__matmul__`).
- **Asynchronous Networking (`std/net.t`, `std/async.t`)**: The standard library introduces non-blocking TCP and UDP socket primitives managed by a cooperative event loop supporting `async fn` and `await`.
- **Expanded Tensor Datatypes (`vm/tvm.c`, `vm/tensor_impl.h`)**: The virtual machine extends tensor operations to support `f64`, `f32`, `f16`, `bf16`, `i32`, and `i8` numeric layouts.
- **Self-Contained Executable Packaging**: The `--build-exe` CLI flag allows developers to bundle the virtual machine runtime and compiled bytecode into a single standalone binary.

---

## [5.2.0] - 2026-10-20

### Added
- **ARM64 JIT Compilation Engine (`vm/jit_arm64.c`)**: The runtime compiles bytecode directly to AArch64 machine instructions using AAPCS64 calling conventions with on-stack replacement and deoptimization guards.
- **Hardware GPU Acceleration**: The `std/vulkan.t` module loads the Vulkan driver to dispatch SPIR-V 1.0 compute shaders, while `std/cublas.t` and `std/cudnn.t` offload tensor calculations to NVIDIA Tensor Cores.

---

## [5.0.0] - 2026-10-15

### Added
- **Transformer Inference Subsystem (`std/transformer.t`)**: The runtime supports autoregressive Transformer inference featuring rotary positional embeddings, grouped-query attention, key-value caching, and SwiGLU feed-forward layers.
- **Neural Network Model Formats (`std/gguf.t`, `std/safetensors.t`)**: The standard library can read model architectures and weight catalogs from GGUF and Safetensors archive files.
- **Memory Mapping and Quantization (`std/mmap.t`, `std/quant.t`)**: The runtime maps large weight files into process memory with zero copies and quantizes tensor buffers into FP16, BF16, Q8_0, and Q4_0 formats.
- **Foreign Function Interface (`std/ffi.t`)**: The module loads external shared libraries (`.dll` and `.so`) and calls exported functions with arbitrary parameter signatures.

---

## [3.0.0] - 2026-10-13

### Added
- **Complete Self-Hosting Toolchain**: The language bootstraps itself without third-party runtimes; the compiler, REPL, formatter, and package manager are implemented in pure T.
- **Direct Source Execution**: The `trun` binary accepts `.t` source files directly and caches compiled bytecode automatically.
- **Hardened Security Architecture**: The virtual machine verifies bytecode structure prior to execution, spawns processes without shell expansion, and rejects directory traversal attempts during archive extraction.
