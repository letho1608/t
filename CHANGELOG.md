# Changelog

All notable changes to the **T Programming Language** are documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [7.1.0] - 2026-11-08

### Added
- **Unboxed Homogeneous Vectors (`std/vec.t`)**: `VecI64` and `VecF64` unboxed 8-byte contiguous buffers with Zero-Cost Iterator Fusion (`fused_map_filter_sum`, `dot`).
- **8-Pass $O(N)$ LSD Radix Sort (`std/vec.t`)**: In-place radix sort for `VecI64`, outperforming standard `qsort -O2` by >6×.
- **Fibonacci Open-Addressing HashMap (`std/fastmap.t`)**: High-performance `IntMap` using multiplicative Fibonacci hashing ($2^{64} / \phi$) optimized for L1/L2 cache locality.
- **Zero-Copy SIMD String Kernels (`std/fastmap.t`)**: 8-way unrolled scanning routines (`str_count_byte`, `str_fnv1a_64`, `str_find_fast`).
- **Rayon-Style Native Thread Pool (`std/thread.t`)**: Zero-spawn C thread pool for parallel reductions and maps (`par_collatz`, `par_reduce_i64`, `par_map_f64`).
- **AOT Pointer & Array Support (`std/aot.t`)**: Native transpilation support for pointer arithmetic and array access (`i64_ptr`, `f64_ptr`, `arr[i]`).

---

## [7.0.0] - 2026-11-01

### Added
- **Algebraic Data Types & Error Propagation (`std/result.t`)**: Full support for `Result<T, E>` (`Ok`, `Err`), `Option<T>` (`Some`, `None`), and postfix `?` operator.
- **Flat Struct Arrays (`std/cstruct.t`)**: `#[repr(C)]` contiguous layout with L1 cache tiling and SIMD `axpy_f64`.
- **Static Borrow & Move Checker (`std/typecheck.t`)**: Detection of use-after-move, use-after-close, and non-exhaustive pattern matching at compile time.
- **Lock-Free Concurrency Primitives (`std/thread.t`)**: `AtomicInt`, `Mutex`, and zero-copy `SharedTensor`.

---

## [6.0.0] - 2026-10-25

### Added
- **Language Expressiveness Extensions**:
  - Python-style Generators (`yield`) and Context Managers (`with`).
  - Operator overloading protocol (`__add__`, `__sub__`, `__mul__`, `__matmul__`).
  - Optional static type annotations (`x: int = 5`).
  - Structural Pattern Matching and Traits.
- **Networking & Async I/O**:
  - Native TCP/UDP sockets (`std/net.t`).
  - Event loop and cooperative async/await runtime (`std/async.t`).
- **Multi-Dtype Tensor Expansion (`vm/tvm.c`, `vm/tensor_impl.h`)**:
  - Support for `f64`, `f32`, `f16`, `bf16`, `i32`, and `i8` tensors.
- **Single-Binary Packaging**:
  - `--build-exe` option bundling C VM runtime and bytecode into a standalone binary.

---

## [5.2.0] - 2026-10-20

### Added
- **ARM64 JIT Backend (`vm/jit_arm64.c`)**:
  - Method JIT, On-Stack Replacement (OSR), and deoptimization for AArch64 (AAPCS64 ABI).
  - Cross-architecture machine code simulator (`a64_sim_exec`) for testing ARM64 on x86-64 hosts.
- **Hardware GPU Acceleration**:
  - Vulkan Loader initialization and direct SPIR-V 1.0 Compute Shader execution (`std/vulkan.t`).
  - NVIDIA cuBLAS GEMM (Tensor Cores FP16/BF16/FP32) and cuDNN Conv2D integration (`std/cublas.t`, `std/cudnn.t`).

---

## [5.0.0] - 2026-10-15

### Added
- **LLM Foundation Model Engine**:
  - CPU-side Transformer inference engine (`std/transformer.t`): RoPE, KV-Cache, GQA, SwiGLU, BPE tokenizer.
  - Model format loaders: GGUF v2/v3 (`std/gguf.t`) and HuggingFace Safetensors (`std/safetensors.t`).
  - Quantization primitives (`std/quant.t`): FP16, BF16, Q8_0, Q4_0.
  - Zero-copy memory mapping (`std/mmap.t`).
- **System FFI & Memory Primitives (`std/ffi.t`)**:
  - Dynamic library loading (`.dll` / `.so`) and C calling convention support.
  - Direct pointer access (`mem_alloc`, `mem_read_ptr`, `Buffer`).

---

## [3.0.0] - 2026-10-13

### Added
- **100% Self-Hosting**:
  - Complete elimination of external Python dependencies; compiler (`self/tcomp.t`), REPL, LSP, formatter, and package manager rewritten entirely in pure T.
- **Direct Source Execution**:
  - Running `.t` files directly (`trun app.t`) with automatic bytecode caching and JIT compilation.
- **Hardened VM Security**:
  - Bytecode integrity verifier (`verify_bytecode`).
  - Safe subprocess spawning bypassing shell interpreters (CWE-78 mitigation).
  - Strict path normalization preventing zip-slip vulnerabilities (CWE-22 mitigation).
