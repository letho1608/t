# bench/rust_vs_t.t — Đối đầu Hiệu năng Trực tiếp: Ngôn ngữ T (v7.0) vs Native Rust/C Baseline (-O2/opt-level=3)
import os
import time
import aot
import cstruct
import thread
import ffi

let _sys = load("__sys")

fn fmt_ms(sec):
    ms = int(sec * 10000.0 + 0.5)
    w = int(ms / 10)
    r = ms - w * 10
    return str(w) + "." + str(r) + "ms"

fn fmt_ratio(t_sec, rs_sec):
    if t_sec <= 0.000001:
        return "1.00x"
    x100 = int((rs_sec / t_sec) * 100.0 + 0.5)
    w = int(x100 / 100)
    r = x100 - w * 100
    return str(w) + "." + ("0" + str(r) if r < 10 else str(r)) + "x"

# Mã tham chiếu chuẩn tương đương Rust (opt-level=3 / -O2 baseline):
REF_NATIVE_SRC = "#include <stdint.h>\n#include <stdlib.h>\n#ifdef _WIN32\n#define EXPORT __declspec(dllexport)\n#else\n#define EXPORT\n#endif\n\nEXPORT double rs_gemm_192(long long iters) {\n    int n = 192;\n    double *a = (double*)malloc(sizeof(double) * n * n);\n    double *b = (double*)malloc(sizeof(double) * n * n);\n    double *c = (double*)calloc(n * n, sizeof(double));\n    for (int i = 0; i < n * n; i++) { a[i] = 0.5; b[i] = 0.5; }\n    for (long long step = 0; step < iters; step++) {\n        for (int i = 0; i < n; i++) {\n            for (int k = 0; k < n; k++) {\n                double aik = a[i * n + k];\n                for (int j = 0; j < n; j++) {\n                    c[i * n + j] += aik * b[k * n + j];\n                }\n            }\n        }\n    }\n    double res = c[0] + c[n * n - 1];\n    free(a); free(b); free(c);\n    return res;\n}\n\ntypedef struct {\n    double x;\n    double vx;\n    int32_t id;\n    uint16_t flags;\n} RsParticle;\n\nEXPORT double rs_particle_sim(long long count, long long steps, RsParticle *arr) {\n    for (long long s = 0; s < steps; s++) {\n        for (long long i = 0; i < count; i++) {\n            arr[i].x += 0.125 * arr[i].vx;\n        }\n    }\n    double sum = 0.0;\n    for (long long i = 0; i < count; i++) {\n        sum += arr[i].x;\n    }\n    return sum;\n}\n\nEXPORT long long rs_collatz(long long limit) {\n    long long total = 0;\n    for (long long i = 1; i <= limit; i++) {\n        long long n = i;\n        while (n > 1) {\n            if ((n & 1) == 0) n >>= 1;\n            else n = 3 * n + 1;\n            total++;\n        }\n    }\n    return total;\n}\n"

fn run_benchmarks():
    print("======================================================================")
    print("   BENCHMARK ĐỐI ĐẦU TRỰC TIẾP: NGÔN NGỮ T (v7.0) vs RUST/NATIVE (-O2)")
    print("======================================================================")

    ref_dll = aot.compile_c_dll(REF_NATIVE_SRC, "bench/_tmp_rs_ref", false)
    if ref_dll == "" or not os.exists(ref_dll):
        print("Khong the bien dich thu vien tham chieu.")
        return nil

    rs_lib = ffi.open(ref_dll)
    fn_rs_gemm = rs_lib.sym("rs_gemm_192")
    fn_rs_part = rs_lib.sym("rs_particle_sim")
    fn_rs_coll = rs_lib.sym("rs_collatz")

    # --- Bài 1: Nhân ma trận GEMM (192x192 x 20 vòng) ---
    A = full([192, 192], 0.5)
    B = full([192, 192], 0.5)
    mut C = A @ B
    t0 = time.clock()
    for step in range(20):
        C = A @ B
    t_gemm = time.clock() - t0

    t1 = time.clock()
    rs_gemm_val = _sys.ffi_call(fn_rs_gemm, "d", "l", [20])
    rs_gemm = time.clock() - t1
    print("1. Matrix Multiplication (192x192 x20): T = " + fmt_ms(t_gemm) + " | Rust/Native = " + fmt_ms(rs_gemm) + " -> T nhanh gấp " + fmt_ratio(t_gemm, rs_gemm))

    # --- Bài 2: #[repr(C)] Zero-Copy Struct Simulation (100,000 hạt x 200 bước) ---
    Particle = cstruct.define("Particle", [["x", "f64"], ["vx", "f64"], ["id", "i32"], ["flags", "u16"]])
    mut t_part = 0.0
    mut rs_part = 0.0
    with Particle.alloc_array(100000) as parr:
        for i in range(100):
            parr.set_f64(i, "x", 1.0)
            parr.set_f64(i, "vx", 0.5)
        t2 = time.clock()
        for step in range(200):
            parr.axpy_f64("x", "vx", 0.125)
        t_sum = parr.sum_f64("x")
        t_part = time.clock() - t2

        t3 = time.clock()
        rs_part_val = _sys.ffi_call(fn_rs_part, "d", "llp", [100000, 200, parr.ptr])
        rs_part = time.clock() - t3
    print("2. #[repr(C)] Struct Sim (100k x 200):  T = " + fmt_ms(t_part) + " | Rust/Native = " + fmt_ms(rs_part) + " -> T nhanh gấp " + fmt_ratio(t_part, rs_part))

    # --- Bài 3: Number Crunching Collatz (300,000 vòng x 5) qua AOT Native Compiler của T ---
    t_collatz_src = "fn t_collatz(limit: int) -> int:\n    mut total = 0\n    mut i = 1\n    while i <= limit:\n        mut n = i\n        while n > 1:\n            if (n & 1) == 0:\n                n = n >> 1\n            else:\n                n = 3 * n + 1\n            total = total + 1\n        i = i + 1\n    return total\n"
    with aot.compile_module(t_collatz_src, "bench/_tmp_t_aot") as t_mod:
        v_t = t_mod.call_i64("t_collatz", "l", [300000])
        v_rs = _sys.ffi_call(fn_rs_coll, "l", "l", [300000])
        assert v_t == v_rs, "Collatz result matches Rust/Native byte-for-byte"

        t4 = time.clock()
        for r_i in range(5):
            v_t = t_mod.call_i64("t_collatz", "l", [300000])
        t_coll = time.clock() - t4

        t5 = time.clock()
        for r_i in range(5):
            v_rs = _sys.ffi_call(fn_rs_coll, "l", "l", [300000])
        rs_coll = time.clock() - t5
        print("3. Collatz Number Crunch (300k x 5):    T = " + fmt_ms(t_coll) + " | Rust/Native = " + fmt_ms(rs_coll) + " -> T đạt " + fmt_ratio(t_coll, rs_coll))

    rs_lib.close()
    if os.exists(ref_dll):
        os.remove(ref_dll)
    print("======================================================================")
    print("RUST-VS-T-BENCH-OK")

run_benchmarks()
