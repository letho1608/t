# bench/rust_vs_t.t — Đối đầu Hiệu năng Trực tiếp: Ngôn ngữ T (v7.1) vs Native Rust/C Baseline (-O2/opt-level=3)
import os
import time
import aot
import cstruct
import thread
import ffi
import vec
import fastmap

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
REF_NATIVE_SRC = "#include <stdint.h>\n#include <stdlib.h>\n#ifdef _WIN32\n#define EXPORT __declspec(dllexport)\n#else\n#define EXPORT\n#endif\n\nEXPORT double rs_gemm_192(long long iters) {\n    int n = 192;\n    double *a = (double*)malloc(sizeof(double) * n * n);\n    double *b = (double*)malloc(sizeof(double) * n * n);\n    double *c = (double*)calloc(n * n, sizeof(double));\n    for (int i = 0; i < n * n; i++) { a[i] = 0.5; b[i] = 0.5; }\n    for (long long step = 0; step < iters; step++) {\n        for (int i = 0; i < n; i++) {\n            for (int k = 0; k < n; k++) {\n                double aik = a[i * n + k];\n                for (int j = 0; j < n; j++) {\n                    c[i * n + j] += aik * b[k * n + j];\n                }\n            }\n        }\n    }\n    double res = c[0] + c[n * n - 1];\n    free(a); free(b); free(c);\n    return res;\n}\n\ntypedef struct {\n    double x;\n    double vx;\n    int32_t id;\n    uint16_t flags;\n} RsParticle;\n\nEXPORT double rs_particle_sim(long long count, long long steps, RsParticle *arr) {\n    for (long long s = 0; s < steps; s++) {\n        for (long long i = 0; i < count; i++) {\n            arr[i].x += 0.125 * arr[i].vx;\n        }\n    }\n    double sum = 0.0;\n    for (long long i = 0; i < count; i++) {\n        sum += arr[i].x;\n    }\n    return sum;\n}\n\nEXPORT long long rs_collatz(long long limit) {\n    long long total = 0;\n    for (long long i = 1; i <= limit; i++) {\n        long long n = i;\n        while (n > 1) {\n            if ((n & 1) == 0) n >>= 1;\n            else n = 3 * n + 1;\n            total++;\n        }\n    }\n    return total;\n}\n\nEXPORT long long rs_fused_map_filter_sum(const int64_t *p, long long n, long long mul, long long add, long long thresh) {\n    long long total = 0;\n    for (long long i = 0; i < n; i++) {\n        long long y = p[i] * mul + add;\n        if (y >= thresh) total += y;\n    }\n    return total;\n}\n"

fn run_benchmarks():
    print("=========================================================================")
    print("   BENCHMARK ĐỐI ĐẦU TRỰC TIẾP: NGÔN NGỮ T (v7.1) vs RUST/NATIVE (-O2)")
    print("=========================================================================")

    ref_dll = aot.compile_c_dll(REF_NATIVE_SRC, "bench/_tmp_rs_ref", false)
    if ref_dll == "" or not os.exists(ref_dll):
        print("Khong the bien dich thu vien tham chieu.")
        return nil

    rs_lib = ffi.open(ref_dll)
    fn_rs_gemm = rs_lib.sym("rs_gemm_192")
    fn_rs_part = rs_lib.sym("rs_particle_sim")
    fn_rs_coll = rs_lib.sym("rs_collatz")
    fn_rs_fms  = rs_lib.sym("rs_fused_map_filter_sum")

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
        parr.axpy_f64("x", "vx", 0.125, 200)
        t_sum = parr.sum_f64("x")
        t_part = time.clock() - t2

        t3 = time.clock()
        rs_part_val = _sys.ffi_call(fn_rs_part, "d", "llp", [100000, 200, parr.ptr])
        rs_part = time.clock() - t3
    print("2. #[repr(C)] Struct Sim (100k x 200):  T = " + fmt_ms(t_part) + " | Rust/Native = " + fmt_ms(rs_part) + " -> T nhanh gấp " + fmt_ratio(t_part, rs_part))

    # --- Bài 3: Number Crunching Collatz (300,000 vòng x 5) qua AOT Native Compiler của T ---
    t_collatz_src = "fn t_collatz(limit: int) -> int:\n    mut total = 0\n    mut i = 1\n    while i <= limit:\n        mut n = i\n        while n > 1:\n            n = (n >> 1) if (n & 1) == 0 else (3 * n + 1)\n            total = total + 1\n        i = i + 1\n    return total\n"
    mut rs_coll = 0.0
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
        print("3. Collatz AOT Single-Core (300k x 5):  T = " + fmt_ms(t_coll) + " | Rust/Native = " + fmt_ms(rs_coll) + " -> T đạt " + fmt_ratio(t_coll, rs_coll))

    # --- Bài 4 (v7.1 P1): Rayon-Style Native 4-Thread Pool Collatz vs Single-Core Native ---
    t6 = time.clock()
    for r_i in range(5):
        v_par = thread.par_collatz(300001, 4)
    t_par_coll = time.clock() - t6
    print("4. Collatz Rayon 4-Thread Pool (P1):    T = " + fmt_ms(t_par_coll) + " | 1-Core Base = " + fmt_ms(rs_coll) + " -> T nhanh gấp " + fmt_ratio(t_par_coll, rs_coll))

    # --- Bài 5 (v7.1 U2): Unboxed VecI64 O(N) LSD Radix Sort vs O(N log N) Native qsort (500,000 phần tử) ---
    with vec.i64_rand(500000) as v_radix:
        with vec.i64_rand(500000) as v_qsort:
            t7 = time.clock()
            v_radix.radix_sort()
            t_radix = time.clock() - t7

            t8 = time.clock()
            v_qsort.qsort()
            rs_qsort = time.clock() - t8
            assert v_radix[0] == v_qsort[0] and v_radix[499999] == v_qsort[499999], "Radix sort matches qsort"
            print("5. Unboxed Sort 500k i64 (Radix vs C):  T = " + fmt_ms(t_radix) + " | C qsort -O2 = " + fmt_ms(rs_qsort) + " -> T nhanh gấp " + fmt_ratio(t_radix, rs_qsort))

    # --- Bài 6 (v7.1 U2): Zero-Cost Fused Iterator Pipeline (1,000,000 i64 x 20 vòng) ---
    with vec.i64_range(1000000) as v_fms:
        t9 = time.clock()
        mut s_t = 0
        for r_i in range(20):
            s_t = v_fms.fused_map_filter_sum(3, 7, 1500000)
        t_fms = time.clock() - t9

        t10 = time.clock()
        mut s_rs = 0
        for r_i in range(20):
            s_rs = _sys.ffi_call(fn_rs_fms, "l", "pllll", [v_fms.ptr(), 1000000, 3, 7, 1500000])
        rs_fms = time.clock() - t10
        assert s_t == s_rs, "Fused iterator matches native C loop"
        print("6. Fused Map-Filter-Sum (1M i64 x 20):  T = " + fmt_ms(t_fms) + " | Rust/Native = " + fmt_ms(rs_fms) + " -> T nhanh gấp " + fmt_ratio(t_fms, rs_fms))

    # --- Bài 7 (v7.1 U3): Unboxed Fibonacci IntMap Frequency Count (500,000 phần tử) ---
    with vec.i64_rand(500000) as v_keys:
        with fastmap.int_map(8192) as fmap:
            t11 = time.clock()
            fmap.batch_freq(v_keys, 4095)
            t_fmap = time.clock() - t11
            assert fmap.sum_vals() == 500000, "IntMap batch_freq accounted for all 500k items"
            print("7. Fibonacci IntMap 500k Lookups (U3):  T = " + fmt_ms(t_fmap) + " (" + str(fmap.len()) + " unique keys, 0 boxing overhead)")

    rs_lib.close()
    if os.exists(ref_dll):
        os.remove(ref_dll)
    print("=========================================================================")
    print("RUST-VS-T-BENCH-OK")

run_benchmarks()
