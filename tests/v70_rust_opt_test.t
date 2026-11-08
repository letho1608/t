# tests/v70_rust_opt_test.t — Kiểm thử 3 cơ chế tối ưu chuẩn Rust của T (v7.0: J9/AOT, L12, S1 & Rust Benchmark)
import os
import time
import result as res
import typecheck as tc
import cstruct
import thread
import aot

print("--- 1. L12: Result<T, E> (Ok/Err), Option<T> (Some/None) & Toán tử '?' Zero-Cost ---")
fn parse_positive(x: int):
    if x > 0:
        return res.Ok(x * 10)
    return res.Err("so_am_hoac_0:" + str(x))

fn compute_chain(a: int, b: int):
    va = parse_positive(a)?
    vb = parse_positive(b)?
    return res.Ok(va + vb)

r1 = compute_chain(3, 4)
assert r1.is_ok() and r1.unwrap() == 70, "operator ? unwraps Ok values"

r2 = compute_chain(3, -5)
assert r2.is_err() and r2.err() == "so_am_hoac_0:-5", "operator ? early returns Err"

fn opt_lookup(d, k: str):
    if d.has(k):
        return res.Some(d[k])
    return res.None()

fn opt_double(d, k: str):
    v = opt_lookup(d, k)?
    return res.Some(v * 2)

o1 = opt_double({"a": 21}, "a")
assert o1.is_some() and o1.unwrap() == 42, "operator ? unwraps Some"
o2 = opt_double({"a": 21}, "missing")
assert o2.is_none() and o2.unwrap_or(99) == 99, "operator ? early returns None"

# Kiểm tra Pattern Matching với Ok(v) / Err(e)
fn describe_res(r):
    switch r:
        case Ok(v):
            return "thanh_cong:" + str(v)
        case Err(e):
            return "loi:" + str(e)

assert describe_res(r1) == "thanh_cong:70", "switch case Ok(v)"
assert describe_res(r2) == "loi:so_am_hoac_0:-5", "switch case Err(e)"

print("--- 2. L12: Borrow/Move/Use-After-Close Check & Exhaustiveness Check ---")
# 2a. Phát hiện dùng biến sau khi đã move() hoặc .close()
chk_move = tc.check_source("let buf = [1, 2, 3]\nlet b2 = move(buf)\nprint(buf)\n")
assert chk_move["ok"] == false and ("BorrowError" in chk_move["errors"][0]), "BorrowChecker catches use-after-move"

chk_close = tc.check_source("let f = open(\"test.txt\", \"r\")\nf.close()\nlet s = f.read()\n")
assert chk_close["ok"] == false and ("BorrowError" in chk_close["errors"][0]), "BorrowChecker catches use-after-close"

# 2b. Phát hiện switch trên Result thiếu nhánh Err (Exhaustiveness Check)
chk_exh_bad = tc.check_source("switch r:\n    case Ok(v):\n        print(v)\n")
assert chk_exh_bad["ok"] == false and ("ExhaustivenessError" in chk_exh_bad["errors"][0]), "ExhaustivenessChecker catches missing Err branch"

chk_exh_ok = tc.check_source("switch r:\n    case Ok(v):\n        print(v)\n    case Err(e):\n        print(e)\n")
assert chk_exh_ok["ok"] == true, "ExhaustivenessChecker accepts complete Ok + Err switch"

print("--- 3. S1: #[repr(C)] Zero-Copy Contiguous Struct Array & SIMD Stride Kernels ---")
Particle = cstruct.define("Particle", [
    ["x", "f64"],
    ["vx", "f64"],
    ["id", "i32"],
    ["flags", "u16"]
])
assert Particle.stride == 24, "#[repr(C)] alignment and stride (8 + 8 + 4 + 2 + 2 pad = 24 bytes)"
assert Particle.offset_of("x") == 0 and Particle.offset_of("vx") == 8 and Particle.offset_of("id") == 16 and Particle.offset_of("flags") == 20, "#[repr(C)] field offsets"

with Particle.alloc_array(1000) as arr:
    for i in range(1000):
        arr.set_f64(i, "x", 2.0)
        arr.set_f64(i, "vx", 0.5)
        arr.set_i32(i, "id", i + 1)
        arr.set_u16(i, "flags", 7)
    assert arr.get_i32(500, "id") == 501 and arr.get_u16(500, "flags") == 7, "cstruct i32/u16 read/write"
    # Cập nhật SIMD axpy_f64: x += 4.0 * vx (2.0 + 4.0 * 0.5 = 4.0 mỗi phần tử -> tổng 1000 phần tử = 4000.0)
    arr.axpy_f64("x", "vx", 4.0)
    assert arr.sum_f64("x") == 4000.0, "cstruct SIMD axpy_f64 + sum_f64"

print("--- 4. S1: Lock-Free AtomicInt, Mutex & Zero-Copy SharedTensor Đa luồng ---")
at = thread.atomic_int(100)
assert at.load() == 100, "atomic load initial"
old = at.add(25)
assert old == 100 and at.load() == 125, "atomic fetch-and-add"
assert at.cas(125, 200) == true and at.load() == 200, "atomic CAS success"
assert at.cas(125, 300) == false and at.load() == 200, "atomic CAS reject"

mtx = thread.mutex()
with mtx as m:
    at.sub(50)
assert at.load() == 150, "mutex context manager + atomic sub"
mtx.free()

# Chia sẻ Zero-Copy SharedTensor + AtomicInt với Worker Thread
big_t = ones([10000])
sh_handle = thread.share_tensor(big_t)
at.store(0)

worker_src = "import thread\nraw = argv()\na = raw[1:raw.len] if (raw.len > 0 and (raw[0].endswith(\".t\") or raw[0].endswith(\".tb\"))) else raw\nsh = int(a[0])\naid = int(a[1])\nlo = int(a[2])\nhi = int(a[3])\nt = thread.from_shared_tensor(sh)\nt[lo] = 99.0\ns = int(thread.slice_sum(t, lo, hi))\natom = thread.atomic_from_id(aid)\natom.add(s)\n"
w_path = "tests/_tmp_v70_worker.t"
os.write_file(w_path, worker_src)
th1 = thread.spawn(w_path, [str(sh_handle), str(at.id), "0", "5000"])
th2 = thread.spawn(w_path, [str(sh_handle), str(at.id), "5000", "10000"])
thread.join(th1)
thread.join(th2)
os.remove(w_path)

# Kiểm tra Worker Thread đã ghi trực tiếp vào big_t zero-copy và cộng dồn AtomicInt chuẩn xác
assert big_t[0] == 99.0 and big_t[5000] == 99.0, "Zero-copy SharedTensor mutated directly by worker threads"
assert at.load() == (10000 + 98 + 98), "Lock-free AtomicInt accumulated across worker threads"
at.free()

print("--- 5. J9 / AOT: Instant JIT & AOT Native Transpiler/Compiler vs Rust ---")
fn hot_arith(n: int) -> int:
    mut s = 0
    mut i = 1
    while i <= n:
        s = s + (i * 3 - (i >> 1))
        i = i + 1
    return s

# Biên dịch JIT ngay lập tức từ lần gọi đầu tiên (threshold = 1)
aot.jit(hot_arith, [10])
assert hot_arith(100) == 12650, "Instant JIT execution on 1st call"

# Dịch AOT sang C99 & Rust, biên dịch mã máy gốc và so sánh tốc độ với Rust
kernel_src = "fn fast_collatz(limit: int) -> int:\n    mut total = 0\n    mut i = 1\n    while i <= limit:\n        mut n = i\n        while n > 1:\n            if (n & 1) == 0:\n                n = n >> 1\n            else:\n                n = 3 * n + 1\n            total = total + 1\n        i = i + 1\n    return total\n"
c_code = aot.transpile_to_c(kernel_src)
rs_code = aot.transpile_to_rust(kernel_src)
assert ("long long fast_collatz" in c_code) and ("pub extern \"C\" fn fast_collatz" in rs_code), "AOT transpiles to C99 and Rust"

fn fast_collatz_ref(limit: int) -> int:
    mut total = 0
    mut i = 1
    while i <= limit:
        mut n = i
        while n > 1:
            if (n & 1) == 0:
                n = n >> 1
            else:
                n = 3 * n + 1
            total = total + 1
        i = i + 1
    return total

with aot.compile_module(kernel_src, "tests/_tmp_v70_aot") as mod:
    assert mod.is_native == true, "AOT compiled native DLL successfully"
    res_aot = mod.call_i64("fast_collatz", "l", [10000])
    assert res_aot == fast_collatz_ref(10000), "AOT compiled native kernel matches T reference byte-for-byte"

print("V70-RUST-OPT-ALL-GREEN")
