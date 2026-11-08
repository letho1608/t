# tests/jitarm64.t — J7 Hoàn chỉnh: Trình biên dịch Bytecode -> AArch64 (ARM64) + OSR + Deopt
# Kiểm chứng:
#   1. Oracle byte (t_arm64_selftest) & dump nhị phân 52 byte (t_arm64_dump).
#   2. Biên dịch hàm T thật sang mã máy AArch64 (arm64_compile) & chạy trên máy ảo AArch64
#      (arm64_run / arm64_osr_run): đệ quy fib, vòng lặp số học, thực float64, bitwise 64-bit,
#      builtin int/float/abs/min/max/len, đọc/ghi list (OP_IDX/OP_SETIDX/OP_LEN) và Deopt an toàn.

let sys = load("__sys")
import os
import path

fn fib_a64(n):
    if n < 2:
        return n
    return fib_a64(n - 1) + fib_a64(n - 2)

fn sum_loop(n):
    mut s = 0
    mut i = 1
    while i <= n:
        s = s + (i * 3 - (i % 5))
        i = i + 1
    return s

fn float_kernel(n, x):
    mut acc = 0.0
    mut i = 1
    while i <= n:
        acc = acc + (float(i) * x) / (float(i) + 1.5)
        i = i + 1
    return acc

fn bitwise_kernel(a, b):
    let x = ((a & b) | (a ^ b)) + (~a & 255)
    let y = (x << 3) >> 1
    return min(abs(y), max(a, b))

fn list_sieve_sum(xs):
    mut i = 0
    let n = len(xs)
    mut s = 0
    while i < n:
        xs[i] = xs[i] * 2 + 1
        s = s + xs[i]
        i = i + 1
    return s

fn div_maybe_deopt(a, b):
    return a % b

struct Vec2:
    x
    y

fn vec_step(p, n):
    mut i = 0
    while i < n:
        p.x = p.x + p.y + i
        p.y = p.y + 1
        i = i + 1
    return Vec2(p.x, p.y)

fn dict_counter(d, n):
    mut i = 0
    while i < n:
        d["cnt"] = d.get("cnt", 0) + i
        d[i] = i * 2
        i = i + 1
    if d.has(2):
        d.remove(2)
    return d["cnt"] + len(d)

fn str_builder(prefix, n):
    mut s = prefix
    mut i = 1
    while i <= n:
        s = s + "-" + str(i)
        i = i + 1
    if s == "k-1-2-3" and s[0] == "k":
        return len(s)
    return 0

fn main():
    assert sys.arm64_selftest() == true, "arm64 emitter lech oracle byte"
    print("arm64 selftest OK")

    tmp = os.mktemp("jarm64")
    out = path.join(tmp, "arm64.bin")
    assert sys.arm64_dump(out) == true
    assert os.exists(out)
    data = os.read_file(out)
    assert data.len == 52, "dump phai 52 byte (16+16+20), got " + str(data.len)
    os.remove(out)
    os.rmtree(tmp)

    assert sys.arm64_dump(path.join(tmp, "nope", "x.bin")) == false
    print("arm64 dump OK")

    # 1. Kiểm tra biên dịch Bytecode -> AArch64 (mỗi lệnh 4 byte, kết thúc bằng ret 0xD65F03C0)
    let code_fib = sys.arm64_compile(fib_a64, 10)
    assert code_fib.len > 32 and (code_fib.len % 4) == 0
    assert ord(code_fib[code_fib.len - 4]) == 0xC0
    assert ord(code_fib[code_fib.len - 3]) == 0x03
    assert ord(code_fib[code_fib.len - 2]) == 0x5F
    assert ord(code_fib[code_fib.len - 1]) == 0xD6
    print("arm64_compile bytecode -> AArch64 (" + str(code_fib.len) + " bytes) OK")

    # 2. Chạy đệ quy fib_a64 trên mã máy AArch64
    let f12 = sys.arm64_run(fib_a64, 12)
    assert f12 == fib_a64(12) and f12 == 144
    print("arm64_run fib(12) =", f12, "OK")

    # 3. Chạy vòng lặp số học & OSR
    let sl = sys.arm64_run(sum_loop, 500)
    assert sl == sum_loop(500)
    let sl_osr = sys.arm64_osr_run(sum_loop, 500)
    assert sl_osr == sl
    print("arm64_run & OSR sum_loop(500) =", sl, "OK")

    # 4. Chạy kernel số thực IEEE-754 float64
    let fk_got = sys.arm64_run(float_kernel, 200, 0.75)
    let fk_ref = float_kernel(200, 0.75)
    assert abs(fk_got - fk_ref) < 0.000000000001
    print("arm64_run float64 kernel OK")

    # 5. Chạy kernel bitwise 64-bit & builtin abs/min/max
    let bw_got = sys.arm64_run(bitwise_kernel, 12345, 6789)
    let bw_ref = bitwise_kernel(12345, 6789)
    assert bw_got == bw_ref
    print("arm64_run bitwise64 + builtins =", bw_got, "OK")

    # 6. Đọc/ghi phần tử list (OP_IDX, OP_SETIDX, OP_LEN)
    let a1 = [1, 2, 3, 4, 5, 6, 7, 8]
    let a2 = [1, 2, 3, 4, 5, 6, 7, 8]
    let ls_got = sys.arm64_run(list_sieve_sum, a1)
    let ls_ref = list_sieve_sum(a2)
    assert ls_got == ls_ref and a1 == a2
    print("arm64_run list read/write =", ls_got, "OK")

    # 7. Kiểm tra Deopt an toàn khi chia dư và tràn
    assert sys.arm64_run(div_maybe_deopt, -17, 5) == (-17 % 5)
    print("arm64_run floor mod & deopt guard OK")

    # 8. J8: Struct (OP_ATTR, OP_SETATTR, NAT___NEW) trên AArch64
    let p_in = Vec2(10, 3)
    assert sys.arm64_compile(vec_step, p_in, 5).len > 0
    let p_out = sys.arm64_run(vec_step, Vec2(10, 3), 20)
    let p_ref = vec_step(Vec2(10, 3), 20)
    assert p_out.x == p_ref.x and p_out.y == p_ref.y
    print("arm64_run J8 struct (ATTR/SETATTR/__new) =", p_out.x, p_out.y, "OK")

    # 9. J8: Dict (OP_IDX, OP_SETIDX, get, has, remove, len) trên AArch64
    let d_in = {"cnt": 5}
    assert sys.arm64_compile(dict_counter, d_in, 6).len > 0
    let d1 = {"cnt": 5}
    let d2 = {"cnt": 5}
    let dc_got = sys.arm64_run(dict_counter, d1, 10)
    let dc_ref = dict_counter(d2, 10)
    assert dc_got == dc_ref and d1["cnt"] == d2["cnt"] and not d1.has(2)
    print("arm64_run J8 dict (get/set/has/remove/len) =", dc_got, "OK")

    # 10. J8: Str (OP_PUSH_STR, NAT_STR, OP_ADD concat, OP_EQ, OP_IDX, OP_LEN) trên AArch64
    assert sys.arm64_compile(str_builder, "k", 3).len > 0
    let sb_got = sys.arm64_run(str_builder, "k", 3)
    let sb_ref = str_builder("k", 3)
    assert sb_got == sb_ref and sb_got == 7
    print("arm64_run J8 str (concat/str/eq/idx/len) =", sb_got, "OK")

main()
print("All jit-arm64 tests passed successfully!")
