# tests/jit2.t — JIT v2.2: hàm số học thuần (int/float/bool, vòng lặp, đệ quy,
# gọi chéo, builtin int/float/abs/min/max, global chỉ đọc) + deopt an toàn.
# Mọi kết quả phải giống hệt interpreter (TVM_NOJIT=1) và t.py.

fn fib(n):
    if n < 2:
        return n
    return fib(n - 1) + fib(n - 2)

fn sumto(n):
    mut s = 0
    mut i = 0
    while i < n:
        s += i % 7
        i += 1
    return s

fn mandel(w, h, maxit):
    mut count = 0
    for y in range(h):
        for x in range(w):
            cr = 2.0 * x / w - 1.5
            ci = 2.0 * y / h - 1.0
            mut zr = 0.0
            mut zi = 0.0
            mut k = 0
            while k < maxit and zr * zr + zi * zi < 4.0:
                t = zr * zr - zi * zi + cr
                zi = 2.0 * zr * zi + ci
                zr = t
                k += 1
            if k == maxit:
                count += 1
    return count

# gọi chéo + builtin
fn sq(x):
    mut r = x * x
    return r
fn hyp2(a, b):
    mut s = sq(a) + sq(b)
    return s
fn clamp(v, lo, hi):
    mut r = max(lo, min(v, hi))
    return r
fn fl(x):
    mut y = float(x) / 2
    return int(y) + abs(0 - x)
fn isodd(n):
    mut r = n % 2 == 1
    return r

mut acc = 0
for i in range(120):
    acc += fib(12) + hyp2(i, 3) + clamp(i, 10, 50) + fl(i)
print(acc, fib(25), sumto(100000), mandel(60, 40, 50))
mut odds = 0
for i in range(100):
    if isodd(i):
        odds += 1
print(odds)

# float: NaN/inf, so sánh
fn fcmp(a, b):
    mut r = 0
    if a < b:
        r += 1
    if a <= b:
        r += 10
    if a == b:
        r += 100
    if a != b:
        r += 1000
    return r
mut fc = 0
for i in range(80):
    fc = fcmp(1.5, 2.5)
nan = 0.0 * 1e400
print(fc, fcmp(2.5, 2.5), fcmp(nan, 1.0), fcmp(1.0, nan))

# % kiểu floor + float %
fn mods(a, b):
    mut r = a % b
    return r
for i in range(80):
    mods(i, 3)
print(mods(-7, 3), mods(7, -3), mods(-6, 3), mods(0, 5))
fn fmods(a, b):
    mut r = a % b
    return r
for i in range(80):
    fmods(1.5 * i, 2.0)
print(fmods(-7.5, 2.0), fmods(7.5, -2.0))

# deopt: chia 0 -> lỗi y hệt interpreter (try bắt được)
fn safe_div(a, b):
    mut r = a / b
    return r
for i in range(80):
    safe_div(i, 2)
try:
    safe_div(1, 0)
except "division":
    print("bắt:", err)
fn imod0(a, b):
    mut r = a % b
    return r
for i in range(80):
    imod0(i, 3)
try:
    imod0(5, 0)
except "modulo":
    print("bắt:", err)

# deopt: đổi kiểu đối số sau khi JIT (int -> float) vẫn đúng
fn add3(a, b, c):
    mut r = a + b + c
    return r
for i in range(80):
    add3(i, 1, 2)
print(add3(1, 2, 3), add3(1.5, 2, 3), add3(1, 2.5, 0.25))

# deopt: global đọc trong hàm đổi kiểu
mut scale = 3
fn scaled(x):
    mut r = x * scale
    return r
for i in range(80):
    scaled(i)
print(scaled(5))
scale = 0.5
print(scaled(5))

# deopt: biến giữ hàm bị gán lại sau khi JIT
fn base(x):
    mut r = x + 1
    return r
mut hook = base
fn caller(x):
    mut r = hook(x) * 2
    return r
for i in range(80):
    caller(i)
print(caller(10))
fn base2(x):
    mut r = x + 100
    return r
hook = base2
print(caller(10))

# đệ quy sâu hơn giới hạn native -> interpreter lo (vẫn đúng)
fn depth(n):
    if n == 0:
        return 0
    return 1 + depth(n - 1)
for i in range(60):
    depth(10)
print(depth(5000))

# bool trả về + and/or + not
fn between(x, lo, hi):
    return x >= lo and x <= hi and not (x == 13)
mut bc = 0
for i in range(100):
    if between(i, 10, 20):
        bc += 1
print(bc, between(13, 0, 99), between(5, 0, 9))

# OSR: một lần gọi, vòng lặp dài; locals float + int
fn osr_mix(n):
    mut x = 0.0
    mut k = 0
    while k < n:
        x = x + k * 0.5
        k += 1
    return x
print(osr_mix(300000))

# OSR với nhánh trả sớm
fn find_first(n, target):
    mut i = 0
    while i < n:
        if i * i >= target:
            return i
        i += 1
    return 0 - 1
print(find_first(1000000, 99980001), find_first(10, 1000))
print("JIT2 OK")
