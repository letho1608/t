# tests/jit.t — hot-path JIT native (chạy: python t.py tests/jit.t | trun jit.tb)
# Hàm số học 2-arg gọi >50 lần -> trun biên dịch x86-64, vẫn đúng + đúng kiểu.
fn add(a, b):
    return a + b

fn mul(a, b):
    return a * b

fn sub(a, b):
    return a - b

fn divi(a, b):
    return a / b

# int path: 200 lần, giữ int
mut s = 0
for i in range(200):
    s = add(s, i)
print(s)
assert s == 19900
assert type(add(1, 2)) == "int"

# float path
mut f = 0.5
for i in range(200):
    f = mul(f, 1.01)
print(f > 3.6)
assert type(mul(1.5, 2)) == "float"

# deopt: đổi kiểu sau khi compile (int->float vẫn đúng)
for i in range(60):
    s = add(s, 0)
assert add(1.5, 2) == 3.5
assert type(add(1, 2)) == "int"

# div luôn float
assert divi(7, 2) == 3.5
assert type(divi(7, 2)) == "float"

# sub
assert sub(10, 4) == 6
mut d = 1000
for i in range(100):
    d = sub(d, 1)
assert d == 900

# hàm không-shape (nhánh) không JIT nhưng vẫn đúng
fn fib(n):
    if n < 2:
        return n
    else:
        return fib(n - 1) + fib(n - 2)
assert fib(15) == 610

# cmp i64 hot (>50 gọi) -> native
fn lt(a, b):
    return a < b

mut c = 0
for i in range(200):
    if lt(i, 100):
        c = c + 1
assert c == 100
assert type(lt(1, 2)) == "bool"

fn eq(a, b):
    return a == b

assert eq(3, 3) == true
for i in range(60):
    eq(i, i)
assert eq(3, 3) == true
assert eq(3, 4) == false

# mod i64 hot
fn md(a, b):
    return a % b

assert md(7, 3) == 1
mut m = 0
for i in range(200):
    m = m + md(i, 7)
assert m == 594

# inline: hàm single-return được chèn thẳng (đúng + nhanh)
fn sq(x):
    return x * x

assert sq(9) == 81
fn sumsq(a, b):
    return sq(a) + sq(b)
assert sumsq(3, 4) == 25
mut sh = 100
fn getsh():
    return sh
assert getsh() == 100
print("JIT OK")
