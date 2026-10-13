# tests/jitmain.t — JIT vòng lặp top-level (global ghi thẳng bộ nhớ, deopt/lối ra
# về interpreter). Phải giống TVM_NOJIT=1 và t.py.

# tích luỹ + break + continue
mut s = 0
for i in range(30000):
    if i % 5 == 0:
        continue
    if i > 25000:
        break
    s += i % 7
print(s)

# lồng nhau + while
mut tot = 0
mut a = 0
while a < 60:
    for b in range(500):
        tot += a * b % 11
    a += 1
print(tot, a)

# global ghi trong vòng lặp, hàm được gọi đọc lại (write-through)
mut counter = 0
fn peek():
    mut r = counter * 2
    return r
mut seen = 0
for i in range(5000):
    counter += 1
    seen += peek()
print(counter, seen)

# đổi kiểu giữa chừng (int -> float): guard deopt, kết quả vẫn đúng
mut x = 0
for i in range(6000):
    if i == 4000:
        x = 0.5
    x += 1
print(x)

# lỗi giữa vòng lặp, try ở ngoài bắt
mut k = 0
try:
    for i in range(10000):
        k += 100 / (5000 - i)
except "division":
    print("bắt:", err)
print(k > 0)

# ghi list ở top-level
mut flags = [0] * 4000
for i in range(4000):
    flags[i] = i * i % 13
mut fs = 0
for v in flags:
    fs += v
print(fs, flags[3999])

# float + bool global
mut acc = 0.0
mut on = false
for i in range(8000):
    on = i % 3 == 0
    if on:
        acc += 0.25
print(acc, on)

# global bị closure bắt (cell) trong vòng lặp: không JIT, vẫn đúng
mut fns = []
for i in range(3):
    fn g():
        return i * 10
    fns.append(g)
mut gs = 0
for f in fns:
    gs += f()
print(gs)
print("JITMAIN OK")
