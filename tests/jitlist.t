# tests/jitlist.t — JIT vòng lặp đọc/ghi list (OSR + deopt có chuyển trạng thái)
# Kết quả phải giống interpreter (TVM_NOJIT=1) và t.py.
fn sieve(n):
    mut flags = [true] * (n + 1)
    mut count = 0
    mut i = 2
    while i <= n:
        if flags[i]:
            count += 1
            mut j = i * i
            while j <= n:
                flags[j] = false
                j += i
        i += 1
    return count
print(sieve(100000))

# đọc list trong hàm thuần (entry JIT)
fn total(xs):
    mut s = 0
    mut i = 0
    while i < len(xs):
        s += xs[i]
        i += 1
    return s
mut data = []
for k in range(1000):
    data.append(k % 13)
mut acc = 0
for r in range(60):
    acc += total(data)
print(acc)

# deopt giữa vòng lặp: phần tử đổi kiểu
fn mixsum(xs):
    mut s = 0.0
    mut i = 0
    while i < xs.len:
        s += xs[i]
        i += 1
    return s
mut ys = []
for k in range(5000):
    ys.append(k * 1.0)
ys[4000] = 7
print(mixsum(ys))

# deopt: lỗi chỉ số ngoài phạm vi phải đúng thông điệp, sau khi đã ghi list
fn fill(xs, n):
    mut i = 0
    while i < n:
        xs[i] = i * 2
        i += 1
    return 0
mut zs = [0] * 3000
try:
    fill(zs, 3005)
except "assign range":
    print("bắt:", err)
print(zs[0], zs[2999], len(zs))
try:
    fill(zs, 3000)
except:
    print("không được tới")
print(zs[1500])

# DP: dãy fibo bằng list
fn fibs(n):
    mut f = [0] * (n + 1)
    f[1] = 1
    mut i = 2
    while i <= n:
        f[i] = (f[i - 1] + f[i - 2]) % 1000000007
        i += 1
    return f[n]
print(fibs(100000))

# list float + bool ghi trong vòng lặp nóng
fn smooth(n):
    mut a = [0.0] * n
    mut seen = [false] * n
    mut i = 1
    while i < n:
        a[i] = a[i - 1] * 0.5 + i
        seen[i] = i % 3 == 0
        i += 1
    mut c = 0
    i = 0
    while i < n:
        if seen[i]:
            c += 1
        i += 1
    return a[n - 1] + c
print(smooth(50000))

# callee bị huỷ (chia 0) giữa vùng OSR đã ghi list -> deopt tại CALL,
# interpreter gọi lại callee và báo đúng lỗi; các phần tử đã ghi vẫn còn
fn inv(x):
    mut r = 1.0 / x
    return r
fn fill_inv(xs, n):
    mut i = 0
    while i < n:
        xs[i] = inv(n - 2000 - i)
        i += 1
    return 0
mut ws = [0.0] * 5000
try:
    fill_inv(ws, 5000)
except "division":
    print("bắt:", err)
print(ws[0], ws[2999], ws[3000])

# list global đọc trong hàm JIT
mut table = []
for k in range(64):
    table.append(k * k)
fn lookup_sum(n):
    mut s = 0
    mut i = 0
    while i < n:
        s += table[i % 64]
        i += 1
    return s
print(lookup_sum(200000))
table[5] = 1.5
print(lookup_sum(100))
print("JITLIST OK")
