# tests/jitdict.t — JIT dict + chuỗi sở hữu được (G7)
# Vòng nóng nào cũng phải cho kết quả khớp interpreter (TVM_NOJIT) và t.py.

# đếm từ như bench/dict.t (nhỏ hơn để test nhanh)
fn count_words(n):
    mut d = {}
    mut i = 0
    while i < n:
        k = "w" + str(i % 1000)
        d[k] = d.get(k, 0) + 1
        i = i + 1
    return d.len

print(count_words(3000))
assert count_words(100) == 100
assert count_words(1500) == 1000

# key int + val float/bool
fn mix(n):
    mut d = {}
    mut i = 0
    while i < n:
        d[i] = i * 1.5
        d[100000 + i] = true
        i = i + 1
    return d.len

print(mix(2000))
assert mix(100) == 200

# đọc d[k] (IDX), thiếu key báo lỗi như interpreter
fn readback(n):
    mut d = {"a": 1, "b": 2}
    mut s = 0
    mut i = 0
    while i < n:
        s = s + d["a"] + d["b"]
        i = i + 1
    return s

print(readback(3000))
assert readback(10) == 30
try:
    d0 = {"a": 1}
    print(d0["zz"])
except:
    print("bắt:", err)

# get trúng giá trị không phải int -> deopt vẫn đúng
fn getmix(n):
    mut d = {"s": "txt", "n": 41}
    mut s = 0
    mut i = 0
    while i < n:
        s = s + d.get("n", 0)
        i = i + 1
    return s

print(getmix(3000))
assert getmix(5) == 205
try:
    mut d1 = {}
    d1[[]] = 1
except:
    print("bắt:", err)

# nối chuỗi nhiều lần + tự nối (k + k cùng wrapper)
fn cat(n):
    mut s = ""
    mut i = 0
    while i < n:
        s = "a" + s + "b"
        i = i + 1
    return s

print(len(cat(100)), cat(3))
assert cat(3) == "aaabbb"
assert len(cat(100)) == 200

fn selfcat(n):
    mut s = "x"
    mut i = 0
    while i < n:
        s = s + s
        if len(s) > 100:
            s = "x" + ""
        i = i + 1
    return len(s)

print(selfcat(5000))
assert selfcat(10) == 8

# giá trị chuỗi/dict/record trong dict nóng
struct VR:
    v

fn vtypes(n):
    mut d = {}
    mut i = 0
    while i < n:
        d["s"] = "v" + str(i)
        d["f"] = i * 0.5
        d["r"] = VR(i)
        i = i + 1
    return d.len

print(vtypes(2000))
assert vtypes(10) == 3

fn show(x):
    mut t = "[" + str(x) + "]"
    return t

for i in range(100):
    show(i)
print(show(42), show(1.5), show(true))
assert show(42) == "[42]"

# dict mới trong entry (không vòng lặp, không SETIDX): NEW_DICT + len + return
fn mkd2(x):
    mut d = {}
    return d.len + x

for i in range(100):
    mkd2(i)
print(mkd2(7))
assert mkd2(7) == 7

# dict global ở top-level (mainmode STOREG)
mut g = {}
mut gi = 0
for i in range(3000):
    g["k" + str(i % 50)] = i
    gi = gi + 1
print(g.len, gi)
assert g.len == 50
assert g["k7"] % 50 == 7
# gán lại dict/chuỗi mới vào global trong vòng top-level (write-through + dec-old)
mut h2 = {"init": 1}
mut msg = ""
for i in range(3000):
    h2 = {}
    h2["k"] = i
    msg = "n" + str(i)
print(h2.len, h2["k"], len(msg))
assert h2["k"] == 2999
print("JITDICT OK")
