# load(): import động lúc chạy (tên module tính được), khớp t.py / VM / eval.t
nm = "tests/loadmod" + "_m"
m = load(nm)
print(type(m), m.VERSION)
print(m.sq(7), m.apply(m.sq, [1, 2, 3]))
print(m.bump(), m.bump(), m.COUNT)
m2 = load(nm)                 # nạp lần 2: dùng lại (không chạy top-level nữa)
print(m2.bump(), m.COUNT[0])
p = m.Pt(3, 4)
print(p.norm2(), type(p))
print(m.hot(5000))
f = m.sq
print(map(f, [4, 5]))
try:
    m.boom()
except:
    print("bắt lỗi:", err)
try:
    load("tests/khong_co_module")
except:
    print("thiếu module: ok")
# field sống: thấy giá trị global hiện tại của module, ghi m.X = v vào module
print(m.LEVEL, m.set_level(5), m.LEVEL, m2.LEVEL)
m.LEVEL = 42
print(m.get_level(), m2.LEVEL, m["LEVEL"])
m["LEVEL"] = 7
print(m.get_level())
mut tot = 0
for i in range(3000):
    tot += m.set_level(i) - m.LEVEL
print(tot, m.LEVEL)
print("xong")
