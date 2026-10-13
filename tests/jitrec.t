# JIT trên record (struct) mượn: đọc field số + gọi method tĩnh, guard kiểu/tag.
# Output phải giống hệt interpreter (TVM_NOJIT) và t.py (tools/test_jit.py).
struct P:
    x, y
    fn norm2(self):
        return self.x * self.x + self.y * self.y
    fn dot(self, o):
        return self.x * o.x + self.y * o.y
    fn scaled(self, k):
        return self.norm2() * k
struct Q:
    x, y
    fn norm2(self):
        return self.x + self.y
struct Flag:
    on, w

fn energy(p, n):
    mut s = 0.0
    mut i = 0
    while i < n:
        s = s + p.norm2() + p.dot(p) * 0.5 + p.scaled(2)
        i = i + 1
    return s

fn cross(p, q, n):
    mut t = 0
    for i in range(n):
        t = t + p.x * q.y - p.y * q.x + i % 3
    return t

fn weigh(f, xs):
    mut s = 0
    for v in xs:
        if f.on:
            s = s + v * f.w
        else:
            s = s - v
    return s

fn n2(p):
    return p.norm2()

a = P(3.0, 4.0)
b = P(1, 2)
print(energy(a, 30000))
print(cross(b, P(5, 7), 30000))
# hàm đã JIT: field đổi kiểu (int -> float) -> guard hỏng -> interpreter chạy lại
c = P(1, 2)
mut acc = 0
for i in range(3000):
    acc = acc + n2(c)
print(acc)
c.x = 2.5
print(n2(c), cross(c, b, 200))
# cùng tên method nhưng kiểu struct khác -> không dùng code của P
q = Q(10, 20)
print(n2(q))
mut acc2 = 0
for i in range(3000):
    acc2 = acc2 + n2(q) + n2(b)
print(acc2)
# field bool + list
f = Flag(true, 3)
xs = [1, 2, 3, 4, 5]
mut tot = 0
for i in range(2000):
    tot = tot + weigh(f, xs)
print(tot)
f.on = false
print(weigh(f, xs))
# field bị sửa giữa các lần gọi: luôn đọc giá trị hiện tại
mut acc3 = 0
for i in range(2000):
    b.x = i
    acc3 = acc3 + n2(b)
print(acc3)
# field không phải số -> lỗi như interpreter
c.y = "z"
try:
    print(n2(c))
except:
    print("lỗi kiểu")
