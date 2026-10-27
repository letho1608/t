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

# G6: JIT tạo record (__new), gán field số (SETATTR), chuỗi hằng
struct V:
    x, y
    fn norm2(self):
        return self.x * self.x + self.y * self.y
    fn moved(self, dx):
        mut r = V(self.x + dx, self.y)
        return r

fn runv(n):
    mut acc = V(0, 0)
    d = V(1, 2)
    mut s = 0
    mut i = 0
    while i < n:
        acc = acc.moved(1)
        s = s + acc.norm2()
        i = i + 1
    return s

print(runv(3000))
assert runv(10) == 385
# gán field số trong vòng nóng + đọc lại
fn bump(p, n):
    mut i = 0
    while i < n:
        p.x = p.x + 1
        p.y = i * 2
        i = i + 1
    return p.x + p.y

qv = V(0, 0)
print(bump(qv, 5000))
assert qv.x == 5000
assert qv.y == 9998
# constructor làm câu lệnh (tạo rồi bỏ) + hàm trả record mới
fn mkv(i):
    return V(i, i * 3)

fn usemk(n):
    mut s = 0
    mut i = 0
    while i < n:
        V(0, 0)
        r = mkv(i)
        s = s + r.x + r.y
        i = i + 1
    return s

print(usemk(3000))
assert usemk(5) == 40
# chuỗi hằng: ==, !=, len, điều kiện, tham số
fn sclass(s):
    if s == "quit":
        return 0 - 1
    if s != "go":
        return 0
    return len(s)

for i in range(200):
    sclass("go")
    sclass("quit")
    sclass("xx")
print(sclass("go"), sclass("quit"), sclass("xx"), sclass(""))
assert sclass("go") == 2
assert sclass("") == 0
# global chuỗi đổi kiểu sau khi JIT -> deopt vẫn đúng
mut mode = "go"
fn chk(n):
    mut c = 0
    mut i = 0
    while i < n:
        if mode == "go":
            c = c + 1
        i = i + 1
    return c

for i in range(200):
    chk(10)
print(chk(3000))
mode = 7
print(chk(100))
# deopt giữa vòng OSR khi local đang giữ record mới
fn fill2(n):
    mut a2 = P(1, 2)
    mut xs2 = [0] * 2003
    mut i = 0
    while i < n:
        a2 = P(i, i + 1)
        xs2[i] = i * 2
        i = i + 1
    return a2.x

try:
    fill2(2005)
except "assign range":
    print("bắt:", err)
print(fill2(10))
# hàm nhiều câu lệnh trả record mới: gọi nóng từ interpreter qua t_jit_call
fn fresh(i):
    mut t = i * 2
    return V(t, t + 1)

for i in range(100):
    fresh(i)
print(fresh(7).x, fresh(7).y)
assert fresh(7).x == 14
assert fresh(3).y == 7
print(map(fresh, [1, 2, 3]))

# G8: SETATTR trong entry khi receiver là local owned hoặc method sửa field
struct Mover:
    x, y
    fn move(self, dx, dy):
        self.x = self.x + dx
        self.y = self.y + dy
        return self

fn entry_mutate(v):
    mut p = Mover(v, v * 2)
    p.x = p.x + 10
    p.y = p.y - 3
    return p.x * 100 + p.y

for i in range(100):
    entry_mutate(i)
assert entry_mutate(5) == 1507

fn hot_move(n):
    mut m = Mover(0, 0)
    mut i = 0
    while i < n:
        m.move(1, 2)
        i = i + 1
    return m.x + m.y

assert hot_move(2500) == 7500

# G9: hàm đệ quy trả record (đoán retty theo shape record)
fn build_rec(n):
    if n <= 0:
        return V(0, 0)
    prev = build_rec(n - 1)
    return V(prev.x + 1, prev.y + n)

for i in range(60):
    build_rec(5)
r_rec = build_rec(10)
assert r_rec.x == 10
assert r_rec.y == 55
print("JITREC2 OK")
