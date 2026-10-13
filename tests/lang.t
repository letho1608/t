# tests/lang.t — cú pháp v2.2: a if c else b, in/not in, for a, b in,
# += -= *= /= %=, switch/case/default, struct (field + method + self),
# set, method mới của str/list/dict, số mũ 1e-5 (khớp cả 4 runtime)

# --- biểu thức điều kiện ---
x = 5
y = "big" if x > 3 else "small"
print(y, 1 if false else 2, "a" if x == 1 else "b" if x == 5 else "c")
fn sign(v):
    return -1 if v < 0 else (0 if v == 0 else 1)
print(sign(-4), sign(0), sign(9))
if x > 3:
    print("stmt-if van chay")

# --- in / not in ---
print(2 in [1, 2], 5 in [1, 2], "b" in "abc", "k" in {"k": 1}, 3 not in [1, 2])
print(2.0 in tensor([1, 2]), "z" not in "abc")

# --- for nhiều biến ---
for i, v in enumerate(["p", "q"]):
    print(i, v)
d = {"a": 1, "b": 2}
for k, v in d.items():
    print(k, v)
for a, b, c in [[1, 2, 3], [4, 5, 6]]:
    print(a + b + c)

# --- gán kèm toán tử ---
mut n = 1
n += 4
n *= 3
n -= 1
n %= 8
print(n)
mut xs = [1, 2, 3]
xs[1] += 10
print(xs)

# --- switch ---
fn kind(v):
    switch v:
        case 1, 2:
            return "small"
        case "x":
            return "ex"
        default:
            return "other"
print(kind(1), kind(2), kind("x"), kind(9))
for t in [1, 7]:
    switch t * 2:
        case 2:
            print("hai")
        case 14:
            print("muoi bon")

# --- struct ---
struct Point:
    x, y
    fn norm2(self):
        return self.x * self.x + self.y * self.y
    fn move(self, dx, dy):
        self.x += dx
        self.y = self.y + dy
        return self
p = Point(3, 4)
print(p, p.norm2(), type(p))
p.move(2, 0)
print(p.x, p["y"], p == Point(5, 4), p != Point(5, 4))
p.move(1, 1).move(1, 1)
print(p)
try:
    p.z = 1
except "no field":
    print("loi:", err)
try:
    print(p.zz)
except "has no":
    print("loi:", err)
struct Node:
    val
    next
mut head = nil
for i in range(3):
    head = Node(i, head)
mut cur = head
mut acc = []
while cur != nil:
    acc.append(cur.val)
    cur = cur.next
print(acc)

# --- set ---
s = set([3, 1, 3, 2])
s.add(7)
print(s, s.len, 3 in s, s.has(9), type(s), set(), len(s))
s.remove(1)
print(s, s.union([9, 3]), s.inter([3, 7, 8]), s.diff([3]))
mut tot = 0
for e in s:
    tot += e
print(tot, sort(s.list()), set("hello"))

# --- method mới ---
mut ys = [1, 2]
ys.insert(0, 9)
ys.extend([5, 6])
print(ys, ys.index(5), ys.pop(), ys)
ys.clear()
print(ys, ys.len)
print("hello".replace("l", "L"), "abc".find("c"), "abc".find("z"))
print("abc".startswith("ab"), "abc".endswith("bc"), "abc".endswith("x"))
dd = {"a": 1, "b": 2}
print(dd.items(), dd.has("a"), dd.has("z"))
dd.remove("a")
print(dd)

# --- số thực + số mũ ---
print(0.1 + 0.2, 1 / 3, 2.5, 1e-5, 1.5e3, 2E2)
# % theo ngữ nghĩa floor (dấu theo số chia) ở mọi runtime
print(-7 % 3, 7 % -3, -7.5 % 2, 7.5 % -2, -6 % 3)

# --- hồi quy: inline lồng nhau (từng dùng chung biến tạm -> sai) ---
fn add(a, b):
    return a + b
fn mul(a, b):
    return a * b
fn sq(v):
    return mul(v, v)
fn useA():
    let k = 5
    let j = 6
    return sq(3) + k + j
fn useB():
    return sq(4)
print(add(1, add(2, 3)), add(add(1, 2), add(3, 4)), useA(), useB())

# --- builtin mềm có thể bị che tên ---
fn wrap():
    set = 3
    return set + 1
print(wrap(), type(set([1])))
print("LANG OK")
