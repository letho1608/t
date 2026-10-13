# tests/closure.t — closure/upvalue (chạy: python t.py tests/closure.t | trun)
fn maker(n):
    mut count = n
    fn inc(d):
        count = count + d
        return count
    return inc

c = maker(10)
assert c(1) == 11
assert c(5) == 16
c2 = maker(0)
assert c2(100) == 100
assert c(0) == 16

fn adder(x):
    fn f(y):
        return x + y
    return f

add7 = adder(7)
assert add7(3) == 10
assert adder(1)(2) == 3

fn outer():
    mut m = 1
    fn mid():
        mut k = 2
        fn inner():
            return m + k
        return inner()
    return mid()
assert outer() == 3

# shadow: local che outer
mut s = "g"
fn sh():
    mut s = "l"
    return s
assert sh() == "l"
assert s == "g"

# global capture (không cần cell)
mut g = 100
fn getg():
    return g + 1
assert getg() == 101
g = 200
assert getg() == 201

# capture tham số + đổi qua cell
fn acc():
    mut t = 0
    fn push(v):
        t = t + v
        return t
    fn get():
        return t
    return [push, get]
r = acc()
assert r[0](5) == 5
assert r[0](5) == 10
assert r[1]() == 10

# for-var capture mỗi vòng tươi
fns = []
for i in range(3):
    fn g():
        return i * 10
    fns.append(g)
assert fns[0]() == 0
assert fns[1]() == 10
assert fns[2]() == 20

# closure làm đối số
fn apply(f, x):
    return f(x)
assert apply(add7, 5) == 12
print("CLOSURE OK")
