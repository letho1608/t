# tests/lambda.t — lambda fn(x): e + comprehension [e for x in xs if c] / {k: v for ...}
sq = fn(x): x * x
print(sq(5), map(fn(v): v + 1, [1, 2, 3]), sort([3, 1, 2], fn(v): 0 - v))
xs = [1, 2, 3, 4, 5, 6]
print([x * x for x in xs], [x for x in xs if x % 2 == 0])
print([[i, j] for i in range(3) for j in range(i)], {k: k * 10 for k in xs if k > 4})
print([a + b for a, b in [[1, 2], [3, 4]]], [c.upper() for c in "abc"])
fn adder(n):
    return fn(x): x + n
add5 = adder(5)
print(add5(1), (fn(a, b): a * b)(6, 7))
print([fn(): i for i in range(3)][2](), [[y * 2 for y in row] for row in [[1, 2], [3]]])
mut k = 0
while len([z for z in range(k)]) < 3:
    k += 1
print(k, f"{[q * 3 for q in range(3)]}", 1 if [v for v in xs if v > 9] else 2)
fn mk():
    mut n = 10
    f = fn(d): n + d
    n = 20
    return f(1)
print(mk())
# tổng hợp với tensor + sort theo khoá lambda
words = ["pear", "fig", "banana", "kiwi"]
print(sort(words, fn(w): w.len), {w: w.len for w in words if w.len > 3})
m = tensor([[i * 3 + j for j in range(3)] for i in range(2)])
print(m, sum([x * x for x in range(10)]), filter(fn(v): v % 3 == 0, range(10)))
print("LAMBDA OK")
