print("hello T")

fn fib(n):
    if n < 2:
        return n
    else:
        return fib(n - 1) + fib(n - 2)

print(fib(15))

let x = 10
mut y = 20
y = y + 5
print(x + y)

t = tensor([1, 2, 3])
print(t + t)
print(t @ t)

mut s = 0
for i in range(100):
    s = s + i
print(s)

fn add(a, b):
    return a + b
let f = add
print(f(3, 4))
