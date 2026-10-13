# bench/fib.t — đệ quy số nguyên
fn fib(n):
    if n < 2:
        return n
    return fib(n - 1) + fib(n - 2)

print(fib(30))
