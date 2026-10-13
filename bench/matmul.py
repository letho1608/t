def mk(n, k):
    return [[((i * 7 + j * 3 + k) % 11) / 10.0 for j in range(n)] for i in range(n)]

A = mk(120, 1)
B = mk(120, 2)
Bt = list(zip(*B))
s = 0.0
for t in range(20):
    C = [[sum(x * y for x, y in zip(row, col)) for col in Bt] for row in A]
    s += sum(sum(r) for r in C)
print(round(s))
