# bench/matmul.t — nhân ma trận 120x120 (tensor C) 20 lần
fn mk(n, k):
    mut rows = []
    for i in range(n):
        mut r = []
        for j in range(n):
            r.append(((i * 7 + j * 3 + k) % 11) / 10.0)
        rows.append(r)
    return tensor(rows)

A = mk(120, 1)
B = mk(120, 2)
mut s = 0.0
for t in range(20):
    s += (A @ B).sum()
print(round(s))
