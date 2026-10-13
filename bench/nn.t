# bench/nn.t — huấn luyện MLP 16-64-1 (batch 512, 300 bước) bằng tensor
fn mkx(n, d):
    mut rows = []
    for i in range(n):
        mut r = []
        for j in range(d):
            r.append(((i * 31 + j * 17) % 97) / 97.0 - 0.5)
        rows.append(r)
    return tensor(rows)

X = mkx(512, 16)
Y = ((X.sum(1) > 0) * 1.0).reshape(512, 1)
mut W1 = mkx(16, 64) * 0.5
mut b1 = zeros([1, 64])
mut W2 = mkx(64, 1) * 0.5
mut b2 = zeros([1, 1])
mut loss = 0.0
for step in range(300):
    h = (X @ W1 + b1).tanh()
    p = (h @ W2 + b2).sigmoid()
    loss = -(Y * p.log() + (1 - Y) * (1 - p).log()).mean()
    dz2 = (p - Y) / 512
    dh = (dz2 @ W2.T) * (1 - h * h)
    W2 -= h.T @ dz2 * 0.5
    b2 -= dz2.sum(0).reshape(1, 1) * 0.5
    W1 -= X.T @ dh * 0.5
    b1 -= dh.sum(0).reshape(1, 64) * 0.5
print(round(loss * 10000))
