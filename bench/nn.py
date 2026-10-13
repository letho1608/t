import numpy as np

def mkx(n, d):
    i = np.arange(n).reshape(-1, 1)
    j = np.arange(d).reshape(1, -1)
    return ((i * 31 + j * 17) % 97) / 97.0 - 0.5

X = mkx(512, 16)
Y = ((X.sum(1) > 0) * 1.0).reshape(512, 1)
W1 = mkx(16, 64) * 0.5
b1 = np.zeros((1, 64))
W2 = mkx(64, 1) * 0.5
b2 = np.zeros((1, 1))
loss = 0.0
for step in range(300):
    h = np.tanh(X @ W1 + b1)
    p = 1 / (1 + np.exp(-(h @ W2 + b2)))
    loss = -(Y * np.log(p) + (1 - Y) * np.log(1 - p)).mean()
    dz2 = (p - Y) / 512
    dh = (dz2 @ W2.T) * (1 - h * h)
    W2 -= h.T @ dz2 * 0.5
    b2 -= dz2.sum(0).reshape(1, 1) * 0.5
    W1 -= X.T @ dh * 0.5
    b1 -= dh.sum(0).reshape(1, 64) * 0.5
print(round(loss * 10000))
