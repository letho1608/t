# tests/tensor.t — tensor n-chiều v2.2 (khớp từng bit giữa t.py, VM C, eval.t)

# --- tạo + shape + in ---
a = tensor([[1, 2, 3], [4, 5, 6]])
print(a, a.shape, a.ndim, a.size, a.len, len(a), type(a))
print(tensor([1, 2, 3]), tensor(5), tensor([]).shape, tensor([[[1], [2]]]).shape)
print(zeros([2, 2]), ones(3), full([2, 3], 7), eye(3))
print(arange(5), arange(2, 5), arange(0, 1, 0.25), arange(5, 0, -2))
print(tensor(a) == a, a == tensor([[1, 2, 3], [4, 5, 7]]), a != a.T)

# --- broadcast + phép từng phần tử ---
b = tensor([10, 20, 30])
print(a + b, a * 2, 1 - a, a / 4)
col = tensor([[100], [200]])
print(a + col, col * b)
print(a ** 2, 2 ** tensor([1, 2, 3]), -a)
print(tensor([1, -2]) / 0, tensor([0]) / 0)

# --- @ ---
m = tensor([[1, 2], [3, 4]])
v = tensor([5, 6])
print(m @ m, m @ v, v @ m, v @ v, m.dot(v))
print(a @ a.T, a.T @ a)

# --- rút gọn ---
print(a.sum(), a.mean(), a.max(), a.min(), a.argmax(), a.argmin())
print(a.sum(0), a.sum(1), a.mean(0), a.max(1), a.min(0), a.argmax(1), a.sum(-1))
print(sum(a), min(a), max(a))

# --- hàm từng phần tử ---
x = tensor([-2.0, -0.5, 0.0, 0.5, 2.0])
print(x.relu(), x.abs(), abs(x), x.sigmoid())
print(x.exp(), x.tanh(), x.floor(), x.ceil())
print(tensor([1, 4, 9]).sqrt(), tensor([1, 2.718281828459045]).log(), x.sin(), x.cos())
print(sqrt(16), exp(0), log(1), floor(2.7), ceil(2.1), floor(-2.5), sin(0), cos(0), tanh(0))
print(sqrt(x.abs()), exp(tensor([0, 1])), pi)
print(a.softmax(), tensor([1, 2, 3]).softmax().sum())
print(x.clip(-1, 1), tensor([3.5]).item())

# --- reshape / transpose / flatten / tolist ---
r = arange(12).reshape(3, 4)
print(r, r.shape, r.reshape([2, -1]).shape, r.reshape(-1).shape)
print(r.T, r.transpose().shape, r.flatten())
print(arange(24).reshape(2, 3, 4).T.shape, arange(8).reshape(2, 2, 2).T)
print(r.tolist(), type(r.tolist()), r.copy() == r)

# --- chỉ số theo hàng, gán, cắt, lặp ---
print(r[1], r[1][2], r[2].shape, a[0] @ a[1])
mut w = zeros([3, 2])
w[0] = tensor([1, 2])
w[2] = 9
w[1][0] = 5
print(w)
mut z = zeros(3)
z[1] = 4.5
print(z, r[1:3], r[::2].shape, r[::-1][0])
mut tot = 0.0
for row in r:
    tot += row.sum()
print(tot, 5 in r, 99 in r)

# --- so sánh từng phần tử -> mặt nạ 0/1 ---
print(a > 2, a <= tensor([1, 5, 3]), 3 >= a, a.eq(a.T.T), a.ne(5), (a > 2).sum())
print(x * (x > 0), tensor([0.0 * 1e400]) < 1)

# --- chỉ số nhiều trục: t[i, j], t[:, j], gán vùng ---
t3 = arange(24).reshape(2, 3, 4)
print(t3[1, 2, 3], t3[1, 2], t3[:, 1], t3[:, 1, 2], t3[0, :, 0])
mut g = zeros([3, 3])
g[1, 1] = 5
g[:, 0] = 7
g[2, :] = tensor([1, 2, 3])
print(g, g[2, 1], g[:, 2].sum())
# lát cắt theo nhiều trục (bản sao), cận âm như Python
s6 = arange(24).reshape(4, 6)
print(s6[0:2, 1], s6[:, 1:3], s6[::-1, 0], s6[1:3, ::2], s6[3, -3:], s6[:, ::-2].shape)
mut z4 = zeros([4, 4])
z4[1:3, 1:3] = 1
z4[0, ::2] = tensor([7, 8])
print(z4)

# --- ngẫu nhiên tất định (splitmix64) ---
seed(7)
u = rand([2, 3])
print(u)
seed(7)
print(rand([2, 3]) == u, rand(2))
seed(123)
g = randn(5)
print(g, g.mean())

# --- lỗi bắt được bằng try ---
fn tryit(f):
    try:
        f()
        return "ok"
    except:
        return err
fn e1():
    return tensor([1, 2]) + tensor([1, 2, 3])
fn e2():
    return tensor([[1, 2], [3]])
fn e3():
    return tensor([[1, 2]]) @ tensor([[1, 2]])
fn e4():
    return arange(6).reshape(4, 2)
fn e5():
    return a.sum(5)
fn e6():
    return a[7]
fn e7():
    return tensor([1, 2]) < "x"
print(tryit(e1), tryit(e2), tryit(e3))
print(tryit(e4), tryit(e5), tryit(e6), tryit(e7))

# --- mạng nơ-ron nhỏ: MLP 2-4-1 học XOR (tanh + sigmoid, cross-entropy, gradient tay) ---
seed(1)
X = tensor([[0, 0], [0, 1], [1, 0], [1, 1]])
Y = tensor([[0], [1], [1], [0]])
mut W1 = randn([2, 4])
mut b1 = zeros([1, 4])
mut W2 = randn([4, 1])
mut b2 = zeros([1, 1])
lr = 1.0
mut loss = 0.0
for step in range(500):
    h = (X @ W1 + b1).tanh()
    p = (h @ W2 + b2).sigmoid()
    loss = -(Y * p.log() + (1 - Y) * (1 - p).log()).mean()
    dz2 = (p - Y) / 4
    dh = (dz2 @ W2.T) * (1 - h * h)
    W2 -= h.T @ dz2 * lr
    b2 -= dz2.sum(0).reshape(1, 1) * lr
    W1 -= X.T @ dh * lr
    b1 -= dh.sum(0).reshape(1, 4) * lr
pred = ((X @ W1 + b1).tanh() @ W2 + b2).sigmoid()
print("loss", loss, "pred", pred.reshape(-1))
mut ok = 0
for i in range(4):
    if (pred[i][0] > 0.5) == (Y[i][0] > 0.5):
        ok += 1
print("XOR dung", ok, "/ 4")
print("TENSOR OK")

# view: hàng / khối liền dùng chung bộ nhớ với tensor gốc; cột / bước != 1 là bản sao
fn view_demo():
    t = arange(12).reshape(3, 4)
    r = t[1]
    r[0] = 99
    print(t)
    t[1, 2] = -1
    print(r)
    u = t[1, :]
    u[3] = 7
    print(t[1], t[1, 1:3])
    c = t[:, 0]
    c[0] = 1000
    print(t[0], c)
    b = t[0:2]
    b[0, 0] = 50
    print(t[0], t[::2][0])
    t3 = zeros([2, 2, 3])
    p = t3[1][0]
    p[2] = 5
    print(t3)
    print(r * 2, r.sum(), r.shape, r.copy())
    return t[2]
keep = view_demo()
print(keep, keep.sum())
# view theo bước: cột, bước != 1, bước âm, transpose, reshape (khớp numpy)
fn strided_demo():
    t = arange(12).reshape(3, 4)
    c = t[:, 1]
    print(c, c.sum(), c.tolist(), 5.0 in c)
    t[0, 1] = 100
    print(c, -c, c * 2, c == tensor([100, 5, 9]))
    c[2] = -9
    print(t)
    e = t[::2]
    e[1, 0] = 77
    print(t[2], e)
    rv = t[:, ::-1]
    print(rv[0], rv[:, 0])
    rv[1, 0] = 0
    print(t[1])
    tt = t.T
    print(tt.shape, tt[1], tt[:, 0])
    tt[0, 2] = 555
    print(t[2, 0], tt @ tensor([1, 0, 0]))
    s2 = t[::2, 1::2]
    print(s2, s2.shape, s2.T)
    s2[1, 1] = 3.5
    print(t[2])
    t[:, 2] = tensor([1, 2, 3])
    print(c.copy(), t[:, 2])
    t[1:3, :] = t[0:2]
    print(t)
    a = arange(6).reshape(1, 6)
    a[:, 1:] = a[:, :5]
    print(a)
    g = arange(8)
    h = g[::-2]
    h[0] = 70
    print(g, h, h.reshape(2, 2))
    m = g.reshape(2, 4)
    m[0, 0] = -1
    print(g[0], m.T.reshape(8))
    print(t[:, 5:5], t[0:0].shape, t[:, ::3].tolist())
    print(tt.T == t, tt.sum(0), tt.sum(1), tt.max())
    return t[:, 3]
kc = strided_demo()
print(kc, kc.mean())
