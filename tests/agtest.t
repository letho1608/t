# tests/agtest.t — std/autograd.t: vi phân ngược trên tensor (khớp mọi runtime)
import autograd as ag

# kiểm gradient từng phép bằng sai phân hữu hạn
fn numgrad(f, x, eps):
    mut g = zeros(x.shape)
    mut i = 0
    flat = x.flatten()
    while i < x.size:
        mut xp = flat.copy()
        mut xm = flat.copy()
        xp[i] = xp[i] + eps
        xm[i] = xm[i] - eps
        d = (f(xp.reshape(x.shape)) - f(xm.reshape(x.shape))) / (2 * eps)
        mut gf = g.flatten()
        gf[i] = d
        g = gf.reshape(x.shape)
        i += 1
    return g

seed(5)
A = randn([3, 4])
B = randn([4, 2])
bias = randn([1, 2])
fn loss_of(a):
    return ((a @ B + bias).tanh() * 1.5).sigmoid().mean()
pa = ag.param(A)
L = ag.mean(ag.sigmoid(ag.mul(ag.tanh(ag.add(ag.matmul(pa, ag.const(B)), ag.const(bias))), 1.5)))
ag.backward(L)
ng = numgrad(loss_of, A, 0.000001)
print("loss", L.val.item(), "khớp", (pa.grad - ng).abs().max() < 0.000001)

# broadcast: gradient của bias cộng dồn theo hàng
pb = ag.param(bias)
L2 = ag.sumall(ag.add(ag.const(A @ B), pb))
ag.backward(L2)
print(pb.grad, pb.grad.shape)

# div / exp / log / relu / sub
pc = ag.param(tensor([[0.5, 2.0], [1.5, -1.0]]))
L3 = ag.sumall(ag.sub(ag.div(ag.exp(pc), 2), ag.log(ag.add(ag.relu(pc), 1))))
ag.backward(L3)
fn f3(c):
    return (c.exp() / 2 - (c.relu() + 1).log()).sum()
print((pc.grad - numgrad(f3, pc.val, 0.000001)).abs().max() < 0.000001)

# huấn luyện MLP XOR chỉ bằng autograd + sgd
seed(1)
X = ag.const(tensor([[0, 0], [0, 1], [1, 0], [1, 1]]))
Y = tensor([[0], [1], [1], [0]])
W1 = ag.param(randn([2, 4]))
b1 = ag.param(zeros([1, 4]))
W2 = ag.param(randn([4, 1]))
b2 = ag.param(zeros([1, 1]))
params = [W1, b1, W2, b2]
mut loss = nil
for step in range(500):
    h = ag.tanh(ag.add(ag.matmul(X, W1), b1))
    p = ag.sigmoid(ag.add(ag.matmul(h, W2), b2))
    # cross-entropy: -mean(Y log p + (1 - Y) log(1 - p))
    t1 = ag.mul(ag.log(p), Y)
    t2 = ag.mul(ag.log(ag.sub(1, p)), 1 - Y)
    loss = ag.mul(ag.mean(ag.add(t1, t2)), -1)
    ag.backward(loss)
    ag.sgd(params, 1.0)
print("xor loss", loss.val.item() < 0.01)
pred = ((X.val @ W1.val + b1.val).tanh() @ W2.val + b2.val).sigmoid()
print([1 if v > 0.5 else 0 for v in pred.flatten().tolist()])
print("AUTOGRAD OK")
