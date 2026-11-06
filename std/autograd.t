# std/autograd.t — vi phân tự động ngược (reverse-mode) trên tensor, viết bằng T.
# Dùng:
#   import autograd as ag
#   w = ag.param(randn([2, 3]))
#   x = ag.const(tensor([[1, 2]]))
#   loss = ag.mean(ag.tanh(ag.matmul(x, w)))
#   ag.backward(loss)
#   print(w.grad)                 # cùng shape với w.val
# Phép có broadcast (add/sub/mul/div) tự cộng dồn gradient về shape gốc.

NEXT_ID = [0]

struct Node:
    val, grad, parents, backfn, op, id

fn mk(val, parents, backfn, op):
    NEXT_ID[0] += 1
    return Node(val, nil, parents, backfn, op, NEXT_ID[0])

fn as_tensor(x):
    if type(x) == "tensor":
        return x
    return tensor([x]) if type(x) == "int" or type(x) == "float" else tensor(x)

# tham số học được / hằng số: lá của đồ thị
fn param(x):
    return mk(as_tensor(x), [], nil, "param")

fn const(x):
    return mk(as_tensor(x), [], nil, "const")

fn lift(x):
    if type(x) == "Node":
        return x
    return const(x)

# gradient g (shape kết quả broadcast) -> shape gốc: cộng các trục đã broadcast
fn unbroadcast(g, shape):
    mut r = g
    while r.ndim > shape.len:
        r = r.sum(0)
    mut ax = 0
    while ax < shape.len:
        if shape[ax] == 1 and r.shape[ax] != 1:
            mut keep = r.shape
            keep[ax] = 1
            r = r.sum(ax).reshape(keep)
        ax += 1
    return r

fn add(a, b):
    a = lift(a)
    b = lift(b)
    return mk(a.val + b.val, [a, b],
        fn(g): [unbroadcast(g, a.val.shape), unbroadcast(g, b.val.shape)], "add")

fn sub(a, b):
    a = lift(a)
    b = lift(b)
    return mk(a.val - b.val, [a, b],
        fn(g): [unbroadcast(g, a.val.shape), unbroadcast(-g, b.val.shape)], "sub")

fn mul(a, b):
    a = lift(a)
    b = lift(b)
    return mk(a.val * b.val, [a, b],
        fn(g): [unbroadcast(g * b.val, a.val.shape), unbroadcast(g * a.val, b.val.shape)], "mul")

fn div(a, b):
    a = lift(a)
    b = lift(b)
    return mk(a.val / b.val, [a, b],
        fn(g): [unbroadcast(g / b.val, a.val.shape),
                unbroadcast(-g * a.val / (b.val * b.val), b.val.shape)], "div")

fn matmul(a, b):
    a = lift(a)
    b = lift(b)
    return mk(a.val @ b.val, [a, b], fn(g): [g @ b.val.T, a.val.T @ g], "matmul")

fn tanh(a):
    a = lift(a)
    y = a.val.tanh()
    return mk(y, [a], fn(g): [g * (1 - y * y)], "tanh")

fn sigmoid(a):
    a = lift(a)
    y = a.val.sigmoid()
    return mk(y, [a], fn(g): [g * y * (1 - y)], "sigmoid")

fn relu(a):
    a = lift(a)
    return mk(a.val.relu(), [a], fn(g): [g * (a.val > 0)], "relu")

fn exp(a):
    a = lift(a)
    y = a.val.exp()
    return mk(y, [a], fn(g): [g * y], "exp")

fn log(a):
    a = lift(a)
    return mk(a.val.log(), [a], fn(g): [g / a.val], "log")

# rút gọn về tensor 1 phần tử (shape [1]): sumall, mean
fn sumall(a):
    a = lift(a)
    return mk(tensor([a.val.sum()]), [a], fn(g): [ones(a.val.shape) * g.item()], "sum")

fn mean(a):
    a = lift(a)
    n = a.val.size
    return mk(tensor([a.val.mean()]), [a], fn(g): [ones(a.val.shape) * (g.item() / n)], "mean")

# ---- A1: backward kernel C gộp (không tensor trung gian) ----
# softmax: Jacobian gộp dx = y * (dy - Σ dy*y) theo trục cuối
fn softmax(a):
    a = lift(a)
    y = a.val.softmax()
    return mk(y, [a], fn(g): [y.softmax_bwd(g)], "softmax")

# gelu chính xác (erf): forward + backward gộp trong C
fn gelu(a):
    a = lift(a)
    y = a.val.gelu()
    return mk(y, [a], fn(g): [a.val.gelu_bwd(g)], "gelu")

fn rms_norm(a, eps = 1e-5):
    a = lift(a)
    y = a.val.rmsnorm(eps)
    return mk(y, [a], fn(g): [a.val.rmsnorm_bwd(g, eps)], "rms_norm")

fn layer_norm(a, eps = 1e-5):
    a = lift(a)
    y = a.val.layernorm(eps)
    return mk(y, [a], fn(g): [a.val.layernorm_bwd(g, eps)], "layer_norm")

# SwiGLU: h = silu(gate) * up; kernel C trả về [dgate, dup] (khớp thứ tự parents)
fn swiglu(gate, up):
    gate = lift(gate)
    up = lift(up)
    y = (gate.val * gate.val.sigmoid()) * up.val
    return mk(y, [gate, up], fn(g): gate.val.swiglu_bwd(g, up.val), "swiglu")

# thứ tự topo (DFS) từ nút gốc
fn topo(root):
    mut order = []
    mut seen = set()
    mut stack = [[root, false]]
    while stack.len > 0:
        top = stack.pop()
        n = top[0]
        if top[1]:
            order.append(n)
            continue
        if n.id in seen:
            continue
        seen.add(n.id)
        stack.append([n, true])
        for p in n.parents:
            if p.id not in seen:
                stack.append([p, false])
    return order

fn backward(root):
    order = topo(root)
    for n in order:
        n.grad = nil
    root.grad = ones(root.val.shape)
    mut i = order.len - 1
    while i >= 0:
        n = order[i]
        if n.backfn != nil and n.grad != nil:
            gs = n.backfn(n.grad)
            mut k = 0
            while k < n.parents.len:
                p = n.parents[k]
                p.grad = gs[k] if p.grad == nil else p.grad + gs[k]
                k += 1
        i -= 1
    return nil

# bước gradient descent cho danh sách tham số: val -= lr * grad
fn sgd(params, lr):
    for p in params:
        if p.grad != nil:
            p.val = p.val - p.grad * lr
    return nil
