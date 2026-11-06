# tests/fusedbwd.t — A1: fused backward kernels (softmax/gelu/rms_norm/layer_norm/swiglu)
# Mỗi phép: (a) gradient phân tích (kernel C gộp) phải khớp sai phân hữu hạn;
#           (b) softmax phải khớp công thức compose trong T (không qua kernel).
import autograd as ag

# sai phân hữu hạn trung tâm (giống tests/agtest.t)
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

fn diff(a, b):
    return (a - b).abs().max()

# --- 1) softmax ---
X1 = tensor([[1.0, 2.0, 0.5], [-1.0, 0.0, 3.0], [0.2, -0.4, 1.1]])
W1 = tensor([[0.3, 0.5, 0.2], [1.0, 0.0, 0.0], [0.1, 0.1, 0.8]])
fn f_s(x):
    return (x.softmax() * W1).sum()
p1 = ag.param(X1)
L1 = ag.sumall(ag.mul(ag.softmax(p1), ag.const(W1)))
ag.backward(L1)
d = diff(p1.grad, numgrad(f_s, X1, 0.000001))
assert d < 0.000001, "softmax gradient sai: " + str(d)
# fused == compose trong T: dx = y * (dy - (dy*y).sum(1).reshape(-1,1))
y1 = X1.softmax()
s1 = (W1 * y1).sum(1).reshape([3, 1])
dx_naive = y1 * (W1 - s1)
d = diff(p1.grad, dx_naive)
assert d < 0.000000001, "softmax fused != compose: " + str(d)

# --- 2) gelu (erf chính xác) ---
X2 = tensor([[0.5, -1.2, 2.0], [1.5, 0.0, -0.3]])
W2 = tensor([[1.0, 0.5, -0.5], [0.2, 1.0, 0.1]])
fn f_g(x):
    return (x.gelu() * W2).sum()
p2 = ag.param(X2)
L2 = ag.sumall(ag.mul(ag.gelu(p2), ag.const(W2)))
ag.backward(L2)
d = diff(p2.grad, numgrad(f_g, X2, 0.000001))
assert d < 0.000001, "gelu gradient sai: " + str(d)

# --- 3) rms_norm ---
X3 = tensor([[1.0, 2.0, 3.0], [-1.0, 0.5, 2.0], [0.1, -0.2, 0.4]])
W3 = tensor([[0.3, -0.1, 0.8], [1.0, 0.0, 0.5], [-0.5, 0.2, 0.7]])
fn f_r(x):
    return (x.rmsnorm(0.00001) * W3).sum()
p3 = ag.param(X3)
L3 = ag.sumall(ag.mul(ag.rms_norm(p3, 0.00001), ag.const(W3)))
ag.backward(L3)
d = diff(p3.grad, numgrad(f_r, X3, 0.000001))
assert d < 0.000001, "rms_norm gradient sai: " + str(d)

# --- 4) layer_norm ---
fn f_l(x):
    return (x.layernorm(0.00001) * W3).sum()
p4 = ag.param(X3)
L4 = ag.sumall(ag.mul(ag.layer_norm(p4, 0.00001), ag.const(W3)))
ag.backward(L4)
d = diff(p4.grad, numgrad(f_l, X3, 0.000001))
assert d < 0.000001, "layer_norm gradient sai: " + str(d)

# --- 5) swiglu (2 tham số, kernel trả [dgate, dup]) ---
G = tensor([[0.5, -1.0], [2.0, 0.1]])
U = tensor([[1.5, 0.5], [-0.5, 2.0]])
W4 = tensor([[1.0, -1.0], [0.5, 0.5]])
fn f_gg(gate):
    return ((gate * gate.sigmoid()) * U * W4).sum()
fn f_uu(up):
    return ((G * G.sigmoid()) * up * W4).sum()
pg = ag.param(G)
pu = ag.param(U)
L5 = ag.sumall(ag.mul(ag.swiglu(pg, pu), ag.const(W4)))
ag.backward(L5)
d = diff(pg.grad, numgrad(f_gg, G, 0.000001))
assert d < 0.000001, "swiglu dgate sai: " + str(d)
d = diff(pu.grad, numgrad(f_uu, U, 0.000001))
assert d < 0.000001, "swiglu dup sai: " + str(d)

print("FUSED-BWD-OK")
