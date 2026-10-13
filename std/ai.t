# std/ai.t — thư viện AI (dùng qua: import ai)
fn zeros(n):
    mut r = []
    for i in range(n):
        r.append(0)
    return tensor(r)

fn ones(n):
    mut r = []
    for i in range(n):
        r.append(1)
    return tensor(r)

fn dot(a, b):
    return a @ b

fn relu(t):
    mut r = []
    for i in range(t.len):
        v = t[i]
        if v > 0:
            r.append(v)
        else:
            r.append(0)
    return tensor(r)

fn mse(a, b):
    mut s = 0
    for i in range(a.len):
        d = a[i] - b[i]
        s = s + d * d
    return s / a.len

fn grad(f, x):
    h = 0.00001
    return (f(x + h) - f(x - h)) / (2 * h)

# linear: w @ x + b  (w, x tensor cùng len)
fn linear(w, x, b):
    return (w @ x) + b

# softmax 1-D, trả tensor
fn softmax(t):
    mut m = t[0]
    for i in range(1, t.len):
        if t[i] > m:
            m = t[i]
    mut s = 0
    mut e = []
    for i in range(t.len):
        v = 2.718281828 ** ((t[i] - m))
        e.append(v)
        s = s + v
    mut r = []
    for v in e:
        r.append(v / s)
    return tensor(r)
