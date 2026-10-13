"""ttensor.py — tensor n-chiều tham chiếu cho t.py (VM C phải khớp từng bit).

Quy ước (khớp vm/tvm.c):
  * float64, row-major, ndim 1..8, shape là tuple int >= 0.
  * Phép từng phần tử + - * / ** có broadcast kiểu numpy; tổng/trung bình cộng
    TUẦN TỰ trái->phải (không Kahan/pairwise) để kết quả trùng bit với C.
  * @: 1D·1D -> float; 2D@2D, 2D@1D, 1D@2D -> tensor; tích luỹ theo k tăng dần.
  * Ngẫu nhiên: splitmix64 (seed mặc định 42); rand -> (z >> 11) * 2^-53;
    randn: Box-Muller một giá trị/lần gọi.
"""
import math

MAXDIM = 8


class TensorError(Exception):
    pass


def _prod(sh):
    n = 1
    for s in sh:
        n *= s
    return n


class _View:
    """dữ liệu của view: phần tử thứ k (row-major) nằm ở b[ix[k]] — b là list của tensor
    chủ (dùng chung bộ nhớ); off/st: vị trí phần tử đầu + bước từng trục trong b.
    ix là range khi vùng chọn liền, list khi theo bước."""
    __slots__ = ("b", "off", "st", "ix")

    def __init__(self, b, off, st, ix):
        self.b = b
        self.off = off
        self.st = st
        self.ix = ix

    def __len__(self):
        return len(self.ix)

    def __iter__(self):
        ix = self.ix
        if isinstance(ix, range):
            return iter(self.b[ix.start:ix.stop])
        b = self.b
        return iter([b[i] for i in ix])

    def __getitem__(self, k):
        if isinstance(k, slice):
            b = self.b
            return [b[i] for i in self.ix[k]]
        return self.b[self.ix[k]]

    def __setitem__(self, k, v):
        if isinstance(k, slice):
            ix = self.ix[k]
            vs = list(v)
            if len(vs) != len(ix):
                raise ValueError("view slice assign")
            for i, x in zip(ix, vs):
                self.b[i] = x
            return
        self.b[self.ix[k]] = v

    def __repr__(self):
        return repr(list(self))


def _phys(t):
    """(list của chủ, vị trí phần tử đầu, bước từng trục) của t"""
    d = t.d
    if isinstance(d, _View):
        return d.b, d.off, d.st
    return d, 0, _strides(t.shape)


def _walk(shape, off, st):
    """vị trí (trong list của chủ) của từng phần tử, theo thứ tự row-major"""
    pos = []

    def rec(q, o):
        if q == len(shape):
            pos.append(o)
            return
        for i in range(shape[q]):
            rec(q + 1, o + i * st[q])
    rec(0, off)
    return pos


def mkview(b, off, shape, st):
    """view shape `shape` trên list b: phần tử đầu ở off, bước st. Vùng rỗng -> tensor
    rỗng mới; bước row-major -> range (liền); còn lại danh sách vị trí."""
    shape = tuple(shape)
    st = tuple(st)
    n = _prod(shape)
    if n == 0:
        return raw([], shape)
    rm = _strides(shape)
    if all(sh <= 1 or a == r for sh, a, r in zip(shape, st, rm)):
        ix = range(off, off + n)
    else:
        ix = _walk(shape, off, st)
    return raw(_View(b, off, st, ix), shape)


class Tensor:
    __slots__ = ("d", "shape")

    def __init__(self, d, shape=None):
        self.d = [float(x) for x in d]
        self.shape = tuple(shape) if shape is not None else (len(self.d),)

    def __len__(self):
        return self.shape[0] if self.shape else 0

    @property
    def ndim(self):
        return len(self.shape)

    def __repr__(self):
        return "tensor(" + repr(self.d) + ", " + repr(self.shape) + ")"

    def __eq__(self, o):
        # so từng phần tử (nan != nan như C; list == của Python coi cùng object là bằng)
        return (isinstance(o, Tensor) and o.shape == self.shape and
                all(x == y for x, y in zip(self.d, o.d)))

    def __ne__(self, o):
        return not self.__eq__(o)

    __hash__ = None

    def fmt(self, fmt_float):
        def rec(off, dim):
            n = self.shape[dim]
            if dim == len(self.shape) - 1:
                return "[" + ", ".join(fmt_float(self.d[off + i]) for i in range(n)) + "]"
            step = _prod(self.shape[dim + 1:])
            return "[" + ", ".join(rec(off + i * step, dim + 1) for i in range(n)) + "]"
        if 0 in self.shape[:-1]:
            return "[]"
        return rec(0, 0)


def raw(d, shape):
    t = Tensor.__new__(Tensor)
    t.d = d
    t.shape = tuple(shape)
    return t


# ---------------- tạo ----------------
def _num(x):
    if isinstance(x, bool) or not isinstance(x, (int, float)):
        raise TensorError("tensor() numbers")
    return float(x)


def from_value(v):
    """tensor(x): list lồng nhau (chữ nhật) / tensor / số."""
    if isinstance(v, Tensor):
        return raw(list(v.d), v.shape)
    if isinstance(v, list):
        shape = []
        cur = v
        while isinstance(cur, list):
            shape.append(len(cur))
            if not cur:
                break
            cur = cur[0]
        if len(shape) > MAXDIM:
            raise TensorError("tensor: quá %d chiều" % MAXDIM)
        out = []

        def walk(x, dim):
            if dim == len(shape):
                if isinstance(x, list):
                    raise TensorError("tensor: ragged")
                out.append(_num(x))
                return
            if not isinstance(x, list) or len(x) != shape[dim]:
                raise TensorError("tensor: ragged")
            for y in x:
                walk(y, dim + 1)
        walk(v, 0)
        return raw(out, shape)
    return raw([_num(v)], (1,))


def shape_arg(s, who):
    if isinstance(s, bool):
        raise TensorError(who + "(shape)")
    if isinstance(s, int):
        sh = (s,)
    elif isinstance(s, list) and s and all(isinstance(x, int) and not isinstance(x, bool) for x in s):
        sh = tuple(s)
    else:
        raise TensorError(who + "(shape)")
    if len(sh) > MAXDIM or any(x < 0 for x in sh):
        raise TensorError(who + "(shape)")
    return sh


def full(shape, v, who="full"):
    sh = shape_arg(shape, who)
    return raw([_num(v)] * _prod(sh), sh)


def arange(*a):
    if not 1 <= len(a) <= 3:
        raise TensorError("arange(n) / arange(a, b[, step])")
    xs = [_num(x) for x in a]
    if len(xs) == 1:
        lo, hi, st = 0.0, xs[0], 1.0
    else:
        lo, hi = xs[0], xs[1]
        st = xs[2] if len(xs) == 3 else 1.0
    if st == 0:
        raise TensorError("arange step 0")
    n = int(math.ceil((hi - lo) / st))
    if n < 0:
        n = 0
    return raw([lo + i * st for i in range(n)], (n,))


def eye(n):
    if isinstance(n, bool) or not isinstance(n, int) or n < 0:
        raise TensorError("eye(n)")
    d = [0.0] * (n * n)
    for i in range(n):
        d[i * n + i] = 1.0
    return raw(d, (n, n))


# ---------------- ngẫu nhiên (splitmix64) ----------------
_M64 = (1 << 64) - 1
_RNG = [42]


def seed(n):
    if isinstance(n, bool) or not isinstance(n, int):
        raise TensorError("seed(int)")
    _RNG[0] = n & _M64


def _next():
    _RNG[0] = (_RNG[0] + 0x9E3779B97F4A7C15) & _M64
    z = _RNG[0]
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & _M64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & _M64
    return z ^ (z >> 31)


def _unif():
    return (_next() >> 11) * (1.0 / 9007199254740992.0)


def _gauss():
    u1 = 1.0 - _unif()
    u2 = _unif()
    return math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)


def rand(shape):
    sh = shape_arg(shape, "rand")
    return raw([_unif() for _ in range(_prod(sh))], sh)


def randn(shape):
    sh = shape_arg(shape, "randn")
    return raw([_gauss() for _ in range(_prod(sh))], sh)


# ---------------- broadcast ----------------
def _bshape(a, b):
    n = max(len(a), len(b))
    a2 = (1,) * (n - len(a)) + a
    b2 = (1,) * (n - len(b)) + b
    out = []
    for x, y in zip(a2, b2):
        if x == y or y == 1:
            out.append(x)
        elif x == 1:
            out.append(y)
        else:
            raise TensorError("tensor shape mismatch")
    return tuple(out), a2, b2


def _strides(sh):
    st = [0] * len(sh)
    acc = 1
    for i in range(len(sh) - 1, -1, -1):
        st[i] = acc
        acc *= sh[i]
    return st


def _bindex(shape_out, sh_in):
    """chỉ số phẳng của từng phần tử output trong tensor input (đã đệm 1)."""
    st_in = _strides(sh_in)
    st_eff = [0 if sh_in[i] == 1 else st_in[i] for i in range(len(sh_in))]
    n = _prod(shape_out)
    idx = [0] * n
    nd = len(shape_out)
    cnt = [0] * nd
    off = 0
    for k in range(n):
        idx[k] = off
        j = nd - 1
        while j >= 0:
            cnt[j] += 1
            off += st_eff[j]
            if cnt[j] < shape_out[j]:
                break
            off -= st_eff[j] * cnt[j]
            cnt[j] = 0
            j -= 1
    return idx


def _apply(op, x, y):
    if op == "+":
        return x + y
    if op == "-":
        return x - y
    if op == "*":
        return x * y
    if op == "/":
        if y == 0.0:
            # khớp C: chia 0 theo IEEE (inf/nan), không lỗi trong tensor
            if x == 0.0 or x != x:
                return float("nan")
            return math.copysign(float("inf"), x) * math.copysign(1.0, y)
        return x / y
    if op == "**":
        return _pow(x, y)
    # so sánh từng phần tử -> 1.0 / 0.0 (nan: luôn 0.0, trừ "ne")
    if op == "<":
        return 1.0 if x < y else 0.0
    if op == ">":
        return 1.0 if x > y else 0.0
    if op == "<=":
        return 1.0 if x <= y else 0.0
    if op == ">=":
        return 1.0 if x >= y else 0.0
    if op == "eq":
        return 1.0 if x == y else 0.0
    if op == "ne":
        return 1.0 if x != y else 0.0
    raise TensorError("bad tensor op " + op)


def _pow(x, y):
    try:
        r = math.pow(x, y)
    except (OverflowError, ValueError):
        # khớp pow() của C: tràn -> ±inf, miền sai -> nan
        if x != x or y != y:
            return float("nan")
        if x < 0 and y != int(y):
            return float("nan")
        if x < 0 and y == int(y) and int(y) % 2 == 1 and abs(x) > 1:
            return float("-inf")
        if x == 0 and y < 0 and y == int(y) and int(y) % 2 == 1:
            return math.copysign(float("inf"), x)
        return float("inf")
    return r


def binop(op, a, b):
    if op == "@":
        return matmul(a, b)
    if not isinstance(a, Tensor):
        x = _scalar(a)
        return raw([_apply(op, x, v) for v in b.d], b.shape)
    if not isinstance(b, Tensor):
        y = _scalar(b)
        return raw([_apply(op, v, y) for v in a.d], a.shape)
    if a.shape == b.shape:
        return raw([_apply(op, x, y) for x, y in zip(a.d, b.d)], a.shape)
    out, a2, b2 = _bshape(a.shape, b.shape)
    ia = _bindex(out, a2)
    ib = _bindex(out, b2)
    ad, bd = a.d, b.d
    return raw([_apply(op, ad[i], bd[j]) for i, j in zip(ia, ib)], out)


def _scalar(x):
    if isinstance(x, bool) or not isinstance(x, (int, float)):
        raise TensorError("bad tensor op")
    return float(x)


def neg(a):
    return raw([-x for x in a.d], a.shape)


def matmul(a, b):
    if not isinstance(a, Tensor) or not isinstance(b, Tensor):
        raise TensorError("matmul cần 2 tensor")
    if a.ndim == 1 and b.ndim == 1:
        if a.shape != b.shape:
            raise TensorError("matmul shape")
        acc = 0.0
        for x, y in zip(a.d, b.d):
            acc += x * y
        return acc
    if a.ndim > 2 or b.ndim > 2:
        raise TensorError("matmul: tối đa 2 chiều")
    m, k = (1, a.shape[0]) if a.ndim == 1 else a.shape
    k2, n = (b.shape[0], 1) if b.ndim == 1 else b.shape
    if k != k2:
        raise TensorError("matmul shape")
    ad, bd = a.d, b.d
    out = [0.0] * (m * n)
    for i in range(m):
        row = i * k
        for j in range(n):
            acc = 0.0
            for q in range(k):
                acc += ad[row + q] * bd[q * n + j]
            out[i * n + j] = acc
    if a.ndim == 1:
        return raw(out, (n,))
    if b.ndim == 1:
        return raw(out, (m,))
    return raw(out, (m, n))


# ---------------- chỉ số / reshape ----------------
def getitem(t, i):
    n = t.shape[0]
    if i < 0 or i >= n:
        raise TensorError("index %d out of range (len %d)" % (i, n))
    b, off, st = _phys(t)
    off += i * st[0]
    if t.ndim == 1:
        return b[off]
    return mkview(b, off, t.shape[1:], st[1:])     # hàng i -> view


def _assign(t, b, shape, off, st, v, scalar_shape1):
    """gán v vào vùng (shape, off, st) trên list b của chủ t"""
    if isinstance(v, Tensor):
        if v.shape != (shape if (shape or not scalar_shape1) else (1,)):
            raise TensorError("tensor assign shape")
        vals = list(v.d)          # chép trước: nguồn có thể chồng lên đích
        for p, x in zip(_walk(shape, off, st), vals):
            b[p] = x
        return
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise TensorError("tensor assign")
    x = float(v)
    for p in _walk(shape, off, st):
        b[p] = x


def setitem(t, i, v):
    n = t.shape[0]
    if i < 0 or i >= n:
        raise TensorError("assign range")
    b, off, st = _phys(t)
    off += i * st[0]
    if t.ndim == 1:
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            raise TensorError("tensor assign")
        b[off] = float(v)
        return
    _assign(t, b, t.shape[1:], off, st[1:], v, False)


def _select(t, key):
    """vùng chọn của key: int chọn (bỏ trục), nil / [lo, hi, step] giữ trục; thiếu
    trục cuối = giữ. -> (list của chủ, shape, vị trí phần tử đầu, bước)"""
    if len(key) > t.ndim:
        raise TensorError("too many indices")
    b, off, st0 = _phys(t)
    shape, st = [], []
    for ax in range(t.ndim):
        k = key[ax] if ax < len(key) else None
        if k is None:
            shape.append(t.shape[ax])
            st.append(st0[ax])
            continue
        if isinstance(k, list):
            if len(k) != 3:
                raise TensorError("slice [lo, hi, step]")
            for x in k:
                if x is not None and (isinstance(x, bool) or not isinstance(x, int)):
                    raise TensorError("slice int")
            if k[2] == 0:
                raise TensorError("slice step 0")
            r = range(t.shape[ax])[k[0]:k[1]:k[2]]
            if len(r):
                off += r.start * st0[ax]
            shape.append(len(r))
            st.append(st0[ax] * r.step)
            continue
        if isinstance(k, bool) or not isinstance(k, int):
            raise TensorError("index int")
        if k < 0 or k >= t.shape[ax]:
            raise TensorError("index %d out of range (len %d)" % (k, t.shape[ax]))
        off += k * st0[ax]
    return b, tuple(shape), off, tuple(st)


def get_multi(t, key):
    b, shape, off, st = _select(t, key)
    if not shape:
        return b[off]
    return mkview(b, off, shape, st)      # luôn là view (liền / theo bước)


def set_multi(t, key, v):
    b, shape, off, st = _select(t, key)
    _assign(t, b, shape, off, st, v, True)


def slice0(t, lo, hi, st):
    if st == 0:
        raise TensorError("slice step 0")
    r = range(t.shape[0])[lo:hi:st]
    b, off, st0 = _phys(t)
    if len(r):
        off += r.start * st0[0]
    return mkview(b, off, (len(r),) + t.shape[1:], (st0[0] * r.step,) + tuple(st0[1:]))


def reshape(t, args):
    if len(args) == 1 and isinstance(args[0], list):
        args = args[0]
    sh = []
    for x in args:
        if isinstance(x, bool) or not isinstance(x, int):
            raise TensorError("reshape(int...)")
        sh.append(x)
    if not sh or len(sh) > MAXDIM or sh.count(-1) > 1 or any(x < -1 for x in sh):
        raise TensorError("reshape shape")
    if -1 in sh:
        rest = _prod([x for x in sh if x != -1])
        if rest == 0 or len(t.d) % rest:
            raise TensorError("reshape shape")
        sh[sh.index(-1)] = len(t.d) // rest
    if _prod(sh) != len(t.d):
        raise TensorError("reshape shape")
    d = t.d
    if not isinstance(d, _View) or isinstance(d.ix, range):     # liền -> view
        bb, off, _ = _phys(t)
        return mkview(bb, off, sh, _strides(sh))
    return raw(list(d), sh)


def transpose(t):
    """view: đảo shape và bước"""
    b, off, st = _phys(t)
    return mkview(b, off, t.shape[::-1], st[::-1])


def tolist(t):
    def rec(off, dim):
        n = t.shape[dim]
        if dim == t.ndim - 1:
            return t.d[off:off + n]
        step = _prod(t.shape[dim + 1:])
        return [rec(off + i * step, dim + 1) for i in range(n)]
    return rec(0, 0)


# ---------------- rút gọn ----------------
def _axis(t, ax):
    if isinstance(ax, bool) or not isinstance(ax, int):
        raise TensorError("axis int")
    if ax < 0:
        ax += t.ndim
    if ax < 0 or ax >= t.ndim:
        raise TensorError("axis out of range")
    return ax


def _reduce_axis(t, ax, fn):
    sh = t.shape
    outer = _prod(sh[:ax])
    n = sh[ax]
    inner = _prod(sh[ax + 1:])
    out = []
    for o in range(outer):
        for i in range(inner):
            out.append(fn([t.d[(o * n + q) * inner + i] for q in range(n)]))
    osh = sh[:ax] + sh[ax + 1:]
    if not osh:
        osh = (1,)
    return raw(out, osh)


def _sum(xs):
    acc = 0.0
    for x in xs:
        acc += x
    return acc


def _mean(xs):
    return _sum(xs) / len(xs) if xs else 0.0


def _max(xs):
    if not xs:
        raise TensorError("max() empty")
    m = xs[0]
    for x in xs[1:]:
        if x > m:
            m = x
    return m


def _min(xs):
    if not xs:
        raise TensorError("min() empty")
    m = xs[0]
    for x in xs[1:]:
        if x < m:
            m = x
    return m


def _argmax(xs):
    if not xs:
        raise TensorError("argmax() empty")
    bi = 0
    for i in range(1, len(xs)):
        if xs[i] > xs[bi]:
            bi = i
    return bi


def _argmin(xs):
    if not xs:
        raise TensorError("argmin() empty")
    bi = 0
    for i in range(1, len(xs)):
        if xs[i] < xs[bi]:
            bi = i
    return bi


REDUCERS = {"sum": _sum, "mean": _mean, "max": _max, "min": _min,
            "argmax": _argmax, "argmin": _argmin}


def reduce(t, name, args):
    fn = REDUCERS[name]
    if not args:
        r = fn(t.d)
        return r if name not in ("argmax", "argmin") else int(r)
    if len(args) != 1:
        raise TensorError(name + "([axis])")
    ax = _axis(t, args[0])
    return _reduce_axis(t, ax, lambda xs: float(fn(xs)))


# ---------------- hàm từng phần tử ----------------
def _exp(x):
    try:
        return math.exp(x)
    except OverflowError:
        return float("inf")


def _log(x):
    if x > 0:
        return math.log(x)
    if x == 0:
        return float("-inf")
    return float("nan")


def _sqrt(x):
    return math.sqrt(x) if x >= 0 else float("nan")


UNARY = {
    "exp": _exp, "log": _log, "sqrt": _sqrt, "tanh": math.tanh,
    "sin": math.sin, "cos": math.cos, "abs": abs,
    "relu": lambda x: x if x > 0 else 0.0,
    "sigmoid": lambda x: 1.0 / (1.0 + _exp(-x)),
    "floor": lambda x: float(math.floor(x)) if x == x and abs(x) != float("inf") else x,
    "ceil": lambda x: float(math.ceil(x)) if x == x and abs(x) != float("inf") else x,
}


def unary(t, name):
    f = UNARY[name]
    return raw([f(x) for x in t.d], t.shape)


def softmax(t):
    """softmax theo trục cuối."""
    n = t.shape[-1]
    out = []
    if n == 0:
        return raw([], t.shape)
    for o in range(0, len(t.d), n):
        row = t.d[o:o + n]
        m = _max(row)
        e = [_exp(x - m) for x in row]
        s = _sum(e)
        out.extend(x / s for x in e)
    return raw(out, t.shape)


def clip(t, lo, hi):
    a, b = _num(lo), _num(hi)
    return raw([a if x < a else (b if x > b else x) for x in t.d], t.shape)


METHODS = ("sum", "mean", "max", "min", "argmax", "argmin", "reshape", "transpose",
           "flatten", "tolist", "copy", "dot", "clip", "softmax", "item", "eq", "ne") + tuple(UNARY)
