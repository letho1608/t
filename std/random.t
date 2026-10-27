# std/random.t — bộ sinh số giả ngẫu nhiên (PRNG) 64-bit có seed cho T
# Thuật toán: SplitMix64 / XorShift64* dùng toán tử bit 64-bit (&, |, ^, <<, >>)
import math
import time

mut _state = [0x123456789ABCDEF0]

fn seed(s = nil):
    mut v = 0
    if s == nil:
        v = int(time.time() * 1000000.0) ^ int(time.clock() * 1000000000.0)
    elif type(s) == "int":
        v = s
    elif type(s) == "float":
        v = f64bits(s)
    else:
        t = str(s)
        v = 0xCBF29CE484222325
        for i in range(t.len):
            v = (v ^ ord(t[i:i + 1])) * 0x100000001B3
    if v == 0:
        v = 0x9E3779B97F4A7C15
    _state[0] = v
    # Khởi động 4 vòng để khuếch tán bit hạt giống
    next_int()
    next_int()
    next_int()
    next_int()
    return nil

fn next_int():
    mut x = _state[0]
    if x == 0:
        x = 0x9E3779B97F4A7C15
    x = x ^ ((x >> 12) & 0x000FFFFFFFFFFFFF)
    x = x ^ (x << 25)
    x = x ^ ((x >> 27) & 0x0000001FFFFFFFFF)
    _state[0] = x
    r = x * 0x2545F4914F6CDD1D
    return r & 0x7FFFFFFFFFFFFFFF

fn random():
    # 53-bit mantissa chuẩn IEEE 754 trong [0.0, 1.0)
    u53 = (next_int() >> 10) & 0x001FFFFFFFFFFFFF
    return float(u53) / 9007199254740992.0

fn uniform(a, b):
    return a + (b - a) * random()

fn randint(lo, hi):
    if lo > hi:
        raise "randint: lo > hi"
    span = hi - lo + 1
    if span <= 1:
        return lo
    return lo + (next_int() % span)

fn randrange(lo, hi = nil, step = 1):
    mut start = 0
    mut stop = lo
    if hi != nil:
        start = lo
        stop = hi
    if step == 0:
        raise "randrange: step == 0"
    mut n = 0
    if step > 0:
        if start >= stop:
            raise "randrange: empty range"
        n = int((stop - start + step - 1) / step)
    else:
        if start <= stop:
            raise "randrange: empty range"
        n = int((start - stop - step - 1) / (0 - step))
    return start + randint(0, n - 1) * step

fn choice(seq):
    n = len(seq)
    if n == 0:
        raise "choice: empty sequence"
    return seq[randint(0, n - 1)]

fn shuffle(lst):
    mut i = lst.len - 1
    while i > 0:
        j = randint(0, i)
        tmp = lst[i]
        lst[i] = lst[j]
        lst[j] = tmp
        i = i - 1
    return lst

fn sample(seq, k):
    n = len(seq)
    if k < 0 or k > n:
        raise "sample: k out of range"
    mut pool = []
    for i in range(n):
        pool.append(seq[i])
    mut out = []
    for i in range(k):
        j = randint(i, n - 1)
        tmp = pool[i]
        pool[i] = pool[j]
        pool[j] = tmp
        out.append(pool[i])
    return out

fn gauss(mu = 0.0, sigma = 1.0):
    mut u1 = random()
    while u1 <= 1e-15:
        u1 = random()
    u2 = random()
    z0 = math.sqrt(-2.0 * math.log(u1)) * math.cos(math.TAU * u2)
    return mu + sigma * z0

fn normal(mu = 0.0, sigma = 1.0):
    return gauss(mu, sigma)
