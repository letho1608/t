# std/math.t — thư viện toán học chuẩn của T

let PI = 3.141592653589793
let TAU = 6.283185307179586
let E = 2.718281828459045
let INF = 1e308 * 10.0
let NAN = INF - INF

let sin = sin
let cos = cos
let tanh = tanh
let sqrt = sqrt
let exp = exp
let log = log

fn tan(x):
    return sin(x) / cos(x)

fn pow(x, y):
    return x ** y

fn floor(x):
    i = int(x)
    return i - 1 if x < 0 and x != i else i

fn ceil(x):
    i = int(x)
    return i + 1 if x > 0 and x != i else i

fn deg(rad_val):
    return rad_val * (180.0 / PI)

fn rad(deg_val):
    return deg_val * (PI / 180.0)

fn sign(x):
    if x > 0:
        return 1
    if x < 0:
        return -1
    return 0

fn trunc(x):
    return int(x)

fn clamp(x, lo, hi):
    if x < lo:
        return lo
    if x > hi:
        return hi
    return x

fn lerp(a, b, t):
    return a + (b - a) * t

fn log2(x):
    return log(x) / log(2.0)

fn log10(x):
    return log(x) / log(10.0)

fn hypot(x, y):
    return sqrt(x * x + y * y)

fn sinh(x):
    return 0.5 * (exp(x) - exp(0.0 - x))

fn cosh(x):
    return 0.5 * (exp(x) + exp(0.0 - x))

fn atan(x):
    if x < 0.0:
        return 0.0 - atan(0.0 - x)
    if x > 1.0:
        return 0.5 * PI - atan(1.0 / x)
    # Thu gọn miền 3 lần: u = x / (1 + sqrt(1 + x^2)) -> |u| <= tan(pi/32) < 0.0985
    mut u = x / (1.0 + sqrt(1.0 + x * x))
    u = u / (1.0 + sqrt(1.0 + u * u))
    u = u / (1.0 + sqrt(1.0 + u * u))
    u2 = u * u
    p = u * (1.0 - u2 * (1.0 / 3.0 - u2 * (1.0 / 5.0 - u2 * (1.0 / 7.0 - u2 * (1.0 / 9.0 - u2 / 11.0)))))
    return 8.0 * p

fn atan2(y, x):
    if x > 0.0:
        return atan(y / x)
    if x < 0.0:
        if y >= 0.0:
            return atan(y / x) + PI
        return atan(y / x) - PI
    if y > 0.0:
        return 0.5 * PI
    if y < 0.0:
        return -0.5 * PI
    return 0.0

fn asin(x):
    if x <= -1.0:
        return -0.5 * PI
    if x >= 1.0:
        return 0.5 * PI
    return atan2(x, sqrt(1.0 - x * x))

fn acos(x):
    return 0.5 * PI - asin(x)

fn isnan(x):
    return x != x

fn isinf(x):
    return x == INF or x == (0.0 - INF)

fn isfinite(x):
    return (not isnan(x)) and (not isinf(x))

fn isclose(a, b, rel_tol = 1e-9, abs_tol = 0.0):
    diff = abs(a - b)
    bound = max(rel_tol * max(abs(a), abs(b)), abs_tol)
    return diff <= bound

fn gcd(a, b):
    mut x = abs(int(a))
    mut y = abs(int(b))
    while y != 0:
        t = x % y
        x = y
        y = t
    return x

fn lcm(a, b):
    if a == 0 or b == 0:
        return 0
    return abs(int(int(a) / gcd(a, b)) * int(b))

fn factorial(n):
    if n < 0:
        raise "factorial: n < 0"
    mut r = 1
    mut i = 2
    while i <= n:
        r = r * i
        i = i + 1
    return r

fn comb(n, k):
    if k < 0 or k > n:
        return 0
    mut kk = k
    if kk > n - kk:
        kk = n - kk
    mut r = 1
    mut i = 1
    while i <= kk:
        r = int((r * (n - kk + i)) / i)
        i = i + 1
    return r

fn perm(n, k = nil):
    kk = n if k == nil else k
    if kk < 0 or kk > n:
        return 0
    mut r = 1
    mut i = 0
    while i < kk:
        r = r * (n - i)
        i = i + 1
    return r

fn isprime(n):
    if n < 2:
        return false
    if n == 2 or n == 3:
        return true
    if n % 2 == 0 or n % 3 == 0:
        return false
    mut i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return false
        i = i + 6
    return true
