# module nạp lúc chạy bằng load("tests/loadmod_m") (xem tests/loadmod.t)
VERSION = "1.0"
mut COUNT = [0]
mut LEVEL = 1
fn set_level(v):
    LEVEL = v
    return LEVEL
fn get_level():
    return LEVEL
fn sq(x):
    return x * x
fn bump():
    COUNT[0] += 1
    return COUNT[0]
fn apply(f, xs):
    return [f(x) for x in xs]
struct Pt:
    x, y
    fn norm2(self):
        return self.x * self.x + self.y * self.y
fn hot(n):
    mut s = 0
    for i in range(n):
        s += sq(i) % 7
    return s
fn boom():
    return 1 / 0
print("loadmod_m: nạp")
