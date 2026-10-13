# tests/tryflow.t — return/break/continue bên trong try phải gỡ handler;
# lỗi xuyên qua callback của native (map/sort) vẫn về đúng handler.
fn f(v):
    try:
        return v * 2
    except:
        return -1

fn g():
    mut out = []
    for i in range(6):
        try:
            if i == 1:
                continue
            if i == 4:
                break
            out.append(i)
        except:
            out.append("E")
    return out

mut w = 0
while true:
    try:
        w += 1
        if w == 3:
            break
    except:
        print("never")
print(f(2), g(), w)

# sau các handler đã gỡ, lỗi mới phải tới đúng handler
try:
    raise "late"
except "late":
    print("caught late", err)

fn h():
    raise "deep"
fn mid():
    try:
        return h()
    except "nope":
        return 0
try:
    mid()
except "deep":
    print("caught deep", err)

# lỗi trong callback của map, bắt ở ngoài; trạng thái VM vẫn lành
fn bad(x):
    if x == 2:
        raise "cb fail"
    return x
try:
    print(map(bad, [1, 2, 3]))
except "cb":
    print("caught cb", err)
print(map(bad, [1, 3]))
fn key(x):
    return 0 - x
print(sort([3, 1, 2], key))

# lồng try + return xuyên nhiều tầng
fn deep(n):
    try:
        try:
            if n == 0:
                return "zero"
            raise "inner"
        except "inner":
            return "handled " + str(n)
    except:
        return "outer"
print(deep(0), deep(5))
try:
    raise "after deep"
except "after":
    print("ok", err)
print("TRY OK")
