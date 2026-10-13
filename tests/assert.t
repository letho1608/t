# tests/assert.t — assert stmt (chạy: python t.py tests/assert.t)
assert 1 + 1 == 2
assert 2 ** 10 == 1024, "power"
assert [1, 2].len == 2, "list len"
assert "x".upper() == "X"
assert tensor([1, 2]).sum() == 3, "tensor sum"
assert {"a": 1}.get("zz", -1) == -1
assert "hello"[1:3] == "el", "slice"
fn sq(x):
    return x * x
assert sq(9) == 81
try:
    assert 1 == 2, "boom-msg"
except boom:
    print("caught:" + err)
# return/break xuyên qua try (không bị except nuốt)
fn early():
    try:
        return 1
    except:
        return 2
assert early() == 1, "return-thru-try"
mut hit = 0
for i in range(5):
    try:
        if i == 2:
            break
    except:
        hit = -99
    hit = hit + 1
assert hit == 2, "break-thru-try"
print("ASSERT OK")
