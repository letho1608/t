# tests/builtins.t — builtins tập trên: abs/round/min/max/sum/any/all/
# sort/reversed/enumerate/zip/map/filter (khớp cả 3 runtime)
xs = [5, 3, 9, 1, 7]
ss = ["pear", "apple", "fig"]

print(abs(-7), abs(7), abs(-2.5), abs(0))
print(round(2.4), round(2.5), round(-2.5), round(3.14159))
print(min(xs), max(xs), min([4, 2, 8]), max(ss))
print(min(3, 9), max(3, 9))
print(sum(xs), sum([1.5, 2.5]), sum([], 10), sum(xs, 100))
print(any([0, 1, 0]), any([0, 0]), all([1, 2]), all([1, 0]))

mut a = [5, 3, 9, 1, 7]
print(sort(a), a)
mut b = ["pear", "apple", "fig"]
print(sort(b))
mut w = [[2, 1], [1, 9], [1, 2]]
fn bykey(p):
    return p[0] * 10 + p[1]
mut c = sort(w, bykey)
print(c)

print(reversed([1, 2, 3]), reversed("abc"))
print(enumerate(["a", "b"]), enumerate([9, 8], 5))
print(zip([1, 2, 3], ["a", "b"]), zip([1, 2], [3, 4], [5, 6]))

fn dbl(x):
    return x * 2
fn big(x):
    return x > 4
print(map(dbl, [1, 2, 3]), map(dbl, []))
print(filter(big, [1, 5, 3, 9, 2]), filter(big, []))

# sort chuỗi theo byte, không phụ thuộc locale
print(sort(["b", "A", "a", "B"]))
# min/max/sum trên tensor
t = tensor([3.0, 1.0, 2.0])
print(min(t), max(t), sum(t))
print("BUILTINS OK")
