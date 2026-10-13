# tests/gcstress.t — chu trình rác đủ loại (list, dict, record, closure qua cell,
# bound method) tạo liên tục; chạy với TVM_GCTHRESH nhỏ để GC quét rất dày.
struct Node:
    val, next
    fn total(self):
        mut s = 0
        mut cur = self
        mut k = 0
        while cur != nil and k < 5:
            s += cur.val
            cur = cur.next
            k += 1
        return s

fn make_cycles(i):
    mut a = [i]
    b = [a, "x"]
    a.append(b)
    mut d = {"k": i}
    d["self"] = d
    n1 = Node(i, nil)
    n2 = Node(i + 1, n1)
    n1.next = n2
    fn counter():
        mut c = 0
        fn inc():
            c += 1
            return inc
        return inc
    f = counter()
    f()
    m = n1.total
    return a[0] + d["k"] + m()

mut acc = 0
for i in range(3000):
    acc += make_cycles(i)
print(acc)

# dữ liệu sống lâu xen kẽ rác: phải giữ nguyên
mut keep = []
for i in range(2000):
    tmp = [i, [i]]
    tmp[1].append(tmp)
    if i % 100 == 0:
        keep.append(Node(i, nil))
mut ks = 0
for n in keep:
    ks += n.val
print(len(keep), ks)

# closure trong vòng lặp giữ biến lặp (cell) + map/sort gọi lại T
fn mk_adders(n):
    mut fs = []
    for k in range(n):
        fn add(x):
            return x + k
        fs.append(add)
    return fs
mut tot = 0
for r in range(200):
    for f in mk_adders(5):
        tot += f(r)
print(tot)
fn keyf(x):
    junk = [x, [x]]
    junk[1].append(junk)
    return 0 - x
print(sort([5, 3, 9, 1], keyf), map(keyf, [1, 2, 3]))
print("GC OK")
