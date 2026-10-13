# tests/mem.t — builtin mem() (chạy: python t.py tests/mem.t | trun)
m = mem()
assert type(m) == "dict"
assert type(m.get("objs", -1)) == "int"
assert type(m.get("gc", -1)) == "int"
# chu trình: refcount không thu được, GC mark-sweep thu được (trun)
mut cyc = []
cyc.append(cyc)
cyc = nil
mut i = 0
mut keep = []
while i < 3000:
    mut c = [i]
    c.append(c)
    keep.append(c[0])
    i = i + 1
m2 = mem()
assert type(m2.get("objs", -1)) == "int"
print("MEM OK")
