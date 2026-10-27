print("== power ==")
print(2 ** 10)
print(2 ** 3 ** 2)
print(9 ** 0.5)
print(tensor([1, 2, 3]) ** 2)

print("== dict ==")
d = {"a": 1, "b": 2}
print(d)
print(d["a"])
d["c"] = 30
print(d.len)
print(d.keys())
print(d.get("zz", -1))
for k in d:
    print(k + "=" + str(d[k]))

print("== slice ==")
s = "hello"
print(s[1:3])
print(s[2:])
print(s[:3])
print(s[::2])
print([1, 2, 3, 4][1:3])
print(tensor([1, 2, 3, 4])[1:3])

print("== file io ==")
f = open("tests/_tmp_out.txt", "w")
write(f, "hello T\nline2")
close(f)
f = open("tests/_tmp_out.txt", "r")
print(read(f))
close(f)
load("__sys").remove("tests/_tmp_out.txt")

import ai
print(ai.softmax(tensor([1, 2, 3])).sum())
