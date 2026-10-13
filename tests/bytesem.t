# tests/bytesem.t — string trong T là BYTE-STRING: 1 char = 1 byte ở mọi runtime
# (t.py, trun.exe, và compiler thuần T). len/ord/chr/index/slice/so sánh đều
# theo byte; upper/lower ASCII-only; split(sep) giữ field rỗng.
s = "Tiệng Việt"
print(s.len)
print(ord("a"), ord("A"), ord("0"))
print(chr(65), chr(233))
print(s[0:5], s[-6:])
print(s.upper())
print(s.lower())
print("a  b".split(" ").len)
print("abc".split("").len)
print("  x \n".strip())
print("-".join(["a", "b", "c"]))
print("abc" < "abd", "Z" < "a")
print(s == "Tiệng Việt")
print("héllo".upper(), "héllo".lower())
print("héllo".upper(), "héllo".lower())
assert s.len == 14
print("héllo".upper(), "héllo".lower())
# chuỗi nhiều byte: literal UTF-8 trong source, mọi runtime đều đọc theo byte
z = "你好"
print(z, z.len, z[0:3])
assert z.len == 6
print("BYTESEM OK")
