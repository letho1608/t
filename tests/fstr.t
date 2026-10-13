# tests/fstr.t — f-string (chạy: python t.py tests/fstr.t | trun)
name = "T"
n = 7
assert f"hello {name}, {n + 1}!" == "hello T, 8!"
assert f"{{x}}" == "{x}"
assert f"{n * 2}" == "14"
assert f"upper {"abc".upper()}" == "upper ABC"
d = {"k": 42}
assert f"v={d['k']}" == "v=42"
t = tensor([1, 2])
assert f"t={t}" == "t=[1, 2]"
assert f"{t.sum()}" == "3"
assert f"" == ""
assert f"{true} {nil}" == "true nil"
fn greet(who):
    return f"hi {who}!"
assert greet("ai") == "hi ai!"
assert f"{d["k"]}!" == "42!"
assert f"{f"{1 + 2}"}" == "3"
print("FSTR OK")
