# test_tpkg.t — Kiểm thử toàn diện tpkg (init/publish/search/install/lock/offline/run/test/release)
import os
import path
import json

trun = os.exepath()
root = os.getcwd()
tpkg_script = path.join(root, "tools", "tpkg.t")
tmp = os.mktemp("tpkg_test")
reg = path.join(tmp, "registry")
cache = path.join(tmp, "cache")
os.mkdir(reg)
os.mkdir(cache)
os.setenv("TPKG_REG", reg)
os.setenv("TPKG_CACHE", cache)

fn run_tpkg(cwd, args_list, expect_ok):
    mut cmd = [trun, tpkg_script]
    for a in args_list:
        cmd.append(a)
    r = os.run(cmd, "", cwd)
    if expect_ok and r["rc"] != 0:
        print("FAIL tpkg", args_list, "rc=" + str(r["rc"]), r["out"], r["err"])
        os.exit(1)
    if not expect_ok and r["rc"] == 0:
        print("FAIL expected error for", args_list)
        os.exit(1)
    return r

# 1. Phát hành 3 phiên bản mathutil: 1.0.0, 1.2.0, 2.0.0
for item in [["1.0.0", "a + b"], ["1.2.0", "a + b + 100"], ["2.0.0", "a * b"]]:
    ver = item[0]
    expr = item[1]
    d = path.join(tmp, "mathutil-" + ver)
    os.mkdir(d)
    os.write_file(path.join(d, "t.json"), json.dumps({
        "name": "mathutil",
        "version": ver,
        "description": "Tien ich toan hoc co ban",
        "main": "mathutil.t",
        "deps": {}
    }, 2))
    os.write_file(path.join(d, "mathutil.t"), "fn calc(a, b):\n    return " + expr + "\n")
    run_tpkg(d, ["publish"], true)

# 2. Phát hành gói strfmt@0.3.1 phụ thuộc bắc cầu mathutil@^1.0.0
d_sf = path.join(tmp, "strfmt")
os.mkdir(d_sf)
os.write_file(path.join(d_sf, "t.json"), json.dumps({
    "name": "strfmt",
    "version": "0.3.1",
    "description": "Format ket qua",
    "main": "strfmt.t",
    "deps": {"mathutil": "^1.0.0"}
}, 2))
os.write_file(path.join(d_sf, "strfmt.t"), "import mathutil\nfn show(x, y):\n    return \"KQ=\" + str(mathutil.calc(x, y))\n")
run_tpkg(d_sf, ["publish"], true)

# 3. Kiểm tra search
rs = run_tpkg(tmp, ["search", "toan"], true)
assert "mathutil@2.0.0" in rs["out"], "search failed: " + rs["out"]
print("OK publish + search")

# 4. Tạo ứng dụng myapp, install strfmt@^0.3.0 -> phải kéo mathutil@1.2.0
app = path.join(tmp, "myapp")
os.mkdir(app)
run_tpkg(app, ["init", "myapp"], true)
run_tpkg(app, ["install", "strfmt@^0.3.0"], true)

lock = json.loads(os.read_file(path.join(app, "lock.json")))
assert lock["packages"]["strfmt"]["version"] == "0.3.1"
assert lock["packages"]["mathutil"]["version"] == "1.2.0"
print("OK semver resolve + lock.json")

# 5. Chạy main.t và test
os.write_file(path.join(app, "main.t"), "import strfmt\nimport mathutil\nprint(strfmt.show(3, 4))\nprint(mathutil.calc(5, 6))\n")
rr = run_tpkg(app, ["run"], true)
assert rr["out"].strip().split("\n") == ["KQ=107", "111"], "run output mismatch: " + rr["out"]

app_tests = path.join(app, "tests")
os.mkdir(app_tests)
os.write_file(path.join(app_tests, "t1.t"), "import mathutil\nassert mathutil.calc(1, 2) == 103\nprint(\"T1-OK\")\n")
rt = run_tpkg(app, ["test"], true)
assert "T1-OK" in rt["out"]
print("OK tpkg run + test")

# 6. Kiểm tra install --offline từ cache khi đã xóa registry
os.remove(path.join(reg, "index.json"))
os.remove(path.join(app, "tpkgs", "mathutil", "mathutil.t"))
run_tpkg(app, ["install", "--offline"], true)
rr2 = run_tpkg(app, ["run"], true)
assert rr2["out"].strip().split("\n") == ["KQ=107", "111"]
print("OK install --offline")

# 7. Kiểm tra phát hiện hỏng SHA-256 trong cache
os.write_file(path.join(cache, "mathutil-1.2.0.tpkg"), "corrupted-bytes")
r_bad = run_tpkg(app, ["install", "--offline"], false)
assert "sha256" in r_bad["err"].lower()
print("OK sha256 corruption check")

# 8. Kiểm tra release minor + patch + CHANGELOG.md
reg2 = path.join(tmp, "reg2")
os.mkdir(reg2)
os.setenv("TPKG_REG", reg2)
rel_pkg = path.join(tmp, "relpkg")
os.mkdir(rel_pkg)
run_tpkg(rel_pkg, ["init", "relpkg"], true)
run_tpkg(rel_pkg, ["release", "minor", "Them ham moi"], true)
mf_rel = json.loads(os.read_file(path.join(rel_pkg, "t.json")))
assert mf_rel["version"] == "0.2.0"
chg = os.read_file(path.join(rel_pkg, "CHANGELOG.md"))
assert "## v0.2.0" in chg and "Them ham moi" in chg
assert os.exists(path.join(reg2, "relpkg-0.2.0.tpkg"))

run_tpkg(rel_pkg, ["release", "patch", "Sua loi nho"], true)
mf_rel2 = json.loads(os.read_file(path.join(rel_pkg, "t.json")))
assert mf_rel2["version"] == "0.2.1"
print("OK release minor/patch + CHANGELOG.md")

os.setenv("TPKG_REG", "")
os.setenv("TPKG_CACHE", "")
os.rmtree(tmp)
print("TPKG-ALL-GREEN")
