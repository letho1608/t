"""test_tpkg.py — kiểm chứng tpkg hoàn toàn offline (registry thư mục cục bộ).
Chạy: python tools/test_tpkg.py   (in TPKG-ALL-GREEN khi xanh)
Kịch bản: tạo 2 phiên bản gói `mathx` -> publish vào kho thư mục -> app phụ
thuộc `mathx@^1.0` (phải chọn 1.2.0, không lấy 2.0.0) -> install -> run ->
xoá kho + tpkgs -> `install --offline` dựng lại từ cache (kiểm sha) ->
`release minor` tăng semver + CHANGELOG.md.
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TPM = [sys.executable, os.path.join(ROOT, "tools", "tpkg.py")]
FAILS = []


def run(args, cwd, env):
    r = subprocess.run(TPM + args, cwd=cwd, env=env, capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    return r


def check(name, cond, extra=""):
    print(("PASS " if cond else "FAIL ") + name)
    if not cond:
        FAILS.append(name)
        if extra:
            print("     ", extra[-600:])


def make_pkg(base, ver, body):
    d = os.path.join(base, "src-" + ver)
    os.makedirs(d)
    with open(os.path.join(d, "t.json"), "w", encoding="utf-8") as f:
        json.dump({"name": "mathx", "version": ver, "main": "mathx.t", "deps": {}}, f)
    with open(os.path.join(d, "mathx.t"), "w", encoding="utf-8") as f:
        f.write(body)
    return d


def main():
    tmp = tempfile.mkdtemp(prefix="tpkgt")
    reg = os.path.join(tmp, "reg")
    env = dict(os.environ, TPKG_REG=reg, TPKG_CACHE=os.path.join(tmp, "cache"),
               TPKG_DATE="2026-01-02", PYTHONIOENCODING="utf-8")
    env.pop("TPKG_OFFLINE", None)
    for ver, k in (("1.0.0", 1), ("1.2.0", 2), ("2.0.0", 3)):
        d = make_pkg(tmp, ver, "fn twice(x):\n    return x * 2\nVER = %d\n" % k)
        r = run(["publish"], d, env)
        check("publish mathx " + ver, r.returncode == 0, r.stdout + r.stderr)
    idx = json.load(open(os.path.join(reg, "index.json"), encoding="utf-8"))
    e = idx["pkgs"].get("mathx", {})
    check("index giữ đủ 3 phiên bản", sorted(e.get("versions", {})) == ["1.0.0", "1.2.0", "2.0.0"])
    check("index version mới nhất = 2.0.0", e.get("version") == "2.0.0")
    r = run(["search", "math"], tmp, env)
    check("search", r.returncode == 0 and "mathx 2.0.0" in r.stdout, r.stdout + r.stderr)

    app = os.path.join(tmp, "app")
    r = run(["init", "app"], tmp, env)
    check("init app", r.returncode == 0, r.stdout + r.stderr)
    man = json.load(open(os.path.join(app, "t.json"), encoding="utf-8"))
    man["deps"] = {"mathx": "mathx@^1.0"}
    json.dump(man, open(os.path.join(app, "t.json"), "w", encoding="utf-8"))
    with open(os.path.join(app, "main.t"), "w", encoding="utf-8") as f:
        f.write("import mathx\nprint(mathx.twice(21), mathx.VER)\n")
    r = run(["install"], app, env)
    check("install mathx@^1.0", r.returncode == 0, r.stdout + r.stderr)
    lock = json.load(open(os.path.join(app, "tpkgs", "lock.json"), encoding="utf-8"))
    check("chọn 1.2.0 (cao nhất thỏa ^1.0)", lock.get("mathx", {}).get("version") == "1.2.0",
          json.dumps(lock))
    r = run(["run", "main.t"], app, env)
    check("run import dep", r.returncode == 0 and r.stdout.strip() == "42 2", r.stdout + r.stderr)

    # offline: kho biến mất, tpkgs bị xoá -> dựng lại từ lock + cache
    shutil.rmtree(reg)
    shutil.rmtree(os.path.join(app, "tpkgs", "mathx"))
    r = run(["install", "--offline"], app, env)
    check("install --offline từ cache", r.returncode == 0, r.stdout + r.stderr)
    r = run(["run", "main.t"], app, env)
    check("run sau offline install", r.returncode == 0 and r.stdout.strip() == "42 2",
          r.stdout + r.stderr)

    # cache bị sửa -> sha lệch phải báo lỗi
    cdir = os.path.join(tmp, "cache")
    sha = lock["mathx"]["sha16"]
    with open(os.path.join(cdir, "sha-" + sha), "ab") as f:
        f.write(b"x")
    r = run(["install", "--offline"], app, env)
    check("offline phát hiện cache hỏng (sha)", r.returncode != 0 and "sha" in (r.stdout + r.stderr))

    # release: semver + changelog
    r = run(["release", "minor", "thêm", "mathx"], app, env)
    man = json.load(open(os.path.join(app, "t.json"), encoding="utf-8"))
    cl = open(os.path.join(app, "CHANGELOG.md"), encoding="utf-8").read()
    check("release minor 0.1.0 -> 0.2.0", r.returncode == 0 and man["version"] == "0.2.0",
          r.stdout + r.stderr)
    check("CHANGELOG có mục 0.2.0", "## 0.2.0 — 2026-01-02" in cl and "- thêm mathx" in cl, cl)
    r = run(["release", "patch", "sửa lỗi"], app, env)
    cl = open(os.path.join(app, "CHANGELOG.md"), encoding="utf-8").read()
    check("CHANGELOG mới nhất ở đầu", cl.index("## 0.2.1") < cl.index("## 0.2.0"), cl)

    shutil.rmtree(tmp, ignore_errors=True)
    print("TPKG-ALL-GREEN" if not FAILS else "TPKG-FAIL %d" % len(FAILS))
    return 1 if FAILS else 0


if __name__ == "__main__":
    raise SystemExit(main())
