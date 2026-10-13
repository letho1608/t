"""run_all.py — chạy TOÀN BỘ kiểm thử của T (dùng cho CI và máy dev).
Chạy:  python tools/run_all.py [--build]
  --build : build trun trước (Windows: vm/build.bat qua MSVC; POSIX: vm/build.sh)
In ALL-GREEN và exit 0 khi mọi bộ đều xanh.
"""
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRUN = os.path.join(ROOT, "trun.exe" if os.name == "nt" else "trun")

# (tên, lệnh, chuỗi phải có trong output)
SUITES = [
    ("ops", [sys.executable, "tools/test_ops.py"], "OPS-OK"),
    ("tcomp-t", [sys.executable, "tools/test_tcomp_t.py"], "TCOMP-T-OK"),
    ("dbg", [sys.executable, "tools/test_dbg.py"], "DBG-ALL-GREEN"),
    ("lsp", [sys.executable, "tools/test_lsp.py"], "LSP-ALL-GREEN"),
    ("tpkg", [sys.executable, "tools/test_tpkg.py"], "TPKG-ALL-GREEN"),
    ("jit", [sys.executable, "tools/test_jit.py"], "JIT-ALL-GREEN"),
    ("tensor", [sys.executable, "tools/test_tensor.py"], "TENSOR-ALL-GREEN"),
]


def build():
    if os.name == "nt":
        r = subprocess.run(["cmd", "/c", os.path.join(ROOT, "vm", "build.bat")], cwd=ROOT)
    else:
        r = subprocess.run(["sh", "vm/build.sh"], cwd=ROOT)
    if r.returncode != 0:
        print("BUILD FAIL")
        raise SystemExit(1)


def main():
    for st in (sys.stdout, sys.stderr):
        try:
            st.reconfigure(encoding="utf-8", errors="replace")
        except Exception:  # noqa: BLE001
            pass
    if "--build" in sys.argv:
        build()
    if not os.path.isfile(TRUN):
        print("thiếu %s — chạy với --build" % TRUN)
        return 1
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    bad = []
    for name, cmd, want in SUITES:
        if not os.path.isfile(os.path.join(ROOT, cmd[1])):
            print("SKIP %-8s (chưa có %s)" % (name, cmd[1]))
            continue
        t0 = time.time()
        r = subprocess.run(cmd, cwd=ROOT, capture_output=True, env=env)
        out = r.stdout.decode("utf-8", "replace") + r.stderr.decode("utf-8", "replace")
        ok = r.returncode == 0 and want in out
        print("%s %-8s %5.1fs" % ("PASS" if ok else "FAIL", name, time.time() - t0))
        if not ok:
            bad.append(name)
            print("\n".join("    " + x for x in out.strip().splitlines()[-25:]))
    # mọi tests/*.t phải chạy được bằng t.py (exit 0)
    tdir = os.path.join(ROOT, "tests")
    for fn in sorted(os.listdir(tdir)):
        if not fn.endswith(".t"):
            continue
        r = subprocess.run([sys.executable, "t.py", os.path.join("tests", fn)], cwd=ROOT,
                           capture_output=True, env=env)
        if r.returncode != 0:
            bad.append("t.py " + fn)
            print("FAIL t.py tests/%s\n    %s" % (fn, r.stderr.decode("utf-8", "replace")[-300:]))
    print("ALL-GREEN" if not bad else "FAILED: " + ", ".join(bad))
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
