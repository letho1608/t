"""test_jit.py — kiểm JIT x86-64 (v2.2): đúng + thực sự kích hoạt + nhanh hơn.
Chạy: python tools/test_jit.py   (in JIT-ALL-GREEN khi xanh)
  1. tests/jit2.t, jit.t, jitlist.t, jitmain.t, jitrec.t, jitdict.t: output VM có JIT == VM tắt JIT (TVM_NOJIT) == t.py
  2. log TVM_JITLOG: đủ số hàm được biên dịch entry và vùng OSR
  3. fib(27) và vòng lặp float: JIT nhanh hơn interpreter ít nhất 3 lần
"""
import os
import subprocess
import sys
import tempfile
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRUN = os.path.join(ROOT, "trun.exe" if os.name == "nt" else "trun")
FAILS = []


def check(name, cond, extra=""):
    print(("PASS " if cond else "FAIL ") + name)
    if not cond:
        FAILS.append(name)
        if extra:
            print("     ", str(extra)[-500:])


def compile_t(src, out):
    r = subprocess.run([sys.executable, os.path.join(ROOT, "vm", "tcomp.py"), src, out],
                       capture_output=True, cwd=ROOT)
    return r.returncode == 0


def run(tb, env_extra=None):
    env = dict(os.environ)
    env.pop("TVM_NOJIT", None)
    env.pop("TVM_JITLOG", None)
    if env_extra:
        env.update(env_extra)
    t0 = time.perf_counter()
    r = subprocess.run([TRUN, tb], capture_output=True, env=env, cwd=ROOT)
    return r, time.perf_counter() - t0


def main():
    if not os.path.isfile(TRUN):
        print("SKIP (thiếu %s)" % TRUN)
        return 0
    tmp = tempfile.mkdtemp(prefix="tjit")
    for name in ("jit2", "jit", "jitlist", "jitmain", "jitrec", "jitdict"):
        src = os.path.join(ROOT, "tests", name + ".t")
        tb = os.path.join(tmp, name + ".tb")
        check("compile " + name, compile_t(src, tb))
        rj, _ = run(tb, {"TVM_JITLOG": "1"})
        rn, _ = run(tb, {"TVM_NOJIT": "1"})
        rp = subprocess.run([sys.executable, os.path.join(ROOT, "t.py"), src],
                            capture_output=True, cwd=ROOT)
        check(name + ": JIT == interpreter", rj.returncode == 0 and rj.stdout == rn.stdout,
              rj.stdout[-300:] + rj.stderr[-300:])
        check(name + ": JIT == t.py", rj.stdout.replace(b"\r\n", b"\n") ==
              rp.stdout.replace(b"\r\n", b"\n"), rp.stdout[-300:] + rp.stderr[-300:])
        if name == "jit2":
            log = rj.stderr.decode("utf-8", "replace")
            ne = log.count("entry ok")
            no = log.count("osr ok")
            check("jit2: >= 12 hàm JIT entry (có %d)" % ne, ne >= 12, log)
            check("jit2: >= 2 vùng OSR (có %d)" % no, no >= 2, log)
        if name == "jitlist":
            log = rj.stderr.decode("utf-8", "replace")
            no = log.count("osr ok")
            check("jitlist: >= 6 vùng OSR có list (có %d)" % no, no >= 6, log)
        if name == "jitrec":
            log = rj.stderr.decode("utf-8", "replace")
            ne, no = log.count("entry ok"), log.count("osr ok")
            check("jitrec: >= 4 hàm/method đọc record JIT (có %d)" % ne, ne >= 4, log)
            check("jitrec: >= 3 vùng OSR đọc record (có %d)" % no, no >= 3, log)
        if name == "jitmain":
            log = rj.stderr.decode("utf-8", "replace")
            nm = log.count("main ok")
            check("jitmain: >= 5 vòng lặp top-level JIT (có %d)" % nm, nm >= 5, log)
        if name == "jitdict":
            log = rj.stderr.decode("utf-8", "replace")
            no = log.count("osr ok")
            check("jitdict: >= 3 vùng OSR dict/chuỗi (có %d)" % no, no >= 3, log)

    # tốc độ: JIT phải nhanh hơn rõ rệt
    prog = os.path.join(tmp, "speed.t")
    with open(prog, "w", encoding="utf-8") as f:
        f.write("fn fib(n):\n    if n < 2:\n        return n\n    return fib(n - 1) + fib(n - 2)\n"
                "fn fl(n):\n    mut x = 0.0\n    mut i = 0\n    while i < n:\n"
                "        x = x * 0.5 + i * 1.5\n        i += 1\n    return x\n"
                "print(fib(27), fl(3000000))\n")
    tb = os.path.join(tmp, "speed.tb")
    compile_t(prog, tb)
    best_j = best_n = None
    out_j = out_n = None
    for _ in range(2):
        rj, tj = run(tb)
        rn, tn = run(tb, {"TVM_NOJIT": "1"})
        best_j = tj if best_j is None else min(best_j, tj)
        best_n = tn if best_n is None else min(best_n, tn)
        out_j, out_n = rj.stdout, rn.stdout
    check("speed: cùng kết quả", out_j == out_n, (out_j, out_n))
    check("speed: JIT %.3fs vs interpreter %.3fs (>= 3x)" % (best_j, best_n),
          best_n >= 3 * best_j)
    print("JIT-ALL-GREEN" if not FAILS else "JIT-FAIL %d" % len(FAILS))
    return 1 if FAILS else 0


if __name__ == "__main__":
    for st in (sys.stdout,):
        try:
            st.reconfigure(encoding="utf-8", errors="replace")
        except Exception:  # noqa: BLE001
            pass
    raise SystemExit(main())
