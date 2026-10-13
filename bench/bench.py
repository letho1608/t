"""bench.py — so tốc độ T (trun: VM C + JIT) với CPython trên cùng thuật toán.
Chạy: python bench/bench.py [tên ...] [--runs N] [--nojit]
Mỗi bài: bench/<tên>.t (T) và bench/<tên>.py (Python), output phải giống nhau.
Thời gian = tốt nhất trong N lần chạy (mặc định 3), tính cả khởi động tiến trình.
"""
import os
import subprocess
import sys
import tempfile
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BENCH = os.path.join(ROOT, "bench")
TRUN = os.path.join(ROOT, "trun.exe" if os.name == "nt" else "trun")
ALL = ["fib", "loop", "loopfn", "mandel", "sieve", "dict", "method", "matmul", "nn"]


def best(cmd, runs, env=None):
    t_best, out = None, None
    for _ in range(runs):
        t0 = time.perf_counter()
        r = subprocess.run(cmd, capture_output=True, env=env)
        dt = time.perf_counter() - t0
        if r.returncode != 0:
            return None, r.stdout + r.stderr
        out = r.stdout
        t_best = dt if t_best is None else min(t_best, dt)
    return t_best, out


def main():
    args = sys.argv[1:]
    runs = 3
    if "--runs" in args:
        i = args.index("--runs")
        runs = int(args[i + 1])
        del args[i:i + 2]
    env = dict(os.environ)
    if "--nojit" in args:
        args.remove("--nojit")
        env["TVM_NOJIT"] = "1"
    names = args or [n for n in ALL if os.path.isfile(os.path.join(BENCH, n + ".t"))]
    tmp = tempfile.mkdtemp(prefix="tbench")
    print("%-8s %10s %10s %8s  %s" % ("bench", "T (trun)", "CPython", "T/Py", "output"))
    for n in names:
        tb = os.path.join(tmp, n + ".tb")
        r = subprocess.run([sys.executable, os.path.join(ROOT, "vm", "tcomp.py"),
                            os.path.join(BENCH, n + ".t"), tb], capture_output=True)
        if r.returncode != 0:
            print("%-8s compile FAIL %s" % (n, r.stderr.decode("utf-8", "replace")[-200:]))
            continue
        tt, ot = best([TRUN, tb], runs, env)
        tp, op = best([sys.executable, os.path.join(BENCH, n + ".py")], runs)
        if tt is None or tp is None:
            print("%-8s run FAIL %s" % (n, (ot if tt is None else op)[-200:]))
            continue
        same = ot.replace(b"\r\n", b"\n") == op.replace(b"\r\n", b"\n")
        print("%-8s %9.3fs %9.3fs %7.2fx  %s" % (
            n, tt, tp, tt / tp, "khớp" if same else "LỆCH: %r vs %r" % (ot[:40], op[:40])))
    return 0


if __name__ == "__main__":
    for st in (sys.stdout,):
        try:
            st.reconfigure(encoding="utf-8", errors="replace")
        except Exception:  # noqa: BLE001
            pass
    raise SystemExit(main())
