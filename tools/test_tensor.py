"""test_tensor.py — kiểm tensor n-chiều của T theo NGỮ NGHĨA numpy + hiệu năng.
Chạy: python tools/test_tensor.py   (in TENSOR-ALL-GREEN khi xanh)
  1. Sinh dữ liệu ngẫu nhiên -> chương trình T (broadcast, @, rút gọn theo trục,
     transpose/reshape, hàm toán học, softmax...) chạy trên VM C và t.py;
     so với numpy (sai số tương đối 1e-9) — bỏ qua nếu không có numpy.
  2. VM C và t.py phải ra output giống hệt nhau (từng bit).
  3. matmul 200x200 trên VM nhanh hơn nhân ma trận Python thuần >= 20 lần.
"""
import ast
import os
import random
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
            print("     ", str(extra)[-600:])


def lit(a):
    return repr(a.tolist()).replace("e+", "e")


def main():
    try:
        import numpy as np
    except ImportError:
        np = None
    if not os.path.isfile(TRUN):
        print("SKIP (thiếu %s)" % TRUN)
        return 0
    tmp = tempfile.mkdtemp(prefix="tten")
    rng = random.Random(2026)

    def arr(*shape):
        n = 1
        for s in shape:
            n *= s
        data = [round(rng.uniform(-3, 3), 6) for _ in range(n)]
        if np is not None:
            return np.array(data, dtype=float).reshape(shape)
        return data

    cases = []   # (biểu thức T, hàm numpy)
    if np is not None:
        A = arr(3, 4)
        B = arr(4, 5)
        C = arr(1, 4)
        D = arr(3, 1)
        V = arr(4)
        W = arr(2, 3, 4)
        P = np.abs(arr(3, 4)) + 0.5
        src = ["A = tensor(%s)" % lit(A), "B = tensor(%s)" % lit(B), "C = tensor(%s)" % lit(C),
               "D = tensor(%s)" % lit(D), "V = tensor(%s)" % lit(V), "W = tensor(%s)" % lit(W),
               "P = tensor(%s)" % lit(P)]
        cases = [
            ("A + C", lambda: A + C), ("A * D", lambda: A * D), ("C - D", lambda: C - D),
            ("A / P", lambda: A / P), ("P ** 0.5", lambda: P ** 0.5), ("2 - A", lambda: 2 - A),
            ("A @ B", lambda: A @ B), ("A @ V", lambda: A @ V), ("V @ B", lambda: V @ B),
            ("V @ V", lambda: V @ V), ("A.T @ A", lambda: A.T @ A),
            ("A.sum()", lambda: A.sum()), ("A.sum(0)", lambda: A.sum(0)),
            ("A.sum(1)", lambda: A.sum(1)), ("A.mean(0)", lambda: A.mean(0)),
            ("A.max(1)", lambda: A.max(1)), ("A.min(0)", lambda: A.min(0)),
            ("W.sum(1)", lambda: W.sum(1)), ("W.mean(-1)", lambda: W.mean(-1)),
            ("A.argmax()", lambda: A.argmax()), ("A.argmax(1)", lambda: A.argmax(1)),
            ("W.T", lambda: W.T), ("W.reshape(4, 6)", lambda: W.reshape(4, 6)),
            ("A.exp()", lambda: np.exp(A)), ("P.log()", lambda: np.log(P)),
            ("A.tanh()", lambda: np.tanh(A)), ("A.sigmoid()", lambda: 1 / (1 + np.exp(-A))),
            ("P.sqrt()", lambda: np.sqrt(P)), ("A.relu()", lambda: np.maximum(A, 0)),
            ("A.clip(-1, 1)", lambda: np.clip(A, -1, 1)),
            ("A.softmax()", lambda: np.exp(A - A.max(-1, keepdims=True)) /
             np.exp(A - A.max(-1, keepdims=True)).sum(-1, keepdims=True)),
            ("(A @ B + 1).softmax().sum(1)", lambda: np.ones(3)),
            ("abs(A)", lambda: np.abs(A)), ("-A", lambda: -A),
            ("A[1]", lambda: A[1]), ("W[1][2]", lambda: W[1][2]),
            ("A[::-1]", lambda: A[::-1]), ("eye(3) @ A", lambda: np.eye(3) @ A),
            ("A[:, 1]", lambda: A[:, 1]), ("W[1, :, 2]", lambda: W[1, :, 2]),
            ("W[:, 2]", lambda: W[:, 2]), ("A[2, 3]", lambda: A[2, 3]),
        ]
        for i, (e, _) in enumerate(cases):
            src.append("print(%s)" % e)
        prog = os.path.join(tmp, "np.t")
        with open(prog, "w", encoding="utf-8") as f:
            f.write("\n".join(src) + "\n")
        tb = os.path.join(tmp, "np.tb")
        r = subprocess.run([sys.executable, os.path.join(ROOT, "vm", "tcomp.py"), prog, tb],
                           capture_output=True, cwd=ROOT)
        check("compile chương trình so numpy", r.returncode == 0, r.stderr)
        rv = subprocess.run([TRUN, tb], capture_output=True, cwd=ROOT)
        rp = subprocess.run([sys.executable, os.path.join(ROOT, "t.py"), prog],
                            capture_output=True, cwd=ROOT)
        check("VM chạy được", rv.returncode == 0, rv.stderr)
        check("VM == t.py (từng bit)", rv.stdout.replace(b"\r\n", b"\n") ==
              rp.stdout.replace(b"\r\n", b"\n"), rp.stderr[-300:])
        lines = rv.stdout.decode("utf-8", "replace").strip().splitlines()
        bad = []
        for (e, fn), ln in zip(cases, lines):
            got = np.array(ast.literal_eval(ln.replace("nan", "None")), dtype=float)
            want = np.array(fn(), dtype=float)
            if got.shape != want.shape and not (got.size == want.size == 1):
                bad.append("%s: shape %s != %s" % (e, got.shape, want.shape))
            elif not np.allclose(got.reshape(want.shape), want, rtol=1e-9, atol=1e-12):
                bad.append("%s: %s != %s" % (e, got.ravel()[:4], want.ravel()[:4]))
        check("%d phép khớp numpy" % (len(cases) - len(bad)), not bad and len(lines) == len(cases),
              "; ".join(bad))
    else:
        print("SKIP so numpy (không có numpy)")

    # hiệu năng matmul: VM C vs Python thuần
    n = 200
    prog = os.path.join(tmp, "mm.t")
    with open(prog, "w", encoding="utf-8") as f:
        f.write("seed(3)\nA = rand([%d, %d])\nB = rand([%d, %d])\n"
                "mut s = 0.0\nfor i in range(5):\n    s += (A @ B).sum()\nprint(s)\n" % (n, n, n, n))
    tb = os.path.join(tmp, "mm.tb")
    subprocess.run([sys.executable, os.path.join(ROOT, "vm", "tcomp.py"), prog, tb],
                   capture_output=True, cwd=ROOT)
    t0 = time.perf_counter()
    rv = subprocess.run([TRUN, tb], capture_output=True)
    tv = time.perf_counter() - t0
    a = [[rng.random() for _ in range(n)] for _ in range(n)]
    b = [[rng.random() for _ in range(n)] for _ in range(n)]
    t0 = time.perf_counter()
    bt = list(zip(*b))
    [[sum(x * y for x, y in zip(row, col)) for col in bt] for row in a]
    tp = (time.perf_counter() - t0) * 5
    check("matmul %dx%d x5: VM %.3fs vs Python thuần %.3fs (>= 20x)" % (n, n, tv, tp),
          rv.returncode == 0 and tp >= 20 * tv, rv.stderr)
    print("TENSOR-ALL-GREEN" if not FAILS else "TENSOR-FAIL %d" % len(FAILS))
    return 1 if FAILS else 0


if __name__ == "__main__":
    for st in (sys.stdout,):
        try:
            st.reconfigure(encoding="utf-8", errors="replace")
        except Exception:  # noqa: BLE001
            pass
    raise SystemExit(main())
