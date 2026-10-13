"""test_tcomp_t.py - kiem chung compiler T thuan T (self/tcomp.t + self/tcli.t, v2.0).

Chay: python tools/test_tcomp_t.py        (exit 0 = xanh)

Quy trinh:
  1. bootstrap 1 lan bang Python: vm/tcomp.py -b self/tcli.t -> tcomp.tb
  2. tu do moi thu T chay tren trun.exe, KHONG con Python:
       trun tcomp.tb [-b] tests/X.t X.tb  &&  trun X.tb  ==  python t.py tests/X.t
  3. tu bien dich: trun tcomp.tb -b self/tcli.t tcomp2.tb
     va phai co fixed point: tcomp2 -> tcomp3 byte-identical
  4. output cua stage1 (tcomp.tb) va stage2 (tcomp2.tb) phai byte-identical
  5. chuoi literal nhieu byte: compiler chay tren VM C va tren t.py phai ra
     cung byte (UTF-8 xuyen tach)
"""
import hashlib
import os
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRUN = os.path.join(ROOT, "trun.exe" if os.name == "nt" else "trun")
CASES = [
    ("tests/demo.t", False), ("tests/core.t", False),
    ("tests/assert.t", False), ("tests/jit.t", False),
    ("tests/closure.t", False), ("tests/fstr.t", False),
    ("tests/mem.t", False), ("tests/bytesem.t", False), ("tests/mods.t", True),
    ("tests/v03.t", True), ("tests/selfhost.t", True),
    ("tests/fmtself.t", True), ("tests/hoist.t", False),
    ("tests/builtins.t", False), ("tests/lang.t", False),
    ("tests/tryflow.t", False), ("tests/jit2.t", False),
    ("tests/gcstress.t", False), ("tests/tensor.t", False),
    ("tests/jitlist.t", False), ("tests/lambda.t", False),
    ("tests/jitmain.t", False), ("tests/agtest.t", False),
]
# suite chạy thêm qua evaluator self-host (eval.t bundle -> tdriver.tb trên VM C)
EVAL_CASES = ["tests/demo.t", "tests/core.t", "tests/assert.t", "tests/closure.t",
              "tests/fstr.t", "tests/hoist.t", "tests/builtins.t", "tests/lang.t",
              "tests/tryflow.t", "tests/bytesem.t", "tests/tensor.t",
              "tests/lambda.t"]


def sh(*args, **kw):
    r = subprocess.run(list(args), capture_output=True, cwd=ROOT, **kw)
    r.out = r.stdout
    r.stdout = r.stdout.decode("utf-8", "replace")
    r.stderr = r.stderr.decode("utf-8", "replace")
    return r


def sha(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def compile_with(stage, src, out, bundle):
    args = [TRUN, stage] + (["-b"] if bundle else []) + [src, out]
    r = sh(*args)
    if r.returncode != 0:
        print("FAIL compile %s %s rc=%d" % (stage, src, r.returncode))
        print("      ", r.stdout.strip()[-200:], r.stderr.strip()[-200:])
        return False
    return os.path.exists(out)


def main():
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
        except Exception:
            pass
    ok = True
    if not os.path.exists(TRUN):
        print("SKIP: khong co %s (build VM truoc)" % TRUN)
        return 0
    tmp = tempfile.mkdtemp(prefix="tct")
    s1 = os.path.join(tmp, "tcomp.tb")
    s2 = os.path.join(tmp, "tcomp2.tb")
    s3 = os.path.join(tmp, "tcomp3.tb")
    cli = os.path.join(ROOT, "self", "tcli.t")

    r = sh(sys.executable, os.path.join(ROOT, "vm", "tcomp.py"), "-b", cli, s1)
    if r.returncode != 0:
        print("FAIL bootstrap", r.stdout[-300:], r.stderr[-300:])
        return 1
    print("bootstrap tcomp.tb", os.path.getsize(s1), "bytes")

    for src, b in CASES:
        name = os.path.basename(src)
        a = os.path.join(tmp, "a_" + name + ".tb")
        c = os.path.join(tmp, "c_" + name + ".tb")
        if not compile_with(s1, src, a, b):
            ok = False
            continue
        run = sh(TRUN, a)
        want = sh(sys.executable, os.path.join(ROOT, "t.py"), src)
        if run.returncode != 0 or run.out != want.out:
            ok = False
            print("FAIL run %s rc=%d" % (src, run.returncode))
            print("      got :", run.stdout.strip()[-200:])
            print("      want:", want.stdout.strip()[-200:], run.stderr.strip()[-200:])
        else:
            print("OK %s (%d bytes)" % (src, os.path.getsize(a)))

    if not compile_with(s1, cli, s2, True):
        print("FAIL self-compile stage2")
        return 1
    print("self-compile tcomp2.tb", os.path.getsize(s2), "bytes")
    if not compile_with(s2, cli, s3, True):
        print("FAIL self-compile stage3")
        return 1
    if sha(s2) != sha(s3):
        ok = False
        print("FAIL khong co fixed point: %s != %s" % (sha(s2)[:12], sha(s3)[:12]))
    else:
        print("OK self-hosting fixed point (%s)" % sha(s2)[:12])

    for src, b in CASES:
        name = os.path.basename(src)
        c = os.path.join(tmp, "c_" + name + ".tb")
        if not compile_with(s2, src, c, b):
            ok = False
            continue
        a = os.path.join(tmp, "a_" + name + ".tb")
        if os.path.exists(a) and sha(a) != sha(c):
            ok = False
            print("FAIL stage1 != stage2 output %s" % src)

    # GC quet rat day (moi canh lui/loi goi) van phai ra dung ket qua
    g = os.path.join(tmp, "a_gcstress.t.tb")
    if os.path.exists(g):
        env = dict(os.environ, TVM_GCTHRESH="3")
        r1 = subprocess.run([TRUN, g], capture_output=True, cwd=ROOT, env=env)
        want = sh(sys.executable, os.path.join(ROOT, "t.py"), "tests/gcstress.t")
        if r1.returncode != 0 or r1.stdout != want.out:
            ok = False
            print("FAIL gcstress voi TVM_GCTHRESH=3")
        else:
            print("OK gcstress voi GC quet day (TVM_GCTHRESH=3)")

    # loi runtime khong bat: VM (bang dong LN01) va t.py deu bao file:dong
    em = os.path.join(tmp, "errmod.t")
    with open(em, "w", encoding="utf-8") as f:
        f.write("fn chia(a, b):\n    mut t = a + 1\n    return t / b\n")
    ep = os.path.join(tmp, "errmain.t")
    with open(ep, "w", encoding="utf-8") as f:
        f.write("import errmod\nprint(\"truoc\")\n\nx = errmod.chia(1, 0)\nprint(\"khong toi\")\n")
    eb = os.path.join(tmp, "errmain.tb")
    if compile_with(s1, ep, eb, False):
        rv = sh(TRUN, eb)
        rp = sh(sys.executable, os.path.join(ROOT, "t.py"), ep)
        want = "tại errmod.t:3"
        if rv.returncode == 0 or want not in rv.stderr or rp.returncode == 0 or want not in rp.stderr \
                or rv.stdout.strip() != "truoc":
            ok = False
            print("FAIL vi tri loi runtime:", rv.stderr.strip()[-120:], "|", rp.stderr.strip()[-120:])
        else:
            print("OK loi runtime bao dung file:dong (VM + t.py)")
    else:
        ok = False

    # compiler Python (vm/tcomp.py) va compiler T phai ra bytecode giong tung byte
    for src, b in CASES:
        name = os.path.basename(src)
        a = os.path.join(tmp, "a_" + name + ".tb")
        p = os.path.join(tmp, "p_" + name + ".tb")
        r = sh(sys.executable, os.path.join(ROOT, "vm", "tcomp.py"), *(["-b"] if b else []), src, p)
        if r.returncode != 0 or not os.path.exists(a) or sha(a) != sha(p):
            ok = False
            print("FAIL py-compiler != T-compiler %s" % src)
    print("OK compiler Python == compiler T (byte-identical, %d suite)" % len(CASES))

    # evaluator self-host (eval.t) chay tren VM C phai khop t.py
    drv = os.path.join(tmp, "tdriver.tb")
    if not compile_with(s1, os.path.join(ROOT, "tools", "tdriver.t"), drv, True):
        ok = False
    else:
        bad = 0
        for src in EVAL_CASES:
            run = sh(TRUN, drv, src)
            want = sh(sys.executable, os.path.join(ROOT, "t.py"), src)
            if run.returncode != 0 or run.out != want.out:
                ok = False
                bad += 1
                print("FAIL eval.t-on-VM %s" % src)
                print("      got :", run.stdout.strip()[-200:], run.stderr.strip()[-200:])
                print("      want:", want.stdout.strip()[-200:])
        if not bad:
            print("OK eval.t tren VM C khop t.py (%d suite)" % len(EVAL_CASES))

    # UTF-8 xuyen tach giua VM C (byte) va t.py (codepoint)
    u = os.path.join(tmp, "utf8.t")
    text = ("a = \"%s\"\nb = \"%s\"\nc = \"%s\"\n"
            "print(a)\nprint(b)\nprint(c)\n"
            % (chr(0x54) + chr(0x69) + chr(0x1EC7) + chr(0x6E) + chr(0x67),
               chr(0x4F60) + chr(0x597D) + chr(0x4E16) + chr(0x754C),
               chr(0x1F600)))
    with open(u, "w", encoding="utf-8") as f:
        f.write(text)
    ub = os.path.join(tmp, "u_c.tb")
    ut = os.path.join(tmp, "u_tpy.tb")
    if not compile_with(s1, u, ub, False):
        ok = False
    drv_t = os.path.join(tmp, "drv.t")
    with open(drv_t, "w", encoding="utf-8") as f:
        f.write('import tcomp\n'
                'r = tcomp.compile_file("%s", false)\n'
                'g = open("%s", "wb")\n'
                'write(g, r[0])\n'
                'close(g)\n'
                % (u.replace("\\", "/"), ut.replace("\\", "/")))
    rd = subprocess.run([sys.executable, os.path.join(ROOT, "t.py"), drv_t],
                        capture_output=True, cwd=ROOT)
    if rd.returncode != 0 or not os.path.exists(ut):
        ok = False
        print("FAIL utf8 compile tren t.py", rd.stdout[-200:].decode("utf-8", "replace"),
              rd.stderr[-300:].decode("utf-8", "replace"))
    elif sha(ub) != sha(ut):
        ok = False
        print("FAIL utf8: VM C va t.py ra byte khac nhau")
    else:
        run = sh(TRUN, ub)
        if run.returncode != 0 or "\u4f60\u597d\u4e16\u754c" not in run.stdout:
            ok = False
            print("FAIL utf8 run", run.returncode, run.stdout[-200:], run.stderr[-200:])
        else:
            print("OK utf8 literal byte-identical tren ca 2 backend")

    print("TCOMP-T-OK" if ok else "TCOMP-T-FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
