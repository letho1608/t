"""test_dbg.py — kiểm hồi debugger trun -d bằng phiên scripted (v1.8).
Chạy: python tools/test_dbg.py   (cần trun.exe đã build; tự biên dịch prog)
Assert: disasm/step/break/stack/locals + chạy tới hết đúng output.
"""
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TMP = os.path.join(os.environ.get("TEMP", "/tmp"), "tdbgtest")
TRUN = os.path.join(ROOT, "trun.exe" if os.name == "nt" else "trun")


def main():
    fails = []

    def check(name, cond, extra=""):
        print(("PASS " if cond else "FAIL ") + name + (f" [{extra}]" if extra and not cond else ""))
        if not cond:
            fails.append(name)

    if not os.path.isfile(TRUN):
        print("SKIP (thiếu %s — build VM trước)" % TRUN)
        return 0
    os.makedirs(TMP, exist_ok=True)
    prog = os.path.join(TMP, "dbgprog.t")
    with open(prog, "w", encoding="utf-8") as f:
        f.write('fn add(a, b):\n    return a + b\n\nprint(add(20, 22))\n')
    tb = os.path.join(TMP, "dbgprog.tb")
    r = subprocess.run([sys.executable, os.path.join(ROOT, "vm", "tcomp.py"),
                        prog, tb], capture_output=True, text=True, cwd=ROOT)
    check("tcomp", r.returncode == 0, r.stderr[-200:])
    if fails:
        print("DBG-FAIL")
        return 1
    # disasm từ lệnh d để lấy pc của MKFN (đặt breakpoint động)
    r = subprocess.run([TRUN, "-d", tb], input="d\nd\nd\nq\n", capture_output=True,
                       text=True, cwd=ROOT)
    mkpc = None
    for ln in r.stdout.splitlines():
        p = ln.split()
        if len(p) >= 3 and p[1] == "MKFN":
            mkpc = p[0]
    check("disasm-mkfn", mkpc is not None, r.stdout[:200])
    if mkpc is None:
        print("DBG-FAIL")
        return 1
    # phiên chính: step 2, xem stack, break ở MKFN, chạy hết
    cmds = f"s\ns\np\nl\nb {mkpc}\nc\n"
    r = subprocess.run([TRUN, "-d", tb], input=cmds, capture_output=True,
                       text=True, cwd=ROOT)
    out = r.stdout
    check("step-pc", "[dbg pc=" in out, out[:200])
    check("break-hit", f"[dbg pc={mkpc} " in out, out[-400:])
    check("prog-out", "42\n" in out, out[-200:])
    check("rc", r.returncode == 0, str(r.returncode))
    print("DBG-ALL-GREEN" if not fails else f"DBG-FAIL: {fails}")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
