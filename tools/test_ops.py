"""test_ops.py — khóa đồng bộ opcode/native: tvm.h <-> tcomp.py <-> self/tcomp.t (v2.0).
Chạy: python tools/test_ops.py   (exit 0 = khớp)
So thứ tự enum trong tvm.h với _OPNAMES trong tcomp.py và opnames() trong self/tcomp.t.
"""
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def t_tables(path):
    """Rút list string từ hàm `fn name(): return [...]` trong file T."""
    src = open(path, encoding="utf-8").read()
    out = {}
    for m in re.finditer(r"^fn (\w+)\(\):\s*\n\s*return \[(.*?)\]", src, re.S | re.M):
        out[m.group(1)] = re.findall(r'"([A-Za-z_0-9]+)"', m.group(2))
    return out


def h_ops():
    src = open(os.path.join(ROOT, "vm", "tvm.h"), encoding="utf-8").read()
    names = []
    for m in re.finditer(r"enum \{(.*?)\};", src, re.S):
        body = re.sub(r"/\*.*?\*/", "", m.group(1), flags=re.S)
        cur = []
        for part in body.split(","):
            tok = part.strip().split()[0] if part.strip().split() else ""
            if tok.startswith("OP_"):
                cur.append(tok[3:])
        if cur:
            names = cur
    return names


def py_ops():
    sys.path.insert(0, os.path.join(ROOT, "vm"))
    import tcomp
    inv = sorted(tcomp.OP.items(), key=lambda kv: kv[1])
    return [k for k, _ in inv]


def main():
    h, p = h_ops(), py_ops()
    ok = True
    if len(h) != len(p):
        print(f"FAIL count: h={len(h)} py={len(p)}")
        ok = False
    for i, (a, b) in enumerate(zip(h, p)):
        if a != b:
            print(f"FAIL [{i}]: h={a} py={b}")
            ok = False
    only_h = [x for x in h if x not in p]
    only_p = [x for x in p if x not in h]
    if only_h:
        print("FAIL only-h:", only_h)
        ok = False
    if only_p:
        print("FAIL only-py:", only_p)
        ok = False
    # self/tcomp.t phải có đúng thứ tự opcode/native như C header
    tt = t_tables(os.path.join(ROOT, "self", "tcomp.t"))
    tops = tt.get("opnames")
    tnat = tt.get("natnames")
    if not tops or not tnat:
        print("FAIL self/tcomp.t thiếu opnames()/natnames()")
        ok = False
    else:
        if tops != h:
            for i, (a, b) in enumerate(zip(tops, h)):
                if a != b:
                    print(f"FAIL tcomp.t op [{i}]: t={a} h={b}")
                    ok = False
                    break
            if len(tops) != len(h):
                print(f"FAIL tcomp.t op count: t={len(tops)} h={len(h)}")
                ok = False
    # NAT_* khớp tcomp.NAT
    sys.path.insert(0, os.path.join(ROOT, "vm"))
    import tcomp
    nh = {}
    for m in re.finditer(r"#define (NAT_\w+)\s+(\d+)", open(os.path.join(ROOT, "vm", "tvm.h"), encoding="utf-8").read()):
        nh[m.group(1)[4:].lower()] = int(m.group(2))
    for k, v in tcomp.NAT.items():
        if nh.get(k) != v:
            print(f"FAIL NAT {k}: h={nh.get(k)} py={v}")
            ok = False
    if tnat and sorted(tnat) != sorted(tcomp.NAT):
        print("FAIL tcomp.t natnames:", [x for x in tnat if x not in tcomp.NAT],
              [x for x in tcomp.NAT if x not in tnat])
        ok = False
    if tcomp.NFIXED != len(tcomp.NAT):
        print(f"FAIL NFIXED {tcomp.NFIXED} != {len(tcomp.NAT)}")
        ok = False
    print(f"OPS-OK ({len(h)} opcodes + {len(tcomp.NAT)} natives, 3 sources in sync)"
          if ok else "OPS-FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
