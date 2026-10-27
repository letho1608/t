# test_ops.t — Kiểm tra bảng OP_* và NAT_* trong vm/tvm.h khớp self/tcomp.t
import os

fn parse_ops(text):
    mut out = []
    mut idx = 0
    mut in_enum = false
    for raw in text.split("\n"):
        line = raw.split("//")[0].split("/*")[0].strip()
        if line.startswith("enum {"):
            in_enum = true
            idx = 0
        if not in_enum:
            continue
        if line.startswith("}"):
            if out.len > 0:
                return out
            in_enum = false
            continue
        for tok in line.split(","):
            t = tok.strip()
            if not t.startswith("OP_"):
                continue
            if "=" in t:
                parts = t.split("=")
                k = parts[0].strip()
                idx = 1
                out.append([k, idx])
            else:
                out.append([t, idx])
            idx = idx + 1
    return out

fn parse_nats(text):
    mut out = []
    for raw in text.split("\n"):
        line = raw.split("//")[0].split("/*")[0].strip()
        if not line.startswith("#define NAT_"):
            continue
        rest = line[8:line.len].strip()
        mut sp = rest.find(" ")
        if sp < 0:
            sp = rest.find("\t")
        if sp < 0:
            continue
        k = rest[0:sp].strip()
        if k == "NAT_NFIXED":
            continue
        v = rest[sp + 1:rest.len].strip()
        out.append([k, int(v)])
    return out

fn parse_list_fn(text, fn_name):
    marker = "fn " + fn_name + "():"
    p = text.find(marker)
    assert p >= 0, "khong tim thay " + marker
    sub = text[p:text.len]
    b1 = sub.find("[")
    b2 = sub.find("]")
    body = sub[b1 + 1:b2]
    mut out = []
    mut i = 0
    while i < body.len:
        if body[i] == "\"":
            mut j = i + 1
            while j < body.len and body[j] != "\"":
                j = j + 1
            out.append(body[i + 1:j])
            i = j + 1
        else:
            i = i + 1
    return out

h_text = os.read_file("vm/tvm.h")
tc_text = os.read_file("self/tcomp.t")

c_op = parse_ops(h_text)
c_nat = parse_nats(h_text)
t_op = parse_list_fn(tc_text, "opnames")
t_nat = parse_list_fn(tc_text, "natnames")

mut errs = []
if c_op.len != t_op.len:
    errs.append("so luong OP lech: C=" + str(c_op.len) + " T=" + str(t_op.len))
mut i = 0
while i < c_op.len and i < t_op.len:
    cn = c_op[i][0]
    ci = c_op[i][1]
    tn = t_op[i]
    if ci != i + 1 or cn != "OP_" + tn:
        errs.append("OP[" + str(i) + "]: C=" + cn + "(" + str(ci) + ") T=OP_" + tn)
    i = i + 1

if c_nat.len != t_nat.len:
    errs.append("so luong NAT lech: C=" + str(c_nat.len) + " T=" + str(t_nat.len))
mut j = 0
while j < c_nat.len and j < t_nat.len:
    cn = c_nat[j][0]
    ci = c_nat[j][1]
    tn = t_nat[j]
    if ci != j or cn.lower() != "nat_" + tn:
        errs.append("NAT[" + str(j) + "]: C=" + cn + "(" + str(ci) + ") T=NAT_" + tn)
    j = j + 1

if errs.len > 0:
    for e in errs:
        print("FAIL:", e)
    os.exit(1)

print("OPS-OK (" + str(c_op.len) + " opcodes, " + str(c_nat.len) + " natives)")
