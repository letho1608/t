# self/tcomp.t — compiler T -> bytecode .tb viết bằng chính T (v2.1 thuần T)
# Thư viện; CLI nằm ở self/tcli.t (from tcomp import main_cli).
# Bootstrap một lần: python vm/tcomp.py -b self/tcli.t tcomp.tb
# Sau đó không cần Python:
#   trun tcomp.tb prog.t prog.tb          (dịch)
#   trun tcomp.tb prog.t prog.tb          (có import thì tự gộp module)
#   trun tcomp.tb -b self/tcli.t t2.tb    (tự biên dịch, tới fixed point)
# .tb: "TB01" + u16 nglobals + u32 len + code. Op theo vm/tvm.h.
# Phạm vi: full core (mọi node/stmt, closure, try, assert, fstr, mem,
# f64bits) + inliner hàm single-return + peephole superop; import qua -b.
# Chỉ 1 primitive C: f64bits. Int 2/4/8 byte và UTF-8 làm thuần T.
# String là byte-string (1 char = 1 byte) ở cả VM C lẫn t.py: UTF-8 pass-through.
import host
import parse2

# ---------- bảng opcode/native ----------
fn opnames():
    return ["PUSH_NIL", "PUSH_BOOL", "PUSH_INT", "PUSH_FLOAT", "PUSH_STR", "NEW_LIST", "NEW_TENSOR", "NEW_DICT", "ADD", "SUB", "MUL", "DIV", "MOD", "POW", "NEG", "NOT", "EQ", "NE", "LT", "GT", "LE", "GE", "CONTAINS", "MATCHGLOB", "TDOT", "IDX", "SETIDX", "SLICE", "LEN", "PRINT", "JMP", "JMPF", "JMPT", "CALL", "RET", "LOAD", "STORE", "DECL", "LOADG", "STOREG", "DECLG", "MKFN", "CAPLOCAL", "CAPUP", "CAPG", "LOADU", "STOREU", "BIND", "BINDG", "RANGE", "ATTR", "RAISE", "TRY_PUSH", "TRY_POP", "FOR_PREP", "POP", "DUP", "INC_L", "INC_G", "DEC_L", "DEC_G", "ADDI_L", "ADDI_G", "ADDS_L", "ADDS_G", "SUBS_L", "SUBS_G", "INCB_L", "INCB_G", "IN", "CALLM", "SETATTR", "NOP", "HALT"]

fn build_op():
    mut d = {}
    mut i = 0
    for n in opnames():
        d[n] = 1 + i
        i = i + 1
    return d

fn natnames():
    return ["range", "len", "str", "int", "float", "type", "tensor", "open",
        "read", "write", "close", "ord", "chr", "argv", "print", "mem",
        "f64bits", "abs", "round", "min", "max", "sum", "any", "all",
        "sort", "reversed", "enumerate", "zip", "map", "filter",
        "set", "input", "__struct", "__new",
        "zeros", "ones", "full", "arange", "eye", "rand", "randn", "seed",
        "sqrt", "exp", "log", "sin", "cos", "tanh", "floor", "ceil", "pi", "load"]

fn build_nat():
    mut d = {}
    mut i = 0
    for n in natnames():
        d[n] = i
        i = i + 1
    return d

fn binop_name(op):
    if op == "+":
        return "ADD"
    if op == "-":
        return "SUB"
    if op == "*":
        return "MUL"
    if op == "/":
        return "DIV"
    if op == "%":
        return "MOD"
    if op == "**":
        return "POW"
    if op == "@":
        return "TDOT"
    if op == "==":
        return "EQ"
    if op == "!=":
        return "NE"
    if op == "<":
        return "LT"
    if op == ">":
        return "GT"
    if op == "<=":
        return "LE"
    if op == ">=":
        return "GE"
    if op == "in":
        return "IN"
    return nil

# ---------- số nguyên -> byte (thuần T) ----------
fn dec_digits(s):
    mut d = []
    for i in range(s.len):
        c = s[i:i + 1]
        d.append(ord(c) - ord("0"))
    return d

fn divmod256(d):
    mut q = []
    mut rem = 0
    for x in d:
        cur = rem * 10 + x
        rem = int(cur - int(cur / 256) * 256)
        q.append(int((cur - rem) / 256))
    while q.len > 1 and q[0] == 0:
        q = q[1:]
    return [q, rem]

fn pow256(n):
    mut r = 1
    mut i = 0
    while i < n:
        r = r * 256
        i = i + 1
    return r

fn be_bytes(v, n):
    if n < 0 or n > 8:
        raise "be_bytes n"
    mut neg = false
    if v < 0:
        neg = true
        v = 0 - v
    s = str(v)
    if s.len > 19:
        raise "int quá lớn (int64)"
    if s.len == 19:
        lim = "9223372036854775807"
        if neg:
            lim = "9223372036854775808"
        if s > lim:
            raise "int quá lớn (int64)"
    mut d = dec_digits(s)
    mut out = []
    mut i = 0
    while i < n:
        r = divmod256(d)
        d = r[0]
        out.append(r[1])
        i = i + 1
    if not (d.len == 1 and d[0] == 0):
        raise "int quá lớn"
    if neg:
        # two's complement: (~v) + 1, carry từ byte thấp lên cao
        mut j = 0
        while j < n:
            out[j] = 255 - out[j]
            j = j + 1
        mut k = 0
        while k < n:
            out[k] = out[k] + 1
            if out[k] < 256:
                break
            out[k] = 0
            k = k + 1
    return out

fn u16_bytes(v):
    return [v % 256, int(v / 256) % 256]

fn rd_u16(code, at):
    return code[at] + code[at + 1] * 256

fn rd_u32(code, at):
    return code[at] + code[at + 1] * 256 + code[at + 2] * 65536 + code[at + 3] * 16777216

fn rd_i32(code, at):
    v = rd_u32(code, at)
    if v >= 2147483648:
        v = v - 4294967296
    return v

fn rd_i64(code, at):
    # little-endian: byte thấp trước
    mut v = 0
    mut i = 7
    while i >= 0:
        v = v * 256 + code[at + i]
        i = i - 1
    if code[at + 7] >= 128:
        mut comp = 0
        i = 7
        while i >= 0:
            comp = comp * 256 + (255 - code[at + i])
            i = i - 1
        return 0 - (comp + 1)
    return v

# ---------- số trong AST (self/lex.t giữ chuỗi; t.py đã ra số) ----------
fn digit_val(c):
    if c >= "0" and c <= "9":
        return ord(c) - ord("0")
    if c >= "a" and c <= "f":
        return ord(c) - ord("a") + 10
    if c >= "A" and c <= "F":
        return ord(c) - ord("A") + 10
    return -1

fn acc_base(s, i, base, acc):
    mut v = acc
    while i < s.len:
        d = digit_val(s[i:i + 1])
        if d < 0 or d >= base:
            raise "số không hợp lệ: " + s
        if v > 1000000000000000000:
            raise "int quá lớn"
        v = v * base + d
        i = i + 1
    return v

fn parse_num(s):
    mut t = ""
    for i in range(s.len):
        c = s[i:i + 1]
        if c != "_":
            t = t + c
    if t.len > 1:
        p = t[0:2]
        if p == "0x" or p == "0X":
            return acc_base(t, 2, 16, 0)
        if p == "0b" or p == "0B":
            return acc_base(t, 2, 2, 0)
        if p == "0o" or p == "0O":
            return acc_base(t, 2, 8, 0)
    if t.split(".").len > 1 or t.split("e").len > 1 or t.split("E").len > 1:
        return float(t)
    return acc_base(t, 0, 10, 0)

fn num_val(v):
    if type(v) == "str":
        return parse_num(v)
    return v

fn lit_val(v):
    if type(v) == "str":
        if v == "true":
            return true
        if v == "false":
            return false
        return nil
    return v

# ---------- trạng thái compiler ----------
# cc = {"b": byte list, "gnames": {}, "fn_names": [], "fnmap": [], "fnorder": [],
#       "loops": [], "fns": [], "op": {}, "nat": {}, "fntab": {}, "topbinds": [],
#       "ichain": [], "inlinemap": {}, "preglobals": []}
# fctx = {"is_main", "scopes", "next", "max", "uval", "id"}
# scope = {"m": {}, "iter", "kids", "next"}

fn new_scope(it):
    return {"m": {}, "iter": it, "kids": [], "next": 0, "imm": {}}

fn new_fctx(is_main, nargs, fid):
    return {"is_main": is_main, "scopes": [new_scope(false)],
        "next": nargs, "max": nargs, "uval": {}, "id": fid}

fn new_c():
    return {"b": [], "gnames": {}, "fn_names": [], "fnmap": [],
        "fnorder": [], "loops": [], "fns": [], "op": build_op(), "nat": build_nat(),
        "fntab": {}, "topbinds": [], "ichain": [], "inlinemap": {},
        "preglobals": [], "ncall": [0], "tdepth": [0], "lines": [], "lfiles": []}

fn curfn(cc):
    return cc["fns"][cc["fns"].len - 1]

fn has_key(d, k):
    for x in d.keys():
        if x == k:
            return true
    return false

fn in_list(xs, v):
    for x in xs:
        if x == v:
            return true
    return false

fn fail(msg):
    raise "tcomp: " + msg

# ---------- emit ----------
fn c_emit(cc, v):
    cc["b"].append(v)

fn c_u16(cc, v):
    for b in u16_bytes(v):
        cc["b"].append(b)

fn c_i32(cc, v):
    for b in be_bytes(v, 4):
        cc["b"].append(b)

fn c_i64(cc, v):
    for b in be_bytes(v, 8):
        cc["b"].append(b)

fn c_f64(cc, x):
    for b in be_bytes(f64bits(x), 8):
        cc["b"].append(b)

fn c_str(cc, s):
    # byte-string: chép thẳng byte ra (UTF-8 pass-through)
    mut bs = []
    for i in range(s.len):
        bs.append(ord(s[i:i + 1]))
    for b in be_bytes(bs.len, 4):
        cc["b"].append(b)
    for b in bs:
        cc["b"].append(b)

fn c_pos(cc):
    return cc["b"].len

fn c_jmp(cc, op):
    c_emit(cc, cc["op"][op])
    at = cc["b"].len
    mut i = 0
    while i < 4:
        cc["b"].append(0)
        i = i + 1
    return at

fn c_patch(cc, at):
    bs = be_bytes(cc["b"].len - (at + 4), 4)
    mut i = 0
    while i < 4:
        cc["b"][at + i] = bs[i]
        i = i + 1

fn c_patch_to(cc, at, target):
    bs = be_bytes(target - (at + 4), 4)
    mut i = 0
    while i < 4:
        cc["b"][at + i] = bs[i]
        i = i + 1

fn c_gslot(cc, name):
    if not has_key(cc["gnames"], name):
        cc["gnames"][name] = cc["nat"].len + cc["gnames"].len
    return cc["gnames"][name]

# ---------- scope ----------
fn push_scope(cc, it):
    fx = curfn(cc)
    sc = fx["scopes"][fx["scopes"].len - 1]
    mut ch = nil
    if sc["next"] < sc["kids"].len:
        ch = sc["kids"][sc["next"]]
    else:
        ch = new_scope(it)
        sc["kids"].append(ch)
    sc["next"] = sc["next"] + 1
    fx["scopes"].append(ch)

fn pop_scope(cc):
    fx = curfn(cc)
    fx["scopes"].pop()

fn reset_walk(cc, fctx):
    rec_reset(fctx["scopes"][0])
    while fctx["scopes"].len > 1:
        fctx["scopes"].pop()

fn rec_reset(sc):
    sc["next"] = 0
    for k in sc["kids"]:
        rec_reset(k)

fn find_in(fx, name):
    mut i = fx["scopes"].len - 1
    while i >= 0:
        sc = fx["scopes"][i]
        if has_key(sc["m"], name):
            return sc["m"][name]
        i = i - 1
    return nil

fn find2_in(fx, name):
    mut i = fx["scopes"].len - 1
    while i >= 0:
        sc = fx["scopes"][i]
        if has_key(sc["m"], name):
            return [sc["m"][name], sc["iter"]]
        i = i - 1
    return nil

fn get_here_t(cc, name):
    sc = curfn(cc)["scopes"]
    sc = sc[sc.len - 1]
    if not has_key(sc["m"], name):
        fail("thiếu prescan " + name)
    return sc["m"][name]

fn use_upval(cc, name, defdepth):
    mut j = cc["fns"].len - 2
    while j > defdepth:
        hj = cc["fns"][j]
        if not has_key(hj["uval"], name):
            hj["uval"][name] = hj["uval"].len
        j = j - 1
    cur = cc["fns"][cc["fns"].len - 1]
    if not has_key(cur["uval"], name):
        cur["uval"][name] = cur["uval"].len
    return ["upval", cur["uval"][name]]

fn use_upval_main(cc, name):
    cur = cc["fns"][cc["fns"].len - 1]
    if not has_key(cur["uval"], name):
        cur["uval"][name] = cur["uval"].len
    return ["upval", cur["uval"][name]]

fn c_resolve(cc, name):
    fx = curfn(cc)
    if not fx["is_main"]:
        s = find_in(fx, name)
        if s != nil:
            return ["local", s]
        mut d = cc["fns"].len - 2
        while d >= 0:
            f2 = cc["fns"][d]
            if f2["is_main"]:
                break
            if find_in(f2, name) != nil:
                return use_upval(cc, name, d)
            d = d - 1
        main = cc["fns"][0]
        r = find2_in(main, name)
        if r != nil:
            if r[1]:
                return use_upval_main(cc, name)
            return ["global", r[0]]
    else:
        s = find_in(fx, name)
        if s != nil:
            return ["global", s]
    if in_list(cc["fn_names"], name) and has_key(cc["gnames"], name):
        return ["global", cc["gnames"][name]]
    if has_key(cc["nat"], name):
        return ["global", cc["nat"][name]]
    return nil

fn emit_load(cc, kind, ix):
    if kind == "local":
        c_emit(cc, cc["op"]["LOAD"])
    elif kind == "global":
        c_emit(cc, cc["op"]["LOADG"])
    else:
        c_emit(cc, cc["op"]["LOADU"])
    c_u16(cc, ix)

fn emit_store_slot(cc, ix):
    if curfn(cc)["is_main"]:
        c_emit(cc, cc["op"]["STOREG"])
    else:
        c_emit(cc, cc["op"]["STORE"])
    c_u16(cc, ix)

fn emit_load_slot(cc, ix):
    if curfn(cc)["is_main"]:
        c_emit(cc, cc["op"]["LOADG"])
    else:
        c_emit(cc, cc["op"]["LOAD"])
    c_u16(cc, ix)

fn emit_bind_slot(cc, ix):
    if curfn(cc)["is_main"]:
        c_emit(cc, cc["op"]["BINDG"])
    else:
        c_emit(cc, cc["op"]["BIND"])
    c_u16(cc, ix)

# tên gần nhất trong tầm nhìn là let/fn? (t.py: "x immutable (use mut)")
fn is_immutable(cc, name):
    mut i = cc["fns"].len - 1
    while i >= 0:
        fx = cc["fns"][i]
        mut j = fx["scopes"].len - 1
        while j >= 0:
            sc = fx["scopes"][j]
            if has_key(sc["m"], name):
                return has_key(sc["imm"], name)
            j = j - 1
        i = i - 1
    return in_list(cc["fn_names"], name)

fn mark_imm(cc, name):
    fx = curfn(cc)
    sc = fx["scopes"][fx["scopes"].len - 1]
    sc["imm"][name] = true

fn c_store(cc, name):
    if is_immutable(cc, name):
        fail(name + " immutable (use mut)")
    r = c_resolve(cc, name)
    if r == nil:
        if has_key(cc["nat"], name):
            fail("gán vào builtin " + name)
        ix = get_here_t(cc, name)
        if curfn(cc)["is_main"]:
            c_emit(cc, cc["op"]["STOREG"])
        else:
            c_emit(cc, cc["op"]["STORE"])
        c_u16(cc, ix)
        return nil
    kind = r[0]
    ix = r[1]
    if kind == "local":
        c_emit(cc, cc["op"]["STORE"])
    elif kind == "global":
        if has_key(cc["nat"], name) and ix == cc["nat"][name]:
            fail("gán vào builtin " + name)
        c_emit(cc, cc["op"]["STOREG"])
    else:
        c_emit(cc, cc["op"]["STOREU"])
    c_u16(cc, ix)

fn c_scoped(cc, stmts):
    push_scope(cc, false)
    c_block(cc, stmts)
    pop_scope(cc)

fn c_block(cc, stmts):
    for s in stmts:
        c_stmt(cc, s)

# ---------- expr ----------
fn c_expr(cc, e):
    k = e[0]
    if k == "num":
        v = num_val(e[1])
        if type(v) == "bool":
            c_emit(cc, cc["op"]["PUSH_BOOL"])
            if v:
                cc["b"].append(1)
            else:
                cc["b"].append(0)
        elif type(v) == "int":
            c_emit(cc, cc["op"]["PUSH_INT"])
            c_i64(cc, v)
        else:
            c_emit(cc, cc["op"]["PUSH_FLOAT"])
            c_f64(cc, v)
    elif k == "lit":
        m = lit_val(e[1])
        if m == nil:
            c_emit(cc, cc["op"]["PUSH_NIL"])
        else:
            c_emit(cc, cc["op"]["PUSH_BOOL"])
            if m:
                cc["b"].append(1)
            else:
                cc["b"].append(0)
    elif k == "str":
        c_emit(cc, cc["op"]["PUSH_STR"])
        c_str(cc, e[1])
    elif k == "fstr":
        ps = e[1]
        if ps.len == 0:
            c_emit(cc, cc["op"]["PUSH_STR"])
            for b in be_bytes(0, 4):
                cc["b"].append(b)
        else:
            mut first = true
            for p in ps:
                if p[0] == "s":
                    c_emit(cc, cc["op"]["PUSH_STR"])
                    c_str(cc, p[1])
                else:
                    c_emit(cc, cc["op"]["LOADG"])
                    c_u16(cc, cc["nat"]["str"])
                    c_expr(cc, p[1])
                    c_emit(cc, cc["op"]["CALL"])
                    cc["b"].append(1)
                if not first:
                    c_emit(cc, cc["op"]["ADD"])
                first = false
    elif k == "var":
        r = c_resolve(cc, e[1])
        if r == nil:
            fail("undef " + e[1])
        emit_load(cc, r[0], r[1])
    elif k == "bltfn":
        if e[1] != "print":
            fail("bare builtin " + e[1])
        c_emit(cc, cc["op"]["LOADG"])
        c_u16(cc, cc["nat"]["print"])
    elif k == "list":
        if e[1].len > 65535:
            fail("list quá lớn")
        for x in e[1]:
            c_expr(cc, x)
        c_emit(cc, cc["op"]["NEW_LIST"])
        c_u16(cc, e[1].len)
    elif k == "dict":
        ks = e[1]
        if ks.len > 65535:
            fail("dict quá lớn")
        mut i = 0
        while i < ks.len:
            c_expr(cc, ks[i])
            c_expr(cc, e[2][i])
            i = i + 1
        c_emit(cc, cc["op"]["NEW_DICT"])
        c_u16(cc, ks.len)
    elif k == "tensor":
        c_emit(cc, cc["op"]["LOADG"])
        c_u16(cc, cc["nat"]["tensor"])
        c_expr(cc, e[1])
        c_emit(cc, cc["op"]["CALL"])
        cc["b"].append(1)
    elif k == "bltin":
        if not has_key(cc["nat"], e[1]):
            fail("bltin " + e[1])
        if e[2].len > 255:
            fail("quá nhiều arg")
        c_emit(cc, cc["op"]["LOADG"])
        c_u16(cc, cc["nat"][e[1]])
        for x in e[2]:
            c_expr(cc, x)
        c_emit(cc, cc["op"]["CALL"])
        cc["b"].append(e[2].len)
    elif k == "un":
        c_expr(cc, e[2])
        if e[1] == "-":
            c_emit(cc, cc["op"]["NEG"])
        elif e[1] == "not":
            c_emit(cc, cc["op"]["NOT"])
        else:
            fail("unary " + e[1])
    elif k == "bin":
        op = e[1]
        if op == "and":
            c_expr(cc, e[2])
            c_emit(cc, cc["op"]["DUP"])
            f = c_jmp(cc, "JMPF")
            c_emit(cc, cc["op"]["POP"])
            c_expr(cc, e[3])
            end = c_jmp(cc, "JMP")
            c_patch(cc, f)
            c_patch(cc, end)
        elif op == "or":
            c_expr(cc, e[2])
            c_emit(cc, cc["op"]["DUP"])
            t = c_jmp(cc, "JMPT")
            c_emit(cc, cc["op"]["POP"])
            c_expr(cc, e[3])
            end = c_jmp(cc, "JMP")
            c_patch(cc, t)
            c_patch(cc, end)
        else:
            bn = binop_name(op)
            if bn == nil:
                fail("binop " + op)
            c_expr(cc, e[2])
            c_expr(cc, e[3])
            c_emit(cc, cc["op"][bn])
    elif k == "cond":
        c_expr(cc, e[1])
        f = c_jmp(cc, "JMPF")
        c_expr(cc, e[2])
        end = c_jmp(cc, "JMP")
        c_patch(cc, f)
        c_expr(cc, e[3])
        c_patch(cc, end)
    elif k == "idx":
        c_expr(cc, e[1])
        c_expr(cc, e[2])
        c_emit(cc, cc["op"]["IDX"])
    elif k == "slice":
        c_expr(cc, e[1])
        for part in [e[2], e[3], e[4]]:
            if part == nil:
                c_emit(cc, cc["op"]["PUSH_NIL"])
            else:
                c_expr(cc, part)
        c_emit(cc, cc["op"]["SLICE"])
    elif k == "attr":
        c_expr(cc, e[1])
        c_emit(cc, cc["op"]["ATTR"])
        c_str(cc, e[2])
    elif k == "call":
        c_expr_call(cc, e)
    else:
        fail("expr " + k)

fn c_expr_call(cc, e):
    mut inlined = false
    if e[1][0] == "var":
        key = in_key(cc, e)
        if has_key(cc["inlinemap"], key) and inlineable(cc, e[1][1], e[2].len) != nil:
            r = cc["inlinemap"][key]
            emit_inline(cc, e, r[1], r[2])
            inlined = true
    if inlined:
        return nil
    if e[1][0] == "attr":
        # obj.m(args): CALLM argc name (gọi method trực tiếp, inline cache)
        c_expr(cc, e[1][1])
        if e[2].len > 255:
            fail("quá nhiều arg")
        for x in e[2]:
            c_expr(cc, x)
        c_emit(cc, cc["op"]["CALLM"])
        cc["b"].append(e[2].len)
        c_str(cc, e[1][2])
        return nil
    c_expr(cc, e[1])
    if e[2].len > 255:
        fail("quá nhiều arg")
    for x in e[2]:
        c_expr(cc, x)
    c_emit(cc, cc["op"]["CALL"])
    cc["b"].append(e[2].len)
    return nil

# ---------- stmt ----------
# mốc dòng: ghi (pc, dòng, file) vào bảng, không sinh bytecode
fn c_line(cc, s):
    mut fi = -1
    mut q = 0
    while q < cc["lfiles"].len:
        if cc["lfiles"][q] == s[2]:
            fi = q
        q = q + 1
    if fi < 0:
        fi = cc["lfiles"].len
        cc["lfiles"].append(s[2])
    ent = [c_pos(cc), s[1], fi]
    ls = cc["lines"]
    if ls.len > 0:
        last = ls[ls.len - 1]
        if last[1] == ent[1] and last[2] == ent[2]:
            return nil
        if last[0] == ent[0]:
            ls[ls.len - 1] = ent
            return nil
    ls.append(ent)
    return nil

fn c_stmt(cc, s):
    k = s[0]
    main = curfn(cc)["is_main"]
    if k == "line":
        c_line(cc, s)
        return nil
    if k == "decl":
        c_expr(cc, s[3])
        if main:
            c_emit(cc, cc["op"]["STOREG"])
        else:
            c_emit(cc, cc["op"]["DECL"])
        c_u16(cc, get_here_t(cc, s[2]))
    elif k == "assign":
        t = s[1]
        if t[0] == "var":
            c_expr(cc, s[2])
            c_store(cc, t[1])
        elif t[0] == "idx":
            c_expr(cc, t[1])
            c_expr(cc, t[2])
            c_expr(cc, s[2])
            c_emit(cc, cc["op"]["SETIDX"])
        elif t[0] == "attr":
            c_expr(cc, t[1])
            c_expr(cc, s[2])
            c_emit(cc, cc["op"]["SETATTR"])
            c_str(cc, t[2])
        else:
            fail("bad assign target")
    elif k == "fn":
        c_fn(cc, s, main)
    elif k == "ret":
        if main:
            fail("return ngoài hàm")
        c_expr(cc, s[1])
        c_emit(cc, cc["op"]["RET"])
    elif k == "print":
        if s[1].len > 255:
            fail("print quá 255 arg")
        for x in s[1]:
            c_expr(cc, x)
        c_emit(cc, cc["op"]["PRINT"])
        cc["b"].append(s[1].len)
    elif k == "expr":
        c_expr(cc, s[1])
        c_emit(cc, cc["op"]["POP"])
    elif k == "if":
        c_expr(cc, s[1])
        f = c_jmp(cc, "JMPF")
        c_scoped(cc, s[2])
        end = c_jmp(cc, "JMP")
        c_patch(cc, f)
        if s[3]:
            if s[3].len == 1 and s[3][0][0] == "if":
                c_stmt(cc, s[3][0])
            else:
                c_scoped(cc, s[3])
        c_patch(cc, end)
    elif k == "while":
        loop = c_pos(cc)
        c_expr(cc, s[1])
        f = c_jmp(cc, "JMPF")
        cc["loops"].append({"breaks": [], "conts": [], "td": cc["tdepth"][0]})
        c_scoped(cc, s[2])
        patch_conts(cc)
        back = c_jmp(cc, "JMP")
        c_patch_to(cc, back, loop)
        c_patch(cc, f)
        patch_breaks(cc)
        cc["loops"].pop()
    elif k == "for":
        c_for(cc, s[1], s[2], s[3])
    elif k == "break":
        if cc["loops"].len == 0:
            fail("break ngoài vòng lặp")
        l = cc["loops"][cc["loops"].len - 1]
        emit_trypops(cc, l)
        l["breaks"].append(c_jmp(cc, "JMP"))
    elif k == "continue":
        if cc["loops"].len == 0:
            fail("continue ngoài vòng lặp")
        l = cc["loops"][cc["loops"].len - 1]
        emit_trypops(cc, l)
        l["conts"].append(c_jmp(cc, "JMP"))
    elif k == "raise":
        c_expr(cc, s[1])
        c_emit(cc, cc["op"]["RAISE"])
    elif k == "assert":
        c_expr(cc, s[1])
        ok = c_jmp(cc, "JMPT")
        if s[2] == nil:
            c_emit(cc, cc["op"]["PUSH_STR"])
            c_str(cc, "assert failed")
        else:
            c_emit(cc, cc["op"]["PUSH_STR"])
            c_str(cc, "assert failed: ")
            c_expr(cc, s[2])
            c_emit(cc, cc["op"]["ADD"])
        c_emit(cc, cc["op"]["RAISE"])
        c_patch(cc, ok)
    elif k == "try":
        c_try(cc, s[1], s[2])
    elif k == "import" or k == "fromimport":
        fail(k + " trực tiếp chưa hỗ trợ - dùng -b để bundle")
    else:
        fail("stmt " + k)

# break/continue nhảy ra khỏi try đang mở trong vòng lặp: gỡ handler trước
fn emit_trypops(cc, l):
    mut i = l["td"]
    while i < cc["tdepth"][0]:
        c_emit(cc, cc["op"]["TRY_POP"])
        i = i + 1

fn patch_conts(cc):
    l = cc["loops"][cc["loops"].len - 1]
    for p in l["conts"]:
        c_patch(cc, p)

fn patch_breaks(cc):
    l = cc["loops"][cc["loops"].len - 1]
    for p in l["breaks"]:
        c_patch(cc, p)

# ---------- fn ----------
fn c_fn(cc, s, main):
    name = s[1]
    args = s[2]
    body = s[3]
    if args.len > 255:
        fail("quá nhiều param")
    slot = get_here_t(cc, name)
    over = c_jmp(cc, "JMP")
    entry = c_pos(cc)
    fctx = cc["fnmap"][fn_index(cc, s)]
    mut i = 0
    while i < args.len:
        a = args[i]
        if fctx["scopes"][0]["m"][a] != i:
            fail("lỗi nội bộ param " + a)
        i = i + 1
    cc["fns"].append(fctx)
    c_block(cc, body)
    c_emit(cc, cc["op"]["PUSH_NIL"])
    c_emit(cc, cc["op"]["RET"])
    cc["fns"].pop()
    c_patch(cc, over)
    c_emit(cc, cc["op"]["MKFN"])
    c_i32(cc, entry)
    cc["b"].append(args.len)
    c_u16(cc, fctx["max"])
    c_u16(cc, fctx["uval"].len)
    for nm in fctx["uval"].keys():
        emit_cap(cc, nm)
    if main:
        c_emit(cc, cc["op"]["STOREG"])
    else:
        c_emit(cc, cc["op"]["DECL"])
    c_u16(cc, slot)

fn fn_index(cc, s):
    mut i = 0
    while i < cc["fnorder"].len:
        if cc["fnorder"][i] == s:
            return i
        i = i + 1
    fail("lỗi nội bộ fnmap")

fn emit_cap(cc, nm):
    enc = curfn(cc)
    if not enc["is_main"]:
        s = find_in(enc, nm)
        if s != nil:
            c_emit(cc, cc["op"]["CAPLOCAL"])
            c_u16(cc, s)
            return nil
        if has_key(enc["uval"], nm):
            c_emit(cc, cc["op"]["CAPUP"])
            c_u16(cc, enc["uval"][nm])
            return nil
        fail("capture " + nm + " trước khi bind")
    r = find2_in(enc, nm)
    if r == nil or not r[1]:
        fail("capture " + nm + " trước khi bind")
    c_emit(cc, cc["op"]["CAPG"])
    c_u16(cc, r[0])
    return nil

# ---------- for ----------
fn is_range_for(it):
    if it[0] == "bltin" and it[1] == "range" and it[2].len >= 1 and it[2].len <= 2:
        return it[2]
    return nil

fn emit_intcall(cc, e):
    c_emit(cc, cc["op"]["LOADG"])
    c_u16(cc, cc["nat"]["int"])
    c_expr(cc, e)
    c_emit(cc, cc["op"]["CALL"])
    cc["b"].append(1)

fn c_for(cc, v, it, body):
    ra = is_range_for(it)
    if ra == nil:
        c_for_general(cc, v, it, body)
    else:
        c_for_range(cc, v, ra, body)

fn c_for_range(cc, v, ra, body):
    push_scope(cc, true)
    v_s = get_here_t(cc, v)
    hi_s = get_here_t(cc, "$hi")
    if ra.len == 1:
        c_emit(cc, cc["op"]["PUSH_INT"])
        c_i64(cc, 0)
    else:
        emit_intcall(cc, ra[0])
    emit_bind_slot(cc, v_s)
    emit_intcall(cc, ra[ra.len - 1])
    emit_bind_slot(cc, hi_s)
    loop = c_pos(cc)
    emit_load_slot(cc, v_s)
    emit_load_slot(cc, hi_s)
    c_emit(cc, cc["op"]["LT"])
    f = c_jmp(cc, "JMPF")
    cc["loops"].append({"breaks": [], "conts": [], "td": cc["tdepth"][0]})
    push_scope(cc, true)
    c_block(cc, body)
    pop_scope(cc)
    patch_conts(cc)
    emit_load_slot(cc, v_s)
    c_emit(cc, cc["op"]["PUSH_INT"])
    c_i64(cc, 1)
    c_emit(cc, cc["op"]["ADD"])
    emit_bind_slot(cc, v_s)
    back = c_jmp(cc, "JMP")
    c_patch_to(cc, back, loop)
    c_patch(cc, f)
    patch_breaks(cc)
    cc["loops"].pop()
    pop_scope(cc)

fn c_for_general(cc, v, it, body):
    push_scope(cc, true)
    it_s = get_here_t(cc, "$it")
    i_s = get_here_t(cc, "$i")
    v_s = get_here_t(cc, v)
    c_expr(cc, it)
    c_emit(cc, cc["op"]["FOR_PREP"])
    emit_store_slot(cc, it_s)
    c_emit(cc, cc["op"]["PUSH_INT"])
    c_i64(cc, 0)
    emit_store_slot(cc, i_s)
    loop = c_pos(cc)
    emit_load_slot(cc, i_s)
    emit_load_slot(cc, it_s)
    c_emit(cc, cc["op"]["LEN"])
    c_emit(cc, cc["op"]["LT"])
    f = c_jmp(cc, "JMPF")
    cc["loops"].append({"breaks": [], "conts": [], "td": cc["tdepth"][0]})
    emit_load_slot(cc, it_s)
    emit_load_slot(cc, i_s)
    c_emit(cc, cc["op"]["IDX"])
    emit_bind_slot(cc, v_s)
    push_scope(cc, true)
    c_block(cc, body)
    pop_scope(cc)
    patch_conts(cc)
    emit_load_slot(cc, i_s)
    c_emit(cc, cc["op"]["PUSH_INT"])
    c_i64(cc, 1)
    c_emit(cc, cc["op"]["ADD"])
    emit_store_slot(cc, i_s)
    back = c_jmp(cc, "JMP")
    c_patch_to(cc, back, loop)
    c_patch(cc, f)
    patch_breaks(cc)
    cc["loops"].pop()
    pop_scope(cc)

# ---------- try ----------
fn has_wild(s):
    return str_contains(s, "*") or str_contains(s, "?")

fn str_contains(hay, needle):
    if needle.len > hay.len:
        return false
    mut i = 0
    while i <= hay.len - needle.len:
        if hay[i:i + needle.len] == needle:
            return true
        i = i + 1
    return false

fn has_bare(handlers):
    for h in handlers:
        if h[0] == nil:
            return true
    return false

fn c_try(cc, body, handlers):
    push_h = c_jmp(cc, "TRY_PUSH")
    cc["tdepth"][0] = cc["tdepth"][0] + 1
    c_scoped(cc, body)
    cc["tdepth"][0] = cc["tdepth"][0] - 1
    c_emit(cc, cc["op"]["TRY_POP"])
    norm = c_jmp(cc, "JMP")
    c_patch(cc, push_h)
    push_scope(cc, false)
    es = get_here_t(cc, "err")
    mut dones = [norm]
    for h in handlers:
        nm = h[0]
        hb = h[1]
        if nm == nil:
            emit_store_slot(cc, es)
            c_scoped(cc, hb)
            dones.append(c_jmp(cc, "JMP"))
            break
        c_emit(cc, cc["op"]["DUP"])
        c_emit(cc, cc["op"]["PUSH_STR"])
        c_str(cc, nm)
        if has_wild(nm):
            c_emit(cc, cc["op"]["MATCHGLOB"])
        else:
            c_emit(cc, cc["op"]["CONTAINS"])
        nx = c_jmp(cc, "JMPF")
        emit_store_slot(cc, es)
        c_scoped(cc, hb)
        dones.append(c_jmp(cc, "JMP"))
        c_patch(cc, nx)
    if not has_bare(handlers):
        c_emit(cc, cc["op"]["RAISE"])
    pop_scope(cc)
    for d in dones:
        c_patch(cc, d)

# ---------- inliner ----------
fn expr_children(e):
    k = e[0]
    if k == "num" or k == "str" or k == "lit" or k == "var" or k == "bltfn":
        return []
    if k == "tensor" or k == "attr":
        return [e[1]]
    if k == "un":
        return [e[2]]
    if k == "idx":
        return [e[1], e[2]]
    if k == "bin":
        return [e[2], e[3]]
    if k == "bltin":
        return e[2]
    if k == "list":
        return e[1]
    if k == "call":
        mut r = [e[1]]
        for x in e[2]:
            r.append(x)
        return r
    if k == "dict":
        mut r = []
        for x in e[1]:
            r.append(x)
        for x in e[2]:
            r.append(x)
        return r
    if k == "slice":
        mut r = [e[1]]
        for p in [e[2], e[3], e[4]]:
            if p != nil:
                r.append(p)
        return r
    if k == "fstr":
        mut r = []
        for p in e[1]:
            if p[0] == "e":
                r.append(p[1])
        return r
    if k == "cond":
        return [e[1], e[2], e[3]]
    return nil

fn count_nodes(e):
    kids = expr_children(e)
    if kids == nil:
        return 1000000
    mut n = 1
    for c in kids:
        n = n + count_nodes(c)
    return n

fn fv_into(acc, e, bound):
    if e[0] == "var":
        if not in_list(bound, e[1]) and not in_list(acc, e[1]):
            acc.append(e[1])
        return nil
    kids = expr_children(e)
    if kids == nil:
        return nil
    for c in kids:
        fv_into(acc, c, bound)
    return nil

fn free_vars(e, bound):
    mut acc = []
    fv_into(acc, e, bound)
    return acc

# Khoá inline: theo TỪNG node call (gắn số định danh vào cuối node) + hàm
# đang dịch. Theo tên hàm thì add(1, add(2, 3)) dùng chung temp -> lệnh
# trong ghi đè lệnh ngoài (sai kết quả).
fn call_id(cc, e):
    if e.len < 4:
        cc["ncall"][0] = cc["ncall"][0] + 1
        e.append(cc["ncall"][0])
    return e[3]

fn in_key(cc, e):
    return str(call_id(cc, e)) + "/" + str(curfn(cc)["id"])

fn inlineable(cc, fname, nargs):
    if not has_key(cc["fntab"], fname):
        return nil
    ent = cc["fntab"][fname]
    if ent[0].len != nargs:
        return nil
    if in_list(cc["ichain"], fname):
        return nil
    if count_nodes(ent[1]) > 40:
        return nil
    for nm in free_vars(ent[1], ent[0]):
        if in_list(cc["topbinds"], nm):
            continue
        if has_key(cc["nat"], nm):
            continue
        return nil
    return ent

fn pre_temp(cc):
    fx = curfn(cc)
    sc = fx["scopes"][fx["scopes"].len - 1]
    nm = "$in" + str(cc["inlinemap"].len) + "$" + str(fx["next"])
    mut ix = 0
    if fx["is_main"]:
        ix = c_gslot(cc, nm + "@" + str(cc["gnames"].len))
    else:
        ix = fx["next"]
        fx["next"] = fx["next"] + 1
        if fx["next"] > fx["max"]:
            fx["max"] = fx["next"]
    sc["m"][nm] = ix
    return [nm, ix]

fn emit_inline(cc, e, ent, temps):
    params = ent[0]
    body = ent[1]
    mut i = 0
    while i < e[2].len:
        c_expr(cc, e[2][i])
        emit_store_slot(cc, temps[i][1])
        i = i + 1
    psc = new_scope(false)
    i = 0
    while i < params.len:
        psc["m"][params[i]] = temps[i][1]
        i = i + 1
    curfn(cc)["scopes"].append(psc)
    cc["ichain"].append(ent[2])
    c_expr(cc, body)
    cc["ichain"].pop()
    curfn(cc)["scopes"].pop()

# ---------- prescan ----------
fn pre_alloc(cc, name):
    fx = curfn(cc)
    sc = fx["scopes"]
    sc = sc[sc.len - 1]
    if has_key(sc["m"], name):
        if in_list(cc["preglobals"], name):
            return sc["m"][name]
        fail("redeclare " + name)
    mut ix = 0
    if fx["is_main"]:
        ix = c_gslot(cc, name + "@" + str(cc["gnames"].len))
    else:
        ix = fx["next"]
        fx["next"] = fx["next"] + 1
        if fx["next"] > fx["max"]:
            fx["max"] = fx["next"]
    sc["m"][name] = ix
    return ix

fn pre_found(cc, name):
    mut i = cc["fns"].len - 1
    while i >= 0:
        if find_in(cc["fns"][i], name) != nil:
            return true
        i = i - 1
    if in_list(cc["fn_names"], name):
        return true
    return false

# pha 1: cấp slot cho mọi biến GLOBAL top-level (let/mut + gán trực tiếp)
# để hàm định nghĩa trước vẫn thấy biến khai báo sau
fn prescan_globals(cc, stmts):
    for s in stmts:
        k = s[0]
        if k == "decl":
            cc["preglobals"].append(s[2])
            pre_alloc(cc, s[2])
        elif k == "assign" and s[1][0] == "var":
            cc["preglobals"].append(s[1][1])
            pre_alloc(cc, s[1][1])

fn prescan(cc, stmts):
    for s in stmts:
        prescan_stmt(cc, s)

fn prescan_scope(cc, stmts, it):
    push_scope(cc, it)
    prescan(cc, stmts)
    pop_scope(cc)

fn prescan_expr(cc, e):
    k = e[0]
    if k == "call" and e[1][0] == "var":
        fname = e[1][1]
        ent = inlineable(cc, fname, e[2].len)
        if ent != nil:
            key = in_key(cc, e)
            if not has_key(cc["inlinemap"], key):
                mut temps = []
                for a in e[2]:
                    temps.append(pre_temp(cc))
                cc["inlinemap"][key] = [fname, ent, temps]
                cc["ichain"].append(fname)
                prescan_expr(cc, ent[1])
                cc["ichain"].pop()
        for x in e[2]:
            prescan_expr(cc, x)
        if e[1][0] != "var":
            prescan_expr(cc, e[1])
        return nil
    if k == "num" or k == "str" or k == "lit" or k == "var" or k == "bltfn":
        return nil
    if k == "tensor":
        prescan_expr(cc, e[1])
    elif k == "un":
        prescan_expr(cc, e[2])
    elif k == "bltin":
        for x in e[2]:
            prescan_expr(cc, x)
    elif k == "bin":
        prescan_expr(cc, e[2])
        prescan_expr(cc, e[3])
    elif k == "idx":
        prescan_expr(cc, e[1])
        prescan_expr(cc, e[2])
    elif k == "attr":
        prescan_expr(cc, e[1])
    elif k == "slice":
        prescan_expr(cc, e[1])
        for part in [e[2], e[3], e[4]]:
            if part != nil:
                prescan_expr(cc, part)
    elif k == "list":
        for x in e[1]:
            prescan_expr(cc, x)
    elif k == "dict":
        for x in e[1]:
            prescan_expr(cc, x)
        for x in e[2]:
            prescan_expr(cc, x)
    elif k == "call":
        prescan_expr(cc, e[1])
        for x in e[2]:
            prescan_expr(cc, x)
    elif k == "fstr":
        for p in e[1]:
            if p[0] == "e":
                prescan_expr(cc, p[1])
    elif k == "cond":
        prescan_expr(cc, e[1])
        prescan_expr(cc, e[2])
        prescan_expr(cc, e[3])
    return nil

fn prescan_stmt(cc, s):
    k = s[0]
    if k == "decl":
        prescan_expr(cc, s[3])
        pre_alloc(cc, s[2])
        if s[1] == "let":
            mark_imm(cc, s[2])
    elif k == "assign" and s[1][0] == "var":
        prescan_expr(cc, s[2])
        if not pre_found(cc, s[1][1]):
            pre_alloc(cc, s[1][1])
    elif k == "assign":
        prescan_expr(cc, s[1][1])
        if s[1][0] == "idx":
            prescan_expr(cc, s[1][2])
        prescan_expr(cc, s[2])
    elif k == "fn":
        if s[2].len > 255:
            fail("quá nhiều param")
        pre_alloc(cc, s[1])
        mark_imm(cc, s[1])
        fctx = new_fctx(false, s[2].len, cc["fnmap"].len + 1)
        mut i = 0
        while i < s[2].len:
            a = s[2][i]
            if has_key(fctx["scopes"][0]["m"], a):
                fail("trùng param " + a)
            fctx["scopes"][0]["m"][a] = i
            i = i + 1
        cc["fnmap"].append(fctx)
        cc["fnorder"].append(s)
        cc["fns"].append(fctx)
        prescan(cc, s[3])
        cc["fns"].pop()
    elif k == "for":
        prescan_expr(cc, s[2])
        push_scope(cc, true)
        if is_range_for(s[2]) == nil:
            for n in ["$it", "$i", s[1]]:
                pre_alloc(cc, n)
        else:
            for n in [s[1], "$hi"]:
                pre_alloc(cc, n)
        push_scope(cc, true)
        prescan(cc, s[3])
        pop_scope(cc)
        pop_scope(cc)
    elif k == "while":
        prescan_expr(cc, s[1])
        prescan_scope(cc, s[2], false)
    elif k == "if":
        prescan_expr(cc, s[1])
        prescan_scope(cc, s[2], false)
        if s[3]:
            if s[3].len == 1 and s[3][0][0] == "if":
                prescan(cc, [s[3][0]])
            else:
                prescan_scope(cc, s[3], false)
    elif k == "try":
        prescan_scope(cc, s[1], false)
        for h in s[2]:
            push_scope(cc, false)
            pre_alloc(cc, "err")
            prescan(cc, h[1])
            pop_scope(cc)
    elif k == "print":
        for x in s[1]:
            prescan_expr(cc, x)
    elif k == "expr" or k == "ret" or k == "raise":
        prescan_expr(cc, s[1])
    elif k == "assert":
        prescan_expr(cc, s[1])
        if s[2] != nil:
            prescan_expr(cc, s[2])
    return nil

# ---------- bundler ----------
fn join_path(parts):
    mut s = ""
    mut i = 0
    while i < parts.len:
        if i > 0:
            s = s + "/"
        s = s + parts[i]
        i = i + 1
    return s

fn base_name(rel):
    parts = rel.split("/")
    return parts[parts.len - 1]

fn leaf_base(leaf):
    if leaf.len > 2 and leaf[leaf.len - 2:leaf.len] == ".t":
        return leaf[0:leaf.len - 2]
    return leaf

fn dir_name(p):
    mut i = p.len - 1
    while i >= 0:
        # cả "/" lẫn "\" (đường dẫn Windows)
        if p[i:i + 1] == "/" or p[i:i + 1] == "\\":
            if i == 0:
                return "/"
            return p[0:i]
        i = i - 1
    return "."

fn file_exists(p):
    f = open(p, "r")
    close(f)
    return true

fn read_file(p):
    f = open(p, "r")
    s = read(f)
    close(f)
    return s

fn find_mod(dotted, refdir):
    rel = join_path(dotted.split(".")) + ".t"
    leaf = base_name(rel)
    stem = leaf_base(leaf)
    tp = "tpkgs/" + stem + "/" + stem + ".t"
    cands = [refdir + "/" + rel, refdir + "/std/" + leaf, refdir + "/self/" + leaf,
        refdir + "/tools/" + leaf, refdir + "/" + tp,
        rel, "std/" + leaf, "self/" + leaf, "tools/" + leaf, tp,
        dir_name(refdir) + "/" + rel, dir_name(refdir) + "/std/" + leaf,
        dir_name(refdir) + "/self/" + leaf, dir_name(refdir) + "/tools/" + leaf,
        dir_name(refdir) + "/" + tp]
    for p in cands:
        try:
            if file_exists(p):
                return p
        except:
            continue
    fail("no module " + dotted)

fn fn_bounds(stmts):
    mut b = []
    for s in stmts:
        if s[0] == "decl":
            if not in_list(b, s[2]):
                b.append(s[2])
        elif s[0] == "fn":
            if not in_list(b, s[1]):
                b.append(s[1])
        elif s[0] == "assign" and s[1][0] == "var":
            if not in_list(b, s[1][1]):
                b.append(s[1][1])
        elif s[0] == "for":
            if not in_list(b, s[1]):
                b.append(s[1])
    return b

fn collect_bounds(stmts, cur, fbound, up):
    for s in stmts:
        k = s[0]
        if k == "fn":
            key = s[1]
            if cur != nil:
                key = cur + "@" + s[1]
            elif up != "":
                key = up + "@" + s[1]
            mut b = []
            for a in s[2]:
                b.append(a)
            for x in fn_bounds(s[3]):
                b.append(x)
            fbound[key] = b
            collect_bounds(s[3], key, fbound, up)
            continue
        if cur != nil:
            if k == "decl":
                if not in_list(fbound[cur], s[2]):
                    fbound[cur].append(s[2])
            elif k == "assign" and s[1][0] == "var":
                if not in_list(fbound[cur], s[1][1]):
                    fbound[cur].append(s[1][1])
            elif k == "for":
                if not in_list(fbound[cur], s[1]):
                    fbound[cur].append(s[1])
        if k == "if":
            collect_bounds(s[2], cur, fbound, up)
            if s[3]:
                if s[3].len == 1 and s[3][0][0] == "if":
                    collect_bounds([s[3][0]], cur, fbound, up)
                else:
                    collect_bounds(s[3], cur, fbound, up)
        elif k == "while":
            collect_bounds(s[2], cur, fbound, up)
        elif k == "for":
            collect_bounds(s[3], cur, fbound, up)
        elif k == "try":
            collect_bounds(s[1], cur, fbound, up)
            for h in s[2]:
                collect_bounds(h[1], cur, fbound, up)

fn unit_prefix(modkey):
    if modkey == "":
        return ""
    mut out = ""
    mut first = true
    for p in modkey.split("."):
        if first:
            out = p
            first = false
        else:
            out = out + "_" + p
    return out + "__"

fn alias_key(modkey, name):
    return modkey + "." + name

fn bundle_load(entry_path, units, done, busy, out):
    for s in units:
        if s[0] == "":
            continue
        if in_list(done, s[0]):
            continue
        if in_list(busy, s[0]):
            fail("circular import " + s[0])
        busy.append(s[0])
        path = s[1]
        prog = parse_file(path)
        refdir = dir_name(path)
        mut deps = []
        for st in prog[1]:
            if st[0] == "import" or st[0] == "fromimport":
                deps.append([st[1], find_mod(st[1], refdir)])
        bundle_load(entry_path, deps, done, busy, out)
        out.append([s[0], prog[1]])
        done.append(s[0])
        busy.pop()
    return nil

fn bundle(entry_path):
    prog = parse_file(entry_path)
    mut units = []
    done = []
    busy = []
    mut deps = []
    refdir = dir_name(entry_path)
    for st in prog[1]:
        if st[0] == "import" or st[0] == "fromimport":
            deps.append([st[1], find_mod(st[1], refdir)])
    bundle_load(entry_path, deps, done, busy, units)
    units.append(["", prog[1]])
    # am: "unit.tên" -> ["mod", dotted] | ["from", dotted, tên]
    mut am = {}
    for u in units:
        for s in u[1]:
            if s[0] == "import":
                a = s[2]
                if a == nil:
                    a = leaf_base(join_path(s[1].split(".")) + ".t")
                am[alias_key(u[0], a)] = ["mod", s[1]]
            elif s[0] == "fromimport":
                for n in s[2]:
                    am[alias_key(u[0], n)] = ["from", s[1], n]
    # xbind: tên global của unit phụ (cross-ref kiểu parse_stmt)
    mut xb = {}
    for u in units:
        if u[0] == "":
            continue
        pre = unit_prefix(u[0])
        for b in fn_bounds(u[1]):
            if not has_key(xb, b):
                xb[b] = pre + b
    mut out = []
    for u in units:
        is_entry = u[0] == ""
        mut ren = {}
        if not is_entry:
            pre = unit_prefix(u[0])
            for b in fn_bounds(u[1]):
                ren[b] = pre + b
        fbound = {}
        collect_bounds(u[1], nil, fbound, u[0])
        for s in u[1]:
            if s[0] == "import":
                continue
            elif s[0] == "fromimport":
                for n in s[2]:
                    out.append(["decl", "let", n, ["var", unit_prefix(s[1]) + n]])
            else:
                out.append(rw_stmt(s, u[0], ren, am, xb, fbound, is_entry, nil))
    return ["prog", out]

fn rw_var(name, up, lren, am, xb):
    if has_key(lren, name):
        return ["var", lren[name]]
    key = alias_key(up, name)
    if has_key(am, key):
        e = am[key]
        if e[0] == "from":
            return ["var", unit_prefix(e[1]) + e[2]]
        return ["var", e[1]]
    if up != "" and has_key(xb, name):
        return ["var", xb[name]]
    return ["var", name]

fn rw_expr(e, up, lren, am, xb):
    k = e[0]
    if k == "var":
        return rw_var(e[1], up, lren, am, xb)
    if k == "num" or k == "str" or k == "lit" or k == "bltfn":
        return e
    if k == "fstr":
        mut parts = []
        for p in e[1]:
            if p[0] == "s":
                parts.append(["s", p[1]])
            else:
                parts.append(["e", rw_expr(p[1], up, lren, am, xb)])
        return ["fstr", parts]
    if k == "list":
        mut r = []
        for x in e[1]:
            r.append(rw_expr(x, up, lren, am, xb))
        return ["list", r]
    if k == "dict":
        mut ks = []
        mut vs = []
        for x in e[1]:
            ks.append(rw_expr(x, up, lren, am, xb))
        for x in e[2]:
            vs.append(rw_expr(x, up, lren, am, xb))
        return ["dict", ks, vs]
    if k == "tensor":
        return ["tensor", rw_expr(e[1], up, lren, am, xb)]
    if k == "bltin":
        mut r = []
        for x in e[2]:
            r.append(rw_expr(x, up, lren, am, xb))
        return ["bltin", e[1], r]
    if k == "un":
        return ["un", e[1], rw_expr(e[2], up, lren, am, xb)]
    if k == "bin":
        return ["bin", e[1], rw_expr(e[2], up, lren, am, xb),
            rw_expr(e[3], up, lren, am, xb)]
    if k == "idx":
        return ["idx", rw_expr(e[1], up, lren, am, xb),
            rw_expr(e[2], up, lren, am, xb)]
    if k == "slice":
        mut r = ["slice", rw_expr(e[1], up, lren, am, xb)]
        for p in [e[2], e[3], e[4]]:
            if p == nil:
                r.append(nil)
            else:
                r.append(rw_expr(p, up, lren, am, xb))
        return r
    if k == "attr":
        base = e[1]
        if base[0] == "var" and not has_key(lren, base[1]):
            key = alias_key(up, base[1])
            if has_key(am, key):
                e2 = am[key]
                if e2[0] == "mod":
                    return ["var", unit_prefix(e2[1]) + e[2]]
                fail(base[1] + " là fromimport, không attr")
        return ["attr", rw_expr(base, up, lren, am, xb), e[2]]
    if k == "call":
        mut args = []
        for x in e[2]:
            args.append(rw_expr(x, up, lren, am, xb))
        return ["call", rw_expr(e[1], up, lren, am, xb), args]
    if k == "cond":
        return ["cond", rw_expr(e[1], up, lren, am, xb), rw_expr(e[2], up, lren, am, xb),
            rw_expr(e[3], up, lren, am, xb)]
    fail("bundler expr " + k)

fn local_ren(base, names, pre):
    # gộp ren của hàm cha (upvalue) với binds của hàm này
    mut d = {}
    for k in base.keys():
        d[k] = base[k]
    for n in names:
        d[n] = pre + n
    return d

fn rw_if_tail(b, up, lren, am, xb, fbound, is_entry, cur):
    if b == nil:
        return nil
    mut out = []
    for s in b:
        out.append(rw_stmt(s, up, lren, am, xb, fbound, is_entry, cur))
    return out

fn rw_stmt(s, up, lren, am, xb, fbound, is_entry, cur):
    k = s[0]
    pre = unit_prefix(up)
    if k == "decl":
        nm = s[2]
        if has_key(lren, nm):
            nm = lren[nm]
        return ["decl", s[1], nm, rw_expr(s[3], up, lren, am, xb)]
    if k == "fn":
        nm = s[1]
        key = nm
        if cur != nil:
            key = cur + "@" + nm
        elif up != "":
            key = up + "@" + nm
        mut myren = lren
        if up != "" and has_key(fbound, key):
            myren = local_ren(lren, fbound[key], pre)
        nm2 = nm
        if up != "":
            nm2 = pre + nm
        mut body = []
        for x in s[3]:
            body.append(rw_stmt(x, up, myren, am, xb, fbound, is_entry, key))
        mut args = []
        for a in s[2]:
            if has_key(myren, a):
                args.append(myren[a])
            else:
                args.append(a)
        return ["fn", nm2, args, body]
    if k == "assign":
        t = s[1]
        if t[0] == "var":
            return ["assign", rw_var(t[1], up, lren, am, xb),
                rw_expr(s[2], up, lren, am, xb)]
        if t[0] == "idx":
            return ["assign", ["idx", rw_expr(t[1], up, lren, am, xb),
                rw_expr(t[2], up, lren, am, xb)],
                rw_expr(s[2], up, lren, am, xb)]
        if t[0] == "attr":
            return ["assign", rw_expr(t, up, lren, am, xb), rw_expr(s[2], up, lren, am, xb)]
        return s
    if k == "ret":
        return ["ret", rw_expr(s[1], up, lren, am, xb)]
    if k == "print":
        mut r = []
        for x in s[1]:
            r.append(rw_expr(x, up, lren, am, xb))
        return ["print", r]
    if k == "expr":
        return ["expr", rw_expr(s[1], up, lren, am, xb)]
    if k == "if":
        mut b2 = nil
        if s[3]:
            if s[3].len == 1 and s[3][0][0] == "if":
                b2 = [rw_stmt(s[3][0], up, lren, am, xb, fbound, is_entry, cur)]
            else:
                b2 = rw_if_tail(s[3], up, lren, am, xb, fbound, is_entry, cur)
        mut a2 = []
        for x in s[2]:
            a2.append(rw_stmt(x, up, lren, am, xb, fbound, is_entry, cur))
        return ["if", rw_expr(s[1], up, lren, am, xb), a2, b2]
    if k == "for":
        mut body = []
        for x in s[3]:
            body.append(rw_stmt(x, up, lren, am, xb, fbound, is_entry, cur))
        v = s[1]
        if has_key(lren, v):
            v = lren[v]
        return ["for", v, rw_expr(s[2], up, lren, am, xb), body]
    if k == "while":
        mut body = []
        for x in s[2]:
            body.append(rw_stmt(x, up, lren, am, xb, fbound, is_entry, cur))
        return ["while", rw_expr(s[1], up, lren, am, xb), body]
    if k == "break" or k == "continue":
        return s
    if k == "try":
        mut body = []
        for x in s[1]:
            body.append(rw_stmt(x, up, lren, am, xb, fbound, is_entry, cur))
        mut hs = []
        for h in s[2]:
            mut hb = []
            for x in h[1]:
                hb.append(rw_stmt(x, up, lren, am, xb, fbound, is_entry, cur))
            hs.append([h[0], hb])
        return ["try", body, hs]
    if k == "raise":
        return ["raise", rw_expr(s[1], up, lren, am, xb)]
    if k == "assert":
        mut msg = nil
        if s[2] != nil:
            msg = rw_expr(s[2], up, lren, am, xb)
        return ["assert", rw_expr(s[1], up, lren, am, xb), msg]
    return s

# ---------- peephole ----------
fn op_size(cc, code, pc):
    o = code[pc]
    op = cc["op"]
    mut sz = 1
    if o == op["PUSH_INT"] or o == op["PUSH_FLOAT"]:
        sz = 9
    elif o == op["PUSH_STR"] or o == op["ATTR"] or o == op["SETATTR"]:
        sz = 5 + rd_u32(code, pc + 1)
    elif o == op["CALLM"]:
        sz = 6 + rd_u32(code, pc + 2)
    elif o == op["MKFN"]:
        sz = 10
    elif o == op["JMP"] or o == op["JMPF"] or o == op["JMPT"] or o == op["TRY_PUSH"]:
        sz = 5
    elif o == op["PUSH_BOOL"] or o == op["CALL"] or o == op["PRINT"] or o == op["RANGE"]:
        sz = 2
    elif o == op["NEW_LIST"] or o == op["NEW_TENSOR"] or o == op["NEW_DICT"] or o == op["LOAD"] or o == op["STORE"] or o == op["DECL"] or o == op["LOADG"] or o == op["STOREG"] or o == op["DECLG"] or o == op["LOADU"] or o == op["STOREU"] or o == op["BIND"] or o == op["BINDG"] or o == op["CAPLOCAL"] or o == op["CAPUP"] or o == op["CAPG"] or o == op["INC_L"] or o == op["INC_G"] or o == op["DEC_L"] or o == op["DEC_G"] or o == op["INCB_L"] or o == op["INCB_G"]:
        sz = 3
    elif o == op["ADDI_L"] or o == op["ADDI_G"]:
        sz = 11
    elif o == op["ADDS_L"] or o == op["ADDS_G"] or o == op["SUBS_L"] or o == op["SUBS_G"]:
        sz = 5
    return sz

fn peephole(cc):
    code = cc["b"]
    op = cc["op"]
    n = code.len
    mut prot = []
    mut i = 0
    while i < n:
        prot.append(0)
        i = i + 1
    prot[0] = 1
    mut ops = []
    mut pc = 0
    while pc < n:
        sz = op_size(cc, code, pc)
        ops.append([pc, code[pc], sz])
        o = code[pc]
        if o == op["JMP"] or o == op["JMPF"] or o == op["JMPT"] or o == op["TRY_PUSH"]:
            t = pc + 5 + rd_i32(code, pc + 1)
            if t >= 0 and t < n:
                prot[t] = 1
        elif o == op["MKFN"]:
            t = rd_i32(code, pc + 1)
            if t >= 0 and t < n:
                prot[t] = 1
        pc = pc + sz
    mut idx = 0
    while idx < ops.len:
        mut done = false
        if idx + 3 < ops.len:
            a0 = ops[idx]
            a1 = ops[idx + 1]
            a2 = ops[idx + 2]
            a3 = ops[idx + 3]
            p0 = a0[0]
            p1 = a1[0]
            p3 = a3[0]
            end = p3 + a3[2]
            mut inside = false
            mut q = p0 + 1
            while q < end:
                if prot[q] != 0:
                    inside = true
                q = q + 1
            if not inside:
                o0 = a0[1]
                o1 = a1[1]
                o2 = a2[1]
                o3 = a3[1]
                ld = o0 == op["LOAD"] or o0 == op["LOADG"]
                isg = o0 == op["LOADG"]
                want0 = op["STORE"]
                want1 = op["BIND"]
                if isg:
                    want0 = op["STOREG"]
                    want1 = op["BINDG"]
                mut rep = nil
                if ld and o1 == op["PUSH_INT"] and (o2 == op["ADD"] or o2 == op["SUB"]) and (o3 == op["STORE"] or o3 == op["STOREG"] or o3 == op["BIND"] or o3 == op["BINDG"]):
                    s = rd_u16(code, p0 + 1)
                    if s == rd_u16(code, p3 + 1) and (o3 == want0 or o3 == want1):
                        k = rd_i64(code, p1 + 1)
                        isb = o3 == want1
                        if o2 == op["ADD"] and k == 1 and not isb:
                            opc = op["INC_L"]
                            if isg:
                                opc = op["INC_G"]
                            rep = [opc] + u16_bytes(s)
                        elif o2 == op["ADD"] and k == 1 and isb:
                            opc = op["INCB_L"]
                            if isg:
                                opc = op["INCB_G"]
                            rep = [opc] + u16_bytes(s)
                        elif o2 == op["SUB"] and k == 1 and not isb:
                            opc = op["DEC_L"]
                            if isg:
                                opc = op["DEC_G"]
                            rep = [opc] + u16_bytes(s)
                        elif o2 == op["ADD"] and not isb:
                            opc = op["ADDI_L"]
                            if isg:
                                opc = op["ADDI_G"]
                            rep = [opc] + u16_bytes(s) + be_bytes(k, 8)
                if rep == nil and ld and (o1 == op["LOAD"] or o1 == op["LOADG"]) and (o2 == op["ADD"] or o2 == op["SUB"]) and (o3 == op["STORE"] or o3 == op["STOREG"]):
                    s = rd_u16(code, p0 + 1)
                    x = rd_u16(code, p1 + 1)
                    same_kind = (o1 == op["LOAD"]) == (not isg) and (o3 == op["STORE"]) == (not isg)
                    if s == rd_u16(code, p3 + 1) and same_kind:
                        opc = op["ADDS_L"]
                        if o2 == op["SUB"]:
                            opc = op["SUBS_L"]
                        if isg:
                            opc = op["ADDS_G"]
                            if o2 == op["SUB"]:
                                opc = op["SUBS_G"]
                        rep = [opc] + u16_bytes(s) + u16_bytes(x)
                if rep != nil and rep.len <= end - p0:
                    mut w = p0
                    for byte in rep:
                        code[w] = byte
                        w = w + 1
                    while w < end:
                        code[w] = op["NOP"]
                        w = w + 1
                    idx = idx + 4
                    done = true
        if not done:
            idx = idx + 1
    return nil

# ---------- topbind / entry ----------
fn topbind_into(acc, stmts):
    for s in stmts:
        k = s[0]
        if k == "decl":
            if not in_list(acc, s[2]):
                acc.append(s[2])
        elif k == "fn":
            if not in_list(acc, s[1]):
                acc.append(s[1])
        elif k == "assign" and s[1][0] == "var":
            if not in_list(acc, s[1][1]):
                acc.append(s[1][1])
        elif k == "for":
            if not in_list(acc, s[1]):
                acc.append(s[1])
            topbind_into(acc, s[3])
        elif k == "while":
            topbind_into(acc, s[2])
        elif k == "if":
            topbind_into(acc, s[2])
            if s[3]:
                if s[3].len == 1 and s[3][0][0] == "if":
                    topbind_into(acc, [s[3][0]])
                else:
                    topbind_into(acc, s[3])
        elif k == "try":
            topbind_into(acc, s[1])
            for h in s[2]:
                topbind_into(acc, h[1])

fn compile_prog(cc, prog):
    if prog[0] != "prog":
        fail("AST gốc")
    topbind_into(cc["topbinds"], prog[1])
    for s in prog[1]:
        if s[0] == "fn" and not has_key(cc["nat"], s[1]):
            if not has_key(cc["gnames"], s[1]):
                c_gslot(cc, s[1])
            if not in_list(cc["fn_names"], s[1]):
                cc["fn_names"].append(s[1])
            mut bs = []
            for x in s[3]:
                if x[0] != "line":
                    bs.append(x)
            if bs.len == 1 and bs[0][0] == "ret":
                cc["fntab"][s[1]] = [s[2], bs[0][1], s[1]]
    main = new_fctx(true, 0, 0)
    cc["main"] = main
    cc["fns"].append(main)
    prescan_globals(cc, prog[1])
    prescan(cc, prog[1])
    reset_walk(cc, main)
    for f in cc["fnmap"]:
        reset_walk(cc, f)
    for s in prog[1]:
        c_stmt(cc, s)
    cc["fns"].pop()
    peephole(cc)
    c_emit(cc, cc["op"]["HALT"])

# ---------- parse/build/CLI ----------
# tên file không thư mục (khớp file_base của t.py)
fn file_base(p):
    mut a = p.split("\\")
    mut b = a[a.len - 1].split("/")
    return b[b.len - 1]

fn parse_file(path):
    f = open(path, "r")
    s = read(f)
    close(f)
    parse2.LINES[0] = file_base(path)
    toks = host.scan(s)
    st = [toks, 0]
    return host.parse_prog(st)

fn has_import(prog):
    for st in prog[1]:
        if st[0] == "import" or st[0] == "fromimport":
            return true
    return false

# có import -> tự gộp module lúc build (-b vẫn nhận để tương thích)
fn parse_prog_file(src, dobundle):
    prog = parse_file(src)
    if dobundle or has_import(prog):
        return bundle(src)
    return prog

fn compile_file(src, dobundle):
    prog = parse_prog_file(src, dobundle)
    cc = new_c()
    compile_prog(cc, prog)
    return build_from_cc(cc)

fn build_from_cc(cc):
    ng = cc["nat"].len + cc["gnames"].len
    if ng > 65535:
        fail("quá nhiều global")
    mut out = [84, 66, 48, 49]
    for b in be_bytes(ng, 2):
        out.append(b)
    ln = cc["b"].len
    for b in be_bytes(ln, 4):
        out.append(b)
    for b in cc["b"]:
        out.append(b)
    # bảng dòng "LN01" (khớp line_table của vm/tcomp.py)
    for b in [76, 78, 48, 49]:
        out.append(b)
    for b in be_bytes(cc["lfiles"].len, 2):
        out.append(b)
    for fnm in cc["lfiles"]:
        for b in be_bytes(fnm.len, 2):
            out.append(b)
        for q in range(fnm.len):
            out.append(ord(fnm[q:q + 1]))
    for b in be_bytes(cc["lines"].len, 4):
        out.append(b)
    for e in cc["lines"]:
        for b in be_bytes(e[0], 4):
            out.append(b)
        for b in be_bytes(e[1], 4):
            out.append(b)
        for b in be_bytes(e[2], 2):
            out.append(b)
    # bảng export "EX01" (khớp export_table của vm/tcomp.py)
    mut ex = []
    for k in cc["main"]["scopes"][0]["m"]:
        if "$" not in k and not k.endswith("__type"):
            ex.append(k)
    for b in [69, 88, 48, 49]:
        out.append(b)
    for b in be_bytes(ex.len, 2):
        out.append(b)
    for k in ex:
        for b in be_bytes(k.len, 2):
            out.append(b)
        for q in range(k.len):
            out.append(ord(k[q:q + 1]))
        for b in be_bytes(cc["main"]["scopes"][0]["m"][k], 2):
            out.append(b)
    mut cs = []
    for b in out:
        cs.append(chr(b))
    return ["".join(cs), cc["gnames"].len]

fn main_cli():
    args = argv()
    mut paths = []
    mut dobundle = false
    for a in args:
        if a == "-b":
            dobundle = true
        else:
            paths.append(a)
    if paths.len < 2:
        print("dùng: tcomp [-b] prog.t prog.tb")
        raise "usage"
    src = paths[paths.len - 2]
    out = paths[paths.len - 1]
    prog = parse_prog_file(src, dobundle)
    cc = new_c()
    compile_prog(cc, prog)
    r = build_from_cc(cc)
    f = open(out, "wb")
    write(f, r[0])
    close(f)
    print("ok")
