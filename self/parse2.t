# self/parse2.t — phần 2 parser T bằng T: stmt, suite, expr các cấp (v0.6 full)
# Nối với parse.t (cùng module khi import cả hai, dùng chung state [toks,pos]).
# Tự chứa helpers nên nạp độc lập vẫn chạy (định nghĩa lại idempotent).
fn is_op(t, v):
    return t[0] == "OP" and t[1] == v

fn is_kw(t, w):
    return t[0] == "KW" and t[1] == w

fn tok_line(t):
    if t.len > 2:
        return "line " + str(t[2]) + ": "
    return ""

fn eat_op(st, v):
    t = st[0][st[1]]
    if t[0] != "OP" or t[1] != v:
        raise tok_line(t) + "want " + v
    st[1] = st[1] + 1
    return t

fn parse_suite(st):
    eat_ind(st)
    mut ss = []
    while not is_ded(st):
        add_mark(ss, st)
        add_stmt(ss, parse_stmt(st))
    eat_ded(st)
    return ss

# LINES[0] != nil (tên file): chèn mốc ["line", dòng, file] trước mỗi câu lệnh
# (compiler dùng làm bảng dòng cho lỗi runtime; mặc định tắt)
LINES = [nil]

fn add_mark(ss, st):
    if LINES[0] != nil:
        t = st[0][st[1]]
        mut ln = 0
        if t.len > 2:
            ln = int(t[2])
        ss.append(["line", ln, LINES[0]])
    return nil

# desugar trả ["multi", [...]] -> trải phẳng vào cùng block
fn add_stmt(ss, s):
    if s[0] == "multi":
        for x in s[1]:
            ss.append(x)
    else:
        ss.append(s)
    return nil

# tên ẩn ($ không lex được nên không đụng tên user)
GENSYM = [0]

fn gensym(kind):
    GENSYM[0] = GENSYM[0] + 1
    return "$" + kind + str(GENSYM[0])

fn eat_name(st):
    t = st[0][st[1]]
    if t[0] != "NAME" and not (t[0] == "KW" and t[1] == "async"):
        raise tok_line(t) + "want NAME"
    st[1] = st[1] + 1
    return t[1]

fn parse_names(st):
    mut ns = [eat_name(st)]
    while is_op(st[0][st[1]], ","):
        st[1] = st[1] + 1
        ns.append(eat_name(st))
    return ns

# --- Type Annotations (v6.0 L10 & v7.0 L12) ---
# Lưu thông tin kiểu của biến, hàm, struct, switches để std/typecheck.t kiểm tra tĩnh
TYPE_ANNS = [{"vars": {}, "muts": {}, "fns": {}, "structs": {"Ok": [["val", "any"]], "Err": [["val", "any"]], "Some": [["val", "any"]], "None": []}, "traits": {}, "switches": []}]

fn reset_type_anns():
    TYPE_ANNS[0] = {"vars": {}, "muts": {}, "fns": {}, "structs": {"Ok": [["val", "any"]], "Err": [["val", "any"]], "Some": [["val", "any"]], "None": []}, "traits": {}, "switches": []}
    return nil

fn parse_type_ann(st):
    t = st[0][st[1]]
    mut base = ""
    if t[0] == "NAME" or t[0] == "KW":
        base = t[1]
        st[1] = st[1] + 1
    else:
        raise tok_line(t) + "want type name"
    if is_op(st[0][st[1]], "["):
        st[1] = st[1] + 1
        mut subs = [parse_type_ann(st)]
        while is_op(st[0][st[1]], ","):
            st[1] = st[1] + 1
            subs.append(parse_type_ann(st))
        eat_op(st, "]")
        base = base + "[" + ", ".join(subs) + "]"
    while is_op(st[0][st[1]], "|"):
        st[1] = st[1] + 1
        base = base + " | " + parse_type_ann(st)
    return base

# --- Structural Pattern Matching (v6.0 L11) ---
# Hỗ trợ:
#   - Wildcard: case _:
#   - Literal / biểu thức: case 1, 2:
#   - List destructuring: case [a, b]: hoặc case [head, *tail]:
#   - Dict destructuring: case {"k": v, "x": 10}:
#   - Struct destructuring: case Point(0, y):
#   - Guard: case pat if cond:
fn is_pat_bind_name(nm):
    if nm == "_":
        return true
    c = ord(nm[0:1])
    # chữ thường a..z hoặc _ là biến bind mới trong pattern destructuring
    return (c >= 97 and c <= 122) or nm[0:1] == "_"

fn compile_subpat(val_expr, pat, conds, binds):
    k = pat[0]
    if k == "var":
        nm = pat[1]
        if nm == "_":
            return nil
        if is_pat_bind_name(nm):
            binds.append(["decl", "mut", nm, val_expr])
            return nil
        conds.append(["bin", "==", val_expr, pat])
        return nil
    if k == "list":
        items = pat[1]
        mut star_idx = -1
        mut i = 0
        while i < items.len:
            it = items[i]
            if it[0] == "pat_star":
                star_idx = i
            i = i + 1
        conds.append(["bin", "==", ["bltin", "type", [val_expr]], ["str", "list"]])
        if star_idx < 0:
            conds.append(["bin", "==", ["attr", val_expr, "len"], ["num", str(items.len)]])
            mut j = 0
            while j < items.len:
                compile_subpat(["idx", val_expr, ["num", str(j)]], items[j], conds, binds)
                j = j + 1
        else:
            conds.append(["bin", ">=", ["attr", val_expr, "len"], ["num", str(items.len - 1)]])
            mut j = 0
            while j < star_idx:
                compile_subpat(["idx", val_expr, ["num", str(j)]], items[j], conds, binds)
                j = j + 1
            star_nm = items[star_idx][1]
            if star_nm != "_":
                binds.append(["decl", "mut", star_nm, ["slice", val_expr, ["num", str(star_idx)], nil, nil]])
        return nil
    if k == "dict":
        ks = pat[1]
        vs = pat[2]
        conds.append(["bin", "==", ["bltin", "type", [val_expr]], ["str", "dict"]])
        mut j = 0
        while j < ks.len:
            kj = ks[j]
            vj = vs[j]
            conds.append(["bin", "in", kj, val_expr])
            compile_subpat(["idx", val_expr, kj], vj, conds, binds)
            j = j + 1
        return nil
    if k == "call" and pat[1][0] == "var":
        sname = pat[1][1]
        pargs = pat[2]
        conds.append(["bin", "==", ["bltin", "type", [val_expr]], ["str", sname]])
        sfields = TYPE_ANNS[0]["structs"].get(sname, nil)
        mut j = 0
        while j < pargs.len:
            if sfields != nil and j < sfields.len:
                fname = sfields[j][0]
                compile_subpat(["attr", val_expr, fname], pargs[j], conds, binds)
            j = j + 1
        return nil
    conds.append(["bin", "==", val_expr, pat])
    return nil

fn parse_pat_atom(st):
    t = st[0][st[1]]
    if is_op(t, "["):
        st[1] = st[1] + 1
        mut items = []
        if not is_op(st[0][st[1]], "]"):
            if is_op(st[0][st[1]], "*"):
                st[1] = st[1] + 1
                items.append(["pat_star", eat_name(st)])
            else:
                items.append(parse_pat_atom(st))
            while is_op(st[0][st[1]], ","):
                st[1] = st[1] + 1
                if is_op(st[0][st[1]], "*"):
                    st[1] = st[1] + 1
                    items.append(["pat_star", eat_name(st)])
                else:
                    items.append(parse_pat_atom(st))
        eat_op(st, "]")
        return ["list", items]
    return parse_or(st)

fn parse_switch(st):
    st[1] = st[1] + 1
    subj = parse_expr(st)
    eat_op(st, ":")
    eat_ind(st)
    mut cases = []
    mut dflt = nil
    while not is_ded(st):
        t = st[0][st[1]]
        if is_kw(t, "case"):
            st[1] = st[1] + 1
            mut vs = [parse_pat_atom(st)]
            while is_op(st[0][st[1]], ","):
                st[1] = st[1] + 1
                vs.append(parse_pat_atom(st))
            mut guard = nil
            if is_kw(st[0][st[1]], "if"):
                st[1] = st[1] + 1
                guard = parse_or(st)
            eat_op(st, ":")
            cases.append([vs, parse_suite(st), guard])
        elif is_kw(t, "default"):
            st[1] = st[1] + 1
            eat_op(st, ":")
            dflt = parse_suite(st)
        else:
            raise tok_line(t) + "want case/default"
    eat_ded(st)
    TYPE_ANNS[0]["switches"].append([subj, cases, dflt != nil])
    tmp = gensym("s")
    mut b = dflt
    mut i = cases.len - 1
    while i >= 0:
        vs = cases[i][0]
        suite = cases[i][1]
        guard = cases[i][2]
        if vs.len == 1:
            pat = vs[0]
            mut conds = []
            mut binds = []
            if pat[0] == "var" and pat[1] == "_":
                # wildcard _
                conds = []
            elif pat[0] == "list" or pat[0] == "dict" or (pat[0] == "call" and pat[1][0] == "var" and TYPE_ANNS[0]["structs"].has(pat[1][1])):
                compile_subpat(["var", tmp], pat, conds, binds)
            elif pat[0] == "var" and guard != nil and is_pat_bind_name(pat[1]):
                binds.append(["decl", "mut", pat[1], ["var", tmp]])
            else:
                conds.append(["bin", "==", ["var", tmp], pat])
            mut c = nil
            for ce in conds:
                if c == nil:
                    c = ce
                else:
                    c = ["bin", "and", c, ce]
            if c == nil:
                c = ["lit", "true"]
            mut body_i = []
            for bd in binds:
                body_i.append(bd)
            if guard != nil:
                # Nếu có guard và có binds, phải tính binds trước khi kiểm guard -> dùng cờ $m
                if binds.len > 0:
                    c = ["bin", "and", c, ["lit", "true"]]
                    # Thay thế bằng nhánh lồng: nếu conds khớp -> gán binds -> nếu guard khớp -> chạy suite
                else:
                    c = ["bin", "and", c, guard]
            for st_item in suite:
                body_i.append(st_item)
            if guard != nil and binds.len > 0:
                # Dùng biến cờ $gm để nếu guard sai thì rơi xuống nhánh dưới (b)
                gm = gensym("gm")
                mut try_b = []
                for bd in binds:
                    try_b.append(bd)
                mut guarded_suite = [["assign", ["var", gm], ["lit", "true"]]]
                for st_item in suite:
                    guarded_suite.append(st_item)
                try_b.append(["if", guard, guarded_suite, nil])
                mut seq_b = [["decl", "mut", gm, ["lit", "false"]], ["if", c, try_b, nil]]
                if b != nil:
                    seq_b.append(["if", ["un", "not", ["var", gm]], b, nil])
                b = seq_b
            else:
                b = [["if", c, body_i, b]]
        else:
            mut c = nil
            for v in vs:
                eq = ["bin", "==", ["var", tmp], v]
                if c == nil:
                    c = eq
                else:
                    c = ["bin", "or", c, eq]
            if guard != nil:
                c = ["bin", "and", c, guard]
            b = [["if", c, suite, b]]
        i = i - 1
    mut out = [["decl", "let", tmp, subj]]
    if b != nil:
        for bi in b:
            out.append(bi)
    return ["multi", out]

# --- Traits / Interfaces (v6.0 L11) ---
TRAITS = [{}]

fn parse_trait(st):
    st[1] = st[1] + 1
    tname = eat_name(st)
    eat_op(st, ":")
    eat_ind(st)
    mut reqs = []
    mut dflts = []
    while not is_ded(st):
        t = st[0][st[1]]
        if is_kw(t, "fn"):
            st[1] = st[1] + 1
            mname = eat_name(st)
            eat_op(st, "(")
            mut args = []
            if not is_op(st[0][st[1]], ")"):
                p0 = eat_name(st)
                if is_op(st[0][st[1]], ":"):
                    st[1] = st[1] + 1
                    parse_type_ann(st)
                args.append(p0)
                while is_op(st[0][st[1]], ","):
                    st[1] = st[1] + 1
                    pn = eat_name(st)
                    if is_op(st[0][st[1]], ":"):
                        st[1] = st[1] + 1
                        parse_type_ann(st)
                    args.append(pn)
            eat_op(st, ")")
            if is_op(st[0][st[1]], "->"):
                st[1] = st[1] + 1
                parse_type_ann(st)
            if is_op(st[0][st[1]], ":"):
                st[1] = st[1] + 1
                body = parse_suite(st)
                dflts.append(["fn", mname, args, body])
            else:
                reqs.append([mname, args.len])
        else:
            raise tok_line(t) + "want fn in trait"
    eat_ded(st)
    TRAITS[0][tname] = {"reqs": reqs, "dflts": dflts}
    TYPE_ANNS[0]["traits"][tname] = {"reqs": reqs, "dflts": dflts}
    return ["multi", []]

fn parse_struct(st):
    st[1] = st[1] + 1
    name = eat_name(st)
    mut impl_traits = []
    if is_op(st[0][st[1]], "("):
        st[1] = st[1] + 1
        if not is_op(st[0][st[1]], ")"):
            impl_traits.append(eat_name(st))
            while is_op(st[0][st[1]], ","):
                st[1] = st[1] + 1
                impl_traits.append(eat_name(st))
        eat_op(st, ")")
    eat_op(st, ":")
    eat_ind(st)
    mut fields = []
    mut field_anns = []
    mut meths = []
    while not is_ded(st):
        t = st[0][st[1]]
        if is_kw(t, "fn") or (is_kw(t, "async") and is_kw(st[0][st[1] + 1], "fn")):
            ms = parse_stmt(st)
            if ms[0] == "multi":
                for mx in ms[1]:
                    meths.append(mx)
            else:
                meths.append(ms)
        elif t[0] == "NAME":
            f0 = eat_name(st)
            mut fty = "any"
            if is_op(st[0][st[1]], ":"):
                st[1] = st[1] + 1
                fty = parse_type_ann(st)
                for g in fields:
                    if g == f0:
                        raise tok_line(t) + "trùng field " + f0
                fields.append(f0)
                field_anns.append([f0, fty])
            else:
                mut fs = [f0]
                while is_op(st[0][st[1]], ","):
                    st[1] = st[1] + 1
                    fs.append(eat_name(st))
                if is_op(st[0][st[1]], ":"):
                    st[1] = st[1] + 1
                    fty = parse_type_ann(st)
                for f in fs:
                    for g in fields:
                        if g == f:
                            raise tok_line(t) + "trùng field " + f
                    fields.append(f)
                    field_anns.append([f, fty])
        else:
            raise tok_line(t) + "want field/fn in struct"
    eat_ded(st)
    TYPE_ANNS[0]["structs"][name] = field_anns
    # Kiểm tra và kế thừa các method từ trait
    for tr_name in impl_traits:
        if not TRAITS[0].has(tr_name):
            raise "struct " + name + " implements unknown trait " + tr_name
        tr_info = TRAITS[0][tr_name]
        for dm in tr_info["dflts"]:
            mut has_m = false
            for m in meths:
                if m[0] == "fn" and m[1] == dm[1]:
                    has_m = true
            if not has_m:
                meths.append(dm)
        for req in tr_info["reqs"]:
            rname = req[0]
            mut has_req = false
            for m in meths:
                if m[0] == "fn" and m[1] == rname:
                    has_req = true
            if not has_req:
                raise "struct " + name + " thiếu method '" + rname + "' của trait " + tr_name
    mut out = []
    for m in meths:
        if m[0] == "fn":
            if m.len > 4:
                out.append(["fn", name + "__" + m[1], m[2], m[3], m[4]])
            else:
                out.append(["fn", name + "__" + m[1], m[2], m[3]])
        else:
            out.append(m)
    mut fstrs = []
    mut fvars = []
    for f in fields:
        fstrs.append(["str", f])
        fvars.append(["var", f])
    mut mks = []
    mut mvs = []
    for m in meths:
        if m[0] == "fn":
            mks.append(["str", m[1]])
            mvs.append(["var", name + "__" + m[1]])
    # let P__type = __struct("P", [fields], {methods}) ; fn P(x, y): return __new(P__type, x, y)
    tp = name + "__type"
    out.append(["decl", "let", tp, ["call", ["var", "__struct"], [["str", name], ["list", fstrs], ["dict", mks, mvs]]]])
    mut params = []
    mut nargs = [["var", tp]]
    for f in fields:
        params.append(f)
        nargs.append(["var", f])
    out.append(["fn", name, params, [["ret", ["call", ["var", "__new"], nargs]]]])
    return ["multi", out]

fn eat_ind(st):
    t = st[0][st[1]]
    if t[0] != "IND":
        raise "want IND"
    st[1] = st[1] + 1

fn eat_ded(st):
    t = st[0][st[1]]
    if t[0] != "DED":
        raise "want DED"
    st[1] = st[1] + 1

fn is_ded(st):
    t = st[0][st[1]]
    return t[0] == "DED"

# lambda / comprehension trong câu lệnh -> hàm ẩn đặt ngay trước câu lệnh
# (PEND: stack danh sách định nghĩa chờ; khớp t.py)
PEND = []

fn hoist(fdef):
    if PEND.len == 0:
        raise "lambda/comprehension ngoài câu lệnh"
    PEND[PEND.len - 1].append(fdef)
    return nil

fn merge_pending():
    top = PEND.pop()
    if top.len > 0:
        if PEND.len == 0:
            raise "lambda/comprehension ngoài câu lệnh"
        for d in top:
            PEND[PEND.len - 1].append(d)
    return nil

# for a in xs / for a, b in xs (-> for $pN in xs: mut a = $pN[0] ...)
fn make_for(vs, it, body):
    if vs.len == 1:
        return ["for", vs[0], it, body]
    tmp = gensym("p")
    mut b2 = []
    mut i = 0
    while i < vs.len:
        b2.append(["decl", "mut", vs[i], ["idx", ["var", tmp], ["num", str(i)]]])
        i = i + 1
    for x in body:
        b2.append(x)
    return ["for", tmp, it, b2]

fn parse_stmt(st):
    PEND.append([])
    s = parse_stmt0(st)
    pend = PEND.pop()
    if pend.len == 0:
        return s
    mut out = []
    for d in pend:
        out.append(d)
    if s[0] == "multi":
        for x in s[1]:
            out.append(x)
    else:
        out.append(s)
    return ["multi", out]

# fn(a, b): biểu_thức  ->  fn $lN(a, b): return biểu_thức
fn parse_lambda(st):
    st[1] = st[1] + 1
    eat_op(st, "(")
    mut args = []
    mut is_va = false
    if not is_op(st[0][st[1]], ")"):
        if is_op(st[0][st[1]], "*"):
            st[1] = st[1] + 1
            args.append(eat_name(st))
            if is_op(st[0][st[1]], ":"):
                st[1] = st[1] + 1
                parse_type_ann(st)
            is_va = true
        else:
            args.append(eat_name(st))
            if is_op(st[0][st[1]], ":"):
                st[1] = st[1] + 1
                parse_type_ann(st)
        while is_op(st[0][st[1]], ","):
            if is_va:
                raise tok_line(st[0][st[1]]) + "*args phải ở cuối"
            st[1] = st[1] + 1
            if is_op(st[0][st[1]], "*"):
                st[1] = st[1] + 1
                args.append(eat_name(st))
                if is_op(st[0][st[1]], ":"):
                    st[1] = st[1] + 1
                    parse_type_ann(st)
                is_va = true
            else:
                args.append(eat_name(st))
                if is_op(st[0][st[1]], ":"):
                    st[1] = st[1] + 1
                    parse_type_ann(st)
    eat_op(st, ")")
    if is_op(st[0][st[1]], "->"):
        st[1] = st[1] + 1
        parse_type_ann(st)
    eat_op(st, ":")
    PEND.append([])
    body = parse_expr(st)
    mut fb = PEND.pop()
    fb.append(["ret", body])
    name = gensym("l")
    if is_va:
        hoist(["fn", name, args, fb, -1])
    else:
        hoist(["fn", name, args, fb])
    return ["var", name]

# --- Generator yield (v6.0 L9) ---
fn has_yield_stmt(s):
    k = s[0]
    if k == "yield":
        return true
    if k == "fn":
        return false
    if k == "if":
        if has_yield_list(s[2]):
            return true
        if s[3] != nil and has_yield_list(s[3]):
            return true
        return false
    if k == "while":
        return has_yield_list(s[2])
    if k == "for":
        return has_yield_list(s[3])
    if k == "try":
        if has_yield_list(s[1]):
            return true
        for h in s[2]:
            if has_yield_list(h[1]):
                return true
        return false
    if k == "multi":
        return has_yield_list(s[1])
    return false

fn has_yield_list(ss):
    for s in ss:
        if has_yield_stmt(s):
            return true
    return false

fn rw_yield_stmt(s, gv):
    k = s[0]
    if k == "yield":
        return ["expr", ["call", ["attr", ["var", gv], "append"], [s[1]]]]
    if k == "ret":
        if s[1] != nil and not (s[1][0] == "lit" and s[1][1] == "nil"):
            return ["multi", [["expr", ["call", ["attr", ["var", gv], "append"], [s[1]]]], ["ret", ["var", gv]]]]
        return ["ret", ["var", gv]]
    if k == "fn":
        return s
    if k == "if":
        mut b2 = nil
        if s[3] != nil:
            b2 = rw_yield_list(s[3], gv)
        return ["if", s[1], rw_yield_list(s[2], gv), b2]
    if k == "while":
        return ["while", s[1], rw_yield_list(s[2], gv)]
    if k == "for":
        return ["for", s[1], s[2], rw_yield_list(s[3], gv)]
    if k == "try":
        mut hs2 = []
        for h in s[2]:
            hs2.append([h[0], rw_yield_list(h[1], gv)])
        return ["try", rw_yield_list(s[1], gv), hs2]
    if k == "multi":
        return ["multi", rw_yield_list(s[1], gv)]
    return s

fn rw_yield_list(ss, gv):
    mut out = []
    for s in ss:
        add_stmt(out, rw_yield_stmt(s, gv))
    return out

# Khử đường cú pháp try ... [except ...] finally: fin
fn rw_fin_stmt(s, fd, fin, ldepth):
    k = s[0]
    if k == "ret":
        rv = gensym("rv")
        mut seq = [["decl", "let", rv, s[1]], ["assign", ["var", fd], ["lit", "true"]]]
        for fs in fin:
            seq.append(fs)
        seq.append(["ret", ["var", rv]])
        return ["multi", seq]
    if (k == "break" or k == "continue") and ldepth == 0:
        mut seq = [["assign", ["var", fd], ["lit", "true"]]]
        for fs in fin:
            seq.append(fs)
        seq.append(s)
        return ["multi", seq]
    if k == "if":
        mut b2 = nil
        if s[3] != nil:
            b2 = rw_fin_list(s[3], fd, fin, ldepth)
        return ["if", s[1], rw_fin_list(s[2], fd, fin, ldepth), b2]
    if k == "while":
        return ["while", s[1], rw_fin_list(s[2], fd, fin, ldepth + 1)]
    if k == "for":
        return ["for", s[1], s[2], rw_fin_list(s[3], fd, fin, ldepth + 1)]
    if k == "try":
        mut hs2 = []
        for h in s[2]:
            hs2.append([h[0], rw_fin_list(h[1], fd, fin, ldepth)])
        return ["try", rw_fin_list(s[1], fd, fin, ldepth), hs2]
    if k == "multi":
        return ["multi", rw_fin_list(s[1], fd, fin, ldepth)]
    return s

fn rw_fin_list(ss, fd, fin, ldepth):
    mut out = []
    for x in ss:
        add_stmt(out, rw_fin_stmt(x, fd, fin, ldepth))
    return out

fn make_try_fin(body, hs, fin):
    fd = gensym("fd")
    fe = gensym("fe")
    b_rw = rw_fin_list(body, fd, fin, 0)
    mut inner = b_rw
    if hs.len > 0:
        mut hs_rw = []
        for h in hs:
            hs_rw.append([h[0], rw_fin_list(h[1], fd, fin, 0)])
        inner = [["try", b_rw, hs_rw]]
    mut run_fin1 = [["assign", ["var", fd], ["lit", "true"]]]
    mut run_fin2 = [["assign", ["var", fd], ["lit", "true"]]]
    for fs in fin:
        run_fin1.append(fs)
        run_fin2.append(fs)
    guard_fin1 = ["if", ["un", "not", ["var", fd]], run_fin1, nil]
    guard_fin2 = ["if", ["un", "not", ["var", fd]], run_fin2, nil]
    outer_h = [nil, [["decl", "let", fe, ["var", "err"]], guard_fin1, ["raise", ["var", fe]]]]
    return ["multi", [["decl", "mut", fd, ["lit", "false"]], ["try", inner, [outer_h]], guard_fin2]]

# [e for ... if ...] / {k: v for ...} -> hàm ẩn $cN() dựng kết quả, gọi tại chỗ;
# định nghĩa ẩn của từng mệnh đề đặt ngay trước mệnh đề đó
fn parse_comp(st, first, key, inner, closer):
    mut clauses = []
    mut go = true
    while go:
        t = st[0][st[1]]
        if is_kw(t, "for"):
            st[1] = st[1] + 1
            vs = parse_names(st)
            if not is_kw(st[0][st[1]], "in"):
                raise tok_line(st[0][st[1]]) + "want in"
            st[1] = st[1] + 1
            PEND.append([])
            it = parse_or(st)
            clauses.append(["for", vs, it, PEND.pop()])
        elif is_kw(t, "if"):
            st[1] = st[1] + 1
            PEND.append([])
            c = parse_or(st)
            clauses.append(["if", c, nil, PEND.pop()])
        else:
            go = false
    eat_op(st, closer)
    r = gensym("r")
    mut body = []
    for d in inner:
        body.append(d)
    mut init = nil
    if key == nil:
        body.append(["expr", ["call", ["attr", ["var", r], "append"], [first]]])
        init = ["list", []]
    else:
        body.append(["assign", ["idx", ["var", r], key], first])
        init = ["dict", [], []]
    mut i = clauses.len - 1
    while i >= 0:
        c = clauses[i]
        mut nb = []
        for d in c[3]:
            nb.append(d)
        if c[0] == "if":
            nb.append(["if", c[1], body, nil])
        else:
            nb.append(make_for(c[1], c[2], body))
        body = nb
        i = i - 1
    mut fb = [["decl", "mut", r, init]]
    for x in body:
        fb.append(x)
    fb.append(["ret", ["var", r]])
    name = gensym("c")
    hoist(["fn", name, [], fb])
    return ["call", ["var", name], []]

fn parse_stmt0(st):
    t = st[0][st[1]]
    if t[0] == "KW":
        if t[1] == "switch":
            return parse_switch(st)
        if t[1] == "struct":
            return parse_struct(st)
        if t[1] == "trait":
            return parse_trait(st)
        if t[1] == "with":
            st[1] = st[1] + 1
            res_expr = parse_expr(st)
            mut as_var = nil
            if is_kw(st[0][st[1]], "as"):
                st[1] = st[1] + 1
                as_var = eat_name(st)
            eat_op(st, ":")
            wbody = parse_suite(st)
            wv = gensym("w")
            mut ibody = []
            if as_var != nil:
                ibody.append(["decl", "mut", as_var, ["bltin", "close", [["var", wv], ["num", "1"]]]])
            else:
                ibody.append(["expr", ["bltin", "close", [["var", wv], ["num", "1"]]]])
            for ws in wbody:
                ibody.append(ws)
            fin = [["expr", ["bltin", "close", [["var", wv]]]]]
            tf = make_try_fin(ibody, [], fin)
            mut out = [["decl", "let", wv, res_expr]]
            for ts in tf[1]:
                out.append(ts)
            return ["multi", out]
        if t[1] == "yield":
            st[1] = st[1] + 1
            return ["yield", parse_expr(st)]
        if t[1] == "let" or t[1] == "mut":
            st[1] = st[1] + 1
            nm = st[0][st[1]][1]
            st[1] = st[1] + 1
            TYPE_ANNS[0]["muts"][nm] = (t[1] == "mut")
            if is_op(st[0][st[1]], ":"):
                st[1] = st[1] + 1
                vty = parse_type_ann(st)
                TYPE_ANNS[0]["vars"][nm] = vty
            eat_op(st, "=")
            return ["decl", t[1], nm, parse_expr(st)]
        if t[1] == "fn" or (t[1] == "async" and is_kw(st[0][st[1] + 1], "fn")):
            if t[1] == "async":
                st[1] = st[1] + 1
            st[1] = st[1] + 1
            nm = st[0][st[1]][1]
            st[1] = st[1] + 1
            eat_op(st, "(")
            mut args = []
            mut param_anns = []
            mut defs = []
            mut is_va = false
            if not is_op(st[0][st[1]], ")"):
                if is_op(st[0][st[1]], "*"):
                    st[1] = st[1] + 1
                    p0 = eat_name(st)
                    args.append(p0)
                    mut pty = "any"
                    if is_op(st[0][st[1]], ":"):
                        st[1] = st[1] + 1
                        pty = parse_type_ann(st)
                    param_anns.append([p0, pty])
                    is_va = true
                else:
                    p0 = st[0][st[1]][1]
                    args.append(p0)
                    st[1] = st[1] + 1
                    mut pty = "any"
                    if is_op(st[0][st[1]], ":"):
                        st[1] = st[1] + 1
                        pty = parse_type_ann(st)
                    param_anns.append([p0, pty])
                    if is_op(st[0][st[1]], "="):
                        st[1] = st[1] + 1
                        defs.append([p0, parse_expr(st)])
                while is_op(st[0][st[1]], ","):
                    if is_va:
                        raise tok_line(st[0][st[1]]) + "*args phải ở cuối danh sách tham số"
                    st[1] = st[1] + 1
                    if is_op(st[0][st[1]], "*"):
                        if defs.len > 0:
                            raise tok_line(st[0][st[1]]) + "không kết hợp tham số mặc định và *args"
                        st[1] = st[1] + 1
                        pn = eat_name(st)
                        args.append(pn)
                        mut pty = "any"
                        if is_op(st[0][st[1]], ":"):
                            st[1] = st[1] + 1
                            pty = parse_type_ann(st)
                        param_anns.append([pn, pty])
                        is_va = true
                    else:
                        pn = st[0][st[1]][1]
                        args.append(pn)
                        st[1] = st[1] + 1
                        mut pty = "any"
                        if is_op(st[0][st[1]], ":"):
                            st[1] = st[1] + 1
                            pty = parse_type_ann(st)
                        param_anns.append([pn, pty])
                        if is_op(st[0][st[1]], "="):
                            st[1] = st[1] + 1
                            defs.append([pn, parse_expr(st)])
                        elif defs.len > 0:
                            raise tok_line(st[0][st[1]]) + "tham số không mặc định sau tham số mặc định"
            eat_op(st, ")")
            mut ret_ty = "any"
            if is_op(st[0][st[1]], "->"):
                st[1] = st[1] + 1
                ret_ty = parse_type_ann(st)
            TYPE_ANNS[0]["fns"][nm] = {"params": param_anns, "ret": ret_ty}
            eat_op(st, ":")
            mut body = parse_suite(st)
            if has_yield_list(body):
                gv = gensym("g")
                mut gbody = [["decl", "mut", gv, ["list", []]]]
                for gs in rw_yield_list(body, gv):
                    gbody.append(gs)
                gbody.append(["ret", ["var", gv]])
                body = gbody
            if is_va:
                return ["fn", nm, args, body, -1]
            if defs.len == 0:
                return ["fn", nm, args, body]
            mut nb = []
            for d in defs:
                nb.append(["if", ["bin", "==", ["var", d[0]], ["lit", "nil"]], [["assign", ["var", d[0]], d[1]]], nil])
            for s in body:
                nb.append(s)
            return ["fn", nm, args, nb, defs.len]
        if t[1] == "return":
            st[1] = st[1] + 1
            return ["ret", parse_expr(st)]
        if t[1] == "if":
            return parse_if(st)
        if t[1] == "for":
            st[1] = st[1] + 1
            vs = parse_names(st)
            if not is_kw(st[0][st[1]], "in"):
                raise tok_line(st[0][st[1]]) + "want in"
            st[1] = st[1] + 1
            it = parse_expr(st)
            eat_op(st, ":")
            body = parse_suite(st)
            return make_for(vs, it, body)
        if t[1] == "while":
            st[1] = st[1] + 1
            c = parse_expr(st)
            eat_op(st, ":")
            return ["while", c, parse_suite(st)]
        if t[1] == "break":
            st[1] = st[1] + 1
            return ["break"]
        if t[1] == "continue":
            st[1] = st[1] + 1
            return ["continue"]
        if t[1] == "try":
            st[1] = st[1] + 1
            eat_op(st, ":")
            body = parse_suite(st)
            mut hs = []
            while is_kw(st[0][st[1]], "except"):
                st[1] = st[1] + 1
                mut nm = nil
                if st[0][st[1]][0] == "NAME" or st[0][st[1]][0] == "STR":
                    nm = st[0][st[1]][1]
                    st[1] = st[1] + 1
                eat_op(st, ":")
                hs.append([nm, parse_suite(st)])
            if is_kw(st[0][st[1]], "finally"):
                st[1] = st[1] + 1
                eat_op(st, ":")
                fin = parse_suite(st)
                return make_try_fin(body, hs, fin)
            return ["try", body, hs]
        if t[1] == "raise":
            st[1] = st[1] + 1
            return ["raise", parse_expr(st)]
        if t[1] == "assert":
            st[1] = st[1] + 1
            c = parse_expr(st)
            mut m = nil
            if is_op(st[0][st[1]], ","):
                st[1] = st[1] + 1
                m = parse_expr(st)
            return ["assert", c, m]
        if t[1] == "import":
            st[1] = st[1] + 1
            mut parts = [st[0][st[1]][1]]
            st[1] = st[1] + 1
            while is_op(st[0][st[1]], "."):
                st[1] = st[1] + 1
                parts.append(st[0][st[1]][1])
                st[1] = st[1] + 1
            mut alias = nil
            if is_kw(st[0][st[1]], "as"):
                st[1] = st[1] + 1
                alias = st[0][st[1]][1]
                st[1] = st[1] + 1
            return ["import", ".".join(parts), alias]
        if t[1] == "from":
            st[1] = st[1] + 1
            mut parts = [st[0][st[1]][1]]
            st[1] = st[1] + 1
            while is_op(st[0][st[1]], "."):
                st[1] = st[1] + 1
                parts.append(st[0][st[1]][1])
                st[1] = st[1] + 1
            if not is_kw(st[0][st[1]], "import"):
                raise "want import"
            st[1] = st[1] + 1
            mut names = [st[0][st[1]][1]]
            st[1] = st[1] + 1
            while is_op(st[0][st[1]], ","):
                st[1] = st[1] + 1
                names.append(st[0][st[1]][1])
                st[1] = st[1] + 1
            return ["fromimport", ".".join(parts), names]
        if t[1] == "print":
            nxt = st[0][st[1] + 1]
            if nxt[0] == "OP" and nxt[1] == "(":
                st[1] = st[1] + 1
                eat_op(st, "(")
                mut a = []
                if not is_op(st[0][st[1]], ")"):
                    a.append(parse_expr(st))
                    while is_op(st[0][st[1]], ","):
                        st[1] = st[1] + 1
                        a.append(parse_expr(st))
                eat_op(st, ")")
                return ["print", a]
    e = parse_expr(st)
    a = st[0][st[1]]
    if a[0] == "OP" and (a[1] == "=" or a[1] == "+=" or a[1] == "-=" or a[1] == "*=" or a[1] == "/=" or a[1] == "%=" or a[1] == "&=" or a[1] == "|=" or a[1] == "^=" or a[1] == "<<=" or a[1] == ">>="):
        if e[0] != "var" and e[0] != "idx" and e[0] != "attr":
            raise tok_line(a) + "bad assign target"
        st[1] = st[1] + 1
        mut v = parse_expr(st)
        if a[1] != "=":
            v = ["bin", a[1][0:a[1].len - 1], e, v]
        return ["assign", e, v]
    return ["expr", e]

fn parse_if(st):
    st[1] = st[1] + 1
    c = parse_expr(st)
    eat_op(st, ":")
    a = parse_suite(st)
    mut elifs = []
    while is_kw(st[0][st[1]], "elif"):
        st[1] = st[1] + 1
        c2 = parse_expr(st)
        eat_op(st, ":")
        b2 = parse_suite(st)
        elifs.append([c2, b2])
    mut b = nil
    if is_kw(st[0][st[1]], "else"):
        st[1] = st[1] + 1
        eat_op(st, ":")
        b = parse_suite(st)
    mut i = elifs.len - 1
    while i >= 0:
        b = [["if", elifs[i][0], elifs[i][1], b]]
        i = i - 1
    return ["if", c, a, b]

fn parse_expr(st):
    mut e = parse_or(st)
    t = st[0][st[1]]
    if is_kw(t, "if") and not is_bol(t):
        st[1] = st[1] + 1
        c = parse_or(st)
        if not is_kw(st[0][st[1]], "else"):
            raise tok_line(st[0][st[1]]) + "want else"
        st[1] = st[1] + 1
        e = ["cond", c, e, parse_expr(st)]
    return e

# token đầu dòng logic (host.scan gắn phần tử thứ 4)
fn is_bol(t):
    return t.len > 3 and t[3] == true

fn parse_or(st):
    e = parse_and(st)
    while is_kw(st[0][st[1]], "or"):
        st[1] = st[1] + 1
        e = ["bin", "or", e, parse_and(st)]
    return e

fn parse_and(st):
    e = parse_not(st)
    while is_kw(st[0][st[1]], "and"):
        st[1] = st[1] + 1
        e = ["bin", "and", e, parse_not(st)]
    return e

fn parse_not(st):
    if is_kw(st[0][st[1]], "not"):
        st[1] = st[1] + 1
        return ["un", "not", parse_not(st)]
    return parse_cmp(st)

fn parse_cmp(st):
    e = parse_bitor(st)
    mut go = true
    while go:
        t = st[0][st[1]]
        if t[0] == "OP" and (t[1] == "==" or t[1] == "!=" or t[1] == "<" or t[1] == ">" or t[1] == "<=" or t[1] == ">="):
            st[1] = st[1] + 1
            e = ["bin", t[1], e, parse_bitor(st)]
        elif is_kw(t, "in"):
            st[1] = st[1] + 1
            e = ["bin", "in", e, parse_bitor(st)]
        elif is_kw(t, "not") and is_kw(st[0][st[1] + 1], "in"):
            st[1] = st[1] + 2
            e = ["un", "not", ["bin", "in", e, parse_bitor(st)]]
        else:
            go = false
    return e

fn parse_bitor(st):
    e = parse_bitxor(st)
    while is_op(st[0][st[1]], "|"):
        st[1] = st[1] + 1
        e = ["bin", "|", e, parse_bitxor(st)]
    return e

fn parse_bitxor(st):
    e = parse_bitand(st)
    while is_op(st[0][st[1]], "^"):
        st[1] = st[1] + 1
        e = ["bin", "^", e, parse_bitand(st)]
    return e

fn parse_bitand(st):
    e = parse_shift(st)
    while is_op(st[0][st[1]], "&"):
        st[1] = st[1] + 1
        e = ["bin", "&", e, parse_shift(st)]
    return e

fn parse_shift(st):
    e = parse_add(st)
    while is_op(st[0][st[1]], "<<") or is_op(st[0][st[1]], ">>"):
        op = st[0][st[1]][1]
        st[1] = st[1] + 1
        e = ["bin", op, e, parse_add(st)]
    return e

fn parse_add(st):
    e = parse_mul(st)
    while is_op(st[0][st[1]], "+") or is_op(st[0][st[1]], "-"):
        op = st[0][st[1]][1]
        st[1] = st[1] + 1
        e = ["bin", op, e, parse_mul(st)]
    return e

fn parse_mul(st):
    e = parse_un(st)
    # @ cùng mức với * / % (như Python): X @ W + b == (X @ W) + b
    while is_op(st[0][st[1]], "*") or is_op(st[0][st[1]], "/") or is_op(st[0][st[1]], "%") or is_op(st[0][st[1]], "@"):
        op = st[0][st[1]][1]
        st[1] = st[1] + 1
        e = ["bin", op, e, parse_un(st)]
    return e

fn parse_un(st):
    if is_kw(st[0][st[1]], "await"):
        st[1] = st[1] + 1
        return parse_un(st)
    if is_op(st[0][st[1]], "-") or is_op(st[0][st[1]], "~"):
        op = st[0][st[1]][1]
        st[1] = st[1] + 1
        return ["un", op, parse_un(st)]
    return parse_pow(st)

fn parse_pow(st):
    e = parse_post(st)
    if is_op(st[0][st[1]], "**"):
        st[1] = st[1] + 1
        e = ["bin", "**", e, parse_pow(st)]
    return e

fn parse_post(st):
    e = parse_atom(st)
    mut go = true
    while go:
        if is_op(st[0][st[1]], "("):
            st[1] = st[1] + 1
            mut a = []
            if not is_op(st[0][st[1]], ")"):
                a.append(parse_expr(st))
                while is_op(st[0][st[1]], ","):
                    st[1] = st[1] + 1
                    a.append(parse_expr(st))
            eat_op(st, ")")
            e = ["call", e, a]
        elif is_op(st[0][st[1]], "[") and comma_ahead(st, st[1] + 1):
            # t[i, j] / t[:, j]: chỉ số nhiều trục -> t[[i, j]] (":" riêng = nil)
            st[1] = st[1] + 1
            mut items = []
            mut more = true
            while more:
                items.append(parse_axis_item(st))
                if is_op(st[0][st[1]], ","):
                    st[1] = st[1] + 1
                else:
                    more = false
            eat_op(st, "]")
            e = ["idx", e, ["list", items]]
        elif is_op(st[0][st[1]], "["):
            st[1] = st[1] + 1
            mut lo = nil
            mut hi = nil
            mut stp = nil
            mut is_slice = false
            if is_op(st[0][st[1]], ":"):
                is_slice = true
            else:
                lo = parse_expr(st)
                if is_op(st[0][st[1]], ":"):
                    is_slice = true
            if is_slice:
                eat_op(st, ":")
                if not is_op(st[0][st[1]], ":") and not is_op(st[0][st[1]], "]"):
                    hi = parse_expr(st)
                if is_op(st[0][st[1]], ":"):
                    st[1] = st[1] + 1
                    if not is_op(st[0][st[1]], "]"):
                        stp = parse_expr(st)
                eat_op(st, "]")
                e = ["slice", e, lo, hi, stp]
            else:
                eat_op(st, "]")
                e = ["idx", e, lo]
        elif is_op(st[0][st[1]], "."):
            st[1] = st[1] + 1
            f = st[0][st[1]][1]
            st[1] = st[1] + 1
            e = ["attr", e, f]
        elif is_op(st[0][st[1]], "?"):
            st[1] = st[1] + 1
            qv = gensym("q")
            qt = gensym("qt")
            hoist(["decl", "let", qv, e])
            hoist(["decl", "let", qt, ["bltin", "type", [["var", qv]]]])
            is_err_cond = ["bin", "or",
                ["bin", "==", ["var", qv], ["lit", "nil"]],
                ["bin", "or",
                    ["bin", "==", ["var", qt], ["str", "Err"]],
                    ["bin", "==", ["var", qt], ["str", "None"]]
                ]
            ]
            hoist(["if", is_err_cond, [["ret", ["var", qv]]], nil])
            is_wrap_cond = ["bin", "or",
                ["bin", "==", ["var", qt], ["str", "Ok"]],
                ["bin", "==", ["var", qt], ["str", "Some"]]
            ]
            e = ["cond", is_wrap_cond, ["attr", ["var", qv], "val"], ["var", qv]]
        else:
            go = false
    return e

# một trục trong chỉ số nhiều trục: số | nil (":") | ["list", [lo, hi, step]]
fn parse_axis_item(st):
    nilv = ["lit", "nil"]
    mut lo = nil
    if not is_op(st[0][st[1]], ":"):
        lo = parse_expr(st)
    if not is_op(st[0][st[1]], ":"):
        return lo
    st[1] = st[1] + 1
    mut hi = nil
    mut sp = nil
    t = st[0][st[1]]
    if not (is_op(t, ":") or is_op(t, ",") or is_op(t, "]")):
        hi = parse_expr(st)
    if is_op(st[0][st[1]], ":"):
        st[1] = st[1] + 1
        t2 = st[0][st[1]]
        if not (is_op(t2, ",") or is_op(t2, "]")):
            sp = parse_expr(st)
    if lo == nil and hi == nil and sp == nil:
        return nilv
    return ["list", [lo if lo != nil else nilv, hi if hi != nil else nilv, sp if sp != nil else nilv]]

# có ',' ở tầng ngoài cùng trước ']' khớp không? (bắt đầu từ chỉ số j)
fn comma_ahead(st, j):
    mut d = 0
    mut k = j
    while k < st[0].len:
        t = st[0][k]
        if t[0] == "EOF":
            return false
        if t[0] == "OP":
            v = t[1]
            if v == "(" or v == "[" or v == "{":
                d = d + 1
            elif v == ")" or v == "]" or v == "}":
                if d == 0:
                    return false
                d = d - 1
            elif v == "," and d == 0:
                return true
        k = k + 1
    return false

fn is_bltin_name(w):
    return w == "range" or w == "len" or w == "str" or w == "int" or w == "float" or w == "type" or w == "open" or w == "read" or w == "write" or w == "close" or w == "ord" or w == "chr" or w == "argv" or w == "mem" or w == "f64bits" or w == "abs" or w == "round" or w == "min" or w == "max" or w == "sum" or w == "any" or w == "all" or w == "sort" or w == "reversed" or w == "enumerate" or w == "zip" or w == "map" or w == "filter"

fn parse_atom(st):
    t = st[0][st[1]]
    if t[0] == "NUM":
        st[1] = st[1] + 1
        return ["num", t[1]]
    if t[0] == "STR":
        st[1] = st[1] + 1
        return ["str", t[1]]
    if t[0] == "FSTR":
        st[1] = st[1] + 1
        mut parts = []
        for p in t[1]:
            if p[0] == "s":
                parts.append(["s", p[1]])
            else:
                st2 = [p[1], 0]
                e = parse_expr(st2)
                if st2[0][st2[1]][0] != "EOF":
                    raise "trailing in f-string"
                parts.append(["e", e])
        return ["fstr", parts]
    if t[0] == "KW" and (t[1] == "true" or t[1] == "false" or t[1] == "nil"):
        st[1] = st[1] + 1
        return ["lit", t[1]]
    if t[0] == "KW" and t[1] == "tensor":
        st[1] = st[1] + 1
        eat_op(st, "(")
        e = parse_expr(st)
        if is_op(st[0][st[1]], ","):
            st[1] = st[1] + 1
            dt = parse_expr(st)
            eat_op(st, ")")
            return ["bltin", "tensor", [e, dt]]
        eat_op(st, ")")
        return ["tensor", e]
    if t[0] == "KW" and is_bltin_name(t[1]):
        st[1] = st[1] + 1
        eat_op(st, "(")
        mut a = []
        if not is_op(st[0][st[1]], ")"):
            a.append(parse_expr(st))
            while is_op(st[0][st[1]], ","):
                st[1] = st[1] + 1
                a.append(parse_expr(st))
        eat_op(st, ")")
        return ["bltin", t[1], a]
    if t[0] == "NAME" or (t[0] == "KW" and t[1] == "async"):
        st[1] = st[1] + 1
        return ["var", t[1]]
    if t[0] == "KW" and t[1] == "print":
        st[1] = st[1] + 1
        return ["bltfn", "print"]
    if is_op(t, "("):
        st[1] = st[1] + 1
        e = parse_expr(st)
        eat_op(st, ")")
        return e
    if is_kw(t, "fn") and is_op(st[0][st[1] + 1], "("):
        return parse_lambda(st)
    if is_op(t, "["):
        st[1] = st[1] + 1
        mut a = []
        PEND.append([])
        if not is_op(st[0][st[1]], "]"):
            a.append(parse_expr(st))
            if is_kw(st[0][st[1]], "for"):
                inner = PEND.pop()
                return parse_comp(st, a[0], nil, inner, "]")
            while is_op(st[0][st[1]], ","):
                st[1] = st[1] + 1
                a.append(parse_expr(st))
        eat_op(st, "]")
        merge_pending()
        return ["list", a]
    if is_op(t, "{"):
        st[1] = st[1] + 1
        mut ks = []
        mut vs = []
        PEND.append([])
        if not is_op(st[0][st[1]], "}"):
            ks.append(parse_expr(st))
            eat_op(st, ":")
            vs.append(parse_expr(st))
            if is_kw(st[0][st[1]], "for"):
                inner = PEND.pop()
                return parse_comp(st, vs[0], ks[0], inner, "}")
            while is_op(st[0][st[1]], ","):
                st[1] = st[1] + 1
                if is_op(st[0][st[1]], "}"):
                    break
                ks.append(parse_expr(st))
                eat_op(st, ":")
                vs.append(parse_expr(st))
        eat_op(st, "}")
        merge_pending()
        return ["dict", ks, vs]
    raise tok_line(t) + "unexpected token " + t[1]
