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

fn is_op2(t, v):
    return t[0] == "OP" and t[1] == v

fn is_kw2(t, w):
    return t[0] == "KW" and t[1] == w

fn eat_op2(st, v):
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
    if t[0] != "NAME":
        raise tok_line(t) + "want NAME"
    st[1] = st[1] + 1
    return t[1]

fn parse_names(st):
    mut ns = [eat_name(st)]
    while is_op(st[0][st[1]], ","):
        st[1] = st[1] + 1
        ns.append(eat_name(st))
    return ns

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
            mut vs = [parse_expr(st)]
            while is_op(st[0][st[1]], ","):
                st[1] = st[1] + 1
                vs.append(parse_expr(st))
            eat_op(st, ":")
            cases.append([vs, parse_suite(st)])
        elif is_kw(t, "default"):
            st[1] = st[1] + 1
            eat_op(st, ":")
            dflt = parse_suite(st)
        else:
            raise tok_line(t) + "want case/default"
    eat_ded(st)
    tmp = gensym("s")
    mut b = dflt
    mut i = cases.len - 1
    while i >= 0:
        mut c = nil
        for v in cases[i][0]:
            eq = ["bin", "==", ["var", tmp], v]
            if c == nil:
                c = eq
            else:
                c = ["bin", "or", c, eq]
        b = [["if", c, cases[i][1], b]]
        i = i - 1
    mut out = [["decl", "let", tmp, subj]]
    if b != nil:
        if cases.len > 0:
            out.append(b[0])
        else:
            out.append(["if", ["lit", "true"], b, nil])
    return ["multi", out]

fn parse_struct(st):
    st[1] = st[1] + 1
    name = eat_name(st)
    eat_op(st, ":")
    eat_ind(st)
    mut fields = []
    mut meths = []
    while not is_ded(st):
        t = st[0][st[1]]
        if is_kw(t, "fn"):
            meths.append(parse_stmt(st))
        elif t[0] == "NAME":
            for f in parse_names(st):
                for g in fields:
                    if g == f:
                        raise tok_line(t) + "trùng field " + f
                fields.append(f)
        else:
            raise tok_line(t) + "want field/fn in struct"
    eat_ded(st)
    mut out = []
    for m in meths:
        out.append(["fn", name + "__" + m[1], m[2], m[3]])
    mut fstrs = []
    mut fvars = []
    for f in fields:
        fstrs.append(["str", f])
        fvars.append(["var", f])
    mut mks = []
    mut mvs = []
    for m in meths:
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
    if not is_op(st[0][st[1]], ")"):
        args.append(eat_name(st))
        while is_op(st[0][st[1]], ","):
            st[1] = st[1] + 1
            args.append(eat_name(st))
    eat_op(st, ")")
    eat_op(st, ":")
    PEND.append([])
    body = parse_expr(st)
    mut fb = PEND.pop()
    fb.append(["ret", body])
    name = gensym("l")
    hoist(["fn", name, args, fb])
    return ["var", name]

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
        if t[1] == "let" or t[1] == "mut":
            st[1] = st[1] + 1
            nm = st[0][st[1]][1]
            st[1] = st[1] + 1
            eat_op(st, "=")
            return ["decl", t[1], nm, parse_expr(st)]
        if t[1] == "fn":
            st[1] = st[1] + 1
            nm = st[0][st[1]][1]
            st[1] = st[1] + 1
            eat_op(st, "(")
            mut args = []
            if not is_op(st[0][st[1]], ")"):
                args.append(st[0][st[1]][1])
                st[1] = st[1] + 1
                while is_op(st[0][st[1]], ","):
                    st[1] = st[1] + 1
                    args.append(st[0][st[1]][1])
                    st[1] = st[1] + 1
            eat_op(st, ")")
            eat_op(st, ":")
            return ["fn", nm, args, parse_suite(st)]
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
    if a[0] == "OP" and (a[1] == "=" or a[1] == "+=" or a[1] == "-=" or a[1] == "*=" or a[1] == "/=" or a[1] == "%="):
        if e[0] != "var" and e[0] != "idx" and e[0] != "attr":
            raise tok_line(a) + "bad assign target"
        st[1] = st[1] + 1
        mut v = parse_expr(st)
        if a[1] != "=":
            v = ["bin", a[1][0:1], e, v]
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
    e = parse_add(st)
    mut go = true
    while go:
        t = st[0][st[1]]
        if t[0] == "OP" and (t[1] == "==" or t[1] == "!=" or t[1] == "<" or t[1] == ">" or t[1] == "<=" or t[1] == ">="):
            st[1] = st[1] + 1
            e = ["bin", t[1], e, parse_add(st)]
        elif is_kw(t, "in"):
            st[1] = st[1] + 1
            e = ["bin", "in", e, parse_add(st)]
        elif is_kw(t, "not") and is_kw(st[0][st[1] + 1], "in"):
            st[1] = st[1] + 2
            e = ["un", "not", ["bin", "in", e, parse_add(st)]]
        else:
            go = false
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
    if is_op(st[0][st[1]], "-"):
        st[1] = st[1] + 1
        return ["un", "-", parse_un(st)]
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
    if t[0] == "NAME":
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
