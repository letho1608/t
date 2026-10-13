# self/parse.t — parser của T viết bằng chính T (v0.4 self-hosting bước 2)
# Input: token list từ lex.tokenize (sau indent_scan). Output: AST tuple/list.
# AST dùng list T: ["num", v], ["bin", op, l, r], ... ("prog" ở gốc)

fn peek(toks, i):
    return toks[i]

fn is_op(t, v):
    return t[0] == "OP" and t[1] == v

fn is_kw(t, w):
    return t[0] == "KW" and t[1] == w

# --- indent scan: source -> source với <IND>/<DED> ---
# Bracket-aware + string-aware (khớp t.py): dòng nối trong (... [...] {...})
# không gây IND/DED; '#' trong string được giữ; tab nở thành 4 spaces.
fn strip_comment(ln):
    if ln.find("#") < 0:
        return ln       # không có '#': vòng dưới cũng trả nguyên dòng
    mut q = ""
    mut i = 0
    mut n = ln.len
    while i < n:
        c = ln[i:i + 1]
        if q != "":
            if c == "\\":
                i = i + 2
            elif c == q:
                q = ""
                i = i + 1
            else:
                i = i + 1
        elif c == "\"" or c == "'":
            q = c
            i = i + 1
        elif c == "#":
            return ln[0:i]
        else:
            i = i + 1
    return ln

fn bracket_delta(code):
    # không nháy / '#': đếm ngoặc thẳng (split của host), khỏi quét từng ký tự
    if code.find("\"") < 0 and code.find("'") < 0 and code.find("#") < 0:
        return code.split("(").len + code.split("[").len + code.split("{").len - code.split(")").len - code.split("]").len - code.split("}").len
    mut q = ""
    mut d = 0
    mut i = 0
    mut n = code.len
    while i < n:
        c = code[i:i + 1]
        if q != "":
            if c == "\\":
                i = i + 2
            elif c == q:
                q = ""
                i = i + 1
            else:
                i = i + 1
        elif c == "\"" or c == "'":
            q = c
            i = i + 1
        elif c == "#":
            i = n
        elif c == "(" or c == "[" or c == "{":
            d = d + 1
            i = i + 1
        elif c == ")" or c == "]" or c == "}":
            d = d - 1
            i = i + 1
        else:
            i = i + 1
    return d

# indent_lines: list [số dòng gốc, text, là-dòng-nối] (marker mang số dòng của code kế)
fn indent_lines(src):
    mut out = []
    mut stack = [0]
    mut depth = 0
    mut no = 0
    mut lines = src.split("\n")
    for raw in lines:
        no = no + 1
        ln = "    ".join(raw.split("\t"))
        code = strip_comment(ln)
        mut st = code.strip()
        if st == "":
            continue
        if st[0:1] == "#":
            continue
        if depth > 0:
            out.append([no, st, true])
            depth = depth + bracket_delta(code)
            if depth < 0:
                depth = 0
            continue
        mut ind = 0
        mut j = 0
        while j < ln.len and ln[j:j + 1] == " ":
            ind = ind + 1
            j = j + 1
        if ind > stack[stack.len - 1]:
            stack.append(ind)
            out.append([no, "<IND>", false])
        while ind < stack[stack.len - 1]:
            stack.pop()
            out.append([no, "<DED>", false])
        out.append([no, st, false])
        depth = depth + bracket_delta(code)
        if depth < 0:
            depth = 0
    while stack.len > 1:
        stack.pop()
        out.append([no, "<DED>", false])
    return out

fn indent_scan(src):
    mut out = []
    for p in indent_lines(src):
        out.append(p[1])
    return "\n".join(out)

# --- parser state: [toks, pos] ---
# (parse_stmt sống ở parse2.t; entry full chương trình là host.parse_prog
#  trong host.t — không định nghĩa parse_prog ở đây để tránh ref treo.)
fn eat_op(st, v):
    t = st[0][st[1]]
    if t[0] != "OP" or t[1] != v:
        raise "want " + v
    st[1] = st[1] + 1
    return t
