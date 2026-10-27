# self/parse.t — tiền xử lý thụt lề của T viết bằng chính T
# Input: mã nguồn T. Output: danh sách [số_dòng, dòng_logic, là_dòng_nối] cho host.scan.

# --- indent scan: source -> danh sách dòng logic kèm <IND>/<DED> ---
# Bracket-aware + string-aware: dòng nối trong (... [...] {...})
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

