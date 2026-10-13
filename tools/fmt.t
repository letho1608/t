# tools/fmt.t — formatter code T viết bằng chính T (v0.9)
# Chuẩn: tab nở 4 spaces, indent 4 spaces/level, rstrip dòng, collapse
# blank thừa, giữ comment. Dùng:
#   import fmt
#   out = fmt.fmt_src(src)
#   fmt.fmt_file("tests/demo.t", "tests/demo.t")
import parse

fn rstrip(ln):
    mut i = ln.len - 1
    while i >= 0 and ln[i:i + 1] == " ":
        i = i - 1
    if i < 0:
        return ""
    return ln[0:i + 1]

fn indent_of(ln):
    mut ind = 0
    mut j = 0
    while j < ln.len and ln[j:j + 1] == " ":
        ind = ind + 1
        j = j + 1
    return ind

fn pad(level):
    mut s = ""
    mut i = 0
    while i < level:
        s = s + "    "
        i = i + 1
    return s

fn fmt_src(src):
    mut out = []
    mut stack = [0]
    mut level = 0
    mut depth = 0
    mut pending_blank = false
    mut lines = src.split("\n")
    for raw in lines:
        ln = "    ".join(raw.split("\t"))
        ln = rstrip(ln)
        st = ln.strip()
        if st == "":
            pending_blank = true
            continue
        if depth > 0:
            if pending_blank and out.len > 0:
                out.append("")
            pending_blank = false
            out.append(pad(level + 1) + st)
            depth = depth + parse.bracket_delta(ln)
            if depth < 0:
                depth = 0
            continue
        ind = indent_of(ln)
        while ind < stack[stack.len - 1]:
            stack.pop()
            level = level - 1
        if ind > stack[stack.len - 1]:
            stack.append(ind)
            level = level + 1
        if pending_blank and out.len > 0:
            out.append("")
        pending_blank = false
        out.append(pad(level) + st)
        depth = depth + parse.bracket_delta(ln)
        if depth < 0:
            depth = 0
    return "\n".join(out)

fn fmt_file(inp, outp):
    f = open(inp, "r")
    src = read(f)
    close(f)
    res = fmt_src(src)
    g = open(outp, "w")
    write(g, res + "\n")
    close(g)
    return res
