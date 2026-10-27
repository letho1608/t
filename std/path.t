# std/path.t — xử lý đường dẫn đa nền tảng cho T

let _sys = load("__sys")

fn sep():
    return "\\" if _sys.platform() == "win32" else "/"

fn to_slash(p):
    return p.replace("\\", "/")

fn isabs(p):
    if len(p) == 0:
        return false
    if p[0] == "/" or p[0] == "\\":
        return true
    if len(p) >= 3 and p[1] == ":" and (p[2] == "/" or p[2] == "\\"):
        return true
    return false

fn normpath(p):
    if len(p) == 0:
        return "."
    s = p.replace("\\", "/")
    mut prefix = ""
    mut rest = s
    if len(s) >= 2 and s[1] == ":":
        prefix = s[0:2]
        rest = s[2:len(s)]
        if len(rest) > 0 and rest[0] == "/":
            prefix = prefix + "/"
            rest = rest[1:len(rest)]
    elif s[0] == "/":
        prefix = "/"
        rest = s[1:len(s)]
    parts = rest.split("/")
    mut out = []
    for part in parts:
        if part == "" or part == ".":
            continue
        elif part == "..":
            if out.len > 0 and out[out.len - 1] != "..":
                out.pop()
            elif prefix == "" or (len(prefix) == 2 and prefix[1] == ":"):
                out.append("..")
        else:
            out.append(part)
    res = prefix + "/".join(out)
    return res if res != "" else "."

fn join(a, b, c = nil, d = nil):
    mut res = a
    for part in [b, c, d]:
        if part != nil and part != "":
            if res == "" or isabs(part):
                res = part
            elif res.endswith("/") or res.endswith("\\"):
                res = res + part
            else:
                res = res + "/" + part
    return res

fn basename(p):
    s = p.replace("\\", "/")
    mut end = len(s)
    while end > 1 and s[end - 1] == "/":
        end = end - 1
    sub = s[0:end]
    parts = sub.split("/")
    return parts[parts.len - 1]

fn dirname(p):
    s = p.replace("\\", "/")
    mut end = len(s)
    while end > 1 and s[end - 1] == "/":
        end = end - 1
    sub = s[0:end]
    idx = sub.find("/")
    if idx < 0:
        return "."
    parts = sub.split("/")
    if parts.len <= 1:
        return "."
    mut head = []
    for i in range(parts.len - 1):
        head.append(parts[i])
    res = "/".join(head)
    return "/" if res == "" and sub[0] == "/" else res

fn ext(p):
    base = basename(p)
    parts = base.split(".")
    if parts.len <= 1 or (parts.len == 2 and parts[0] == ""):
        return ""
    return "." + parts[parts.len - 1]

fn stem(p):
    base = basename(p)
    e = ext(base)
    if e == "":
        return base
    return base[0:len(base) - len(e)]

fn abspath(p):
    return _sys.abspath(p)

fn exists(p):
    return _sys.exists(p)

fn isfile(p):
    return _sys.isfile(p)

fn isdir(p):
    return _sys.isdir(p)
