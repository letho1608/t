# self/lex.t — lexer của T viết bằng chính T (v0.4 self-hosting bước 1)
# Input: source string. Output: list of [type, val, line] (val là str).
# Dùng: import lex ... lex.tokenize(src)

fn is_digit(ch):
    c = ord(ch)
    return c >= 48 and c <= 57

fn is_alpha(ch):
    c = ord(ch)
    return (c >= 65 and c <= 90) or (c >= 97 and c <= 122) or ch == "_"

fn is_alnum(ch):
    return is_alpha(ch) or is_digit(ch)

fn keywords():
    return ["let", "mut", "fn", "return", "if", "else", "elif", "for", "in",
        "while", "break", "continue", "true", "false", "nil", "and", "or",
        "not", "print", "tensor", "range", "len", "str", "int", "float",
        "type", "dict", "keys", "open", "read", "write", "close", "ord",
        "chr", "argv", "assert", "mem", "f64bits", "import", "from", "as", "try", "except", "finally",
        "raise", "abs", "round", "min", "max", "sum", "any", "all",
        "sort", "reversed", "enumerate", "zip", "map", "filter",
        "switch", "case", "default", "struct"]

# tập keyword dựng một lần (tra dict thay vì quét list mỗi định danh)
KWSET = {}
for k in keywords():
    KWSET[k] = true

fn is_kw(w):
    return w in KWSET

fn scan_fstr(src, i, n, line):
    q = src[i:i + 1]
    mut j = i + 1
    mut parts = []
    mut lit = ""
    while j < n:
        c = src[j:j + 1]
        if c == q:
            parts.append(["s", lit])
            return [parts, j + 1]
        if c == "\\" and j + 1 < n:
            e = src[j + 1:j + 2]
            if e == "n":
                lit = lit + "\n"
            elif e == "t":
                lit = lit + "\t"
            elif e == "r":
                lit = lit + "\r"
            elif e == "\\":
                lit = lit + "\\"
            elif e == q:
                lit = lit + q
            elif e == "{":
                lit = lit + "{"
            elif e == "}":
                lit = lit + "}"
            else:
                lit = lit + e
            j = j + 2
        elif c == "{" and j + 1 < n and src[j + 1:j + 2] == "{":
            lit = lit + "{"
            j = j + 2
        elif c == "}" and j + 1 < n and src[j + 1:j + 2] == "}":
            lit = lit + "}"
            j = j + 2
        elif c == "{":
            parts.append(["s", lit])
            lit = ""
            mut depth = 1
            mut k = j + 1
            mut qq = ""
            while k < n and depth > 0:
                d = src[k:k + 1]
                if qq != "":
                    if d == "\\":
                        k = k + 2
                    elif d == qq:
                        qq = ""
                        k = k + 1
                    else:
                        k = k + 1
                elif d == "\"" or d == "'":
                    qq = d
                    k = k + 1
                elif d == "(" or d == "[" or d == "{":
                    depth = depth + 1
                    k = k + 1
                elif d == ")" or d == "]" or d == "}":
                    depth = depth - 1
                    if depth == 0:
                        break
                    k = k + 1
                else:
                    k = k + 1
            if depth != 0:
                raise "unclosed { in f-string line " + str(line)
            parts.append(["e", tokenize(src[j + 1:k])])
            j = k + 1
        elif c == "}":
            raise "lone } in f-string line " + str(line)
        elif c == "\n":
            raise "f-string no newline line " + str(line)
        else:
            lit = lit + c
            j = j + 1
    raise "unterminated f-string line " + str(line)

fn tokenize(src):
    mut toks = []
    mut i = 0
    mut n = src.len
    mut line = 1
    while i < n:
        c = src[i]
        if c == "\n":
            line = line + 1
            i = i + 1
        elif c == " " or c == "\t" or c == "\r":
            i = i + 1
        elif c == "#":
            while i < n and src[i] != "\n":
                i = i + 1
        elif c == "\"" or c == "'":
            q = c
            mut s = ""
            mut j = i + 1
            while j < n and src[j] != q:
                if src[j] == "\\" and j + 1 < n:
                    e = src[j + 1]
                    if e == "n":
                        s = s + "\n"
                    elif e == "t":
                        s = s + "\t"
                    elif e == "r":
                        s = s + "\r"
                    elif e == "\\":
                        s = s + "\\"
                    else:
                        s = s + e
                    j = j + 2
                else:
                    s = s + src[j]
                    j = j + 1
            if j >= n:
                raise "unterminated string line " + str(line)
            toks.append(["STR", s, str(line)])
            i = j + 1
        elif is_digit(c):
            mut j = i
            mut dot = false
            mut nxt = ""
            if j + 1 < n:
                nxt = src[j + 1:j + 2]
            if c == "0" and (nxt == "x" or nxt == "X" or nxt == "b" or nxt == "B" or nxt == "o" or nxt == "O"):
                j = j + 2
                while j < n and is_alnum(src[j:j + 1]):
                    j = j + 1
            else:
                while j < n:
                    d1 = src[j:j + 1]
                    if is_digit(d1) or d1 == "_":
                        j = j + 1
                    elif d1 == "." and not dot:
                        dot = true
                        j = j + 1
                    else:
                        break
                # số mũ: 1e5, 2.5E-3
                if j < n and (src[j:j + 1] == "e" or src[j:j + 1] == "E"):
                    mut k = j + 1
                    if k < n and (src[k:k + 1] == "+" or src[k:k + 1] == "-"):
                        k = k + 1
                    if k < n and is_digit(src[k:k + 1]):
                        while k < n and is_digit(src[k:k + 1]):
                            k = k + 1
                        j = k
            toks.append(["NUM", src[i:j], str(line)])
            i = j
        elif is_alpha(c):
            mut j = i
            while j < n and is_alnum(src[j]):
                j = j + 1
            w = src[i:j]
            if (w == "f" or w == "F") and j < n and (src[j:j + 1] == "\"" or src[j:j + 1] == "'"):
                r = scan_fstr(src, j, n, line)
                toks.append(["FSTR", r[0], str(line)])
                i = r[1]
            elif is_kw(w):
                toks.append(["KW", w, str(line)])
                i = j
            else:
                toks.append(["NAME", w, str(line)])
                i = j
        else:
            three = src[i:i + 3]
            two = src[i:i + 2]
            if three == "<<=" or three == ">>=":
                toks.append(["OP", three, str(line)])
                i = i + 3
            elif two == "==" or two == "!=" or two == "<=" or two == ">=" or two == "**" or two == "->" or two == "+=" or two == "-=" or two == "*=" or two == "/=" or two == "%=" or two == "<<" or two == ">>" or two == "&=" or two == "|=" or two == "^=":
                toks.append(["OP", two, str(line)])
                i = i + 2
            elif c == "+" or c == "-" or c == "*" or c == "/" or c == "%" or c == "@" or c == "<" or c == ">" or c == "=" or c == "(" or c == ")" or c == ":" or c == "," or c == "[" or c == "]" or c == "{" or c == "}" or c == "." or c == "&" or c == "|" or c == "^" or c == "~":
                toks.append(["OP", c, str(line)])
                i = i + 1
            else:
                raise "bad char line " + str(line)
    toks.append(["EOF", "", str(line)])
    return toks
