# std/json.t — bộ phân tích và tuần tự hoá JSON thuần T (chuẩn RFC 8259)

fn _is_ws(c):
    return c == " " or c == "\t" or c == "\n" or c == "\r"

fn _is_digit(c):
    o = ord(c)
    return o >= 48 and o <= 57

fn _skip_ws(s, i, n):
    mut p = i
    while p < n and _is_ws(s[p]):
        p = p + 1
    return p

fn _hex_val(c):
    o = ord(c)
    if o >= 48 and o <= 57:
        return o - 48
    if o >= 65 and o <= 70:
        return o - 55
    if o >= 97 and o <= 102:
        return o - 87
    raise "JSON: mã hex không hợp lệ"

fn _utf8_from_cp(cp):
    if cp <= 127:
        return chr(cp)
    if cp <= 2047:
        return chr(192 + int(cp / 64)) + chr(128 + (cp % 64))
    return chr(224 + int(cp / 4096)) + chr(128 + (int(cp / 64) % 64)) + chr(128 + (cp % 64))

fn _parse_str(s, i, n):
    if s[i] != "\"":
        raise "JSON: thiếu dấu mở chuỗi"
    mut p = i + 1
    mut parts = []
    mut start = p
    while p < n:
        c = s[p]
        if c == "\"":
            if p > start:
                parts.append(s[start:p])
            return ["".join(parts), p + 1]
        if c == "\\":
            if p > start:
                parts.append(s[start:p])
            p = p + 1
            if p >= n:
                raise "JSON: chuỗi kết thúc đột ngột sau \\"
            esc = s[p]
            if esc == "\"":
                parts.append("\"")
                p = p + 1
            elif esc == "\\":
                parts.append("\\")
                p = p + 1
            elif esc == "/":
                parts.append("/")
                p = p + 1
            elif esc == "b":
                parts.append(chr(8))
                p = p + 1
            elif esc == "f":
                parts.append(chr(12))
                p = p + 1
            elif esc == "n":
                parts.append("\n")
                p = p + 1
            elif esc == "r":
                parts.append("\r")
                p = p + 1
            elif esc == "t":
                parts.append("\t")
                p = p + 1
            elif esc == "u":
                if p + 4 >= n:
                    raise "JSON: mã \\uXXXX thiếu ký tự"
                cp = (_hex_val(s[p + 1]) * 4096 +
                      _hex_val(s[p + 2]) * 256 +
                      _hex_val(s[p + 3]) * 16 +
                      _hex_val(s[p + 4]))
                parts.append(_utf8_from_cp(cp))
                p = p + 5
            else:
                raise "JSON: ký tự thoát không hợp lệ: \\" + esc
            start = p
        else:
            if ord(c) < 32:
                raise "JSON: ký tự điều khiển trong chuỗi"
            p = p + 1
    raise "JSON: chuỗi chưa đóng"

fn _parse_num(s, i, n):
    mut p = i
    mut is_flt = false
    if p < n and s[p] == "-":
        p = p + 1
    if p >= n or not _is_digit(s[p]):
        raise "JSON: số không hợp lệ"
    if s[p] == "0":
        p = p + 1
    else:
        while p < n and _is_digit(s[p]):
            p = p + 1
    if p < n and s[p] == ".":
        is_flt = true
        p = p + 1
        if p >= n or not _is_digit(s[p]):
            raise "JSON: phần thập phân rỗng"
        while p < n and _is_digit(s[p]):
            p = p + 1
    if p < n and (s[p] == "e" or s[p] == "E"):
        is_flt = true
        p = p + 1
        if p < n and (s[p] == "+" or s[p] == "-"):
            p = p + 1
        if p >= n or not _is_digit(s[p]):
            raise "JSON: số mũ rỗng"
        while p < n and _is_digit(s[p]):
            p = p + 1
    raw = s[i:p]
    return [float(raw) if is_flt else int(raw), p]

fn _parse_val(s, i, n):
    mut p = _skip_ws(s, i, n)
    if p >= n:
        raise "JSON: dữ liệu rỗng hoặc kết thúc bất ngờ"
    c = s[p]
    if c == "\"":
        return _parse_str(s, p, n)
    if c == "-" or _is_digit(c):
        return _parse_num(s, p, n)
    if c == "{":
        p = _skip_ws(s, p + 1, n)
        mut obj = {}
        if p < n and s[p] == "}":
            return [obj, p + 1]
        while true:
            p = _skip_ws(s, p, n)
            if p >= n or s[p] != "\"":
                raise "JSON: khoá object phải là chuỗi"
            kr = _parse_str(s, p, n)
            key = kr[0]
            p = _skip_ws(s, kr[1], n)
            if p >= n or s[p] != ":":
                raise "JSON: thiếu dấu ':' sau khoá object"
            vr = _parse_val(s, p + 1, n)
            obj[key] = vr[0]
            p = _skip_ws(s, vr[1], n)
            if p >= n:
                raise "JSON: object chưa đóng"
            if s[p] == "}":
                return [obj, p + 1]
            if s[p] != ",":
                raise "JSON: thiếu dấu ',' hoặc '}' trong object"
            p = p + 1
    if c == "[":
        p = _skip_ws(s, p + 1, n)
        mut arr = []
        if p < n and s[p] == "]":
            return [arr, p + 1]
        while true:
            vr = _parse_val(s, p, n)
            arr.append(vr[0])
            p = _skip_ws(s, vr[1], n)
            if p >= n:
                raise "JSON: mảng chưa đóng"
            if s[p] == "]":
                return [arr, p + 1]
            if s[p] != ",":
                raise "JSON: thiếu dấu ',' hoặc ']' trong mảng"
            p = p + 1
    if p + 4 <= n and s[p:p + 4] == "true":
        return [true, p + 4]
    if p + 5 <= n and s[p:p + 5] == "false":
        return [false, p + 5]
    if p + 4 <= n and s[p:p + 4] == "null":
        return [nil, p + 4]
    raise "JSON: giá trị không hợp lệ tại vị trí " + str(p)

fn loads(s):
    if type(s) != "str":
        raise "JSON loads: đầu vào phải là str"
    n = len(s)
    r = _parse_val(s, 0, n)
    end = _skip_ws(s, r[1], n)
    if end != n:
        raise "JSON: dữ liệu thừa sau giá trị gốc"
    return r[0]

fn _escape_str(s):
    mut out = ["\""]
    n = len(s)
    mut start = 0
    mut i = 0
    while i < n:
        c = s[i]
        o = ord(c)
        mut rep = nil
        if c == "\"":
            rep = "\\\""
        elif c == "\\":
            rep = "\\\\"
        elif c == "\n":
            rep = "\\n"
        elif c == "\r":
            rep = "\\r"
        elif c == "\t":
            rep = "\\t"
        elif o == 8:
            rep = "\\b"
        elif o == 12:
            rep = "\\f"
        elif o < 32:
            hex = "0123456789abcdef"
            rep = "\\u00" + hex[int(o / 16)] + hex[o % 16]
        if rep != nil:
            if i > start:
                out.append(s[start:i])
            out.append(rep)
            start = i + 1
        i = i + 1
    if n > start:
        out.append(s[start:n])
    out.append("\"")
    return "".join(out)

fn _spaces(n):
    if n <= 0:
        return ""
    return "".join([" "] * n)

fn _dump_rec(v, indent, depth):
    t = type(v)
    if t == "nil":
        return "null"
    if t == "bool":
        return "true" if v else "false"
    if t == "int" or t == "float":
        return str(v)
    if t == "str":
        return _escape_str(v)
    if t == "tensor":
        return _dump_rec(v.tolist(), indent, depth)
    if t == "list":
        if v.len == 0:
            return "[]"
        mut items = []
        for elem in v:
            items.append(_dump_rec(elem, indent, depth + 1))
        if indent <= 0:
            return "[" + ", ".join(items) + "]"
        pad = _spaces(indent * (depth + 1))
        end_pad = _spaces(indent * depth)
        return "[\n" + pad + (",\n" + pad).join(items) + "\n" + end_pad + "]"
    if t == "dict":
        ks = v.keys()
        if ks.len == 0:
            return "{}"
        mut pairs = []
        sep = ": "
        for k in ks:
            ks_str = _escape_str(str(k))
            vs_str = _dump_rec(v[k], indent, depth + 1)
            pairs.append(ks_str + sep + vs_str)
        if indent <= 0:
            return "{" + ", ".join(pairs) + "}"
        pad = _spaces(indent * (depth + 1))
        end_pad = _spaces(indent * depth)
        return "{\n" + pad + (",\n" + pad).join(pairs) + "\n" + end_pad + "}"
    raise "JSON dumps: kiểu không hỗ trợ: " + t

fn dumps(v, indent = 0):
    return _dump_rec(v, indent, 0)
