# std/csv.t — Phân tích & xuất dữ liệu CSV chuẩn RFC 4180 cho T
import os

fn parse_row(line, delim = ","):
    mut out = []
    mut cur = ""
    mut in_q = false
    mut i = 0
    n = line.len
    while i < n:
        c = line[i:i + 1]
        if in_q:
            if c == "\"":
                if i + 1 < n and line[i + 1:i + 2] == "\"":
                    cur = cur + "\""
                    i = i + 2
                else:
                    in_q = false
                    i = i + 1
            else:
                cur = cur + c
                i = i + 1
        else:
            if c == "\"" and cur == "":
                in_q = true
                i = i + 1
            elif c == delim:
                out.append(cur)
                cur = ""
                i = i + 1
            elif c == "\r":
                i = i + 1
            else:
                cur = cur + c
                i = i + 1
    out.append(cur)
    return out

fn format_row(fields, delim = ","):
    mut parts = []
    for f in fields:
        s = str(f) if f != nil else ""
        if delim in s or "\"" in s or "\n" in s or "\r" in s:
            parts.append("\"" + s.replace("\"", "\"\"") + "\"")
        else:
            parts.append(s)
    return delim.join(parts)

fn loads(text, delim = ","):
    mut rows = []
    for raw in text.split("\n"):
        ln = raw.replace("\r", "")
        if ln == "":
            continue
        rows.append(parse_row(ln, delim))
    return rows

fn dumps(rows, delim = ","):
    mut lines = []
    for r in rows:
        lines.append(format_row(r, delim))
    return "\n".join(lines) + ("\n" if lines.len > 0 else "")

fn loads_dicts(text, delim = ","):
    rows = loads(text, delim)
    if rows.len == 0:
        return []
    hdr = rows[0]
    mut out = []
    for i in range(1, rows.len):
        r = rows[i]
        mut d = {}
        for j in range(hdr.len):
            d[hdr[j]] = r[j] if j < r.len else ""
        out.append(d)
    return out

fn dumps_dicts(dicts, headers, delim = ","):
    mut rows = [headers]
    for d in dicts:
        mut r = []
        for h in headers:
            r.append(d.get(h, ""))
        rows.append(r)
    return dumps(rows, delim)

fn read_csv(path, delim = ","):
    return loads(os.read_file(path), delim)

fn write_csv(path, rows, delim = ","):
    return os.write_file(path, dumps(rows, delim))
