# std/http.t — Tiện ích URL, Query String & HTTP Client cho T
import os

fn _hex_digit(n):
    return "0123456789ABCDEF"[n:n + 1]

fn _from_hex(c):
    o = ord(c)
    if o >= 48 and o <= 57:
        return o - 48
    if o >= 65 and o <= 70:
        return o - 55
    if o >= 97 and o <= 102:
        return o - 87
    return -1

fn _is_unreserved(c):
    o = ord(c)
    return (o >= 48 and o <= 57) or (o >= 65 and o <= 90) or (o >= 97 and o <= 122) or c == "-" or c == "_" or c == "." or c == "~"

fn urlencode(s):
    mut out = ""
    for i in range(s.len):
        c = s[i:i + 1]
        if _is_unreserved(c):
            out = out + c
        elif c == " ":
            out = out + "+"
        else:
            b = ord(c) & 255
            out = out + "%" + _hex_digit((b >> 4) & 15) + _hex_digit(b & 15)
    return out

fn urldecode(s):
    mut out = ""
    mut i = 0
    while i < s.len:
        c = s[i:i + 1]
        if c == "+":
            out = out + " "
            i = i + 1
        elif c == "%" and i + 2 < s.len:
            h1 = _from_hex(s[i + 1:i + 2])
            h2 = _from_hex(s[i + 2:i + 3])
            if h1 >= 0 and h2 >= 0:
                out = out + chr((h1 << 4) | h2)
                i = i + 3
            else:
                out = out + c
                i = i + 1
        else:
            out = out + c
            i = i + 1
    return out

fn parse_query(qs):
    mut s = qs
    if s.startswith("?"):
        s = s[1:]
    mut out = {}
    if s == "":
        return out
    for part in s.split("&"):
        if part == "":
            continue
        eq = part.find("=")
        if eq < 0:
            out[urldecode(part)] = ""
        else:
            k = urldecode(part[0:eq])
            v = urldecode(part[eq + 1:])
            out[k] = v
    return out

fn build_query(params):
    mut parts = []
    for k in params.keys():
        parts.append(urlencode(str(k)) + "=" + urlencode(str(params[k])))
    return "&".join(parts)

fn parse_url(url):
    mut s = url
    mut scheme = "http"
    mut host = ""
    mut port = 80
    mut path = "/"
    mut query = ""
    mut fragment = ""
    frag_idx = s.find("#")
    if frag_idx >= 0:
        fragment = s[frag_idx + 1:]
        s = s[0:frag_idx]
    q_idx = s.find("?")
    if q_idx >= 0:
        query = s[q_idx + 1:]
        s = s[0:q_idx]
    sc_idx = s.find("://")
    if sc_idx >= 0:
        scheme = s[0:sc_idx].lower()
        s = s[sc_idx + 3:]
    port = 443 if scheme == "https" else 80
    sl_idx = s.find("/")
    mut authority = s
    if sl_idx >= 0:
        authority = s[0:sl_idx]
        path = s[sl_idx:]
    colon = authority.find(":")
    if colon >= 0:
        host = authority[0:colon]
        port = int(authority[colon + 1:])
    else:
        host = authority
    return {
        "scheme": scheme,
        "host": host,
        "port": port,
        "path": path,
        "query": query,
        "fragment": fragment
    }

fn request(method, url, body = "", headers = nil):
    mut cmd = ["curl", "-sS", "-i", "-X", method.upper()]
    if headers != nil:
        for k in headers.keys():
            cmd.append("-H")
            cmd.append(str(k) + ": " + str(headers[k]))
    if body != "":
        cmd.append("--data-raw")
        cmd.append(body)
    cmd.append(url)
    res = os.run(cmd)
    if res["code"] != 0:
        raise "http request failed: " + res["stderr"]
    raw = res["stdout"]
    mut sep = "\r\n\r\n"
    mut idx = raw.find(sep)
    if idx < 0:
        sep = "\n\n"
        idx = raw.find(sep)
    if idx < 0:
        return {"status": 200, "headers": {}, "body": raw}
    hdr_block = raw[0:idx]
    resp_body = raw[idx + sep.len:]
    lines = hdr_block.replace("\r", "").split("\n")
    mut status = 200
    if lines.len > 0:
        sp = lines[0].split(" ")
        if sp.len >= 2:
            status = int(sp[1])
    mut resp_headers = {}
    for i in range(1, lines.len):
        ln = lines[i]
        c = ln.find(":")
        if c > 0:
            resp_headers[ln[0:c].strip().lower()] = ln[c + 1:].strip()
    return {"status": status, "headers": resp_headers, "body": resp_body}

fn get(url, headers = nil):
    return request("GET", url, "", headers)

fn post(url, body, headers = nil):
    return request("POST", url, body, headers)
