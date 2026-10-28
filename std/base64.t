# std/base64.t — RFC 4648 Base64 Encoder / Decoder cho T

let B64_CHARS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"

fn _build_decode_table():
    mut table = {}
    mut i = 0
    while i < len(B64_CHARS):
        let c = B64_CHARS[i]
        table[c] = i
        i += 1
    return table

let _DECODE_TABLE = _build_decode_table()

# Mã hóa chuỗi thành Base64
fn b64encode(s):
    let n = len(s)
    if n == 0:
        return ""

    mut out = []
    mut i = 0
    let full_triples = int(n / 3) * 3

    while i < full_triples:
        let b0 = ord(s[i])
        let b1 = ord(s[i + 1])
        let b2 = ord(s[i + 2])
        let triple = (b0 << 16) | (b1 << 8) | b2

        let c0 = B64_CHARS[(triple >> 18) & 63]
        let c1 = B64_CHARS[(triple >> 12) & 63]
        let c2 = B64_CHARS[(triple >> 6) & 63]
        let c3 = B64_CHARS[triple & 63]

        out.append(c0 + c1 + c2 + c3)
        i += 3

    let rem = n - full_triples
    if rem == 1:
        let b0 = ord(s[i])
        let triple = b0 << 16
        let c0 = B64_CHARS[(triple >> 18) & 63]
        let c1 = B64_CHARS[(triple >> 12) & 63]
        out.append(c0 + c1 + "==")
    elif rem == 2:
        let b0 = ord(s[i])
        let b1 = ord(s[i + 1])
        let triple = (b0 << 16) | (b1 << 8)
        let c0 = B64_CHARS[(triple >> 18) & 63]
        let c1 = B64_CHARS[(triple >> 12) & 63]
        let c2 = B64_CHARS[(triple >> 6) & 63]
        out.append(c0 + c1 + c2 + "=")

    return "".join(out)

# Giải mã chuỗi Base64
fn b64decode(s):
    let n = len(s)
    if n == 0:
        return ""

    # Lọc bỏ khoảng trắng / xuống dòng
    mut clean = []
    mut i = 0
    while i < n:
        let c = s[i]
        if c != " " and c != "\n" and c != "\r" and c != "\t":
            clean.append(c)
        i += 1

    let clen = len(clean)
    if clen % 4 != 0:
        raise "Invalid base64 string length"

    mut out = []
    mut j = 0
    while j < clen:
        let c0 = clean[j]
        let c1 = clean[j + 1]
        let c2 = clean[j + 2]
        let c3 = clean[j + 3]

        let v0 = _DECODE_TABLE.get(c0, nil)
        let v1 = _DECODE_TABLE.get(c1, nil)
        if v0 == nil or v1 == nil:
            raise "Invalid character in base64 string"

        if c2 == "=":
            # Chỉ có 1 byte hợp lệ
            let triple = (v0 << 18) | (v1 << 12)
            out.append(chr((triple >> 16) & 255))
        elif c3 == "=":
            # Có 2 bytes hợp lệ
            let v2 = _DECODE_TABLE.get(c2, nil)
            if v2 == nil:
                raise "Invalid character in base64 string"
            let triple = (v0 << 18) | (v1 << 12) | (v2 << 6)
            out.append(chr((triple >> 16) & 255))
            out.append(chr((triple >> 8) & 255))
        else:
            # Đủ 3 bytes
            let v2 = _DECODE_TABLE.get(c2, nil)
            let v3 = _DECODE_TABLE.get(c3, nil)
            if v2 == nil or v3 == nil:
                raise "Invalid character in base64 string"
            let triple = (v0 << 18) | (v1 << 12) | (v2 << 6) | v3
            out.append(chr((triple >> 16) & 255))
            out.append(chr((triple >> 8) & 255))
            out.append(chr(triple & 255))

        j += 4

    return "".join(out)
