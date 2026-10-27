# std/re.t — Thư viện biểu thức chính quy (Regular Expressions) thuần T
# Hỗ trợ: ^, $, ., *, +, ?, {m}, {m,}, {m,n}, [abc], [a-z], [^...], (a|b),
#         \d, \D, \w, \W, \s, \S, \b, và các ký tự escape (\., \*, \+, \?, \\...)

fn _is_digit(c):
    o = ord(c)
    return o >= 48 and o <= 57

fn _is_word(c):
    o = ord(c)
    return (o >= 48 and o <= 57) or (o >= 65 and o <= 90) or (o >= 97 and o <= 122) or c == "_"

fn _is_space(c):
    return c == " " or c == "\t" or c == "\n" or c == "\r" or c == "\x0b" or c == "\x0c"

fn _parse_int(s):
    mut v = 0
    if s.len == 0:
        return -1
    for i in range(s.len):
        c = s[i:i + 1]
        if not _is_digit(c):
            return -1
        v = v * 10 + (ord(c) - 48)
    return v

fn _parse_class(pat, i):
    mut j = i
    mut neg = false
    if j < pat.len and pat[j:j + 1] == "^":
        neg = true
        j = j + 1
    mut items = []
    while j < pat.len and pat[j:j + 1] != "]":
        mut c = pat[j:j + 1]
        if c == "\\" and j + 1 < pat.len:
            esc = pat[j + 1:j + 2]
            if esc == "d" or esc == "D" or esc == "w" or esc == "W" or esc == "s" or esc == "S":
                items.append(["esc", esc])
                j = j + 2
                continue
            elif esc == "n":
                c = "\n"
            elif esc == "t":
                c = "\t"
            elif esc == "r":
                c = "\r"
            else:
                c = esc
            j = j + 2
        else:
            j = j + 1
        if j + 1 < pat.len and pat[j:j + 1] == "-" and pat[j + 1:j + 2] != "]":
            mut c2 = pat[j + 1:j + 2]
            if c2 == "\\" and j + 2 < pat.len:
                c2 = pat[j + 2:j + 3]
                j = j + 3
            else:
                j = j + 2
            items.append(["rng", ord(c), ord(c2)])
        else:
            items.append(["ch", c])
    if j >= pat.len:
        raise "re: unclosed ["
    return [["class", neg, items], j + 1]

fn _parse_seq(pat, i, stop_ch, gid_box):
    mut branches = []
    mut cur = []
    mut j = i
    while j < pat.len:
        c = pat[j:j + 1]
        if c == stop_ch:
            break
        if c == "|":
            branches.append(["seq", cur])
            cur = []
            j = j + 1
            continue
        mut node = nil
        if c == "^":
            node = ["bol"]
            j = j + 1
        elif c == "$":
            node = ["eol"]
            j = j + 1
        elif c == ".":
            node = ["dot"]
            j = j + 1
        elif c == "[":
            r = _parse_class(pat, j + 1)
            node = r[0]
            j = r[1]
        elif c == "(":
            gid_box[0] = gid_box[0] + 1
            my_gid = gid_box[0]
            r = _parse_seq(pat, j + 1, ")", gid_box)
            if r[1] >= pat.len or pat[r[1]:r[1] + 1] != ")":
                raise "re: unclosed ("
            node = ["grp", my_gid, r[0]]
            j = r[1] + 1
        elif c == "\\" and j + 1 < pat.len:
            esc = pat[j + 1:j + 2]
            if esc == "d" or esc == "D" or esc == "w" or esc == "W" or esc == "s" or esc == "S" or esc == "b":
                node = ["esc", esc]
            elif esc == "n":
                node = ["lit", "\n"]
            elif esc == "t":
                node = ["lit", "\t"]
            elif esc == "r":
                node = ["lit", "\r"]
            else:
                node = ["lit", esc]
            j = j + 2
        else:
            node = ["lit", c]
            j = j + 1
        # Hậu tố lặp: *, +, ?, {m,n}
        if j < pat.len:
            q = pat[j:j + 1]
            if q == "*":
                node = ["rep", node, 0, -1]
                j = j + 1
            elif q == "+":
                node = ["rep", node, 1, -1]
                j = j + 1
            elif q == "?":
                node = ["rep", node, 0, 1]
                j = j + 1
            elif q == "{":
                mut k = j + 1
                while k < pat.len and pat[k:k + 1] != "}":
                    k = k + 1
                if k < pat.len:
                    inside = pat[j + 1:k]
                    parts = inside.split(",")
                    if parts.len == 1:
                        m = _parse_int(parts[0].strip())
                        if m >= 0:
                            node = ["rep", node, m, m]
                            j = k + 1
                    elif parts.len == 2:
                        m = _parse_int(parts[0].strip())
                        hi_s = parts[1].strip()
                        mut mx = -1
                        if hi_s != "":
                            mx = _parse_int(hi_s)
                        if m >= 0 and (hi_s == "" or mx >= m):
                            node = ["rep", node, m, mx]
                            j = k + 1
        cur.append(node)
    if branches.len == 0:
        return [["seq", cur], j]
    branches.append(["seq", cur])
    return [["alt", branches], j]

fn compile(pat):
    gid_box = [0]
    r = _parse_seq(pat, 0, "", gid_box)
    if r[1] != pat.len:
        raise "re: unexpected )"
    return {"ast": r[0], "ngroups": gid_box[0], "pat": pat}

fn _match_esc(esc, c):
    if esc == "d":
        return _is_digit(c)
    if esc == "D":
        return not _is_digit(c)
    if esc == "w":
        return _is_word(c)
    if esc == "W":
        return not _is_word(c)
    if esc == "s":
        return _is_space(c)
    if esc == "S":
        return not _is_space(c)
    return false

fn _match_class(neg, items, c):
    o = ord(c)
    mut hit = false
    for it in items:
        if it[0] == "ch" and it[1] == c:
            hit = true
            break
        elif it[0] == "rng" and o >= it[1] and o <= it[2]:
            hit = true
            break
        elif it[0] == "esc" and _match_esc(it[1], c):
            hit = true
            break
    return (not hit) if neg else hit

fn _is_boundary(text, pos):
    prev_w = _is_word(text[pos - 1:pos]) if pos > 0 else false
    next_w = _is_word(text[pos:pos + 1]) if pos < text.len else false
    return prev_w != next_w

fn _copy_grps(grps):
    mut out = []
    for g in grps:
        out.append(g)
    return out

fn _match_nodes(nodes, idx, text, pos, grps):
    if idx >= nodes.len:
        return [pos, grps]
    nd = nodes[idx]
    k = nd[0]
    if k == "bol":
        if pos == 0:
            return _match_nodes(nodes, idx + 1, text, pos, grps)
        return nil
    if k == "eol":
        if pos == text.len:
            return _match_nodes(nodes, idx + 1, text, pos, grps)
        return nil
    if k == "lit":
        if pos < text.len and text[pos:pos + 1] == nd[1]:
            return _match_nodes(nodes, idx + 1, text, pos + 1, grps)
        return nil
    if k == "dot":
        if pos < text.len and text[pos:pos + 1] != "\n":
            return _match_nodes(nodes, idx + 1, text, pos + 1, grps)
        return nil
    if k == "esc":
        if nd[1] == "b":
            if _is_boundary(text, pos):
                return _match_nodes(nodes, idx + 1, text, pos, grps)
            return nil
        if pos < text.len and _match_esc(nd[1], text[pos:pos + 1]):
            return _match_nodes(nodes, idx + 1, text, pos + 1, grps)
        return nil
    if k == "class":
        if pos < text.len and _match_class(nd[1], nd[2], text[pos:pos + 1]):
            return _match_nodes(nodes, idx + 1, text, pos + 1, grps)
        return nil
    if k == "seq":
        mut combined = []
        for item in nd[1]:
            combined.append(item)
        for j in range(idx + 1, nodes.len):
            combined.append(nodes[j])
        return _match_nodes(combined, 0, text, pos, grps)
    if k == "alt":
        rest = nodes[idx + 1:]
        for br in nd[1]:
            mut combined = [br]
            for r in rest:
                combined.append(r)
            res = _match_nodes(combined, 0, text, pos, _copy_grps(grps))
            if res != nil:
                return res
        return nil
    if k == "grp":
        gid = nd[1]
        inner = nd[2]
        rest = nodes[idx + 1:]
        # Khớp inner trước, rồi ghi nhận chuỗi nhóm gid và khớp tiếp rest
        return _match_group(inner, rest, gid, text, pos, grps)
    if k == "rep":
        child = nd[1]
        mn = nd[2]
        mx = nd[3]
        rest = nodes[idx + 1:]
        return _match_rep(child, mn, mx, 0, rest, text, pos, grps)
    return nil

fn _match_group(inner, rest, gid, text, start_pos, grps):
    # Tìm các vị trí kết thúc khả dĩ của inner (ưu tiên tham lam)
    if inner[0] == "alt":
        for br in inner[1]:
            res = _match_group(br, rest, gid, text, start_pos, _copy_grps(grps))
            if res != nil:
                return res
        return nil
    # inner là ["seq", items]
    res1 = _match_nodes(inner[1], 0, text, start_pos, _copy_grps(grps))
    if res1 == nil:
        return nil
    end_pos = res1[0]
    mut g2 = res1[1]
    g2[gid - 1] = text[start_pos:end_pos]
    return _match_nodes(rest, 0, text, end_pos, g2)

fn _match_rep(child, mn, mx, cnt, rest, text, pos, grps):
    if mx >= 0 and cnt > mx:
        return nil
    # Tham lam (greedy): thử lặp thêm 1 lần trước nếu chưa chạm mx
    if mx < 0 or cnt < mx:
        step = _match_nodes([child], 0, text, pos, _copy_grps(grps))
        if step != nil and step[0] > pos:
            deeper = _match_rep(child, mn, mx, cnt + 1, rest, text, step[0], step[1])
            if deeper != nil:
                return deeper
    if cnt >= mn:
        return _match_nodes(rest, 0, text, pos, grps)
    return nil

fn _ensure_rx(pat):
    if type(pat) == "dict" and "ast" in pat:
        return pat
    return compile(pat)

fn _exec_at(rx, text, start):
    mut grps = []
    for i in range(rx["ngroups"]):
        grps.append(nil)
    r = _match_nodes([rx["ast"]], 0, text, start, grps)
    if r == nil:
        return nil
    ep = r[0]
    return {
        "match": text[start:ep],
        "start": start,
        "end": ep,
        "groups": r[1]
    }

fn match(pat, text):
    rx = _ensure_rx(pat)
    return _exec_at(rx, text, 0)

fn fullmatch(pat, text):
    rx = _ensure_rx(pat)
    m = _exec_at(rx, text, 0)
    return m != nil and m["end"] == text.len

fn search(pat, text):
    rx = _ensure_rx(pat)
    for i in range(text.len + 1):
        m = _exec_at(rx, text, i)
        if m != nil:
            return m
    return nil

fn test(pat, text):
    return search(pat, text) != nil

fn findall(pat, text):
    rx = _ensure_rx(pat)
    mut out = []
    mut i = 0
    while i <= text.len:
        m = _exec_at(rx, text, i)
        if m != nil:
            out.append(m["match"])
            if m["end"] > i:
                i = m["end"]
            else:
                i = i + 1
        else:
            i = i + 1
    return out

fn sub(pat, repl, text, count = 0):
    rx = _ensure_rx(pat)
    mut out = ""
    mut i = 0
    mut done = 0
    while i <= text.len:
        if count > 0 and done >= count:
            out = out + text[i:]
            break
        m = _exec_at(rx, text, i)
        if m != nil:
            out = out + repl
            done = done + 1
            if m["end"] > i:
                i = m["end"]
            else:
                if i < text.len:
                    out = out + text[i:i + 1]
                i = i + 1
        else:
            if i < text.len:
                out = out + text[i:i + 1]
            i = i + 1
    return out

fn split(pat, text):
    rx = _ensure_rx(pat)
    mut out = []
    mut last = 0
    mut i = 0
    while i <= text.len:
        m = _exec_at(rx, text, i)
        if m != nil and m["end"] > i:
            out.append(text[last:i])
            last = m["end"]
            i = m["end"]
        else:
            i = i + 1
    out.append(text[last:])
    return out
