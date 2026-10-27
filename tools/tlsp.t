# tlsp.t — Language Server Protocol (JSON-RPC 2.0 qua stdio) cho Ngôn ngữ T (v3.0, thuần T)
import os
import json
import host

BUILTIN_DOCS = {
    "print": ["print(x, ...)", "In các giá trị ra stdout, cách nhau bởi dấu cách."],
    "len": ["len(x) -> int", "Độ dài của chuỗi, danh sách, dict hoặc tensor."],
    "range": ["range(n) | range(a, b) -> list", "Tạo dãy số nguyên."],
    "str": ["str(x) -> str", "Chuyển giá trị bất kỳ sang chuỗi."],
    "int": ["int(x) -> int", "Ép kiểu sang số nguyên 64-bit."],
    "float": ["float(x) -> float", "Ép kiểu sang số thực 64-bit."],
    "type": ["type(x) -> str", "Trả về tên kiểu của giá trị."],
    "open": ["open(path[, mode]) -> file", "Mở file để đọc/ghi."],
    "read": ["read(f) -> str", "Đọc toàn bộ nội dung file."],
    "write": ["write(f, s) -> int", "Ghi chuỗi s vào file f."],
    "close": ["close(f)", "Đóng file f."],
    "argv": ["argv() -> list[str]", "Danh sách tham số dòng lệnh."],
    "load": ["load(path) -> record", "Nạp module .t/.tb lúc chạy, trả về bản ghi export."],
    "sqrt": ["sqrt(x) -> float|Tensor", "Căn bậc hai."],
    "sin": ["sin(x) -> float|Tensor", "Sin (radian)."],
    "cos": ["cos(x) -> float|Tensor", "Cos (radian)."],
    "tanh": ["tanh(x) -> float|Tensor", "Tang hiperbolic."],
    "exp": ["exp(x) -> float|Tensor", "e mũ x."],
    "log": ["log(x) -> float|Tensor", "Logarit tự nhiên ln(x)."],
    "abs": ["abs(x) -> num|Tensor", "Trị tuyệt đối."],
    "round": ["round(x) -> int", "Làm tròn tới số nguyên gần nhất."],
    "floor": ["floor(x) -> int|Tensor", "Làm tròn xuống."],
    "ceil": ["ceil(x) -> int|Tensor", "Làm tròn lên."],
    "ord": ["ord(ch) -> int", "Mã byte của ký tự đầu."],
    "chr": ["chr(n) -> str", "Ký tự 1-byte từ mã 0..255."],
    "tensor": ["tensor(list) -> Tensor", "Tạo tensor n-chiều từ danh sách lồng nhau."],
    "zeros": ["zeros(shape) -> Tensor", "Tạo tensor toàn 0 với shape cho trước."],
    "ones": ["ones(shape) -> Tensor", "Tạo tensor toàn 1 với shape cho trước."],
    "full": ["full(shape, val) -> Tensor", "Tạo tensor điền giá trị val."],
    "arange": ["arange(n) -> Tensor", "Tensor 1D [0, 1, ..., n-1]."],
    "eye": ["eye(n) -> Tensor", "Ma trận đơn vị n x n."],
    "rand": ["rand(shape) -> Tensor", "Tensor ngẫu nhiên phân phối đều [0, 1)."],
    "randn": ["randn(shape) -> Tensor", "Tensor ngẫu nhiên chuẩn N(0, 1)."],
    "sum": ["sum(xs[, init])", "Tổng phần tử."],
    "min": ["min(xs) | min(a, b)", "Giá trị nhỏ nhất."],
    "max": ["max(xs) | max(a, b)", "Giá trị lớn nhất."]
}

KEYWORDS = [
    "fn", "let", "mut", "return", "if", "elif", "else", "while", "for", "in",
    "break", "continue", "switch", "case", "default", "struct", "import", "from", "as",
    "try", "except", "raise", "assert", "and", "or", "not", "true", "false", "nil"
]

fn is_id_ch(c):
    o = ord(c)
    return (o >= 65 and o <= 90) or (o >= 97 and o <= 122) or (o >= 48 and o <= 57) or c == "_" or o >= 128

fn is_id_start(c):
    o = ord(c)
    return (o >= 65 and o <= 90) or (o >= 97 and o <= 122) or c == "_" or o >= 128

fn parse_line_num(msg):
    p = msg.find("line ")
    if p >= 0:
        sub = msg[p + 5:msg.len].strip()
        mut num_s = ""
        mut k = 0
        while k < sub.len and ord(sub[k]) >= 48 and ord(sub[k]) <= 57:
            num_s = num_s + sub[k]
            k = k + 1
        if num_s != "":
            return int(num_s)
    return 1

fn analyze(text):
    lines = text.split("\n")
    mut syms = {}
    mut diags = []
    try:
        toks = host.scan(text)
        st = [toks, 0]
        host.parse_prog(st)
    except:
        msg = str(err)
        ln = parse_line_num(msg)
        l0 = (ln - 1) if ln > 0 else 0
        end_c = lines[l0].len if l0 < lines.len else 1
        diags.append({
            "range": {"start": {"line": l0, "character": 0}, "end": {"line": l0, "character": end_c}},
            "severity": 1,
            "source": "tlsp",
            "message": msg
        })

    mut li = 0
    while li < lines.len:
        raw = lines[li]
        s = raw.split("#")[0].strip()
        if s.startswith("fn "):
            rest = s[3:s.len].strip()
            lp = rest.find("(")
            rp = rest.find(")")
            if lp > 0 and rp >= lp:
                name = rest[0:lp].strip()
                params = rest[lp + 1:rp].strip()
                col = raw.find(name)
                syms[name] = {
                    "name": name,
                    "kind": 12,
                    "line": li,
                    "col": col if col >= 0 else 0,
                    "end_col": (col if col >= 0 else 0) + name.len,
                    "detail": "fn " + name + "(" + params + ")",
                    "doc": "Hàm `" + name + "` khai báo tại dòng " + str(li + 1) + "."
                }
        elif s.startswith("struct "):
            rest = s[7:s.len].strip()
            cp = rest.find(":")
            name = (rest[0:cp] if cp >= 0 else rest).strip()
            if name != "":
                col = raw.find(name)
                syms[name] = {
                    "name": name,
                    "kind": 23,
                    "line": li,
                    "col": col if col >= 0 else 0,
                    "end_col": (col if col >= 0 else 0) + name.len,
                    "detail": "struct " + name,
                    "doc": "Kiểu bản ghi `" + name + "` tại dòng " + str(li + 1) + "."
                }
        elif s.startswith("let ") or s.startswith("mut "):
            mode = s[0:3]
            rest = s[4:s.len].strip()
            eq = rest.find("=")
            if eq > 0:
                name = rest[0:eq].strip()
                col = raw.find(name)
                syms[name] = {
                    "name": name,
                    "kind": 13,
                    "line": li,
                    "col": col if col >= 0 else 0,
                    "end_col": (col if col >= 0 else 0) + name.len,
                    "detail": mode + " " + name,
                    "doc": "Biến (`" + mode + "`) tại dòng " + str(li + 1) + "."
                }
        li = li + 1
    return [syms, diags, lines]

fn word_at(lines, line, ch):
    if line < 0 or line >= lines.len:
        return ""
    s = lines[line]
    if ch < 0 or ch > s.len:
        return ""
    mut i = ch
    while i > 0 and is_id_ch(s[i - 1]):
        i = i - 1
    mut j = ch
    while j < s.len and is_id_ch(s[j]):
        j = j + 1
    return s[i:j]

fn find_word_occurrences(text, target):
    mut out = []
    if target == "":
        return out
    lines = text.split("\n")
    mut li = 0
    while li < lines.len:
        raw = lines[li]
        code = raw.split("#")[0]
        mut i = 0
        n = code.len
        while i < n:
            if code[i] == "\"":
                i = i + 1
                while i < n and code[i] != "\"":
                    if code[i] == "\\":
                        i = i + 2
                    else:
                        i = i + 1
                i = i + 1
                continue
            if is_id_start(code[i]):
                mut j = i + 1
                while j < n and is_id_ch(code[j]):
                    j = j + 1
                if code[i:j] == target:
                    out.append({
                        "start": {"line": li, "character": i},
                        "end": {"line": li, "character": j}
                    })
                i = j
            else:
                i = i + 1
        li = li + 1
    return out

fn send_msg(payload):
    body = json.dumps(payload)
    hdr = "Content-Length: " + str(body.len) + "\r\n\r\n"
    os.stdout_write(hdr + body)

fn read_msg():
    mut content_len = -1
    while true:
        line = os.stdin_line()
        if line == nil:
            return nil
        s = line.strip()
        if s == "":
            break
        if s.lower().startswith("content-length:"):
            content_len = int(s[15:s.len].strip())
    if content_len < 0:
        return nil
    body = os.stdin_read(content_len)
    if body == nil or body == "":
        return nil
    return json.loads(body)

fn main():
    mut docs = {}
    while true:
        msg = read_msg()
        if msg == nil:
            break
        method = msg["method"] if msg.has("method") else ""
        req_id = msg["id"] if msg.has("id") else nil
        params = msg["params"] if msg.has("params") else {}

        if method == "initialize":
            send_msg({
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "capabilities": {
                        "textDocumentSync": 1,
                        "hoverProvider": true,
                        "definitionProvider": true,
                        "referencesProvider": true,
                        "renameProvider": true,
                        "documentSymbolProvider": true,
                        "workspaceSymbolProvider": true,
                        "completionProvider": {"triggerCharacters": ["."]}
                    },
                    "serverInfo": {"name": "tlsp", "version": "3.0.0"}
                }
            })
        elif method == "initialized":
            continue
        elif method == "shutdown":
            send_msg({"jsonrpc": "2.0", "id": req_id, "result": nil})
        elif method == "exit":
            break
        elif method == "textDocument/didOpen":
            td = params["textDocument"]
            uri = td["uri"]
            text = td["text"]
            res = analyze(text)
            docs[uri] = {"text": text, "syms": res[0], "lines": res[2]}
            send_msg({
                "jsonrpc": "2.0",
                "method": "textDocument/publishDiagnostics",
                "params": {"uri": uri, "diagnostics": res[1]}
            })
        elif method == "textDocument/didChange":
            uri = params["textDocument"]["uri"]
            changes = params["contentChanges"]
            if changes.len > 0:
                text = changes[changes.len - 1]["text"]
                res = analyze(text)
                docs[uri] = {"text": text, "syms": res[0], "lines": res[2]}
                send_msg({
                    "jsonrpc": "2.0",
                    "method": "textDocument/publishDiagnostics",
                    "params": {"uri": uri, "diagnostics": res[1]}
                })
        elif method == "textDocument/hover":
            uri = params["textDocument"]["uri"]
            pos = params["position"]
            d = docs[uri] if docs.has(uri) else {"syms": {}, "lines": []}
            w = word_at(d["lines"], pos["line"], pos["character"])
            mut res = nil
            if d["syms"].has(w):
                sym = d["syms"][w]
                res = {"contents": {"kind": "markdown", "value": "```t\n" + sym["detail"] + "\n```\n" + sym["doc"]}}
            elif BUILTIN_DOCS.has(w):
                b = BUILTIN_DOCS[w]
                res = {"contents": {"kind": "markdown", "value": "```t\n" + b[0] + "\n```\n" + b[1]}}
            send_msg({"jsonrpc": "2.0", "id": req_id, "result": res})
        elif method == "textDocument/definition":
            uri = params["textDocument"]["uri"]
            pos = params["position"]
            d = docs[uri] if docs.has(uri) else {"syms": {}, "lines": []}
            w = word_at(d["lines"], pos["line"], pos["character"])
            mut res = nil
            if d["syms"].has(w):
                sym = d["syms"][w]
                res = {
                    "uri": uri,
                    "range": {
                        "start": {"line": sym["line"], "character": sym["col"]},
                        "end": {"line": sym["line"], "character": sym["end_col"]}
                    }
                }
            send_msg({"jsonrpc": "2.0", "id": req_id, "result": res})
        elif method == "textDocument/completion":
            uri = params["textDocument"]["uri"]
            d = docs[uri] if docs.has(uri) else {"syms": {}}
            mut items = []
            for k in d["syms"].keys():
                s = d["syms"][k]
                items.append({"label": s["name"], "kind": 3 if s["kind"] == 12 else 6, "detail": s["detail"]})
            for k in BUILTIN_DOCS.keys():
                b = BUILTIN_DOCS[k]
                items.append({"label": k, "kind": 3, "detail": b[0], "documentation": b[1]})
            for kw in KEYWORDS:
                items.append({"label": kw, "kind": 14, "detail": "keyword"})
            send_msg({"jsonrpc": "2.0", "id": req_id, "result": items})
        elif method == "textDocument/documentSymbol":
            uri = params["textDocument"]["uri"]
            d = docs[uri] if docs.has(uri) else {"syms": {}}
            mut out = []
            for k in d["syms"].keys():
                s = d["syms"][k]
                r = {
                    "start": {"line": s["line"], "character": s["col"]},
                    "end": {"line": s["line"], "character": s["end_col"]}
                }
                out.append({
                    "name": s["name"],
                    "detail": s["detail"],
                    "kind": s["kind"],
                    "range": r,
                    "selectionRange": r
                })
            send_msg({"jsonrpc": "2.0", "id": req_id, "result": out})
        elif method == "textDocument/references":
            uri = params["textDocument"]["uri"]
            pos = params["position"]
            d = docs[uri] if docs.has(uri) else {"text": "", "syms": {}, "lines": []}
            w = word_at(d["lines"], pos["line"], pos["character"])
            mut locs = []
            if w != "":
                for u in docs.keys():
                    for r in find_word_occurrences(docs[u]["text"], w):
                        locs.append({"uri": u, "range": r})
            send_msg({"jsonrpc": "2.0", "id": req_id, "result": locs})
        elif method == "textDocument/rename":
            uri = params["textDocument"]["uri"]
            pos = params["position"]
            new_name = params["newName"] if params.has("newName") else ""
            d = docs[uri] if docs.has(uri) else {"text": "", "syms": {}, "lines": []}
            w = word_at(d["lines"], pos["line"], pos["character"])
            if w == "" or new_name == "":
                send_msg({"jsonrpc": "2.0", "id": req_id, "result": nil})
            else:
                mut changes = {}
                for u in docs.keys():
                    mut edits = []
                    for r in find_word_occurrences(docs[u]["text"], w):
                        edits.append({"range": r, "newText": new_name})
                    if edits.len > 0:
                        changes[u] = edits
                send_msg({"jsonrpc": "2.0", "id": req_id, "result": {"changes": changes}})
        elif method == "workspace/symbol":
            q = params["query"].lower() if params.has("query") else ""
            mut out = []
            for u in docs.keys():
                syms = docs[u]["syms"]
                for k in syms.keys():
                    s = syms[k]
                    if q == "" or (q in s["name"].lower()):
                        out.append({
                            "name": s["name"],
                            "kind": s["kind"],
                            "location": {
                                "uri": u,
                                "range": {
                                    "start": {"line": s["line"], "character": s["col"]},
                                    "end": {"line": s["line"], "character": s["end_col"]}
                                }
                            }
                        })
            send_msg({"jsonrpc": "2.0", "id": req_id, "result": out})
        elif req_id != nil:
            send_msg({"jsonrpc": "2.0", "id": req_id, "result": nil})

main()
