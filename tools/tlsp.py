#!/usr/bin/env python3
"""tlsp.py — Language Server tối thiểu cho ngôn ngữ T (v1.3, JSON-RPC/stdio).
Chạy:  python tools/tlsp.py            (nói LSP với editor qua stdin/stdout)
Test:   python tools/test_lsp.py        (phiên giả lập, assert đầy đủ)
Hỗ trợ: initialize, textDocument/didOpen, didChange, hover, definition,
completion, documentSymbol, shutdown. Chẩn đoán lỗi parse trực tiếp.
Định vị symbol bằng phân tích dòng (fn/let/mut/for/param), không cần AST line.
"""
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from t import P, lex2  # noqa: E402

KEYWORD_DOC = {
    "let": "khai báo bất biến: `let x = 1`",
    "mut": "khai báo khả biến: `mut x = 1`",
    "fn": "định nghĩa hàm: `fn add(a, b):` + `return`",
    "return": "trả giá trị khỏi hàm",
    "if": "rẽ nhánh (kèm `elif`/`else`)",
    "elif": "rẽ nhánh tiếp theo",
    "else": "nhánh cuối",
    "for": "lặp: `for i in range(n):`",
    "in": "duyệt trong `for`; toán tử thành viên `x in xs` / `x not in xs`",
    "while": "lặp khi điều kiện đúng",
    "break": "thoát vòng lặp",
    "continue": "sang vòng lặp tiếp",
    "and": "và (ngắn mạch, trả toán hạng)",
    "or": "hoặc (ngắn mạch, trả toán hạng)",
    "not": "phủ định",
    "true": "đúng", "false": "sai", "nil": "rỗng",
    "print": "in các đối số, cách nhau bởi dấu cách",
    "tensor": "tensor n-chiều float64: `tensor([[1, 2], [3, 4]])`; `+ - * / **` từng phần tử"
              " (broadcast), `@` nhân ma trận; `.shape .T .reshape() .sum(axis) .softmax()`...",
    "range": "`range(n)` 0..n-1, `range(a, b)` a..b-1",
    "len": "độ dài str/list/tensor/dict",
    "str": "đổi sang chuỗi", "int": "đổi sang số nguyên",
    "float": "đổi sang số thực", "type": "tên kiểu giá trị",
    "import": "`import ai`, `import ai as x`",
    "from": "`from ai import relu, mse`",
    "as": "bí danh import",
    "try": "bắt lỗi (kèm `except [tên]:`, biến `err`)",
    "except": "nhánh bắt lỗi theo tên (match chuỗi)",
    "raise": "ném lỗi chuỗi: `raise \"boom\"`",
    "assert": "`assert cond, msg` — sai thì TError",
    "open": "`open(path[, mode])` -> file",
    "read": "`read(f)` toàn bộ nội dung",
    "write": "`write(f, s)` -> số byte",
    "close": "`close(f)`",
    "ord": "`ord(\"a\")` -> 97", "chr": "`chr(97)` -> \"a\"",
    "argv": "`argv()` tham số dòng lệnh",
    "dict": "từ điển: `{\"a\": 1}`, `d[k]`, `.keys()/.values()/.get()`",
    "keys": "xem `dict`",
    "switch": "rẽ nhiều nhánh: `switch x:` + `case 1, 2:` + `default:`",
    "case": "nhánh của `switch` (nhiều giá trị cách bởi dấu phẩy)",
    "default": "nhánh mặc định của `switch`",
    "struct": "kiểu bản ghi: `struct P:` + dòng field `x, y` + method `fn m(self):`;"
              " tạo `P(1, 2)`, đọc/ghi `p.x`",
    "set": "tập hợp: `set([1, 2])`, `.add/.has/.remove/.union/.inter/.diff/.list`",
    "input": "`input([prompt])` đọc 1 dòng stdin (nil khi EOF)",
    "abs": "trị tuyệt đối", "round": "làm tròn nửa xa 0",
    "min": "`min(xs)` / `min(a, b)`", "max": "`max(xs)` / `max(a, b)`",
    "sum": "`sum(xs[, init])`", "any": "có phần tử truthy", "all": "mọi phần tử truthy",
    "sort": "`sort(xs[, key])` sắp tại chỗ", "reversed": "đảo list/str",
    "enumerate": "`enumerate(xs[, start])` -> [[i, x], ...]",
    "zip": "`zip(a, b, ...)`", "map": "`map(f, xs)`", "filter": "`filter(f, xs)`",
    "zeros": "`zeros(shape)` tensor 0", "ones": "`ones(shape)` tensor 1",
    "full": "`full(shape, v)`", "arange": "`arange(n)` / `arange(a, b[, step])`",
    "eye": "`eye(n)` ma trận đơn vị", "rand": "`rand(shape)` đều [0, 1)",
    "randn": "`randn(shape)` chuẩn N(0, 1)", "seed": "`seed(n)` đặt hạt giống ngẫu nhiên",
    "sqrt": "căn (số hoặc tensor)", "exp": "e mũ x", "log": "ln x",
    "sin": "sin", "cos": "cos", "tanh": "tanh", "floor": "làm tròn xuống",
    "ceil": "làm tròn lên", "pi": "3.141592653589793",
}
BUILTIN_FN = {"range", "len", "str", "int", "float", "type", "tensor", "open",
              "read", "write", "close", "ord", "chr", "argv", "print",
              "abs", "round", "min", "max", "sum", "any", "all", "sort",
              "reversed", "enumerate", "zip", "map", "filter", "set", "input",
              "zeros", "ones", "full", "arange", "eye", "rand", "randn", "seed",
              "sqrt", "exp", "log", "sin", "cos", "tanh", "floor", "ceil", "pi"}

RE_FN = re.compile(r"^\s*fn\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(")
RE_FNARGS = re.compile(r"^\s*fn\s+[A-Za-z_][A-Za-z0-9_]*\s*\(([^)]*)\)")
RE_DECL = re.compile(r"^\s*(?:let|mut)\s+([A-Za-z_][A-Za-z0-9_]*)")
RE_FOR = re.compile(r"^\s*for\s+([A-Za-z_][A-Za-z0-9_]*(?:\s*,\s*[A-Za-z_][A-Za-z0-9_]*)*)\s+in\b")
RE_STRUCT = re.compile(r"^\s*struct\s+([A-Za-z_][A-Za-z0-9_]*)\s*:")
RE_WORD = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def word_at(line, col):
    for m in RE_WORD.finditer(line):
        if m.start() <= col < m.end():
            return m.group(0)
    return None


def scan_symbols(text):
    """-> (defs, refs_ignored). defs: list dict(name, kind, line, detail)."""
    defs = []
    for i, raw in enumerate(text.split("\n")):
        m = RE_FN.match(raw)
        if m:
            a = RE_FNARGS.match(raw)
            params = [p.strip() for p in a.group(1).split(",")] if a else []
            params = [p for p in params if p]
            defs.append({"name": m.group(1), "kind": "fn", "line": i,
                         "detail": f"fn({', '.join(params)})"})
            for p in params:
                if RE_WORD.fullmatch(p):
                    defs.append({"name": p, "kind": "param", "line": i,
                                 "detail": "tham số"})
            continue
        m = RE_DECL.match(raw)
        if m:
            defs.append({"name": m.group(1), "kind": "var", "line": i,
                         "detail": "biến"})
            continue
        m = RE_FOR.match(raw)
        if m:
            for nm in m.group(1).split(","):
                defs.append({"name": nm.strip(), "kind": "var", "line": i,
                             "detail": "biến lặp"})
            continue
        m = RE_STRUCT.match(raw)
        if m:
            defs.append({"name": m.group(1), "kind": "struct", "line": i,
                         "detail": "struct"})
    return defs


def find_refs(text, word):
    """Mọi vị trí xuất hiện word (dùng cho references/rename)."""
    out = []
    for i, raw in enumerate(text.split("\n")):
        code = raw.split("#", 1)[0]
        for m in RE_WORD.finditer(code):
            if m.group(0) == word:
                out.append({"line": i, "start": m.start(), "end": m.end()})
    return out


def diagnose(text):
    nlines = max(1, len(text.split("\n")))

    def mkdiag(line, msg):
        line = min(max(0, line), nlines - 1)
        return [{"range": {"start": {"line": line, "character": 0},
                           "end": {"line": line, "character": 1}},
                 "severity": 1, "message": msg}]

    try:
        P(lex2(text)).parse()
        return []
    except SyntaxError as ex:
        msg = str(ex)
        line = 0
        m = re.search(r"line (\d+)", msg)
        if m:
            line = max(0, int(m.group(1)) - 1)
        return mkdiag(line, "cú pháp: " + msg)
    except Exception as ex:  # noqa: BLE001
        return mkdiag(0, "lỗi: " + str(ex))


class Server:
    def __init__(self, rin, wout):
        self.rin = rin
        self.wout = wout
        self.docs = {}
        self.running = True
        self.shutdown_asked = False

    def read_msg(self):
        headers = {}
        while True:
            line = self.rin.readline()
            if not line:
                return None
            line = line.strip()
            if not line:
                break
            if ":" in line:
                k, v = line.split(":", 1)
                headers[k.strip().lower()] = v.strip()
        n = int(headers.get("content-length", "0"))
        if n <= 0:
            return None
        raw = self.rin.read(n)
        if not raw:
            return None
        return json.loads(raw)

    def send(self, obj):
        data = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.wout.write(f"Content-Length: {len(data)}\r\n\r\n".encode("latin-1"))
        self.wout.write(data)
        self.wout.flush()

    def send_resp(self, mid, result):
        self.send({"jsonrpc": "2.0", "id": mid, "result": result})

    def send_err(self, mid, code, message):
        self.send({"jsonrpc": "2.0", "id": mid,
                   "error": {"code": code, "message": message}})

    def notify(self, method, params):
        self.send({"jsonrpc": "2.0", "method": method, "params": params})

    def doc_of(self, uri):
        return self.docs.get(uri, "")

    def handle(self, msg):
        mid = msg.get("id")
        method = msg.get("method", "")
        params = msg.get("params") or {}
        if method == "initialize":
            self.send_resp(mid, {"capabilities": {
                "textDocumentSync": 1,
                "hoverProvider": True,
                "definitionProvider": True,
                "referencesProvider": True,
                "renameProvider": True,
                "completionProvider": {"triggerCharacters": ["."]},
                "documentSymbolProvider": True,
                "workspaceSymbolProvider": True,
            }, "serverInfo": {"name": "tlsp", "version": "1.3"}})
        elif method == "initialized":
            pass
        elif method == "shutdown":
            self.shutdown_asked = True
            if mid is not None:
                self.send_resp(mid, None)
        elif method == "exit":
            self.running = False
        elif method == "textDocument/didOpen":
            doc = params["textDocument"]
            self.docs[doc["uri"]] = doc["text"]
            self.notify("textDocument/publishDiagnostics",
                        {"uri": doc["uri"],
                         "diagnostics": diagnose(doc["text"])})
        elif method == "textDocument/didChange":
            uri = params["textDocument"]["uri"]
            for ch in params.get("contentChanges", []):
                self.docs[uri] = ch["text"]
            self.notify("textDocument/publishDiagnostics",
                        {"uri": uri, "diagnostics": diagnose(self.docs[uri])})
        elif method == "textDocument/hover":
            pos = params["position"]
            lines = self.doc_of(params["textDocument"]["uri"]).split("\n")
            w = ""
            if 0 <= pos["line"] < len(lines):
                w = word_at(lines[pos["line"]], pos["character"]) or ""
            if not w:
                self.send_resp(mid, None)
                return
            doc = KEYWORD_DOC.get(w)
            if doc is None:
                ds = [d for d in scan_symbols(self.doc_of(params["textDocument"]["uri"]))
                      if d["name"] == w]
                if ds:
                    d = ds[0]
                    doc = f"{d['kind']} `{w}` (dòng {d['line'] + 1}): {d['detail']}"
            if doc is None:
                self.send_resp(mid, None)
            else:
                self.send_resp(mid, {"contents": {"kind": "markdown",
                                                  "value": f"**{w}** — {doc}"}})
        elif method == "textDocument/definition":
            pos = params["position"]
            text = self.doc_of(params["textDocument"]["uri"])
            lines = text.split("\n")
            w = ""
            if 0 <= pos["line"] < len(lines):
                w = word_at(lines[pos["line"]], pos["character"]) or ""
            hit = None
            if w:
                for d in scan_symbols(text):
                    if d["name"] == w and d["line"] <= pos["line"]:
                        if hit is None or d["line"] > hit["line"]:
                            hit = d
            if hit is None:
                self.send_resp(mid, None)
            else:
                self.send_resp(mid, {"uri": params["textDocument"]["uri"],
                                     "range": {"start": {"line": hit["line"],
                                                         "character": 0},
                                               "end": {"line": hit["line"],
                                                       "character": len(w)}}})
        elif method == "textDocument/completion":
            text = self.doc_of(params["textDocument"]["uri"])
            items = [{"label": k, "kind": 14, "detail": v}
                     for k, v in sorted(KEYWORD_DOC.items())]
            seen = set()
            for d in scan_symbols(text):
                if d["name"] not in seen:
                    seen.add(d["name"])
                    items.append({"label": d["name"], "kind": 6,
                                  "detail": d["kind"]})
            self.send_resp(mid, {"isIncomplete": False, "items": items})
        elif method == "textDocument/documentSymbol":
            text = self.doc_of(params["textDocument"]["uri"])
            syms = []
            for d in scan_symbols(text):
                kind = {"fn": 12, "struct": 23}.get(d["kind"], 13)
                syms.append({"name": d["name"], "kind": kind,
                             "range": {"start": {"line": d["line"], "character": 0},
                                       "end": {"line": d["line"], "character": len(d["name"])}},
                             "selectionRange": {"start": {"line": d["line"], "character": 0},
                                                "end": {"line": d["line"], "character": len(d["name"])}},
                             "detail": d["detail"]})
            self.send_resp(mid, syms)
        elif method == "textDocument/references":
            pos = params["position"]
            uri = params["textDocument"]["uri"]
            text = self.doc_of(uri)
            lines = text.split("\n")
            w = ""
            if 0 <= pos["line"] < len(lines):
                w = word_at(lines[pos["line"]], pos["character"]) or ""
            locs = []
            if w:
                for r in find_refs(text, w):
                    locs.append({"uri": uri,
                                 "range": {"start": {"line": r["line"],
                                                     "character": r["start"]},
                                           "end": {"line": r["line"],
                                                   "character": r["end"]}}})
            self.send_resp(mid, locs)
        elif method == "textDocument/rename":
            pos = params["position"]
            uri = params["textDocument"]["uri"]
            new = params["newName"]
            text = self.doc_of(uri)
            lines = text.split("\n")
            w = ""
            if 0 <= pos["line"] < len(lines):
                w = word_at(lines[pos["line"]], pos["character"]) or ""
            if not w or not RE_WORD.fullmatch(new):
                self.send_resp(mid, None)
                return
            edits = []
            for r in find_refs(text, w):
                edits.append({"range": {"start": {"line": r["line"],
                                                  "character": r["start"]},
                                        "end": {"line": r["line"],
                                                "character": r["end"]}},
                              "newText": new})
            self.send_resp(mid, {"documentChanges": [
                {"textDocument": {"uri": uri, "version": None}, "edits": edits}]})
        elif method == "workspace/symbol":
            q = (params.get("query") or "").lower()
            syms = []
            for uri, text in self.docs.items():
                for d in scan_symbols(text):
                    if q in d["name"].lower():
                        syms.append({"name": d["name"], "kind": 12,
                                     "location": {"uri": uri,
                                                  "range": {"start": {"line": d["line"],
                                                                      "character": 0},
                                                            "end": {"line": d["line"],
                                                                    "character": len(d["name"])}}}})
            self.send_resp(mid, syms)
        else:
            if mid is not None:
                self.send_err(mid, -32601, "chưa hỗ trợ: " + method)

    def run(self):
        while self.running:
            try:
                msg = self.read_msg()
            except Exception:  # noqa: BLE001
                return
            if msg is None:
                return
            try:
                self.handle(msg)
            except Exception as ex:  # noqa: BLE001
                if msg.get("id") is not None:
                    self.send_err(msg.get("id"), -32603, str(ex))


def main():
    rin = sys.stdin
    wout = sys.stdout.buffer
    Server(rin, wout).run()


if __name__ == "__main__":
    main()
