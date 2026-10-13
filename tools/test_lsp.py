"""test_lsp.py — điều khiển tools/tlsp.py qua JSON-RPC, assert đầy đủ.
Chạy: python tools/test_lsp.py   (exit 0 = xanh)
"""
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class Conn:
    def __init__(self, proc):
        self.proc = proc
        self.buf = b""
        self.next_id = 1
        self.saved = []

    def send(self, obj):
        data = json.dumps(obj).encode("utf-8")
        self.proc.stdin.write(f"Content-Length: {len(data)}\r\n\r\n".encode("latin-1"))
        self.proc.stdin.write(data)
        self.proc.stdin.flush()

    def read_msg(self):
        out = self.proc.stdout
        headers = {}
        while True:
            line = out.readline()
            if not line:
                raise EOFError
            line = line.strip()
            if not line:
                break
            if b":" in line:
                k, v = line.split(b":", 1)
                headers[k.strip().lower()] = v.strip()
        n = int(headers.get(b"content-length", b"0"))
        return json.loads(out.read(n).decode("utf-8"))

    def req(self, method, params):
        mid = self.next_id
        self.next_id += 1
        self.send({"jsonrpc": "2.0", "id": mid, "method": method, "params": params})
        while True:
            m = self.read_msg()
            if m.get("id") == mid:
                return m
            # bỏ notification xen giữa (diagnostics) vào kho
            self.saved.append(m)

def main():
    proc = subprocess.Popen(
        [sys.executable, os.path.join(ROOT, "tools", "tlsp.py")],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
        cwd=ROOT)
    c = Conn(proc)
    c.saved = []
    fails = []

    def check(name, cond, extra=""):
        print(("PASS " if cond else "FAIL ") + name + (f" [{extra}]" if extra and not cond else ""))
        if not cond:
            fails.append(name)

    # 1. initialize
    r = c.req("initialize", {"processId": 1, "rootUri": None, "capabilities": {}})
    check("initialize", r["result"]["serverInfo"]["name"] == "tlsp")
    check("caps", r["result"]["capabilities"]["hoverProvider"] is True)
    c.send({"jsonrpc": "2.0", "method": "initialized", "params": {}})

    # 2. didOpen file đúng -> diagnostics rỗng
    good = 'fn add(a, b):\n    return a + b\n\nlet x = add(1, 2)\nprint(x)\n'
    uri = "file:///demo.t"
    c.send({"jsonrpc": "2.0", "method": "textDocument/didOpen",
            "params": {"textDocument": {"uri": uri, "text": good}}})
    m = c.read_msg()
    check("open-notify", m["method"] == "textDocument/publishDiagnostics", str(m)[:80])
    check("open-diag-empty", m["params"]["diagnostics"] == [], str(m)[:160])

    # 3. didOpen file sai -> có diagnostics dòng đúng
    bad = 'fn add(a, b)\n    return a + b\n'
    uri2 = "file:///bad.t"
    c.send({"jsonrpc": "2.0", "method": "textDocument/didOpen",
            "params": {"textDocument": {"uri": uri2, "text": bad}}})
    m = c.read_msg()
    ds = m["params"]["diagnostics"]
    check("bad-diag-one", len(ds) == 1, str(ds)[:160])
    # lỗi thiếu ':' báo ở IND dòng 2 (kẹp trong file) — trỏ cuối hàm là hợp lý
    check("bad-diag-line", ds and ds[0]["range"]["start"]["line"] == 1, str(ds)[:160])

    # 4. didChange sửa hết lỗi -> diagnostics rỗng
    c.send({"jsonrpc": "2.0", "method": "textDocument/didChange",
            "params": {"textDocument": {"uri": uri2},
                       "contentChanges": [{"text": "print(1)\n"}]}})
    m = c.read_msg()
    check("change-fixed", m["params"]["diagnostics"] == [], str(m)[:160])

    # 5. hover keyword + builtin + user fn
    r = c.req("textDocument/hover",
              {"textDocument": {"uri": uri},
               "position": {"line": 1, "character": 5}})
    check("hover-return", r["result"] is not None and "return" in r["result"]["contents"]["value"],
          str(r)[:160])
    r = c.req("textDocument/hover",
              {"textDocument": {"uri": uri},
               "position": {"line": 4, "character": 1}})
    check("hover-print", r["result"] is not None and "print" in r["result"]["contents"]["value"],
          str(r)[:160])
    r = c.req("textDocument/hover",
              {"textDocument": {"uri": uri},
               "position": {"line": 3, "character": 9}})
    check("hover-userfn", r["result"] is not None and "add" in r["result"]["contents"]["value"],
          str(r)[:160])
    r = c.req("textDocument/hover",
              {"textDocument": {"uri": uri},
               "position": {"line": 0, "character": 0}})
    check("hover-fn-kw", r["result"] is not None and "fn" in r["result"]["contents"]["value"],
          str(r)[:160])

    # 6. definition: dùng ở dòng 3 -> định nghĩa dòng 0
    r = c.req("textDocument/definition",
              {"textDocument": {"uri": uri},
               "position": {"line": 3, "character": 9}})
    check("def-add", r["result"] is not None and r["result"]["range"]["start"]["line"] == 0,
          str(r)[:160])
    # dùng x ở dòng 4 (col 6 = chữ x) -> định nghĩa dòng 3
    r = c.req("textDocument/definition",
              {"textDocument": {"uri": uri},
               "position": {"line": 4, "character": 6}})
    check("def-x", r["result"] is not None and r["result"]["range"]["start"]["line"] == 3,
          str(r)[:160])

    # 7. completion có keyword + tên file
    r = c.req("textDocument/completion",
              {"textDocument": {"uri": uri}, "position": {"line": 4, "character": 0}})
    labels = [i["label"] for i in r["result"]["items"]]
    check("complete-kw", "while" in labels and "tensor" in labels, str(labels[:8]))
    check("complete-user", "add" in labels and "x" in labels, str(labels[:40]))

    # 8. documentSymbol liệt kê fn + var
    r = c.req("textDocument/documentSymbol",
              {"textDocument": {"uri": uri}})
    names = [s["name"] for s in r["result"]]
    check("symbols", "add" in names and "x" in names, str(names))

    # 9. method lạ -> lỗi -32601 chứ không chết
    r = c.req("workspace/xyz", {})
    check("unknown-err", r.get("error", {}).get("code") == -32601, str(r)[:120])

    # 10. references: add xuất hiện 2 lần (def dòng 0, dùng dòng 3)
    r = c.req("textDocument/references",
              {"textDocument": {"uri": uri},
               "position": {"line": 3, "character": 9},
               "context": {"includeDeclaration": True}})
    locs = r["result"] or []
    check("refs-add", len(locs) == 2, str(locs)[:200])
    check("refs-lines", sorted(l["range"]["start"]["line"] for l in locs) == [0, 3],
          str(locs)[:200])

    # 11. rename add -> plus (2 edits, đúng vị trí)
    r = c.req("textDocument/rename",
              {"textDocument": {"uri": uri},
               "position": {"line": 0, "character": 4},
               "newName": "plus"})
    edits = (r["result"] or {}).get("documentChanges", [{}])[0].get("edits", [])
    check("rename-n", len(edits) == 2, str(edits)[:200])
    check("rename-text", all(e["newText"] == "plus" for e in edits), str(edits)[:120])
    # rename tên bậy -> null
    r = c.req("textDocument/rename",
              {"textDocument": {"uri": uri},
               "position": {"line": 0, "character": 4},
               "newName": "9x"})
    check("rename-bad", r["result"] is None, str(r)[:120])

    # 12. workspace/symbol qua 2 file
    uri3 = "file:///lib.t"
    c.send({"jsonrpc": "2.0", "method": "textDocument/didOpen",
            "params": {"textDocument": {"uri": uri3,
                                        "text": "fn helper():\n    return 1\n"}}})
    m = c.read_msg()
    check("open3-diag", m["params"]["diagnostics"] == [], str(m)[:120])
    r = c.req("workspace/symbol", {"query": "hel"})
    names = [s["name"] for s in r["result"]]
    check("wsym", "helper" in names, str(names))
    r = c.req("workspace/symbol", {"query": ""})
    names = [s["name"] for s in r["result"]]
    check("wsym-all", "add" in names and "helper" in names, str(names))

    # 13. shutdown/exit sạch
    r = c.req("shutdown", None)
    check("shutdown", r["result"] is None, str(r)[:80])
    c.send({"jsonrpc": "2.0", "method": "exit", "params": {}})
    proc.wait(timeout=10)
    check("exit-code", proc.returncode == 0, str(proc.returncode))

    print("LSP-ALL-GREEN" if not fails else f"LSP-FAIL: {fails}")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
