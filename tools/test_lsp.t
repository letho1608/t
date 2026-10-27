# test_lsp.t — Kiểm thử Language Server (tools/tlsp.t) qua JSON-RPC stdio
import os
import json

trun = os.exepath()

fn frame(msg):
    body = json.dumps(msg)
    return "Content-Length: " + str(body.len) + "\r\n\r\n" + body

fn parse_stream(raw):
    mut msgs = []
    mut i = 0
    n = raw.len
    while i < n:
        p = raw[i:n].find("Content-Length:")
        if p < 0:
            break
        start = i + p
        sub = raw[start:n]
        mut sep = sub.find("\r\n\r\n")
        mut sep_len = 4
        if sep < 0:
            sep = sub.find("\n\n")
            sep_len = 2
        assert sep >= 0, "missing header separator"
        hdr = sub[0:sep]
        line0 = hdr.split("\n")[0]
        clen = int(line0[15:line0.len].strip())
        body_start = start + sep + sep_len
        body = raw[body_start:body_start + clen]
        msgs.append(json.loads(body))
        i = body_start + clen
    return msgs

fn find_id(msgs, id):
    for m in msgs:
        if m.has("id") and m["id"] == id:
            return m
    assert false, "missing response for id=" + str(id)
    return nil

uri = "file:///demo.t"
good_code = "struct Diem:\n    x, y\n\nfn tinh_tong(a, b):\n    return a + b\n\nlet ket_qua = tinh_tong(3, 4)\nprint(sqrt(16))\n"
bad_code = "fn loi(\n    return 1\n"

reqs = [
    {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}},
    {"jsonrpc": "2.0", "method": "textDocument/didOpen", "params": {"textDocument": {"uri": uri, "languageId": "t", "version": 1, "text": good_code}}},
    {"jsonrpc": "2.0", "id": 2, "method": "textDocument/hover", "params": {"textDocument": {"uri": uri}, "position": {"line": 6, "character": 16}}},
    {"jsonrpc": "2.0", "id": 3, "method": "textDocument/hover", "params": {"textDocument": {"uri": uri}, "position": {"line": 7, "character": 7}}},
    {"jsonrpc": "2.0", "id": 4, "method": "textDocument/definition", "params": {"textDocument": {"uri": uri}, "position": {"line": 6, "character": 16}}},
    {"jsonrpc": "2.0", "id": 5, "method": "textDocument/documentSymbol", "params": {"textDocument": {"uri": uri}}},
    {"jsonrpc": "2.0", "id": 6, "method": "textDocument/completion", "params": {"textDocument": {"uri": uri}, "position": {"line": 6, "character": 0}}},
    {"jsonrpc": "2.0", "id": 7, "method": "textDocument/references", "params": {"textDocument": {"uri": uri}, "position": {"line": 6, "character": 16}, "context": {"includeDeclaration": true}}},
    {"jsonrpc": "2.0", "id": 8, "method": "textDocument/rename", "params": {"textDocument": {"uri": uri}, "position": {"line": 3, "character": 5}, "newName": "cong_hai"}},
    {"jsonrpc": "2.0", "id": 9, "method": "workspace/symbol", "params": {"query": "Diem"}},
    {"jsonrpc": "2.0", "method": "textDocument/didChange", "params": {"textDocument": {"uri": uri, "version": 2}, "contentChanges": [{"text": bad_code}]}},
    {"jsonrpc": "2.0", "id": 10, "method": "shutdown", "params": {}},
    {"jsonrpc": "2.0", "method": "exit", "params": {}}
]

mut stdin_buf = ""
for r in reqs:
    stdin_buf = stdin_buf + frame(r)

res = os.run([trun, "tools/tlsp.t"], stdin_buf)
assert res["rc"] == 0, "tlsp failed: " + res["err"]
msgs = parse_stream(res["out"])

r1 = find_id(msgs, 1)
assert r1["result"]["capabilities"]["hoverProvider"] == true
assert r1["result"]["capabilities"]["referencesProvider"] == true
assert r1["result"]["capabilities"]["renameProvider"] == true

r2 = find_id(msgs, 2)
assert "fn tinh_tong(a, b)" in r2["result"]["contents"]["value"]

r3 = find_id(msgs, 3)
assert "sqrt(x)" in r3["result"]["contents"]["value"]

r4 = find_id(msgs, 4)
assert r4["result"]["range"]["start"]["line"] == 3

r5 = find_id(msgs, 5)
mut sym_names = []
for s in r5["result"]:
    sym_names.append(s["name"])
assert "Diem" in sym_names and "tinh_tong" in sym_names and "ket_qua" in sym_names

r6 = find_id(msgs, 6)
mut labels = []
for c in r6["result"]:
    labels.append(c["label"])
assert "tinh_tong" in labels and "tensor" in labels and "switch" in labels

r7 = find_id(msgs, 7)
assert r7["result"].len == 2

r8 = find_id(msgs, 8)
edits = r8["result"]["changes"][uri]
assert edits.len == 2 and edits[0]["newText"] == "cong_hai"

r9 = find_id(msgs, 9)
assert r9["result"].len == 1 and r9["result"][0]["name"] == "Diem"

mut diag_msgs = []
for m in msgs:
    if m.has("method") and m["method"] == "textDocument/publishDiagnostics":
        diag_msgs.append(m)
assert diag_msgs.len == 2
assert diag_msgs[0]["params"]["diagnostics"].len == 0
assert diag_msgs[1]["params"]["diagnostics"].len == 1

print("LSP-ALL-GREEN")
