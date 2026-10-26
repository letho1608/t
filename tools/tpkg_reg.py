#!/usr/bin/env python3
"""tpkg_reg.py — registry tối thiểu cho tpkg (v1.6, stdlib http.server).
Kho: thư mục chứa *.tpkg + index.json tự sinh (mọi phiên bản):
  {"pkgs": {"ten": {"version": "<cao nhất>", "file": "ten-1.2.0.tpkg",
                     "sha16": "...", "size": 123,
                     "versions": {"1.0.0": {"file", "sha16", "size"}, ...}}}}
Không cần server: client trỏ TPKG_REG vào thẳng thư mục kho (hoặc file://)
là chạy offline (search/install/publish).
Chạy:  python tools/tpkg_reg.py KHO_DIR [PORT=8765]
Client trỏ registry: env TPKG_REG=http://host:port  (tpm search/install tên)
"""
import hashlib
import json
import os
import sys
import zipfile
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer


def ver_key(v):
    out = []
    for p in str(v).split("."):
        d = "".join(c for c in p if c.isdigit())
        out.append(int(d) if d else 0)
    while len(out) < 3:
        out.append(0)
    return tuple(out[:3])


def build_index(root):
    pkgs = {}
    for fn in sorted(os.listdir(root)):
        if not fn.endswith(".tpkg"):
            continue
        full = os.path.join(root, fn)
        with open(full, "rb") as f:
            data = f.read()
        name, ver = split_name(fn[:-len(".tpkg")])
        try:
            with zipfile.ZipFile(os.path.join(root, fn)) as z:
                inner = [n for n in z.namelist() if n.endswith("t.json")]
        except zipfile.BadZipFile:
            inner = []
        e = pkgs.setdefault(name, {"versions": {}})
        e["versions"][ver] = {"file": fn,
                              "sha16": hashlib.sha256(data).hexdigest()[:16],
                              "size": len(data), "has_manifest": bool(inner)}
    for name, e in pkgs.items():
        best = max(e["versions"], key=ver_key)
        e.update({"version": best}, **e["versions"][best])
    tmp = os.path.join(root, "index.json.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump({"pkgs": pkgs}, f, indent=2, sort_keys=True)
    os.replace(tmp, os.path.join(root, "index.json"))
    return pkgs


def split_name(base):
    if "-" in base:
        name, ver = base.rsplit("-", 1)
        if ver and ver[0].isdigit():
            return name, ver
    return base, "0.0.0"


class Handler(SimpleHTTPRequestHandler):
    token = ""

    def __init__(self, *a, directory=None, **k):
        super().__init__(*a, directory=directory, **k)

    def log_message(self, format, *a):  # noqa: A002 - khớp base handler
        pass

    def do_PUT(self):
        if not Handler.token:
            self.send_response(403)
            self.end_headers()
            self.wfile.write(b"PUT tat: khong co TPKG_TOKEN")
            return
        if self.headers.get("X-Token") != Handler.token:
            self.send_response(403)
            self.end_headers()
            return
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0 or length > 64 * 1024 * 1024:
            self.send_response(400)
            self.end_headers()
            return
        data = self.rfile.read(length)
        name = os.path.basename((self.headers.get("X-Name") or "").strip().replace("\\", "/"))
        if not name.endswith(".tpkg") or not name or name.startswith("."):
            self.send_response(400)
            self.end_headers()
            return
        root = self.directory
        with open(os.path.join(root, name), "wb") as f:
            f.write(data)
        pkgs = build_index(root)
        body = json.dumps({"ok": True, "pkgs": len(pkgs)}).encode()
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main():
    root = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else "tpkgs-reg")
    port = int(sys.argv[2]) if len(sys.argv) > 2 else 8765
    token = os.environ.get("TPKG_TOKEN") or (sys.argv[3] if len(sys.argv) > 3 else "")
    os.makedirs(root, exist_ok=True)
    pkgs = build_index(root)
    print(f"[reg] {len(pkgs)} pkgs <- {root} (index.json)")
    if not token:
        print("[reg] CẢNH BÁO: khong co TPKG_TOKEN — PUT bi tat (chi GET duoc)")
    Handler.token = token

    def factory(*a, **k):
        return Handler(*a, directory=root, **k)

    srv = ThreadingHTTPServer(("127.0.0.1", port), factory)
    print(f"[reg] http://127.0.0.1:{port}/")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
