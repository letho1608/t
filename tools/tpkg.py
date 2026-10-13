#!/usr/bin/env python3
"""tpkg.py — package manager tối thiểu cho T (v1.5, local-first + URL).
Dùng:
  python tools/tpkg.py init myapp        # scaffold t.json + main.t + tests/
  python tools/tpkg.py install           # chép deps vào tpkgs/ + lock.json
  python tools/tpkg.py run main.t        # chạy với tpkgs/ trên đường tìm module
  python tools/tpkg.py test              # chạy tests/*.t, báo pass/fail
  python tools/tpkg.py pack              # zip project -> name-version.tpkg
  python tools/tpkg.py publish [.tpkg]   # đẩy lên registry (TPKG_REG/TPKG_TOKEN)
  python tools/tpkg.py search [chuỗi]    # tìm trong registry
  python tools/tpkg.py fetch <url> <dir> # tải .tpkg (http/https/file) -> dir
  python tools/tpkg.py install --offline # chỉ dùng lock.json + cache (kiểm sha)
  python tools/tpkg.py release patch|minor|major [ghi chú...]
                                          # tăng semver trong t.json + CHANGELOG.md
Registry (TPKG_REG hoặc "registry" trong t.json):
  http://host:port   — server tools/tpkg_reg.py
  D:/duong/kho, file:///duong/kho — thư mục kho cục bộ, chạy hoàn toàn offline
Cache archive: $TPKG_CACHE hoặc ~/.tpkg/cache (theo sha16; dùng khi mất mạng).
Quy ước dep trong t.json "deps": {"ten": "./duong/den/dep"} hoặc URL
(http://.../x.tpkg, file:///.../x.tpkg — tự bung vào tpkgs/ten/).
Sau install: import ten tìm tpkgs/ten/ten.t (hoặc main trong t.json).
"""
import datetime
import hashlib
import json
import os
import shutil
import subprocess
import sys
import urllib.parse
import urllib.request
import zipfile

TROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tpkg_reg import build_index  # noqa: E402

OFFLINE = [False]


def cache_dir():
    d = os.environ.get("TPKG_CACHE") or os.path.join(os.path.expanduser("~"), ".tpkg", "cache")
    os.makedirs(d, exist_ok=True)
    return d


def local_reg_dir(reg):
    """Registry là thư mục cục bộ? -> đường dẫn, ngược lại None."""
    if reg.startswith("file://"):
        p = urllib.parse.unquote(urllib.parse.urlparse(reg).path)
        if os.name == "nt" and len(p) > 2 and p[0] == "/" and p[2] == ":":
            p = p[1:]
        return p
    if "://" not in reg:
        return reg
    return None


def fetch_bytes(url):
    """Tải url (http/https/file/đường dẫn). Lưu cache theo url; mất mạng thì
    đọc lại cache. Trả (data, sha16)."""
    key = hashlib.sha256(url.encode("utf-8")).hexdigest()[:16]
    cpath = os.path.join(cache_dir(), "url-" + key)
    data = None
    if not OFFLINE[0]:
        try:
            if "://" not in url:
                with open(url, "rb") as f:
                    data = f.read()
            else:
                with urllib.request.urlopen(url, timeout=60) as r:
                    data = r.read()
        except Exception as ex:  # noqa: BLE001
            if not os.path.isfile(cpath):
                fail(f"tải {url}: {ex} (không có trong cache)")
            print(f"[tpm] offline: dùng cache cho {url}")
    if data is None:
        if not os.path.isfile(cpath):
            fail(f"offline: {url} chưa có trong cache")
        with open(cpath, "rb") as f:
            data = f.read()
    else:
        with open(cpath, "wb") as f:
            f.write(data)
    sha = hashlib.sha256(data).hexdigest()[:16]
    with open(os.path.join(cache_dir(), "sha-" + sha), "wb") as f:
        f.write(data)
    return data, sha


def fail(msg):
    print(f"tpm: {msg}", file=sys.stderr)
    raise SystemExit(1)


def read_manifest(d):
    p = os.path.join(d, "t.json")
    if not os.path.isfile(p):
        fail(f"thiếu {p} (chạy 'tpm init' trước)")
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def dep_main(depdir, depname):
    mj = os.path.join(depdir, "t.json")
    if os.path.isfile(mj):
        try:
            with open(mj, encoding="utf-8") as f:
                m = json.load(f).get("main")
            if m:
                return m
        except (OSError, ValueError):
            pass
    return depname + ".t"


def cmd_init(args):
    name = args[0] if args else "myapp"
    d = os.path.abspath(name)
    os.makedirs(os.path.join(d, "tests"), exist_ok=True)
    with open(os.path.join(d, "t.json"), "w", encoding="utf-8") as f:
        json.dump({"name": os.path.basename(name), "version": "0.1.0",
                   "entry": "main.t", "deps": {}}, f, indent=2)
    with open(os.path.join(d, "main.t"), "w", encoding="utf-8") as f:
        f.write('print("hello " + "tpkg")\n')
    with open(os.path.join(d, "tests", "main.t"), "w", encoding="utf-8") as f:
        f.write('assert 1 + 1 == 2\nprint("MAIN-TEST OK")\n')
    print(f"[tpm] init {d}")


def reg_url(proj):
    if os.environ.get("TPKG_REG"):
        return os.environ["TPKG_REG"].rstrip("/")
    try:
        man = read_manifest(proj)
        if man.get("registry"):
            return man["registry"].rstrip("/")
    except SystemExit:
        pass
    return "http://127.0.0.1:8765"


def reg_index(reg):
    loc = local_reg_dir(reg)
    if loc is not None:
        if not os.path.isdir(loc):
            fail(f"registry {reg}: không có thư mục {loc}")
        return build_index(loc)
    import urllib.request as _u
    try:
        with _u.urlopen(reg + "/index.json", timeout=15) as r:
            return json.load(r).get("pkgs", {})
    except Exception as ex:  # noqa: BLE001
        fail(f"registry {reg}: {ex}")


def reg_file_url(reg, fn):
    loc = local_reg_dir(reg)
    if loc is not None:
        return os.path.join(loc, fn)
    return reg + "/" + fn


def cmd_search(args):
    q = args[0].lower() if args else ""
    pkgs = reg_index(reg_url(os.getcwd())) or {}
    hit = 0
    for name in sorted(pkgs):
        if q in name.lower():
            e = pkgs[name]
            print(f"{name} {e.get('version', '?')} ({e.get('size', 0)}B)")
            hit += 1
    if not hit:
        print("(không có)")


def cmd_install(args):
    proj = os.getcwd()
    man = read_manifest(proj)
    deps = man.get("deps", {})
    tp = os.path.join(proj, "tpkgs")
    os.makedirs(tp, exist_ok=True)
    if "--offline" in args or os.environ.get("TPKG_OFFLINE"):
        OFFLINE[0] = True
        return install_offline(tp, deps)
    lock = {}
    for depname, spec in deps.items():
        if "://" in spec:
            install_url(tp, depname, spec, lock)
        elif os.path.isdir(os.path.normpath(os.path.join(proj, spec))):
            install_path(tp, proj, depname, spec, lock)
        else:
            install_name(tp, depname, spec, lock)
    with open(os.path.join(tp, "lock.json"), "w", encoding="utf-8") as f:
        json.dump(lock, f, indent=2)
    print(f"[tpm] lock {len(lock)} deps")


def install_offline(tp, deps):
    """Cài lại đúng như lock.json, chỉ từ cache, kiểm sha16."""
    lp = os.path.join(tp, "lock.json")
    if not os.path.isfile(lp):
        fail("offline: thiếu tpkgs/lock.json (chạy 'tpm install' khi có mạng trước)")
    with open(lp, encoding="utf-8") as f:
        lock = json.load(f)
    for depname in deps:
        e = lock.get(depname)
        if e is None:
            fail(f"offline: lock.json không có {depname}")
        if "path" in e:
            install_path(tp, os.getcwd(), depname, e["path"], {})
            continue
        cp = os.path.join(cache_dir(), "sha-" + e.get("sha16", ""))
        if not os.path.isfile(cp):
            fail(f"offline: {depname} (sha {e.get('sha16')}) không có trong cache")
        with open(cp, "rb") as f:
            data = f.read()
        if hashlib.sha256(data).hexdigest()[:16] != e.get("sha16"):
            fail(f"offline: sha {depname} lệch")
        unpack_into(os.path.join(tp, depname), data)
        check_dep_module(os.path.join(tp, depname), depname)
        print(f"[tpm] install {depname} <- cache (sha {e['sha16']})")
    print(f"[tpm] offline OK {len(deps)} deps")


def ver_tuple(v):
    parts = []
    for p in str(v).split("."):
        try:
            parts.append(int("".join(c for c in p if c.isdigit()) or 0))
        except ValueError:
            parts.append(0)
    while len(parts) < 3:
        parts.append(0)
    return tuple(parts[:3])


def ver_satisfies(have, spec):
    """spec: None | '1.0.0' | '^1.2' | '~1.2.3' | '>=1.2'."""
    if not spec:
        return True
    h = ver_tuple(have)
    if spec.startswith("^"):
        base = ver_tuple(spec[1:])
        return h >= base and h[0] == base[0]
    if spec.startswith("~"):
        base = ver_tuple(spec[1:])
        return h >= base and h[:2] == base[:2]
    if spec.startswith(">="):
        return h >= ver_tuple(spec[2:])
    return h == ver_tuple(spec)


def install_name(tp, depname, spec, lock):
    # spec là tên registry, kèm version ("mymath", "mymath@1.0.0",
    # "mymath@^1.0", "mymath@>=1.2").
    reg = reg_url(os.getcwd())
    if "@" in spec:
        want, ver = spec.split("@", 1)
    else:
        want, ver = spec, None
    pkgs = reg_index(reg)
    if not pkgs or want not in pkgs:
        fail(f"registry không có {want}")
    assert pkgs is not None
    e = pkgs[want]
    vers = e.get("versions") or {e.get("version", "0.0.0"): e}
    ok = [v for v in vers if ver_satisfies(v, ver)]
    if not ok:
        fail(f"{want} có {sorted(vers, key=ver_tuple)}, không bản nào thỏa {ver}")
    best = max(ok, key=ver_tuple)
    e = dict(vers[best], version=best)
    url = reg_file_url(reg, e["file"])
    if e.get("sha16"):
        print(f"[tpm] {want} {e.get('version')} (sha {e['sha16']})")
    install_url(tp, depname, url, lock)
    lock[depname]["version"] = e.get("version")
    lock[depname]["registry"] = reg


def install_path(tp, proj, depname, spec, lock):
    src = os.path.normpath(os.path.join(proj, spec))
    if not os.path.isdir(src):
        fail(f"dep {depname}: không phải thư mục: {src}")
    dst = os.path.join(tp, depname)
    if os.path.isdir(dst):
        shutil.rmtree(dst)
    shutil.copytree(src, dst, ignore=shutil.ignore_patterns("tpkgs", "__pycache__"))
    check_dep_module(dst, depname)
    main = dep_main(dst, depname)
    lock[depname] = {"path": spec, "main": main}
    print(f"[tpm] install {depname} <- {spec} (main {main})")


def check_dep_module(dst, depname):
    main = dep_main(dst, depname)
    if not os.path.isfile(os.path.join(dst, main)):
        fail(f"dep {depname}: thiếu {main} trong {dst}")
    # quy ước import: module ten <-> tpkgs/ten/ten.t (loader tìm đúng chỗ này)
    if not os.path.isfile(os.path.join(dst, depname + ".t")):
        fail(f"dep {depname}: thiếu {depname}.t (quy ước import) trong {dst}")


def install_url(tp, depname, url, lock):
    dst = os.path.join(tp, depname)
    print(f"[tpm] fetch {url}")
    data, sha = fetch_bytes(url)
    unpack_into(dst, data)
    check_dep_module(dst, depname)
    main = dep_main(dst, depname)
    lock[depname] = {"url": url, "sha16": sha, "main": main}
    print(f"[tpm] install {depname} <- {url} (sha {sha})")


def unpack_into(dst, data):
    tmp = dst + ".tmp.zip"
    with open(tmp, "wb") as f:
        f.write(data)
    if os.path.isdir(dst):
        shutil.rmtree(dst)
    os.makedirs(dst)
    try:
        with zipfile.ZipFile(tmp) as z:
            names = z.namelist()
            # bung: nếu zip bọc 1 thư mục gốc thì lột vỏ
            tops = {n.split("/")[0] for n in names if "/" in n and not n.endswith("/")}
            if len(tops) == 1 and all(n.startswith(tuple(tops)[0] + "/") or n == tuple(tops)[0] + "/" for n in names):
                top = tuple(tops)[0] + "/"
                for n in names:
                    if n == top:
                        continue
                    rel = n[len(top):]
                    if not rel:
                        continue
                    out = os.path.join(dst, rel)
                    if n.endswith("/"):
                        os.makedirs(out, exist_ok=True)
                    else:
                        os.makedirs(os.path.dirname(out) or dst, exist_ok=True)
                        with z.open(n) as src, open(out, "wb") as f:
                            shutil.copyfileobj(src, f)
            else:
                z.extractall(dst)
    finally:
        if os.path.isfile(tmp):
            os.remove(tmp)


def t_run(path):
    r = subprocess.run([sys.executable, os.path.join(TROOT, "t.py"), path],
                       capture_output=True, text=True, cwd=os.getcwd())
    return r


def cmd_run(args):
    if not args:
        fail("dùng: tpm run <file.t> [args...]")
    r = subprocess.run([sys.executable, os.path.join(TROOT, "t.py")] + args,
                       cwd=os.getcwd())
    raise SystemExit(r.returncode)


def cmd_test(args):
    tdir = os.path.join(os.getcwd(), "tests")
    files = sorted(f for f in os.listdir(tdir) if f.endswith(".t")) if os.path.isdir(tdir) else []
    if not files:
        fail("không có tests/*.t")
    bad = 0
    for fn in files:
        r = t_run(os.path.join(tdir, fn))
        ok = r.returncode == 0
        print(("PASS " if ok else "FAIL ") + fn)
        if not ok:
            bad += 1
            print(r.stderr[-400:] + r.stdout[-400:])
    print("TPM-TEST-ALL-GREEN" if not bad else f"TPM-TEST-FAIL {bad}")
    raise SystemExit(1 if bad else 0)


def cmd_pack(args):
    proj = os.getcwd()
    man = read_manifest(proj)
    out = f"{man.get('name', 'pkg')}-{man.get('version', '0.1.0')}.tpkg"
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for root, dirs, files in os.walk(proj):
            if "tpkgs" in dirs:
                dirs.remove("tpkgs")
            if "__pycache__" in dirs:
                dirs.remove("__pycache__")
            for fn in files:
                full = os.path.join(root, fn)
                z.write(full, os.path.relpath(full, proj))
    print(f"[tpm] pack {out}")
    return out


def cmd_publish(args):
    proj = os.getcwd()
    man = read_manifest(proj)
    reg = reg_url(proj)
    name = man.get("name", "pkg")
    if args:
        arches = [a for a in args if a.endswith(".tpkg")]
        path = arches[0] if arches else None
    else:
        path = None
    if path is None:
        path = cmd_pack([])
        path = os.path.join(proj, path)
    with open(path, "rb") as f:
        data = f.read()
    loc = local_reg_dir(reg)
    if loc is not None:
        os.makedirs(loc, exist_ok=True)
        shutil.copyfile(path, os.path.join(loc, os.path.basename(path)))
        pkgs = build_index(loc)
        print(f"[tpm] publish {name} -> {loc} (local, {len(pkgs)} pkgs)")
        return
    import urllib.request as _u
    req = _u.Request(reg + "/publish", data=data, method="PUT",
                     headers={"Content-Length": str(len(data)),
                              "X-Name": os.path.basename(path),
                              "X-Token": os.environ.get("TPKG_TOKEN", "")})
    try:
        with _u.urlopen(req, timeout=60) as r:
            print(f"[tpm] publish {name} -> {reg} ({r.status})")
    except Exception as ex:  # noqa: BLE001
        fail(f"publish: {ex}")


def bump(ver, part):
    ma, mi, pa = ver_tuple(ver)
    if part == "major":
        return f"{ma + 1}.0.0"
    if part == "minor":
        return f"{ma}.{mi + 1}.0"
    if part == "patch":
        return f"{ma}.{mi}.{pa + 1}"
    fail("dùng: tpm release patch|minor|major [ghi chú...]")


def git_notes(proj, since):
    """Commit từ tag v<since> (hoặc toàn bộ) nếu project là git repo."""
    try:
        rng = [f"v{since}..HEAD"] if since else []
        r = subprocess.run(["git", "log", "--pretty=format:%s"] + rng, cwd=proj,
                           capture_output=True, text=True, timeout=20)
        if r.returncode == 0:
            return [x for x in r.stdout.splitlines() if x.strip()]
    except (OSError, subprocess.SubprocessError):
        pass
    return []


def cmd_release(args):
    proj = os.getcwd()
    man = read_manifest(proj)
    if not args:
        fail("dùng: tpm release patch|minor|major [ghi chú...]")
    old = man.get("version", "0.0.0")
    new = bump(old, args[0])
    notes = [" ".join(args[1:])] if args[1:] else git_notes(proj, old)
    if not notes:
        notes = ["(không có ghi chú)"]
    man["version"] = new
    with open(os.path.join(proj, "t.json"), "w", encoding="utf-8") as f:
        json.dump(man, f, indent=2)
    cl = os.path.join(proj, "CHANGELOG.md")
    prev = ""
    if os.path.isfile(cl):
        with open(cl, encoding="utf-8") as f:
            prev = f.read()
        if prev.startswith("# Changelog"):
            prev = prev[len("# Changelog"):].lstrip("\n")
    day = os.environ.get("TPKG_DATE") or datetime.date.today().isoformat()
    sec = f"## {new} — {day}\n\n" + "".join(f"- {n}\n" for n in notes) + "\n"
    with open(cl, "w", encoding="utf-8") as f:
        f.write("# Changelog\n\n" + sec + prev)
    print(f"[tpm] release {man.get('name', 'pkg')} {old} -> {new}")


def main():
    a = sys.argv[1:]
    if not a or a[0] in ("-h", "--help", "help"):
        print(__doc__)
        return
    cmd, rest = a[0], a[1:]
    if cmd == "init":
        cmd_init(rest)
    elif cmd == "install":
        cmd_install(rest)
    elif cmd == "run":
        cmd_run(rest)
    elif cmd == "test":
        cmd_test(rest)
    elif cmd == "pack":
        cmd_pack(rest)
    elif cmd == "publish":
        cmd_publish(rest)
    elif cmd == "search":
        cmd_search(rest)
    elif cmd == "release":
        cmd_release(rest)
    elif cmd == "fetch":
        if len(rest) != 2:
            fail("dùng: tpm fetch <url> <dir>")
        url, d = rest
        os.makedirs(d, exist_ok=True)
        data = b""
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                data = r.read()
        except Exception as ex:  # noqa: BLE001
            fail(f"tải {url}: {ex}")
        tmp = os.path.join(d, ".fetch.tmp.zip")
        with open(tmp, "wb") as f:
            f.write(data)
        try:
            with zipfile.ZipFile(tmp) as z:
                z.extractall(d)
        finally:
            os.remove(tmp)
        print(f"[tpm] fetch {url} -> {d} "
              f"(sha {hashlib.sha256(data).hexdigest()[:16]})")
    else:
        fail(f"lệnh lạ: {cmd}")


if __name__ == "__main__":
    main()
