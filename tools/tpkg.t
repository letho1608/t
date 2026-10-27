# tpkg.t — Trình quản lý gói cho Ngôn ngữ T (v3.0, thuần T)
import os
import path
import json
import tpkg_reg

MANIFEST = "t.json"
LOCKFILE = "lock.json"

fn die(msg):
    os.stderr_write("tpkg loi: " + msg + "\n")
    os.exit(1)

fn parse_ver(s):
    ps = s.strip().split(".")
    if ps.len != 3:
        die("version khong hop le '" + s + "' (can X.Y.Z)")
    return [int(ps[0]), int(ps[1]), int(ps[2])]

fn ver_cmp(a, b):
    if a[0] != b[0]:
        return a[0] - b[0]
    if a[1] != b[1]:
        return a[1] - b[1]
    return a[2] - b[2]

fn match_spec(ver_str, spec):
    v = parse_ver(ver_str)
    mut s = spec.strip()
    if s == "" or s == "*":
        return true
    if s.startswith("^"):
        b = parse_ver(s[1:s.len])
        if b[0] > 0:
            return v[0] == b[0] and ver_cmp(v, b) >= 0
        if b[1] > 0:
            return v[0] == 0 and v[1] == b[1] and v[2] >= b[2]
        return v[0] == 0 and v[1] == 0 and v[2] == b[2]
    if s.startswith("~"):
        b = parse_ver(s[1:s.len])
        return v[0] == b[0] and v[1] == b[1] and v[2] >= b[2]
    if s.startswith(">="):
        return ver_cmp(v, parse_ver(s[2:s.len])) >= 0
    if s.startswith("<="):
        return ver_cmp(v, parse_ver(s[2:s.len])) <= 0
    if s.startswith(">"):
        return ver_cmp(v, parse_ver(s[1:s.len])) > 0
    if s.startswith("<"):
        return ver_cmp(v, parse_ver(s[1:s.len])) < 0
    if s.startswith("=="):
        s = s[2:s.len].strip()
    return ver_cmp(v, parse_ver(s)) == 0

fn pick_version(versions, spec):
    mut cands = []
    for v in versions:
        if match_spec(v, spec):
            cands.append(v)
    if cands.len == 0:
        return nil
    mut best = cands[0]
    for v in cands:
        if ver_cmp(parse_ver(v), parse_ver(best)) > 0:
            best = v
    return best

fn load_json(p):
    if not os.exists(p):
        die("khong tim thay " + p)
    return json.loads(os.read_file(p))

fn save_json(p, data):
    os.write_file(p, json.dumps(data, 2) + "\n")

fn cache_dir():
    mut d = os.getenv("TPKG_CACHE")
    if d == "":
        mut home = os.getenv("USERPROFILE")
        if home == "":
            home = os.getenv("HOME")
        d = path.join(home, ".tpkg", "cache")
    os.mkdir(d)
    return d

fn registry_url(manifest):
    mut env = os.getenv("TPKG_REG")
    if env == "":
        env = os.getenv("TPKG_REGISTRY")
    if env != "":
        return env
    if manifest != nil and manifest.has("registry"):
        return manifest["registry"]
    return ""

fn load_index(reg):
    if reg == "":
        die("chua cau hinh registry (dat TPKG_REG hoac truong 'registry' trong t.json)")
    mut p = reg
    if p.startswith("file://"):
        p = p[7:p.len]
    idx_path = path.join(p, "index.json")
    return [load_json(idx_path), p]

fn fetch_pkg(base, rel, name, ver, want_sha, offline):
    cpath = path.join(cache_dir(), name + "-" + ver + ".tpkg")
    if os.exists(cpath):
        got = os.sha256_file(cpath)
        if got == want_sha:
            return cpath
        if offline:
            die("file cache " + cpath + " bi hong sha256 va dang --offline")
        os.remove(cpath)
    if offline:
        die("khong co " + name + "-" + ver + ".tpkg trong cache (--offline)")
    src = path.join(base, rel)
    data = os.read_file(src)
    got = os.sha256(data)
    if got != want_sha:
        die("sha256 lech cho " + name + "@" + ver + ": mong " + want_sha + ", nhan " + got)
    os.write_file(cpath, data)
    return cpath

fn unpack_pkg(archive, dest_dir):
    os.mkdir(dest_dir)
    os.zip_unpack(archive, dest_dir)

fn _collect_files(root, rel_dir, out):
    cur = root if rel_dir == "" else path.join(root, rel_dir)
    for fn_name in os.listdir(cur):
        if fn_name == "tpkgs" or fn_name == "t_modules" or fn_name == "__pycache__" or fn_name.startswith(".") or fn_name.endswith(".tpkg") or fn_name.endswith(".tb"):
            continue
        full = path.join(cur, fn_name)
        rel = fn_name if rel_dir == "" else (rel_dir + "/" + fn_name)
        if os.isdir(full):
            _collect_files(root, rel, out)
        else:
            out[rel] = os.read_file(full)

fn pack_dir(src_dir, out_dir):
    mf = load_json(path.join(src_dir, MANIFEST))
    name = mf["name"]
    ver = mf["version"]
    parse_ver(ver)
    os.mkdir(out_dir)
    archive = path.join(out_dir, name + "-" + ver + ".tpkg")
    mut entries = {}
    _collect_files(src_dir, "", entries)
    if not os.zip_pack(entries, archive):
        die("zip_pack that bai: " + archive)
    return [archive, os.sha256_file(archive), mf]

fn cmd_init(av):
    if os.exists(MANIFEST):
        die(MANIFEST + " da ton tai")
    name = av[0] if av.len > 0 else path.basename(os.getcwd())
    mf = {
        "name": name,
        "version": "0.1.0",
        "description": "",
        "main": "main.t",
        "deps": {}
    }
    save_json(MANIFEST, mf)
    if not os.exists("main.t"):
        os.write_file("main.t", "print(\"xin chao tu " + name + "\")\n")
    print("da khoi tao goi " + name + "@0.1.0")

fn cmd_pack(av):
    out = av[0] if av.len > 0 else "."
    res = pack_dir(".", out)
    print("packed " + res[0] + " (sha256=" + res[1][0:16] + "...)")

fn cmd_publish(av):
    mf = load_json(MANIFEST)
    mut reg = av[0] if av.len > 0 else registry_url(mf)
    if reg == "":
        die("can chi dinh thu muc registry (tham so hoac TPKG_REG)")
    if reg.startswith("file://"):
        reg = reg[7:reg.len]
    res = pack_dir(".", reg)
    tpkg_reg.build_index(reg)
    print("published " + mf["name"] + "@" + mf["version"] + " -> " + res[0] + " (sha256=" + res[1][0:16] + "...)")

fn resolve_all(idx, root_deps):
    pkgs = idx["packages"] if idx.has("packages") else {}
    mut resolved = {}
    mut queue = []
    for k in root_deps.keys():
        queue.append([k, root_deps[k]])
    mut qi = 0
    while qi < queue.len:
        item = queue[qi]
        qi = qi + 1
        name = item[0]
        spec = item[1]
        if not pkgs.has(name):
            die("khong tim thay goi '" + name + "' trong registry")
        vers = pkgs[name]["versions"]
        if resolved.has(name):
            cur = resolved[name]["version"]
            if not match_spec(cur, spec):
                die("xung dot phien ban cho " + name + ": da chon " + cur + ", nhung can '" + spec + "'")
            continue
        v = pick_version(vers.keys(), spec)
        if v == nil:
            die("khong co phien ban nao cua '" + name + "' khop '" + spec + "'")
        info = vers[v]
        resolved[name] = {
            "version": v,
            "sha256": info["sha256"],
            "path": info["path"],
            "deps": info["deps"] if info.has("deps") else {}
        }
        d = resolved[name]["deps"]
        for dk in d.keys():
            queue.append([dk, d[dk]])
    return resolved

fn cmd_install(av):
    mut offline = false
    mut specs = []
    for a in av:
        if a == "--offline":
            offline = true
        else:
            specs.append(a)
    mf = load_json(MANIFEST)
    if not mf.has("deps"):
        mf["deps"] = {}
    for s in specs:
        if "@" in s:
            ps = s.split("@")
            mf["deps"][ps[0]] = ps[1]
        else:
            mf["deps"][s] = "*"
    if specs.len > 0:
        save_json(MANIFEST, mf)

    deps = mf["deps"]
    if deps.keys().len == 0 and not os.exists(LOCKFILE):
        print("khong co phu thuoc nao can cai")
        return nil

    reg = registry_url(mf)
    mut locked = nil
    mut base = ""
    if offline and os.exists(LOCKFILE):
        locked = load_json(LOCKFILE)["packages"]
    else:
        ib = load_index(reg)
        idx = ib[0]
        base = ib[1]
        if os.exists(LOCKFILE) and specs.len == 0:
            old = load_json(LOCKFILE)
            old_pkgs = old["packages"] if old.has("packages") else {}
            mut ok = true
            for k in deps.keys():
                if not old_pkgs.has(k) or not match_spec(old_pkgs[k]["version"], deps[k]):
                    ok = false
            if ok:
                locked = old_pkgs
        if locked == nil:
            locked = resolve_all(idx, deps)
            save_json(LOCKFILE, {"lockfileVersion": 1, "packages": locked})

    mod_dir = "tpkgs"
    os.mkdir(mod_dir)
    for name in locked.keys():
        info = locked[name]
        ver = info["version"]
        sha = info["sha256"]
        rel = info["path"] if info.has("path") else (name + "-" + ver + ".tpkg")
        arc = fetch_pkg(base, rel, name, ver, sha, offline)
        unpack_pkg(arc, path.join(mod_dir, name))
        print("installed " + name + "@" + ver + " (" + sha[0:12] + ")")

fn cmd_search(av):
    q = av[0].lower() if av.len > 0 else ""
    mf = load_json(MANIFEST) if os.exists(MANIFEST) else nil
    ib = load_index(registry_url(mf))
    pkgs = ib[0]["packages"] if ib[0].has("packages") else {}
    mut hits = 0
    for name in pkgs.keys():
        p = pkgs[name]
        desc = p["description"] if p.has("description") else ""
        if q == "" or (q in name.lower()) or (q in desc.lower()):
            vers = p["versions"].keys()
            mut latest = pick_version(vers, "*")
            if latest == nil:
                latest = "?"
            print(name + "@" + latest + "  " + desc + "  (versions: " + ", ".join(vers) + ")")
            hits = hits + 1
    if hits == 0:
        print("(khong tim thay goi nao)")

fn cmd_run(av):
    mf = load_json(MANIFEST)
    entry = av[0] if av.len > 0 else (mf["main"] if mf.has("main") else "main.t")
    r = os.run([os.exepath(), entry])
    os.stdout_write(r["out"])
    os.stderr_write(r["err"])
    if r["rc"] != 0:
        os.exit(r["rc"])

fn cmd_test(av):
    mf = load_json(MANIFEST) if os.exists(MANIFEST) else {}
    mut files = []
    if mf.has("test"):
        files = [mf["test"]]
    elif os.isdir("tests"):
        for fn_name in os.listdir("tests"):
            if fn_name.endswith(".t"):
                files.append(path.join("tests", fn_name))
    elif os.exists("test.t"):
        files = ["test.t"]
    if files.len == 0:
        die("khong tim thay file test")
    for f in files:
        print("==> test " + f)
        r = os.run([os.exepath(), f])
        os.stdout_write(r["out"])
        os.stderr_write(r["err"])
        if r["rc"] != 0:
            os.exit(r["rc"])
    print("OK (" + str(files.len) + " test file)")

fn bump_ver(ver_str, kind):
    v = parse_ver(ver_str)
    if kind == "major":
        return str(v[0] + 1) + ".0.0"
    if kind == "minor":
        return str(v[0]) + "." + str(v[1] + 1) + ".0"
    if kind == "patch":
        return str(v[0]) + "." + str(v[1]) + "." + str(v[2] + 1)
    parse_ver(kind)
    return kind

fn update_changelog(new_ver, notes):
    p = "CHANGELOG.md"
    old = os.read_file(p) if os.exists(p) else "# Changelog\n\n"
    sec = "## v" + new_ver + "\n\n- " + (notes if notes != "" else ("Release v" + new_ver)) + "\n\n"
    mut out = ""
    if old.startswith("# "):
        nl = old.find("\n")
        head = old[0:nl]
        rest = old[nl + 1:old.len].strip()
        out = head + "\n\n" + sec + (rest + "\n" if rest != "" else "")
    else:
        out = "# Changelog\n\n" + sec + old
    os.write_file(p, out)

fn cmd_release(av):
    kind = av[0] if av.len > 0 else "patch"
    notes = " ".join(av[1:av.len]) if av.len > 1 else ""
    mf = load_json(MANIFEST)
    old_ver = mf["version"] if mf.has("version") else "0.1.0"
    new_ver = bump_ver(old_ver, kind)
    mf["version"] = new_ver
    save_json(MANIFEST, mf)
    update_changelog(new_ver, notes)
    print("bumped " + mf["name"] + ": " + old_ver + " -> " + new_ver)
    reg = registry_url(mf)
    if reg != "":
        cmd_publish([reg])
    else:
        cmd_pack(["dist"])

fn main():
    raw = argv()
    a = raw[1:raw.len] if (raw.len > 0 and raw[0].endswith(".tb") or raw.len > 0 and raw[0].endswith(".t")) else raw
    if a.len == 0 or a[0] == "-h" or a[0] == "--help":
        print("tpkg - Trinh quan ly goi cho Ngon ngu T (init|install|run|test|pack|publish|search|release)")
        return nil
    cmd = a[0]
    rest = a[1:a.len]
    if cmd == "init":
        cmd_init(rest)
    elif cmd == "pack":
        cmd_pack(rest)
    elif cmd == "publish":
        cmd_publish(rest)
    elif cmd == "install":
        cmd_install(rest)
    elif cmd == "search":
        cmd_search(rest)
    elif cmd == "run":
        cmd_run(rest)
    elif cmd == "test":
        cmd_test(rest)
    elif cmd == "release":
        cmd_release(rest)
    else:
        die("lenh khong ro '" + cmd + "'")

main()
