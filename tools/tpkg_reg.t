# tpkg_reg.t — Xây index.json cho static registry tpkg từ thư mục chứa *.tpkg
import os
import path
import json

fn split_name(base):
    if not base.endswith(".tpkg"):
        return nil
    stem = base[0:base.len - 5]
    mut i = stem.len - 1
    mut rdash = -1
    while i >= 0:
        if stem[i] == "-":
            rdash = i
            break
        i = i - 1
    if rdash <= 0:
        return nil
    name = stem[0:rdash]
    ver = stem[rdash + 1:stem.len]
    parts = ver.split(".")
    if parts.len != 3:
        return nil
    for p in parts:
        if p.len == 0:
            return nil
        mut j = 0
        while j < p.len:
            c = ord(p[j])
            if c < 48 or c > 57:
                return nil
            j = j + 1
    return [name, ver]

fn _ver_key(ver):
    ps = ver.split(".")
    return int(ps[0]) * 1000000 + int(ps[1]) * 1000 + int(ps[2])

fn _sort_strs(xs):
    mut out = xs[0:xs.len]
    n = out.len
    mut i = 0
    while i < n:
        mut j = i + 1
        while j < n:
            if out[j] < out[i]:
                t = out[i]
                out[i] = out[j]
                out[j] = t
            j = j + 1
        i = i + 1
    return out

fn _sort_vers(xs):
    mut out = xs[0:xs.len]
    n = out.len
    mut i = 0
    while i < n:
        mut j = i + 1
        while j < n:
            if _ver_key(out[j]) < _ver_key(out[i]):
                t = out[i]
                out[i] = out[j]
                out[j] = t
            j = j + 1
        i = i + 1
    return out

fn _scan_dir(root, rel_prefix, found):
    names = _sort_strs(os.listdir(root))
    for fn_name in names:
        full = path.join(root, fn_name)
        rel = fn_name if rel_prefix == "" else (rel_prefix + "/" + fn_name)
        if os.isdir(full):
            _scan_dir(full, rel, found)
        elif fn_name.endswith(".tpkg"):
            found.append([full, rel, fn_name])

fn build_index(root):
    mut found = []
    _scan_dir(root, "", found)
    mut pkgs = {}
    tmp = os.mktemp("tpkg_idx")
    for item in found:
        full = item[0]
        rel = item[1]
        fn_name = item[2]
        nv = split_name(fn_name)
        if nv == nil:
            continue
        name = nv[0]
        ver = nv[1]
        raw = os.read_file(full)
        h = os.sha256(raw)
        mut desc = ""
        mut deps = {}
        ex_dir = path.join(tmp, name + "_" + ver)
        os.mkdir(ex_dir)
        os.zip_unpack(full, ex_dir)
        mut mf = path.join(ex_dir, "t.json")
        if not os.exists(mf):
            mf = path.join(ex_dir, "tpkg.json")
        if os.exists(mf):
            meta = json.loads(os.read_file(mf))
            if meta.has("description"):
                desc = meta["description"]
            if meta.has("deps"):
                deps = meta["deps"]
        if not pkgs.has(name):
            pkgs[name] = {"description": desc, "versions": {}}
        elif desc != "":
            pkgs[name]["description"] = desc
        pkgs[name]["versions"][ver] = {
            "sha256": h,
            "path": rel,
            "size": raw.len,
            "deps": deps
        }
    os.rmtree(tmp)
    mut out_pkgs = {}
    for k in _sort_strs(pkgs.keys()):
        p = pkgs[k]
        mut vs = {}
        for vk in _sort_vers(p["versions"].keys()):
            vs[vk] = p["versions"][vk]
        out_pkgs[k] = {"description": p["description"], "versions": vs}
    idx = {"version": 1, "packages": out_pkgs}
    idx_path = path.join(root, "index.json")
    os.write_file(idx_path, json.dumps(idx, 2) + "\n")
    return idx
