# std/typecheck.t — Bộ kiểm tra kiểu tĩnh tuỳ chọn (Static Type Checker, v6.0 L10)
import os
import host

fn base_type(ty):
    if ty == "" or ty == "any" or ty == "nil":
        return ty
    p = ty.find("[")
    if p > 0:
        return ty[0:p]
    return ty

fn type_compatible(expected, actual):
    if expected == "" or actual == "" or expected == "any" or actual == "any":
        return true
    if expected == actual:
        return true
    if expected == "float" and actual == "int":
        return true
    if expected == "num" and (actual == "int" or actual == "float"):
        return true
    eb = base_type(expected)
    ab = base_type(actual)
    if eb == ab and (eb == "list" or eb == "dict" or eb == "set" or eb == "tensor"):
        if expected == eb or actual == ab:
            return true
    return false

fn infer_expr(e, env, anns, errs, cur_fn):
    if e == nil or type(e) != "list" or e.len == 0:
        return "any"
    tag = e[0]
    if tag == "num":
        return "float" if ("." in e[1]) else "int"
    if tag == "int":
        return "int"
    if tag == "float":
        return "float"
    if tag == "str" or tag == "fstr":
        return "str"
    if tag == "bool":
        return "bool"
    if tag == "nil":
        return "nil"
    if tag == "lit":
        if e[1] == "true" or e[1] == "false":
            return "bool"
        return "nil"
    if tag == "list":
        mut elem_t = ""
        for item in e[1]:
            it = infer_expr(item, env, anns, errs, cur_fn)
            if elem_t == "":
                elem_t = it
            elif elem_t != it:
                elem_t = "any"
        if elem_t != "" and elem_t != "any":
            return "list[" + elem_t + "]"
        return "list"
    if tag == "dict":
        return "dict"
    if tag == "tensor":
        return "tensor"
    if tag == "var":
        nm = e[1]
        if env.has(nm):
            return env[nm]
        if anns["vars"].has(nm):
            return anns["vars"][nm]
        return "any"
    if tag == "un":
        op = e[1]
        t1 = infer_expr(e[2], env, anns, errs, cur_fn)
        if op == "not":
            return "bool"
        if op == "-":
            return t1
        return "int"
    if tag == "bin":
        op = e[1]
        t1 = infer_expr(e[2], env, anns, errs, cur_fn)
        t2 = infer_expr(e[3], env, anns, errs, cur_fn)
        if op == "==" or op == "!=" or op == "<" or op == ">" or op == "<=" or op == ">=" or op == "and" or op == "or" or op == "in":
            return "bool"
        if op == "+":
            if t1 == "str" and t2 == "str":
                return "str"
            if (t1 == "str" and t2 != "str" and t2 != "any") or (t2 == "str" and t1 != "str" and t1 != "any"):
                errs.append("TypeError: khong the cong '" + t1 + "' voi '" + t2 + "'" + (" trong ham " + cur_fn if cur_fn != "" else ""))
                return "any"
        if op == "/":
            return "float"
        if t1 == "float" or t2 == "float":
            return "float"
        if t1 == "int" and t2 == "int":
            return "int"
        return "any"
    if tag == "bltin" or tag == "call":
        mut fn_nm = ""
        if tag == "bltin":
            fn_nm = e[1]
        elif e[1][0] == "var":
            fn_nm = e[1][1]
        args = e[2]
        mut arg_types = []
        for a in args:
            arg_types.append(infer_expr(a, env, anns, errs, cur_fn))
        if fn_nm == "int" or fn_nm == "len" or fn_nm == "ord" or fn_nm == "round":
            return "int"
        if fn_nm == "float" or fn_nm == "sqrt" or fn_nm == "sin" or fn_nm == "cos" or fn_nm == "exp" or fn_nm == "log":
            return "float"
        if fn_nm == "str" or fn_nm == "chr" or fn_nm == "type":
            return "str"
        if fn_nm == "tensor" or fn_nm == "zeros" or fn_nm == "ones" or fn_nm == "full" or fn_nm == "arange" or fn_nm == "eye" or fn_nm == "rand" or fn_nm == "randn":
            return "tensor"
        if fn_nm != "" and anns["fns"].has(fn_nm):
            finfo = anns["fns"][fn_nm]
            p_anns = finfo["params"]
            mut k = 0
            while k < args.len and k < p_anns.len:
                pnm = p_anns[k][0]
                want_t = p_anns[k][1]
                got_t = arg_types[k]
                if want_t != "" and want_t != "any" and not type_compatible(want_t, got_t):
                    errs.append("TypeError: tham so '" + pnm + "' cua ham '" + fn_nm + "' can kieu '" + want_t + "', nhung nhan '" + got_t + "'")
                k = k + 1
            return finfo["ret"] if finfo["ret"] != "" else "any"
        return "any"
    if tag == "attr":
        obj_t = infer_expr(e[1], env, anns, errs, cur_fn)
        fld = e[2]
        if anns["structs"].has(obj_t):
            st_info = anns["structs"][obj_t]
            for fp in st_info["fields"]:
                if fp[0] == fld:
                    return fp[1]
        if fld == "len":
            return "int"
        return "any"
    return "any"

fn check_stmts(stmts, env, anns, errs, cur_fn, want_ret):
    for s in stmts:
        if s == nil or type(s) != "list" or s.len == 0:
            continue
        tag = s[0]
        if tag == "line":
            continue
        if tag == "decl":
            nm = s[2]
            val_t = infer_expr(s[3], env, anns, errs, cur_fn)
            mut want_t = ""
            if env.has(nm) and env[nm] != "any":
                want_t = env[nm]
            elif cur_fn == "" and anns["vars"].has(nm):
                want_t = anns["vars"][nm]
            if want_t != "" and not type_compatible(want_t, val_t):
                errs.append("TypeError: bien '" + nm + "' co kieu '" + want_t + "', khong the gan gia tri kieu '" + val_t + "'")
            if want_t != "":
                env[nm] = want_t
            elif not env.has(nm) or env[nm] == "any":
                env[nm] = val_t
        elif tag == "assign":
            if s[1][0] == "var":
                nm = s[1][1]
                val_t = infer_expr(s[2], env, anns, errs, cur_fn)
                mut want_t = ""
                if env.has(nm) and env[nm] != "any":
                    want_t = env[nm]
                elif cur_fn == "" and anns["vars"].has(nm):
                    want_t = anns["vars"][nm]
                if want_t != "" and not type_compatible(want_t, val_t):
                    errs.append("TypeError: bien '" + nm + "' co kieu '" + want_t + "', khong the gan gia tri kieu '" + val_t + "'")
        elif tag == "ret" or tag == "return":
            rt = infer_expr(s[1], env, anns, errs, cur_fn)
            if want_ret != "" and want_ret != "any" and not type_compatible(want_ret, rt):
                errs.append("TypeError: ham '" + cur_fn + "' khai bao tra ve '" + want_ret + "', nhung return '" + rt + "'")
        elif tag == "expr":
            infer_expr(s[1], env, anns, errs, cur_fn)
        elif tag == "if":
            infer_expr(s[1], env, anns, errs, cur_fn)
            check_stmts(s[2], env, anns, errs, cur_fn, want_ret)
            if s[3] != nil:
                check_stmts(s[3], env, anns, errs, cur_fn, want_ret)
        elif tag == "while":
            infer_expr(s[1], env, anns, errs, cur_fn)
            check_stmts(s[2], env, anns, errs, cur_fn, want_ret)
        elif tag == "for":
            infer_expr(s[2], env, anns, errs, cur_fn)
            check_stmts(s[3], env, anns, errs, cur_fn, want_ret)
        elif tag == "try":
            check_stmts(s[1], env, anns, errs, cur_fn, want_ret)
            for h in s[2]:
                check_stmts(h[1], env, anns, errs, cur_fn, want_ret)
        elif tag == "fn":
            fn_nm = s[1]
            params = s[2]
            body = s[3]
            mut fn_env = {}
            for k in env.keys():
                fn_env[k] = env[k]
            mut fn_ret = ""
            if anns["fns"].has(fn_nm):
                finfo = anns["fns"][fn_nm]
                fn_ret = finfo["ret"]
                p_anns = finfo["params"]
                for p in params:
                    mut found_pt = "any"
                    for pa in p_anns:
                        if pa[0] == p:
                            found_pt = pa[1]
                    fn_env[p] = found_pt
            else:
                for p in params:
                    fn_env[p] = "any"
            check_stmts(body, fn_env, anns, errs, fn_nm, fn_ret)
    return nil

fn check_source(src):
    host.reset_type_anns()
    toks = host.scan(src)
    st = [toks, 0]
    prog = host.parse_prog(st)
    anns = host.get_type_anns()
    mut errs = []
    mut env = {}
    for k in anns["vars"].keys():
        env[k] = anns["vars"][k]
    check_stmts(prog[1], env, anns, errs, "", "")
    return {"ok": errs.len == 0, "errors": errs, "annotations": anns}

fn check_file(path):
    src = os.read_file(path)
    return check_source(src)
