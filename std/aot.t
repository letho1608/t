# std/aot.t — Type-Guided Instant JIT & AOT Native Compiler (C / Rust CDYLIB) cho Ngôn ngữ T (v7.0 J9/AOT)
import os
import host
import ffi

let _sys = load("__sys")

fn jit(fn_val, sample_args = nil):
    if sample_args != nil:
        _sys.jit_compile(fn_val, sample_args)
    else:
        _sys.jit_compile(fn_val)
    return fn_val

fn map_type_rust(ty):
    if ty == "i64_ptr" or ty == "ptr":
        return "*mut i64"
    if ty == "f64_ptr":
        return "*mut f64"
    if ty == "float" or ty == "f64" or ty == "float64":
        return "f64"
    if ty == "bool":
        return "i64"
    return "i64"

fn map_type_c(ty):
    if ty == "i64_ptr" or ty == "ptr":
        return "long long*"
    if ty == "f64_ptr":
        return "double*"
    if ty == "float" or ty == "f64" or ty == "float64":
        return "double"
    return "long long"

fn emit_expr_rust(e, env):
    if e == nil or type(e) != "list" or e.len == 0:
        return "0"
    tag = e[0]
    if tag == "num":
        return e[1]
    if tag == "lit":
        if e[1] == "true":
            return "1"
        return "0"
    if tag == "var":
        return e[1]
    if tag == "idx":
        arr = emit_expr_rust(e[1], env)
        idx = emit_expr_rust(e[2], env)
        return "unsafe { *" + arr + ".add((" + idx + ") as usize) }"
    if tag == "un":
        op = e[1]
        v = emit_expr_rust(e[2], env)
        if op == "not":
            return "(if " + v + " == 0 { 1 } else { 0 })"
        if op == "~":
            return "(!" + v + ")"
        return "(" + op + v + ")"
    if tag == "bin":
        op = e[1]
        l = emit_expr_rust(e[2], env)
        r = emit_expr_rust(e[3], env)
        if op == "and":
            return "(if (" + l + " != 0) && (" + r + " != 0) { 1 } else { 0 })"
        if op == "or":
            return "(if (" + l + " != 0) || (" + r + " != 0) { 1 } else { 0 })"
        if op == "==" or op == "!=" or op == "<" or op == ">" or op == "<=" or op == ">=":
            return "(" + l + " " + op + " " + r + ")"
        return "(" + l + " " + op + " " + r + ")"
    if tag == "cond":
        c = emit_expr_rust(e[1], env)
        t = emit_expr_rust(e[2], env)
        f = emit_expr_rust(e[3], env)
        return "(if " + c + " { " + t + " } else { " + f + " })"
    if tag == "bltin" or tag == "call":
        mut fn_nm = ""
        if tag == "bltin":
            fn_nm = e[1]
        elif e[1][0] == "var":
            fn_nm = e[1][1]
        args = e[2]
        mut as_str = []
        for a in args:
            as_str.append(emit_expr_rust(a, env))
        if fn_nm == "int":
            return "(" + as_str[0] + " as i64)"
        if fn_nm == "float":
            return "(" + as_str[0] + " as f64)"
        if fn_nm == "abs":
            return "(" + as_str[0] + ").abs()"
        return fn_nm + "(" + ", ".join(as_str) + ")"
    return "0"

fn make_pad(indent: int) -> str:
    mut s = ""
    mut i = 0
    while i < indent:
        s = s + "    "
        i = i + 1
    return s

fn emit_stmts_rust(stmts, indent, env):
    mut out = ""
    pad = make_pad(indent)
    for s in stmts:
        if s == nil or type(s) != "list" or s.len == 0:
            continue
        tag = s[0]
        if tag == "line":
            continue
        if tag == "decl":
            nm = s[2]
            val = emit_expr_rust(s[3], env)
            out = out + pad + "let mut " + nm + " = " + val + ";\n"
        elif tag == "assign":
            if s[1][0] == "var":
                nm = s[1][1]
                val = emit_expr_rust(s[2], env)
                out = out + pad + nm + " = " + val + ";\n"
            elif s[1][0] == "idx":
                arr = emit_expr_rust(s[1][1], env)
                idx = emit_expr_rust(s[1][2], env)
                val = emit_expr_rust(s[2], env)
                out = out + pad + "unsafe { *" + arr + ".add((" + idx + ") as usize) = " + val + "; }\n"
        elif tag == "ret" or tag == "return":
            val = emit_expr_rust(s[1], env)
            out = out + pad + "return " + val + ";\n"
        elif tag == "break":
            out = out + pad + "break;\n"
        elif tag == "continue":
            out = out + pad + "continue;\n"
        elif tag == "if":
            c = emit_expr_rust(s[1], env)
            out = out + pad + "if " + c + " {\n"
            out = out + emit_stmts_rust(s[2], indent + 1, env)
            out = out + pad + "}"
            if s[3] != nil:
                out = out + " else {\n"
                out = out + emit_stmts_rust(s[3], indent + 1, env)
                out = out + pad + "}\n"
            else:
                out = out + "\n"
        elif tag == "while":
            c = emit_expr_rust(s[1], env)
            out = out + pad + "while " + c + " {\n"
            out = out + emit_stmts_rust(s[2], indent + 1, env)
            out = out + pad + "}\n"
        elif tag == "for":
            v = s[1]
            it = s[2]
            if it[0] == "bltin" and it[1] == "range":
                rargs = it[2]
                mut start_s = "0"
                mut stop_s = "0"
                if rargs.len == 1:
                    stop_s = emit_expr_rust(rargs[0], env)
                elif rargs.len >= 2:
                    start_s = emit_expr_rust(rargs[0], env)
                    stop_s = emit_expr_rust(rargs[1], env)
                out = out + pad + "let mut " + v + ": i64 = " + start_s + ";\n"
                out = out + pad + "let _end_" + v + ": i64 = " + stop_s + ";\n"
                out = out + pad + "while " + v + " < _end_" + v + " {\n"
                out = out + emit_stmts_rust(s[3], indent + 1, env)
                out = out + pad + "    " + v + " += 1;\n"
                out = out + pad + "}\n"
        elif tag == "expr":
            out = out + pad + emit_expr_rust(s[1], env) + ";\n"
    return out

fn emit_expr_c(e, env):
    if e == nil or type(e) != "list" or e.len == 0:
        return "0"
    tag = e[0]
    if tag == "num":
        return e[1]
    if tag == "lit":
        if e[1] == "true":
            return "1"
        return "0"
    if tag == "var":
        return e[1]
    if tag == "idx":
        arr = emit_expr_c(e[1], env)
        idx = emit_expr_c(e[2], env)
        return arr + "[" + idx + "]"
    if tag == "un":
        op = e[1]
        v = emit_expr_c(e[2], env)
        if op == "not":
            return "(!" + v + ")"
        return "(" + op + v + ")"
    if tag == "bin":
        op = e[1]
        l = emit_expr_c(e[2], env)
        r = emit_expr_c(e[3], env)
        if op == "and":
            return "((" + l + ") && (" + r + "))"
        if op == "or":
            return "((" + l + ") || (" + r + "))"
        return "(" + l + " " + op + " " + r + ")"
    if tag == "cond":
        c = emit_expr_c(e[1], env)
        t = emit_expr_c(e[2], env)
        f = emit_expr_c(e[3], env)
        return "((" + c + ") ? (" + t + ") : (" + f + "))"
    if tag == "bltin" or tag == "call":
        mut fn_nm = ""
        if tag == "bltin":
            fn_nm = e[1]
        elif e[1][0] == "var":
            fn_nm = e[1][1]
        args = e[2]
        mut as_str = []
        for a in args:
            as_str.append(emit_expr_c(a, env))
        if fn_nm == "int":
            return "((long long)(" + as_str[0] + "))"
        if fn_nm == "float":
            return "((double)(" + as_str[0] + "))"
        return fn_nm + "(" + ", ".join(as_str) + ")"
    return "0"

fn emit_stmts_c(stmts, indent, env):
    mut out = ""
    pad = make_pad(indent)
    for s in stmts:
        if s == nil or type(s) != "list" or s.len == 0:
            continue
        tag = s[0]
        if tag == "line":
            continue
        if tag == "decl":
            nm = s[2]
            val = emit_expr_c(s[3], env)
            is_flt = ("." in val)
            cty = "double" if is_flt else "long long"
            out = out + pad + cty + " " + nm + " = " + val + ";\n"
        elif tag == "assign":
            if s[1][0] == "var":
                nm = s[1][1]
                val = emit_expr_c(s[2], env)
                out = out + pad + nm + " = " + val + ";\n"
            elif s[1][0] == "idx":
                arr = emit_expr_c(s[1][1], env)
                idx = emit_expr_c(s[1][2], env)
                val = emit_expr_c(s[2], env)
                out = out + pad + arr + "[" + idx + "] = " + val + ";\n"
        elif tag == "ret" or tag == "return":
            val = emit_expr_c(s[1], env)
            out = out + pad + "return " + val + ";\n"
        elif tag == "break":
            out = out + pad + "break;\n"
        elif tag == "continue":
            out = out + pad + "continue;\n"
        elif tag == "if":
            c = emit_expr_c(s[1], env)
            out = out + pad + "if (" + c + ") {\n"
            out = out + emit_stmts_c(s[2], indent + 1, env)
            out = out + pad + "}"
            if s[3] != nil:
                out = out + " else {\n"
                out = out + emit_stmts_c(s[3], indent + 1, env)
                out = out + pad + "}\n"
            else:
                out = out + "\n"
        elif tag == "while":
            c = emit_expr_c(s[1], env)
            out = out + pad + "while (" + c + ") {\n"
            out = out + emit_stmts_c(s[2], indent + 1, env)
            out = out + pad + "}\n"
        elif tag == "for":
            v = s[1]
            it = s[2]
            if it[0] == "bltin" and it[1] == "range":
                rargs = it[2]
                mut start_s = "0"
                mut stop_s = "0"
                if rargs.len == 1:
                    stop_s = emit_expr_c(rargs[0], env)
                elif rargs.len >= 2:
                    start_s = emit_expr_c(rargs[0], env)
                    stop_s = emit_expr_c(rargs[1], env)
                out = out + pad + "for (long long " + v + " = " + start_s + ", _end_" + v + " = " + stop_s + "; " + v + " < _end_" + v + "; ++" + v + ") {\n"
                out = out + emit_stmts_c(s[3], indent + 1, env)
                out = out + pad + "}\n"
        elif tag == "expr":
            out = out + pad + emit_expr_c(s[1], env) + ";\n"
    return out

fn transpile_to_rust(src: str) -> str:
    host.reset_type_anns()
    toks = host.scan(src)
    st = [toks, 0]
    prog = host.parse_prog(st)
    anns = host.get_type_anns()
    mut out = "#![allow(unused_parens, unused_mut, non_snake_case)]\n\n"
    for s in prog[1]:
        if type(s) == "list" and s.len > 0 and s[0] == "fn":
            fn_nm = s[1]
            params = s[2]
            body = s[3]
            mut p_strs = []
            mut ret_r = "i64"
            if anns["fns"].has(fn_nm):
                finfo = anns["fns"][fn_nm]
                ret_r = map_type_rust(finfo["ret"])
                p_anns = finfo["params"]
                for p in params:
                    mut pty = "i64"
                    for pa in p_anns:
                        if pa[0] == p:
                            pty = map_type_rust(pa[1])
                    p_strs.append("mut " + p + ": " + pty)
            else:
                for p in params:
                    p_strs.append("mut " + p + ": i64")
            out = out + "#[no_mangle]\npub extern \"C\" fn " + fn_nm + "(" + ", ".join(p_strs) + ") -> " + ret_r + " {\n"
            out = out + emit_stmts_rust(body, 1, {})
            out = out + "}\n\n"
    return out

fn transpile_to_c(src: str) -> str:
    host.reset_type_anns()
    toks = host.scan(src)
    st = [toks, 0]
    prog = host.parse_prog(st)
    anns = host.get_type_anns()
    mut out = "#include <stdint.h>\n#include <math.h>\n#ifdef _WIN32\n#define T_EXPORT __declspec(dllexport)\n#else\n#define T_EXPORT\n#endif\n\n"
    for s in prog[1]:
        if type(s) == "list" and s.len > 0 and s[0] == "fn":
            fn_nm = s[1]
            params = s[2]
            body = s[3]
            mut p_strs = []
            mut ret_c = "long long"
            if anns["fns"].has(fn_nm):
                finfo = anns["fns"][fn_nm]
                ret_c = map_type_c(finfo["ret"])
                p_anns = finfo["params"]
                for p in params:
                    mut pty = "long long"
                    for pa in p_anns:
                        if pa[0] == p:
                            pty = map_type_c(pa[1])
                    p_strs.append(pty + " " + p)
            else:
                for p in params:
                    p_strs.append("long long " + p)
            out = out + "T_EXPORT " + ret_c + " " + fn_nm + "(" + (", ".join(p_strs) if p_strs.len > 0 else "void") + ") {\n"
            out = out + emit_stmts_c(body, 1, {})
            out = out + "}\n\n"
    return out

struct AotModule:
    lib
    syms
    rust_code
    c_code
    dll_path
    is_native

    fn sym(self, name: str) -> int:
        if not self.syms.has(name):
            self.syms[name] = self.lib.sym(name)
        return self.syms[name]

    fn call_i64(self, name: str, argtypes: str, args) -> int:
        return _sys.ffi_call(self.sym(name), "l", argtypes, args)

    fn call_f64(self, name: str, argtypes: str, args) -> float:
        return _sys.ffi_call(self.sym(name), "d", argtypes, args)

    fn close(self):
        if self.lib != nil:
            self.lib.close()
            self.lib = nil
        if self.dll_path != "" and os.exists(self.dll_path):
            os.remove(self.dll_path)

    fn __enter__(self):
        return self

    fn __exit__(self):
        self.close()

fn find_rustc() -> str:
    home = os.getenv("USERPROFILE", os.getenv("HOME", ""))
    if home != "":
        cand = home + "\\.cargo\\bin\\rustc.exe"
        if os.exists(cand):
            return cand
        cand2 = home + "/.cargo/bin/rustc"
        if os.exists(cand2):
            return cand2
    return "rustc"

fn compile_c_dll(c_code: str, out_prefix = "", fast_avx2 = true) -> str:
    base = out_prefix if out_prefix != "" else os.mktemp("taot_")
    c_path = base + ".c"
    ext = ".dll" if os.platform() == "win32" else ".so"
    dll_path = base + ext
    os.write_file(c_path, c_code)
    if os.platform() == "win32":
        bat_path = base + "_build.bat"
        opt = "/O2 /Oi /Ot /arch:AVX2 /fp:fast" if fast_avx2 else "/O2"
        bat_src = "@echo off\nsetlocal\nwhere cl >nul 2>nul\nif not errorlevel 1 goto havecl\nset \"VSWHERE=%ProgramFiles(x86)%\\Microsoft Visual Studio\\Installer\\vswhere.exe\"\nif exist \"%VSWHERE%\" (\n  for /f \"usebackq tokens=*\" %%i in (`\"%VSWHERE%\" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do set \"VSDIR=%%i\"\n)\nif defined VSDIR (\n  call \"%VSDIR%\\VC\\Auxiliary\\Build\\vcvars64.bat\" >nul 2>nul\n) else (\n  call \"C:\\Program Files (x86)\\Microsoft Visual Studio\\2022\\BuildTools\\VC\\Auxiliary\\Build\\vcvars64.bat\" >nul 2>nul\n)\n:havecl\ncl /nologo /LD " + opt + " \"" + os.abspath(c_path) + "\" /Fe\"" + os.abspath(dll_path) + "\" >nul 2>nul\n"
        os.write_file(bat_path, bat_src)
        r = os.run(["cmd.exe", "/c", bat_path])
        if os.exists(bat_path):
            os.remove(bat_path)
    else:
        opt_flag = "-ffast-math" if fast_avx2 else "-fno-fast-math"
        r = os.run(["cc", "-shared", "-O3", opt_flag, "-fPIC", c_path, "-o", dll_path])
    if os.exists(c_path):
        os.remove(c_path)
    if os.exists(base + ".obj"):
        os.remove(base + ".obj")
    if os.exists(base + ".lib"):
        os.remove(base + ".lib")
    if os.exists(base + ".exp"):
        os.remove(base + ".exp")
    ps1 = base.split("/")
    ps2 = ps1[ps1.len - 1].split("\\")
    stem = ps2[ps2.len - 1]
    if os.exists(stem + ".obj"):
        os.remove(stem + ".obj")
    if os.exists(dll_path):
        return dll_path
    return ""

fn compile_module(src: str, out_prefix = ""):
    rs_code = transpile_to_rust(src)
    c_code = transpile_to_c(src)
    base = out_prefix if out_prefix != "" else os.mktemp("taot_")
    dll_path = compile_c_dll(c_code, base, true)
    if dll_path != "" and os.exists(dll_path):
        lib = ffi.open(dll_path)
        return AotModule(lib, {}, rs_code, c_code, dll_path, true)
    return AotModule(nil, {}, rs_code, c_code, "", false)
