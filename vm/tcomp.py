"""tcomp.py — compiler T -> bytecode .tb v1 (full language + bundler).
File .tb: magic b"TB01" + u16 nglobals + u32 len + code bytes.
OP được dựng theo đúng thứ tự enum trong vm/tvm.h (đừng sửa tay số).
Dùng:
  python vm/tcomp.py prog.t prog.tb        # có import thì tự bundle
  python vm/tcomp.py -b prog.t prog.tb     # ép bundle (giữ tương thích)
Phạm vi v1: mọi syntax; giới hạn: closure chỉ bắt global (không bắt biến
hàm ngoài), fn lồng tham chiếu nhau phải định nghĩa trước khi dùng.
Ngữ nghĩa scope khớp t.py: decl trong block (kể cả top-level) không lọt ra.
"""
import os, struct, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from t import P, lex2, file_base

_OPNAMES = """PUSH_NIL PUSH_BOOL PUSH_INT PUSH_FLOAT PUSH_STR NEW_LIST
NEW_TENSOR NEW_DICT ADD SUB MUL DIV MOD POW NEG NOT EQ NE LT GT LE GE
CONTAINS MATCHGLOB TDOT IDX SETIDX SLICE LEN PRINT JMP JMPF JMPT CALL RET LOAD STORE
DECL LOADG STOREG DECLG MKFN CAPLOCAL CAPUP CAPG LOADU STOREU BIND BINDG RANGE ATTR
RAISE TRY_PUSH TRY_POP FOR_PREP POP DUP INC_L INC_G DEC_L DEC_G ADDI_L ADDI_G
ADDS_L ADDS_G SUBS_L SUBS_G INCB_L INCB_G IN CALLM SETATTR NOP HALT""".split()
OP = {n: 0x01 + i for i, n in enumerate(_OPNAMES)}

NAT = {n: i for i, n in enumerate(
    ["range", "len", "str", "int", "float", "type", "tensor", "open",
     "read", "write", "close", "ord", "chr", "argv", "print", "mem",
     "f64bits", "abs", "round", "min", "max", "sum", "any", "all",
     "sort", "reversed", "enumerate", "zip", "map", "filter",
     # builtin "mềm" (không phải keyword: user được che tên)
     "set", "input", "__struct", "__new",
     # tensor + toán học (builtin mềm)
     "zeros", "ones", "full", "arange", "eye", "rand", "randn", "seed",
     "sqrt", "exp", "log", "sin", "cos", "tanh", "floor", "ceil", "pi", "load"])}
NFIXED = len(NAT)
BINOP = {"+": "ADD", "-": "SUB", "*": "MUL", "/": "DIV", "%": "MOD",
         "**": "POW", "@": "TDOT", "==": "EQ", "!=": "NE", "<": "LT",
         ">": "GT", "<=": "LE", ">=": "GE", "in": "IN"}
BLTIN_CALL = set(NAT)


class CErr(SystemExit):
    pass


def fail(msg):
    raise CErr(f"tcomp: {msg}")


def range_args(it):
    """for-range pattern: trả [args] nếu it là range(a[,b]), else None."""
    if it[0] == "bltin" and it[1] == "range" and 1 <= len(it[2]) <= 2:
        return it[2]
    return None


def count_nodes(e):
    """Đếm node AST expr (giới hạn inline)."""
    k = e[0]
    if k in ("num", "str", "lit", "var", "bltfn"):
        return 1
    if k in ("tensor", "un"):
        return 1 + count_nodes(e[1] if k == "tensor" else e[2])
    if k == "bltin":
        return 1 + sum(count_nodes(x) for x in e[2])
    if k == "bin":
        return 1 + count_nodes(e[2]) + count_nodes(e[3])
    if k in ("idx", "attr"):
        return 1 + count_nodes(e[1]) + (count_nodes(e[2]) if k == "idx" else 0)
    if k == "slice":
        n = 1 + count_nodes(e[1])
        for p in (e[2], e[3], e[4]):
            if p is not None:
                n += count_nodes(p)
        return n
    if k == "list":
        return 1 + sum(count_nodes(x) for x in e[1])
    if k == "dict":
        return 1 + sum(count_nodes(x) for x in e[1]) + sum(count_nodes(x) for x in e[2])
    if k == "call":
        return 1 + count_nodes(e[1]) + sum(count_nodes(x) for x in e[2])
    if k == "fstr":
        n = 1
        for kk, v in e[1]:
            if kk == "e":
                n += count_nodes(v)
        return n
    if k == "cond":
        return 1 + count_nodes(e[1]) + count_nodes(e[2]) + count_nodes(e[3])
    return 1000000


def free_vars(e, bound):
    """Tên var tự do trong expr (trừ bound)."""
    k = e[0]
    if k == "var":
        return {e[1]} - bound
    if k in ("num", "str", "lit", "bltfn"):
        return set()
    if k == "tensor":
        return free_vars(e[1], bound)
    if k == "un":
        return free_vars(e[2], bound)
    if k == "bltin":
        s = set()
        for x in e[2]:
            s |= free_vars(x, bound)
        return s
    if k == "bin":
        return free_vars(e[2], bound) | free_vars(e[3], bound)
    if k == "idx":
        return free_vars(e[1], bound) | free_vars(e[2], bound)
    if k == "attr":
        return free_vars(e[1], bound)
    if k == "slice":
        s = free_vars(e[1], bound)
        for p in (e[2], e[3], e[4]):
            if p is not None:
                s |= free_vars(p, bound)
        return s
    if k == "list":
        s = set()
        for x in e[1]:
            s |= free_vars(x, bound)
        return s
    if k == "dict":
        s = set()
        for x in e[1]:
            s |= free_vars(x, bound)
        for x in e[2]:
            s |= free_vars(x, bound)
        return s
    if k == "call":
        s = free_vars(e[1], bound)
        for x in e[2]:
            s |= free_vars(x, bound)
        return s
    if k == "fstr":
        s = set()
        for kk, v in e[1]:
            if kk == "e":
                s |= free_vars(v, bound)
        return s
    if k == "cond":
        return free_vars(e[1], bound) | free_vars(e[2], bound) | free_vars(e[3], bound)
    return set()


INLINE_MAXNODES = 40


class Scope:
    """Scope từ vựng: name -> slot. iter=True: scope theo vòng lặp (mỗi
    vòng tươi) — fn lồng capture biến trong đó phải snapshot (CAPG)."""
    def __init__(self, it=False):
        self.m = {}
        self.imm = set()  # tên bất biến (let / fn) — gán lại -> lỗi như t.py
        self.iter = it
        self.kids = []    # cây scope con (prescan build, emit tái dùng)
        self._next = 0

    def has_here(self, name):
        return name in self.m


class FnCtx:
    def __init__(self, is_main, nargs=0):
        self.is_main = is_main
        self.scopes = [Scope()]
        self.next = nargs
        self.max = nargs
        self.uval = {}   # upvalue name -> idx (chỉ fn thường)

    def push_scope(self, it=False):
        parent = self.scopes[-1]
        if parent._next < len(parent.kids):
            child = parent.kids[parent._next]
        else:
            child = Scope(it)
            parent.kids.append(child)
        parent._next += 1
        self.scopes.append(child)

    def pop_scope(self):
        self.scopes.pop()

    def reset_walk(self):
        """Về root để walk lại (giữ slot + cây scope)."""
        def rec(sc):
            sc._next = 0
            for k in sc.kids:
                rec(k)
        rec(self.scopes[0])
        self.scopes = [self.scopes[0]]

    def find(self, name):
        for sc in reversed(self.scopes):
            if name in sc.m:
                return sc.m[name]
        return None

    def find2(self, name):
        """-> (slot, is_iter_scope) của scope gần nhất chứa tên."""
        for sc in reversed(self.scopes):
            if name in sc.m:
                return (sc.m[name], sc.iter)
        return None

    def alloc_here(self, name):
        """Cấp slot trong scope hiện tại (prescan). Trùng -> fail."""
        sc = self.scopes[-1]
        if name in sc.m:
            fail(f"redeclare {name}")
        ix = self.next
        self.next += 1
        if self.next > self.max:
            self.max = self.next
        sc.m[name] = ix
        return ix


class C:
    def __init__(self):
        self.b = bytearray()
        self.gnames = {}   # top name -> global idx (cấp khi bind lần đầu)
        self.fn_names = set()  # tên fn top-level (cho phép tham chiếu trước)
        self.fnmap = {}    # id(stmt fn) -> FnCtx (prescan build, emit dùng)
        self.fntab = {}    # tên fn top single-return -> (params, bodyexpr)
        self.topbinds = set()  # mọi tên bind ở top (cho rule free-var inline)
        self.inlinemap = {}  # (id(node call), id(fn đang dịch)) -> (fname, [temps])
        self.ichain = []   # fn đang inline (chống đệ quy inline)
        self.preglobals = []  # biến global top-level đã cấp ở pha 1
        self.loops = []
        self.fnstack = []
        self.lines = []    # (pc, dòng, chỉ số file) — bảng dòng cho lỗi runtime
        self.lfiles = []
        self.tdepth = 0    # số try đang mở (break/continue phải TRY_POP)

    def u8(self, v): self.b.append(v & 0xFF)
    def u16(self, v): self.b += struct.pack("<H", v)
    def i32(self, v): self.b += struct.pack("<i", v)
    def i64(self, v): self.b += struct.pack("<q", v)
    def f64(self, v): self.b += struct.pack("<d", v)
    def pos(self): return len(self.b)

    def jmp(self, op):
        self.u8(OP[op])
        at = len(self.b)
        self.b += struct.pack("<i", 0)
        return at

    def patch(self, at):
        struct.pack_into("<i", self.b, at, len(self.b) - (at + 4))

    def patch_to(self, at, target):
        struct.pack_into("<i", self.b, at, target - (at + 4))

    def curfn(self):
        return self.fnstack[-1] if self.fnstack else None

    def gslot(self, name):
        if name not in self.gnames:
            self.gnames[name] = NFIXED + len(self.gnames)
        return self.gnames[name]

    def get_here(self, name):
        """Slot đã prescan trong scope hiện tại (emit). Thiếu -> drift."""
        sc = self.curfn().scopes[-1]
        if name not in sc.m:
            fail(f"lỗi nội bộ: thiếu prescan {name}")
        return sc.m[name]

    def pre_alloc(self, name):
        """Cấp slot prescan trong scope hiện tại (main-aware). Trùng -> fail."""
        fn = self.curfn()
        sc = fn.scopes[-1]
        if sc.has_here(name):
            fail(f"redeclare {name}")
        if fn.is_main:
            ix = self.gslot(name + "@" + str(len(self.gnames)))
        else:
            ix = fn.next
            fn.next += 1
            if fn.next > fn.max:
                fn.max = fn.next
        sc.m[name] = ix
        return ix

    def pre_found(self, name):
        """Prescan: tên đã thấy ở scope ngoài/main/fn/builtin chưa?"""
        for f2 in reversed(self.fnstack):
            if f2.find(name) is not None:
                return True
        if name in self.fn_names or name in NAT:
            return True
        return False

    def _resolve(self, name):
        fn = self.curfn()
        if fn is not None and not fn.is_main:
            s = fn.find(name)
            if s is not None:
                return ("local", s)
            # tìm ở fn ngoài -> upvalue (ghi chuỗi truyền dẫn)
            for d in range(len(self.fnstack) - 2, -1, -1):
                f2 = self.fnstack[d]
                if f2.is_main:
                    break
                if f2.find(name) is not None:
                    return self.use_upval(name, d)
            main = self.fnstack[0]
            r = main.find2(name)
            if r is not None:
                s, it = r
                if it:
                    # biến vòng-lặp ở top: snapshot mỗi lần định nghĩa fn
                    return self.use_upval_main(name)
                return ("global", s)
        elif fn is not None:
            s = fn.find(name)
            if s is not None:
                return ("global", s)
        if name in self.fn_names:
            # fn top-level: cho tham chiếu trước (đệ quy/tương hỗ, như t.py lúc chạy)
            return ("global", self.gnames[name])
        if name in NAT:
            return ("global", NAT[name])
        return None

    def use_upval_main(self, name):
        cur = self.fnstack[-1]
        if name not in cur.uval:
            cur.uval[name] = len(cur.uval)
        return ("upval", cur.uval[name])

    def inlineable(self, fname, nargs):
        """Đủ điều kiện inline? single-return thuần expr, đúng arity,
        không đệ quy (chain), free-var trong topbinds/NAT."""
        if os.environ.get("TCOMP_NOINLINE"):
            return None
        e = self.fntab.get(fname)
        if e is None:
            return None
        params, body = e
        if len(params) != nargs or fname in self.ichain:
            return None
        if count_nodes(body) > INLINE_MAXNODES:
            return None
        if free_vars(body, set(params)) - self.topbinds - set(NAT):
            return None
        return (params, body)

    def resolve(self, name):
        r = self._resolve(name)
        if r is None:
            fail(f"undef {name}")
        return r

    def use_upval(self, name, defdepth):
        for j in range(len(self.fnstack) - 2, defdepth, -1):
            hj = self.fnstack[j]
            if name not in hj.uval:
                hj.uval[name] = len(hj.uval)
        cur = self.fnstack[-1]
        if name not in cur.uval:
            cur.uval[name] = len(cur.uval)
        return ("upval", cur.uval[name])

    def emit_load(self, kind, ix):
        self.u8(OP["LOAD"] if kind == "local" else
                OP["LOADG"] if kind == "global" else OP["LOADU"])
        self.u16(ix)

    # ---------------- expr ----------------
    def expr(self, e):
        k = e[0]
        if k == "num":
            v = e[1]
            if isinstance(v, bool):
                self.b += bytes([OP["PUSH_BOOL"], 1 if v else 0])
            elif isinstance(v, int):
                self.u8(OP["PUSH_INT"]); self.i64(v)
            else:
                self.u8(OP["PUSH_FLOAT"]); self.f64(float(v))
        elif k == "lit":
            m = {True: 1, False: 0, None: None}[e[1]]
            if m is None:
                self.u8(OP["PUSH_NIL"])
            else:
                self.b += bytes([OP["PUSH_BOOL"], m])
        elif k == "str":
            s = e[1].encode("utf-8")
            self.u8(OP["PUSH_STR"])
            self.b += struct.pack("<I", len(s)); self.b += s
        elif k == "fstr":
            ps = e[1]
            if len(ps) == 0:
                self.u8(OP["PUSH_STR"])
                self.b += struct.pack("<I", 0)
            else:
                first = True
                for kind, val in ps:
                    if kind == "s":
                        s = val.encode("utf-8")
                        self.u8(OP["PUSH_STR"])
                        self.b += struct.pack("<I", len(s)); self.b += s
                    else:
                        self.u8(OP["LOADG"]); self.u16(NAT["str"])
                        self.expr(val)
                        self.b += bytes([OP["CALL"], 1])
                    if not first:
                        self.u8(OP["ADD"])
                    first = False
        elif k == "var":
            kind, ix = self.resolve(e[1])
            self.emit_load(kind, ix)
        elif k == "bltfn":
            if e[1] != "print":
                fail(f"bare builtin {e[1]}")
            self.u8(OP["LOADG"]); self.u16(NAT["print"])
        elif k == "list":
            if len(e[1]) > 65535:
                fail("list quá lớn")
            for x in e[1]:
                self.expr(x)
            self.u8(OP["NEW_LIST"]); self.u16(len(e[1]))
        elif k == "dict":
            ks, vs = e[1], e[2]
            if len(ks) > 65535:
                fail("dict quá lớn")
            for kk, vv in zip(ks, vs):
                self.expr(kk); self.expr(vv)
            self.u8(OP["NEW_DICT"]); self.u16(len(ks))
        elif k == "tensor":
            self.u8(OP["LOADG"]); self.u16(NAT["tensor"])
            self.expr(e[1])
            self.b += bytes([OP["CALL"], 1])
        elif k == "bltin":
            if e[1] not in BLTIN_CALL:
                fail(f"bltin {e[1]}")
            if len(e[2]) > 255:
                fail("quá nhiều arg")
            self.u8(OP["LOADG"]); self.u16(NAT[e[1]])
            for x in e[2]:
                self.expr(x)
            self.b += bytes([OP["CALL"], len(e[2])])
        elif k == "un":
            self.expr(e[2])
            if e[1] == "-":
                self.u8(OP["NEG"])
            elif e[1] == "not":
                self.u8(OP["NOT"])
            else:
                fail(f"unary {e[1]}")
        elif k == "bin":
            op = e[1]
            if op == "and":
                self.expr(e[2])
                self.u8(OP["DUP"])
                f = self.jmp("JMPF")
                self.u8(OP["POP"])
                self.expr(e[3])
                end = self.jmp("JMP")
                self.patch(f)
                self.patch(end)
            elif op == "or":
                self.expr(e[2])
                self.u8(OP["DUP"])
                t = self.jmp("JMPT")
                self.u8(OP["POP"])
                self.expr(e[3])
                end = self.jmp("JMP")
                self.patch(t)
                self.patch(end)
            elif op in BINOP:
                self.expr(e[2]); self.expr(e[3])
                self.u8(OP[BINOP[op]])
            else:
                fail(f"binop {op}")
        elif k == "cond":
            self.expr(e[1])
            f = self.jmp("JMPF")
            self.expr(e[2])
            end = self.jmp("JMP")
            self.patch(f)
            self.expr(e[3])
            self.patch(end)
        elif k == "idx":
            self.expr(e[1]); self.expr(e[2])
            self.u8(OP["IDX"])
        elif k == "slice":
            self.expr(e[1])
            for part in (e[2], e[3], e[4]):
                if part is None:
                    self.u8(OP["PUSH_NIL"])
                else:
                    self.expr(part)
            self.u8(OP["SLICE"])
        elif k == "attr":
            self.expr(e[1])
            s = e[2].encode("utf-8")
            self.u8(OP["ATTR"])
            self.b += struct.pack("<I", len(s)); self.b += s
        elif k == "call":
            if e[1][0] == "var" and self.ikey(e) in self.inlinemap:
                self.emit_inline(e)
            elif e[1][0] == "attr":
                # obj.m(args): CALLM argc name (gọi method trực tiếp, inline cache)
                self.expr(e[1][1])
                if len(e[2]) > 255:
                    fail("quá nhiều arg")
                for x in e[2]:
                    self.expr(x)
                s = e[1][2].encode("utf-8")
                self.b += bytes([OP["CALLM"], len(e[2])])
                self.b += struct.pack("<I", len(s)); self.b += s
            else:
                self.expr(e[1])
                if len(e[2]) > 255:
                    fail("quá nhiều arg")
                for x in e[2]:
                    self.expr(x)
                self.b += bytes([OP["CALL"], len(e[2])])
        else:
            fail(f"expr {k}")

    def is_immutable(self, name):
        """Tên gần nhất trong tầm nhìn là let/fn? (t.py: 'x immutable (use mut)')"""
        for f2 in reversed(self.fnstack):
            for sc in reversed(f2.scopes):
                if name in sc.m:
                    return name in sc.imm
        return name in self.fn_names

    def do_store(self, name):
        """Gán: local->STORE, global->STOREG, upval->STOREU, lạ->auto-decl
        (prescan đã cấp)."""
        fn = self.curfn()
        if self.is_immutable(name):
            fail(f"{name} immutable (use mut)")
        r = self._resolve(name)
        if r is None:
            if name in NAT:
                fail(f"gán vào builtin {name}")
            self.u8(OP["STOREG"] if (fn is not None and fn.is_main) else OP["STORE"])
            self.u16(self.get_here(name))
            return
        kind, ix = r
        if kind == "local":
            self.u8(OP["STORE"])
        elif kind == "global":
            if name in NAT and ix == NAT.get(name):
                fail(f"gán vào builtin {name}")
            self.u8(OP["STOREG"])
        else:
            self.u8(OP["STOREU"])
        self.u16(ix)

    # ---------------- stmt ----------------
    def stmt(self, s):
        k = s[0]
        fn = self.curfn()
        main = fn is not None and fn.is_main
        if k == "line":
            if s[2] not in self.lfiles:
                self.lfiles.append(s[2])
            ent = (self.pos(), s[1], self.lfiles.index(s[2]))
            if self.lines and self.lines[-1][1:] == ent[1:]:
                return
            if self.lines and self.lines[-1][0] == ent[0]:
                self.lines[-1] = ent
            else:
                self.lines.append(ent)
            return
        if k == "decl":
            self.expr(s[3])
            if not main:
                self.u8(OP["DECL"]); self.u16(self.get_here(s[2]))
            else:
                self.u8(OP["STOREG"]); self.u16(self.get_here(s[2]))
        elif k == "assign":
            t, val = s[1], s[2]
            if t[0] == "var":
                self.expr(val)
                self.do_store(t[1])
            elif t[0] == "idx":
                self.expr(t[1]); self.expr(t[2]); self.expr(val)
                self.u8(OP["SETIDX"])
            elif t[0] == "attr":
                self.expr(t[1])
                self.expr(val)
                s = t[2].encode("utf-8")
                self.u8(OP["SETATTR"])
                self.b += struct.pack("<I", len(s)); self.b += s
            else:
                fail("bad assign target")
        elif k == "fn":
            _, name, args, body = s
            if len(args) > 255:
                fail("quá nhiều param")
            if not main:
                slot = self.get_here(name)
                islocal = True
            else:
                # top: prescan đã bind (kể cả prefn đệ quy)
                slot = self.get_here(name)
                islocal = False
            over = self.jmp("JMP")
            entry = self.pos()
            fctx = self.fnmap[id(s)]
            # params đã prescan vào scope0 — kiểm tra khớp
            for i, a in enumerate(args):
                if fctx.scopes[0].m.get(a) != i:
                    fail(f"lỗi nội bộ param {a}")
            self.fnstack.append(fctx)
            self.block(body)
            self.u8(OP["PUSH_NIL"]); self.u8(OP["RET"])
            self.fnstack.pop()
            self.patch(over)
            self.u8(OP["MKFN"]); self.i32(entry)
            self.u8(len(args)); self.u16(fctx.max); self.u16(len(fctx.uval))
            # capture theo thứ tự uval (khớp LOADU idx)
            for nm in fctx.uval:
                enc = self.curfn()
                if enc is not None and not enc.is_main:
                    s = enc.find(nm)
                    if s is not None:
                        self.u8(OP["CAPLOCAL"]); self.u16(s); continue
                    if nm in enc.uval:
                        self.u8(OP["CAPUP"]); self.u16(enc.uval[nm]); continue
                    fail(f"capture {nm} trước khi bind")
                # bao ngoài là main: chỉ capture được biến vòng-lặp (CAPG
                # snapshot); biến main thường đã resolve global từ trước
                r = enc.find2(nm) if enc is not None else None
                if r is None or not r[1]:
                    fail(f"capture {nm} trước khi bind")
                self.u8(OP["CAPG"]); self.u16(r[0])
            if islocal:
                self.u8(OP["DECL"]); self.u16(slot)
            else:
                self.u8(OP["STOREG"]); self.u16(slot)
        elif k == "ret":
            if main:
                fail("return ngoài hàm")
            self.expr(s[1])
            self.u8(OP["RET"])
        elif k == "print":
            if len(s[1]) > 255:
                fail("print quá 255 arg")
            for x in s[1]:
                self.expr(x)
            self.b += bytes([OP["PRINT"], len(s[1])])
        elif k == "expr":
            self.expr(s[1])
            self.u8(OP["POP"])
        elif k == "if":
            _, cond, a, b = s
            self.expr(cond)
            f = self.jmp("JMPF")
            self.scoped(a)
            end = self.jmp("JMP")
            self.patch(f)
            if b:
                if len(b) == 1 and isinstance(b[0], tuple) and b[0][0] == "if":
                    self.stmt(b[0])
                else:
                    self.scoped(b)
            self.patch(end)
        elif k == "while":
            _, cond, body = s
            loop = self.pos()
            self.expr(cond)
            f = self.jmp("JMPF")
            self.loops.append({"breaks": [], "conts": [], "td": self.tdepth})
            self.scoped(body)
            for p in self.loops[-1]["conts"]:
                self.patch(p)
            back = self.jmp("JMP")
            self.patch_to(back, loop)
            self.patch(f)
            for p in self.loops[-1]["breaks"]:
                self.patch(p)
            self.loops.pop()
        elif k == "for":
            _, v, it, body = s
            self.scoped_for(v, it, body)
        elif k == "break":
            if not self.loops:
                fail("break ngoài vòng lặp")
            for _ in range(self.tdepth - self.loops[-1]["td"]):
                self.u8(OP["TRY_POP"])
            self.loops[-1]["breaks"].append(self.jmp("JMP"))
        elif k == "continue":
            if not self.loops:
                fail("continue ngoài vòng lặp")
            for _ in range(self.tdepth - self.loops[-1]["td"]):
                self.u8(OP["TRY_POP"])
            self.loops[-1]["conts"].append(self.jmp("JMP"))
        elif k == "raise":
            self.expr(s[1])
            self.u8(OP["RAISE"])
        elif k == "assert":
            _, cond, msg = s
            self.expr(cond)
            ok = self.jmp("JMPT")
            if msg is None:
                ms = "assert failed".encode()
                self.u8(OP["PUSH_STR"])
                self.b += struct.pack("<I", len(ms)); self.b += ms
            else:
                pre = "assert failed: ".encode()
                self.u8(OP["PUSH_STR"])
                self.b += struct.pack("<I", len(pre)); self.b += pre
                self.expr(msg)
                self.u8(OP["ADD"])
            self.u8(OP["RAISE"])
            self.patch(ok)
        elif k == "try":
            _, body, handlers = s
            push_h = self.jmp("TRY_PUSH")
            self.tdepth += 1
            self.scoped(body)
            self.tdepth -= 1
            self.u8(OP["TRY_POP"])
            norm = self.jmp("JMP")
            self.patch(push_h)
            fn.push_scope()
            es = self.get_here("err")
            dones = [norm]
            for nm, hb in handlers:
                if nm is None:
                    self.emit_store_slot(es)
                    self.scoped(hb)
                    dones.append(self.jmp("JMP"))
                    break
                self.u8(OP["DUP"])
                nb = nm.encode()
                self.u8(OP["PUSH_STR"])
                self.b += struct.pack("<I", len(nb)); self.b += nb
                if "*" in nm or "?" in nm:
                    self.u8(OP["MATCHGLOB"])
                else:
                    self.u8(OP["CONTAINS"])
                nx = self.jmp("JMPF")
                self.emit_store_slot(es)
                self.scoped(hb)
                dones.append(self.jmp("JMP"))
                self.patch(nx)
            else:
                self.u8(OP["RAISE"])
            fn.pop_scope()
            for d in dones:
                self.patch(d)
        elif k in ("import", "fromimport"):
            fail(f"{k} trực tiếp chưa hỗ trợ — dùng 'tcomp.py -b' để bundle")
        else:
            fail(f"stmt {k}")

    def emit_store_slot(self, ix):
        fn = self.curfn()
        if fn.is_main:
            self.u8(OP["STOREG"])
        else:
            self.u8(OP["STORE"])
        self.u16(ix)

    def emit_load_slot(self, ix):
        fn = self.curfn()
        if fn.is_main:
            self.u8(OP["LOADG"])
        else:
            self.u8(OP["LOAD"])
        self.u16(ix)

    def emit_bind_slot(self, ix):
        # gán mới, vứt giá trị cũ (không xuyên cell) — biến lặp mỗi vòng tươi
        fn = self.curfn()
        if fn is not None and fn.is_main:
            self.u8(OP["BINDG"])
        else:
            self.u8(OP["BIND"])
        self.u16(ix)

    def emit_inline(self, e):
        """Inline call đã prescan: tính args vào temps, biên dịch body expr
        với params gắn temps. Kết quả nằm trên stack."""
        fname, temps = self.inlinemap[self.ikey(e)]
        ent = self.inlineable(fname, len(e[2]))
        if ent is None:
            fail(f"lỗi nội bộ inline {fname}")
        params, body = ent
        fn = self.curfn()
        for t, a in zip(temps, e[2]):
            self.expr(a)
            if fn is not None and fn.is_main:
                self.u8(OP["STOREG"])
            else:
                self.u8(OP["STORE"])
            self.u16(t)
        # scope param TẠM, không qua kids-tree (prescan không có nó)
        pscope = Scope()
        for p, t in zip(params, temps):
            pscope.m[p] = t
        fn.scopes.append(pscope)
        self.ichain.append(fname)
        self.expr(body)
        self.ichain.pop()
        fn.scopes.pop()

    def scoped_for(self, v, it, body):
        fn = self.curfn()
        ra = range_args(it)
        if ra is None:
            self.scoped_for_general(v, it, body)
        else:
            self.scoped_for_range(v, ra, body)

    def emit_intcall(self, e):
        self.u8(OP["LOADG"]); self.u16(NAT["int"])
        self.expr(e)
        self.b += bytes([OP["CALL"], 1])

    def scoped_for_range(self, v, ra, body):
        # for v in range(a[,b]) -> v=int(a); $hi=int(b|a); while v<$hi (không list)
        fn = self.curfn()
        fn.push_scope(True)
        v_s = self.get_here(v)
        hi_s = self.get_here("$hi")
        if len(ra) == 1:
            self.u8(OP["PUSH_INT"]); self.i64(0)
        else:
            self.emit_intcall(ra[0])
        self.emit_bind_slot(v_s)
        self.emit_intcall(ra[-1])
        self.emit_bind_slot(hi_s)
        loop = self.pos()
        self.emit_load_slot(v_s)
        self.emit_load_slot(hi_s)
        self.u8(OP["LT"])
        f = self.jmp("JMPF")
        self.loops.append({"breaks": [], "conts": [], "td": self.tdepth})
        fn.push_scope(True)
        for x in body:
            self.stmt(x)
        fn.pop_scope()
        for p in self.loops[-1]["conts"]:
            self.patch(p)
        self.emit_load_slot(v_s)
        self.u8(OP["PUSH_INT"]); self.i64(1)
        self.u8(OP["ADD"])
        self.emit_bind_slot(v_s)
        back = self.jmp("JMP")
        self.patch_to(back, loop)
        self.patch(f)
        for p in self.loops[-1]["breaks"]:
            self.patch(p)
        self.loops.pop()
        fn.pop_scope()

    def scoped_for_general(self, v, it, body):
        fn = self.curfn()
        fn.push_scope(True)
        it_s = self.get_here("$it")
        i_s = self.get_here("$i")
        v_s = self.get_here(v)
        self.expr(it)
        self.u8(OP["FOR_PREP"])
        self.emit_store_slot(it_s)
        self.u8(OP["PUSH_INT"]); self.i64(0)
        self.emit_store_slot(i_s)
        loop = self.pos()
        self.emit_load_slot(i_s)
        self.emit_load_slot(it_s)
        self.u8(OP["LEN"])
        self.u8(OP["LT"])
        f = self.jmp("JMPF")
        self.loops.append({"breaks": [], "conts": [], "td": self.tdepth})
        self.emit_load_slot(it_s)
        self.emit_load_slot(i_s)
        self.u8(OP["IDX"])
        self.emit_bind_slot(v_s)
        # body trong scope RIÊNG iter (decl trong body tươi mỗi vòng)
        fn.push_scope(True)
        for x in body:
            self.stmt(x)
        fn.pop_scope()
        for p in self.loops[-1]["conts"]:
            self.patch(p)
        self.emit_load_slot(i_s)
        self.u8(OP["PUSH_INT"]); self.i64(1)
        self.u8(OP["ADD"])
        self.emit_store_slot(i_s)
        back = self.jmp("JMP")
        self.patch_to(back, loop)
        self.patch(f)
        for p in self.loops[-1]["breaks"]:
            self.patch(p)
        self.loops.pop()
        fn.pop_scope()

    def scoped(self, stmts):
        fn = self.curfn()
        fn.push_scope()
        for s in stmts:
            self.stmt(s)
        fn.pop_scope()

    def block(self, stmts):
        for s in stmts:
            self.stmt(s)

    # ---------- prescan: cấp slot trước cho mọi bind (kể cả bind sau) ----------
    def prescan(self, stmts):
        for s in stmts:
            self.prescan_stmt(s)

    def ikey(self, e):
        """Khoá inline: theo TỪNG node call + hàm đang dịch. (Theo tên hàm thì
        add(1, add(2, 3)) dùng chung temp -> lệnh trong ghi đè lệnh ngoài;
        chỉ theo node thì thân inline dùng chung giữa 2 hàm lấy nhầm slot.)"""
        return (id(e), id(self.curfn()))

    def prescan_expr(self, e):
        """Tìm điểm inline trong expr: cấp temps, ghi inlinemap, đệ quy."""
        k = e[0]
        if k == "call" and e[1][0] == "var":
            fname = e[1][1]
            ent = self.inlineable(fname, len(e[2]))
            if ent is not None and self.ikey(e) not in self.inlinemap:
                temps = []
                for a in e[2]:
                    temps.append(self.pre_temp())
                self.inlinemap[self.ikey(e)] = (fname, temps)
                self.ichain.append(fname)
                _, body = ent
                self.prescan_expr(body)
                self.ichain.pop()
            for x in e[2]:
                self.prescan_expr(x)
            if e[1][0] != "var":
                self.prescan_expr(e[1])
            return
        if k in ("num", "str", "lit", "var", "bltfn"):
            return
        if k == "tensor":
            self.prescan_expr(e[1])
        elif k == "un":
            self.prescan_expr(e[2])
        elif k == "bltin":
            for x in e[2]:
                self.prescan_expr(x)
        elif k == "bin":
            self.prescan_expr(e[2]); self.prescan_expr(e[3])
        elif k == "idx":
            self.prescan_expr(e[1]); self.prescan_expr(e[2])
        elif k == "attr":
            self.prescan_expr(e[1])
        elif k == "slice":
            self.prescan_expr(e[1])
            for p in (e[2], e[3], e[4]):
                if p is not None:
                    self.prescan_expr(p)
        elif k == "list":
            for x in e[1]:
                self.prescan_expr(x)
        elif k == "dict":
            for x in e[1]:
                self.prescan_expr(x)
            for x in e[2]:
                self.prescan_expr(x)
        elif k == "call":
            self.prescan_expr(e[1])
            for x in e[2]:
                self.prescan_expr(x)
        elif k == "fstr":
            for kk, v in e[1]:
                if kk == "e":
                    self.prescan_expr(v)
        elif k == "cond":
            self.prescan_expr(e[1]); self.prescan_expr(e[2]); self.prescan_expr(e[3])

    def pre_temp(self):
        """Temp ẩn $inN trong scope hiện tại (user không viết được $)."""
        fn = self.curfn()
        sc = fn.scopes[-1]
        nm = f"$in{len(self.inlinemap)}_${fn.next}"
        if fn.is_main:
            ix = self.gslot(nm + "@" + str(len(self.gnames)))
        else:
            ix = fn.next
            fn.next += 1
            if fn.next > fn.max:
                fn.max = fn.next
        sc.m[nm] = ix
        return ix

    def prescan_scope(self, stmts, it=False):
        self.curfn().push_scope(it)
        self.prescan(stmts)
        self.curfn().pop_scope()

    def prescan_stmt(self, s):
        k = s[0]
        fn = self.curfn()
        if k == "decl":
            self.prescan_expr(s[3])
            self.pre_alloc(s[2])
            if s[1] == "let":
                self.curfn().scopes[-1].imm.add(s[2])
        elif k == "assign" and s[1][0] == "var":
            self.prescan_expr(s[2])
            if not self.pre_found(s[1][1]):
                self.pre_alloc(s[1][1])
        elif k == "assign":
            self.prescan_expr(s[1][1])
            if s[1][0] == "idx":
                self.prescan_expr(s[1][2])
            self.prescan_expr(s[2])
        elif k == "fn":
            _, name, args, body = s
            if len(args) > 255:
                fail("quá nhiều param")
            self.pre_alloc(name)
            fn.scopes[-1].imm.add(name)
            fctx = FnCtx(False, len(args))
            for i, a in enumerate(args):
                if a in fctx.scopes[0].m:
                    fail(f"trùng param {a}")
                fctx.scopes[0].m[a] = i
            self.fnmap[id(s)] = fctx
            self.fnstack.append(fctx)
            self.prescan(body)
            self.fnstack.pop()
        elif k == "for":
            _, v, it, body = s
            self.prescan_expr(it)
            fn.push_scope(True)
            if range_args(it) is None:
                for n in ("$it", "$i", v):
                    self.pre_alloc(n)
            else:
                for n in (v, "$hi"):
                    self.pre_alloc(n)
            fn.push_scope(True)
            self.prescan(body)
            fn.pop_scope()
            fn.pop_scope()
        elif k == "while":
            self.prescan_expr(s[1])
            self.prescan_scope(s[2])
        elif k == "if":
            self.prescan_expr(s[1])
            self.prescan_scope(s[2])
            b = s[3]
            if b:
                if len(b) == 1 and isinstance(b[0], tuple) and b[0][0] == "if":
                    self.prescan([b[0]])
                else:
                    self.prescan_scope(b)
        elif k == "try":
            self.prescan_scope(s[1])
            for nm, hb in s[2]:
                fn.push_scope()
                self.pre_alloc("err")
                self.prescan(hb)
                fn.pop_scope()
        elif k in ("print",):
            for x in s[1]:
                self.prescan_expr(x)
        elif k in ("expr", "ret", "raise"):
            self.prescan_expr(s[1])
        elif k == "assert":
            self.prescan_expr(s[1])
            if s[2] is not None:
                self.prescan_expr(s[2])
        elif k in ("break", "continue"):
            pass
        elif k in ("import", "fromimport"):
            pass

    def pre_alloc(self, name):
        fn = self.curfn()
        sc = fn.scopes[-1]
        if sc.has_here(name):
            # đã cấp ở pha 1 (prescan_globals) -> dùng lại
            if name in self.preglobals:
                return sc.m[name]
            fail(f"redeclare {name}")
        if fn.is_main:
            ix = self.gslot(name + "@" + str(len(self.gnames)))
        else:
            ix = fn.next
            fn.next += 1
            if fn.next > fn.max:
                fn.max = fn.next
        sc.m[name] = ix
        return ix

    def pre_found(self, name):
        for f2 in reversed(self.fnstack):
            if f2.find(name) is not None:
                return True
        return name in self.fn_names

    def prescan_globals(self, stmts):
        """Pha 1: cấp slot cho mọi biến GLOBAL top-level (let/mut + gán).
        Nhờ vậy hàm định nghĩa trước vẫn thấy biến khai báo sau."""
        for s in stmts:
            k = s[0]
            if k == "decl":
                self.preglobals.append(s[2])
                self.pre_alloc(s[2])
            elif k == "assign" and s[1][0] == "var":
                self.preglobals.append(s[1][1])
                self.pre_alloc(s[1][1])

    def pre_global(self, name):
        """(giữ cho tương lai) cấp slot global ở root scope của main."""
        main = self.fnstack[0]
        sc = main.scopes[0]
        if name in sc.m:
            return sc.m[name]
        ix = self.gslot(name + "@" + str(len(self.gnames)))
        sc.m[name] = ix
        return ix

    def patch_to(self, at, target):
        struct.pack_into("<i", self.b, at, target - (at + 4))


def topbind_walk(stmts, acc):
    """Mọi tên bind ở top (mọi độ sâu block, KHÔNG vào thân fn) cho rule
    free-var inline (chỉ global thật mới được tham chiếu từ body inline)."""
    for s in stmts:
        k = s[0]
        if k == "decl":
            acc.add(s[2])
        elif k == "fn":
            acc.add(s[1])
        elif k == "assign" and s[1][0] == "var":
            acc.add(s[1][1])
        elif k == "for":
            acc.add(s[1])
            topbind_walk(s[3], acc)
        elif k == "while":
            topbind_walk(s[2], acc)
        elif k == "if":
            topbind_walk(s[2], acc)
            if s[3]:
                if len(s[3]) == 1 and isinstance(s[3][0], tuple) and s[3][0][0] == "if":
                    topbind_walk([s[3][0]], acc)
                else:
                    topbind_walk(s[3], acc)
        elif k == "try":
            topbind_walk(s[1], acc)
            for _, hb in s[2]:
                topbind_walk(hb, acc)


def op_size(code, pc):
    """Độ dài lệnh tại pc (mirror oplen trong tvm.c)."""
    op = code[pc]
    R = OP
    if op in (R["ADD"], R["SUB"], R["MUL"], R["DIV"], R["MOD"], R["POW"],
              R["NEG"], R["NOT"], R["EQ"], R["NE"], R["LT"], R["GT"],
              R["LE"], R["GE"], R["CONTAINS"], R["MATCHGLOB"], R["TDOT"], R["IDX"],
              R["SETIDX"], R["SLICE"], R["LEN"], R["RET"], R["POP"],
              R["DUP"], R["RAISE"], R["TRY_POP"], R["FOR_PREP"],
              R["HALT"], R["NOP"], R["IN"]):
        return 1
    if op in (R["PUSH_BOOL"], R["CALL"], R["PRINT"], R["RANGE"]):
        return 2
    if op in (R["PUSH_INT"], R["PUSH_FLOAT"]):
        return 9
    if op in (R["NEW_LIST"], R["NEW_TENSOR"], R["NEW_DICT"], R["LOAD"],
              R["STORE"], R["DECL"], R["LOADG"], R["STOREG"], R["DECLG"],
              R["LOADU"], R["STOREU"], R["BIND"], R["BINDG"],
              R["CAPLOCAL"], R["CAPUP"], R["CAPG"]):
        return 3
    if op in (R["JMP"], R["JMPF"], R["JMPT"], R["TRY_PUSH"]):
        return 5
    if op in (R["PUSH_STR"], R["ATTR"], R["SETATTR"]):
        ln = struct.unpack("<I", bytes(code[pc + 1:pc + 5]))[0]
        return 5 + ln
    if op == R["CALLM"]:
        ln = struct.unpack("<I", bytes(code[pc + 2:pc + 6]))[0]
        return 6 + ln
    if op == R["MKFN"]:
        return 10
    if op in (R["ADDI_L"], R["ADDI_G"]):
        return 11
    if op in (R["INC_L"], R["INC_G"], R["DEC_L"], R["DEC_G"],
                R["INCB_L"], R["INCB_G"]):
        return 3
    if op in (R["ADDS_L"], R["ADDS_G"], R["SUBS_L"], R["SUBS_G"]):
        return 5
    return 1


def peephole(c):
    """Gộp dãy op quen thuộc thành superop + NOP (giữ nguyên vị trí)."""
    b = c.b
    n = len(b)
    # thu thập pc được nhảy tới / entry (không thay trong đó)
    prot = {0}
    pc = 0
    ops = []
    while pc < n:
        sz = op_size(b, pc)
        ops.append((pc, b[pc], sz))
        o = b[pc]
        if o in (OP["JMP"], OP["JMPF"], OP["JMPT"], OP["TRY_PUSH"]):
            rel = struct.unpack("<i", bytes(b[pc + 1:pc + 5]))[0]
            prot.add(pc + 5 + rel)
        elif o == OP["MKFN"]:
            e = struct.unpack("<i", bytes(b[pc + 1:pc + 5]))[0]
            prot.add(e)
        pc += sz
    def rd_u16(at):
        return struct.unpack("<H", bytes(b[at:at + 2]))[0]
    i = 0
    while i < len(ops):
        if i + 3 < len(ops):
            p0, o0, s0 = ops[i]
            p1, o1, s1 = ops[i + 1]
            p2, o2, s2 = ops[i + 2]
            p3, o3, s3 = ops[i + 3]
            end = p3 + s3
            inside = any(p in prot for p in range(p0 + 1, end))
            if not inside:
                ld = o0 in (OP["LOAD"], OP["LOADG"])
                # STORE/STOREG/BIND/BINDG cuối (gán) — khớp loại + slot.
                # BIND (biến lặp) chỉ hợp INCB (tươi mỗi vòng, không xuyên cell).
                if ld and o1 == OP["PUSH_INT"] and o2 in (OP["ADD"], OP["SUB"]) and o3 in (OP["STORE"], OP["STOREG"], OP["BIND"], OP["BINDG"]):
                    s = rd_u16(p0 + 1)
                    isg = (o0 == OP["LOADG"])
                    want = ("STORE", "BIND") if not isg else ("STOREG", "BINDG")
                    if s == rd_u16(p3 + 1) and o3 == OP[want[0]] or s == rd_u16(p3 + 1) and o3 == OP[want[1]]:
                        k = struct.unpack("<q", bytes(b[p1 + 1:p1 + 9]))[0]
                        isb = (o3 == OP[want[1]])
                        if o2 == OP["ADD"] and k == 1 and not isb:
                            opc = OP["INC_G"] if isg else OP["INC_L"]
                            rep = bytes([opc]) + struct.pack("<H", s)
                        elif o2 == OP["ADD"] and k == 1 and isb:
                            opc = OP["INCB_G"] if isg else OP["INCB_L"]
                            rep = bytes([opc]) + struct.pack("<H", s)
                        elif o2 == OP["SUB"] and k == 1 and not isb:
                            opc = OP["DEC_G"] if isg else OP["DEC_L"]
                            rep = bytes([opc]) + struct.pack("<H", s)
                        elif o2 == OP["ADD"] and not isb:
                            opc = OP["ADDI_G"] if isg else OP["ADDI_L"]
                            rep = bytes([opc]) + struct.pack("<H", s) + struct.pack("<q", k)
                        else:
                            rep = None
                        # SUB k!=1 hoặc BIND phi-INC: giữ nguyên (hiếm)
                        if rep is not None and len(rep) <= end - p0:
                            b[p0:p0 + len(rep)] = rep
                            for q in range(p0 + len(rep), end):
                                b[q] = OP["NOP"]
                            i += 4
                            continue
                if ld and o1 in (OP["LOAD"], OP["LOADG"]) and o2 in (OP["ADD"], OP["SUB"]) and o3 in (OP["STORE"], OP["STOREG"]):
                    s = rd_u16(p0 + 1)
                    x = rd_u16(p1 + 1)
                    isg = (o0 == OP["LOADG"])
                    # cùng loại local/global trên cả 3 lệnh
                    if s == rd_u16(p3 + 1) and (o1 == OP["LOAD"]) == (not isg) and (o3 == OP["STORE"]) == (not isg):
                        if o2 == OP["ADD"]:
                            opc = OP["ADDS_G"] if isg else OP["ADDS_L"]
                        else:
                            opc = OP["SUBS_G"] if isg else OP["SUBS_L"]
                        rep = bytes([opc]) + struct.pack("<H", s) + struct.pack("<H", x)
                        if len(rep) <= end - p0:
                            b[p0:p0 + len(rep)] = rep
                            for q in range(p0 + len(rep), end):
                                b[q] = OP["NOP"]
                            i += 4
                            continue
        i += 1


def compile_prog(prog):
    c = C()
    if prog[0] != "prog":
        fail("AST gốc")
    # pre-alloc tên fn top-level (đệ quy + tương hỗ) + bảng inline + topbinds
    topbind_walk(prog[1], c.topbinds)
    for s in prog[1]:
        if s[0] == "fn" and s[1] not in NAT:
            if s[1] not in c.gnames:
                c.gslot(s[1])
            c.fn_names.add(s[1])
            bs = body_stmts(s[3])
            if len(bs) == 1 and bs[0][0] == "ret":
                c.fntab[s[1]] = (list(s[2]), bs[0][1])
    main = FnCtx(True)
    c.main = main
    c.fnstack.append(main)
    c.prescan_globals(prog[1])     # pha 1: global top-level (kể cả khai sau)
    c.prescan(prog[1])          # cấp slot trước (kể cả bind sau)
    for fctx in [main] + list(c.fnmap.values()):
        fctx.reset_walk()
    for s in prog[1]:
        c.stmt(s)
    c.fnstack.pop()
    peephole(c)
    c.b.append(OP["HALT"])
    return c


# ---------- bundler ----------
def find_mod(dotted, refdir):
    rel = dotted.replace(".", "/") + ".t"
    leaf = rel.split("/")[-1]
    stem = leaf[:-2] if leaf.endswith(".t") else leaf
    tp = os.path.join("tpkgs", stem, stem + ".t")
    cand = [os.path.join(refdir, rel),
            os.path.join(refdir, "std", leaf),
            os.path.join(refdir, "self", leaf),
            os.path.join(refdir, "tools", leaf),
            os.path.join(refdir, tp),
            os.path.join(os.getcwd(), rel),
            os.path.join(os.getcwd(), "std", leaf),
            os.path.join(os.getcwd(), "self", leaf),
            os.path.join(os.getcwd(), "tools", leaf),
            os.path.join(os.getcwd(), tp),
            os.path.join(os.path.dirname(refdir), rel),
            os.path.join(os.path.dirname(refdir), "std", leaf),
            os.path.join(os.path.dirname(refdir), "self", leaf),
            os.path.join(os.path.dirname(refdir), "tools", leaf),
            os.path.join(os.path.dirname(refdir), tp)]
    seen = []
    for p in cand:
        a = os.path.normpath(p)
        if a not in seen:
            seen.append(a)
    for p in seen:
        if os.path.isfile(p):
            return p
    fail(f"no module {dotted!r}")


def parse_file(path):
    with open(path, encoding="utf-8") as f:
        return P(lex2(f.read()), file_base(path)).parse()


def body_stmts(body):
    """thân hàm bỏ mốc dòng (nhận diện hàm single-return để inline)"""
    return [x for x in body if x[0] != "line"]


def collect_bounds(stmts, cur, fbound):
    for s in stmts:
        k = s[0]
        if k == "fn":
            key = id(s)
            b = set(s[2])
            gather(s[3], b)
            fbound[key] = b
            collect_bounds(s[3], key, fbound)
            continue
        if cur is not None:
            if k == "decl":
                fbound[cur].add(s[2])
            elif k == "assign" and s[1][0] == "var":
                fbound[cur].add(s[1][1])
            elif k == "for":
                fbound[cur].add(s[1])
        if k == "if":
            collect_bounds(s[2], cur, fbound)
            if s[3]:
                if len(s[3]) == 1 and isinstance(s[3][0], tuple) and s[3][0][0] == "if":
                    collect_bounds([s[3][0]], cur, fbound)
                else:
                    collect_bounds(s[3], cur, fbound)
        elif k == "while":
            collect_bounds(s[2], cur, fbound)
        elif k == "for":
            collect_bounds(s[3], cur, fbound)
        elif k == "try":
            collect_bounds(s[1], cur, fbound)
            for _, hb in s[2]:
                collect_bounds(hb, cur, fbound)


def gather(stmts, b):
    for s in stmts:
        k = s[0]
        if k == "decl":
            b.add(s[2])
        elif k == "fn":
            b.add(s[1])
        elif k == "assign" and s[1][0] == "var":
            b.add(s[1][1])
        elif k == "for":
            b.add(s[1])


def bundle(entry_path):
    entry_path = os.path.abspath(entry_path)
    units = []
    done, busy = set(), set()

    def load(modkey, path):
        if modkey in done:
            return
        if modkey in busy:
            fail(f"circular import {modkey}")
        busy.add(modkey)
        prog = parse_file(path)
        refdir = os.path.dirname(os.path.abspath(path))
        for s in prog[1]:
            if s[0] == "import":
                load(s[1], find_mod(s[1], refdir))
            elif s[0] == "fromimport":
                load(s[1], find_mod(s[1], refdir))
        units.append((modkey, prog[1]))
        done.add(modkey)
        busy.discard(modkey)

    load("", entry_path)
    alias_mod = {}
    for modkey, stmts in units:
        for s in stmts:
            if s[0] == "import":
                a = s[2] or s[1].split(".")[-1]
                alias_mod[(modkey, a)] = s[1]
            elif s[0] == "fromimport":
                for n in s[2]:
                    alias_mod[(modkey, n)] = (s[1], n)
    # map tên global của các unit phụ (cho cross-ref như parse_stmt trong parse.t)
    xbind = {}
    for modkey, stmts in units:
        if modkey == "":
            continue
        pre = modkey.replace(".", "_") + "__"
        binds = []
        for s in stmts:
            k = s[0]
            if k == "decl" and s[2] not in binds:
                binds.append(s[2])
            elif k == "fn" and s[1] not in binds:
                binds.append(s[1])
            elif k == "assign" and s[1][0] == "var" and s[1][1] not in binds:
                binds.append(s[1][1])
        for b in binds:
            if b not in xbind:
                xbind[b] = pre + b
    out = []
    for modkey, stmts in units:
        pre = "" if modkey == "" else modkey.replace(".", "_") + "__"
        binds = []
        for s in stmts:
            k = s[0]
            if k == "decl" and s[2] not in binds:
                binds.append(s[2])
            elif k == "fn" and s[1] not in binds:
                binds.append(s[1])
            elif k == "assign" and s[1][0] == "var" and s[1][1] not in binds:
                binds.append(s[1][1])
        ren = {b: pre + b for b in binds}
        fbound = {}
        collect_bounds(stmts, None, fbound)
        for s in stmts:
            if s[0] == "import":
                continue
            elif s[0] == "fromimport":
                _, mod, names = s
                for n in names:
                    out.append(("decl", "let", n,
                                ("var", mod.replace(".", "_") + "__" + n)))
            else:
                out.append(rewrite_stmt(s, modkey, pre, ren, alias_mod, fbound, xbind, None, ren))
    return ("prog", out)


def rewrite_var(name, modkey, pre, ren, alias_mod, fbound, xbind, cur, lren):
    if name in lren:
        return ("var", lren[name])
    am = alias_mod.get((modkey, name))
    if isinstance(am, tuple):
        mod, orig = am
        return ("var", mod.replace(".", "_") + "__" + orig)
    if name in ren:
        return ("var", ren[name])
    if modkey != "" and name in xbind:
        return ("var", xbind[name])
    return ("var", name)


def rw_target(t, modkey, pre, ren, alias_mod, fbound, xbind, cur, lren):
    if t[0] == "var":
        return rewrite_var(t[1], modkey, pre, ren, alias_mod, fbound, xbind, cur, lren)
    if t[0] == "idx":
        return ("idx",
                rewrite_expr(t[1], modkey, pre, ren, alias_mod, fbound, xbind, cur, lren),
                rewrite_expr(t[2], modkey, pre, ren, alias_mod, fbound, xbind, cur, lren))
    if t[0] == "attr":
        r = rewrite_expr(t, modkey, pre, ren, alias_mod, fbound, xbind, cur, lren)
        return r
    return t


def rewrite_expr(e, modkey, pre, ren, alias_mod, fbound, xbind, cur, lren):
    k = e[0]
    if k == "var":
        return rewrite_var(e[1], modkey, pre, ren, alias_mod, fbound, xbind, cur, lren)
    if k in ("num", "str", "lit", "bltfn"):
        return e
    if k == "fstr":
        return ("fstr", [("s", v) if kk == "s" else
                         ("e", rewrite_expr(v, modkey, pre, ren, alias_mod, fbound, xbind, cur, lren))
                         for kk, v in e[1]])
    if k == "list":
        return ("list", [rewrite_expr(x, modkey, pre, ren, alias_mod, fbound, xbind, cur, lren) for x in e[1]])
    if k == "dict":
        return ("dict",
                [rewrite_expr(x, modkey, pre, ren, alias_mod, fbound, xbind, cur, lren) for x in e[1]],
                [rewrite_expr(x, modkey, pre, ren, alias_mod, fbound, xbind, cur, lren) for x in e[2]])
    if k == "tensor":
        return ("tensor", rewrite_expr(e[1], modkey, pre, ren, alias_mod, fbound, xbind, cur, lren))
    if k == "bltin":
        return ("bltin", e[1], [rewrite_expr(x, modkey, pre, ren, alias_mod, fbound, xbind, cur, lren) for x in e[2]])
    if k == "un":
        return ("un", e[1], rewrite_expr(e[2], modkey, pre, ren, alias_mod, fbound, xbind, cur, lren))
    if k == "bin":
        return ("bin", e[1], rewrite_expr(e[2], modkey, pre, ren, alias_mod, fbound, xbind, cur, lren),
                rewrite_expr(e[3], modkey, pre, ren, alias_mod, fbound, xbind, cur, lren))
    if k == "idx":
        return ("idx", rewrite_expr(e[1], modkey, pre, ren, alias_mod, fbound, xbind, cur, lren),
                rewrite_expr(e[2], modkey, pre, ren, alias_mod, fbound, xbind, cur, lren))
    if k == "slice":
        return ("slice", rewrite_expr(e[1], modkey, pre, ren, alias_mod, fbound, xbind, cur, lren),
                rewrite_expr(e[2], modkey, pre, ren, alias_mod, fbound, xbind, cur, lren) if e[2] is not None else None,
                rewrite_expr(e[3], modkey, pre, ren, alias_mod, fbound, xbind, cur, lren) if e[3] is not None else None,
                rewrite_expr(e[4], modkey, pre, ren, alias_mod, fbound, xbind, cur, lren) if e[4] is not None else None)
    if k == "attr":
        base = e[1]
        if base[0] == "var":
            am = alias_mod.get((modkey, base[1]))
            if isinstance(am, str):
                return ("var", am.replace(".", "_") + "__" + e[2])
            if isinstance(am, tuple):
                fail(f"{base[1]} là fromimport, không attr")
        return ("attr", rewrite_expr(base, modkey, pre, ren, alias_mod, fbound, xbind, cur, lren), e[2])
    if k == "call":
        return ("call", rewrite_expr(e[1], modkey, pre, ren, alias_mod, fbound, xbind, cur, lren),
                [rewrite_expr(x, modkey, pre, ren, alias_mod, fbound, xbind, cur, lren) for x in e[2]])
    if k == "cond":
        return ("cond", rewrite_expr(e[1], modkey, pre, ren, alias_mod, fbound, xbind, cur, lren),
                rewrite_expr(e[2], modkey, pre, ren, alias_mod, fbound, xbind, cur, lren),
                rewrite_expr(e[3], modkey, pre, ren, alias_mod, fbound, xbind, cur, lren))
    fail(f"bundler expr {k}")


def rewrite_stmt(s, modkey, pre, ren, alias_mod, fbound, xbind, cur, lren):
    k = s[0]
    if k == "decl":
        return ("decl", s[1], lren.get(s[2], s[2]),
                rewrite_expr(s[3], modkey, pre, ren, alias_mod, fbound, xbind, cur, lren))
    if k == "fn":
        # hàm lồng trong hàm của module: đổi tên theo lren (khớp chỗ dùng)
        nm = ren.get(s[1], s[1]) if cur is None else lren.get(s[1], s[1])
        key = id(s)
        myren = lren
        if modkey != "" and key in fbound:
            myren = dict(lren)
            for b in fbound[key]:
                myren[b] = pre + b
        return ("fn", nm, [myren.get(a, a) for a in s[2]],
                [rewrite_stmt(x, modkey, pre, ren, alias_mod, fbound, xbind, key, myren) for x in s[3]])
    if k == "assign":
        t = s[1]
        if t[0] == "var":
            t2 = rw_target(t, modkey, pre, ren, alias_mod, fbound, xbind, cur, lren)
            return ("assign", t2,
                    rewrite_expr(s[2], modkey, pre, ren, alias_mod, fbound, xbind, cur, lren))
        return ("assign", rw_target(t, modkey, pre, ren, alias_mod, fbound, xbind, cur, lren),
                rewrite_expr(s[2], modkey, pre, ren, alias_mod, fbound, xbind, cur, lren))
    if k == "ret":
        return ("ret", rewrite_expr(s[1], modkey, pre, ren, alias_mod, fbound, xbind, cur, lren))
    if k == "print":
        return ("print", [rewrite_expr(x, modkey, pre, ren, alias_mod, fbound, xbind, cur, lren) for x in s[1]])
    if k == "expr":
        return ("expr", rewrite_expr(s[1], modkey, pre, ren, alias_mod, fbound, xbind, cur, lren))
    if k == "if":
        b = s[3]
        if b and len(b) == 1 and isinstance(b[0], tuple) and b[0][0] == "if":
            b2 = [rewrite_stmt(b[0], modkey, pre, ren, alias_mod, fbound, xbind, cur, lren)]
        elif b:
            b2 = [rewrite_stmt(x, modkey, pre, ren, alias_mod, fbound, xbind, cur, lren) for x in b]
        else:
            b2 = None
        return ("if", rewrite_expr(s[1], modkey, pre, ren, alias_mod, fbound, xbind, cur, lren),
                [rewrite_stmt(x, modkey, pre, ren, alias_mod, fbound, xbind, cur, lren) for x in s[2]], b2)
    if k == "for":
        return ("for", lren.get(s[1], s[1]),
                rewrite_expr(s[2], modkey, pre, ren, alias_mod, fbound, xbind, cur, lren),
                [rewrite_stmt(x, modkey, pre, ren, alias_mod, fbound, xbind, cur, lren) for x in s[3]])
    if k == "while":
        return ("while", rewrite_expr(s[1], modkey, pre, ren, alias_mod, fbound, xbind, cur, lren),
                [rewrite_stmt(x, modkey, pre, ren, alias_mod, fbound, xbind, cur, lren) for x in s[2]])
    if k in ("break", "continue"):
        return s
    if k == "try":
        return ("try", [rewrite_stmt(x, modkey, pre, ren, alias_mod, fbound, xbind, cur, lren) for x in s[1]],
                [(n, [rewrite_stmt(x, modkey, pre, ren, alias_mod, fbound, xbind, cur, lren) for x in hb])
                 for n, hb in s[2]])
    if k == "raise":
        return ("raise", rewrite_expr(s[1], modkey, pre, ren, alias_mod, fbound, xbind, cur, lren))
    if k == "assert":
        return ("assert", rewrite_expr(s[1], modkey, pre, ren, alias_mod, fbound, xbind, cur, lren),
                rewrite_expr(s[2], modkey, pre, ren, alias_mod, fbound, xbind, cur, lren) if s[2] is not None else None)
    return s


def line_table(c):
    """phần "LN01" sau code: u16 nfile + (u16 len + tên)* + u32 n + (u32 pc, u32 dòng, u16 file)*"""
    out = bytearray(b"LN01") + struct.pack("<H", len(c.lfiles))
    for fn in c.lfiles:
        b = fn.encode("utf-8")
        out += struct.pack("<H", len(b)) + b
    out += struct.pack("<I", len(c.lines))
    for pc, ln, fi in c.lines:
        out += struct.pack("<IIH", pc, ln, fi)
    return bytes(out)


def export_table(c):
    """phần "EX01" sau LN01: u16 n + (u16 len + tên + u16 slot + u8 mut)* — global ở
    scope gốc của main (bỏ tên ẩn chứa "$" và kiểu struct "X__type"); load() của VM
    dùng. mut = 0 với let/fn: `m.X = v` báo "X immutable (use mut)"."""
    sc = c.main.scopes[0]
    ex = [(k, v) for k, v in sc.m.items()
          if "$" not in k and not k.endswith("__type")]
    out = bytearray(b"EX01") + struct.pack("<H", len(ex))
    for k, v in ex:
        b = k.encode("utf-8")
        im = k in sc.imm or k in c.fn_names
        out += struct.pack("<H", len(b)) + b + struct.pack("<HB", v, 0 if im else 1)
    return bytes(out)


def has_import(prog):
    return any(st[0] in ("import", "fromimport") for st in prog[1])


def build(src_path, out_path, dobundle):
    prog = parse_file(src_path)
    if dobundle or has_import(prog):
        # có import -> tự gộp module lúc build (không cần -b)
        prog = bundle(src_path)
    c = compile_prog(prog)
    ng = NFIXED + len(c.gnames)
    if ng > 65535:
        fail("quá nhiều global")
    with open(out_path, "wb") as f:
        f.write(b"TB01" + struct.pack("<H", ng) + struct.pack("<I", len(c.b)) + bytes(c.b))
        f.write(line_table(c))
        f.write(export_table(c))
    print(f"[tcomp] {src_path} -> {out_path} ({len(c.b)} bytes, globals={len(c.gnames)})")


def main():
    a = sys.argv[1:]
    dobundle = False
    if a and a[0] == "-b":
        dobundle = True
        a = a[1:]
    if len(a) != 2:
        print("dùng: python vm/tcomp.py [-b] prog.t prog.tb")
        raise SystemExit(2)
    try:
        build(a[0], a[1], dobundle)
    except CErr as ex:
        print(str(ex), file=sys.stderr)
        raise SystemExit(1)
    except SyntaxError as ex:
        print(f"tcomp: {ex}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()



