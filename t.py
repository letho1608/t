#!/usr/bin/env python3
"""T language interpreter v0.1 — tree-walking + inline cache + Tensor primitive."""
import sys
# Tree-walking lồng nhau (nhất là eval.t chạy dưới eval) tốn frame Python
# (mỗi call T ~ chục frame host). Nâng giới hạn để self-host nest 3 vẫn chạy.
try: sys.setrecursionlimit(10000)
except Exception: pass

# ---------- Lexer ----------
KEYWORDS = {"let","mut","fn","return","if","else","elif","for","in","while",
            "break","continue","true","false","nil","and","or","not","print",
            "tensor","range","len","str","int","float","type","dict","keys",
            "open","read","write","close","ord","chr","argv","assert","mem",
            "f64bits",
            "abs","round","min","max","sum","any","all",
            "sort","reversed","enumerate","zip","map","filter",
            "import","from","as","try","except","raise",
            "switch","case","default","struct"}

class Tok:
    __slots__=("type","val","line","bol")
    def __init__(self,t,v,ln): self.type=t; self.val=v; self.line=ln; self.bol=False
    def __repr__(self): return f"Tok({self.type},{self.val!r})"

def lex_fstr(src, i, n, line):
    """Quét f-string từ dấu quote tại i. Trả (parts, next_i).
    parts: list ("s", literal) / ("e", expr_src). Hỗ trợ {{ }},
    escape như string thường, expr lồng với mọi bracket + string trong."""
    q = src[i]; j = i + 1
    parts = []
    lit = ""
    while j < n:
        c = src[j]
        if c == q:
            parts.append(("s", lit)); lit = ""
            return parts, j + 1
        if c == "\\" and j + 1 < n:
            e2 = src[j + 1]
            if e2 == "n": lit += "\n"
            elif e2 == "t": lit += "\t"
            elif e2 == "r": lit += "\r"
            elif e2 == "\\": lit += "\\"
            elif e2 == q: lit += q
            elif e2 == "{": lit += "{"
            elif e2 == "}": lit += "}"
            else: lit += e2
            j += 2
            continue
        if c == "{" and j + 1 < n and src[j + 1] == "{":
            lit += "{"; j += 2; continue
        if c == "}" and j + 1 < n and src[j + 1] == "}":
            lit += "}"; j += 2; continue
        if c == "{":
            parts.append(("s", lit)); lit = ""
            depth = 1; k = j + 1
            qq = None
            while k < n and depth > 0:
                d = src[k]
                if qq:
                    if d == "\\" and k + 1 < n:
                        k += 2; continue
                    if d == qq:
                        qq = None
                    k += 1; continue
                if d in ("'", '"'):
                    qq = d; k += 1; continue
                if d in "([{":
                    depth += 1
                elif d in ")]}":
                    depth -= 1
                    if depth == 0:
                        break
                k += 1
            if depth != 0:
                raise SyntaxError(f"line {line}: unclosed '{{' trong f-string")
            parts.append(("e", src[j + 1:k]))
            j = k + 1
            continue
        if c == "}":
            raise SyntaxError(f"line {line}: '}}' lẻ trong f-string")
        if c == "\n":
            raise SyntaxError(f"line {line}: f-string không xuống dòng")
        lit += c; j += 1
    raise SyntaxError(f"line {line}: unterminated f-string")


def lex(src,line=1):
    toks=[]; i=0; n=len(src)
    while i<n:
        c=src[i]
        if c=="\n": line+=1; i+=1; continue
        if c in " \t\r": i+=1; continue
        if c=="#":
            while i<n and src[i]!="\n": i+=1
            continue
        if c=='"' or c=="'":
            q=c; j=i+1; s=""
            while j<n and src[j]!=q:
                if src[j]=="\\" and j+1<n:
                    e2=src[j+1]
                    if e2=="n": s+="\n"
                    elif e2=="t": s+="\t"
                    elif e2=="r": s+="\r"
                    elif e2=="\\": s+="\\"
                    elif e2==q: s+=q
                    else: s+=e2
                    j+=2
                else: s+=src[j]; j+=1
            if j>=n: raise SyntaxError(f"line {line}: unterminated string")
            toks.append(Tok("STR",s,line)); i=j+1; continue
        if c.isdigit() or (c=="." and i+1<n and src[i+1].isdigit()):
            j=i; dot=False
            if c=="0" and i+1<n and src[i+1] in "xX":
                j=i+2
                while j<n and src[j] in "0123456789abcdefABCDEF_": j+=1
                t=src[i+2:j].replace("_","")
                if not t: raise SyntaxError(f"line {line}: hex thiếu chữ số")
                toks.append(Tok("NUM",int(t,16),line)); i=j; continue
            if c=="0" and i+1<n and src[i+1] in "bB":
                j=i+2
                while j<n and src[j] in "01_": j+=1
                t=src[i+2:j].replace("_","")
                if not t: raise SyntaxError(f"line {line}: binary thiếu chữ số")
                toks.append(Tok("NUM",int(t,2),line)); i=j; continue
            if c=="0" and i+1<n and src[i+1] in "oO":
                j=i+2
                while j<n and src[j] in "01234567_": j+=1
                t=src[i+2:j].replace("_","")
                if not t: raise SyntaxError(f"line {line}: octal thiếu chữ số")
                toks.append(Tok("NUM",int(t,8),line)); i=j; continue
            while j<n and (src[j].isdigit() or src[j]=="_"):
                j+=1
            if j<n and src[j]=="." and not dot:
                dot=True; j+=1
                while j<n and (src[j].isdigit() or src[j]=="_"): j+=1
            # số mũ: 1e5, 2.5E-3
            if j<n and src[j] in "eE":
                k=j+1
                if k<n and src[k] in "+-": k+=1
                if k<n and src[k].isdigit():
                    while k<n and src[k].isdigit(): k+=1
                    dot=True; j=k
            t=src[i:j].replace("_","")
            toks.append(Tok("NUM",float(t) if dot else int(t),line)); i=j; continue
        if c.isalpha() or c=="_":
            j=i
            while j<n and (src[j].isalnum() or src[j]=="_"): j+=1
            w=src[i:j]
            if w in ("f", "F") and j<n and src[j] in ("'", '"'):
                parts, j = lex_fstr(src, j, n, line)
                toks.append(Tok("FSTR",parts,line)); i=j; continue
            toks.append(Tok("KW" if w in KEYWORDS else "NAME",w,line)); i=j; continue
        two=src[i:i+2]
        if two in ("==","!=","<=",">=","->","**","+=","-=","*=","/=","%="):
            toks.append(Tok("OP",two,line)); i+=2; continue
        if c in "+-*/%@<>=():,[]{}.":
            toks.append(Tok("OP",c,line)); i+=1; continue
        raise SyntaxError(f"line {line}: bad char {c!r}")
    toks.append(Tok("EOF","",line))
    return toks

# ---------- Parser ----------
def file_base(p):
    """tên file (không thư mục) — mốc dòng dùng cho thông báo lỗi runtime"""
    return str(p).replace("\\","/").split("/")[-1]

class P:
    def __init__(self,toks,lpath=None):
        # lpath != None: chèn mốc ("line", dòng, file) trước mỗi câu lệnh
        # (compiler -> bảng dòng trong .tb; t.py -> vị trí lỗi runtime)
        self.t=toks; self.i=0; self.lpath=lpath
    def _mark(self,ss):
        if self.lpath is not None:
            ss.append(("line",self.pk().line,self.lpath))
    def pk(self): return self.t[self.i]
    def eat(self,ty=None,val=None):
        t=self.pk()
        if ty and t.type!=ty: raise SyntaxError(f"line {t.line}: want {ty} got {t}")
        if val is not None and t.val!=val: raise SyntaxError(f"line {t.line}: want {val!r} got {t.val!r}")
        self.i+=1; return t
    def at(self,val): return self.pk().type=="OP" and self.pk().val==val
    def atkw(self,w): return self.pk().type=="KW" and self.pk().val==w
    def _next_is(self,ty,val):
        u=self.t[self.i+1] if self.i+1<len(self.t) else None
        return u is not None and u.type==ty and u.val==val
    def _comma_ahead(self):
        # có ',' ở tầng ngoài cùng trước ']' khớp không?
        d=0; j=self.i
        while j<len(self.t):
            t=self.t[j]
            if t.type=="OP" and t.val in ("(","[","{"): d+=1
            elif t.type=="OP" and t.val in (")","]","}"):
                if d==0: return False
                d-=1
            elif t.type=="OP" and t.val=="," and d==0: return True
            elif t.type=="EOF": return False
            j+=1
        return False
    def _slice_ahead(self):
        # lookahead: is there a ':' before the matching ']'? (depth-aware)
        d=0; j=self.i+1
        while j<len(self.t):
            t=self.t[j]
            if t.type=="OP" and t.val in ("(","[","{"): d+=1
            elif t.type=="OP" and t.val in (")","]","}"):
                if d==0: return t.val==":"  # unreachable; guarded by caller
                d-=1
                if t.val=="]" and d<0: return False
            elif t.type=="OP" and t.val==":" and d==0: return True
            elif t.type=="OP" and t.val=="]" and d==0: return False
            j+=1
        return False
    def _print_stmt(self):
        self.i+=1; self.eat("OP","(")
        a=[]
        if not self.at(")"):
            a.append(self.expr())
            while self.at(","): self.i+=1; a.append(self.expr())
        self.eat("OP",")"); return ("print",a)
    def parse(self):
        stmts=[]
        while self.pk().type!="EOF":
            self._mark(stmts); add_stmt(stmts,self.stmt())
        return ("prog",stmts)
    def _names(self):
        ns=[self.eat("NAME").val]
        while self.at(","): self.i+=1; ns.append(self.eat("NAME").val)
        return ns
    def _switch(self):
        # switch x: case a, b: ... default: ...  -> let $sN = x ; if/elif chain
        self.i+=1; subj=self.expr(); self.eat("OP",":"); self.eat("IND","{")
        cases=[]; dflt=None
        while not (self.pk().type=="DED"):
            if self.atkw("case"):
                self.i+=1; vs=[self.expr()]
                while self.at(","): self.i+=1; vs.append(self.expr())
                self.eat("OP",":"); cases.append((vs,self.suite()))
            elif self.atkw("default"):
                self.i+=1; self.eat("OP",":"); dflt=self.suite()
            else:
                raise SyntaxError(f"line {self.pk().line}: want case/default got {self.pk()}")
        self.eat("DED","}")
        tmp=gensym("s")
        b=dflt
        for vs,body in reversed(cases):
            c=None
            for v in vs:
                eq=("bin","==",("var",tmp),v)
                c=eq if c is None else ("bin","or",c,eq)
            b=[("if",c,body,b)]
        out=[("decl","let",tmp,subj)]
        if b: out.append(b[0] if cases else ("if",("lit",True),b,None))
        return ("multi",out)
    def _struct(self):
        # struct P: x, y / fn m(self): ...  ->
        #   fn P__m(self, ...) ; let P__type = __struct("P", [fields], {methods})
        #   fn P(x, y): return __new(P__type, x, y)
        self.i+=1; name=self.eat("NAME").val; self.eat("OP",":"); self.eat("IND","{")
        fields=[]; meths=[]
        while not (self.pk().type=="DED"):
            if self.atkw("fn"):
                m=self.stmt(); meths.append(m)
            elif self.pk().type=="NAME":
                for f in self._names():
                    if f in fields: raise SyntaxError(f"line {self.pk().line}: trùng field {f}")
                    fields.append(f)
            else:
                raise SyntaxError(f"line {self.pk().line}: want field/fn in struct got {self.pk()}")
        self.eat("DED","}")
        out=[]
        for m in meths:
            out.append(("fn",name+"__"+m[1],m[2],m[3]))
        tp=name+"__type"
        out.append(("decl","let",tp,("call",("var","__struct"),[("str",name),
              ("list",[("str",f) for f in fields]),
              ("dict",[("str",m[1]) for m in meths],[("var",name+"__"+m[1]) for m in meths])])))
        out.append(("fn",name,list(fields),
                    [("ret",("call",("var","__new"),[("var",tp)]+[("var",f) for f in fields]))]))
        return ("multi",out)
    def stmt(self):
        # lambda / comprehension trong câu lệnh được "nâng" thành định nghĩa hàm
        # ẩn đặt ngay trước câu lệnh (hàm không side effect, bắt biến theo tham chiếu)
        _PEND.append([])
        try: st=self._stmt()
        finally: pend=_PEND.pop()
        if not pend: return st
        return ("multi",pend+(st[1] if st[0]=="multi" else [st]))
    def _lambda(self):
        # fn(a, b): biểu_thức  ->  fn $lN(a, b): return biểu_thức
        self.i+=1; self.eat("OP","(")
        args=[]
        if not self.at(")"):
            args.append(self.eat("NAME").val)
            while self.at(","): self.i+=1; args.append(self.eat("NAME").val)
        self.eat("OP",")"); self.eat("OP",":")
        _PEND.append([])
        try: body=self.expr()
        finally: inner=_PEND.pop()
        name=gensym("l")
        _hoist(("fn",name,args,inner+[("ret",body)]),self.pk().line)
        return ("var",name)
    def _comp(self,first,key,inner,close):
        """[e for ... if ...] / {k: v for ...}: hàm ẩn $cN() dựng kết quả, gọi tại chỗ"""
        # mỗi mệnh đề giữ định nghĩa ẩn (lambda...) của biểu thức của nó, đặt
        # ngay trước mệnh đề trong hàm $cN (thấy được biến lặp của tầng ngoài)
        clauses=[]
        while True:
            if self.atkw("for"):
                self.i+=1; vs=self._names(); self.eat("KW","in")
                _PEND.append([])
                try: it=self.orx()
                finally: defs=_PEND.pop()
                clauses.append(("for",vs,it,defs))
            elif self.atkw("if"):
                self.i+=1
                _PEND.append([])
                try: c=self.orx()
                finally: defs=_PEND.pop()
                clauses.append(("if",c,None,defs))
            else: break
        self.eat("OP",close)
        r=gensym("r")
        if key is None:
            body=inner+[("expr",("call",("attr",("var",r),"append"),[first]))]
            init=("list",[])
        else:
            body=inner+[("assign",("idx",("var",r),key),first)]
            init=("dict",[],[])
        for c in reversed(clauses):
            if c[0]=="if": body=c[3]+[("if",c[1],body,None)]
            else: body=c[3]+[make_for(c[1],c[2],body)]
        name=gensym("c")
        fbody=[("decl","mut",r,init)]+body+[("ret",("var",r))]
        _hoist(("fn",name,[],fbody),self.pk().line)
        return ("call",("var",name),[])
    def _stmt(self):
        t = self.pk()
        if t.type == "KW" and t.val == "print" and self._next_is("OP", "("):
            return self._print_stmt()
        if t.type=="KW":
            if t.val=="switch": return self._switch()
            if t.val=="struct": return self._struct()
            if t.val in ("let","mut"):
                self.i+=1; name=self.eat("NAME").val; self.eat("OP","=")
                return ("decl",t.val,name,self.expr())
            if t.val=="fn":
                self.i+=1; name=self.eat("NAME").val; self.eat("OP","(")
                args=[]
                if not self.at(")"):
                    args.append(self.eat("NAME").val)
                    while self.at(","): self.i+=1; args.append(self.eat("NAME").val)
                self.eat("OP",")"); self.eat("OP",":")
                body=self.suite(); return ("fn",name,args,body)
            if t.val=="return":
                self.i+=1; return ("ret",self.expr())
            if t.val=="if":
                self.i+=1; cond=self.expr(); self.eat("OP",":"); a=self.suite()
                # elif chain: if/elif*/else? -> nested ("if",...) in else slot
                elifs=[]
                while self.pk().type=="KW" and self.pk().val=="elif":
                    self.i+=1; c2=self.expr(); self.eat("OP",":"); b2=self.suite()
                    elifs.append((c2,b2))
                b=None
                if self.pk().type=="KW" and self.pk().val=="else":
                    self.i+=1; self.eat("OP",":"); b=self.suite()
                for (c2,b2) in reversed(elifs):
                    b=[("if",c2,b2,b)]
                return ("if",cond,a,b)
            if t.val=="for":
                self.i+=1; vs=self._names(); self.eat("KW","in")
                it=self.expr(); self.eat("OP",":"); body=self.suite()
                return make_for(vs,it,body)
            if t.val=="while":
                self.i+=1; c=self.expr(); self.eat("OP",":"); return ("while",c,self.suite())
            if t.val=="break": self.i+=1; return ("break",)
            if t.val=="continue": self.i+=1; return ("continue",)
            if t.val=="import":
                self.i+=1
                parts=[self.eat("NAME").val]
                while self.at("."): self.i+=1; parts.append(self.eat("NAME").val)
                alias=None
                if self.atkw("as"): self.i+=1; alias=self.eat("NAME").val
                return ("import",".".join(parts),alias)
            if t.val=="from":
                self.i+=1
                parts=[self.eat("NAME").val]
                while self.at("."): self.i+=1; parts.append(self.eat("NAME").val)
                self.eat("KW","import")
                names=[self.eat("NAME").val]
                while self.at(","): self.i+=1; names.append(self.eat("NAME").val)
                return ("fromimport",".".join(parts),names)
            if t.val=="try":
                self.i+=1; self.eat("OP",":"); body=self.suite()
                handlers=[]
                while self.pk().type=="KW" and self.pk().val=="except":
                    self.i+=1; name=None
                    if self.pk().type=="NAME": name=self.eat("NAME").val
                    elif self.pk().type=="STR": name=self.eat("STR").val
                    self.eat("OP",":"); handlers.append((name,self.suite()))
                return ("try",body,handlers)
            if t.val=="raise":
                self.i+=1; return ("raise",self.expr())
            if t.val=="assert":
                self.i+=1; c=self.expr(); m=None
                if self.at(","): self.i+=1; m=self.expr()
                return ("assert",c,m)
        # expr stmt or assign (print-as-variable falls through to expr)
        e=self.expr()
        p=self.pk()
        if p.type=="OP" and p.val in ("=","+=","-=","*=","/=","%="):
            if e[0] not in ("var","idx","attr"):
                raise SyntaxError(f"line {p.line}: bad assign target")
            self.i+=1; v=self.expr()
            if p.val!="=": v=("bin",p.val[0],e,v)
            return ("assign",e,v)
        return ("expr",e)
    def suite(self):
        # next token must be INDENT marker
        self.eat("IND","{")
        ss=[]
        while not (self.pk().type=="DED" and self.pk().val=="}"):
            self._mark(ss); add_stmt(ss,self.stmt())
        self.eat("DED","}")
        return ss
    def expr(self):
        # a if c else b (kết hợp phải, độ ưu tiên thấp nhất như Python)
        e=self.orx()
        if self.atkw("if") and not self.pk().bol:
            self.i+=1; c=self.orx(); self.eat("KW","else")
            e=("cond",c,e,self.expr())
        return e
    def orx(self):
        e=self.andx()
        while self.atkw("or"): self.i+=1; e=("bin","or",e,self.andx())
        return e
    def andx(self):
        e=self.notx()
        while self.atkw("and"): self.i+=1; e=("bin","and",e,self.notx())
        return e
    def notx(self):
        if self.atkw("not"): self.i+=1; return ("un","not",self.notx())
        return self.cmp()
    def cmp(self):
        e=self.add()
        while True:
            if self.pk().type=="OP" and self.pk().val in ("==","!=","<",">","<=",">="):
                op=self.pk().val; self.i+=1; e=("bin",op,e,self.add())
            elif self.atkw("in"):
                self.i+=1; e=("bin","in",e,self.add())
            elif self.atkw("not") and self._next_is("KW","in"):
                self.i+=2; e=("un","not",("bin","in",e,self.add()))
            else: break
        return e
    def add(self):
        e=self.mul()
        while self.pk().type=="OP" and self.pk().val in ("+","-"):
            op=self.pk().val; self.i+=1; e=("bin",op,e,self.mul())
        return e
    def mul(self):
        e=self.un()
        # @ cùng mức với * / % (như Python): X @ W + b == (X @ W) + b
        while self.pk().type=="OP" and self.pk().val in ("*","/","%","@"):
            op=self.pk().val; self.i+=1; e=("bin",op,e,self.un())
        return e
    def un(self):
        if self.pk().type=="OP" and self.pk().val=="-":
            self.i+=1; return ("un","-",self.un())
        return self.pow()
    def pow(self):
        # right-associative: a ** b ** c = a ** (b ** c)
        e=self.post()
        if self.pk().type=="OP" and self.pk().val=="**":
            self.i+=1; e=("bin","**",e,self.pow())
        return e
    def post(self):
        e=self.atom()
        while True:
            if self.pk().type=="OP" and self.pk().val=="(":
                self.i+=1; a=[]
                if not self.at(")"):
                    a.append(self.expr())
                    while self.at(","): self.i+=1; a.append(self.expr())
                self.eat("OP",")"); e=("call",e,a)
            elif self.pk().type=="OP" and self.pk().val=="[":
                self.i+=1
                if self._comma_ahead():
                    # t[i, j] / t[:, j]: chỉ số nhiều trục -> t[[i, j]] (":" riêng = nil)
                    items=[]
                    while True:
                        if self.at(":") and (self._next_is("OP",",") or self._next_is("OP","]")):
                            self.i+=1; items.append(("lit",None))
                        else: items.append(self.expr())
                        if self.at(","): self.i+=1; continue
                        break
                    self.eat("OP","]"); e=("idx",e,("list",items))
                # slice a[lo:hi] / a[lo:hi:step] — empty side = open end
                elif self.at(":") or self._slice_ahead():
                    lo=None; hi=None; st=None
                    if not self.at(":"): lo=self.expr()
                    self.eat("OP",":")
                    if not self.at(":") and not self.at("]"): hi=self.expr()
                    if self.at(":"):
                        self.i+=1
                        if not self.at("]"): st=self.expr()
                    self.eat("OP","]"); e=("slice",e,lo,hi,st)
                else:
                    ix=self.expr(); self.eat("OP","]"); e=("idx",e,ix)
            elif self.pk().type=="OP" and self.pk().val==".":
                self.i+=1; f=self.pk()
                if f.type not in ("NAME","KW"): raise SyntaxError(f"line {f.line}: want field")
                self.i+=1; e=("attr",e,f.val)
            else: break
        return e
    def atom(self):
        t=self.pk()
        if t.type=="NUM": self.i+=1; return ("num",t.val)
        if t.type=="STR": self.i+=1; return ("str",t.val)
        if t.type=="FSTR":
            self.i+=1; parts=[]
            for kind, val in t.val:
                if kind == "s":
                    parts.append(("s",val))
                else:
                    parts.append(("e",parse_fstr_expr(val)))
            return ("fstr",parts)
        if t.type=="KW" and t.val in ("true","false","nil"):
            self.i+=1; return ("lit",{"true":True,"false":False,"nil":None}[t.val])
        if t.type=="KW" and t.val=="tensor":
            self.i+=1; self.eat("OP","("); e=self.expr(); self.eat("OP",")")
            return ("tensor",e)
        if t.type=="KW" and t.val in ("range","len","str","int","float","type",
                                     "open","read","write","close",
                                     "ord","chr","argv","mem","f64bits",
                                     "abs","round","min","max","sum","any","all",
                                     "sort","reversed","enumerate","zip","map",
                                     "filter"):
            self.i+=1; self.eat("OP","("); a=[]
            if not self.at(")"):
                a.append(self.expr())
                while self.at(","): self.i+=1; a.append(self.expr())
            self.eat("OP",")"); return ("bltin",t.val,a)
        if t.type=="NAME":
            self.i+=1; return ("var",t.val)
        if t.type=="KW" and t.val=="print":
            # bare `print` used as value (no parens): treat as builtin fn value
            self.i+=1; return ("bltfn","print")
        if t.type=="OP" and t.val=="(":
            self.i+=1; e=self.expr(); self.eat("OP",")"); return e
        if t.type=="KW" and t.val=="fn" and self._next_is("OP","("):
            return self._lambda()
        if t.type=="OP" and t.val=="[":
            self.i+=1; a=[]
            _PEND.append([])     # lambda trong phần tử có thể dùng biến của comprehension
            try:
                if not self.at("]"):
                    a.append(self.expr())
                    if self.atkw("for"):
                        inner=_PEND.pop()
                        try: return self._comp(a[0],None,inner,"]")
                        finally: _PEND.append([])
                    while self.at(","): self.i+=1; a.append(self.expr())
                self.eat("OP","]")
            finally: _merge_pending()
            return ("list",a)
        if t.type=="OP" and t.val=="{":
            self.i+=1; ks=[]; vs=[]
            _PEND.append([])
            try:
                if not self.at("}"):
                    ks.append(self.expr()); self.eat("OP",":"); vs.append(self.expr())
                    if self.atkw("for"):
                        inner=_PEND.pop()
                        try: return self._comp(vs[0],ks[0],inner,"}")
                        finally: _PEND.append([])
                    while self.at(","):
                        self.i+=1
                        if self.at("}"): break
                        ks.append(self.expr()); self.eat("OP",":"); vs.append(self.expr())
                self.eat("OP","}")
            finally: _merge_pending()
            return ("dict",ks,vs)
        raise SyntaxError(f"line {t.line}: unexpected {t}")

def strip_comment(ln):
    """Strip trailing # comment, but ignore # inside string literals."""
    q=None; i=0; n=len(ln)
    while i<n:
        c=ln[i]
        if q:
            if c=="\\" and i+1<n: i+=2; continue
            if c==q: q=None
            i+=1; continue
        if c in ("'",'"'): q=c; i+=1; continue
        if c=="#": return ln[:i]
        i+=1
    return ln

def bracket_delta(code):
    """Net open-close brackets in a code line, ignoring strings/comments."""
    q=None; d=0; i=0; n=len(code)
    while i<n:
        c=code[i]
        if q:
            if c=="\\" and i+1<n: i+=2; continue
            if c==q: q=None
            i+=1; continue
        if c in ("'",'"'): q=c; i+=1; continue
        if c=="#": break
        if c in "([{" : d+=1
        elif c in ")]}": d-=1
        i+=1
    return d

def indent_lines(src):
    """Convert indentation to IND / DED markers. Tabs expand to 4 spaces.
    Blank lines and comments do not affect indent stack; trailing comments
    after code are stripped so `x = 1  # note` parses cleanly.
    (String-aware: # inside quotes is kept.)
    Bracket-aware: lines inside (... [...] {...}) are continuations and
    never trigger IND/DED (fixes multi-line list/dict/call literals).
    Trả list (số dòng gốc, text); marker mang số dòng của dòng code kế."""
    out=[]; stack=[0]; depth=0; no=0
    for raw in src.split("\n"):
        no+=1
        ln=raw.expandtabs(4)
        code=strip_comment(ln)
        if not code.strip(): continue
        if depth>0:
            # continuation: join to previous logical line
            out.append((no,code.strip(),True))
            depth+=bracket_delta(code)
            if depth<0: depth=0
            continue
        ind=len(ln)-len(ln.lstrip(" "))
        if ind>stack[-1]: stack.append(ind); out.append((no,"<IND>",False))
        while ind<stack[-1]: stack.pop(); out.append((no,"<DED>",False))
        out.append((no,code.strip(),False))
        depth+=bracket_delta(code)
        if depth<0: depth=0
    while len(stack)>1: stack.pop(); out.append((no,"<DED>",False))
    return out

def indent_scan(src):
    return "\n".join(t for _,t,_ in indent_lines(src))

def lex2(src):
    # Line-based: lines exactly <IND>/<DED> become markers, other lines are
    # lexed as code. This keeps dict/list braces `{` `}` intact (old string
    # replace turned every `}` into DED, breaking dict literals).
    # Mỗi token mang số dòng GỐC trong file (báo lỗi đúng chỗ); token đầu
    # mỗi dòng logic có bol=True (phân biệt `if` câu lệnh với `a if c else b`).
    res=[]; line=1
    for line,raw,cont in indent_lines(src):
        if raw=="<IND>": res.append(Tok("IND","{",line)); continue
        if raw=="<DED>": res.append(Tok("DED","}",line)); continue
        first=not cont
        for t in lex(raw,line):
            if t.type=="EOF": continue
            if first: t.bol=True; first=False
            res.append(t)
    res.append(Tok("EOF","",line))
    return res

_PEND=[]   # stack: định nghĩa hàm ẩn (lambda/comprehension) chờ đặt trước câu lệnh

def _hoist(fdef,line):
    if not _PEND: raise SyntaxError(f"line {line}: lambda/comprehension ngoài câu lệnh")
    _PEND[-1].append(fdef)

def _merge_pending():
    top=_PEND.pop()
    if top:
        if not _PEND: raise SyntaxError("lambda/comprehension ngoài câu lệnh")
        _PEND[-1].extend(top)

def make_for(vs,it,body):
    """for a in xs / for a, b in xs (-> for $pN in xs: mut a = $pN[0] ...)"""
    if len(vs)==1: return ("for",vs[0],it,body)
    tmp=gensym("p")
    pre=[("decl","mut",n,("idx",("var",tmp),("num",i))) for i,n in enumerate(vs)]
    return ("for",tmp,it,pre+body)

_GENSYM=[0]
def gensym(kind):
    """Tên ẩn ($ không lex được nên không đụng tên user)."""
    _GENSYM[0]+=1
    return f"${kind}{_GENSYM[0]}"

def add_stmt(ss,s):
    """Desugar trả ("multi",[...]) -> trải phẳng vào cùng block."""
    if s[0]=="multi": ss.extend(s[1])
    else: ss.append(s)

def parse_fstr_expr(src):
    """Parse biểu thức trong f-string bằng parser con (khớp P.expr)."""
    p = P(lex2(src))
    e = p.expr()
    if p.pk().type != "EOF":
        raise SyntaxError(f"thừa trong f-string: {p.pk()}")
    return e


# ---------- Values ----------
import os as _os

class TError(RuntimeError):
    pass

class Module:
    __slots__=("name","env")
    def __init__(self,n,e): self.name=n; self.env=e
    def __repr__(self): return f"module({self.name})"

class TFile:
    __slots__=("path","mode","h")
    def __init__(self,p,m):
        self.path=p; self.mode=m
        if "b" in m:
            self.h=open(p,m)
        else:
            # latin-1: 1 char = 1 byte, khớp byte-string của VM C
            self.h=open(p,m,encoding="latin-1",newline="")
    def __repr__(self): return f"file({self.path!r},{self.mode!r})"

LOADED={}  # abspath -> Module (circular-import guard)

# tensor n-chiều: hiện thực tham chiếu ở ttensor.py (VM C khớp từng bit)
try:
    import ttensor as TT
except ImportError:
    sys.path.insert(0,_os.path.dirname(_os.path.abspath(__file__)))
    import ttensor as TT
Tensor=TT.Tensor

def tens(fn,*a):
    """gọi hàm ttensor, đổi TensorError -> TError (try/except của T bắt được)"""
    try: return fn(*a)
    except TT.TensorError as ex: raise TError(str(ex))

class TSet:
    """Tập hợp: giữ thứ tự chèn (khớp TDict kind=set của VM C)."""
    __slots__=("d",)
    def __init__(self,xs=()):
        self.d={}
        for x in xs: self.d[dkey(x)]=True
    def __len__(self): return len(self.d)
    def __eq__(self,o): return isinstance(o,TSet) and set(self.d)==set(o.d)
    def __ne__(self,o): return not self.__eq__(o)
    __hash__=None

class TStruct:
    """Kiểu struct (tạo 1 lần lúc định nghĩa): tên + tên field + bảng method."""
    __slots__=("name","fields","m")
    def __init__(self,name,fields,m): self.name=name; self.fields=fields; self.m=m

class TRec:
    """Bản ghi struct: tên + field có thứ tự + bảng method."""
    __slots__=("name","f","m")
    def __init__(self,name,f,m): self.name=name; self.f=f; self.m=m
    def __eq__(self,o): return isinstance(o,TRec) and o.name==self.name and o.f==self.f
    def __ne__(self,o): return not self.__eq__(o)
    __hash__=None

class Fn:
    __slots__=("name","args","body","env","cache")
    def __init__(self,n,a,b,e): self.name=n; self.args=a; self.body=b; self.env=e; self.cache=None

class Ret(Exception):
    def __init__(self,v): self.v=v
class Brk(Exception): pass
class Cnt(Exception): pass

class Env:
    __slots__=("m","mut","up","isb")
    def __init__(self,up=None): self.m={}; self.mut=set(); self.up=up; self.isb=False
    def decl(self,n,v,mut):
        if n in self.m: raise RuntimeError(f"redeclare {n}")
        self.m[n]=v
        if mut: self.mut.add(n)
    def get(self,n):
        e=self
        while e:
            if n in e.m: return e.m[n]
            e=e.up
        raise RuntimeError(f"undef {n}")
    def has(self,n):
        # env builtin (isb) không tính: gán `set = 1` tạo biến mới che builtin
        e=self
        while e and not e.isb:
            if n in e.m: return True
            e=e.up
        return False
    def set(self,n,v):
        e=self
        while e and not e.isb:
            if n in e.m:
                if n not in e.mut: raise RuntimeError(f"{n} immutable (use mut)")
                e.m[n]=v; return
            e=e.up
        raise RuntimeError(f"undef {n}")

def truthy(v):
    if v is None or v is False: return False
    if v is True: return True
    if isinstance(v,(int,float)) and v==0: return False
    if isinstance(v,str) and v=="": return False
    if isinstance(v,Tensor): return len(v.d)!=0
    if isinstance(v,(list,dict,TSet)) and len(v)==0: return False
    return True

def fmt(v):
    if v is None: return "nil"
    if v is True: return "true"
    if v is False: return "false"
    if isinstance(v,float): return fmt_float(v)
    if isinstance(v,Tensor): return v.fmt(fmt_float)
    if isinstance(v,list): return "["+", ".join(fmt(x) for x in v)+"]"
    if isinstance(v,dict):
        return "{"+", ".join(fmt(k)+": "+fmt(x) for k,x in v.items())+"}"
    if isinstance(v,TSet):
        if not v.d: return "set()"
        return "{"+", ".join(fmt(k) for k in v.d)+"}"
    if isinstance(v,TRec):
        return v.name+"{"+", ".join(k+": "+fmt(x) for k,x in v.f.items())+"}"
    if isinstance(v,TStruct): return "<struct "+v.name+">"
    if isinstance(v,Fn) or callable(v): return "<fn>"
    return str(v)

def fmt_float(x):
    """Khớp VM C: số nguyên in không phần lẻ; còn lại repr ngắn nhất
    (VM thử %.15g/%.16g/%.17g, chọn cái đầu tiên round-trip)."""
    if x!=x: return "nan"
    if x in (float("inf"),float("-inf")): return "inf" if x>0 else "-inf"
    if x.is_integer() and abs(x)<9.2e18: return str(int(x))
    for p in (15,16,17):
        s="%.*g"%(p,x)
        if float(s)==x: return s
    return "%.17g"%x

def tprint(s, nl=True):
    """Ghi raw byte ra stdout (khớp fwrite của VM C; string là byte-string)."""
    try:
        b = s.encode("latin-1", "replace")
    except Exception:
        b = s.encode("latin-1", "replace")
    buf = getattr(sys.stdout, "buffer", None)
    if buf is None:
        sys.stdout.write(s + ("\n" if nl else ""))
        return
    buf.write(b + (b"\n" if nl else b""))
    buf.flush()

def t_match(hay, pat):
    """Match except: substring thường; có * ? thì glob (chỉ 2 ký tự đó đặc biệt)."""
    if "*" not in pat and "?" not in pat:
        return pat in hay
    si = pi = 0
    star = -1
    ss = 0
    n, m = len(hay), len(pat)
    while si < n:
        if pi < m and (pat[pi] == "?" or pat[pi] == hay[si]):
            si += 1; pi += 1
        elif pi < m and pat[pi] == "*":
            star = pi; pi += 1; ss = si
        elif star != -1:
            pi = star + 1; ss += 1; si = ss
        else:
            return False
    while pi < m and pat[pi] == "*":
        pi += 1
    return pi == m


def dkey(v):
    if isinstance(v,(int,float,str,bool)) or v is None: return v
    raise TError(f"unhashable dict key {fmt(v)[:30]}")

def tolist(v):
    if isinstance(v,Tensor): return v.d
    if isinstance(v,list): return v
    raise RuntimeError("not iterable container")

def t_in(a,b):
    """a in b: chuỗi con / phần tử list / key dict|set / số trong tensor."""
    if isinstance(b,str):
        if not isinstance(a,str): raise TError("in: str needs str")
        return a in b
    if isinstance(b,list): return any(_teq(a,x) for x in b)
    if isinstance(b,dict):
        return any(_teq(a,k) for k in b)
    if isinstance(b,TSet): return any(_teq(a,k) for k in b.d)
    if isinstance(b,Tensor):
        if isinstance(a,bool) or not isinstance(a,(int,float)): return False
        return any(a==x for x in b.d)
    raise TError(f"in: unsupported {tname(b)}")

def _teq(a,b):
    """== khớp tval_eq của VM: bool khác số; int==float theo giá trị."""
    if isinstance(a,bool)!=isinstance(b,bool): return False
    return a==b

def tname(v):
    if v is None: return "nil"
    if isinstance(v,bool): return "bool"
    if isinstance(v,int): return "int"
    if isinstance(v,float): return "float"
    if isinstance(v,str): return "str"
    if isinstance(v,list): return "list"
    if isinstance(v,dict): return "dict"
    if isinstance(v,TSet): return "set"
    if isinstance(v,TRec): return v.name
    if isinstance(v,TStruct): return "type"
    if isinstance(v,Tensor): return "tensor"
    if isinstance(v,(Fn,Module)) or callable(v): return "fn"
    if isinstance(v,TFile): return "file"
    return "obj"

def binop(op,a,b):
    if op=="in": return t_in(a,b)
    if isinstance(a,Tensor) or isinstance(b,Tensor):
        if op in ("+","-","*","/","**","@"):
            return tens(TT.binop,op,a,b)
        if op in ("==","!="):
            r=_teq(a,b)
            return r if op=="==" else not r
        if op in ("<",">","<=",">="): return tens(TT.binop,op,a,b)
        raise TError("bad tensor op")
    if op=="+":
        if isinstance(a,str) or isinstance(b,str): return fmt(a)+fmt(b)
        if isinstance(a,list) and isinstance(b,list): return a+b
        return a+b
    if op=="-": return a-b
    if op=="*":
        if isinstance(a,list) and isinstance(b,int): return a*b
        if isinstance(a,int) and isinstance(b,list): return b*a
        return a*b
    if op=="**":
        return a**b
    if op=="/":
        if b==0: raise TError("division by zero")
        return a/b
    if op=="%":
        if b==0: raise TError("modulo by zero")
        return a%b
    if op=="==": return a==b
    if op=="!=": return a!=b
    if op=="<": return a<b
    if op==">": return a>b
    if op=="<=": return a<=b
    if op==">=": return a>=b
    if op=="and": return b if truthy(a) else a
    if op=="or": return a if truthy(a) else b
    raise RuntimeError(f"bad op {op}")

HOT_N=50
hotcount={}

def ev(e,env):
    k=e[0]
    if k=="num" or k=="str": return e[1]
    if k=="fstr":
        out = []
        for kind, val in e[1]:
            out.append(val if kind == "s" else fmt(ev(val,env)))
        return "".join(out)
    if k=="lit": return e[1]
    if k=="var": return env.get(e[1])
    if k=="cond": return ev(e[2],env) if truthy(ev(e[1],env)) else ev(e[3],env)
    if k=="bltfn":
        if e[1]=="print": return lambda *a: tprint(" ".join(fmt(x) for x in a))
        raise RuntimeError("bltfn")
    if k=="list": return [ev(x,env) for x in e[1]]
    if k=="dict":
        ks=[ev(x,env) for x in e[1]]; vs=[ev(x,env) for x in e[2]]
        return dict(zip([dkey(x) for x in ks],vs))
    if k=="slice":
        c=ev(e[1],env)
        lo=ev(e[2],env) if e[2] is not None else None
        hi=ev(e[3],env) if e[3] is not None else None
        st=ev(e[4],env) if e[4] is not None else None
        lo=int(lo) if lo is not None else None
        hi=int(hi) if hi is not None else None
        st=int(st) if st is not None else None
        if st==0: raise TError("slice step 0")
        if isinstance(c,Tensor): return tens(TT.slice0,c,lo,hi,st)
        if isinstance(c,(str,list)): return c[lo:hi:st]
        raise TError("slice on non-sequence")
    if k=="tensor":
        return tens(TT.from_value,ev(e[1],env))
    if k=="bltin":
        if e[1]=="len":
            v=ev(e[2][0],env)
            if isinstance(v,(TRec,TStruct)) or v is None or isinstance(v,(bool,int,float)) or callable(v) or isinstance(v,(Fn,Module)):
                raise TError("len() here")
            return len(v)
        if e[1]=="str": return fmt(ev(e[2][0],env))
        if e[1]=="int": return int(float(ev(e[2][0],env)))
        if e[1]=="float": return float(ev(e[2][0],env))
        if e[1]=="type": return tname(ev(e[2][0],env))
        if e[1]=="open":
            a=[ev(x,env) for x in e[2]]
            if len(a)<1 or len(a)>2: raise TError("open(path [, mode])")
            mode=fmt(a[1]) if len(a)==2 else "r"
            try: return TFile(fmt(a[0]),mode)
            except OSError as ex: raise TError(f"open: {ex}")
        if e[1]=="read":
            a=[ev(x,env) for x in e[2]]
            if len(a)!=1 or not isinstance(a[0],TFile): raise TError("read(f)")
            try: return a[0].h.read()
            except OSError as ex: raise TError(f"read: {ex}")
        if e[1]=="write":
            a=[ev(x,env) for x in e[2]]
            if len(a)!=2 or not isinstance(a[0],TFile): raise TError("write(f, s)")
            try:
                s = fmt(a[1])
                if "b" in a[0].mode:
                    return a[0].h.write(s.encode("latin-1"))
                return a[0].h.write(s)
            except OSError as ex: raise TError(f"write: {ex}")
            except (UnicodeEncodeError, ValueError): raise TError("write: byte 0-255")
        if e[1]=="close":
            a=[ev(x,env) for x in e[2]]
            if len(a)!=1 or not isinstance(a[0],TFile): raise TError("close(f)")
            try: a[0].h.close()
            except OSError: pass
            return None
        if e[1]=="ord":
            a=[ev(x,env) for x in e[2]]
            if len(a)!=1: raise TError("ord(s)")
            s=fmt(a[0])
            if len(s)!=1: raise TError("ord() 1 char")
            return ord(s)
        if e[1]=="chr":
            a=[ev(x,env) for x in e[2]]
            if len(a)!=1: raise TError("chr(n)")
            n=int(a[0])
            if n<0 or n>255: raise TError("chr() 0-255")
            return chr(n)
        if e[1]=="argv": return list(sys.argv[1:])
        if e[1]=="f64bits":
            import struct as _st
            a=[ev(x,env) for x in e[2]]
            if len(a)!=1 or isinstance(a[0],bool) or not isinstance(a[0],(int,float)):
                raise TError("f64bits(x number)")
            return _st.unpack("<q", _st.pack("<d", float(a[0])))[0]
        if e[1]=="mem":
            if e[2]:
                raise TError("mem() 0 arg")
            import gc as _gc
            return {"objs": len(_gc.get_objects()), "gc": 0}
        if e[1]=="range":
            a=[ev(x,env) for x in e[2]]
            a=[int(x) for x in a]
            return list(range(*a))
        if e[1]=="abs":
            a=[ev(x,env) for x in e[2]]
            if len(a)!=1: raise TError("abs() number")
            v=a[0]
            if isinstance(v,Tensor): return TT.unary(v,"abs")
            if isinstance(v,bool) or not isinstance(v,(int,float)): raise TError("abs() number")
            return -v if v<0 else v
        if e[1]=="round":
            a=[ev(x,env) for x in e[2]]
            if len(a)!=1: raise TError("round() number")
            v=a[0]
            if isinstance(v,bool) or not isinstance(v,(int,float)): raise TError("round() number")
            import math as _m
            f=float(v)
            r=_m.floor(abs(f)+0.5)*(-1.0 if f<0 else 1.0)
            return r if isinstance(v,float) else int(r)
        if e[1]=="min" or e[1]=="max":
            a=[ev(x,env) for x in e[2]]
            if len(a)==2:
                x,y=a
                if not _cmp_ok(x,y): raise TError(f"{e[1]}() number/str")
                return x if (x<=y if e[1]=="min" else x>=y) else y
            if len(a)!=1: raise TError(f"{e[1]}(list) or {e[1]}(a, b)")
            xs=_seq(a[0],e[1])
            if not xs: raise TError(f"{e[1]}() empty")
            best=xs[0]
            for x in xs[1:]:
                if not _cmp_ok(x,best): raise TError(f"{e[1]}() number/str")
                if (x<best if e[1]=="min" else x>best): best=x
            return best
        if e[1]=="sum":
            a=[ev(x,env) for x in e[2]]
            if not 1<=len(a)<=2: raise TError("sum(list[, init])")
            xs=_seq(a[0],"sum")
            isf=len(a)==2 and isinstance(a[1],float)
            acc=a[1] if len(a)==2 else 0
            for x in xs:
                if isinstance(x,bool) or not isinstance(x,(int,float)): raise TError("sum() number")
                if isinstance(x,float): isf=True
                acc=acc+x
            return float(acc) if isf else int(acc)
        if e[1]=="any" or e[1]=="all":
            a=[ev(x,env) for x in e[2]]
            if len(a)!=1: raise TError(f"{e[1]}(list)")
            xs=_seq(a[0],e[1])
            if e[1]=="any": return any(truthy(x) for x in xs)
            return all(truthy(x) for x in xs)
        if e[1]=="sort":
            a=[ev(x,env) for x in e[2]]
            if not 1<=len(a)<=2: raise TError("sort(list[, cmp])")
            if not isinstance(a[0],list): raise TError("sort() list")
            key=a[1] if len(a)==2 else None
            if key is not None and not (isinstance(key,Fn) or callable(key)): raise TError("sort() fn")
            if key is None:
                for x in a[0]:
                    if not _cmp_ok(x,x): raise TError("sort() number/str")
                a[0].sort(key=_msortkey)
            else:
                a[0].sort(key=lambda x: _msortkey(call_fn(key,[x])))
            return a[0]
        if e[1]=="reversed":
            a=[ev(x,env) for x in e[2]]
            if len(a)!=1: raise TError("reversed(list|str)")
            v=a[0]
            if isinstance(v,str): return v[::-1]
            if isinstance(v,list): return list(reversed(v))
            if isinstance(v,Tensor): return [v.d[i] for i in range(len(v.d)-1,-1,-1)]
            raise TError("reversed(list|str)")
        if e[1]=="enumerate":
            a=[ev(x,env) for x in e[2]]
            if not 1<=len(a)<=2: raise TError("enumerate(list[, start])")
            st=int(a[1]) if len(a)==2 else 0
            return [[st+i,v] for i,v in enumerate(_seq(a[0],"enumerate"))]
        if e[1]=="zip":
            a=[ev(x,env) for x in e[2]]
            if len(a)<2: raise TError("zip(list, list, ...)")
            seqs=[_seq(x,"zip") for x in a]
            return [list(t) for t in zip(*seqs)]
        if e[1]=="map":
            a=[ev(x,env) for x in e[2]]
            if len(a)!=2: raise TError("map(fn, list)")
            f=a[0]
            if not (isinstance(f,Fn) or callable(f)): raise TError("map() fn")
            return [call_fn(f,[v]) for v in _seq(a[1],"map")]
        if e[1]=="filter":
            a=[ev(x,env) for x in e[2]]
            if len(a)!=2: raise TError("filter(fn, list)")
            f=a[0]
            if not (isinstance(f,Fn) or callable(f)): raise TError("filter() fn")
            return [v for v in _seq(a[1],"filter") if truthy(call_fn(f,[v]))]
        raise RuntimeError("bltin")
    if k=="un":
        v=ev(e[2],env)
        if e[1]=="-":
            if isinstance(v,Tensor): return TT.neg(v)
            if isinstance(v,bool) or not isinstance(v,(int,float)): raise TError("neg here")
            return -v
        return not truthy(v)
    if k=="bin":
        # short-circuit and/or (fixes `j < n and src[j]` bounds guards)
        if e[1]=="and":
            a=ev(e[2],env)
            return a if not truthy(a) else ev(e[3],env)
        if e[1]=="or":
            a=ev(e[2],env)
            return a if truthy(a) else ev(e[3],env)
        return binop(e[1],ev(e[2],env),ev(e[3],env))
    if k=="idx":
        c=ev(e[1],env)
        if isinstance(c,dict):
            kk=dkey(ev(e[2],env))
            if kk not in c: raise TError(f"key {fmt(kk)[:60]} missing")
            return c[kk]
        if isinstance(c,TRec):
            kk=ev(e[2],env)
            if not isinstance(kk,str) or kk not in c.f: raise TError(f"{c.name} has no field {fmt(kk)[:40]}")
            return c.f[kk]
        if isinstance(c,(TSet,Fn,Module)) or c is None or isinstance(c,(bool,int,float)):
            raise TError("index here")
        i=ev(e[2],env)
        if isinstance(c,Tensor) and isinstance(i,list): return tens(TT.get_multi,c,i)
        if isinstance(i,bool) or not isinstance(i,int): raise TError("index int")
        if isinstance(c,Tensor): return tens(TT.getitem,c,i)
        n=len(c)
        if i<0 or i>=n: raise TError(f"index {i} out of range (len {n})")
        return c[i]
    if k=="attr":
        o=ev(e[1],env); f=e[2]
        if isinstance(o,Module):
            try: return o.env.get(f)
            except RuntimeError: raise TError(f"module {o.name} has no {f!r}")
        if isinstance(o,TRec):
            if f in o.f: return o.f[f]
            if f in o.m:
                m=o.m[f]
                return lambda *a: call_fn(m,[o]+list(a))
            raise TError(f"{o.name} has no .{f}")
        if isinstance(o,str):
            if f=="len": return len(o)
            # ASCII-only, khớp byte-wise của VM C
            if f=="upper": return lambda: "".join(chr(ord(c)-32) if "a"<=c<="z" else c for c in o)
            if f=="lower": return lambda: "".join(chr(ord(c)+32) if "A"<=c<="Z" else c for c in o)
            if f=="split": return lambda d=" ": list(o) if d=="" else o.split(fmt(d))
            if f=="join": return lambda p: o.join(fmt(x) for x in tolist(p) if True) if isinstance(p,(list,Tensor)) else o.join(fmt(x) for x in p)
            if f=="strip": return lambda: o.strip(" \t\n\r")
            if f=="find": return lambda s: o.find(_needstr(s,"find"))
            if f=="replace": return lambda a,b: o.replace(_needstr(a,"replace"),_needstr(b,"replace")) if a!="" else o
            if f=="startswith": return lambda s: o.startswith(_needstr(s,"startswith"))
            if f=="endswith": return lambda s: o.endswith(_needstr(s,"endswith"))
            raise TError(f"str has no .{f}")
        if isinstance(o,list):
            if f=="len": return len(o)
            if f=="append": return lambda x: (o.append(x), None)[1]
            if f=="pop": return lambda i=-1: _lpop(o,i)
            if f=="clear": return lambda: (o.clear(), None)[1]
            if f=="insert": return lambda i,x: (_linsert(o,i,x), None)[1]
            if f=="extend": return lambda xs: (o.extend(_seq(xs,"extend")), None)[1]
            if f=="index": return lambda x: _lindex(o,x)
            raise TError(f"list has no .{f}")
        if isinstance(o,dict):
            if f=="len": return len(o)
            if f=="keys": return lambda: list(o.keys())
            if f=="values": return lambda: list(o.values())
            if f=="items": return lambda: [[k,v] for k,v in o.items()]
            if f=="get": return lambda k,d=None: o.get(dkey(k),d)
            if f=="has": return lambda k: t_in(k,o)
            if f=="remove": return lambda k: _dremove(o,k)
            raise TError(f"dict has no .{f}")
        if isinstance(o,TSet):
            if f=="len": return len(o.d)
            if f=="add": return lambda x: (o.d.__setitem__(dkey(x),True), None)[1]
            if f=="has": return lambda x: t_in(x,o)
            if f=="remove": return lambda x: _dremove(o.d,x)
            if f=="list": return lambda: list(o.d.keys())
            if f=="union": return lambda s: TSet(list(o.d)+list(_setarg(s,"union").d))
            if f=="inter":
                return lambda s: TSet([x for x in o.d if t_in(x,_setarg(s,"inter"))])
            if f=="diff":
                return lambda s: TSet([x for x in o.d if not t_in(x,_setarg(s,"diff"))])
            raise TError(f"set has no .{f}")
        if isinstance(o,Tensor): return tensor_attr(o,f)
        raise TError(f".{f} on {fmt(o)[:30]}")
    if k=="call":
        f=ev(e[1],env); a=[ev(x,env) for x in e[2]]
        if isinstance(f,Fn):
            key=(id(f),tuple(type(x).__name__ for x in a))
            hotcount[key]=hotcount.get(key,0)+1
            if f.cache and f.cache[0]==key:
                fast=f.cache[1]
                if fast=="numloop" and all(isinstance(x,(int,float)) for x in a):
                    return fastnum(f,a)
            if len(a)!=len(f.args): raise RuntimeError(f"{f.name} arity")
            le=Env(f.env)
            for n,v in zip(f.args,a): le.decl(n,v,True)
            here=CUR_LINE[0]   # lỗi trong callee giữ dòng của callee; về bình thường thì trả lại
            try: run(f.body,le)
            except Ret as r:
                CUR_LINE[0]=here
                return r.v
            CUR_LINE[0]=here
            return None
        if callable(f): return f(*a)
        raise RuntimeError("not callable")
    raise RuntimeError(f"expr {k}")

def fastnum(f,a):
    le=Env(f.env)
    for n,v in zip(f.args,a): le.decl(n,v,True)
    try: run(f.body,le)
    except Ret as r: return r.v
    return None


def call_fn(f,a):
    """Gọi một giá trị hàm (Fn hoặc native) — dùng cho map/filter/sort."""
    if isinstance(f,Fn):
        if len(a)!=len(f.args): raise TError(f"{f.name} arity")
        le=Env(f.env)
        for n,v in zip(f.args,a): le.decl(n,v,True)
        try: run(f.body,le)
        except Ret as r: return r.v
        return None
    if callable(f): return f(*a)
    raise TError("not callable")


def _needstr(s,who):
    if not isinstance(s,str): raise TError(f"{who}() str")
    return s

def _lpop(o,i=-1):
    if isinstance(i,bool) or not isinstance(i,int): raise TError("pop(i)")
    if not o: raise TError("pop empty")
    if i<0: i+=len(o)
    if i<0 or i>=len(o): raise TError("pop range")
    return o.pop(i)

def _linsert(o,i,x):
    if isinstance(i,bool) or not isinstance(i,int): raise TError("insert(i, x)")
    n=len(o)
    if i<0: i+=n
    if i<0: i=0
    if i>n: i=n
    o.insert(i,x)

def _lindex(o,x):
    for i,y in enumerate(o):
        if _teq(x,y): return i
    raise TError("index: not found")

def _dremove(d,k):
    for kk in list(d):
        if _teq(kk,k):
            del d[kk]; return None
    raise TError(f"key {fmt(k)[:60]} missing")

def _setarg(s,who):
    if isinstance(s,TSet): return s
    if isinstance(s,list): return TSet(s)
    raise TError(f"{who}() set")

def _seq(v,who):
    """list/tensor/str -> list (dùng cho sort/sum/map/filter/zip/enumerate)."""
    if isinstance(v,list): return v
    if isinstance(v,Tensor): return v.d
    if isinstance(v,str): return list(v)
    raise TError(f"{who}() list")


def _cmp_ok(x,y):
    nx=isinstance(x,(int,float)) and not isinstance(x,bool)
    ny=isinstance(y,(int,float)) and not isinstance(y,bool)
    if nx and ny: return True
    return isinstance(x,str) and isinstance(y,str)


def _msortkey(v):
    """Khóa sort mặc định: số trước chuỗi, so sánh chuỗi theo byte."""
    if isinstance(v,str): return (1,v.encode("latin-1","replace"))
    return (0,float(v))
    return None

def tensor_attr(o,f):
    if f=="len": return len(o)
    if f=="shape": return list(o.shape)
    if f=="ndim": return o.ndim
    if f=="size": return len(o.d)
    if f=="T": return TT.transpose(o)
    if f in TT.REDUCERS: return lambda *a: tens(TT.reduce,o,f,a)
    if f in TT.UNARY: return lambda: TT.unary(o,f)
    if f=="reshape": return lambda *a: tens(TT.reshape,o,list(a))
    if f=="transpose": return lambda: TT.transpose(o)
    if f=="flatten": return lambda: TT.raw(list(o.d),(len(o.d),))
    if f=="tolist": return lambda: TT.tolist(o)
    if f=="copy": return lambda: TT.raw(list(o.d),o.shape)
    if f=="dot": return lambda b: tens(TT.matmul,o,b)
    if f=="clip": return lambda lo,hi: tens(TT.clip,o,lo,hi)
    if f=="softmax": return lambda: TT.softmax(o)
    if f=="eq": return lambda x: tens(TT.binop,"eq",o,x)
    if f=="ne": return lambda x: tens(TT.binop,"ne",o,x)
    if f=="item":
        def item():
            if len(o.d)!=1: raise TError("item() cần tensor 1 phần tử")
            return o.d[0]
        return item
    raise TError(f"tensor has no .{f}")

def set_index(c,key,v):
    """c[key] = v (cũng là đích của c.field = v) — khớp OP_SETIDX của VM."""
    if isinstance(c,Tensor) and isinstance(key,list):
        if isinstance(v,bool) or not isinstance(v,(int,float,Tensor)): raise TError("tensor assign")
        tens(TT.set_multi,c,key,v); return
    if isinstance(c,dict):
        c[dkey(key)]=v; return
    if isinstance(c,TRec):
        if not isinstance(key,str) or key not in c.f:
            raise TError(f"{c.name} has no field {fmt(key)[:40]}")
        c.f[key]=v; return
    if isinstance(c,(list,Tensor)):
        if isinstance(key,bool) or not isinstance(key,int): raise TError("index int")
        if isinstance(c,Tensor):
            if isinstance(v,bool) or not isinstance(v,(int,float,Tensor)): raise TError("tensor assign")
            tens(TT.setitem,c,key,v); return
        n=len(c)
        if key<0 or key>=n: raise TError("assign range")
        c[key]=v
        return
    raise TError("assign here")

def tensor_iter(t): return [TT.getitem(t,i) for i in range(len(t))]

def run(stmts,env):
    for s in stmts:
        k=s[0]
        if k=="line": CUR_LINE[0]=(s[2],s[1]); continue
        if k=="decl": env.decl(s[2],ev(s[3],env),s[1]=="mut")
        elif k=="assign":
            t=s[1]; v=ev(s[2],env)
            if t[0]=="var":
                if env.has(t[1]): env.set(t[1],v)
                else: env.decl(t[1],v,True)  # bare x = ... auto-declares mut
            else:
                c=env.get(t[1][1]) if t[1][0]=="var" else ev(t[1],env)
                key=t[2] if t[0]=="attr" else ev(t[2],env)
                set_index(c,key,v)
        elif k=="fn": env.decl(s[1],Fn(s[1],s[2],s[3],env),False)
        elif k=="ret": raise Ret(ev(s[1],env))
        elif k=="print":
            vals=[ev(x,env) for x in s[1]]   # tính hết đối số rồi mới in (khớp VM)
            tprint(" ".join(fmt(v) for v in vals))
        elif k=="expr": ev(s[1],env)
        elif k=="if":
            if truthy(ev(s[1],env)): run(s[2],Env(env))
            elif s[3]: run(s[3],Env(env))
        elif k=="for":
            it=ev(s[2],env)
            if isinstance(it,dict): it=list(it.keys())
            elif isinstance(it,TSet): it=list(it.d.keys())
            elif isinstance(it,Tensor): it=tensor_iter(it)
            elif not isinstance(it,(list,str)): raise TError(f"for: not iterable {tname(it)}")
            for v in it:
                try: le=Env(env); le.decl(s[1],v,True); run(s[3],le)
                except Cnt: continue
                except Brk: break
        elif k=="while":
            while truthy(ev(s[1],env)):
                try: run(s[2],Env(env))
                except Cnt: continue
                except Brk: break
        elif k=="break": raise Brk()
        elif k=="continue": raise Cnt()
        elif k=="raise": raise TError(fmt(ev(s[1],env)))
        elif k=="assert":
            if not truthy(ev(s[1],env)):
                d="assert failed" if s[2] is None else "assert failed: "+fmt(ev(s[2],env))
                raise TError(d)
        elif k=="try":
            try: run(s[1],Env(env))
            except (Ret,Brk,Cnt): raise  # return/break/continue xuyên qua try
            except Exception as ex:
                # Bắt mọi lỗi (kể cả ValueError từ int()/index) thành TError
                # để code T có thể try/except; giữ tên type để match.
                msg=str(ex); hit=False
                for name,body in s[2]:
                    if name is None or t_match(msg, name) or type(ex).__name__==name:
                        le=Env(env); le.decl("err",msg,True); run(body,le); hit=True; break
                if not hit: raise
        elif k=="import":
            mod=load_mod(s[1],env)
            env.decl(s[2] or s[1].split(".")[-1],mod,False)
        elif k=="fromimport":
            mod=load_mod(s[1],env)
            for n in s[2]:
                try: env.decl(n,mod.env.get(n),False)
                except RuntimeError: raise TError(f"cannot import {n} from {s[1]}")
        else: raise RuntimeError(f"stmt {k}")

# ---------- builtin "mềm": global định sẵn, user được phép che tên ----------
def b_set(*a):
    if len(a)==0: return TSet()
    if len(a)!=1: raise TError("set([list])")
    v=a[0]
    if isinstance(v,TSet): return TSet(list(v.d))
    if isinstance(v,dict): return TSet(list(v.keys()))
    return TSet(_seq(v,"set"))

def b_input(*a):
    if len(a)>1: raise TError("input([prompt])")
    if a: tprint(fmt(a[0]),nl=False)
    try: ln=sys.stdin.buffer.readline()
    except (OSError,ValueError): return None
    if not ln: return None
    s=ln.decode("latin-1")
    if s.endswith("\n"): s=s[:-1]
    if s.endswith("\r"): s=s[:-1]
    return s

def b_struct(name,fields,meths):
    if not isinstance(fields,list) or not isinstance(meths,dict):
        raise TError("__struct(name, fields, methods)")
    return TStruct(fmt(name),list(fields),meths)

def b_new(tp,*vals):
    if not isinstance(tp,TStruct) or len(vals)!=len(tp.fields):
        raise TError("__new(type, values...)")
    return TRec(tp.name,dict(zip(tp.fields,vals)),tp.m)

import math as _math

def _mathfn(name,f):
    """hàm toán học: số -> số (inf/nan như C, không ném lỗi); tensor -> từng phần tử"""
    def g(*a):
        if len(a)!=1: raise TError(f"{name}(x)")
        x=a[0]
        if isinstance(x,Tensor): return TT.unary(x,name)
        if isinstance(x,bool) or not isinstance(x,(int,float)): raise TError(f"{name}() number")
        return f(float(x))
    return g

def _floorceil(name,f):
    def g(*a):
        if len(a)!=1: raise TError(f"{name}(x)")
        x=a[0]
        if isinstance(x,Tensor): return TT.unary(x,name)
        if isinstance(x,bool) or not isinstance(x,(int,float)): raise TError(f"{name}() number")
        if isinstance(x,int): return x
        if x!=x or x in (float("inf"),float("-inf")): raise TError(f"{name}() finite")
        return int(f(x))
    return g

def _tensor_ctor(fn,name):
    def g(*a): return tens(fn,*a)
    g.__name__=name
    return g

SOFT_BUILTINS={"set":b_set,"input":b_input,"__struct":b_struct,"__new":b_new,
    "zeros":_tensor_ctor(lambda s: TT.full(s,0,"zeros"),"zeros"),
    "ones":_tensor_ctor(lambda s: TT.full(s,1,"ones"),"ones"),
    "full":_tensor_ctor(lambda s,v: TT.full(s,v,"full"),"full"),
    "arange":_tensor_ctor(TT.arange,"arange"),
    "eye":_tensor_ctor(TT.eye,"eye"),
    "rand":_tensor_ctor(TT.rand,"rand"),
    "randn":_tensor_ctor(TT.randn,"randn"),
    "seed":_tensor_ctor(lambda n: (TT.seed(n),None)[1],"seed"),
    "sqrt":_mathfn("sqrt",TT.UNARY["sqrt"]),
    "exp":_mathfn("exp",TT.UNARY["exp"]),
    "log":_mathfn("log",TT.UNARY["log"]),
    "sin":_mathfn("sin",TT.UNARY["sin"]),
    "cos":_mathfn("cos",TT.UNARY["cos"]),
    "tanh":_mathfn("tanh",TT.UNARY["tanh"]),
    "floor":_floorceil("floor",_math.floor),
    "ceil":_floorceil("ceil",_math.ceil),
    "pi":_math.pi}

def builtin_env():
    e=Env(); e.isb=True
    for k,v in SOFT_BUILTINS.items(): e.m[k]=v
    return e

BUILTINS=None

def root_env():
    global BUILTINS
    if BUILTINS is None: BUILTINS=builtin_env()
    return Env(BUILTINS)

CUR_FILE=[None]
CUR_LINE=[None]   # (file, dòng) của câu lệnh đang chạy (mốc "line")

def load_mod(dotted,env):
    rel=dotted.replace(".","/")+".t"
    leaf=rel.split("/")[-1]
    base=_os.path.dirname(_os.path.abspath(CUR_FILE[0])) if CUR_FILE[0] else _os.getcwd()
    cwd=_os.getcwd()
    # search: same dir, same dir/std|self|tools|tpkgs, cwd..., parent...
    # (fixes `import ai` from tests/*.t finding std/ai.t at project root,
    #  `import lex` finding self/lex.t, `import fmt` finding tools/fmt.t,
    #  `import dep` finding tpkgs/dep/dep.t sau `tpm install`)
    tpleaf = _os.path.join("tpkgs", leaf[:-2], leaf[:-2] + ".t") if leaf.endswith(".t") else None
    cand=[_os.path.join(base,rel),
          _os.path.join(base,"std",leaf),
          _os.path.join(base,"self",leaf),
          _os.path.join(base,"tools",leaf),
          _os.path.join(cwd,rel),
          _os.path.join(cwd,"std",leaf),
          _os.path.join(cwd,"self",leaf),
          _os.path.join(cwd,"tools",leaf),
          _os.path.join(_os.path.dirname(base),rel),
          _os.path.join(_os.path.dirname(base),"std",leaf),
          _os.path.join(_os.path.dirname(base),"self",leaf),
          _os.path.join(_os.path.dirname(base),"tools",leaf)]
    if tpleaf:
        cand += [_os.path.join(base,tpleaf),
                 _os.path.join(cwd,tpleaf),
                 _os.path.join(_os.path.dirname(base),tpleaf)]
    # dedup preserving order
    seen=set(); uniq=[]
    for c in cand:
        a=_os.path.normpath(c)
        if a not in seen: seen.add(a); uniq.append(a)
    path=next((c for c in uniq if _os.path.isfile(c)),None)
    if path is None: raise TError(f"no module {dotted!r} (searched {uniq[0]!r} ...)")
    ap=_os.path.abspath(path)
    if ap in LOADED: return LOADED[ap]
    menv=root_env(); mod=Module(dotted,menv); LOADED[ap]=mod
    old=CUR_FILE[0]; CUR_FILE[0]=ap
    try:
        with open(ap,encoding="latin-1",newline="") as f:
            run(P(lex2(f.read()),file_base(ap)).parse()[1],menv)
    finally: CUR_FILE[0]=old
    return mod

def run_src(src,env=None):
    env=env or root_env()
    prog=P(lex2(src)).parse()
    run(prog[1],env)
    return env

def main():
    for st in (sys.stdout, sys.stderr):
        try: st.reconfigure(encoding="utf-8", errors="backslashreplace")  # type: ignore[attr-defined]
        except Exception: pass
    if len(sys.argv)>1:
        CUR_FILE[0]=_os.path.abspath(sys.argv[1])
        try:
            with open(sys.argv[1],encoding="latin-1",newline="") as f:
                prog=P(lex2(f.read()),file_base(sys.argv[1])).parse()
            run(prog[1],root_env())
        except (Ret,Brk,Cnt):
            raise
        except RecursionError:
            sys.stdout.flush(); sys.stderr.write("T error: recursion too deep\n"); raise SystemExit(1)
        except (TError,SyntaxError,RuntimeError,ValueError,TypeError,ZeroDivisionError,
                IndexError,KeyError,OSError,AttributeError,OverflowError) as ex:
            if _os.environ.get("TPY_TRACE"): raise
            sys.stdout.flush()
            loc=f"\n  tại {CUR_LINE[0][0]}:{CUR_LINE[0][1]}" if CUR_LINE[0] and not isinstance(ex,SyntaxError) else ""
            sys.stderr.write(f"T error: {ex}{loc}\n"); raise SystemExit(1)
    else:
        repl()

def repl():
    """REPL: khối kết thúc bằng dòng trống; biến sống qua các khối.
    Biểu thức đơn được in giá trị (khác nil). Thoát: EOF hoặc `exit`."""
    tprint("T v2.2 REPL  (khoi lenh ket thuc bang dong trong; thoat: exit)")
    env=root_env(); buf=""
    while True:
        tprint("... " if buf else ">>> ",nl=False)
        try: ln=sys.stdin.buffer.readline().decode("latin-1")
        except (EOFError,OSError): break
        if ln=="": break
        ln=ln.rstrip("\r\n")
        if not buf and ln.strip()=="exit": break
        if ln.strip() and (buf or ln.rstrip().endswith(":")):
            buf+=ln+"\n"; continue
        if ln.strip(): buf=ln+"\n"
        if not buf: continue
        try:
            prog=P(lex2(buf)).parse()
            if len(prog[1])==1 and prog[1][0][0]=="expr":
                v=ev(prog[1][0][1],env)
                if v is not None: tprint(fmt(v))
            else:
                run(prog[1],env)
        except (Ret,Brk,Cnt):
            tprint("error: return/break ngoài hàm/vòng lặp")
        except Exception as ex:
            tprint(f"error: {ex}")
        buf=""

def main_big_stack():
    """Chạy main trong thread stack lớn: đệ quy T sâu (vài nghìn tầng, mỗi tầng
    tốn nhiều frame Python) không làm tràn stack C của CPython."""
    import threading
    ok=False
    for mb in (255,128,64):          # Windows: tối đa < 256MB
        try: threading.stack_size(mb*1024*1024); ok=True; break
        except (ValueError,RuntimeError,OverflowError): pass
    if not ok: return main()
    sys.setrecursionlimit(mb*1024*2) # ~ 512 byte stack C / frame Python là dư
    box=[0]
    def runner():
        try: main()
        except SystemExit as ex: box[0]=ex.code if isinstance(ex.code,int) else (0 if ex.code is None else 1)
    th=threading.Thread(target=runner); th.start(); th.join()
    sys.stdout.flush()
    raise SystemExit(box[0])

if __name__=="__main__": main_big_stack()
