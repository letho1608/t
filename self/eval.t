# self/eval.t — evaluator của T viết bằng chính T (v0.8 self-host full)
# Chạy AST do host.scan + host.parse_prog sinh ra, ngữ nghĩa khớp t.py.
# Giá trị T ánh xạ trực tiếp sang giá trị host (int/float/str/bool/nil/
# list/tensor/dict/file), nên số học/index/slice/print ủy thác cho host.
# Tín hiệu điều khiển: ["ok"] | ["ret", v] | ["brk"] | ["cnt"] | ["err", msg]
# Lỗi user (raise) đi bằng tín hiệu ["err", msg]; lỗi host bên trong expr
# được ném lên với tiền tố "__terr__:" rồi bóc ở biên stmt.
import lex
import parse
import parse2
import host

# --- tag check KHÔNG đệ quy (không gọi type()): đọc key, lỗi -> nil ---
# (v[k] trên non-dict/dict-thiếu-key ném lỗi host, T try bắt được -> nil)
fn tag_get(v, k):
    try:
        return v[k]
    except:
        return nil

fn dict_has(d, k):
    for x in d.keys():
        if x == k:
            return true
    return false

fn terr_prefix():
    return "__terr__:"

fn raise_err(msg):
    raise "__terr__:" + msg

fn is_sig(s, k):
    return type(s) == "list" and s.len > 0 and s[0] == k

fn strip_terr(e):
    p = terr_prefix()
    if e.len >= p.len and e[0:9] == p:
        return e[9:]
    return e

# --- dict/str helpers (T không có toán tử `in`) ---
fn str_contains(hay, needle):
    if needle == "":
        return true
    parts = hay.split(needle)
    return parts.len > 1

fn has_wild(p):
    return str_contains(p, "*") or str_contains(p, "?")

fn glob_match(s, p):
    if p == "":
        return s == ""
    if p[0:1] == "*":
        if glob_match(s, p[1:]):
            return true
        if s == "":
            return false
        return glob_match(s[1:], p)
    if s == "":
        return false
    if p[0:1] == "?" or p[0:1] == s[0:1]:
        return glob_match(s[1:], p[1:])
    return false

fn exc_match(msg, nm):
    if has_wild(nm):
        return glob_match(msg, nm)
    return str_contains(msg, nm)

# --- env: {"slots": {}, "mut": {}, "up": parent-or-nil} ---
fn env_new(parent):
    return {"slots": {}, "mut": {}, "up": parent}

fn env_decl(env, name, val, is_mut):
    slots = env["slots"]
    if dict_has(slots, name):
        raise "redeclare " + name
    slots[name] = val
    if is_mut:
        env["mut"][name] = true
    return nil

fn env_has(env, name):
    mut e = env
    while e != nil:
        if dict_has(e["slots"], name):
            return true
        e = e["up"]
    return false

fn env_get(env, name):
    mut e = env
    while e != nil:
        slots = e["slots"]
        if dict_has(slots, name):
            return slots[name]
        e = e["up"]
    if name == "set" or name == "input" or name == "__struct" or name == "__new" or name == "load":
        return {"__bltin__": name}
    sb = soft_bltin(name)
    if sb != nil:
        return sb
    raise "undef " + name

# builtin mềm của host (tensor + toán học): trả thẳng hàm/giá trị của host
fn soft_bltin(name):
    if name == "zeros":
        return zeros
    if name == "ones":
        return ones
    if name == "full":
        return full
    if name == "arange":
        return arange
    if name == "eye":
        return eye
    if name == "rand":
        return rand
    if name == "randn":
        return randn
    if name == "seed":
        return seed
    if name == "sqrt":
        return sqrt
    if name == "exp":
        return exp
    if name == "log":
        return log
    if name == "sin":
        return sin
    if name == "cos":
        return cos
    if name == "tanh":
        return tanh
    if name == "floor":
        return floor
    if name == "ceil":
        return ceil
    if name == "pi":
        return pi
    return nil

fn env_set(env, name, val):
    mut e = env
    while e != nil:
        slots = e["slots"]
        if dict_has(slots, name):
            if not dict_has(e["mut"], name):
                raise name + " immutable (use mut)"
            slots[name] = val
            return nil
        e = e["up"]
    raise "undef " + name

# --- tags: userfn / module / bltin(print). Chỉ đọc tag (trung thực mọi level),
# KHÔNG gọi type() ở đây (type() qua bltin sẽ dối ở tầng eval lồng nhau).
fn is_userfn(v):
    return tag_get(v, "__tfn__") == true

fn is_module(v):
    return tag_get(v, "__tmod__") == true

fn is_bltin(v):
    return tag_get(v, "__bltin__") != nil

fn is_rec(v):
    return tag_get(v, "__trec__") == true

fn is_bound(v):
    return tag_get(v, "__tbound__") == true

fn mk_rec(name, fields, vals, meths):
    mut f = {}
    mut i = 0
    while i < fields.len:
        f[fields[i]] = vals[i]
        i = i + 1
    return {"__trec__": true, "name": name, "f": f, "m": meths}

# field của bản ghi; bản ghi module (load) lấy lại giá trị hiện tại từ env module
fn rec_f(o):
    if dict_has(o, "menv"):
        sl = o["menv"]["slots"]
        f = o["f"]
        for k in o["keys"]:
            f[k] = sl[k]
    return o["f"]

fn rec_set(o, k, v):
    if not dict_has(o["f"], k):
        raise o["name"] + " has no field " + str(k)
    if dict_has(o, "menv") and not dict_has(o["menv"]["mut"], k):
        raise k + " immutable (use mut)"
    o["f"][k] = v
    if dict_has(o, "menv"):
        o["menv"]["slots"][k] = v
    return nil

fn rec_eq(a, b):
    if not is_rec(a) or not is_rec(b):
        return false
    return a["name"] == b["name"] and rec_f(a) == rec_f(b)

fn mk_userfn(name, params, body, closure):
    return {"__tfn__": true, "name": name, "params": params, "body": body, "closure": closure}

fn mk_module(name, env):
    return {"__tmod__": true, "name": name, "env": env}

# --- num/lit ---
fn digit_val(c):
    if c >= "0" and c <= "9":
        return ord(c) - ord("0")
    if c >= "a" and c <= "f":
        return ord(c) - ord("a") + 10
    if c >= "A" and c <= "F":
        return ord(c) - ord("A") + 10
    return -1

fn acc_base(s, i, base, acc):
    mut v = acc
    while i < s.len:
        d = digit_val(s[i:i + 1])
        if d < 0 or d >= base:
            raise "số không hợp lệ: " + s
        if v > 1000000000000000000:
            raise "int quá lớn"
        v = v * base + d
        i = i + 1
    return v

fn num_val(s):
    mut t = ""
    for i in range(s.len):
        c = s[i:i + 1]
        if c != "_":
            t = t + c
    if t.len > 1:
        p = t[0:2]
        if p == "0x" or p == "0X":
            return acc_base(t, 2, 16, 0)
        if p == "0b" or p == "0B":
            return acc_base(t, 2, 2, 0)
        if p == "0o" or p == "0O":
            return acc_base(t, 2, 8, 0)
    if t.split(".").len > 1 or t.split("e").len > 1 or t.split("E").len > 1:
        return float(t)
    return acc_base(t, 0, 10, 0)

fn lit_val(s):
    if s == "true":
        return true
    if s == "false":
        return false
    return nil

# --- attr động (host không có getattr) ---
fn attr_get(o, f):
    if is_module(o):
        menv = o["env"]
        if env_has(menv, f):
            return env_get(menv, f)
        raise "module " + o["name"] + " has no " + f
    if is_rec(o):
        if dict_has(o["f"], f):
            return rec_f(o)[f]
        if dict_has(o["m"], f):
            return {"__tbound__": true, "fn": o["m"][f], "self": o}
        raise o["name"] + " has no ." + f
    if is_userfn(o) or is_bltin(o) or is_bound(o):
        raise "no attr " + f + " on fn"
    if type(o) == "set":
        if f == "len":
            return o.len
        if f == "add":
            return o.add
        if f == "has":
            return o.has
        if f == "remove":
            return o.remove
        if f == "list":
            return o.list
        if f == "union":
            return o.union
        if f == "inter":
            return o.inter
        if f == "diff":
            return o.diff
        raise "set has no ." + f
    if type(o) == "str":
        if f == "len":
            return o.len
        if f == "upper":
            return o.upper
        if f == "lower":
            return o.lower
        if f == "split":
            return o.split
        if f == "join":
            return o.join
        if f == "strip":
            return o.strip
        if f == "find":
            return o.find
        if f == "replace":
            return o.replace
        if f == "startswith":
            return o.startswith
        if f == "endswith":
            return o.endswith
        raise "str has no ." + f
    if type(o) == "list":
        if f == "len":
            return o.len
        if f == "append":
            return o.append
        if f == "pop":
            return o.pop
        if f == "next":
            return o.next
        if f == "clear":
            return o.clear
        if f == "insert":
            return o.insert
        if f == "extend":
            return o.extend
        if f == "index":
            return o.index
        raise "list has no ." + f
    if type(o) == "dict":
        if f == "len":
            return o.len
        if f == "keys":
            return o.keys
        if f == "values":
            return o.values
        if f == "get":
            return o.get
        if f == "items":
            return o.items
        if f == "has":
            return o.has
        if f == "remove":
            return o.remove
        raise "dict has no ." + f
    raise "no attr ." + f + " here"

fn tensor_attr(o, f):
    if f == "len":
        return o.len
    if f == "shape":
        return o.shape
    if f == "ndim":
        return o.ndim
    if f == "size":
        return o.size
    if f == "T":
        return o.T
    if f == "sum":
        return o.sum
    if f == "mean":
        return o.mean
    if f == "max":
        return o.max
    if f == "min":
        return o.min
    if f == "argmax":
        return o.argmax
    if f == "argmin":
        return o.argmin
    if f == "reshape":
        return o.reshape
    if f == "transpose":
        return o.transpose
    if f == "flatten":
        return o.flatten
    if f == "tolist":
        return o.tolist
    if f == "copy":
        return o.copy
    if f == "dot":
        return o.dot
    if f == "clip":
        return o.clip
    if f == "softmax":
        return o.softmax
    if f == "softmax_bwd":
        return o.softmax_bwd
    if f == "gelu":
        return o.gelu
    if f == "gelu_bwd":
        return o.gelu_bwd
    if f == "rmsnorm":
        return o.rmsnorm
    if f == "rmsnorm_bwd":
        return o.rmsnorm_bwd
    if f == "layernorm":
        return o.layernorm
    if f == "layernorm_bwd":
        return o.layernorm_bwd
    if f == "swiglu_bwd":
        return o.swiglu_bwd
    if f == "conv2d":
        return o.conv2d
    if f == "conv2d_bwd":
        return o.conv2d_bwd
    if f == "item":
        return o.item
    if f == "eq":
        return o.eq
    if f == "ne":
        return o.ne
    if f == "data_ptr":
        return o.data_ptr
    if f == "raw_ptr":
        return o.raw_ptr
    if f == "dtype":
        return o.dtype
    if f == "itemsize":
        return o.itemsize
    if f == "nbytes":
        return o.nbytes
    if f == "to":
        return o.to
    if f == "astype":
        return o.astype
    if f == "exp":
        return o.exp
    if f == "log":
        return o.log
    if f == "sqrt":
        return o.sqrt
    if f == "tanh":
        return o.tanh
    if f == "sin":
        return o.sin
    if f == "cos":
        return o.cos
    if f == "abs":
        return o.abs
    if f == "relu":
        return o.relu
    if f == "sigmoid":
        return o.sigmoid
    if f == "floor":
        return o.floor
    if f == "ceil":
        return o.ceil
    raise "tensor has no ." + f

# gọi host-callable với argv động (0..4 đủ cho methods)
fn call_host(f, args):
    n = args.len
    if n == 0:
        return f()
    if n == 1:
        return f(args[0])
    if n == 2:
        return f(args[0], args[1])
    if n == 3:
        return f(args[0], args[1], args[2])
    if n == 4:
        return f(args[0], args[1], args[2], args[3])
    raise "too many args"

# --- builtins ---
fn bltin_type(v):
    if is_rec(v):
        return v["name"]
    if is_userfn(v) or is_bltin(v) or is_bound(v):
        return "fn"
    if is_module(v):
        return "fn"
    return type(v)

fn bltin_str(v):
    if is_rec(v):
        if dict_has(v["m"], "__str__"):
            return do_call(v["m"]["__str__"], [v], {})
        mut parts = []
        rf = rec_f(v)
        for k in rf.keys():
            parts.append(k + ": " + bltin_str(rf[k]))
        return v["name"] + "{" + ", ".join(parts) + "}"
    if is_bound(v):
        return "<fn>"
    if is_userfn(v):
        return "<fn " + v["name"] + ">"
    if is_bltin(v):
        return "<fn " + v["__bltin__"] + ">"
    if is_module(v):
        return "module(" + v["name"] + ")"
    return str(v)

# --- builtins tập trên (khớp t.py và vm/tvm.c) ---
fn b_num(v, who):
    if type(v) == "int" or type(v) == "float":
        return v
    raise who + "() number"

fn b_seq(v, who):
    if type(v) == "list":
        return v
    if type(v) == "str":
        return v.split("")
    if type(v) == "tensor":
        return tensor_to_list(v)
    raise who + "() list"

fn tensor_to_list(t):
    # phần tử phẳng (khớp t.py/VM: sum/min/max/map trên tensor n-chiều)
    f = t.flatten()
    mut out = []
    mut i = 0
    while i < len(f):
        out.append(f[i])
        i = i + 1
    return out

fn b_cmp(x, y, who):
    # trả -1/0/1; số-vs-số, chuỗi-vs-chuỗi (byte)
    tx = type(x)
    ty = type(y)
    if (tx == "int" or tx == "float") and (ty == "int" or ty == "float"):
        if x < y:
            return -1
        if x > y:
            return 1
        return 0
    if tx == "str" and ty == "str":
        if x < y:
            return -1
        if x > y:
            return 1
        return 0
    raise who + "() number/str"

fn do_abs(args):
    if args.len != 1:
        raise "abs() number"
    if type(args[0]) == "tensor":
        return abs(args[0])
    v = b_num(args[0], "abs")
    if v < 0:
        return -v
    return v

fn do_round(args):
    if args.len != 1:
        raise "round() number"
    v = b_num(args[0], "round")
    a = v
    if a < 0:
        a = -a
    mut f = int(a + 0.5)
    if v < 0:
        f = -f
    if type(v) == "float":
        return float(f)
    return f

fn do_minmax(args, ismin):
    who = "max"
    if ismin:
        who = "min"
    if args.len == 2:
        b_cmp(args[0], args[1], who)
        if ismin:
            if args[0] <= args[1]:
                return args[0]
            return args[1]
        if args[0] >= args[1]:
            return args[0]
        return args[1]
    if args.len != 1:
        raise who + "(list) or " + who + "(a, b)"
    xs = b_seq(args[0], who)
    if len(xs) == 0:
        raise who + "() empty"
    best = xs[0]
    mut i = 1
    while i < len(xs):
        x = xs[i]
        b_cmp(x, best, who)
        if ismin:
            if x < best:
                best = x
        else:
            if x > best:
                best = x
        i = i + 1
    return best

fn do_sum(args):
    if args.len < 1 or args.len > 2:
        raise "sum(list[, init])"
    xs = b_seq(args[0], "sum")
    mut isf = false
    mut acc = 0
    if args.len == 2:
        acc = b_num(args[1], "sum")
        if type(args[1]) == "float":
            isf = true
    mut i = 0
    while i < len(xs):
        x = b_num(xs[i], "sum")
        if type(x) == "float":
            isf = true
        acc = acc + x
        i = i + 1
    if isf:
        return float(acc)
    return int(acc)

fn do_anyall(args, isall):
    who = "any"
    if isall:
        who = "all"
    if args.len != 1:
        raise who + "(list)"
    xs = b_seq(args[0], who)
    mut res = isall
    mut i = 0
    while i < len(xs):
        t = xs[i]
        if isall:
            if not t:
                res = false
                break
        else:
            if t:
                res = true
                break
        i = i + 1
    return res

fn sless(x, y):
    # x < y ? (số với số, chuỗi với chuỗi theo byte)
    return b_cmp(x, y, "sort") < 0

fn sort_list(xs, mode):
    # merge sort ổn định; mode "key": xs là list các cặp [khoá, giá trị]
    mut n = xs.len
    if n < 2:
        return xs
    mut mid = int(n / 2)
    mut left = xs[0:mid]
    mut right = xs[mid:n]
    left = sort_list(left, mode)
    right = sort_list(right, mode)
    mut out = []
    mut i = 0
    mut j = 0
    while i < len(left) and j < len(right):
        ka = left[i]
        kb = right[j]
        if mode == "key":
            ka = ka[0]
            kb = kb[0]
        if sless(kb, ka):
            out.append(right[j])
            j = j + 1
        else:
            out.append(left[i])
            i = i + 1
    while i < len(left):
        out.append(left[i])
        i = i + 1
    while j < len(right):
        out.append(right[j])
        j = j + 1
    return out

fn do_sort(args, mods):
    if args.len < 1 or args.len > 2:
        raise "sort(list[, key])"
    if type(args[0]) != "list":
        raise "sort() list"
    mut src = args[0]
    if args.len == 1:
        mut i = 0
        while i < len(src):
            b_cmp(src[i], src[i], "sort")
            i = i + 1
        # sắp TẠI CHỖ và trả chính list đó (khớp t.py / VM)
        sorted1 = sort_list(src[0:len(src)], nil)
        src.clear()
        for x in sorted1:
            src.append(x)
        return src
    f = args[1]
    if not is_userfn(f) and not is_bltin(f):
        raise "sort() fn"
    mut pairs = []
    mut i = 0
    while i < len(src):
        pairs.append([do_call(f, [src[i]], mods), src[i]])
        i = i + 1
    mut sorted = sort_list(pairs, "key")
    mut out = []
    i = 0
    while i < len(sorted):
        out.append(sorted[i][1])
        i = i + 1
    src.clear()
    for x in out:
        src.append(x)
    return src

fn do_reversed(args):
    if args.len != 1:
        raise "reversed(list|str)"
    v = args[0]
    if type(v) == "str":
        mut chars = v.split("")
        mut out = []
        mut i = len(chars) - 1
        while i >= 0:
            out.append(chars[i])
            i = i - 1
        return "".join(out)
    xs = b_seq(v, "reversed")
    mut out = []
    mut i = len(xs) - 1
    while i >= 0:
        out.append(xs[i])
        i = i - 1
    return out

fn do_enumerate(args):
    if args.len < 1 or args.len > 2:
        raise "enumerate(list[, start])"
    mut st = 0
    if args.len == 2:
        st = int(args[1])
    xs = b_seq(args[0], "enumerate")
    mut out = []
    mut i = 0
    while i < len(xs):
        out.append([st + i, xs[i]])
        i = i + 1
    return out

fn do_zip(args):
    if args.len < 2:
        raise "zip(list, list, ...)"
    mut seqs = []
    mut n = -1
    for x in args:
        s = b_seq(x, "zip")
        seqs.append(s)
        if n < 0 or len(s) < n:
            n = len(s)
    mut out = []
    mut i = 0
    while i < n:
        mut row = []
        for s in seqs:
            row.append(s[i])
        out.append(row)
        i = i + 1
    return out

fn do_map(args, mods):
    if args.len != 2:
        raise "map(fn, list)"
    f = args[0]
    if not is_userfn(f) and not is_bltin(f):
        raise "map() fn"
    xs = b_seq(args[1], "map")
    mut out = []
    mut i = 0
    while i < len(xs):
        out.append(do_call(f, [xs[i]], mods))
        i = i + 1
    return out

fn do_filter(args, mods):
    if args.len != 2:
        raise "filter(fn, list)"
    f = args[0]
    if not is_userfn(f) and not is_bltin(f):
        raise "filter() fn"
    xs = b_seq(args[1], "filter")
    mut out = []
    mut i = 0
    while i < len(xs):
        x = xs[i]
        if do_call(f, [x], mods):
            out.append(x)
        i = i + 1
    return out

fn eval_bltin(name, args, mods):
    if name == "len":
        v = args[0]
        if is_rec(v) and dict_has(v["m"], "__len__"):
            return do_call(v["m"]["__len__"], [v], mods)
        if is_userfn(v) or is_bltin(v) or is_module(v):
            raise "len() on fn"
        return len(v)
    if name == "tensor":
        if args.len == 1:
            return tensor(args[0])
        return tensor(args[0], args[1])
    if name == "str":
        return bltin_str(args[0])
    if name == "int":
        return int(args[0])
    if name == "float":
        return float(args[0])
    if name == "type":
        return bltin_type(args[0])
    if name == "range":
        if args.len == 1:
            return range(args[0])
        return range(args[0], args[1])
    if name == "open":
        if args.len == 1:
            return open(args[0])
        return open(args[0], args[1])
    if name == "read":
        return read(args[0])
    if name == "write":
        return write(args[0], args[1])
    if name == "close":
        if args.len == 2:
            if is_rec(args[0]) and dict_has(args[0]["m"], "__enter__"):
                return do_call(args[0]["m"]["__enter__"], [args[0]], mods)
            return args[0]
        if args[0] == nil:
            return nil
        if is_rec(args[0]):
            if dict_has(args[0]["m"], "__exit__"):
                do_call(args[0]["m"]["__exit__"], [args[0]], mods)
                return nil
            if dict_has(args[0]["m"], "close"):
                do_call(args[0]["m"]["close"], [args[0]], mods)
                return nil
            return nil
        return close(args[0])
    if name == "ord":
        return ord(args[0])
    if name == "chr":
        return chr(args[0])
    if name == "argv":
        return argv()
    if name == "mem":
        return {"objs": 0, "gc": 0}
    if name == "f64bits":
        return f64bits(args[0])
    if name == "abs":
        return do_abs(args)
    if name == "round":
        return do_round(args)
    if name == "min":
        return do_minmax(args, true)
    if name == "max":
        return do_minmax(args, false)
    if name == "sum":
        return do_sum(args)
    if name == "any":
        return do_anyall(args, false)
    if name == "all":
        return do_anyall(args, true)
    if name == "sort":
        return do_sort(args, mods)
    if name == "reversed":
        return do_reversed(args)
    if name == "enumerate":
        return do_enumerate(args)
    if name == "zip":
        return do_zip(args)
    if name == "map":
        return do_map(args, mods)
    if name == "filter":
        return do_filter(args, mods)
    raise "unknown bltin " + name

fn do_print(args):
    mut out = ""
    mut i = 0
    while i < args.len:
        if i > 0:
            out = out + " "
        out = out + bltin_str(args[i])
        i = i + 1
    print(out)
    return nil

# --- eval expr ---
fn eval_expr(e, env, mods):
    k = e[0]
    if k == "num":
        return num_val(e[1])
    if k == "str":
        return e[1]
    if k == "fstr":
        mut out = ""
        for p in e[1]:
            if p[0] == "s":
                out = out + p[1]
            else:
                out = out + bltin_str(eval_expr(p[1], env, mods))
        return out
    if k == "lit":
        return lit_val(e[1])
    if k == "cond":
        if eval_expr(e[1], env, mods):
            return eval_expr(e[2], env, mods)
        return eval_expr(e[3], env, mods)
    if k == "var":
        return env_get(env, e[1])
    if k == "bltfn":
        return {"__bltin__": e[1]}
    if k == "list":
        mut r = []
        for x in e[1]:
            r.append(eval_expr(x, env, mods))
        return r
    if k == "dict":
        mut d = {}
        mut i = 0
        while i < e[1].len:
            kk = eval_expr(e[1][i], env, mods)
            vv = eval_expr(e[2][i], env, mods)
            d[kk] = vv
            i = i + 1
        return d
    if k == "tensor":
        v = eval_expr(e[1], env, mods)
        return tensor(v)
    if k == "bltin":
        mut a = []
        for x in e[2]:
            a.append(eval_expr(x, env, mods))
        return eval_bltin(e[1], a, mods)
    if k == "un":
        v = eval_expr(e[2], env, mods)
        if e[1] == "-":
            if is_rec(v) and dict_has(v["m"], "__neg__"):
                return do_call(v["m"]["__neg__"], [v], mods)
            return 0 - v
        if e[1] == "~":
            return ~v
        return not v
    if k == "bin":
        op = e[1]
        if op == "and":
            a = eval_expr(e[2], env, mods)
            if a:
                return eval_expr(e[3], env, mods)
            return a
        if op == "or":
            a = eval_expr(e[2], env, mods)
            if a:
                return a
            return eval_expr(e[3], env, mods)
        a = eval_expr(e[2], env, mods)
        b = eval_expr(e[3], env, mods)
        if is_rec(a):
            mut dn = nil
            if op == "+":
                dn = "__add__"
            elif op == "-":
                dn = "__sub__"
            elif op == "*":
                dn = "__mul__"
            elif op == "/":
                dn = "__div__"
            elif op == "%":
                dn = "__mod__"
            elif op == "**":
                dn = "__pow__"
            elif op == "@":
                dn = "__matmul__"
            elif op == "==":
                dn = "__eq__"
            elif op == "!=":
                dn = "__ne__"
            elif op == "<":
                dn = "__lt__"
            elif op == ">":
                dn = "__gt__"
            elif op == "<=":
                dn = "__le__"
            elif op == ">=":
                dn = "__ge__"
            if dn != nil and dict_has(a["m"], dn):
                return do_call(a["m"][dn], [a, b], mods)
            if op == "!=" and dict_has(a["m"], "__eq__"):
                return not do_call(a["m"]["__eq__"], [a, b], mods)
        if op == "+":
            return a + b
        if op == "-":
            return a - b
        if op == "*":
            return a * b
        if op == "/":
            return a / b
        if op == "%":
            return a % b
        if op == "**":
            return a ** b
        if op == "@":
            return a @ b
        if op == "==":
            if is_rec(a) or is_rec(b):
                return rec_eq(a, b)
            return a == b
        if op == "!=":
            if is_rec(a) or is_rec(b):
                return not rec_eq(a, b)
            return a != b
        if op == "in":
            return a in b
        if op == "<":
            return a < b
        if op == ">":
            return a > b
        if op == "<=":
            return a <= b
        if op == ">=":
            return a >= b
        if op == "&":
            return a & b
        if op == "|":
            return a | b
        if op == "^":
            return a ^ b
        if op == "<<":
            return a << b
        if op == ">>":
            return a >> b
        raise "bad op " + op
    if k == "idx":
        c = eval_expr(e[1], env, mods)
        ix = eval_expr(e[2], env, mods)
        if is_rec(c):
            if dict_has(c["f"], ix):
                return rec_f(c)[ix]
            if dict_has(c["m"], "__getitem__"):
                return do_call(c["m"]["__getitem__"], [c, ix], mods)
            raise c["name"] + " has no field " + str(ix)
        return c[ix]
    if k == "slice":
        c = eval_expr(e[1], env, mods)
        mut lo = nil
        mut hi = nil
        mut st = nil
        if e[2] != nil:
            lo = eval_expr(e[2], env, mods)
        if e[3] != nil:
            hi = eval_expr(e[3], env, mods)
        if e[4] != nil:
            st = eval_expr(e[4], env, mods)
        if lo == nil and hi == nil and st == nil:
            return c[::]
        if lo == nil and st == nil:
            return c[:hi]
        if hi == nil and st == nil:
            return c[lo:]
        if st == nil:
            return c[lo:hi]
        if lo == nil and hi == nil:
            return c[::st]
        if lo == nil:
            return c[:hi:st]
        if hi == nil:
            return c[lo::st]
        return c[lo:hi:st]
    if k == "attr":
        o = eval_expr(e[1], env, mods)
        if type(o) == "tensor":
            return tensor_attr(o, e[2])
        return attr_get(o, e[2])
    if k == "call":
        f = eval_expr(e[1], env, mods)
        mut a = []
        for x in e[2]:
            a.append(eval_expr(x, env, mods))
        return do_call(f, a, mods)
    raise "bad expr " + k

fn do_call(f, args, mods):
    if is_userfn(f):
        params = f["params"]
        ndef = f.get("ndef", 0)
        le = env_new(f["closure"])
        if ndef == -1:
            fixed = params.len - 1
            if args.len < fixed:
                raise_err(f["name"] + " arity")
            mut i = 0
            while i < fixed:
                env_decl(le, params[i], args[i], true)
                i = i + 1
            env_decl(le, params[fixed], args[fixed:], true)
        else:
            if args.len < params.len - ndef or args.len > params.len:
                raise_err(f["name"] + " arity")
            mut i = 0
            while i < params.len:
                v = args[i] if i < args.len else nil
                env_decl(le, params[i], v, true)
                i = i + 1
        r = exec_block(f["body"], le, mods)
        if is_sig(r, "ret"):
            return r[1]
        if is_sig(r, "err"):
            raise_err(r[1])
        if is_sig(r, "brk"):
            raise_err("break outside loop")
        if is_sig(r, "cnt"):
            raise_err("continue outside loop")
        return nil
    if is_bound(f):
        mut a2 = [f["self"]]
        for x in args:
            a2.append(x)
        return do_call(f["fn"], a2, mods)
    if is_bltin(f):
        b = f["__bltin__"]
        if b == "print":
            return do_print(args)
        if b == "set":
            if args.len == 0:
                return set()
            return set(args[0])
        if b == "input":
            if args.len == 0:
                return input()
            return input(args[0])
        if b == "__struct":
            return {"__ttype__": true, "name": args[0], "fields": args[1], "m": args[2]}
        if b == "__new":
            tp = args[0]
            return mk_rec(tp["name"], tp["fields"], args[1:], tp["m"])
        if b == "load":
            # module nạp lúc chạy -> bản ghi "module" (global top-level, bỏ "$"/"__type")
            if args.len != 1 or type(args[0]) != "str":
                raise_err("load(name str)")
            if args[0] == "__sys":
                hs = load("__sys")
                ks = ["clock", "time", "sleep", "getenv", "setenv", "platform", "exepath", "getcwd", "chdir", "abspath", "exists", "isfile", "isdir", "mtime", "mkdir", "remove", "listdir", "mktemp", "sha256", "run", "stdin_line", "stdin_read", "stdout_write", "stderr_write", "zip_pack", "zip_unpack", "exit"]
                mut vs = []
                for k in ks:
                    vs.append(hs[k])
                return mk_rec("module", ks, vs, {})
            m = load_mod(args[0], mods)
            mut ks = []
            mut vs = []
            sl = m["env"]["slots"]
            for k in sl:
                if "$" not in k and not k.endswith("__type") and not is_module(sl[k]):
                    ks.append(k)
                    vs.append(sl[k])
            r = mk_rec("module", ks, vs, {})
            r["menv"] = m["env"]
            r["keys"] = ks
            return r
        raise_err("bad bltin value")
    if is_module(f):
        raise_err("module not callable")
    return call_host(f, args)

# --- import ---
fn dotted_path(dotted):
    return "/".join(dotted.split(".")) + ".t"

fn mod_leaf(dotted):
    parts = dotted.split(".")
    return parts[parts.len - 1] + ".t"

fn mod_stem(dotted):
    parts = dotted.split(".")
    return parts[parts.len - 1]

fn try_open(path):
    try:
        return open(path, "r")
    except:
        return nil

fn try_load(path, mods):
    fh = try_open(path)
    if fh == nil:
        return nil
    content = read(fh)
    close(fh)
    toks = host.scan(content)
    st = [toks, 0]
    ast = host.parse_prog(st)
    menv = env_new(nil)
    r = exec_block(ast[1], menv, mods)
    if is_sig(r, "err"):
        raise_err(r[1])
    return menv

fn load_mod(dotted, mods):
    if dict_has(mods, dotted):
        return mods[dotted]
    rel = dotted_path(dotted)
    leaf = mod_leaf(dotted)
    stem = mod_stem(dotted)
    mut menv = try_load(rel, mods)
    if menv == nil:
        menv = try_load("std/" + leaf, mods)
    if menv == nil:
        menv = try_load("self/" + leaf, mods)
    if menv == nil:
        menv = try_load("tools/" + leaf, mods)
    if menv == nil:
        menv = try_load("tpkgs/" + stem + "/" + stem + ".t", mods)
    if menv == nil:
        raise "no module " + dotted
    m = mk_module(dotted, menv)
    mods[dotted] = m
    return m

# --- stmt/block ---
fn exec_stmt(s, env, mods):
    k = s[0]
    if k == "line":
        return ["ok"]
    if k == "decl":
        v = eval_expr(s[3], env, mods)
        env_decl(env, s[2], v, s[1] == "mut")
        return ["ok"]
    if k == "assign":
        t = s[1]
        v = eval_expr(s[2], env, mods)
        if t[0] == "var":
            if env_has(env, t[1]):
                env_set(env, t[1], v)
            else:
                env_decl(env, t[1], v, true)
            return ["ok"]
        c = eval_expr(t[1], env, mods)
        mut ix = nil
        if t[0] == "attr":
            ix = t[2]
        else:
            ix = eval_expr(t[2], env, mods)
        if is_rec(c):
            if t[0] == "idx" and not dict_has(c["f"], ix) and dict_has(c["m"], "__setitem__"):
                do_call(c["m"]["__setitem__"], [c, ix, v], mods)
            else:
                rec_set(c, ix, v)
        elif t[0] == "attr" and type(c) != "dict":
            raise "assign here"
        else:
            c[ix] = v
        return ["ok"]
    if k == "fn":
        uf = mk_userfn(s[1], s[2], s[3], env)
        if s.len > 4:
            uf["ndef"] = s[4]
        env_decl(env, s[1], uf, false)
        return ["ok"]
    if k == "ret":
        return ["ret", eval_expr(s[1], env, mods)]
    if k == "print":
        mut a = []
        for x in s[1]:
            a.append(eval_expr(x, env, mods))
        do_print(a)
        return ["ok"]
    if k == "expr":
        eval_expr(s[1], env, mods)
        return ["ok"]
    if k == "if":
        if eval_expr(s[1], env, mods):
            return exec_block(s[2], env_new(env), mods)
        if s[3] != nil:
            return exec_block(s[3], env_new(env), mods)
        return ["ok"]
    if k == "for":
        mut it = eval_expr(s[2], env, mods)
        if is_rec(it) and dict_has(it["m"], "__iter__"):
            it = do_call(it["m"]["__iter__"], [it], mods)
        if type(it) == "dict":
            it = it.keys()
        for v in it:
            le = env_new(env)
            env_decl(le, s[1], v, true)
            r = exec_block(s[3], le, mods)
            if is_sig(r, "ret") or is_sig(r, "err"):
                return r
            if is_sig(r, "brk"):
                break
        return ["ok"]
    if k == "while":
        while eval_expr(s[1], env, mods):
            r = exec_block(s[2], env_new(env), mods)
            if is_sig(r, "ret") or is_sig(r, "err"):
                return r
            if is_sig(r, "brk"):
                break
        return ["ok"]
    if k == "break":
        return ["brk"]
    if k == "continue":
        return ["cnt"]
    if k == "raise":
        return ["err", bltin_str(eval_expr(s[1], env, mods))]
    if k == "assert":
        if eval_expr(s[1], env, mods):
            return ["ok"]
        mut d = "assert failed"
        if s[2] != nil:
            d = "assert failed: " + bltin_str(eval_expr(s[2], env, mods))
        return ["err", d]
    if k == "try":
        r = exec_block(s[1], env_new(env), mods)
        if not is_sig(r, "err"):
            return r
        msg = r[1]
        for h in s[2]:
            nm = h[0]
            if nm == nil or exc_match(msg, nm):
                le = env_new(env)
                env_decl(le, "err", msg, true)
                return exec_block(h[1], le, mods)
        return r
    if k == "import":
        m = load_mod(s[1], mods)
        nm = s[2]
        if nm == nil:
            parts = s[1].split(".")
            nm = parts[parts.len - 1]
        env_decl(env, nm, m, false)
        return ["ok"]
    if k == "fromimport":
        m = load_mod(s[1], mods)
        for n in s[2]:
            if not env_has(m["env"], n):
                return ["err", "cannot import " + n + " from " + s[1]]
            env_decl(env, n, env_get(m["env"], n), false)
        return ["ok"]
    return ["err", "bad stmt " + k]

fn exec_block(b, env, mods):
    mut i = 0
    while i < b.len:
        mut r = nil
        mut ok = true
        try:
            r = exec_stmt(b[i], env, mods)
        except:
            r = ["err", strip_terr(err)]
            ok = true
        if is_sig(r, "ret") or is_sig(r, "err") or is_sig(r, "brk") or is_sig(r, "cnt"):
            return r
        i = i + 1
    return ["ok"]

# --- entry ---
fn run_src(src):
    mods = {}
    toks = host.scan(src)
    st = [toks, 0]
    ast = host.parse_prog(st)
    env = env_new(nil)
    r = exec_block(ast[1], env, mods)
    if is_sig(r, "err"):
        print("error: " + r[1])
    return r

