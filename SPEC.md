# Ngôn ngữ T — Đặc tả thiết kế v2.2 (xem §21 cho phần mới; các mục cũ giữ làm lịch sử)

> Mục tiêu: nhanh tối đa (đánh bại Python/Ruby, tiệm cận C), syntax đơn giản như
> Python/Go, an toàn, AI-first, chạy trực tiếp không qua giai đoạn compile riêng.

## 1. Lối đi mới: "T-Model" (Single-Pass Direct Execution + Adaptive Hot-Path Native Caching)

Thay vì pipeline cổ điển (source → AST → IR → machine code), T dùng:

- **Cold code**: tree-walking interpreter thực thi AST trực tiếp, zero startup
  cost, không compile, không bytecode.
- **Hot code**: bộ đếm thực thi trên mỗi hàm. Vượt ngưỡng N lần gọi cùng kiểu đối
  số → đánh dấu specialize + inline cache (v0.1, trong interpreter); v0.2 sẽ
  biên dịch hàm đó sang machine code x86-64 ngay trong bộ nhớ (embedded
  mini-backend chỉ cho số học + tensor, KHÔNG dùng LLVM), patch con trỏ hàm.
- **Tensor là kiểu nguyên thủy**: n-chiều (1..8) float64 contiguous, phép toán
  vector hóa + broadcast (v2.2, §21.4).

## 2. Syntax (Python/Go-like, thụt lề)

```
# comment
let x = 10            # immutable
mut y = 20            # mutable
fn add(a, b):         # type suy luận
    return a + b
if x > 5:
    print("lon")
else:
    print("nho")
for i in range(10):
    print(i)
while y > 0:
    y = y - 1
t = tensor([1, 2, 3])
print(t + t)          # [2, 4, 6]
print(t @ t)          # dot: 14
```

Keywords: `let mut fn return if else elif for in while break continue
true false nil and or not print tensor range len str int float type
import from as try except raise assert`

## v0.9 mới: assert, formatter, escape/try cứng cáp

```
assert 1 + 1 == 2
assert xs.len == 3, "ghi chú khi fail"  # fail -> TError, bắt được bằng try
import fmt
out = fmt.fmt_src(src)      # indent 4, rstrip, collapse blank, giữ comment
fmt.fmt_file("a.t", "a.t")  # idempotent + giữ nguyên AST
```

- `try/except` bắt mọi exception (kể cả `ValueError` từ `int()`/index), giữ
  tên type để match; `return/break/continue` xuyên qua `try` (không bị nuốt).
- Escape string đủ `\\n \\t \\r \\\\` ở cả `t.py` lẫn `self/lex.t` (trước đây
  self-host biến `"\\r"` thành `"r"`, từng làm nhánh space nuốt ký tự `r`).
- `sys.setrecursionlimit(10000)` cho tree-walking lồng (eval-chạy-eval).
- `load_mod` tìm thêm `tools/` (cho `import fmt`).

## v0.2 mới: module, exception, attribute, builtins

```
import ai                 # nạp std/ai.t hoặc ./ai.t (guard circular)
import ai as x            # alias
from ai import relu, mse  # nạp tên trực tiếp
mod.fn()                  # truy cập qua dấu chấm

try:                      # bắt lỗi theo tên (match chuỗi lỗi)
    risky()
except boom:
    print("caught: " + err)   # err chứa message
raise "boom-xyz"          # ném lỗi (chuỗi)
```

- Builtins mới: `str() int() float() type()`.
- Attribute: str `.upper .lower .split .join .strip .len`,
  list `.append .pop .len`, tensor `.sum .mean .max .min .len`,
  dict `.keys .values .get .len`.
- Lỗi runtime (chia 0, index sai, module thiếu...) là `TError` với message rõ.

## v0.3 mới: power, dict, slice, file I/O

```
print(2 ** 10)        # 1024, kết hợp phải: 2 ** 3 ** 2 = 512
d = {"a": 1}          # dict literal, key int/float/str/bool/nil
d["c"] = 30           # gán key mới
for k in d: ...       # duyệt keys
s[1:3] s[2:] s[::2]   # slice str/list/tensor, đầu mở được
f = open("a.txt", "w")
write(f, "hi")
close(f)
f = open("a.txt", "r")
print(read(f))        # open/read/write/close builtins
```

## v0.4–v0.6: thoát Python (self-host + VM C + JIT)

- **v0.4 self-hosting** (`self/lex.t`, `self/parse.t`, `self/parse2.t`): lexer +
  parser của T viết bằng chính T (tokenize, indent_scan, recursive-descent đủ
  các cấp op, hỗ trợ `**` kết hợp phải). Builtins mới phục vụ self-host:
  `ord() chr() argv[]`, escape `\n \t \r` trong string literal.
- **v0.5 bytecode VM bằng C** (`vm/tvm.h`, `vm/tvm.c`, `vm/trun.c`,
  `vm/tcomp.py`): stack machine, value tagged union (nil/bool/int/float/
  str/list/tensor/fn), refcount, opcodes `OP_*`, file `.tb` (magic `TB01`).
  `tcomp.py` là compiler tạm (T→.tb); VM chạy độc lập Python sau khi build.
  Build: `cl vm/tvm.c vm/trun.c` → `trun prog.tb`.
- **v0.6 hot-path native JIT** (`vm/jit_x64.c`): đếm `OP_CALL` mỗi hàm
  (`hot[256]`); vượt 50 lần + shape số học → emit machine code x86-64
  (addsd/mulsd) vào `VirtualAlloc(XRW)`, patch con trỏ native, lần sau gọi
  thẳng — đúng T-Model "cold tree-walking + hot native", không LLVM.

## 3. Type system: gradual, specialize tại lần gọi đầu

- Tag: `int | float | str | bool | nil | list | tensor | fn`.
- Gọi hàm lần đầu với kiểu cụ thể → specialize; lần sau cùng kiểu chạy đường
  nhanh (inline cache). Không báo lỗi kiểu tĩnh — đơn giản như Python.
- `tensor`: n-chiều float64 (v2.2). `+ - * / **` element-wise có broadcast, `@` nhân ma trận.

## 4. Memory: an toàn không GC phức tạp

- CPython refcount thu hồi tức thì; không con trỏ trần ở tầng T.
- `let` immutable mặc định → giảm aliasing bug; muốn đổi phải `mut`.

## 5. Semantics

- `int` 64-bit, trộn int+float → float. `/` luôn float. Chia/mod cho 0 và
  index sai → `TError` với thông điệp rõ ràng (thay vì crash traceback).
- Hàm first-class, closure + đệ quy qua lexical scope chain.
- `==` so giá trị; `and/or` ngắn mạch; `not` phủ định.
- `range(n)` → 0..n-1; `range(a,b)` → a..b-1. `len()` cho str/list/tensor.
- Index `a[0]` (out-of-range → lỗi rõ ràng). Gán trần `x = ...` khi biến chưa tồn
  tại sẽ tự khai báo mutable (tương thích code viết nhanh kiểu Python).

## 6. Toolchain v0.1

```
t\
  SPEC.md  t.py  README.md
  std\ai.t        # zeros ones dot matmul relu mse grad số (viết bằng T)
  tests\*.t       # test suite
  bench\          # so sánh T vs Python
```

Chạy: `python t.py prog.t` · REPL: `python t.py`

## 7. Lộ trình

- v0.1: tree-walking + inline cache + tensor + REPL/tests (hiện tại).
- v0.2: hot-path counter + embedded x86-64 backend cho hàm số học.
- v0.3: tensor 2-D, SIMD, arena cho tensor, FFI Python/C.
- v1.0: LSP, formatter, package manager.

## 8. v0.7 đã verify (2026-09-26) — sửa để "hoàn chỉnh" chạy được
- Lexer/parser: `lex2` giữ `{}` dict, `indent_scan` string/bracket-aware,
  parser `elif*` chain, `and/or` ngắn mạch (giữ `j < n and src[j]` an toàn).
- Runtime: tensor `**`, methods `upper/lower/strip/sum/mean/max/min/keys/values`
  dạng callable (giữ `len` là property), `load_mod` tìm `std/|self/` + cwd + parent.
- Self-host: `lex.t` (+`->`), `parse.t` (indent_scan port từ Python),
  `parse2.t` full (not/unary/slice/dict/builtins/import-dots/from),
  mới `host.t` (`scan`+`parse_prog`) — xem STATUS.md.
- VM: `tvm.h` +`OP_POP`, `tvm.c` chạy số học/print/vars, `tcomp.py` opcode
  khớp header (straight-line); build C cần MSVC: `cl vm/tvm.c vm/trun.c`.

## 9. v0.8 self-host full (2026-09-26) — evaluator bằng chính T

- Mới `self/eval.t` (~500 dòng T): env chain (slots/mut/up), `eval_expr` mọi
  node (`num/str/lit/var/list/dict/tensor/bltin/un/bin/idx/slice/attr/call`),
  `exec_stmt/exec_block` mọi stmt (`decl/assign/fn/ret/print/expr/if/for/
  while/break/continue/raise/try/import/fromimport`), tín hiệu
  `ok/ret/brk/cnt/err`, lỗi user qua `["err", msg]`, lỗi host bọc tiền tố
  `__terr__:` bóc ở biên stmt, `and/or` ngắn mạch, `load_mod` đọc file +
  `host.scan/parse_prog` (tìm `./`, `std/`, `self/`), guard circular.
- Giá trị ánh xạ trực tiếp sang host nên số học/index/slice ủy thác cho host
  (đúng ngữ nghĩa t.py); userfn/module là dict tag `__tfn__/__tmod__`.
- `tests/selfhost.t` xanh; cả 4 suite chạy qua `eval.run_src` cho output
  đồng nhất với `t.py`. `.py` giờ chỉ còn là thang bootstrap.

## 10. v0.9 bổ sung (2026-09-26) — assert + fmt + nest-3 cứng
- `assert cond[, msg]` trong `t.py` + `lex/parse2/eval` self-host.
- `tools/fmt.t` formatter thuần T (dùng `parse.strip_comment/bracket_delta`).
- `is_userfn/is_module/is_bltin` trong `eval.t` đọc tag trực tiếp (`tag_get`
  + `try v[k]`), không qua `type()` — vì `type()` dối ở tầng eval lồng
  (đã gây `tensor has no .dot` ở nest 3). Xem STATUS.md.
- `t.py` `except Exception` (thay vì chỉ `TError/RuntimeError`) nhưng
  re-raise `Ret/Brk/Cnt` — từng nuốt `return` trong `try` (lòi qua `tag_get`).

## 11. v1.0 thoát Python (2026-09-26) — C VM + tcomp full + bundler

- `vm/tvm.h` (~50 opcode) + `vm/tvm.c` (~1450 dòng): stack machine, frame
  call (args/locals/arity), refcount, try qua `setjmp/longjmp` (khôi phục
  sp/frame/pc, đẩy errmsg), natives: số học int/float (+tensor broadcast,
  `/` float, `%`, `**`, `@`), str/list/tensor/dict/file methods, range/len/
  type/open/read/write/close/ord/chr/argv/print, so sánh sâu cho list/
  tensor/dict, slice kiểu Python (âm/step), `for` qua `FOR_PREP`.
- `vm/tcomp.py`: compiler full (mọi node; scope block đúng kể cả top-level;
  fn top pre-alloc cho đệ quy/tương hỗ; `and/or` ngắn mạch bằng `DUP`+jump;
  `for` desugar, `try` + `CONTAINS` match chuỗi, `assert`) + **bundler**
  (`-b`): inline import với prefix `mod__`, viết lại `mod.fn`, cross-unit
  fallback cho helper dùng chung (vd `parse_stmt` trong `parse.t`).
- `.tb`: `TB01` + u16 nglobals + u32 len + code; 15 global đầu là natives.
- Verify: 7 suite `t.py` vs `trun` khớp byte-for-byte; `tdriver.tb`
  (bundle eval) chạy demo/mods/core/v03/assert/selfhost bằng pure C.
- Giới hạn: closure chỉ global; fn lồng tham chiếu nhau phải định nghĩa
  trước; `import` runtime chỉ qua bundle.

## 12. v1.1 JIT hot-path móc vào vòng chạy (2026-09-26)

- `TFn`: `ncalls/tried/nat_kind/nat_op/nat_t0/nat_t1/native`; `OP_CALL` đếm,
  đạt 50 thì `t_jit_maybe` scan shape `LOAD0,LOAD1,BINOP,RET` (+,-,*,/;
  `/` chỉ double theo chuẩn `t.py`), speculate kiểu arg, emit x86-64
  (Windows ABI: int qua RCX/RDX/RAX, double qua XMM0/XMM1) vào VirtualAlloc.
- Mỗi lần gọi check tag strict `(t0,t1)`; lệch kiểu/div-0 → deopt về
  interpreter (giữ `type(1+2)=="int"`, `type(1.5+2)=="float"`).
  Tắt bằng `TVM_NOJIT=1`. Non-Windows: stub (không JIT).
- `tests/jit.t`: int/float/deopt/div/fn-thường đều đúng, khớp `t.py`.
- Microbench 2M lần gọi `add`: JIT 0.35s vs tắt 0.47s (~1.3x; vòng lặp
  chính vẫn bytecode nên Amdahl — inline loop là việc tiếp theo).

## 13. v1.2 closure thật (upvalue cell, 2026-09-26)

- `T_CELL` (hộp giữ 1 `TVal` + refcount); `LOAD/STORE/LOADG/STOREG` xuyên
  cell (đọc thay giá trị, ghi thay vào trong); `DECL/BIND/BINDG` thay mới.
- `TFn.upvals` (mảng cell) + `TFrame.fn`; `MKFN` thêm u16 nupvals;
  `CAPLOCAL` (box frame-slot tại chỗ + share), `CAPG` (box global),
  `CAPUP` (share upval cha — truyền dẫn nhiều tầng), `LOADU/STOREU`.
- `tcomp`: free-var analysis khi biên dịch thân fn (ghi `uval` + chuỗi);
  capture lười tại `MKFN` (đúng thứ tự LOADU); biến lặp `for` gán bằng
  `BIND/BINDG` (tươi mỗi vòng → capture đúng từng vòng, khớp `t.py`);
  scope vòng-lặp gắn cờ `iter` (fn lồng capture biến trong đó thì snapshot
  qua `CAPG`, biến main thường vẫn share global).
- `tests/closure.t`: counter, factory, lồng 3 tầng, shadow, global,
  2 closure share 1 cell, for-capture từng vòng, closure làm đối số.
- Giới hạn còn lại: capture/bắt phải bind trước khi định nghĩa fn.


## 14. v1.3 LSP tối thiểu (2026-09-26)

- `tools/tlsp.py`: server JSON-RPC/stdio (`Content-Length` framing):
  `initialize` (capabilities + `serverInfo`), `didOpen/didChange` (lưu bản
  + `publishDiagnostics` từ `t.py` parse, dòng kẹp trong file), `hover`
  (43 keyword/builtin + biến-hàm tự định nghĩa), `definition` (hit gần nhất
  cùng/trước dòng), `completion` (keyword + tên file), `documentSymbol`
  (fn/var/param), `shutdown`/`exit` sạch, method lạ → `-32601`.
- `tools/test_lsp.py`: phiên scripted qua subprocess (20 assert:
  init/caps/diagnostics/hover/definition/completion/symbols/error/shutdown).
- Symbol không cần AST line (phân tích dòng: `fn`/`let`/`mut`/`for`/param).

## 15. v1.4 package manager + UNDEF + JIT-cmp + for-range (2026-09-26)

- `tools/tpkg.py`: `init/install/run/test/pack`, manifest `t.json`
  (`deps` đường dẫn local), quy ước `tpkgs/<ten>/<ten>.t`; cả 3 loader
  (`t.py`, `eval.t`, `tcomp`) đều tìm `tpkgs/`.
- `T_UNDEF`: slot/frame/global khởi tạo UNDEF; đọc (`LOAD/LOADG/LOADU`)
  chưa bind → lỗi `undef variable` (khớp `t.py`, hết nil lặng lẽ).
- `tcomp` prescan 2 lượt (cây scope dùng chung): bind sau vẫn biên dịch
  (đọc trước bind → lỗi runtime đúng chỗ); capture-trước-bind đúng nhờ
  `DECL` xuyên cell; fn lồng tương hỗ nhau.
- JIT thêm so sánh i64 (`cmp/setcc`, kind 3) và `%` i64 (`idiv` qua R8,
  deopt khi y==0/MIN%-1); so sánh f64 bỏ qua (NaN).
- `for v in range(a[,b])` desugar thành while + `int()` (không cấp list),
  `hi` hoist 1 lần — khớp `t.py` kể cả float/side-effect.

## 16. v1.5 f-string + tpkg URL (2026-09-26)

- f-string f"...": {expr} long bracket, {{ }}, escape nhu string.
- t.py lex_fstr + parse con; self/lex.t scan_fstr; parse2 rap AST;
  eval.t noi bltin_str; tcomp literal + str() + ADD (C dung san).
- tpkg fetch <url> <dir> + dep URL (http/file, timeout 60, sha lock).
- tests/fstr.t khop 3 runtime.

## 17. v1.6 GC chu trình + frame-reuse + registry + LSP-full (2026-09-26)

- GC mark-sweep trong tvm.c (LIST/DICT/FN/NATIVE, intrusive doubly-linked,
  trigger ở loop-top, roots stack/frame/global/upval); bltin mem() 3 runtime
  (-> {objs, gc}); tests/mem.t. Chương trình 80k chu trình: 2 runs, bounded.
- Frame-locals tái dùng (lcap, UNDEF fill): fib 0.045s -> 0.023s, jbench +1.8x.
- Registry tools/tpkg_reg.py (index.json + static) + tpm search/install tên
  (@version, sha lock) qua TPKG_REG; fixture http://127.0.0.1 xanh.
- LSP references/rename/workspace-symbol (27 assert); f-string quote lồng
  cùng loại vốn đã đúng (thêm test); eval capture-trước-bind vốn đã đúng.


## 18. v1.7 inliner hàm single-return (2026-09-26)

- Điều kiện: hàm top-level, 1 lệnh return chứa expr thuần (không if/loop/try),
  <=40 node, đúng arity, không đệ quy (chain), free-var trong topbinds/NAT.
- Prescan tìm điểm inline (inlinemap + temps $in trong scope hiện tại),
  emit thay CALL bằng tính args + body (params gán temps, scope tạm).
- TCOMP_NOINLINE=1 để tắt. Bảo toàn ngữ nghĩa: args tính 1 lần đúng thứ tự,
  global đọc lúc chạy, closure vẫn gọi thường.
- Kết quả: jbench hết CALL (xem disasm) nhưng vẫn ~0.2s (dispatch-bound) —
  trace-JIT vòng lặp là hướng tiếp theo.


## 19. v1.8 debugger + fmt CLI (2026-09-26)

- vm/tdbg.c: hook dau vong run (tru khi tat), breakpoint pc, step N chuan
  (sN dat N-1 vi op hien tai luon chay), stack/locals/globals/disasm,
  stdin pipe duoc; trun -d prog.tb; tools/test_dbg.py xanh.
- tools/ffmt.t: CLI cho fmt.t (argv cuoi, --check exit 1), chay ca tren
  t.py va trun (bundle); xoa parse.parse_prog chet (treo parse_stmt).


## 20. v1.9 superop + semver + POSIX JIT (2026-09-26)

- Peephole trong tcomp (NOP-fill, giu vi tri): LOAD,PUSH,ADD/SUB,STORE ->
  INC/DEC/ADDI (+INCB cho BIND bien lap); LOAD,LOAD,ADD/SUB,STORE -> ADDS/SUBS.
  Ham LOAD/STORE qua slot_get/set (forceinline). Kich hoat rong khap suite.
- Semver tpkg: @1.0.0, @^1.0, @~1.2, @>=1.2 (ver_tuple + satisfies, test pin/range).
- JIT 2 ABI: emitter tham so hoa (Win RCX/RDX, SysV RDI/RSI), mmap/mprotect
  tren POSIX (chua test thuc thi ngoai Windows).
- Perf: fib(20) 0.011s (JIT-cmp + frame-reuse), jbench 2M ~0.25s (dispatch-bound).

## 21. v2.2 (2026-10-13) — đặc tả phần mới

### 21.1 Từ khoá và cú pháp mới

Từ khoá thêm: `switch case default struct`. Toán tử thêm: `in`, `not in`,
`+= -= *= /= %=`, biểu thức điều kiện `a if c else b`.

```
stmt   := ... | 'for' NAME (',' NAME)* 'in' expr ':' suite
              | target ('=' | '+=' | '-=' | '*=' | '/=' | '%=') expr
              | 'switch' expr ':' IND ('case' expr (',' expr)* ':' suite)* ['default' ':' suite] DED
              | 'struct' NAME ':' IND (NAME (',' NAME)* | fn_def)* DED
target := NAME | postfix '[' expr ']' | postfix '.' NAME
expr   := or_expr ['if' or_expr 'else' expr]      # 'if' không đứng đầu dòng logic
```

Desugar (giống nhau ở `t.py`, `self/parse2.t`; compiler/evaluator không cần biết):

- `for a, b in xs: body` → `for $pN in xs: mut a = $pN[0]; mut b = $pN[1]; body`
- `x op= e` → `x = x op e` (đích `a[i]`/`o.f` tính biểu thức con 2 lần)
- `switch e: case v1, v2: A ... default: D` → `let $sN = e` + chuỗi
  `if $sN == v1 or $sN == v2: A elif ...: ... else: D`
- `struct P: x, y / fn m(self, ...)` → `fn P__m(self, ...)`,
  `let P__type = __struct("P", ["x", "y"], {"m": P__m})`,
  `fn P(x, y): return __new(P__type, x, y)`

Tên `$...` không lex được nên không đụng tên người dùng.

### 21.2 Độ ưu tiên (thấp → cao)

| Mức | Toán tử |
|---|---|
| 1 | `a if c else b` (kết hợp phải) |
| 2 | `or` |
| 3 | `and` |
| 4 | `not` |
| 5 | `== != < > <= >= in not in` |
| 6 | `+ -` |
| 7 | `* / % @` (**`@` từ v2.2 cùng mức `*`, như Python**) |
| 8 | `-` một ngôi |
| 9 | `**` (kết hợp phải) |
| 10 | gọi `f()`, chỉ số `a[i]`, lát cắt `a[i:j:k]`, thuộc tính `a.b` |

### 21.3 Ngữ nghĩa

- `x in c`: chuỗi con (cả hai là str); phần tử (list, so bằng `==` của T: bool khác
  số); key (dict/set); số trong tensor (mọi phần tử). Kiểu khác → lỗi `in: unsupported`.
- `%`: `a % b` có dấu theo `b` (floor, như Python) cho int và float, ở mọi runtime.
- Biến khai bằng `let` và tên hàm `fn` là bất biến: gán lại → lỗi
  `x immutable (use mut)` (t.py lúc chạy; compiler lúc dịch).
- `set`: giữ thứ tự chèn; in `{1, 2}` / `set()`; `type` → `"set"`; phần tử phải
  hashable (int/float/str/bool/nil).
- `struct`: record có field cố định (thứ tự khai báo); `p.f` đọc field, nếu không
  phải field thì tìm method (gắn `self`); `p.f = v` chỉ cho field có sẵn; `p["f"]`
  tương đương `p.f`; in `P{x: 1, y: 2}`; `type(p)` → `"P"`; `==` so tên kiểu + field.
- Builtin "mềm" (`set input zeros ones full arange eye rand randn seed sqrt exp log
  sin cos tanh floor ceil pi`) là global định sẵn: khai báo/gán cùng tên sẽ che.
- `print(a, b, ...)` tính hết đối số rồi mới in.

### 21.4 Tensor

- Giá trị: float64 row-major, `shape` 1..8 chiều; `len(t)` = `shape[0]`;
  `t.size` = số phần tử; truthy khi `size > 0`.
- Tạo: `tensor(list lồng nhau chữ nhật | tensor | số)`, `zeros/ones(shape)`,
  `full(shape, v)`, `arange(n)`/`arange(a, b[, step])`, `eye(n)`, `rand/randn(shape)`;
  `seed(n)`: splitmix64 (mặc định 42), `rand` = `(z >> 11) * 2^-53`, `randn` Box-Muller.
- Phép từng phần tử `+ - * / ** < > <= >=` (so sánh ra 1.0/0.0) với broadcast kiểu
  numpy; chia 0 theo IEEE (inf/nan, không lỗi). `==`/`!=` so toàn bộ (ra bool);
  từng phần tử: `.eq(x)`, `.ne(x)`.
- `@`: 1D·1D → số; 2D@2D; 2D@1D → 1D; 1D@2D → 1D; tích luỹ theo k tăng dần.
- Rút gọn `sum mean max min argmax argmin([axis])`: không `axis` → số (argmax → int);
  có `axis` (âm được) → tensor bỏ trục đó. Cộng tuần tự trái→phải.
- Chỉ số: `t[i]` trên 1-D → số; n-chiều → **bản sao** hàng (shape[1:]);
  `t[i] = số | tensor shape[1:]`; `t[a:b:c]` cắt theo trục 0; `for row in t`.
- Biến đổi: `reshape(...)` (một `-1`), `T`/`transpose()` (đảo mọi trục),
  `flatten() tolist() copy() clip(lo, hi) softmax()` (trục cuối) `item()`;
  từng phần tử `exp log sqrt tanh sin cos abs relu sigmoid floor ceil`
  (cũng là builtin cho số: `sqrt(2)`; `floor/ceil` của số ra int).
- Bảo đảm: VM C và `t.py` cho kết quả **trùng từng bit** trên cùng nền tảng
  (cùng libm); so với numpy sai số tương đối ≤ 1e-9 (`tools/test_tensor.py`).

### 21.6 v2.3

- Lambda: `fn(a, b): biểu_thức` (biểu thức một dòng). Comprehension:
  `[e for x in xs if c ...]`, `{k: v for ...}`. Desugar trong parser thành hàm ẩn
  `$lN` / `$cN()` đặt ngay trước câu lệnh chứa nó (bắt biến theo tham chiếu như
  closure); `$cN()` được gọi tại chỗ nên thứ tự tính và short-circuit giữ nguyên.
- Tensor: `t[i, j, ...]` (parse thành `t[[i, j]]`, `:` riêng = `nil` = cả trục);
  thiếu trục cuối = giữ nguyên; gán vùng nhận số hoặc tensor đúng shape.
- `.tb` có thể kèm phần `"LN01"` sau code: u16 số file + (u16 len + tên)* + u32 n +
  (u32 pc, u32 dòng, u16 file)*; VM dùng để in vị trí lỗi chưa bị bắt.
- JIT vòng lặp top-level: vùng [đầu vòng, sau JMP lùi); global trong vùng là "local
  giả" đọc/ghi thẳng `TVal` (guard tag; ô cũ phải vô hướng); nhảy ra ngoài vùng
  hoặc guard hỏng → deopt (đẩy stack ảo, đặt pc).
- Lát cắt nhiều trục: mỗi trục trong `t[...]` là số (chọn, bỏ trục), `nil`/`:`
  (cả trục) hoặc `[lo, hi, step]` (parse từ `lo:hi:step`, phần trống = `nil`;
  ngữ nghĩa slice của Python, kể cả âm và step âm). Kết quả là bản sao.
- `load(tên)`: builtin mềm (NAT 51). Trả bản ghi kiểu `"module"` với field = global
  top-level của module (bỏ tên chứa `$` và `X__type`), giá trị chụp lúc top-level
  chạy xong; nạp một lần theo đường dẫn, nạp vòng là lỗi. `.tb` kèm phần `"EX01"`
  sau `LN01`: u16 n + (u16 len + tên + u16 slot)*. VM: mỗi hàm nhớ prog của nó,
  gọi/return/try đổi prog hiện hành; global của module là gốc GC; JIT không gọi
  thẳng hàm thuộc prog khác.

### 21.5 VM / JIT

- Opcode thêm: `IN`, `CALLM argc name` (gọi method, inline cache theo pc),
  `SETATTR name`. Native thêm: 21 (tổng 51). Bảng opcode có static assert.
- JIT (§README): hàm số học được đặc hoá theo kiểu đối số sau 50 lần gọi; OSR
  sau 2000 vòng lặp. Entry chỉ cho hàm thuần (đọc list được) → guard hỏng thì huỷ
  và interpreter chạy lại. OSR được ghi phần tử list → guard hỏng thì deopt: native
  chép frame ra `ctx->save`, VM ghi locals + đẩy stack ảo rồi chạy tiếp tại pc của
  lệnh hỏng. Biến môi trường: `TVM_NOJIT`, `TVM_JITLOG`, `TVM_JITDUMP`, `TVM_GCTHRESH`.

