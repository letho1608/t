# Ngôn ngữ T — Báo cáo trạng thái v3.0 (2026-10-27)

> Vị trí: `D:\Code\t`. Trạng thái: **`.\trun.exe tools\run_all.t` → ALL-GREEN** trên
> Windows (MSVC 19.44): 35/35 bộ kiểm thử (7 subsystem + 28 `tests/*.t`), hoàn toàn tự thân
> thuần `.t` (0 file `.py` trong compiler/runtime/tools), hoàn tất G1–G12 JIT x86-64,
> bảo mật bytecode/process/archive, tham số mặc định, thư viện chuẩn đầy đủ (`math`, `json`, `os`, `path`, `time`, `autograd`, `ai`).

## v3.0 — Tự thân hoàn toàn (`.t`), hoàn tất JIT G8/G9/G10/G12, chuẩn hóa bảo mật & stdlib

- **Loại bỏ hoàn toàn sự phụ thuộc Python**: Compiler (`self/tcomp.t` → `tcomp.tb`), REPL (`tools/repl.t` → `trepl.tb`), Formatter (`tools/ffmt.t`), LSP (`tools/tlsp.t`), Package Manager (`tools/tpkg.t`, `tools/tpkg_reg.t`), Test Suite (`tools/run_all.t`, `tools/test_*.t`) và Benchmark Runner (`bench/bench.t`) đều viết 100% bằng T và chạy trực tiếp trên `trun`.
- **Chạy trực tiếp file nguồn `.t` và `-c`**: `trun prog.t` và `trun -c "code"` tự động gọi `tcomp.tb` dịch ngầm và thực thi ngay trên VM + JIT.
- **Tham số mặc định**: `fn f(a, b = 10, c = "x"):` hỗ trợ trên cả VM C, compiler T và evaluator `self/eval.t`.
- **Hoàn tất JIT G8, G9, G10, G12 (`vm/jit_x64.c`)**:
  - **G8**: `OP_SETATTR` trong entry mode khi receiver là record cục bộ (`is_own_rec_t` hoặc `sv == 1`) hoặc không thể abort sau điểm ghi (kèm guard tag toàn bộ field trước khi ghi).
  - **G9**: Hàm đệ quy trả về record (`compile_entry` đăng ký trước các struct shape vào `j.ops` và suy đoán kiểu trả về `own_of_shape(s)` / `bor_of_shape(s)`).
  - **G10**: Cờ sở hữu động `OWN(jc, i)` ngay trên stack frame native thay thế `nodec` tĩnh khi biến cục bộ nâng cấp `borrowed -> owned` trong vòng lặp, giải phóng bộ nhớ tức thời mỗi vòng lặp và đồng bộ chính xác khi OSR deopt.
  - **G12**: Hỗ trợ `OP_STORE` chuỗi mượn `JT_S`, điều kiện `if d` trên dict, dict literal `{"k": v}` (`k > 0`), `d.has(k)`, `d.remove(k)`, chỉ số chuỗi `s[i]` (`tjit_str_idx`), và truyền chuỗi mới `JT_SN` làm đối số gọi hàm/method T.
- **Bảo mật hệ thống (`vm/tvm.c`)**:
  - `verify_bytecode`: Kiểm tra toàn vẹn bytecode `.tb` (magic, độ dài lệnh, phạm vi opcode, chỉ số global, bảng `LN01`/`EX01`, guard `OP_HALT`) trước khi thực thi.
  - Khởi tạo tiến trình con trực tiếp qua OS API (`CreateProcessA` trên Windows, `fork`/`execvp` trên POSIX) với mảng tham số độc lập, loại bỏ hoàn toàn `system()` (chống CWE-78 OS Command Injection).
  - Giải nén gói `.tpkg` kiểm tra chặt tên mục nén (`zip_safe_arcname`), chặn tuyệt đối `..`, đường dẫn tuyệt đối và ký tự ổ đĩa (chống CWE-22 Zip-Slip).

## v2.6 — JIT dict + chuỗi sở hữu được (G7) + G11

- Kiểu mới `JT_D`/`JT_DN` (dict mượn/mới), `JT_SN` (chuỗi mới, wrapper `TVal*`
  malloc giữ 1 ref — KHÔNG track GC nên mọi đường đều phải dec đúng hạn).
- `NEW_DICT {}` → helper `tjit_dict_new0` (tracked); `str(x)` → `tjit_str_of`
  (dùng chung `fmt_str`, infallible); nối chuỗi `ADD` → `tjit_str_cat` (tự nhả
  toán hạng owned trong helper theo mask, guard `a == b` chống double-free).
- `d[k]`/`d[k] = v` đoán int: 1 lần tìm (`tjit_dict_getD`/`tjit_dict_set`),
  thiếu key/khác int → deopt (interpreter báo lỗi/dùng default đúng ngữ nghĩa);
  `d.get(k, dflt-int)` tương tự; `SETIDX` OSR-only (ghi nhìn thấy được).
- Kỷ luật ownership: `LOAD` owned → alias mượn; `STORE`/`POP`/điều kiện/hậu
  `CALL` nhả temp owned; `STORE` dùng entry-compare (`[r12+8i]`) để lần ghi đầu
  giữ ref interpreter; epilogue `RET` nhả local chuỗi chết (trừ khi trả `JT_S`
  có thể alias); callee mượn (chuẩn hoá owned → borrowed, `SN` cấm làm đối số
  gọi hàm để tránh alias-UAF).
- Hạn chế có chủ đích (rớt về interpreter, vẫn đúng): dict literal k>0, đọc
  non-int, method dict khác `get`, key float/bool, `STORE` chuỗi mượn, `if`
  trên dict, method ghi field gọi từ vòng nóng.
- `tests/jitdict.t` mới (counter, key int, IDX/get/deopt, concat, self-concat,
  entry trả chuỗi, dict global mainmode, unhashable): 6 OSR + 1 entry + 1 main;
  bench dict 0.44× → 0.37× CPython (`--runs 3`).
- Ba bug G7 đã bắt bằng test đọc giá trị (không chỉ len): emitter `mov_r8_rax`
  sai REX, `get` ghi default đè receiver, `str_eq` trùng tên 2 đối/3 đối.
- Tối ưu v2.6.1: wrapper `TVal*` của chuỗi JIT (`str()`/nối/so sánh/`dec`) chuyển
  từ `malloc`/`free` CRT sang pool 32 byte (`tjit_wrap_free`) — bottleneck đo được
  trên `mb_str` (3M vòng: 0.36s → 0.14s); bench dict 0.37× → **0.23×** CPython.
- Ba tối ưu v2.6.1 tiếp theo (ALL-GREEN, GC dày khớp):
  1. **Cache `str(i)` số nhỏ** (`mk_str_int`, cache `-256..2047`, ref vĩnh viễn):
     `mb_str` 0.14 → 0.099s, `dict.t` 0.041 → 0.033s.
  2. **Counter fusion** — `tjit_dict_getD` ghi index thấy được (`tjit_last_i`),
     `tjit_dict_set` nhận hint (bit5 `kv`) và VALIDATE (`i < len && keys[i]==k`)
     rồi ghi thẳng; sai hint fallback `dict_find` (chậm, không bao giờ sai kết quả):
     `mb_dictint` 0.143 → 0.108s.
  3. **`key_eq_fast`** — so sánh khoá int/str inline né C-call `tval_eq` trong
     `dict_find` (lợi cả interpreter lẫn JIT, thay cho probe int-key viết tay trong
     asm chỉ lợi JIT + nhiều rủi ro): `mb_get` 0.077 → 0.053s, `mb_dictint`
     0.108 → 0.092s, `dict.t` → 0.031s (JIT ~2.2× / 7.8× CPython dict/loop).

## v2.5 — JIT tạo/gán record + chuỗi hằng (G6 một phần)

- `__new(Kiểu, v...)` (field toàn số) biên dịch sang helper `tjit_rec_new`:
  record mới sở hữu được (`JT_RN0+shape`, refc=1 + `gc_track`); callee luôn mượn
  đối số (chuẩn hoá owned → borrowed khi tra spec).
- `SETATTR` field số trong vùng OSR (guard tag cũ cùng kiểu → không refcount);
  `STORE` record mới vào local (nhả ref cũ nếu chắc chắn native-owned, cấm
  `nodec` cho local từng nâng borrowed → owned); `POP` record mới nhả ref;
  `return` record (owned move, mượn retain); deopt đồng bộ local owned (move),
  borrowed-record (free + retain), stack (move/retain theo sở hữu).
- Chuỗi hằng `PUSH_STR` mượn cache (không inc, cache sống suốt native);
  `==`/`!=` qua `tjit_str_eq`, `len()`, điều kiện `if s`/`while s`; global chuỗi
  đọc kèm guard tag (đổi kiểu → deopt, `tests/jitrec.t`: `mode = 7` vẫn đúng).
- `tests/jitrec.t` thêm khối G6 (constructor nóng, SETATTR, POP-owned,
  return-record, chuỗi, deopt-move khi OOB): JIT 5 entry/3 OSR → 10 entry/7 OSR;
  `bench/method.t` 0.48× → 0.22× CPython. GC dày (`TVM_GCTHRESH=1`) khớp byte.
- Còn lại ở interpreter: tạo dict, nối chuỗi, `s[i]`, `d[k]`/`d[k] = v`, method
  dict/list/str, closure/`try` trong vùng nóng, method ghi field gọi từ vòng
  nóng (entry thuần không side-effect nên `p.move(1)` trong loop vẫn interpreted).

## v2.4 — view theo bước, module tôn trọng mut, tối ưu toàn diện

## v2.4 — view theo bước, module tôn trọng mut, tối ưu toàn diện

- Tensor: mọi chỉ số/lát cắt (cột, bước ≠ 1, bước âm), `.T`, `reshape` của tensor
  liền là **view** như numpy (VM: cache dày + bộ đếm phiên bản của chủ bộ nhớ).
- `load()`: `m.X = v` báo `X immutable (use mut)` với global `let`/`fn`/`struct`
  (EX01 thêm cờ mut).
- Tensor: kernel từng phần tử tách `switch` khỏi vòng lặp, broadcast theo trục trong
  cùng, pool buffer, matmul đọc A theo bước + `n == 1` thành tích vô hướng, gcc bật
  vector hoá động. `bench/nn`: 0.55s → 0.20s (Windows), 0.52s → 0.32s (Linux).
- VM: pool khối nhỏ (GCNode, vỏ list/dict/native, cell, chuỗi ngắn); superop bỏ qua
  đệm NOP (trước đây 27% lệnh thực thi là NOP); push/pop inline; bỏ refcount cho
  số; `==` chuỗi nhanh; field struct trúng inline cache ngay trong `OP_ATTR`; in số
  nguyên không qua printf; chuỗi 1 byte dùng chung. Interpreter (không JIT)
  nhanh hơn 15–30%; dict 0.81× → 0.48×, method 0.71× → 0.48× CPython.
- Compiler T tự host: 0.95s → 0.43s khi dịch chính nó (keyword tra dict, `in` thay
  vòng lặp, bảng kích thước opcode, `be_bytes` đường nhanh, peephole kiểm opcode
  trước). Output byte-identical, fixed point giữ.
- `-DTVM_PROF`: profile số lệnh theo hàm T và theo opcode.
- JIT đọc field số + gọi method của struct (record mượn, guard kiểu/tag;
  `tests/jitrec.t`): code số học trên struct nhanh ~13× so với interpreter.
- Vòng dispatch giữ `pc` trong thanh ghi: loop 0.330 → 0.297s, sieve 0.513 → 0.447s
  (không JIT).

## v2.3 — số dòng lỗi runtime, lambda/comprehension, autograd, JIT top-level

- `python tools/run_all.py` → ALL-GREEN (Windows + Linux gcc 13); `test_tcomp_t` 25 suite,
  compiler Python == compiler T từng byte (kể cả bảng dòng `LN01`).
- Lỗi runtime: `T error: <msg>` + `  tại <file>:<dòng>` ở `trun` và `t.py`.
- Cú pháp: lambda `fn(x): e`, list/dict comprehension (nhiều `for`/`if`), tensor `t[i, j]`/`t[:, j]`.
- `std/autograd.t`: reverse-mode autodiff trên tensor (kiểm sai phân hữu hạn, MLP XOR).
- JIT vòng lặp top-level: `bench/loop.t` 0.377s → 0.033s (CPython 0.494s).
- Tensor lát cắt nhiều trục `t[0:2, 1:3]`, `t[::-1, 0]` (khớp numpy).
- `load("tên")`: import động lúc chạy (VM nạp `.tb` riêng, tự dịch `.t` khi cần;
  field module sống; `test_tcomp_t` 25 suite, eval.t 13 suite, kèm GC quét dày).
- Tensor view cho vùng chọn liền khối (`t[i]`, `t[a:b]`, `t[i, j:k]`): `bench/nn.t`
  0.550s → 0.51s. Git: mỗi tính năng một commit.
- View theo bước như numpy (cột, bước ≠ 1, bước âm, `.T`, `reshape`); `m.X = v` trên
  module tôn trọng `mut`.

## v2.2 — hoàn thiện mục còn thiếu + JIT mở rộng + tensor thật

Verify (2026-10-13):

```
python tools/run_all.py
PASS ops        OPS-OK (74 opcodes + 51 natives, 3 sources in sync)
PASS tcomp-t    20 suite; fixed point e4f49eec3745 (tcomp.tb 80532 byte);
                compiler Python == compiler T (20 suite, từng byte);
                eval.t trên VM khớp t.py (11 suite); gcstress với TVM_GCTHRESH=3
PASS dbg / lsp / tpkg (offline) / jit / tensor (38 phép khớp numpy)
ALL-GREEN
```

- **Ngôn ngữ**: `a if c else b`, `in`/`not in`, `for a, b in`, `+= -= *= /= %=`,
  `switch/case/default`, `struct` (field, method, `self`, gán `p.x`), `set`, method
  mới cho str/list/dict, số mũ, builtin "mềm" che được tên, lỗi cú pháp đúng dòng,
  `let`/`fn` bất biến kiểm cả ở compiler, `@` cùng độ ưu tiên với `*`.
- **Toolchain**: REPL cho `trun` (`trepl.tb`), compiler tự gộp module khi có
  `import`, LSP cập nhật, tpkg registry thư mục/offline + cache sha + `release`,
  `vm/build.sh` + `vm/build.bat` + CI GitHub Actions, verify POSIX thật.
- **JIT** (`vm/jit_x64.c`, viết lại): thông dịch trừu tượng suy kiểu int/float/bool/list,
  biên dịch cả hàm (vòng lặp, nhánh, đệ quy, gọi chéo, `int/float/abs/min/max/len`),
  đọc/ghi phần tử list có guard, OSR cho vòng lặp nóng; entry: huỷ + chạy lại
  (hàm thuần), OSR: deopt có chuyển trạng thái về frame interpreter; ngân sách stack 256KB.
- **Tensor** (`ttensor.py` + `vm/tensor_impl.h`): 1..8 chiều, broadcast, matmul,
  rút gọn theo trục, hàm toán học, softmax, so sánh từng phần tử, ngẫu nhiên
  splitmix64 tất định; VM và t.py khớp từng bit.
- **Hiệu năng VM**: dict băm, chuỗi 1 khối + cache hash, hằng chuỗi cache,
  đường nhanh số học/LOAD/STORE/IDX/SETIDX, `CALLM`/`SETATTR` + inline cache,
  kiểu struct dùng chung (record 2 lần cấp phát).

Bench (Windows, CPython 3.10, `python bench/bench.py --runs 3`):

| bench | T | CPython | T/Py |
|---|---:|---:|---:|
| fib | 0.020s | 0.351s | 0.06× |
| loop | 0.462s | 0.535s | 0.86× |
| loopfn | 0.056s | 0.797s | 0.07× |
| mandel | 0.038s | 0.412s | 0.09× |
| sieve (JIT list) | 0.100s | 0.475s | 0.21× |
| dict | 0.132s | 0.157s | 0.84× |
| method | 0.125s | 0.180s | 0.69× |
| matmul | 0.025s | 2.399s | 0.01× |
| nn (so numpy) | 0.568s | 0.753s | 0.75× |

Trước v2.2 (cùng máy): fib 1.06×, loop 1.25×, loopfn 1.69×, mandel 1.71×,
sieve 2.06×, dict 9.14×, method 4.18× (T chậm hơn CPython ở mọi bài).

Bug có sẵn phát hiện và sửa: xem `TODO.md` mục 1.5 (inliner sai kết quả,
handler try bị bỏ lại khi return, use-after-free trong callback, GC quét, slice
bước âm, `%` số âm, dict rỗng truthy, disassembler, build POSIX, eval.t sort...).

# Báo cáo v2.0 (2026-09-28, compiler thuần T tự host) — lịch sử

## 0. v2.0 — compiler T thuần T (`self/tcomp.t`)

`vm/tcomp.py` đã có một bản song song viết bằng chính ngôn ngữ T:
`self/tcomp.t` (~1460 dòng: lexer/parser dùng `self/host.t`, bundler,
codegen đầy đủ mọi node/stmt, emit `.tb`). `self/tcli.t` là wrapper CLI.

- **Một primitive C duy nhất**: `f64bits` (đóng gói double IEEE-754 chuẩn).
  Mọi phần còn lại — int 2/4/8 byte (two's complement tay), UTF-8 — thuần T.
- **Bootstrap một lần**: `python vm/tcomp.py -b self/tcli.t tcomp.tb`
  (72538 bytes). Từ đó **không còn Python**:
  `trun tcomp.tb [-b] prog.t prog.tb`.
- **Self-hosting fixed point**: `tcomp.tb` → `tcomp2.tb` → `tcomp3.tb`,
  stage 2 và 3 **byte-identical** (sha `a67efc335f3f`).
- **14/14 suite** `trun tcomp.tb` dịch rồi chạy → output khớp `python t.py`.
  Stage 1 và stage 2 cho ra `.tb` giống nhau từng byte.
- **UTF-8**: literal nhiều byte (Việt/Trung/emoji) cho ra bytecode giống nhau
  khi compiler chạy trên VM C (byte) và trên `t.py` (codepoint) — `c_str` tự
  dò backend rồi encode.
- **Khác biệt có chủ đích**: bản T không inliner/peephole nên bytecode dài hơn
  (`selfhost.t` 31010 vs 36021 byte) nhưng ngữ nghĩa/output giống hệt.
  (Từ v2.1 cả hai bản đều có inliner + peephole nên `.tb` trùng nhau.)
- Test vĩnh viễn: `python tools/test_tcomp_t.py` → `TCOMP-T-OK`.

Lỗi đã phát hiện và sửa trong lúc port (đều là bug thật, không workaround):

1. **Bundler: local của module đụng tên global của entry** — bundler chỉ đổi
   tên *hàm* và *global top-level* của unit phụ, còn biến cục bộ trong hàm giữ
   tên trần. `tests/fmtself.t` có global `a`/`b`/`src`/`out` trùng local của
   `host.t`/`parse.t` → `a = parse_suite(st)` trong `parse_if` ghi đè global `a`
   của entry, bytecode lệch ⇒ `fmt keeps AST` fail. Sửa: `local_ren()` đổi tên
   **mọi** bind (param, `decl`, gán, biến `for`) của hàm trong unit phụ, gộp với
   ren của hàm cha để upvalue vẫn đúng; key `fbound` phải mang prefix unit để
   hàm lồng nhau khớp. Sửa đồng thời ở `self/tcomp.t` và `vm/tcomp.py`.
2. **Block-scope của T**: biến gán *trong* nhánh `if/else` không tồn tại sau
   nhánh → `undef islocal` trong `c_fn`, `done` trong `c_try`. Sửa bằng cách
   trả về cặp giá trị từ hàm phụ (`fn_bind`, `has_bare`) thay vì gán trong nhánh.
3. **`be_bytes` số âm**: dùng `255 - b` (bitwise NOT) nhưng thiếu `+1`, và cộng
   carry đi ngược chiều → mọi jump lùi (`JMP`, `MKFN` entry) sai 1..n byte →
   `unknown opcode` / `pc overrun`. Sửa thành `(~v) + 1` với carry từ byte
   thấp lên cao. Test riêng: khớp `struct.pack` cho 1/2/4/8 byte, cả âm dương.
4. **`self/tcomp.t` thiếu `MATCHGLOB`** trong `opnames()` → lệch toàn bộ opcode
   từ 24 trở đi. Đã khoá lại bằng `tools/test_ops.py` (3 nguồn: `tvm.h`,
   `tcomp.py`, `self/tcomp.t`).
5. **`main_cli()` kiểm tra sai số tham số** (`< 3` thay vì `< 2`) và bị gọi cả
   khi `import tcomp` dưới `t.py` → tách CLI ra `self/tcli.t`.
6. **`build_from_cc` nối chuỗi O(n²)** (100KB+ byte) → dùng list + `join`.
7. **Closure không thấy biến khai báo sau hàm** (prescan 2 pha chỉ dựng global
   trước, không dựng local của hàm) → `tests/hoist.t` khoá hành vi: global khai
   báo sau hàm và mutual recursion OK; biến `for` khai báo *sau* hàm thì không
   capture (giới hạn nhất quán của cả 4 runtime).

## 0b. v2.1 — 13 builtin tập trên + 2 bug VM nghiêm trọng

**Builtins mới** (`abs round min max sum any all sort reversed enumerate zip
map filter`) — native của VM C (`NAT_NFIXED` 17 → 30), `t.py`, và `self/eval.t`
mô phỏng trong T; đồng bộ 3 nguồn khoá bằng `tools/test_ops.py`
(`OPS-OK (71 opcodes + 30 natives, 3 sources in sync)`), hành vi khoá bằng
`tests/builtins.t` (chạy trên cả 3 runtime).

- `map(f, xs)`, `filter(f, xs)`: VM C gọi lại hàm T qua `tvm_call_value()`
  (nested `tvm_run` + `vm->stop_nf` để dừng đúng ở `RET` của callee).
- `sort(xs[, key])`: sắp **tự sắp trong list** và trả lại chính nó; tham số 2 là
  **hàm khóa 1 tham số** (không phải comparator) — khớp `t.py` (`key=` của
  Python). Quicksort 3-way median-of-3, ổn định, O(n log n).
- `min/max` nhận iterable hoặc 2 số; `round` nửa ra xa số 0 (`round(2.5)=3`);
  chuỗi so sánh theo byte (số trước chuỗi).

**Bug VM (đều là bug thật, không workaround):**

1. **`open()` mất chữ `b` khi build `-O2`** → file mở ở *text mode* → CRLF
   translation → mọi byte `0x0A` trong `.tb` bị thành `0x0D 0x0A`, bytecode hỏng
   âm thầm (chỉ biểu hiện ở literal số có chữ số 0, vd `10` → `2573`). Nguyên
   nhân: buffer chứa mode khai trong block kết thúc **trước** `fopen` nên đời
   sống hết, optimizer tái dùng stack slot đó. Sửa: `rb`/`k` nâng lên phạm vi
   hàm.
2. **GCC ≥ 12 (mingw-w64) inline hỏng vòng lặp interpreter** vì có `setjmp`
   trong thân: build `-O1/-O2/-Os` → `try/except` **im lặng** (không in lỗi, exit
   0), chỉ `-O0` hoặc `-fno-inline` mới đúng. Sửa: `#pragma GCC
   optimize("no-inline")` ở đầu `vm/tvm.c` + ghi chú trong README.

**Quy trình build Windows (mới, xác minh):** không có GCC trên máy, dùng
Docker + mingw-w64:
`docker build -t tbuild -f Dockerfile.mingw .` rồi
`docker run --rm -v D:\Code\t:/src tbuild x86_64-w64-mingw32-gcc -O2 -o trun.exe vm/tvm.c vm/jit_x64.c vm/tdbg.c vm/trun.c -lm`.
Kết quả: `test_ops` / `test_lsp` / `test_dbg` / `test_tcomp_t` (14 suite +
fixed point `a67efc335f3f`) + `bench/bench.py` (fib(20) 0.216s) đều xanh.

## 1. Test suite — XANH HẾT (python t.py == trun.exe, 14 suite)

| File | Nội dung | Chạy? |
|---|---|---|
| `tests/demo.t` | hello, fib(15)=610, tensor +/@, for/while, first-class fn | ✅ pass |
| `tests/core.t` | số học, string, list, so sánh, logic, elif, while+break/continue, fact(10), tensor idx/gán | ✅ pass |
| `tests/mods.t` | import 2 dạng, try/except 2 case, builtins, methods | ✅ pass (sau fix load_mod + methods) |
| `tests/v03.t` | `**`, dict, slice, file I/O, softmax end-to-end (=1) | ✅ pass (sau fix lex2 dict) |
| `bench/bench.py` | T fib(20) ~0.2–0.4s vs Py ~0.002–0.004s (tree-walking, đúng kỳ vọng) | ✅ pass |
| `tests/selfhost.t` | lexer+parser+evaluator `.t` thuần (fib10=55, tensor, dict, try, import ai) | ✅ pass |
| `tests/assert.t` | assert + return/break xuyên try | ✅ pass |
| `tests/fmtself.t` | fmt idempotent + giữ AST (demo) | ✅ pass |
| `tests/jit.t` | hot-path native int/float/deopt/div (JIT OK) | ✅ pass |
| `tests/closure.t` | counter/factory/lồng-3/shadow/global/share-cell/for-capture | ✅ pass |
| `tests/fstr.t` | f-string lồng/method/dict/tensor/rỗng (FSTR OK) | ✅ pass |
| `tests/mem.t` | builtin mem() shape + chu trình (MEM OK) | ✅ pass |
| `tests/bytesem.t` | byte-string: len/ord/chr/slice/split, latin-1, file nhị phân | ✅ pass |
| `tests/hoist.t` | global khai báo sau hàm + mutual recursion | ✅ pass |
| `tests/builtins.t` | 13 builtin tập trên (map/filter/sort khoá/reversed/zip/enumerate…) | ✅ pass |
| `tools/test_lsp.py` | LSP scripted 27 assert | ✅ pass |
| `tools/test_dbg.py` | debugger scripted (step/break/stack/disasm) | ✅ pass |
| `tools/test_ops.py` | đồng bộ 71 opcode + 17 native trên 3 nguồn | ✅ pass |
| `tools/test_tcomp_t.py` | compiler T thuần: 14 suite + fixed point + UTF-8 | ✅ pass |
| `tpkg` flow | init/install/run/test/pack với dep local (81/27 qua cả 3 runtime) | ✅ pass |

## 1b. Self-host full v0.8 (mới, đã verify)
- `self/eval.t` (~500 dòng T): env chain, mọi node expr/stmt, tín hiệu
  `ok/ret/brk/cnt/err`, `__terr__:` wrap, `load_mod` (`./`, `std/`, `self/`).
- `demo/core/mods/v03` chạy qua `eval.run_src` → output **đồng nhất** `t.py`.
- 2 bug lúc viết eval đã fix: `tensor_attr`/`attr` trả lambda (không gọi eager);
  `d.keys()()` thừa ngoặc trong code `.t` (host `keys()` đã trả list).

## 1c. v0.9 bổ sung (đã verify)

- `assert cond[, msg]` + `tests/assert.t` (gồm regression return/break xuyên try).
- `tools/fmt.t` + `tests/fmtself.t` (idempotent + AST-equal trên demo).
- Bug nest-3 (eval^3) đã fix, 3 gốc:
  1. `lex.t` thiếu escape `\\r` (+`\\\\`) → self-host parse `"\\r"` thành `"r"`,
     nhánh space nuốt ký tự `r` (`return` → `eturn`). Fix escapes.
  2. `t.py` `except Exception` nuốt `Ret/Brk/Cnt` (`return` trong `try` mất).
     Fix: re-raise control-flow trước.
  3. `eval.t` `is_*` gọi `type()` dối ở tầng lồng (`tensor has no .dot`).
     Fix: `tag_get` đọc key trực tiếp (`try v[k]`), không qua `type()`.
- `sys.setrecursionlimit(10000)` cho tree-walking lồng; `load_mod` tìm `tools/`.
- Verify: 7 suite host xanh; assert/fmtself/selfhost + demo/core/mods/v03
  chạy qua `eval` (nest-2) xanh; assert/fmtself/selfhost chạy qua eval^2
  (nest-3) xanh, output đồng nhất.

## 1d. v1.0 thoát Python (đã verify bằng trun.exe build từ MSVC 19.44)

- Build: `cl /O2 /Fetrun.exe vm/tvm.c vm/trun.c` (sạch, chỉ còn TVM_TRACE tắt).
- `tcomp` thẳng cho demo/core/assert; `-b` cho mods/v03/selfhost/fmtself.
- 7/7 suite `t.py` vs `trun` khớp byte-for-byte (demo 7, core 22, mods 12,
  v03 25, assert 2, selfhost 12, fmtself 1 dòng).
- Escape: `tcomp -b tools/tdriver.t tdriver.tb` (27KB, 83 globals, 0 lỗi) rồi
  `trun tdriver.tb tests/{demo,mods,core,v03,assert,selfhost}.t` — pure C chạy
  evaluator `.t` chạy chương trình `.t`, output khớp `t.py` (kể cả fib(15)=610,
  `import ai` qua file I/O của C, softmax=1). Không còn tiến trình Python.
- fib(20) trực tiếp: `trun` 0.045s, đúng 6765 (CPython ~0.004s — VM v1 ưu tiên
  đúng trước nhanh sau: refcount + dispatch còn nặng).
- Bug VM đã fix khi verify (5 cái): try rơi vào RAISE (thiếu JMP cuối handler);
  `split` đếm argc sai + nhánh `argc==1`; SETIDX-dict free key 2 lần;
  assert `POP` thừa sau `JMPT`; opcode `PRINT` đếm nhầm trước khi chốt header.

## 1e. v1.1 JIT hot-path (đã verify bằng trun.exe build từ MSVC 19.44)

- `jit_x64.c` viết lại: emit `+,-,*` i64 (RAX/RCX/RDX) và `+,-,*,/` f64
  (XMM0/XMM1), scan shape `LOAD0,LOAD1,BINOP,RET`, speculate `(t0,t1)`,
  strict tag + deopt (giữ đúng kiểu int/float, div-0 báo lỗi như interpreter).
- `OP_CALL` móc đếm + `t_jit_maybe`/`t_jit_call`; `t_jit_free_fn` thu hồi
  VirtualAlloc; `TVM_NOJIT=1` để tắt; build gồm `vm/jit_x64.c`.
- `tests/jit.t` khớp `t.py` (int giữ int, float giữ float, đổi kiểu đúng).
- Microbench 2M gọi `add`: 0.35s (JIT) vs 0.47s (tắt) ~1.3x; 8/8 suite vẫn MATCH.

## 1f. v1.2 closure thật (đã verify bằng trun.exe build từ MSVC 19.44)

- `T_CELL` + `LOAD/STORE/LOADG/STOREG` xuyên cell; `TFn.upvals` + `TFrame.fn`;
  `CAPLOCAL/CAPG/CAPUP/LOADU/STOREU/BIND/BINDG` (khớp `tcomp`).
- `tcomp` free-var + capture lười + `BIND` biến lặp + scope `iter`.
- `tests/closure.t` khớp `t.py` (counter, factory, lồng 3, shadow, global
  g=200, share-cell, for-capture 0/10/20, apply); qua `eval` và escape cũng xanh.
- Bug đã fix khi verify (3 cái): assert-test sai do counter gọi 2 lần
  (test sai, không phải VM); `BIND` thiếu bản global (main NULL-deref);
  fn-lồng ở top share global thay vì snapshot (thêm `CAPG` + cờ `iter`).
- 9/9 suite `t.py` vs `trun` MATCH; escape (pure-C driver) xanh.

## 1h. v1.5 f-string + tpkg URL (đã verify)

- `f"..."`: `{expr}` lồng bracket, `{{`/`}}`, escape đầy đủ; `t.py`
  (`lex_fstr` + parse con), `self/lex.t` (`scan_fstr`), `parse2.t`,
  `eval.t`, `tcomp` (+bundler), C dùng sẵn — `tests/fstr.t` khớp 3 runtime.
- `tpkg fetch` + dep URL (`http/file`, sha lock); fixture `file://` xanh
  (install/run/test/pack + dep qua `t.py`/`eval`/`trun -b`: 81/27/144).

## 1i. v1.6 GC + frame-reuse + registry + LSP-full (đã verify)

- GC mark-sweep chu trình (intrusive doubly-linked, trigger loop-top):
  80k chu trình → 2 runs, objs bounded, không crash; `mem()` 3 runtime.
- Frame-locals tái dùng: fib(20) 0.045s→0.023s; jbench 2M calls 0.47s→0.27s
  (tắt JIT), 0.35s→0.20s (bật JIT).
- Registry + `tpm search/install tên@version` qua HTTP thật (fixture xanh).
- LSP references/rename/workspace-symbol: 27/27 assert xanh.
- 11/11 suite `t.py` vs `trun` MATCH (thêm mem).

## 1k. v1.9 superop + semver + POSIX JIT (đã verify)

- Peephole `tcomp` (NOP-fill): INC/DEC/ADDI/ADDS/SUBS/INCB; helper slot
  forceinline; kích hoạt rộng khắp suite, output khớp.
- Semver: `@1.0.0/@^/@~/@>=` (pin đúng, mismatch exit 1, range cài được).
- JIT 2 ABI (Win RCX/RDX, SysV RDI/RSI + mmap/mprotect; POSIX chưa chạy thật).
- Perf hiện tại: fib(20) 0.011s; jbench 2M ~0.25s (dispatch-bound).

## 1j. v1.7 inliner hàm single-return (đã verify)

- Điều kiện: top-level, 1 `return` thuần expr, ≤40 node, đúng arity,
  không đệ quy, free-var trong topbinds/NAT; `TCOMP_NOINLINE=1` để tắt.
- Prescan + `inlinemap` + temps `$in`; emit thay CALL bằng tính args + body.
- `tests/jit.t` thêm sq/sumsq/getsh; jbench hết CALL nhưng vẫn ~0.2s
  (dispatch-bound) — trace-JIT vòng lặp là hướng tiếp theo.

## 1g. v1.4 tpkg + UNDEF + JIT-cmp + for-range (đã verify)

- `tools/tpkg.py` (`init/install/run/test/pack`) + fixture dep local:
  install/run/test/pack xanh; dep chạy qua `t.py`, `eval`, `trun -b` (81/27).
- `T_UNDEF`: đọc trước bind → `undef variable` (hết nil lặng lẽ, khớp `t.py`).
- Prescan 2 lượt (cây scope dùng chung): capture-trước-bind (`g()` trước
  `let x` → 5), fn lồng tương hỗ (`f`/`h` → 42), `DECL` xuyên cell.
- JIT: so sánh i64 (`cmp/setcc`, kind 3), `%` i64 (`idiv` qua R8, deopt
  y==0/MIN%-1); `tests/jit.t` thêm cmp/mod, khớp `t.py`.
- `for v in range(a[,b])` desugar while + `int()` + `$hi` hoist (không list).
- Bug đã fix: `leaf` gồm `.t` (tpkgs sai đường dẫn); `cqo` phá RDX (giữ
  divisor ở R8); scope try-body lệch prescan/emit; assert `POP` thừa (cũ).

## 1m. v1.9 debugger + fmt CLI + glob + test_ops (đã verify)

- `trun -d` (vm/tdbg.c): step chuẩn từng op (sửa off-by-one), breakpoint pc,
  stack/locals/globals/disasm, stdin pipe; `tools/test_dbg.py` 6/6 xanh.
- `tools/ffmt.t` CLI (format tại chỗ + `--check`); chạy cả `t.py` lẫn `trun`;
  xóa `parse.parse_prog` chết (treo `parse_stmt`, chặn bundle).
- `except "pat*"` glob (`*`,`?`, byte-wise): `OP_MATCHGLOB` + matcher C,
  `t_match` (t.py), `glob_match` (eval.t); `except` nhận STR; vào `mods.t`.
- `tools/test_ops.py`: khóa sync 71 opcode + 16 natives (từng bắt lệch
  `MATCHGLOB` trùng tên); `vm/build.sh` POSIX; `TVM_INLINE` portable.

## 2. Bug đã fix ở v0.7 (so với v0.6)

1. **lex2 nuốt `{}` dict** — bản cũ replace string `<IND>`→`: {` rồi đổi mọi `}` thành
   DED, khiến `{"a": 1}` parse lỗi. Fix: lex theo dòng, dòng `<IND>`/`<DED>`
   chính xác mới thành marker, brace của dict/list giữ nguyên.
2. **`import ai` không thấy `std/ai.t`** khi chạy từ `tests/` — load_mod chỉ tìm
   2 chỗ. Fix: tìm `same-dir`, `same-dir/std|self`, `cwd`, `cwd/std|self`,
   `parent`, `parent/std|self`; thêm `self/` nên `import lex` cũng chạy.
3. **Tensor `**`** — nhánh tensor thiếu `**` nên `tensor([1,2,3]) ** 2` lỗi.
   Fix: thêm `**` element-wise vào nhánh tensor.
4. **Attribute methods không nhất quán** — `upper/sum/keys/...` trả giá trị thô
   nên `a.sum()` / `"hi".upper()` / `d.keys()` báo "not callable". Fix: các
   method 0-arg (`upper lower strip sum mean max min keys values`) trả lambda;
   `len` giữ nguyên là property (tương thích `t.len` + `t.len()`? — chuẩn là
   property `t.len`, method gọi `t.sum()`).
5. **`indent_scan` naivety** — `#` trong string bị cắt; dòng nối trong
   `[...]` gây IND/DED giả (làm `self/lex.t` parse lỗi). Fix: string-aware
   `strip_comment` + `bracket_delta`, depth-tracking.
6. **Parser chỉ 1 `elif`** — `lex.t` có chuỗi 4 `elif` nên parse lỗi.
   Fix: vòng lặp `elif*` + `else?` thành nested-if chain (cả `t.py` và `self/parse2.t`).
7. **`and/or` không ngắn mạch** — `ev` tính cả 2 vế rồi mới `binop`, nên
   `j < n and src[j]` vẫn index-error khi `j==n`. Fix: ngắn mạch trong `ev`.
8. **`self/parse2.t` thiếu nhiều syntax** — `for...in` không check, thiếu
   `from/as/dots`, thiếu `not`/unary/slice/dict/builtins/tensor. Fix: viết lại
   đầy đủ ngang `t.py` + thêm `self/host.t` glue (`scan`+`parse_prog`).
9. **`bench/bench.py` + `vm/tcomp.py` hardcode `C:/Users/Admin/t`** — fix dùng
   path tương đối theo `__file__`.
10. **VM opcode lệch** — `tcomp.py` dùng `PRINT=0x20/HALT=0xFF` trong khi
    `tvm.h` là `0x1B/0x25`. Fix: đồng bộ theo header, thêm `OP_POP`, implement
    `PUSH_STR/ADD/SUB/MUL/DIV/MOD/POW/NEG/PRINT/LOAD/STORE/DECL/POP` trong `tvm.c`.

## 3. Self-host (v0.7, đã verify)

- `import lex` + `lex.tokenize("let x = 1 + 2")` → 7 toks ✅
- `import parse2` parse expr `1+2*3`, `2**3**2` (kết hợp phải), `a[1:3]` slice ✅
- `import host` — `host.scan` + `host.parse_prog` full program
  (`let`/`fn`/`print`/`return`/`call`) ra AST đúng format `t.py` ✅
- `self/lex.t` (95 dòng), `self/parse.t` (indent bracket-aware),
  `self/parse2.t` (full syntax), `self/host.t` (glue) — nạp xanh qua `t.py`.

## 4. VM C + JIT (v0.6–v0.7, chưa build trên máy này)

- `vm/tvm.h` (+`OP_POP`), `vm/tvm.c` (đã implement số học/print/vars),
  `vm/tcomp.py` (số học/print/let/mut/assign, opcode đúng header),
  `vm/trun.c`, `vm/jit_x64.c` (giữ nguyên).
- Đã verify bytecode bằng simulator Python (`vmtest.t` → 130 bytes, decode đúng).
- Cần MSVC trên máy thật: `cl vm/tvm.c vm/trun.c` → `trun vmtest.tb`.
- Full scope (if/loop/fn/list/tensor/dict) vẫn là roadmap — VM hiện chạy
  chương trình số học thẳng (straight-line), interpreter vẫn là runtime chính.

## 5. Lệnh verify (đã chạy xanh)

```
python D:\Code\t\t.py D:\Code\t\tests\demo.t
python D:\Code\t\t.py D:\Code\t\tests\core.t
python D:\Code\t\t.py D:\Code\t\tests\mods.t
python D:\Code\t\t.py D:\Code\t\tests\v03.t
python D:\Code\t\bench\bench.py
python D:\Code\t\vm\tcomp.py <prog-so-hoc>.t <out>.tb
python D:\Code\t\tools\test_ops.py        # OPS-OK
python D:\Code\t\tools\test_tcomp_t.py   # TCOMP-T-OK (compiler thuần T)
python D:\Code\t\tools\test_lsp.py       # LSP-ALL-GREEN
python D:\Code\t\tools\test_dbg.py       # DBG-ALL-GREEN
```
