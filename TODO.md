# T — trạng thái & việc còn thiếu

Cập nhật: 2026-10-16 · phiên bản v2.6 · mọi mục ✅ đã verify bằng `python tools/run_all.py`
(Windows MSVC + Linux gcc 13 qua Docker).

Tóm một dòng: ngôn ngữ T có 4 runtime khớp từng byte, compiler thuần T tự biên dịch
(fixed point), VM C có **method JIT x86-64 + OSR**, **tensor n-chiều** khớp numpy,
REPL/LSP/debugger/formatter/package manager, CI 2 nền tảng.

---

## 1. Đã có

### 1.1 Runtime / compiler

| Thành phần | File | Trạng thái |
|---|---|---|
| Interpreter tham chiếu (+ REPL, đệ quy sâu ≥ 20k tầng) | `t.py` + `ttensor.py` | ✅ |
| Compiler Python | `vm/tcomp.py` | ✅ |
| Compiler thuần T (tự gộp module khi có import) | `self/tcomp.t` | ✅ fixed point |
| VM C (+ debugger `-d`, REPL khi không đối số) | `vm/tvm.c` `vm/trun.c` `vm/tdbg.c` | ✅ |
| JIT x86-64 (Win64 + System V) | `vm/jit_x64.c` | ✅ |
| Tensor n-chiều trong C | `vm/tensor_impl.h` | ✅ |
| Evaluator thuần T | `self/eval.t` | ✅ |

- Compiler Python và compiler T ra bytecode **giống từng byte** trên mọi suite (trước v2.2
  sai ở `jit`, `selfhost`, `fmtself` do inliner khác nhau).
- `tests/*.t` chạy qua `t.py`, VM (compiler T), và 11 suite qua `eval.t` trên VM: output khớp.

### 1.2 Ngôn ngữ (v2.2 mới)

| Tính năng | Trạng thái |
|---|---|
| `a if c else b` | ✅ |
| `x in xs`, `x not in xs` (str/list/dict/set/tensor) | ✅ |
| `for a, b in ...` (L5) | ✅ |
| `+= -= *= /= %=` | ✅ |
| `switch` / `case a, b` / `default` (L3) | ✅ |
| `struct` (field + method + `self`, `p.x = v`) (L2) | ✅ |
| `set` + `.add/.has/.remove/.union/.inter/.diff/.list` (L1) | ✅ |
| dict `.items/.has/.remove`; list `.clear/.insert/.extend/.index`; str `.find/.replace/.startswith/.endswith` | ✅ |
| số mũ `1e-5`; `%` kiểu floor ở mọi runtime | ✅ |
| `let`/`fn` bất biến được kiểm cả ở compiler (trước đây chỉ `t.py`) | ✅ |
| builtin "mềm" che được tên (`set input zeros ... pi`) | ✅ |
| Lỗi cú pháp báo **đúng số dòng** (t.py + compiler T) | ✅ |

### 1.3 Toolchain

| # | Hạng mục | Trạng thái |
|---|---|---|
| L4 | REPL cho `trun` (`tools/repl.t` → `trepl.tb`, giữ biến qua các khối) | ✅ |
| T1 | `import` khi build `.tb`: compiler tự gộp module, không cần `-b` | ✅ (liên kết lúc build) |
| T2 | LSP biết từ khoá/builtin mới (`switch case default struct`, tensor, set...) | ✅ |
| T3 | tpkg offline: registry thư mục / `file://`, index mọi phiên bản, cache sha, `install --offline` | ✅ `tools/test_tpkg.py` |
| N1 | Verify POSIX thật: build gcc 13 + toàn bộ test + JIT System V trên Linux | ✅ |
| N2 | CI: `.github/workflows/ci.yml` (Linux + Windows), `tools/run_all.py` | ✅ (workflow chưa chạy trên GitHub vì thư mục chưa là repo) |
| N3 | `tpm release patch/minor/major` + `CHANGELOG.md` (git log nếu có) | ✅ |

### 1.4 Hiệu năng

| # | Hạng mục | Trạng thái |
|---|---|---|
| P1 | JIT: cả hàm số học (vòng lặp, nhánh, đệ quy, gọi chéo, builtin số), OSR, deopt an toàn | ✅ fib/mandel nhanh hơn CPython 10–15× |
| J1 | JIT đọc/ghi phần tử list (+ `len`, `.len`), OSR + deopt có chuyển trạng thái | ✅ sieve nhanh hơn CPython ~5× (`tests/jitlist.t`) |
| P2 | Đường nhanh IDX/SETIDX/LEN list, số học/so sánh int-float, LOAD/STORE | ✅ |
| P3 | Inline cache theo pc cho ATTR/CALLM/SETATTR + opcode `CALLM`/`SETATTR` | ✅ method nhanh hơn 6× |
| — | dict băm (giữ thứ tự chèn), chuỗi một khối + cache hash, hằng chuỗi cache | ✅ |

### 1.4b v2.3 (2026-10-13)

| # | Hạng mục | Trạng thái |
|---|---|---|
| R2 | Lỗi runtime báo `tại file:dòng` (bảng `LN01` trong `.tb`, mốc dòng trong AST khi biên dịch) | ✅ VM + t.py |
| L6 | `fn(x): e` (lambda), `[e for x in xs if c]`, `{k: v for ...}` (desugar thành hàm ẩn) | ✅ `tests/lambda.t` |
| G1 | Tensor chỉ số nhiều trục `t[i, j]`, `t[:, j]`, gán vùng; autograd `std/autograd.t` | ✅ `tests/tensor.t`, `tests/agtest.t` |
| J2 | JIT vòng lặp top-level (global ghi thẳng, lối ra = deopt về pc) | ✅ loop nhanh hơn CPython 14× (`tests/jitmain.t`) |
| J3 | JIT đọc field số + gọi method struct (record mượn, guard kiểu/tag) | ✅ ~13× so với interpreter (`tests/jitrec.t`) |
| J4 | VM giữ `pc` trong thanh ghi ở vòng dispatch | ✅ loop/sieve nhanh hơn 10–13% (không JIT) |
| J5 | JIT tối đa 4 bản đặc hoá / hàm (mỗi bộ kiểu đối số một bản) | ✅ `tests/jitrec.t` (P int + P float) |
| G2 | Lát cắt theo nhiều trục `t[0:2, 1:3]`, `t[::-1, 0]` (ngữ nghĩa slice của Python) | ✅ `tests/tensor.t` |
| R1 | `load("tên")`: import động lúc chạy, tên tính được; VM nạp `tên.tb` riêng (phần `EX01`) | ✅ `tests/loadmod.t` (4 runtime) |
| R3 | `load()`: field sống (đọc global hiện tại, `m.X = v` ghi vào module); VM tự dịch `tên.t` khi `.tb` thiếu/cũ | ✅ `tests/loadmod.t` |
| G3 | Tensor view: `t[i]`, `t[a:b]`, `t[i, j]`… khi vùng chọn là khối liền thì dùng chung bộ nhớ | ✅ `tests/tensor.t` |
| G4 | View theo bước: cột `t[:, j]`, `t[::2]`, bước âm, `.T`, `reshape` (như numpy) | ✅ `tests/tensor.t` |
| G5 | `m.X = v` trên module của `load()` tôn trọng `mut` (let/fn/struct báo lỗi) | ✅ `tests/loadmod.t` |

Lỗi có sẵn sửa thêm: compiler T không tìm module khi đường dẫn dùng `\`; bundler của
compiler Python không đổi tên hàm lồng trong hàm của module; thông điệp
"unhashable dict key" khác nhau giữa VM và t.py; `vm/build.bat` lưu xuống dòng LF
khiến cmd.exe đọc lệch các dòng có chữ UTF-8 (đã chuyển CRLF + `.gitattributes`).

### 1.4c v2.5 (2026-10-16)

| # | Hạng mục | Trạng thái |
|---|---|---|
| G6 | JIT tạo record (`P(x, y)` field số qua `__new`), gán field số (`SETATTR`, OSR), chuỗi hằng (`PUSH_STR` + `==`/`!=`/`len()`/`if s`) | ✅ record mượn + sở hữu được, refcount trong code máy; deopt move/retain; `tests/jitrec.t` 10 entry/7 OSR; bench method 0.48× → 0.22× CPython |
| — | `trun --version` 2.5; fix JITLOAD đọc `s->st[d]` ngoài biên; guard `__new` loại kiểu module; NULL-check shape cho gcc | ✅ `run_all` xanh Win + Linux, `TVM_GCTHRESH=1` khớp byte |

### 1.4d v2.6 (2026-10-16)

| # | Hạng mục | Trạng thái |
|---|---|---|
| G7 | JIT dict + chuỗi sở hữu được: `NEW_DICT {}` → `JT_DN`, `str()` → `JT_SN`, nối chuỗi `ADD`, `d[k]`/`d[k] = v` đoán int (1 lần tìm, deopt khi thiếu/khác int), `d.get(k, dflt)` | ✅ helper C + refcount trong code máy (entry-compare, POP/RET-decs, deopt move/retain); `tests/jitdict.t` mới (6 OSR); bench dict 0.43× → 0.37× CPython |
| G11 | `STORE` chuỗi mới vào local + `return` chuỗi tạo mới trong native | ✅ (làm cùng G7: `JT_SN` + epilogue RET decs; wrapper malloc không track GC nên dec đúng hạn mọi đường) |
| — | Lỗi phát hiện khi làm G7 (đều đã sửa, có test đọc giá trị bắt cả 3): emitter `mov_r8_rax` sai byte REX (`4C`→`49`, r8 không bao giờ được ghi); codegen `d.get` ghi default đè slot receiver trước khi gọi helper; `tjit_str_eq` 2 bản trùng tên | ✅ |

### 1.4e v2.6.1 (2026-10-27) — tối ưu wrapper chuỗi JIT

| # | Hạng mục | Trạng thái |
|---|---|---|
| — | Bottleneck G7 đo được: mỗi `str()`/nối chuỗi trong vòng JIT cấp phát wrapper `TVal*` bằng `malloc`/`free` CRT (Windows heap đắt) — `mb_str` 3M vòng chỉ hơn interpreter 1.31×; dict-int đã 2.70× nên chắc chắn lỗi ở chuỗi | ✅ đổi toàn bộ wrapper sang pool 32 byte (`pk_alloc`/`pk_free`, thêm `tjit_wrap_free`); wrapper hằng số `OP_PUSH_STR` (sống vĩnh viễn) giữ `malloc` |
| — | Sau tối ưu: `mb_str` 1.31× → **3.31×**, bench dict **0.37× → 0.23×** CPython (0.067s → 0.041s), method 0.041 → 0.037s; GC dày (`TVM_GCTHRESH=1/3`) khớp; ALL-GREEN Windows MSVC | ✅ |

Ghi chú đo (best of 5, `--runs 5`, 3M vòng cho micro):

| micro | JIT trước | JIT sau pool | +cache str(int) | +fusion hint | +key_eq_fast | NOJIT |
|---|---|---|---|---|---|---|
| `mb_str` (str+nối) | 0.361s | 0.137s | 0.099s | — | — | 0.453s |
| `mb_dictint` (dict key int get+set) | 0.148s | 0.149s | — | 0.108s | **0.092s** | 0.387s |
| `mb_get` (chỉ `d.get`) | — | — | — | — | **0.053s** | 0.261s |
| `mb_set` (chỉ `d[k]=v`) | — | — | — | — | **0.094s** | 0.233s |
| `bench/dict.t` | 0.067s | 0.041s | 0.033s | 0.032s | **0.031s** | 0.067s |

Ba mục tối ưu tiếp theo (đã làm, ALL-GREEN, GC dày khớp):

1. **Cache `str(i)` số nhỏ** (`mk_str_int`, cache `-256..2047` ref vĩnh viễn như
   `str1_cache`): `mb_str` 0.137 → 0.099s, `dict.t` 0.041 → 0.033s.
2. **Counter fusion qua hint index** (`tjit_last_i` + validate trong `tjit_dict_set`,
   emitter luôn đặt bit5): `mb_dictint` 0.143 → 0.108s. Sai hint chỉ chậm, không
   bao giờ sai kết quả (validate `i < len && keys[i] == k`; keys không bao giờ trùng).
3. **`key_eq_fast`** — so sánh khoá int/str inline né C-call `tval_eq` trong
   `dict_find` (lợi cả interpreter lẫn JIT, thay cho probe int-key viết tay trong
   asm vốn chỉ lợi JIT và nhiều rủi ro): `mb_get` 0.077 → 0.053s, `mb_dictint`
   0.108 → 0.092s, `mb_set` 0.107 → 0.094s, `dict.t` → 0.031s.

### 1.5 Sửa lỗi có sẵn phát hiện khi làm v2.2

- Inliner dùng chung biến tạm → `add(1, add(2, 3))` ra 7 (compiler T); thân inline dùng chung
  giữa 2 hàm → slot sai (compiler Python).
- `return/break/continue` trong `try` không gỡ handler → lỗi về sau nhảy vào hàm đã trả về.
- `tvm_call_value` free giá trị trả về một lần thừa (use-after-free khi callback trả object).
- GC: vòng quét giữ con trỏ node đã bị free lồng nhau; GC chạy giữa lời gọi lồng từ native.
- Slice bước âm khi bỏ trống cận (`xs[::-1]`) ra rỗng trên VM.
- `%` số âm lệch giữa VM (C) và `t.py`; dict rỗng truthy trên `t.py`; `print` tính đối số lệch thứ tự.
- Disassembler: `MKFN` dài sai, thiếu superop; build POSIX lỗi `MAP_ANONYMOUS`.
- `eval.t`: `sort` so sánh sai và không sắp tại chỗ; chỉ số float.
- README hướng dẫn build thiếu `jit_x64.c tdbg.c` (lỗi link); `INSTALL_TO_D.bat` hỏng.

---

## 2. Còn có thể làm tiếp (ưu tiên từ trên xuống)

| # | Hạng mục | Ghi chú |
|---|---|---|
| G8 | `SETATTR` trong entry khi receiver là local owned mới tạo trong cùng hàm (không alias ngoài) | hiện `p.move(1)` trong vòng nóng kéo cả vòng về interpreter |
| G9 | Đoán `retty` thêm shape record của hàm đang biên dịch (hiện chỉ thử I/F/B) | hàm đệ quy trả record (`build(n)`) luôn rớt về interpreter |
| G10 | Tag ownership ở runtime thay cho `nodec` tĩnh (ghi đè local upgraded vẫn `dec` được) | hiện mỗi vòng orphan ~200 byte chờ GC khi local từng nâng borrowed → owned |
| G12 | Dict literal `{"k": v}` (k>0), method dict (`keys`/`values`/`has`/`remove`), key float/bool, chuỗi làm đối số gọi hàm T, `STORE` chuỗi mượn, điều kiện dict | rớt về interpreter (đúng), chờ G8/G10 hoặc khi cần |

Đã xong: G6, G7, G11 (bảng trên). Còn lại ở interpreter: `s[i]`, method dict/list/str
(khác `get`/`len`), closure/`try` trong vùng nóng.

## 3. Giới hạn đã biết (không phải bug)

| Giới hạn | Ghi chú |
|---|---|
| Biến `for` khai báo **sau** hàm thì closure không capture | `tests/hoist.t` khoá hành vi |
| `==`/`!=` trên tensor so toàn bộ (bool) | so từng phần tử: `.eq()` / `.ne()`; `< > <= >=` là từng phần tử |
| `mem()` trên `t.py`/`eval.t` trả `objs=0, gc=0` | chỉ `trun` có số liệu GC thật |
| Thông điệp vài lỗi hiếm (vd unhashable key) khác chữ giữa runtime | output bình thường khớp từng byte |
