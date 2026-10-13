# T — trạng thái & việc còn thiếu

Cập nhật: 2026-10-13 · phiên bản v2.2 · mọi mục ✅ đã verify bằng `python tools/run_all.py`
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

## 2. Còn có thể làm tiếp

| # | Hạng mục | Ghi chú |
|---|---|---|
| G6 | JIT tạo / gán record, chuỗi, dict (cần đếm tham chiếu trong code máy) | hiện chạy ở interpreter |
| G7 | JIT nhiều bản đặc hoá cho một hàm (mỗi hàm chỉ 1 bộ kiểu đối số) | kiểu khác -> interpreter |

## 3. Giới hạn đã biết (không phải bug)

| Giới hạn | Ghi chú |
|---|---|
| Biến `for` khai báo **sau** hàm thì closure không capture | `tests/hoist.t` khoá hành vi |
| `==`/`!=` trên tensor so toàn bộ (bool) | so từng phần tử: `.eq()` / `.ne()`; `< > <= >=` là từng phần tử |
| `mem()` trên `t.py`/`eval.t` trả `objs=0, gc=0` | chỉ `trun` có số liệu GC thật |
| Thông điệp vài lỗi hiếm (vd unhashable key) khác chữ giữa runtime | output bình thường khớp từng byte |
