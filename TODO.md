# T — trạng thái & việc còn thiếu

Cập nhật: 2026-11-06 · phiên bản v5.1 · mọi mục ✅ đã verify bằng `.\trun.exe tools\run_all.t`
→ 48/48 ALL-GREEN (Windows MSVC 19.44; Linux gcc 13 qua CI `vm/build.sh`).
Lưu ý: các mục nhắc `t.py`/`vm/tcomp.py`/`*.py` ở phần lịch sử là trạng thái trước hạng mục T4 —
toàn bộ file `.py` của T đã bị xoá, giờ chỉ còn `.t`.

Tóm một dòng: ngôn ngữ T có 4 runtime khớp từng byte, compiler thuần T tự biên dịch
(fixed point), VM C có **method JIT x86-64 + OSR**, **tensor n-chiều** khớp numpy,
REPL/LSP/debugger/formatter/package manager, CI 2 nền tảng.

---

## 1. Đã có

### 1.1 Runtime / compiler

| Thành phần | File | Trạng thái |
|---|---|---|
| Interpreter tham chiếu (từng là `t.py` + `ttensor.py`, đã xoá ở T4) | `self/eval.t` + REPL | ✅ |
| Compiler Python (từng là `vm/tcomp.py`, đã xoá ở T4) | thay bằng `self/tcomp.t` | ✅ |
| Compiler thuần T (tự gộp module khi có import) | `self/tcomp.t` | ✅ fixed point |
| VM C (+ debugger `-d`, REPL khi không đối số) | `vm/tvm.c` `vm/trun.c` `vm/tdbg.c` | ✅ |
| JIT x86-64 (Win64 + System V) | `vm/jit_x64.c` | ✅ |
| Tensor n-chiều trong C | `vm/tensor_impl.h` | ✅ |
| Evaluator thuần T | `self/eval.t` | ✅ |

- Compiler T ra bytecode **giống từng byte** trên mọi suite (trước T4 còn so với compiler Python;
  trước v2.2 sai ở `jit`, `selfhost`, `fmtself` do inliner khác nhau).
- `tests/*.t` chạy qua VM (compiler T), và 11 suite qua `eval.t` trên VM: output khớp.

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
| Lỗi cú pháp báo **đúng số dòng** (compiler T + VM) | ✅ |

### 1.3 Toolchain

| # | Hạng mục | Trạng thái |
|---|---|---|
| L4 | REPL cho `trun` (`tools/repl.t` → `trepl.tb`, giữ biến qua các khối) | ✅ |
| T1 | `import` khi build `.tb`: compiler tự gộp module, không cần `-b` | ✅ (liên kết lúc build) |
| T2 | LSP biết từ khoá/builtin mới (`switch case default struct`, tensor, set...) | ✅ |
| T3 | tpkg offline: registry thư mục / `file://`, index mọi phiên bản, cache sha, `install --offline` | ✅ `tools/test_tpkg.t` |
| N1 | Verify POSIX thật: build gcc 13 + toàn bộ test + JIT System V trên Linux | ✅ |
| N2 | CI: `.github/workflows/ci.yml` (gitleaks + trivy, build/test Linux gcc & Windows MSVC chạy `tools/run_all.t`) | ✅ workflow đã có trong repo; còn cần xác nhận trạng thái xanh trên GitHub Actions |
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

## 2. Hoàn thiện v3.0 & v3.1 (2026-10-27) — Đã hoàn tất 100%

| # | Hạng mục | Trạng thái |
|---|---|---|
| G8 | `SETATTR` trong entry khi receiver là local owned hoặc không abort sau điểm ghi | ✅ `vm/jit_x64.c`, `tests/jitrec.t` |
| G9 | Đoán `retty` thêm shape record của hàm đang biên dịch (hàm đệ quy trả record `build(n)`) | ✅ `vm/jit_x64.c`, `tests/jitrec.t` |
| G10 | Tag ownership ở runtime (`OWN(jc, i)`) thay cho `nodec` tĩnh (ghi đè local upgraded giải phóng ngay trong vòng lặp) | ✅ `vm/jit_x64.c`, `tests/jitrec.t` |
| G12 | Dict literal `{"k": v}` (k>0), method dict (`has`/`remove`), chuỗi làm đối số gọi hàm T, `STORE` chuỗi mượn, `s[i]`, điều kiện dict | ✅ `vm/jit_x64.c`, `tests/jitdict.t` |
| S1 | Bảo mật VM: `verify_bytecode` kiểm tra toàn vẹn `.tb`, bỏ `system()` thay bằng `CreateProcessA`/`execvp` chống CWE-78, `zip_safe_arcname` chống CWE-22 | ✅ `vm/tvm.c` |
| L7 | Tham số mặc định cho hàm (`fn f(a, b = 10):`), chạy trực tiếp `trun prog.t` và `trun -c "..."`, thư viện chuẩn `math`, `json`, `os`, `path`, `time` | ✅ `tests/stdlib.t` |
| T4 | Chuyển toàn bộ công cụ và bộ kiểm thử sang thuần `.t` (`tools/*.t`, `bench/bench.t`), loại bỏ hoàn toàn các file `.py` của T | ✅ `trun tools/run_all.t` |
| L8 | Cú pháp mở rộng v3.1: Toán tử bitwise 64-bit (`&`, `\|`, `^`, `~`, `<<`, `>>`, compound `&=..>>=`), `try/except/finally` bảo đảm dọn dẹp, tham số biến thiên `*args` | ✅ `self/parse2.t`, `vm/tvm.c`, `tests/stdlib.t` |
| J6 | JIT x86-64 bitwise: mã máy gốc cho `not`, `and`, `or`, `xor`, `shl`, `sar` với guard tràn bit | ✅ `vm/jit_x64.c`, `tests/stdlib.t` |
| S2 | Thư viện chuẩn mở rộng: `std/random.t` (PRNG 64-bit có hạt giống), `std/re.t` (Regex engine thuần T), `std/csv.t` (RFC 4180), `std/http.t` (URL & HTTP client) | ✅ `tests/stdlib.t` |
| P4 | Kho mã nguồn chuẩn: `LICENSE` (MIT), 4 bài mẫu trong `examples/`, tiện ích mở rộng VS Code trong `editors/vscode/` | ✅ `examples/*.t` |

## 3. Giới hạn hiện tại (không phải bug)

| Giới hạn | Ghi chú |
|---|---|
| Biến `for` khai báo **sau** hàm thì closure không capture | `tests/hoist.t` khoá hành vi |
| `==`/`!=` trên tensor so toàn bộ (bool) | so từng phần tử: `.eq()` / `.ne()`; `< > <= >=` là từng phần tử |
| Thông điệp vài lỗi hiếm (vd unhashable key) khác chữ giữa runtime | output bình thường khớp từng byte |
| Chạy đơn luồng cho 1 tiến trình VM | Không có tranh chấp lock/GIL; đa luồng nằm trong lộ trình v4.0 |

---

## 4. Hoàn thiện v4.0 & v5.0 (2026-10-28) — Động cơ LLM & Hệ thống Công nghiệp (100% Hoàn tất)

| # | Hạng mục | Trạng thái |
|---|---|---|
| **F1** | **C FFI (`std/ffi.t`)** | ✅ Nạp động `.dll` / `.so` (`LoadLibraryA` / `dlopen`, `GetProcAddress` / `dlsym`), gọi trực tiếp hàm C ngoài, dispatch lên tới 16 đối số đa kiểu (int, pointer, string, double) — đủ cho `cublasDgemm_v2` 14 đối số (M6). `tests/ffi_test.t`. |
| **T1** | **Đa luồng & Kênh (Worker Threads & Channels `std/thread.t`)** | ✅ Kiến trúc Actor/Worker không GIL (`CRITICAL_SECTION`/`pthread_mutex_t`, `CONDITION_VARIABLE`), hàng đợi MPMC cách ly sâu bộ nhớ 100%, TVM thread-local pool. `tests/thread_test.t`. |
| **M1** | **Memory-Mapped I/O (`std/mmap.t`)** | ✅ Ánh xạ file SSD dung lượng lớn vào RAM ảo zero-copy (`CreateFileMappingA`/`MapViewOfFile`, `mmap`), nạp trực tiếp tensor float và lượng tử vào TVM. `tests/mmap_test.t`. |
| **M2** | **Bộ đọc GGUF & Safetensors (`std/gguf.t`, `std/safetensors.t`)** | ✅ Bộ phân tích nhị phân chuẩn GGUF v2/v3 và Hugging Face Safetensors: đọc metadata, danh mục tensor, offset, nạp tensor trực tiếp qua mmap. `tests/gguf_test.t`, `tests/safetensors_test.t`. |
| **M3** | **Lượng tử hóa (`std/quant.t` — FP16, Q4_0, Q8_0)** | ✅ Nhân nén và giải lượng tử hóa tốc độ cao cho các khối Q4_0 (18 bytes/32 weights), Q8_0 (34 bytes/32 weights) và FP16, tối ưu hóa giảm ~85% dung lượng mô hình. `tests/mmap_test.t`, `tests/gguf_test.t`. |
| **M4** | **Động cơ Transformer suy luận (`std/transformer.t`, `examples/llm_inference.t`)** | ✅ Kiến trúc LLaMA/Mistral/Qwen: RMSNorm, RoPE, KV-Cache, Grouped-Query Attention (GQA), SwiGLU FFN, BPE Tokenizer, Sampler (top-k, top-p, temperature), streaming generation đạt >2300 tokens/giây trên CPU. Benchmark GPT-2 nhanh hơn NumPy 1.36×–1.82× và nhanh hơn PyTorch CPU 12×. `tests/transformer_test.t`. |

### 4.1 v5.1 (2026-10-28) — Cơ sở Dữ liệu Nhúng, Thao tác Bộ nhớ FFI, Base64 & Tăng tốc SIMD Matmul

| # | Hạng mục | Trạng thái |
|---|---|---|
| **F2** | **Thao tác Bộ nhớ & Con trỏ FFI (`vm/tvm.c`, `std/ffi.t`)** | ✅ Native memory primitives: `mem_alloc`, `mem_free`, `mem_read_ptr`, `mem_write_ptr`, `mem_read_i32`, `mem_write_i32`, `mem_read_str`, `mem_read_f64`, `mem_write_f64` và struct `Buffer` tiện ích. Sửa calling convention x64 Windows ABI khi hàm C trả về `double` nhận tham số integer/pointer. |
| **F3** | **SQLite Database Nhúng (`std/sqlite.t`)** | ✅ Client SQLite thuần T kết nối trực tiếp driver OS có sẵn (`winsqlite3.dll` trên Windows 10/11, `libsqlite3.so` trên Linux/macOS) 0 phụ thuộc cài đặt ngoài. Hỗ trợ Transactions (`begin`, `commit`, `rollback`), Prepared Statements, Parameter Binding (`?`), mapping hàng tự động sang Dictionary. `tests/sqlite_test.t`. |
| **U1** | **Base64 RFC 4648 (`std/base64.t`)** | ✅ Bộ mã hóa và giải mã Base64 RFC 4648 tiêu chuẩn, xử lý chính xác padding `=`, `==` và truyền dữ liệu nhị phân đầy đủ 0..255 byte round-trip. `tests/base64_test.t`. |
| **G13**| **Tối ưu hóa SIMD & Unrolling Matmul (`vm/tensor_impl.h`)** | ✅ Tối ưu hoá vòng lặp `ten_matmul` với 4-way & 8-way unrolling và đa bộ tích lũy độc lập (parallel accumulators). Trình biên dịch C tự động sinh mã vector AVX2/SSE, tăng tốc độ nhân vector-matrix cho suy luận LLM. |
| **M5** | **Quản lý GPU Driver CUDA & CPU Fallback (`std/cuda.t`)** | ✅ Tương tác CUDA Driver API qua FFI (`nvcuda.dll` / `libcuda.so`), truy vấn danh sách GPU, tên thiết bị, tổng dung lượng VRAM, phiên bản driver, tự động kích hoạt Graceful CPU Fallback khi không có GPU. `tests/cuda_test.t`. |

---

## 5. Phân tích Hiệu năng AI: Suy luận (Inference) vs Huấn luyện (Training)

| Khía cạnh | Suy luận (Inference / Generation) | Huấn luyện (Training / Backpropagation) |
|---|---|---|
| **Kích thước xử lý** | Batch size = 1 (từng vector token liên tiếp) | Mini-batch lớn ($B = 32, 64, 128, 256$, seq = $1024..4096$) |
| **Bản chất tính toán** | **Memory/Latency-Bound** (Nghẽn bởi độ trễ điều phối vòng lặp và băng thông đọc RAM) | **Compute-Bound** (Nghẽn bởi năng lực tính toán FLOPs thuần túy của CPU/GPU) |
| **Trọng số thời gian** | 70% – 90% là chi phí gọi hàm, quản lý frame, dispatch table, boxing đối tượng | 99% là các phép nhân ma trận khổng lồ ($M \times K \times N$) ở cả Forward và Backward pass |
| **Hiện trạng T vs Python** | **T vượt trội hoàn toàn**: nhanh hơn NumPy **1.36× – 1.82×**, nhanh hơn PyTorch CPU **12.2× – 12.6×** nhờ zero-overhead C runtime và JIT x86-64 | **PyTorch có lợi thế ở batch lớn**: nhờ Intel MKL / oneDNN đa luồng Assembly chia tile L3 cache. T vẫn nhanh hơn ở mô hình nhỏ / batch nhỏ / Online Learning |

---

## 6. Lộ trình Mở rộng Tiếp theo (Hỗ trợ Toàn diện Huấn luyện & Tăng tốc Phần cứng)

| # | Hạng mục | Mục tiêu & Kỹ thuật |
|---|---|---|
| **T2** | **Multi-threaded Cache-Blocked GEMM** | ✅ `vm/tensor_impl.h`: `ten_matmul` chia C theo hàng cho worker (`CreateThread`/`pthread_create`, main chạy task 0), kernel i-k-j chặn cache MC=64 / KC=128 / NC=128; thứ tự cộng mỗi `C[i,j]` vẫn q tăng dần nên **trùng bit** với tuần tự. Env `TVM_GEMM_THREADS` (=1 tuần tự, =N đúng N luồng, auto theo số nhân), ngưỡng `m*n*k ≥ 1M && m ≥ 32`. Đo 192³×30: 111.7ms → 44.2ms (**2.53×** trên 8 nhân, checksum trùng). `tests/gemm_mt.t` (trùng byte giữa 1/6/auto + khớp tham chiếu ba vòng). |
| **M6** | **GPU Tensor Offloading (cuBLAS / cuDNN)** | ✅ `std/cublas.t` qua `std/ffi.t` + `std/cuda.t`: probe `cudart`/`cublas` (`cudaMalloc`/`cudaFree`/`cudaMemcpy`/`cudaGetDeviceCount`, `cublasCreate_v2`/`cublasDgemm_v2` 14 đối số — cần FFI nâng 6→16 đối số trong `vm/tvm.c` và tensor method `data_ptr()` để đưa buffer row-major sang GPU). Đổi chiều row↔column-major bằng trick `C^T = B^T @ A^T` nên chỉ dùng `cublasDgemm_v2`. Thiếu GPU/driver/lib hoặc bất kỳ bước nào lỗi → fallback êm về `@` CPU (GEMM đa luồng T2). `tests/cublas_test.t` (không GPU: fallback khớp `@` < 1e-9 + round-trip `mem_read/write_f64` + `data_ptr`). FP16/BF16 Tensor Cores và cuDNN convolution vẫn là việc mở rộng sau. |
| **M7** | **Cross-Platform GPU (Vulkan Compute Shader)** | Tăng tốc GPU phi độc quyền cho cả AMD Radeon, Intel Arc và Apple qua Vulkan Compute / SPIR-V, mang lại khả năng tăng tốc phần cứng trên mọi thiết bị. |
| **A1** | **Fused Backward Kernels trong `std/autograd.t`** | ✅ Kernel C gộp trong `vm/tensor_impl.h`: `softmax_bwd` (Jacobian gộp), `gelu_bwd` + forward `gelu` (erf chính xác), `rmsnorm`/`rmsnorm_bwd`, `layernorm`/`layernorm_bwd`, `swiglu_bwd` → `[dgate, dup]` — mỗi kernel vài pass trên bộ nhớ gốc, **không tensor trung gian**. `std/autograd.t` thêm `softmax`, `gelu`, `rms_norm`, `layer_norm`, `swiglu`; `std/ai.t` thêm `gelu`; `self/eval.t` cập nhật whitelist method. `tests/fusedbwd.t`: khớp sai phân hữu hạn 1e-6 cho cả 5 phép + softmax khớp compose trong T. |
| **J7** | **JIT ARM64 (AArch64)** | Trình phát mã máy gốc AArch64 cho Apple Silicon (M1/M2/M3/M4) và máy chủ Linux ARM64 (Ampere Altra, AWS Graviton, Raspberry Pi). |
