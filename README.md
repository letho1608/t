# T language — README (v2.4: JIT x86-64 + tensor n-chiều (view như numpy) + autograd)

Ngôn ngữ T: cú pháp Python/Go-like (thụt lề), tensor là kiểu nguyên thủy,
AI-first. Bốn runtime cho output **giống nhau từng byte**:

| Runtime | Lệnh | Ghi chú |
|---|---|---|
| Interpreter tham chiếu | `python t.py prog.t` | + REPL (`python t.py`) |
| VM C + JIT | `trun prog.tb` | không cần Python; REPL (`trun`) |
| Compiler thuần T | `trun tcomp.tb prog.t prog.tb` | tự biên dịch tới fixed point |
| Evaluator thuần T | `trun tdriver.tb prog.t` | `self/eval.t` chạy trên VM |

## Build + chạy nhanh

```
vm\build.bat                    # Windows (MSVC): trun.exe + tcomp.tb + trepl.tb
sh vm/build.sh                  # Linux/macOS (cc): ./trun + tcomp.tb + trepl.tb
python tools/run_all.py         # TOÀN BỘ test (7 bộ + mọi tests/*.t) -> ALL-GREEN

python t.py tests/demo.t        # chạy bằng interpreter
python vm/tcomp.py prog.t prog.tb && trun prog.tb    # dịch bằng Python, chạy bằng C
trun tcomp.tb prog.t prog.tb    # dịch KHÔNG cần Python (có import thì tự gộp module)
trun                            # REPL (nạp trepl.tb cạnh trun)
```

## Ngôn ngữ (v2.2)

```
let x = 10                      # bất biến (gán lại -> lỗi "x immutable (use mut)")
mut n = 0                       # khả biến
n += 5                          # += -= *= /= %=
y = "lớn" if x > 3 else "nhỏ"   # biểu thức điều kiện
print(2 in [1, 2], "k" not in {"k": 1}, "b" in "abc")

for i, v in enumerate(["a", "b"]):     # nhiều biến lặp
    print(i, v)
for k, v in {"a": 1}.items():
    print(k, v)

v = 2
switch v:                       # nhiều nhánh
    case 1, 2:
        print("nhỏ")
    case "x":
        print("chữ")
    default:
        print("khác")

struct Point:                   # bản ghi + method
    x, y
    fn norm2(self):
        return self.x * self.x + self.y * self.y
    fn move(self, dx):
        self.x += dx
        return self
p = Point(3, 4)
print(p.norm2(), p.move(1).x, p, type(p))   # 25 4 Point{x: 4, y: 4} Point

s = set([3, 1, 3])              # tập hợp
s.add(7)
print(s, 3 in s, s.union([9]), s.inter([3]), s.diff([1]))

sq = fn(x): x * x               # lambda (biểu thức một dòng, closure)
print(map(fn(v): v + 1, [1, 2]), sort(["bb", "a"], fn(w): w.len))
print([x * x for x in range(5) if x % 2 == 0])          # comprehension
print({k: k * 10 for k in [1, 2]}, [[i, j] for i in range(2) for j in range(i)])
```

Lỗi runtime không bị bắt in kèm vị trí: `T error: division by zero` /
`  tại prog.t:7` (cả `t.py` lẫn `trun`; `.tb` mang bảng dòng `LN01`; với hàm
một dòng `return` đã được inline, VM báo dòng ở chỗ gọi).

Có sẵn: `fn` + closure, `if/elif/else`, `while`, `for`, `break/continue`,
`try/except "mẫu*"` (+ biến `err`), `raise`, `assert`, f-string, dict, slice
(`a[::-1]`), số `0x/0b/0o/1e-5/1_000`, `import` / `from ... import`.
`import` gộp module lúc build; **`load("tên")`** nạp module lúc chạy (tên tính được),
trả bản ghi kiểu `"module"`: `m = load("plugins/" + ten)` rồi `m.f(x)`, `m.HANG`.
`t.py`/eval.t nạp `tên.t`; VM nạp `tên.tb` (thư mục hiện tại rồi thư mục của
chương trình chính); nếu `tên.t` mới hơn `tên.tb` (hoặc chưa có) VM tự dịch bằng
`tcomp.tb` (`$TCOMP`, cạnh `trun`, hoặc thư mục hiện tại). Nạp một lần (cache theo
đường dẫn); field luôn là giá trị hiện tại của module, `m.X = v` ghi vào module;
lỗi trong module báo đúng `tại tên.t:dòng`.
`%` theo ngữ nghĩa floor như Python (`-7 % 3 == 2`) ở mọi runtime.

Builtin: `range len str int float type tensor open read write close ord chr
argv print mem f64bits abs round min max sum any all sort reversed enumerate
zip map filter`; builtin "mềm" (có thể đặt tên trùng để che): `set input load
zeros ones full arange eye rand randn seed sqrt exp log sin cos tanh floor ceil pi`.
Method: str `upper lower split join strip find replace startswith endswith`,
list `append pop clear insert extend index`, dict `keys values get items has remove`,
set `add has remove list union inter diff`.

## Tensor n-chiều

```
a = tensor([[1, 2, 3], [4, 5, 6]])
print(a.shape, a.T, a @ a.T, a + tensor([10, 20, 30]))   # broadcast kiểu numpy
print(a.sum(), a.sum(0), a.mean(1), a.argmax(1), (a > 2).sum())
r = arange(12).reshape(3, 4)
print(r[1], r[::-1].shape, r.softmax(), r.relu().sqrt())
seed(7)
w = randn([4, 2]) * 0.1         # ngẫu nhiên tất định (splitmix64), giống nhau ở mọi runtime
```

- float64, row-major, 1..8 chiều; `+ - * / **` và `< > <= >=` từng phần tử có
  broadcast; `@`: 1D·1D → số, 2D@2D, 2D@1D, 1D@2D. **`@` cùng độ ưu tiên với
  `*`** (như Python): `X @ W + b` = `(X @ W) + b`.
- `==`/`!=` so **toàn bộ** tensor (ra bool); so từng phần tử dùng `.eq(x)` / `.ne(x)`.
- Thuộc tính `shape ndim size len T`; method `sum mean max min argmax argmin`
  (có/không `axis`), `reshape transpose flatten tolist copy dot clip softmax item
  eq ne exp log sqrt tanh sin cos abs relu sigmoid floor ceil`.
- `t[i]` trên tensor n-chiều trả về **view** của hàng (dùng chung bộ nhớ:
  `t[i][j] = v` đổi `t`); `t[i] = hàng|số` để gán cả hàng.
  Chỉ số nhiều trục: `t[i, j]`, `t[:, j]` (cả cột), `t[1, :] = tensor([...])`,
  khoảng theo từng trục `t[0:2, 1:3]`, `t[::-1, 0]`. Mọi chỉ số/lát cắt (kể cả
  cột `t[:, j]`, bước `t[::2]`, bước âm), `.T`/`transpose()` và `reshape` của tensor
  liền đều là **view** như numpy (ghi qua view đổi gốc). `flatten()`, `copy()` và
  phép toán luôn tạo tensor mới. Cần bản độc lập thì `.copy()`.
- VM C và `t.py` (`ttensor.py`) khớp **từng bit** (cộng tuần tự, `@` tích luỹ theo
  k tăng dần); `tools/test_tensor.py` còn đối chiếu 38 phép với numpy.
- Ví dụ học máy: `tests/tensor.t` (MLP học XOR), `bench/nn.t`.
- **Autograd** (`std/autograd.t`, thuần T): `import autograd as ag`;
  `ag.param/const`, `add sub mul div matmul tanh sigmoid relu exp log sumall mean`
  (broadcast tự cộng dồn gradient), `ag.backward(loss)`, `ag.sgd(params, lr)`.
  Kiểm bằng sai phân hữu hạn trong `tests/agtest.t`.

## JIT x86-64 (vm/jit_x64.c)

Biên dịch **cả hàm số học** sang mã máy: int/float/bool, biến cục bộ,
`if/while/for range`, so sánh, `and/or`, `%` floor, đệ quy / gọi hàm JIT khác,
`int() float() abs() min() max() len()`, đọc global số/list (có guard), **đọc và
ghi phần tử list** (`xs[i]`, `xs[i] = v`, `xs.len`) với kiểu phần tử được đoán và
guard từng lần truy cập, **đọc field số của struct** (`p.x`) và **gọi method struct**
(`p.norm2()`, phân giải tĩnh theo kiểu) trên record mượn (đối số / global), guard
kiểu struct + tag field. Tạo / gán record vẫn chạy ở interpreter.

- Kích hoạt: hàm được gọi ≥ 50 lần (đặc hoá theo kiểu đối số, tối đa 4 bản mỗi
  hàm), hoặc **OSR** khi
  một vòng lặp trong hàm chạy ≥ 2000 vòng (kể cả hàm chỉ gọi 1 lần), hoặc vòng lặp
  **top-level** chạy ≥ 2000 vòng (global đọc/ghi thẳng bộ nhớ có guard; thoát vòng
  lặp = deopt về đúng pc).
- Deopt an toàn: hàm gọi qua entry là hàm thuần (chỉ đọc list) nên guard hỏng
  (chia 0, global đổi kiểu, hàm bị gán lại, đệ quy vượt 256KB stack native) →
  huỷ native, interpreter chạy lại. Vùng OSR được phép **ghi list**: guard hỏng
  → **deopt có chuyển trạng thái** (native chép locals + stack ảo ra, VM ghi lại
  vào frame và chạy tiếp đúng lệnh hỏng) → kết quả và thông điệp lỗi y hệt.
- Windows (Win64 ABI) + Linux/macOS (System V). `TVM_NOJIT=1` tắt,
  `TVM_JITLOG=1` in hàm được biên dịch, `TVM_JITDUMP=dir` ghi mã máy.

## Hiệu năng (`python bench/bench.py`, Windows, CPython 3.10, tốt nhất 7 lần)

| Bài | T (trun) | CPython | T/Py |
|---|---:|---:|---:|
| fib(30) đệ quy | 0.024s | 0.377s | 0.06× |
| vòng lặp toàn cục 5M | 0.041s | 0.629s | 0.07× |
| vòng lặp trong hàm 10M | 0.054s | 0.974s | 0.06× |
| mandelbrot (float) | 0.043s | 0.496s | 0.09× |
| sàng nguyên tố 2M (list) | 0.129s | 0.625s | 0.21× |
| dict + chuỗi | 0.100s | 0.208s | 0.48× |
| struct + method | 0.108s | 0.223s | 0.48× |
| matmul 120×120 ×20 | 0.029s | 2.858s (list Python) | 0.01× |
| MLP 512×16→64→1, 300 bước | 0.246s | 0.947s (**numpy**) | 0.26× |

Thời gian tính cả khởi động tiến trình (máy đo lúc có tải; tỷ lệ T/Py ổn định giữa
các lần chạy). Linux (gcc 13, CPython 3.12 + numpy): số học/JIT/sieve/matmul vượt
xa; MLP 0.32s so với numpy 0.43s; dict/method ngang CPython 3.12 (±20%).
Compiler T tự host dịch chính nó trong 0.43s (compiler Python 0.36s).

## Công cụ

| Công cụ | Lệnh |
|---|---|
| REPL | `python t.py` hoặc `trun` (khối kết thúc bằng dòng trống, `exit` để thoát) |
| Debugger | `trun -d prog.tb` (`s` step, `b pc`, `c`, `p` stack, `l` locals, `g`, `d` disasm) |
| Formatter | `python t.py tools/ffmt.t [--check] prog.t` |
| LSP | `python tools/tlsp.py` (hover, definition, references, rename, completion, symbols) |
| Package | `python tools/tpkg.py init/install/run/test/pack/publish/search/fetch/release` |
| Bench | `python bench/bench.py [tên...] [--nojit]` |

tpkg: registry có thể là server (`tools/tpkg_reg.py`) **hoặc thư mục cục bộ /
`file://`** (offline hoàn toàn); index giữ mọi phiên bản, `mathx@^1.0` chọn bản
cao nhất thoả; archive được cache theo sha (`~/.tpkg/cache`), `install --offline`
dựng lại từ `lock.json` + cache (kiểm sha); `release patch|minor|major` tăng
semver + ghi `CHANGELOG.md` (lấy từ git log nếu có).

## Kiểm thử

`python tools/run_all.py` chạy: `test_ops` (đồng bộ 74 opcode + 51 native giữa
`vm/tvm.h`, `vm/tcomp.py`, `self/tcomp.t`), `test_tcomp_t` (25 suite: compiler T
== t.py, fixed point, compiler Python == compiler T từng byte, eval.t trên VM,
GC quét dày, UTF-8), `test_dbg`, `test_lsp`, `test_tpkg`, `test_jit`,
`test_tensor`, và mọi `tests/*.t` qua `t.py`. CI: `.github/workflows/ci.yml`
(Linux gcc + Windows MSVC).

## Kiến trúc

- `t.py` + `ttensor.py`: lexer/parser (desugar `for a, b`, `switch`, `struct`,
  `+=`), interpreter tham chiếu.
- `vm/tcomp.py` / `self/tcomp.t`: compiler → `.tb` (prescan slot, closure qua
  cell, inliner theo từng call-site, peephole superop, bundler module).
- `vm/tvm.c`: stack VM, refcount + GC mark-sweep chu trình (4 pha an toàn),
  dict băm (giữ thứ tự chèn), chuỗi một khối + cache hash, hằng chuỗi cache,
  inline cache ATTR/CALLM/SETATTR, `vm/tensor_impl.h` (tensor), `vm/jit_x64.c`
  (JIT), `vm/tdbg.c` (debugger), `vm/trun.c` (runner/REPL).
- Lưu ý khi sửa `vm/tvm.c`: giữ `#pragma GCC optimize("no-inline")` (vòng lặp có
  `setjmp`; mingw GCC ≥ 12 từng inline hỏng); buffer mode của `open()` phải sống
  tới `fopen`.

Xem thêm: `SPEC.md` (đặc tả), `STATUS.md` (nhật ký verify), `TODO.md`.
