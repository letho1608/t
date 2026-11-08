# tests/v60_test.t — Kiểm thử toàn diện cho lộ trình v6.0 (L9, L10, L11, N4, G14, J8, T5)
import os
import autograd as ag
import typecheck as tc
import net
import async

let _sys = load("__sys")

print("--- 1. L9: Generators (yield + .next() + for) ---")
fn fib_gen(n):
    mut a = 0
    mut b = 1
    mut i = 0
    while i < n:
        yield a
        tmp = a + b
        a = b
        b = tmp
        i = i + 1

g = fib_gen(6)
assert g.next() == 0, "gen next 0"
assert g.next() == 1, "gen next 1"
mut rest = []
for x in g:
    rest.append(x)
assert rest == [1, 2, 3, 5], "for over remaining generator"

print("--- 2. L9: Context Manager (with expr as v:) ---")
mut log_events = []
struct CtxTracker:
    name
    fn __enter__(self):
        log_events.append("enter:" + self.name)
        return self.name + "_val"
    fn __exit__(self):
        log_events.append("exit:" + self.name)

with CtxTracker("db") as conn:
    log_events.append("use:" + conn)
assert log_events == ["enter:db", "use:db_val", "exit:db"], "with context manager normal flow"

# Kiểm tra __exit__ vẫn chạy khi có ngoại lệ trong khối with
mut caught_with = false
try:
    with CtxTracker("tx") as tx:
        log_events.append("before_err")
        bad = 1 / 0
except e:
    caught_with = true
assert caught_with, "with rethrows exception"
assert log_events[log_events.len - 1] == "exit:tx", "with calls __exit__ on exception"

print("--- 3. L9: Operator Overloading (Struct & autograd.Node) ---")
struct Vec2:
    x
    y
    fn __add__(self, o):
        return Vec2(self.x + o.x, self.y + o.y)
    fn __sub__(self, o):
        return Vec2(self.x - o.x, self.y - o.y)
    fn __mul__(self, s):
        return Vec2(self.x * s, self.y * s)
    fn __div__(self, s):
        return Vec2(self.x / s, self.y / s)
    fn __matmul__(self, o):
        return self.x * o.x + self.y * o.y
    fn __neg__(self):
        return Vec2(-self.x, -self.y)
    fn __eq__(self, o):
        return self.x == o.x and self.y == o.y
    fn __lt__(self, o):
        return (self.x * self.x + self.y * self.y) < (o.x * o.x + o.y * o.y)
    fn __len__(self):
        return 2
    fn __str__(self):
        return f"<{self.x},{self.y}>"
    fn __getitem__(self, idx):
        return self.x if idx == 0 else self.y
    fn __setitem__(self, idx, val):
        if idx == 0:
            self.x = val
        else:
            self.y = val
    fn __iter__(self):
        return [self.x, self.y]

v1 = Vec2(3, 4)
v2 = Vec2(1, 2)
v3 = v1 + v2
assert v3 == Vec2(4, 6), "Vec2 __add__ & __eq__"
assert (v1 - v2) == Vec2(2, 2), "Vec2 __sub__"
assert (v2 * 3) == Vec2(3, 6), "Vec2 __mul__"
assert (v3 / 2) == Vec2(2, 3), "Vec2 __div__"
assert (v1 @ v2) == 11, "Vec2 __matmul__"
assert (-v2) == Vec2(-1, -2), "Vec2 __neg__"
assert v2 < v1, "Vec2 __lt__"
assert len(v1) == 2, "Vec2 __len__"
assert str(v1) == "<3,4>", "Vec2 __str__"
assert v1[0] == 3 and v1[1] == 4, "Vec2 __getitem__"
v1[0] = 9
assert v1.x == 9, "Vec2 __setitem__"
mut coords = []
for c in v2:
    coords.append(c)
assert coords == [1, 2], "Vec2 __iter__"

# Kiểm tra nạp chồng toán tử trực tiếp trên autograd.Node
nx = ag.param(tensor([2.0, 3.0]))
ny = ag.param(tensor([4.0, 5.0]))
nz = ag.sumall(nx * ny + nx - ny)
ag.backward(nz)
assert nx.grad.tolist() == [5.0, 6.0], "autograd.Node operator overloading backward x"
assert ny.grad.tolist() == [1.0, 2.0], "autograd.Node operator overloading backward y"

print("--- 4. L10: Type Annotations & Static Type Checker ---")
let typed_x: int = 42
let typed_s: str = "hello"
fn typed_add(a: int, b: int = 10) -> int:
    return a + b
assert typed_add(typed_x) == 52, "typed function call"

r_ok = tc.check_source("let a: int = 10\nfn f(x: int) -> int:\n    return x + a\nlet b: int = f(5)\n")
assert r_ok["ok"] == true, "typecheck valid source"

r_bad = tc.check_source("let a: int = \"wrong\"\nfn f(x: int) -> str:\n    return x\nf(\"bad_arg\")\n")
assert r_bad["ok"] == false and r_bad["errors"].len >= 2, "typecheck catches type mismatches"

print("--- 5. L11: Traits & Structural Pattern Matching ---")
trait Describable:
    fn kind(self) -> str
    fn describe(self) -> str:
        return "Kind=" + self.kind()

struct Item(Describable):
    label: str
    fn kind(self) -> str:
        return self.label

it = Item("Book")
assert it.kind() == "Book", "trait required method"
assert it.describe() == "Kind=Book", "trait inherited default method"

fn match_val(v):
    switch v:
        case [0, x]:
            return "pair0:" + str(x)
        case [h, *tail] if tail.len >= 2:
            return "head:" + str(h) + ",len:" + str(tail.len)
        case {"type": "pt", "x": px}:
            return "dict_pt:" + str(px)
        case Vec2(0, py) if py > 0:
            return "vec_y:" + str(py)
        case _:
            return "other"

assert match_val([0, 99]) == "pair0:99", "pattern [0, x]"
assert match_val([10, 20, 30, 40]) == "head:10,len:3", "pattern [h, *tail] with guard"
assert match_val({"type": "pt", "x": 77}) == "dict_pt:77", "pattern dict destructuring"
assert match_val(Vec2(0, 15)) == "vec_y:15", "pattern struct destructuring with guard"
assert match_val(Vec2(0, -5)) == "other", "pattern guard rejected"

print("--- 6. N4: Native TCP/UDP Sockets & Async/Await ---")
srv = net.tcp_listen("127.0.0.1", 0)
assert srv.port > 0, "tcp_listen ephemeral port"
cli = net.tcp_connect("127.0.0.1", srv.port)
peer = srv.accept()
cli.send("ping-from-client")
msg = peer.recv(1024)
assert msg == "ping-from-client", "TCP loopback send/recv"
peer.send("pong-from-server")
reply = cli.recv(1024)
assert reply == "pong-from-server", "TCP loopback reply"
cli.close()
peer.close()
srv.close()

u1 = net.udp_bind("127.0.0.1", 0)
u2 = net.udp_bind("127.0.0.1", 0)
u1.send_to("udp-hello", "127.0.0.1", u2.port)
pkt = u2.recv_from(1024)
assert pkt["data"] == "udp-hello", "UDP loopback send_to/recv_from"
u1.close()
u2.close()

# Kiểm tra async fn / await + std/async.t
async fn async_step(x):
    yield x * 2
    return x * 10

async fn async_main():
    t1 = async.spawn(async_step(3))
    t2 = async.spawn(async_step(7))
    res = await async.gather([t1, t2])
    return res[0] + res[1]

assert async.run(async_main()) == 100, "async/await + async.gather"

print("--- 7. G14: Multi-Dtype Native Tensor ---")
tf32 = tensor([1.5, -2.5, 4.0], "float32")
assert tf32.dtype == "float32" and tf32.itemsize == 4 and tf32.nbytes == 12, "f32 tensor metadata"
assert tf32.raw_ptr() != 0, "f32 raw_ptr"

tf16 = tf32.to("float16")
assert tf16.dtype == "float16" and tf16.itemsize == 2 and tf16.nbytes == 6, "f16 tensor conversion"
assert tf16.tolist() == [1.5, -2.5, 4.0], "f16 values preserved"

tbf16 = tf32.astype("bfloat16")
assert tbf16.dtype == "bfloat16" and tbf16.itemsize == 2, "bf16 tensor conversion"

ti8 = zeros([2, 4], "int8")
assert ti8.dtype == "int8" and ti8.itemsize == 1 and ti8.nbytes == 8, "int8 zeros tensor"

ti32 = full([3], 42, "int32")
assert ti32.dtype == "int32" and ti32.itemsize == 4 and ti32.tolist() == [42.0, 42.0, 42.0], "int32 full tensor"

print("--- 8. T5: Single-Binary Executable (--build-exe) ---")
tmp_src = "tests/_tmp_hello_exe.t"
tmp_exe = "tests/_tmp_hello_exe.exe"
os.write_file(tmp_src, "print(\"STANDALONE-EXE-OK:\" + str(20 + 22))\n")
rc_build = _sys.build_exe(tmp_src, tmp_exe)
assert rc_build == true, "_sys.build_exe succeeded"
assert os.exists(tmp_exe), "standalone exe created"

run_res = os.run([tmp_exe])
assert run_res["rc"] == 0 and ("STANDALONE-EXE-OK:42" in run_res["out"]), "standalone exe runs independently"

os.remove(tmp_src)
os.remove(tmp_exe)

print("V60-ALL-GREEN")
