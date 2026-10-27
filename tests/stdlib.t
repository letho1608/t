# tests/stdlib.t — kiểm thử tính năng mở rộng & thư viện chuẩn (math, json, path, time, os, random, re, csv, http)
import math
import json
import path
import time
import os
import random
import re
import csv
import http

# 1. Tham số mặc định (default parameters) cho fn và struct method
fn greet(name, prefix = "Xin chào", punct = "!"):
    return prefix + " " + name + punct

assert greet("T") == "Xin chào T!"
assert greet("T", "Hello") == "Hello T!"
assert greet("T", "Hi", ".") == "Hi T."

struct Counter:
    val
    fn inc(self, step = 1, scale = 1):
        self.val = self.val + step * scale
        return self.val

mut c = Counter(10)
assert c.inc() == 11
assert c.inc(4) == 15
assert c.inc(2, 5) == 25

# 2. std/math
assert math.gcd(48, 18) == 6
assert math.lcm(12, 18) == 36
assert math.factorial(6) == 720
assert math.comb(10, 3) == 120
assert math.perm(5, 2) == 20
assert math.clamp(15, 0, 10) == 10
assert math.clamp(-5, 0, 10) == 0
assert math.isclose(math.sin(math.PI / 2.0), 1.0)
assert math.isclose(math.deg(math.PI), 180.0)
assert math.isclose(math.rad(180.0), math.PI)
assert math.isclose(math.hypot(3.0, 4.0), 5.0)
assert math.isclose(math.log2(8.0), 3.0)
assert math.isclose(math.log10(1000.0), 3.0)
assert math.isprime(2) and math.isprime(97) and (not math.isprime(100))
assert math.pow(2, 10) == 1024
assert math.trunc(3.9) == 3 and math.trunc(-3.9) == -3
assert math.isclose(math.lerp(10.0, 20.0, 0.25), 12.5)
assert math.isclose(math.cosh(0.0), 1.0)
assert math.isclose(math.acos(1.0), 0.0)
assert math.isfinite(42.0) and (not math.isfinite(math.INF)) and (not math.isfinite(math.NAN))

# 3. std/json
let sample = {
    "name": "Ngôn ngữ T",
    "version": 3,
    "active": true,
    "nothing": nil,
    "items": [1, 2.5, "a\nb\t\"c\\d"]
}
let dumped = json.dumps(sample)
let loaded = json.loads(dumped)
assert loaded["name"] == "Ngôn ngữ T"
assert loaded["version"] == 3
assert loaded["active"] == true
assert loaded["nothing"] == nil
assert loaded["items"][0] == 1
assert loaded["items"][1] == 2.5
assert loaded["items"][2] == "a\nb\t\"c\\d"
let pretty = json.dumps({"a": [1, 2]}, 2)
assert json.loads(pretty)["a"][1] == 2
assert json.loads("\"\\u0041\\u0042\"") == "AB"

# 4. std/path
assert path.join("a", "b", "c.t") == "a/b/c.t"
assert path.basename("dir/sub/file.tb") == "file.tb"
assert path.dirname("dir/sub/file.tb") == "dir/sub"
assert path.ext("dir/sub/file.tb") == ".tb"
assert path.stem("dir/sub/file.tb") == "file"
assert path.normpath("a/b/../c/./d") == "a/c/d"
assert path.to_slash("a\\b\\c.t") == "a/b/c.t"

# 5. std/time & std/os (bao gồm kiểm thử bảo mật zip-slip)
let t0 = time.clock()
let mres = time.measure(fn(): 7 * 6)
assert mres[0] == 42 and mres[1] >= 0.0
assert time.time() > 1700000000.0
assert len(os.sha256("abc")) == 64
assert os.sha256("abc") == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"

let tmp = os.mktemp("t_stdtest")
let zpath = path.join(tmp, "test.zip")
let outdir = path.join(tmp, "out")
os.makedirs(path.join(tmp, "nested/dir"))
assert os.isdir(path.join(tmp, "nested/dir"))
os.zip_pack({"hello.txt": "xin chao", "sub/a.t": "x = 1\n"}, zpath)
let files = os.zip_unpack(zpath, outdir)
assert files.len == 2
assert os.read_file(path.join(outdir, "hello.txt")) == "xin chao"
assert os.read_file(path.join(outdir, "sub/a.t")) == "x = 1\n"
os.copy_file(path.join(outdir, "hello.txt"), path.join(outdir, "hello_copy.txt"))
assert os.read_file(path.join(outdir, "hello_copy.txt")) == "xin chao"

# Kiểm thử chặn Zip-Slip (CWE-22)
mut caught_zipslip = false
try:
    os.zip_pack({"../evil.txt": "bad"}, zpath)
except:
    caught_zipslip = true
assert caught_zipslip

os.rmtree(tmp)
assert not os.exists(tmp)
assert time.clock() >= t0

# 6. Toán tử thao tác bit (&, |, ^, ~, <<, >>, &=, |=, ^=, <<=, >>=) + JIT
assert (0b1100 & 0b1010) == 0b1000
assert (0b1100 | 0b1010) == 0b1110
assert (0b1100 ^ 0b1010) == 0b0110
assert (~0) == -1
assert (~(-1)) == 0
assert (1 << 10) == 1024
assert (1024 >> 3) == 128
mut bw = 0x0F
bw |= 0xF0
assert bw == 0xFF
bw &= 0x3C
assert bw == 0x3C
bw ^= 0x0F
assert bw == 0x33
bw <<= 2
assert bw == 0xCC
bw >>= 1
assert bw == 0x66

fn jit_bit_hash(n):
    mut h = 0x12345678
    for i in range(n):
        h = (h ^ (i << 3)) + (h >> 2)
        h = (h & 0x7FFFFFFF) | (~i & 3)
    return h

assert jit_bit_hash(5000) == 188391165
for _ in range(60):
    assert jit_bit_hash(20) == 1490954924

# 7. Tham số biến thiên (*args) cho hàm, lambda, và struct method
fn sum_with_base(base, *nums):
    mut s = base
    for x in nums:
        s = s + x
    return s

assert sum_with_base(10) == 10
assert sum_with_base(10, 1, 2, 3, 4) == 20
let va_lam = fn(prefix, *parts): prefix + ":" + "-".join([str(p) for p in parts])
assert va_lam("id", 10, 20, 30) == "id:10-20-30"

struct Logger:
    tag
    fn fmt(self, *items):
        return "[" + self.tag + "] " + ", ".join([str(x) for x in items])

let lg = Logger("INFO")
assert lg.fmt("ok", 200, true) == "[INFO] ok, 200, true"

# 8. Khối try / except / finally
mut fin_log = []
fn test_fin_ret(x):
    try:
        fin_log.append("try:" + str(x))
        if x < 0:
            raise "neg"
        return x * 10
    except neg:
        fin_log.append("catch")
        return -1
    finally:
        fin_log.append("fin:" + str(x))

assert test_fin_ret(5) == 50
assert test_fin_ret(-3) == -1
assert fin_log == ["try:5", "fin:5", "try:-3", "catch", "fin:-3"]

mut uncaught_fin = false
mut outer_caught = false
try:
    try:
        raise "boom"
    finally:
        uncaught_fin = true
except boom:
    outer_caught = true
assert uncaught_fin and outer_caught

# 9. std/random
random.seed(42)
let r1 = [random.randint(1, 100) for _ in range(5)]
random.seed(42)
let r2 = [random.randint(1, 100) for _ in range(5)]
assert r1 == r2
let uval = random.uniform(5.0, 10.0)
assert uval >= 5.0 and uval < 10.0
assert random.randrange(10, 20, 2) % 2 == 0
mut deck = [1, 2, 3, 4, 5, 6]
random.shuffle(deck)
assert len(deck) == 6
let smp = random.sample(deck, 3)
assert len(smp) == 3
assert random.choice(deck) in deck
let gval = random.normal(0.0, 1.0)
assert math.isfinite(gval)

# 10. std/re
assert re.fullmatch("^[a-z]+_[0-9]{2,4}$", "user_2026")
assert not re.fullmatch("^[a-z]+_[0-9]{2,4}$", "user_2")
let m = re.search("(\\w+)@(\\w+\\.\\w+)", "Liên hệ: admin@tlang.org ngay")
assert m != nil
assert m["match"] == "admin@tlang.org"
assert m["groups"] == ["admin", "tlang.org"]
assert re.findall("\\d+", "a12b345c6") == ["12", "345", "6"]
assert re.sub("\\s+", "-", "xin   chao\tngon ngu T") == "xin-chao-ngon-ngu-T"
assert re.split("[,;]\\s*", "a, b;  c,d") == ["a", "b", "c", "d"]
assert re.test("^foo|bar$", "xxbar")

# 11. std/csv
let csv_txt = csv.dumps([["name", "note", "score"], ["An", "Gioi, \"Xuat sac\"", 95], ["Binh", "Kha", 82]])
let csv_rows = csv.loads(csv_txt)
assert csv_rows.len == 3
assert csv_rows[1][1] == "Gioi, \"Xuat sac\""
let csv_dicts = csv.loads_dicts(csv_txt)
assert csv_dicts.len == 2 and csv_dicts[0]["name"] == "An" and csv_dicts[1]["score"] == "82"
let csv_txt2 = csv.dumps_dicts(csv_dicts, ["name", "note", "score"])
assert csv.loads(csv_txt2) == csv_rows

# 12. std/http
let enc = http.urlencode("xin chào & a=b")
assert http.urldecode(enc) == "xin chào & a=b"
let q = http.parse_query("?page=2&q=ngon+ngu+T&tag=ai")
assert q["page"] == "2" and q["q"] == "ngon ngu T" and q["tag"] == "ai"
let u = http.parse_url("https://api.tlang.org:8443/v1/search?q=jit#top")
assert u["scheme"] == "https" and u["host"] == "api.tlang.org" and u["port"] == 8443
assert u["path"] == "/v1/search" and u["query"] == "q=jit" and u["fragment"] == "top"

print("STDLIB OK")
