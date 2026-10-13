# tests/fmtself.t — formatter tự verify (chạy: python t.py tests/fmtself.t)
import fmt
import host
f = open("tests/demo.t", "r")
src = read(f)
close(f)
out = fmt.fmt_src(src)
assert fmt.fmt_src(out) == out, "fmt idempotent"
a = host.parse_prog([host.scan(src), 0])
b = host.parse_prog([host.scan(out), 0])
assert str(a) == str(b), "fmt keeps AST"
print("FMT OK")
