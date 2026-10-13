# tools/ffmt.t — CLI cho formatter (viết bằng chính T)
# Dùng:
#   python t.py tools/ffmt.t prog.t              # format tại chỗ
#   python t.py tools/ffmt.t --check prog.t      # exit 1 nếu chưa chuẩn
#   trun fmt.tb prog.t                           # tương tự, không cần Python
import fmt

args = argv()
mut check = false
for a in args:
    if a == "--check":
        check = true

mut bad = 0
for i in range(1, args.len):
    p = args[i]
    if p == "--check":
        continue
    f = open(p, "r")
    src = read(f)
    close(f)
    out = fmt.fmt_src(src)
    if out == src.strip():
        print("ok " + p)
    elif check:
        print("needs-fmt " + p)
        bad = 1
    else:
        g = open(p, "w")
        write(g, out + "\n")
        close(g)
        print("fmt " + p)
if bad == 1:
    raise "needs formatting"
