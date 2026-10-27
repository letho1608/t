# tools/tdriver.t — chạy chương trình .t bằng evaluator self-host (self/eval.t).
# Dùng:
#   trun tools/tdriver.t tests/demo.t
#   trun tdriver.tb tests/demo.t
import eval

args = argv()
p = args[args.len - 1]
f = open(p, "r")
src = read(f)
close(f)
eval.run_src(src)
