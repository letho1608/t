# tools/tdriver.t — chạy chương trình .t bằng evaluator self-host đã bundle.
# Dùng sau khi bundle+biên dịch (không cần Python):
#   trun tdriver.tb tests/demo.t
# Chạy thử bằng interpreter (kết quả phải giống):
#   python t.py tools/tdriver.t tests/demo.t
import eval

args = argv()
p = args[args.len - 1]
f = open(p, "r")
src = read(f)
close(f)
eval.run_src(src)
