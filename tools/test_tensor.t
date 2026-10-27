# test_tensor.t — Kiểm thử toàn diện Tensor n-D trên VM C + benchmark matmul
import os
import path
import time

trun = os.exepath()
tmp = os.mktemp("ttensor")

r0 = os.run([trun, "tests/tensor.t"])
assert r0["rc"] == 0 and "TENSOR OK" in r0["out"], "tests/tensor.t failed: " + r0["out"] + r0["err"]
print("OK tests/tensor.t")

extra_src = path.join(tmp, "extra.t")
code = "let a = zeros([2, 3, 4])\nassert a.shape == [2, 3, 4] and len(a) == 2 and a.sum() == 0.0\nlet b = ones([2, 3])\nassert b.shape == [2, 3] and b.sum() == 6.0\nlet c = arange(24).reshape([2, 3, 4])\nassert c[0, 0, 0] == 0.0 and c[1, 2, 3] == 23.0\nc[0, 1, 2] = 99.0\nassert c[0, 1, 2] == 99.0\nc[0, 1, 2] = 6.0\nlet sub = c[1]\nassert sub.shape == [3, 4] and sub[0, 0] == 12.0\nsub[0, 0] = -5.0\nassert c[1, 0, 0] == -5.0\nsub[0, 0] = 12.0\nlet m = arange(6).reshape([2, 3])\nlet mt = m.T\nassert mt.shape == [3, 2] and mt[2, 1] == 5.0\nassert mt.tolist() == [[0.0, 3.0], [1.0, 4.0], [2.0, 5.0]]\nlet x = tensor([[1, 2, 3], [4, 5, 6]])\nlet y = tensor([10, 20, 30])\nassert (x + y).tolist() == [[11.0, 22.0, 33.0], [14.0, 25.0, 36.0]]\nlet col = tensor([[100], [200]])\nassert (x + col).tolist() == [[101.0, 102.0, 103.0], [204.0, 205.0, 206.0]]\nassert x.sum(0).tolist() == [5.0, 7.0, 9.0]\nassert x.sum(1).tolist() == [6.0, 15.0]\nassert x.mean(0).tolist() == [2.5, 3.5, 4.5]\nassert x.max(1).tolist() == [3.0, 6.0]\nassert x.min(0).tolist() == [1.0, 2.0, 3.0]\nlet p = arange(6).reshape([2, 3])\nlet q = arange(12).reshape([3, 4])\nassert (p @ q).tolist() == [[20.0, 23.0, 26.0, 29.0], [56.0, 68.0, 80.0, 92.0]]\nprint(\"EXTRA-OK\")\n"
os.write_file(extra_src, code)

r1 = os.run([trun, extra_src])
assert r1["rc"] == 0 and "EXTRA-OK" in r1["out"], "extra tensor test failed: " + r1["out"] + r1["err"]
print("OK extra n-D tensor semantics")

# Benchmark matmul 120x120
b_src = path.join(tmp, "bmm.t")
b_code = "let N = 120\nlet a = arange(N * N).reshape([N, N])\nlet b = arange(N * N).reshape([N, N])\nlet c = a @ b\nprint(int(c[0, 0]))\n"
os.write_file(b_src, b_code)
t0 = time.clock()
rb = os.run([trun, b_src])
dt = int((time.clock() - t0) * 1000.0)
assert rb["rc"] == 0, "bmm failed: " + rb["err"]
print("OK matmul 120x120:", rb["out"].strip(), "(" + str(dt) + " ms)")
os.rmtree(tmp)
print("TENSOR-ALL-GREEN")
