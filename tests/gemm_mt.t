# tests/gemm_mt.t — T2: GEMM đa luồng + chặn cache (vm/tensor_impl.h)
# Bằng chứng:
#   1) output của big.t phải TRÙNG BYTE giữa TVM_GEMM_THREADS=1 (tuần tự),
#      =6 (đa luồng) và auto -> song song không đổi kết quả (trùng bit);
#   2) kết quả phải khớp tham chiếu ba vòng trong T (tính đúng);
#   3) TVM_GEMM_THREADS=0/1 và giá trị lớn hơn số nhân đều không crash.
import os
import path

trun = os.exepath()
tmp = os.mktemp("tgemm")

BIG = "m = 192\nk = 192\nn = 192\nmut A = zeros([m, k])\nmut B = zeros([k, n])\nfor i in range(m):\n    for q in range(k):\n        A[i, q] = ((i * 37 + q * 11) % 101) * 0.25 - 3.0\nfor q in range(k):\n    for j in range(n):\n        B[q, j] = ((q * 13 + j * 7) % 97) * 0.125 - 2.0\nC = A @ B\nfor i in range(m):\n    for j in range(n):\n        print(C[i, j])\n"

REF = "m = 40\nk = 40\nn = 40\nmut A = zeros([m, k])\nmut B = zeros([k, n])\nfor i in range(m):\n    for q in range(k):\n        A[i, q] = ((i * 5 + q * 3) % 17) * 0.1 - 0.8\nfor q in range(k):\n    for j in range(n):\n        B[q, j] = ((q * 7 + j * 2) % 13) * 0.05 - 0.3\nC = A @ B\nmut R = zeros([m, n])\nfor i in range(m):\n    for j in range(n):\n        mut acc = 0.0\n        for q in range(k):\n            acc = acc + A[i, q] * B[q, j]\n        R[i, j] = acc\nd = (C - R).abs().max()\nassert d < 0.000000001\nprint(d)\n"

fn write_src(name, src):
    p = path.join(tmp, name + ".t")
    os.write_file(p, src)
    return p

fn run_threads(src_path, threads):
    os.setenv("TVM_GEMM_THREADS", threads)
    r = os.run([trun, src_path])
    os.setenv("TVM_GEMM_THREADS", "")
    return r

fn main():
    big = write_src("big", BIG)
    ref = write_src("ref", REF)

    # 1) trùng bit: tuần tự vs đa luồng vs auto
    r1 = run_threads(big, "1")
    r6 = run_threads(big, "6")
    ra = run_threads(big, "")
    ok = r1["rc"] == 0 and r6["rc"] == 0 and ra["rc"] == 0
    assert ok, "big.t loi: " + r1["err"] + r6["err"] + ra["err"]
    lines = r1["out"].split("\n").len
    assert lines > 36000, "output qua ngan: " + str(lines)
    assert r1["out"] == r6["out"], "GEMM 6 luong khac tuan tu (khong trung bit)"
    assert r1["out"] == ra["out"], "GEMM auto khac tuan tu (khong trung bit)"

    # 2) đúng: khớp tham chiếu ba vòng của T (40x40x40, có số lẻ thập phân)
    q1 = run_threads(ref, "1")
    q6 = run_threads(ref, "6")
    assert q1["rc"] == 0 and q6["rc"] == 0, "ref.t loi: " + q1["err"] + q6["err"]
    assert q1["out"] == q6["out"], "ref da luong khac tuan tu"

    # 3) giá trị biên của env không crash: 0 (auto), 1, 999 (clamp về m/32)
    for v in ["0", "1", "999"]:
        rv = run_threads(ref, v)
        assert rv["rc"] == 0, "env TVM_GEMM_THREADS=" + v + " loi: " + rv["err"]
        assert rv["out"] == q1["out"], "env " + v + " cho ket qua khac"

    os.rmtree(tmp)
    print("GEMM-MT-ALL-GREEN")

main()
