# bench.t — Benchmark Ngôn ngữ T (Bytecode VM + x86-64 JIT) so sánh trực tiếp với CPython
import os
import path
import time

trun = os.exepath()
tmp = os.mktemp("tbench")

CASES = [
    ["fib(35) de quy",       "fib"],
    ["vong lap 10M",         "loop"],
    ["vong lap 10M (fn)",    "loopfn"],
    ["mandelbrot 200x200",   "mandel"],
    ["sieve 100kx10",        "sieve"],
    ["dict 100k str-key",    "dict"],
    ["method 2M Vec.step",   "method"],
    ["matmul 150x150",       "matmul"],
    ["nn MLP 2->16->1 XOR",  "nn"]
]

fn pad_right(s, w):
    mut out = s
    while out.len < w:
        out = out + " "
    return out

fn pad_left(s, w):
    mut out = s
    while out.len < w:
        out = " " + out
    return out

fn fmt_num(x, digits):
    mut scale = 1
    mut i = 0
    while i < digits:
        scale = scale * 10
        i = i + 1
    v = int(x * scale + 0.5)
    ip = int(v / scale)
    fp = v - ip * scale
    mut fs = str(fp)
    while fs.len < digits:
        fs = "0" + fs
    return str(ip) + "." + fs

fn best_time(cmd, runs):
    mut best = 1000000.0
    mut out = ""
    mut i = 0
    while i < runs:
        t0 = time.clock()
        r = os.run(cmd)
        dt = time.clock() - t0
        assert r["rc"] == 0, "command failed: " + " ".join(cmd) + "\n" + r["err"]
        out = r["out"].strip()
        if dt < best:
            best = dt
        i = i + 1
    return [best, out]

fn find_python():
    for cand in ["python", "python3"]:
        r = os.run([cand, "--version"])
        if r["rc"] == 0:
            return [cand, (r["out"] + r["err"]).strip()]
    return ["", ""]

fn main():
    py_info = find_python()
    py = py_info[0]
    py_ver = py_info[1]
    assert py != "", "Khong tim thay python de so sanh"

    print("==========================================================================")
    print("SO SANH HIEU NANG: T v3.0 (trun + x86-64 JIT) vs " + py_ver)
    print("==========================================================================")
    print(pad_right("Bai do", 22) + " | " + pad_left("Python (ms)", 12) + " | " + pad_left("T VM+JIT (ms)", 13) + " | " + pad_left("T nhanh gap", 12) + " | KQ")
    print("--------------------------------------------------------------------------")

    mut ratios = []
    for item in CASES:
        label = item[0]
        name = item[1]
        t_src = path.join("bench", name + ".t")
        py_src = path.join("bench", name + ".py")
        tb = path.join(tmp, name + ".tb")

        rc = os.run([trun, "tcomp.tb", "-q", t_src, tb])
        assert rc["rc"] == 0, "compile failed: " + t_src + " " + rc["err"]

        rp = best_time([py, py_src], 3)
        rt = best_time([trun, tb], 3)

        py_ms = rp[0] * 1000.0
        t_ms = rt[0] * 1000.0
        speedup = py_ms / (t_ms if t_ms > 0.1 else 0.1)
        ratios.append(speedup)

        match_ok = (rp[1] == rt[1]) or (name == "nn" and rt[1].startswith("loss_ok 1"))
        status = "OK" if match_ok else ("LECH: " + rt[1] + " != " + rp[1])

        print(pad_right(label, 22) + " | " +
              pad_left(fmt_num(py_ms, 1), 12) + " | " +
              pad_left(fmt_num(t_ms, 1), 13) + " | " +
              pad_left(fmt_num(speedup, 2) + "x", 12) + " | " + status)

    mut sum_r = 0.0
    mut log_sum = 0.0
    for r in ratios:
        sum_r = sum_r + r
        log_sum = log_sum + log(r)
    avg_r = sum_r / ratios.len
    geo_r = exp(log_sum / ratios.len)

    os.rmtree(tmp)
    print("--------------------------------------------------------------------------")
    print("Trung binh nhan (GeoMean): T nhanh gap " + fmt_num(geo_r, 2) + "x so voi Python")
    print("Trung binh cong (Avg)    : T nhanh gap " + fmt_num(avg_r, 2) + "x so voi Python")
    print("==========================================================================")

main()
