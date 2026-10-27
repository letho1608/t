# test_jit.t — Kiểm chứng x86-64 baseline JIT: entry + OSR + main + rec + dict + speedup
import os
import path
import time

trun = os.exepath()
tmp = os.mktemp("tjit")

fn compile_tb(name):
    src = path.join("tests", name + ".t")
    tb = path.join(tmp, name + ".tb")
    r = os.run([trun, "tcomp.tb", src, tb])
    assert r["rc"] == 0, "compile failed: " + name + " " + r["err"]
    return tb

fn run_env(tb, nojit, jitlog):
    os.setenv("TVM_NOJIT", "1" if nojit else "")
    os.setenv("TVM_JITLOG", "1" if jitlog else "")
    r = os.run([trun, tb])
    os.setenv("TVM_NOJIT", "")
    os.setenv("TVM_JITLOG", "")
    return r

fn count_sub(s, sub):
    return s.split(sub).len - 1

fn main():
    cases = [
        ["jit2", 14, 1],
        ["jit", 1, 0],
        ["jitlist", 2, 1],
        ["jitmain", 0, 0],
        ["jitrec", 9, 1],
        ["jitdict", 2, 1]
    ]
    for item in cases:
        name = item[0]
        min_entry = item[1]
        min_osr = item[2]
        tb = compile_tb(name)
        r_jit = run_env(tb, false, true)
        r_int = run_env(tb, true, false)
        assert r_jit["rc"] == 0 and r_int["rc"] == 0, name + " exit!=0"
        assert r_jit["out"] == r_int["out"], name + " stdout mismatch:\nJIT:\n" + r_jit["out"] + "\nINT:\n" + r_int["out"]
        n_entry = count_sub(r_jit["err"], "entry ok")
        n_osr = count_sub(r_jit["err"], "osr ok")
        n_main = count_sub(r_jit["err"], "main ok")
        assert n_entry >= min_entry, name + ": entry ok=" + str(n_entry) + " < " + str(min_entry)
        assert n_osr >= min_osr, name + ": osr ok=" + str(n_osr) + " < " + str(min_osr)
        if name == "jitmain":
            assert n_main >= 1, "jitmain: main ok=" + str(n_main) + " < 1"
        print("OK " + name + ": stdout khop interpreter, entry_ok=" + str(n_entry) + ", osr_ok=" + str(n_osr) + ", main_ok=" + str(n_main))

    # Microbench fib(33) + sum 3M loop: JIT phai nhanh hon interpreter >= 3x
    bench_src = path.join(tmp, "b.t")
    bench_tb = path.join(tmp, "b.tb")
    os.write_file(bench_src, "fn fib(n):\n    if n < 2:\n        return n\n    return fib(n - 1) + fib(n - 2)\nfn loop(n):\n    mut s = 0\n    mut i = 0\n    while i < n:\n        s = s + i\n        i = i + 1\n    return s\nprint(fib(33), loop(3000000))\n")
    rc = os.run([trun, "tcomp.tb", bench_src, bench_tb])
    assert rc["rc"] == 0
    t0 = time.clock()
    rj = run_env(bench_tb, false, false)
    dt_j = (time.clock() - t0) * 1000.0
    t1 = time.clock()
    ri = run_env(bench_tb, true, false)
    dt_i = (time.clock() - t1) * 1000.0
    assert rj["rc"] == 0 and ri["rc"] == 0
    assert rj["out"].strip() == ri["out"].strip()
    speedup = dt_i / (dt_j if dt_j > 1.0 else 1.0)
    print("OK speedup fib(33)+loop(3M): JIT=" + str(int(dt_j)) + "ms, interp=" + str(int(dt_i)) + "ms (" + str(int(speedup * 10) / 10.0) + "x)")
    assert speedup >= 3.0, "JIT speedup qua thap: " + str(speedup) + "x < 3x"
    os.rmtree(tmp)
    print("JIT-ALL-GREEN")

main()
