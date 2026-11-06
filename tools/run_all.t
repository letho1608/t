# run_all.t — Chạy toàn bộ suite kiểm thử của Ngôn ngữ T (v3.0, thuần T)
import os
import time

trun = os.exepath()

SUITES = [
    ["ops",     "tools/test_ops.t",    "OPS-OK"],
    ["tcomp-t", "tools/test_tcomp.t",  "TCOMP-T-OK"],
    ["dbg",     "tools/test_dbg.t",    "DBG-ALL-GREEN"],
    ["lsp",     "tools/test_lsp.t",    "LSP-ALL-GREEN"],
    ["tpkg",    "tools/test_tpkg.t",   "TPKG-ALL-GREEN"],
    ["jit",     "tools/test_jit.t",    "JIT-ALL-GREEN"],
    ["tensor",  "tools/test_tensor.t", "TENSOR-ALL-GREEN"]
]

T_TESTS = [
    "tests/demo.t", "tests/core.t", "tests/assert.t", "tests/jit.t",
    "tests/closure.t", "tests/fstr.t", "tests/mem.t", "tests/bytesem.t",
    "tests/mods.t", "tests/v03.t", "tests/selfhost.t", "tests/fmtself.t",
    "tests/hoist.t", "tests/builtins.t", "tests/lang.t", "tests/tryflow.t",
    "tests/jit2.t", "tests/gcstress.t", "tests/tensor.t", "tests/jitlist.t",
    "tests/lambda.t", "tests/jitmain.t", "tests/agtest.t", "tests/aitest.t",
    "tests/stdlib.t", "tests/jitrec.t", "tests/jitdict.t", "tests/loadmod.t",
    "tests/mmap_test.t", "tests/ffi_test.t", "tests/thread_test.t",
    "tests/gguf_test.t", "tests/safetensors_test.t", "tests/transformer_test.t",
    "tests/test_run_gguf.t", "tests/sqlite_test.t", "tests/base64_test.t",
    "tests/cuda_test.t", "tests/gemm_mt.t", "tests/fusedbwd.t",
    "tests/cublas_test.t", "tests/vulkan_test.t"
]

fn pad_right(s, w):
    mut out = s
    while out.len < w:
        out = out + " "
    return out

fn fmt_sec(dt):
    ms = int(dt * 100.0 + 0.5)
    sec = int(ms / 100)
    rem = ms - sec * 100
    return str(sec) + "." + ("0" + str(rem) if rem < 10 else str(rem)) + "s"

fn run_step(name, cmd, marker):
    t0 = time.clock()
    r = os.run(cmd)
    dt = time.clock() - t0
    ok = (r["rc"] == 0) and (marker == "" or (marker in r["out"]))
    tag = "PASS" if ok else "FAIL"
    print("  [" + tag + "] " + pad_right(name, 22) + " (" + fmt_sec(dt) + ")")
    if not ok:
        tail = (r["out"] + "\n" + r["err"]).strip().split("\n")
        mut start = tail.len - 12
        if start < 0:
            start = 0
        mut i = start
        while i < tail.len:
            print("         " + tail[i])
            i = i + 1
    return ok

fn main():
    t0 = time.clock()
    mut passed = 0
    mut failed = 0

    print("=== 1/2 Tool & Subsystem Suites (" + str(SUITES.len) + ") ===")
    for item in SUITES:
        if run_step(item[0], [trun, item[1]], item[2]):
            passed = passed + 1
        else:
            failed = failed + 1

    print("\n=== 2/2 Direct Execution Suite: trun tests/*.t (" + str(T_TESTS.len) + ") ===")
    for t in T_TESTS:
        if run_step(t, [trun, t], ""):
            passed = passed + 1
        else:
            failed = failed + 1

    if os.exists("tests/loadmod_m.tb"):
        os.remove("tests/loadmod_m.tb")

    total = passed + failed
    dt = time.clock() - t0
    print("\n====================================================")
    status = "ALL-GREEN" if failed == 0 else "FAILED"
    print("Summary: " + str(passed) + "/" + str(total) + " passed in " + fmt_sec(dt) + " -> " + status)
    if failed > 0:
        os.exit(1)

main()
