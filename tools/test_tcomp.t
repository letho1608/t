# test_tcomp.t — Kiểm chứng compiler T thuần T (self/tcomp.t + self/tcli.t, v3.0)
# Quy trình:
#   1. Self-compile stage1 (tcomp.tb) -> stage2 (tcomp2.tb) -> stage3 (tcomp3.tb)
#   2. Kiểm tra fixed-point: sha256(tcomp2.tb) == sha256(tcomp3.tb)
#   3. Dịch và chạy toàn bộ suite tests/*.t bằng stage1 & stage2, kiểm tra byte-identical
#   4. Kiểm tra GC quét dày (TVM_GCTHRESH=3), load() tự động dịch .t -> .tb, báo lỗi file:dòng (LN01)
#   5. Kiểm tra evaluator self-host (self/eval.t qua tools/tdriver.t) khớp kết quả với VM C
import os
import path

trun = os.exepath()
tmp = os.mktemp("tct")

CASES = [
    ["tests/demo.t", false], ["tests/core.t", false],
    ["tests/assert.t", false], ["tests/jit.t", false],
    ["tests/closure.t", false], ["tests/fstr.t", false],
    ["tests/mem.t", false], ["tests/bytesem.t", false], ["tests/mods.t", true],
    ["tests/v03.t", true], ["tests/selfhost.t", true],
    ["tests/fmtself.t", true], ["tests/hoist.t", false],
    ["tests/builtins.t", false], ["tests/lang.t", false],
    ["tests/tryflow.t", false], ["tests/jit2.t", false],
    ["tests/gcstress.t", false], ["tests/tensor.t", false],
    ["tests/jitlist.t", false], ["tests/lambda.t", false],
    ["tests/jitmain.t", false], ["tests/agtest.t", false],
    ["tests/aitest.t", false], ["tests/stdlib.t", false],
    ["tests/jitrec.t", false], ["tests/jitdict.t", false],
    ["tests/loadmod_m.t", false], ["tests/loadmod.t", false]
]

EVAL_CASES = [
    "tests/demo.t", "tests/core.t", "tests/assert.t", "tests/closure.t",
    "tests/fstr.t", "tests/hoist.t", "tests/builtins.t", "tests/lang.t",
    "tests/tryflow.t", "tests/bytesem.t", "tests/tensor.t",
    "tests/lambda.t", "tests/loadmod.t"
]

fn compile_with(stage, src, out, bundle):
    mut args = [trun, stage]
    if bundle:
        args.append("-b")
    args.append("-q")
    args.append(src)
    args.append(out)
    r = os.run(args)
    if r["rc"] != 0 or not os.exists(out):
        print("FAIL compile", stage, src, "rc=" + str(r["rc"]), r["out"], r["err"])
        return false
    return true

fn main():
    mut ok = true
    s1 = "tcomp.tb"
    s2 = path.join(tmp, "tcomp2.tb")
    s3 = path.join(tmp, "tcomp3.tb")
    cli = "self/tcli.t"

    if not compile_with(s1, cli, s2, true):
        print("FAIL self-compile stage2")
        os.exit(1)
    if not compile_with(s2, cli, s3, true):
        print("FAIL self-compile stage3")
        os.exit(1)
    h2 = os.sha256_file(s2)
    h3 = os.sha256_file(s3)
    if h2 != h3:
        print("FAIL khong co fixed point:", h2[0:12], "!=", h3[0:12])
        os.exit(1)
    print("OK self-hosting fixed point (" + h2[0:12] + ")")

    os.setenv("TCOMP", s2)
    if os.exists("tests/loadmod_m.tb"):
        os.remove("tests/loadmod_m.tb")

    mut outputs = {}
    for item in CASES:
        src = item[0]
        b = item[1]
        name = path.basename(src)
        a = path.join(tmp, "a_" + name + ".tb")
        c = path.join(tmp, "c_" + name + ".tb")
        if not compile_with(s1, src, a, b) or not compile_with(s2, src, c, b):
            ok = false
            continue
        if os.sha256_file(a) != os.sha256_file(c):
            ok = false
            print("FAIL stage1 != stage2 output", src)
            continue
        run = os.run([trun, a])
        if run["rc"] != 0:
            ok = false
            print("FAIL run", src, "rc=" + str(run["rc"]), run["out"], run["err"])
        else:
            outputs[src] = run["out"]
            print("OK " + src + " (" + str(os.read_file(a).len) + " bytes)")

    # GC quét dày (TVM_GCTHRESH=3)
    g_gc = path.join(tmp, "a_gcstress.t.tb")
    if os.exists(g_gc):
        os.setenv("TVM_GCTHRESH", "3")
        r1 = os.run([trun, g_gc])
        os.setenv("TVM_GCTHRESH", "")
        if r1["rc"] != 0 or r1["out"] != outputs["tests/gcstress.t"]:
            ok = false
            print("FAIL gcstress voi TVM_GCTHRESH=3")
        else:
            print("OK gcstress voi GC quet day (TVM_GCTHRESH=3)")

    g_lm = path.join(tmp, "a_loadmod.t.tb")
    if os.exists(g_lm):
        os.setenv("TVM_GCTHRESH", "3")
        r1 = os.run([trun, g_lm])
        os.setenv("TVM_GCTHRESH", "")
        if r1["rc"] != 0 or r1["out"] != outputs["tests/loadmod.t"]:
            ok = false
            print("FAIL loadmod voi TVM_GCTHRESH=3")
        else:
            print("OK load() voi GC quet day (TVM_GCTHRESH=3)")
    if not os.exists("tests/loadmod_m.tb"):
        ok = false
        print("FAIL load() khong tu dich tests/loadmod_m.t")

    # Lỗi runtime báo đúng file:dòng (LN01)
    em = path.join(tmp, "errmod.t")
    ep = path.join(tmp, "errmain.t")
    eb = path.join(tmp, "errmain.tb")
    os.write_file(em, "fn chia(a, b):\n    mut t = a + 1\n    return t / b\n")
    os.write_file(ep, "import errmod\nprint(\"truoc\")\n\nx = errmod.chia(1, 0)\nprint(\"khong toi\")\n")
    if compile_with(s2, ep, eb, false):
        rv = os.run([trun, eb])
        if rv["rc"] == 0 or not ("errmod.t:3" in rv["err"]) or rv["out"].strip() != "truoc":
            ok = false
            print("FAIL vi tri loi runtime:", rv["err"])
        else:
            print("OK loi runtime bao dung file:dong (LN01)")
    else:
        ok = false

    # Evaluator self-host (eval.t qua tools/tdriver.t) chạy trên VM C khớp kết quả VM C
    drv = path.join(tmp, "tdriver.tb")
    if not compile_with(s2, "tools/tdriver.t", drv, true):
        ok = false
    else:
        mut bad = 0
        for src in EVAL_CASES:
            re = os.run([trun, drv, src])
            if re["rc"] != 0 or re["out"] != outputs[src]:
                ok = false
                bad = bad + 1
                print("FAIL eval.t-on-VM", src, re["out"], re["err"])
        if bad == 0:
            print("OK eval.t tren VM C khop VM C (" + str(EVAL_CASES.len) + " suite)")

    os.setenv("TCOMP", "")
    os.rmtree(tmp)
    if not ok:
        print("TCOMP-T-FAIL")
        os.exit(1)
    print("TCOMP-T-OK")

main()
