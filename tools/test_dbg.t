# test_dbg.t — Kiểm thử debugger tích hợp (trun -d) trên bytecode .tb
import os
import path

trun = os.exepath()
tmp = os.mktemp("tdbg")
src = path.join(tmp, "dbg_sample.t")
tb = path.join(tmp, "dbg_sample.tb")

code = "fn binh_phuong(n):\n    mut r = n * n\n    return r\n\nmut tong = 0\nfor i in range(4):\n    tong = tong + binh_phuong(i)\nprint(tong)\n"
os.write_file(src, code)

rc = os.run([trun, "tcomp.tb", src, tb])
assert rc["rc"] == 0, "compile failed: " + rc["err"]

script = "d\ns\ns\np 0\nl\nb 5\nc\nl\nc\nc\nc\nc\n"
rd = os.run([trun, "-d", tb], script)
assert rd["rc"] == 0, "trun -d failed: " + rd["err"]

out = rd["out"]
checks = [
    ["prompt tdbg", "(tdbg)"],
    ["disasm JMP/MKFN", "MKFN"],
    ["stack", "stack"],
    ["frame depth", "frame depth="],
    ["dat break", "break 5"],
    ["hit break", "[dbg pc=5"],
    ["locals l0", "l0 = 0"],
    ["ket qua cuoi 14", "14"]
]
for item in checks:
    if not (item[1] in out):
        print("FAIL [" + item[0] + "]: khong thay '" + item[1] + "' trong output:\n" + out)
        os.exit(1)

os.rmtree(tmp)
print("DBG-ALL-GREEN")
