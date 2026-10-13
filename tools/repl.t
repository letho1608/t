# tools/repl.t — REPL của T chạy trên VM C (không cần Python).
# Build một lần:  trun tcomp.tb -b tools/repl.t trepl.tb
# Dùng:           trun            (trun tự nạp trepl.tb cạnh trun.exe)
# Câu lệnh 1 dòng chạy ngay; dòng kết thúc bằng ':' mở khối, khối kết thúc
# bằng dòng trống. Biểu thức đơn được in giá trị (khác nil). Thoát: exit / EOF.
# Biến, hàm, struct sống qua các lần nhập (evaluator self-host self/eval.t).
import eval
import host

fn run_chunk(src, env, mods):
    ast = host.parse_prog([host.scan(src), 0])
    stmts = ast[1]
    if stmts.len == 1 and stmts[0][0] == "expr":
        v = eval.eval_expr(stmts[0][1], env, mods)
        if v != nil:
            print(eval.bltin_str(v))
        return nil
    r = eval.exec_block(stmts, env, mods)
    if eval.is_sig(r, "err"):
        print("error: " + r[1])
    elif eval.is_sig(r, "ret") or eval.is_sig(r, "brk") or eval.is_sig(r, "cnt"):
        print("error: return/break ngoài hàm/vòng lặp")
    return nil

fn main():
    print("T v2.2 REPL (trun)  (khoi lenh ket thuc bang dong trong; thoat: exit)")
    env = eval.env_new(nil)
    mods = {}
    mut buf = ""
    while true:
        ln = input("... " if buf != "" else ">>> ")
        if ln == nil:
            break
        s = ln.strip()
        if buf == "" and s == "exit":
            break
        if s != "" and (buf != "" or s.endswith(":")):
            buf += ln + "\n"
            continue
        if s != "":
            buf = ln + "\n"
        if buf == "":
            continue
        try:
            run_chunk(buf, env, mods)
        except:
            print("error: " + eval.strip_terr(err))
        buf = ""

main()
