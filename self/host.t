# self/host.t — glue self-host: scan(src) = indent_lines + tokenize
# Dùng:
#   import host
#   toks = host.scan("let x = 1\nprint(x)")
#   st = [toks, 0]
#   ast = host.parse_prog(st)
import lex
import parse
import parse2

# Token mang số dòng GỐC trong file (dạng chuỗi) -> lỗi parse báo đúng dòng.
fn scan(src):
    mut toks = []
    mut no = "1"
    for p in parse.indent_lines(src):
        no = str(p[0])
        ln = p[1]
        if ln == "<IND>":
            toks.append(["IND", "{", no])
        elif ln == "<DED>":
            toks.append(["DED", "}", no])
        else:
            mut subt = []
            try:
                subt = lex.tokenize(ln)
            except:
                raise "line " + no + ": " + err
            # token đầu dòng logic: bol=true (phân biệt `if` câu lệnh với a if c else b)
            mut bol = not p[2]
            for tk in subt:
                if tk[0] != "EOF":
                    toks.append([tk[0], tk[1], no, bol])
                    bol = false
    toks.append(["EOF", "", no])
    return toks

fn parse_prog(st):
    mut ss = []
    while st[0][st[1]][0] != "EOF":
        parse2.add_mark(ss, st)
        parse2.add_stmt(ss, parse2.parse_stmt(st))
    return ["prog", ss]

fn reset_type_anns():
    return parse2.reset_type_anns()

fn get_type_anns():
    return parse2.TYPE_ANNS[0]


