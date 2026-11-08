# tools/tcheck.t — Công cụ dòng lệnh kiểm tra kiểu tĩnh cho Ngôn ngữ T (v6.0 L10)
import os
import typecheck

fn main():
    raw = argv()
    a = raw[1:raw.len] if (raw.len > 0 and (raw[0].endswith(".t") or raw[0].endswith(".tb"))) else raw
    if a.len == 0 or a[0] == "-h" or a[0] == "--help":
        print("Cach dung: trun tools/tcheck.t <file1.t> [file2.t ...]")
        return nil
    mut total_errs = 0
    for f in a:
        if not os.exists(f):
            os.stderr_write("tcheck: khong tim thay file " + f + "\n")
            total_errs = total_errs + 1
            continue
        res = typecheck.check_file(f)
        if res["ok"]:
            print("[OK] " + f)
        else:
            for e in res["errors"]:
                os.stderr_write(f + ": " + e + "\n")
                total_errs = total_errs + 1
    if total_errs > 0:
        os.exit(1)

main()
