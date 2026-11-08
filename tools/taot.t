# tools/taot.t — Công cụ CLI dịch mã nguồn T sang C99 / Rust hoặc thư viện mã máy gốc (.dll/.so) (v7.0 J9/AOT)
import os
import aot

fn main():
    raw = argv()
    a = raw[1:raw.len] if (raw.len > 0 and (raw[0].endswith(".t") or raw[0].endswith(".tb"))) else raw
    if a.len == 0 or a[0] == "-h" or a[0] == "--help":
        print("Cach dung: trun tools/taot.t [--emit-c | --emit-rs | --compile] <file.t> [out]")
        return nil
    mode = a[0]
    if mode == "--emit-c" and a.len >= 2:
        src = os.read_file(a[1])
        c_code = aot.transpile_to_c(src)
        if a.len >= 3:
            os.write_file(a[2], c_code)
        else:
            print(c_code)
    elif mode == "--emit-rs" and a.len >= 2:
        src = os.read_file(a[1])
        rs_code = aot.transpile_to_rust(src)
        if a.len >= 3:
            os.write_file(a[2], rs_code)
        else:
            print(rs_code)
    else:
        in_file = a[1] if mode == "--compile" and a.len >= 2 else a[0]
        src = os.read_file(in_file)
        print(aot.transpile_to_c(src))

main()
