# tests/jitarm64.t — J7 skeleton: emitter AArch64 byte-verify + dump
# Máy x86-64 không chạy được mã ARM64; kiểm chứng byte phát ra theo oracle
# viết tay (t_arm64_selftest) và dump file để kiểm bằng `objdump -m aarch64`.
# JIT x64 trên máy này vẫn phải hoạt động bình thường (đã có suite jit*.t).

let sys = load("__sys")
import os
import path

fn main():
    assert sys.arm64_selftest() == true, "arm64 emitter lech oracle byte"
    print("arm64 selftest OK")

    tmp = os.mktemp("jarm64")
    out = path.join(tmp, "arm64.bin")
    assert sys.arm64_dump(out) == true
    assert os.exists(out)
    data = os.read_file(out)
    assert data.len == 52, "dump phai 52 byte (16+16+20), got " + str(data.len)
    os.remove(out)
    os.rmtree(tmp)

    # dump tới thư mục không tồn tại -> false êm, không crash
    assert sys.arm64_dump(path.join(tmp, "nope", "x.bin")) == false
    print("arm64 dump OK")

main()
print("All jit-arm64 tests passed successfully!")
