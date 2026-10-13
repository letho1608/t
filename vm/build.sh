#!/bin/sh
# build.sh — build trun trên POSIX (Linux/macOS x86-64) + bootstrap tcomp.tb/trepl.tb.
# Đã verify: Debian (gcc 13, glibc) x86-64 — toàn bộ tools/run_all.py xanh.
#   sh vm/build.sh            # -> ./trun (+ tcomp.tb, trepl.tb nếu có python3)
#   python3 vm/tcomp.py tests/demo.t /tmp/demo.tb && ./trun /tmp/demo.tb
set -e
cd "$(dirname "$0")/.."
CC="${CC:-cc}"
# gcc 12+ ở -O2 chỉ vector hoá vòng lặp biết trước số lần lặp ("very-cheap"): kernel
# tensor cần cost model động (phép cộng dồn vẫn tuần tự -> kết quả trùng bit)
VEC=""
if $CC --version 2>/dev/null | grep -q "Free Software Foundation"; then VEC=" -fvect-cost-model=dynamic"; fi
CFLAGS="${CFLAGS:--O2$VEC -std=c99 -Wall}"
echo "[build] $CC $CFLAGS"
$CC $CFLAGS -o trun vm/tvm.c vm/jit_x64.c vm/tdbg.c vm/trun.c -lm
echo "[build] OK: ./trun"
PY="${PYTHON:-python3}"
if command -v "$PY" >/dev/null 2>&1; then
    "$PY" vm/tcomp.py -b self/tcli.t tcomp.tb >/dev/null
    ./trun tcomp.tb -b tools/repl.t trepl.tb >/dev/null
    echo "[build] OK: tcomp.tb + trepl.tb"
else
    echo "[build] không có $PY: bỏ qua bootstrap tcomp.tb"
fi
