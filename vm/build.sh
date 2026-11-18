#!/bin/sh
# build.sh — build trun on POSIX (Linux/macOS x86-64 / ARM64) + self-host tcomp.tb/trepl.tb.
#   sh vm/build.sh            # -> ./trun + tcomp.tb + trepl.tb
#   ./trun tools/run_all.t    # -> ALL-GREEN
set -e
cd "$(dirname "$0")/.."
CC="${CC:-cc}"
VEC=""
if $CC --version 2>/dev/null | grep -q "Free Software Foundation"; then VEC=" -fvect-cost-model=dynamic"; fi
CFLAGS="${CFLAGS:--O2$VEC -std=c99 -Wall}"
echo "[build] $CC $CFLAGS"
$CC $CFLAGS -o trun vm/tvm.c vm/jit_x64.c vm/jit_arm64.c vm/tdbg.c vm/trun.c -lm
echo "[build] OK: ./trun"
if [ -f tcomp.tb ]; then
    ./trun tcomp.tb -b self/tcli.t tcomp.new.tb >/dev/null
    mv -f tcomp.new.tb tcomp.tb
    ./trun tcomp.tb -b tools/repl.t trepl.tb >/dev/null
    echo "[build] OK: tcomp.tb + trepl.tb (self-hosted)"
fi
