/* trun.c — runner: nạp .tb (magic TB01 + u16 nglobals + u32 len + code) và chạy VM.
 *   trun prog.tb [args...]     chạy chương trình
 *   trun -d prog.tb            debugger từng lệnh
 *   trun                       REPL (nạp trepl.tb cạnh trun.exe, hoặc $TREPL)
 */
#include "tvm.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifdef _WIN32
#include <io.h>
#include <fcntl.h>
#define TRUN_SEP '\\'
#else
#include <unistd.h>
#define TRUN_SEP '/'
#endif

#define TRUN_VERSION "2.2"

static int run_tb(const char *path, char **xargv, int xargc, int dbg) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "no file %s\n", path); return 1; }
    char magic[4];
    if (fread(magic, 1, 4, f) != 4 || memcmp(magic, "TB01", 4)) {
        fprintf(stderr, "bad magic\n"); fclose(f); return 1;
    }
    uint16_t ng = 0;
    if (fread(&ng, 2, 1, f) != 1) { fprintf(stderr, "bad nglobals\n"); fclose(f); return 1; }
    uint32_t len = 0;
    if (fread(&len, 4, 1, f) != 1) { fprintf(stderr, "bad len\n"); fclose(f); return 1; }
    TProg p; memset(&p, 0, sizeof(p));
    p.nglobals = ng;
    p.code = (uint8_t*)malloc(len ? len : 1); p.len = len;
    if (len && fread(p.code, 1, len, f) != len) {
        fprintf(stderr, "short read\n"); fclose(f); free(p.code); return 1;
    }
    {
        /* bảng dòng tuỳ chọn: "LN01" u16 nfile (u16 len + tên)* u32 n (u32 pc, u32 dòng, u16 file)* */
        char mg[4];
        if (fread(mg, 1, 4, f) == 4 && !memcmp(mg, "LN01", 4)) {
            uint16_t nf = 0;
            uint32_t nl = 0;
            int ok = fread(&nf, 2, 1, f) == 1;
            if (ok) p.lfiles = (char**)calloc(nf ? nf : 1, sizeof(char*));
            for (uint16_t i = 0; ok && i < nf; i++) {
                uint16_t ln = 0;
                ok = fread(&ln, 2, 1, f) == 1;
                if (!ok) break;
                p.lfiles[i] = (char*)calloc((size_t)ln + 1, 1);
                ok = fread(p.lfiles[i], 1, ln, f) == ln;
                p.nlfiles = (uint16_t)(i + 1);
            }
            if (ok) ok = fread(&nl, 4, 1, f) == 1;
            if (ok && nl) {
                p.lpc = (uint32_t*)malloc(sizeof(uint32_t) * nl);
                p.lline = (uint32_t*)malloc(sizeof(uint32_t) * nl);
                p.lfile = (uint16_t*)malloc(sizeof(uint16_t) * nl);
                for (uint32_t i = 0; ok && i < nl; i++) {
                    ok = fread(&p.lpc[i], 4, 1, f) == 1 && fread(&p.lline[i], 4, 1, f) == 1 &&
                         fread(&p.lfile[i], 2, 1, f) == 1;
                    if (ok) p.nlines = i + 1;
                }
            }
        }
    }
    fclose(f);
    p.xargv = xargv;
    p.xargc = xargc;
    if (dbg) dbg_enable();
    TVM vm; tvm_init(&vm, &p);
    int rc = tvm_run(&vm);
    fflush(stdout);
    if (rc) {
        const char *file; uint32_t line;
        if (tvm_pc_line(&p, vm.errpc, &file, &line))
            fprintf(stderr, "T error: %s\n  tại %s:%u\n", vm.errmsg, file, (unsigned)line);
        else
            fprintf(stderr, "T error: %s\n", vm.errmsg);
    }
    tvm_free(&vm); free(p.code);
    for (uint16_t i = 0; i < p.nlfiles; i++) free(p.lfiles[i]);
    free(p.lfiles); free(p.lpc); free(p.lline); free(p.lfile);
    return rc ? 1 : 0;
}

/* trepl.tb: $TREPL, rồi cạnh file thực thi, rồi thư mục hiện tại */
static int find_repl(const char *argv0, char *out, size_t n) {
    const char *env = getenv("TREPL");
    FILE *f;
    if (env && *env) {
        f = fopen(env, "rb");
        if (f) { fclose(f); snprintf(out, n, "%s", env); return 1; }
    }
    if (argv0) {
        const char *s1 = strrchr(argv0, '/');
        const char *s2 = strrchr(argv0, TRUN_SEP);
        const char *sl = s1 > s2 ? s1 : s2;
        if (sl) {
            snprintf(out, n, "%.*s%ctrepl.tb", (int)(sl - argv0), argv0, TRUN_SEP);
            f = fopen(out, "rb");
            if (f) { fclose(f); return 1; }
        }
    }
    snprintf(out, n, "trepl.tb");
    f = fopen(out, "rb");
    if (f) { fclose(f); return 1; }
    return 0;
}

static void usage(void) {
    fprintf(stderr,
        "T %s — usage:\n"
        "  trun prog.tb [args...]   chạy bytecode\n"
        "  trun -d prog.tb          debugger\n"
        "  trun                     REPL (cần trepl.tb: trun tcomp.tb -b tools/repl.t trepl.tb)\n"
        "  trun --version\n", TRUN_VERSION);
}

int main(int argc, char **argv) {
    int ai = 1, dbg = 0;
#ifdef _WIN32
    /* nhị phân: không CRLF — output phải khớp byte với t.py */
    _setmode(_fileno(stdout), _O_BINARY);
    _setmode(_fileno(stderr), _O_BINARY);
    _setmode(_fileno(stdin), _O_BINARY);
#endif
    if (argc < 2) {
        char rp[1024];
        if (!find_repl(argv[0], rp, sizeof(rp))) { usage(); return 2; }
        return run_tb(rp, NULL, 0, 0);
    }
    if (!strcmp(argv[1], "--version") || !strcmp(argv[1], "-v")) {
        printf("T %s (trun VM C, JIT x86-64)\n", TRUN_VERSION);
        return 0;
    }
    if (!strcmp(argv[1], "-h") || !strcmp(argv[1], "--help")) { usage(); return 0; }
    if (!strcmp(argv[1], "-d")) { dbg = 1; ai++; }
    if (ai >= argc) { usage(); return 2; }
    return run_tb(argv[ai], argv + ai + 1, argc - ai - 1, dbg);
}
