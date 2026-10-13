/* trun.c — runner: nạp .tb (magic TB01 + u16 nglobals + u32 len + code) và chạy VM.
 *   trun prog.tb [args...]     chạy chương trình
 *   trun -d prog.tb            debugger từng lệnh
 *   trun                       REPL (nạp trepl.tb cạnh trun.exe, hoặc $TREPL)
 */
#if !defined(_WIN32) && !defined(_DEFAULT_SOURCE)
#define _DEFAULT_SOURCE      /* readlink */
#endif
#include "tvm.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifdef _WIN32
#include <windows.h>
#include <io.h>
#include <fcntl.h>
#define TRUN_SEP '\\'
#else
#include <unistd.h>
#define TRUN_SEP '/'
#endif

#define TRUN_VERSION "2.4"

static int run_tb(const char *path, char **xargv, int xargc, int dbg) {
    TProg p; memset(&p, 0, sizeof(p));
    char msg[256];
    if (tvm_read_tb(path, &p, msg, sizeof(msg)) != 0) {
        fprintf(stderr, "%s\n", msg); return 1;
    }
    p.xargv = xargv;
    p.xargc = xargc;
    if (dbg) dbg_enable();
    TVM vm; tvm_init(&vm, &p);
    int rc = tvm_run(&vm);
    fflush(stdout);
    if (rc) {
        const char *file; uint32_t line;
        if (tvm_pc_line(vm.errprog ? vm.errprog : &p, vm.errpc, &file, &line))
            fprintf(stderr, "T error: %s\n  tại %s:%u\n", vm.errmsg, file, (unsigned)line);
        else
            fprintf(stderr, "T error: %s\n", vm.errmsg);
    }
    tvm_free(&vm);
    tvm_prog_release(&p);
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
    {
        static char exe[1024];
        DWORD k = GetModuleFileNameA(NULL, exe, sizeof(exe));
        tvm_exe_path = (k > 0 && k < sizeof(exe)) ? exe : argv[0];
    }
#else
    {
        static char exe[1024];
        ssize_t k = readlink("/proc/self/exe", exe, sizeof(exe) - 1);
        if (k > 0) { exe[k] = 0; tvm_exe_path = exe; } else tvm_exe_path = argv[0];
    }
#endif
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
