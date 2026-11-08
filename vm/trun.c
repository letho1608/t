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

#define TRUN_VERSION "3.0"

static int is_tb_file(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) return 0;
    unsigned char hdr[4];
    size_t n = fread(hdr, 1, 4, f);
    fclose(f);
    return n == 4 && memcmp(hdr, "TB01", 4) == 0;
}

static void make_tmp_path(char *out, size_t n, const char *ext) {
    static unsigned int seq = 0;
    seq++;
#ifdef _WIN32
    char dir[512];
    DWORD dlen = GetTempPathA((DWORD)sizeof(dir), dir);
    if (dlen == 0 || dlen >= sizeof(dir)) snprintf(dir, sizeof(dir), ".");
    snprintf(out, n, "%strun_%lu_%u%s", dir, (unsigned long)GetCurrentProcessId(), seq, ext);
#else
    const char *dir = getenv("TMPDIR");
    if (!dir || !*dir) dir = "/tmp";
    snprintf(out, n, "%s/trun_%lu_%u%s", dir, (unsigned long)getpid(), seq, ext);
#endif
}

static int run_tb(const char *path, char **xargv, int xargc, int dbg) {
    TProg p; memset(&p, 0, sizeof(p));
    char msg[256];
    char tmp_tb[1024];
    int used_tmp = 0;
    const char *load_path = path;

    if (!is_tb_file(path)) {
        FILE *chk = fopen(path, "rb");
        if (!chk) {
            fprintf(stderr, "không mở được %s\n", path);
            return 1;
        }
        fclose(chk);
        make_tmp_path(tmp_tb, sizeof(tmp_tb), ".tb");
        if (tvm_compile_src(path, tmp_tb, 1, msg, sizeof(msg)) < 0) {
            remove(tmp_tb);
            fprintf(stderr, "%s\n", msg);
            return 1;
        }
        load_path = tmp_tb;
        used_tmp = 1;
    }

    if (tvm_read_tb(load_path, &p, msg, sizeof(msg)) != 0) {
        if (used_tmp) remove(tmp_tb);
        fprintf(stderr, "%s\n", msg);
        return 1;
    }
    if (used_tmp) {
        remove(tmp_tb);
        free(p.path);
        size_t plen = strlen(path);
        p.path = (char*)malloc(plen + 1);
        if (p.path) memcpy(p.path, path, plen + 1);
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

static int run_code(const char *code, char **xargv, int xargc, int dbg) {
    char tmp_t[1024];
    make_tmp_path(tmp_t, sizeof(tmp_t), ".t");
    FILE *f = fopen(tmp_t, "wb");
    if (!f) {
        fprintf(stderr, "không tạo được file tạm\n");
        return 1;
    }
    size_t len = strlen(code);
    fwrite(code, 1, len, f);
    if (len == 0 || code[len - 1] != '\n') fwrite("\n", 1, 1, f);
    fclose(f);
    int rc = run_tb(tmp_t, xargv, xargc, dbg);
    remove(tmp_t);
    return rc;
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
        "  trun prog.t [args...]                biên dịch & chạy trực tiếp file nguồn .t\n"
        "  trun prog.tb [args...]               chạy bytecode .tb\n"
        "  trun -c \"code\" [args...]             chạy đoạn mã T trực tiếp\n"
        "  trun --build-exe <src.t|.tb> <out>   đóng gói thành file thực thi độc lập\n"
        "  trun -d prog.t|.tb                   debugger từng lệnh\n"
        "  trun                                 REPL (cần trepl.tb)\n"
        "  trun --version\n", TRUN_VERSION);
}

/* Kiểm tra xem chính file thực thi hiện tại có nhúng bytecode ở cuối (footer TBEXE001) không.
   Nếu có, trích xuất ra file tạm .tb và chạy với toàn bộ đối số dòng lệnh. */
static int check_embedded_exe(int argc, char **argv, int *out_rc) {
    if (!tvm_exe_path) return 0;
    FILE *f = fopen(tvm_exe_path, "rb");
    if (!f) return 0;
    if (fseek(f, 0, SEEK_END) != 0) { fclose(f); return 0; }
    long fsz = ftell(f);
    if (fsz <= 24) { fclose(f); return 0; }
    if (fseek(f, fsz - 16, SEEK_SET) != 0) { fclose(f); return 0; }
    unsigned char tail[16];
    if (fread(tail, 1, 16, f) != 16 || memcmp(tail + 8, "TBEXE001", 8) != 0) {
        fclose(f);
        return 0;
    }
    uint64_t tb_sz = 0;
    memcpy(&tb_sz, tail, 8);
    if (tb_sz == 0 || (long)(tb_sz + 16) >= fsz) {
        fclose(f);
        return 0;
    }
    if (fseek(f, fsz - 16 - (long)tb_sz, SEEK_SET) != 0) {
        fclose(f);
        return 0;
    }
    unsigned char *buf = (unsigned char*)malloc((size_t)tb_sz);
    if (!buf || fread(buf, 1, (size_t)tb_sz, f) != (size_t)tb_sz) {
        free(buf);
        fclose(f);
        return 0;
    }
    fclose(f);
    if (tb_sz < 4 || memcmp(buf, "TB01", 4) != 0) {
        free(buf);
        return 0;
    }
    char tmp_tb[1024];
    make_tmp_path(tmp_tb, sizeof(tmp_tb), ".tb");
    FILE *fo = fopen(tmp_tb, "wb");
    if (!fo) { free(buf); return 0; }
    fwrite(buf, 1, (size_t)tb_sz, fo);
    fclose(fo);
    free(buf);
    *out_rc = run_tb(tmp_tb, argv + 1, argc - 1, 0);
    remove(tmp_tb);
    return 1;
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
    /* nhị phân: không CRLF — output phải khớp byte chuẩn */
    _setmode(_fileno(stdout), _O_BINARY);
    _setmode(_fileno(stderr), _O_BINARY);
    _setmode(_fileno(stdin), _O_BINARY);
#endif
    {
        int emb_rc = 0;
        if (check_embedded_exe(argc, argv, &emb_rc)) return emb_rc;
    }
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
    if (!strcmp(argv[1], "--build-exe")) {
        char msg[256];
        if (argc < 4) { usage(); return 2; }
        if (tvm_build_exe(argv[2], argv[3], msg, sizeof(msg)) != 0) {
            fprintf(stderr, "T error: %s\n", msg[0] ? msg : "build-exe failed");
            return 1;
        }
        return 0;
    }
    if (!strcmp(argv[1], "-d")) { dbg = 1; ai++; }
    if (ai >= argc) { usage(); return 2; }
    if (!strcmp(argv[ai], "-c")) {
        ai++;
        if (ai >= argc) { usage(); return 2; }
        return run_code(argv[ai], argv + ai + 1, argc - ai - 1, dbg);
    }
    return run_tb(argv[ai], argv + ai + 1, argc - ai - 1, dbg);
}
