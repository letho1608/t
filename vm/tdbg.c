/* tdbg.c — debugger instruction-level cho trun (v1.8).
 * Dùng: trun -d prog.tb [args]  (lệnh: s[n] step, c cont, b pc, p stack,
 * l locals, g globals, d disasm, q quit). Stdin pipe được (cho test).
 */
#include "tvm.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int tvm_dbg_on;      /* đọc trực tiếp trong vòng lặp VM (không gọi hàm) */
#define g_on tvm_dbg_on
static int g_step;
static size_t g_bps[64];
static int g_nbps;

void dbg_enable(void) { g_on = 1; g_step = 0; }
int dbg_enabled(void) { return g_on; }

static const char *opname(uint8_t op) { return tvm_opname(op); }

static size_t oplen(uint8_t *c, size_t n, size_t pc) { return tvm_oplen(c, n, pc); }

void dbg_dis(TProg *p, size_t at, int count) {
    uint8_t *c = p->code;
    size_t n = p->len, pc = at;
    int i;
    for (i = 0; i < count && pc < n; i++) {
        uint8_t op = c[pc];
        size_t L = oplen(c, n, pc);
        printf("  %4zu %-10s", pc, opname(op));
        if (op == OP_PUSH_BOOL || op == OP_PRINT || op == OP_CALL ||
            op == OP_RANGE) {
            if (pc + 1 < n) printf("%u", c[pc + 1]);
        } else if (op == OP_PUSH_INT) {
            int64_t v = 0;
            if (pc + 9 <= n) memcpy(&v, c + pc + 1, 8);
            printf("%lld", (long long)v);
        } else if (op == OP_PUSH_FLOAT) {
            double v = 0;
            if (pc + 9 <= n) memcpy(&v, c + pc + 1, 8);
            printf("%g", v);
        } else if (op == OP_PUSH_STR || op == OP_ATTR) {
            uint32_t len = 0;
            if (pc + 5 <= n) memcpy(&len, c + pc + 1, 4);
            printf("\"%.*s\"", len > 24 ? 24 : (int)len,
                   (len && pc + 5 <= n) ? (char *)(c + pc + 5) : "");
        } else if (op == OP_NEW_LIST || op == OP_NEW_TENSOR || op == OP_NEW_DICT ||
                   op == OP_LOAD || op == OP_STORE || op == OP_DECL ||
                   op == OP_LOADG || op == OP_STOREG || op == OP_DECLG ||
                   op == OP_LOADU || op == OP_STOREU || op == OP_BIND ||
                   op == OP_BINDG || op == OP_CAPLOCAL || op == OP_CAPUP ||
                   op == OP_CAPG || op == OP_INC_L || op == OP_INC_G ||
                   op == OP_DEC_L || op == OP_DEC_G || op == OP_INCB_L ||
                   op == OP_INCB_G) {
            uint16_t v = 0;
            if (pc + 3 <= n) memcpy(&v, c + pc + 1, 2);
            printf("%u", v);
        } else if (op == OP_JMP || op == OP_JMPF || op == OP_JMPT ||
                   op == OP_TRY_PUSH) {
            int32_t r = 0;
            if (pc + 5 <= n) memcpy(&r, c + pc + 1, 4);
            printf("%+d -> %zu", r, pc + 5 + r);
        } else if (op == OP_MKFN) {
            int32_t e = 0;
            uint8_t na = 0;
            uint16_t nl = 0, nu = 0;
            if (pc + 10 <= n) {
                memcpy(&e, c + pc + 1, 4);
                na = c[pc + 5];
                memcpy(&nl, c + pc + 6, 2);
                memcpy(&nu, c + pc + 8, 2);
            }
            printf("entry=%d nargs=%u nlocals=%u nupvals=%u", e, na, nl, nu);
        } else if (op == OP_ADDS_L || op == OP_ADDS_G || op == OP_SUBS_L ||
                   op == OP_SUBS_G) {
            uint16_t d0 = 0, s0 = 0;
            if (pc + 5 <= n) { memcpy(&d0, c + pc + 1, 2); memcpy(&s0, c + pc + 3, 2); }
            printf("%u, %u", d0, s0);
        } else if (op == OP_ADDI_L || op == OP_ADDI_G) {
            uint16_t d0 = 0; int64_t k = 0;
            if (pc + 11 <= n) { memcpy(&d0, c + pc + 1, 2); memcpy(&k, c + pc + 3, 8); }
            printf("%u, %lld", d0, (long long)k);
        }
        putchar('\n');
        pc += L;
    }
    fflush(stdout);
}

static void show_stack(TVM *vm) {
    size_t i;
    printf("stack[%zu]:", vm->sp);
    for (i = 0; i < vm->sp; i++) {
        putchar(' ');
        tval_print(&vm->stack[i]);
    }
    putchar('\n');
}

static void show_frame(TVM *vm) {
    size_t i;
    if (vm->nframes == 0) { puts("(no frame)"); return; }
    {
        TFrame *fr = &vm->frames[vm->nframes - 1];
        printf("frame depth=%zu locals=%u globals=%u\n",
               vm->nframes, fr->nlocals, vm->prog->nglobals);
        for (i = 0; i < fr->nlocals; i++) {
            printf("  l%zu = ", i);
            tval_print(&fr->locals[i]);
            putchar('\n');
        }
    }
    fflush(stdout);
}

static void show_globals(TVM *vm) {
    uint16_t i;
    for (i = 0; i < vm->prog->nglobals; i++) {
        if (vm->prog->globals[i].t == T_NIL) continue;
        printf("  g%u = ", i);
        tval_print(&vm->prog->globals[i]);
        putchar('\n');
    }
    fflush(stdout);
}

static int at_break(size_t pc) {
    int i;
    for (i = 0; i < g_nbps; i++)
        if (g_bps[i] == pc) return 1;
    return 0;
}

/* Gọi ở đầu vòng chạy. Trả 1 để tiếp tục op, 0 để thoát run (quit).
 * g_step: -1 chạy tự do (chỉ breakpoint dừng); >=0: số op BỎ QUA trước
 * khi dừng (vì op tại prompt luôn chạy 1 lần sau mỗi lệnh). */
int dbg_hook(TVM *vm) {
    char line[128];
    if (!at_break(vm->pc)) {
        if (g_step > 0) {
            g_step--;
            return 1;
        }
        if (g_step < 0) return 1;
    }
    g_step = 0;
    printf("[dbg pc=%zu sp=%zu] ", vm->pc, vm->sp);
    fflush(stdout);
    for (;;) {
        printf("(tdbg) ");
        fflush(stdout);
        if (!fgets(line, sizeof(line), stdin)) {
            g_step = -1;
            return 1;  /* EOF: chạy tiếp tới hết */
        }
        if (line[0] == 's') {
            /* prompt xong op hiện tại vẫn chạy 1 lần -> sN đặt N-1 */
            int k = atoi(line + 1);
            g_step = k > 1 ? k - 1 : 0;
            return 1;
        } else if (line[0] == 'c') {
            g_step = -1;
            return 1;
        } else if (line[0] == 'b') {
            if (g_nbps < 64) {
                g_bps[g_nbps++] = (size_t)atoll(line + 1);
                printf("break %zu\n", g_bps[g_nbps - 1]);
            } else puts("het cho breakpoint");
        } else if (line[0] == 'p') {
            show_stack(vm);
        } else if (line[0] == 'l') {
            show_frame(vm);
        } else if (line[0] == 'g') {
            show_globals(vm);
        } else if (line[0] == 'd') {
            int k = atoi(line + 1);
            dbg_dis(vm->prog, vm->pc, k > 0 ? k : 12);
        } else if (line[0] == 'q') {
            exit(0);
        } else if (line[0] == 'h') {
            puts("s[n] step | c cont | b pc | p stack | l locals | g globals | d disasm | q quit");
        } else {
            puts("? (h tro giup)");
        }
        fflush(stdout);
    }
}
