/* jit_x64.c — v2.2 method JIT x86-64 cho hàm số học thuần.
 *
 * Phạm vi: hàm chỉ dùng biến cục bộ + hằng + global số (chỉ đọc) với kiểu
 * int / float / bool; if/elif/while/for-range, so sánh, and/or, % kiểu floor,
 * gọi đệ quy chính nó hoặc gọi hàm khác cũng JIT được, int()/float()/abs()/
 * min()/max(). Kiểu suy ra bằng thông dịch trừu tượng trên bytecode, đặc hoá
 * theo kiểu đối số ở lần biên dịch (guard tag mỗi lần gọi).
 *
 * Deopt an toàn: hàm thuần (không ghi global, không cấp phát, không gọi hàm
 * có side effect) nên MỌI guard hỏng (chia 0, global đổi kiểu, hàm bị gán
 * lại, đệ quy quá sâu) chỉ cần huỷ toàn bộ lời gọi native rồi để interpreter
 * chạy lại từ đầu — kết quả, lỗi và thông điệp lỗi y hệt interpreter.
 *
 * OSR: vòng lặp nóng trong một lần gọi dài -> biên dịch vùng từ đầu vòng lặp
 * với kiểu locals hiện tại, nhảy vào native giữa chừng; huỷ thì interpreter
 * chạy tiếp từ đầu vòng lặp với locals của chính nó (native chỉ sửa bản sao).
 *
 * ABI native: int64 f(int64 *in, JitCtx *ctx). Mọi giá trị là 64 bit (int64,
 * double dạng bit, bool 0/1). Win64: rcx/rdx; System V: rdi/rsi.
 * Frame: [rsp+0..31] shadow, [rsp+32+8*i] locals, sau đó là stack ảo.
 */
#if !defined(_WIN32)
/* -std=c99 ẩn MAP_ANONYMOUS/mprotect trên glibc; macOS cần _DARWIN_C_SOURCE */
#ifndef _DEFAULT_SOURCE
#define _DEFAULT_SOURCE
#endif
#ifndef _DARWIN_C_SOURCE
#define _DARWIN_C_SOURCE
#endif
#endif
#include "tvm.h"
#include <string.h>
#include <stdlib.h>
#include <stddef.h>
#include <stdio.h>

#if defined(_WIN32)
#include <windows.h>
#define JIT_SYSV 0
static void *jit_alloc_exec(size_t n) {
    return VirtualAlloc(NULL, n, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
}
static int jit_seal(void *m, size_t n) {
    DWORD old;
    if (!VirtualProtect(m, n, PAGE_EXECUTE_READ, &old)) return 0;
    FlushInstructionCache(GetCurrentProcess(), m, n);
    return 1;
}
static void jit_free_exec(void *m, size_t n) { (void)n; VirtualFree(m, 0, MEM_RELEASE); }
#else
#include <sys/mman.h>
#include <unistd.h>
#if !defined(MAP_ANONYMOUS) && defined(MAP_ANON)
#define MAP_ANONYMOUS MAP_ANON
#endif
#define JIT_SYSV 1   /* POSIX x86-64: System V ABI */
static void *jit_alloc_exec(size_t n) {
    void *m = mmap(NULL, n, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    return (m == MAP_FAILED) ? NULL : m;
}
static int jit_seal(void *m, size_t n) {
    __builtin___clear_cache((char *)m, (char *)m + n);
    return mprotect(m, n, PROT_READ | PROT_EXEC) == 0;
}
static void jit_free_exec(void *m, size_t n) { munmap(m, n); }
#endif

/* code sinh ra đọc thẳng TVal: tag (4 byte) ở offset 0, giá trị ở offset 8 */
typedef char tval_layout_check[(offsetof(TVal, u) == 8 && sizeof(TType) == 4) ? 1 : -1];

#define OP_BAD 0xFF   /* micro-op: lệnh không JIT được */

/* kiểu trừu tượng */
/* JT_LI/LF/LB: list mà phần tử (đoán lúc biên dịch) là int/float/bool;
   JT_LX: list phần tử lẫn lộn (chỉ dùng len) */
enum { JT_UNK = 0, JT_I, JT_F, JT_B, JT_FN, JT_NAT, JT_LI, JT_LF, JT_LB, JT_LX };
static int is_list_t(uint8_t t) { return t >= JT_LI && t <= JT_LX; }
static uint8_t list_elem(uint8_t t) { return t == JT_LI ? JT_I : t == JT_LF ? JT_F : t == JT_LB ? JT_B : JT_UNK; }
/* code sinh ra đọc TList: len/items theo offset thật; TVal 32 byte */
typedef char tval_size_check[(sizeof(TVal) == 32) ? 1 : -1];

#define JIT_STACK_BUDGET (256 * 1024)   /* native dùng quá ngần này stack C -> huỷ */
#define JIT_MAX_ABORTS 64
#define JIT_MAX_OSR 8
#define JIT_MAX_NEST 4          /* biên dịch callee lồng nhau */

/* ctx: [0] cờ (1 huỷ, 2 deopt có trạng thái), [4] mã deopt, [8] đáy stack cho
   phép (rsp < đây -> huỷ: đệ quy quá sâu), [16] buffer chép frame khi deopt */
typedef struct { int32_t abort; int32_t deopt_id; char *stack_limit; int64_t *save; } JitCtx;
typedef int64_t (*JitEntry)(int64_t *in, JitCtx *ctx);

typedef struct { uint8_t *mem; size_t size; } JitUnit;

/* điểm deopt: tiếp tục interpreter tại pc với stack/locals theo kiểu đã biết */
typedef struct {
    size_t pc;
    int depth;
    uint8_t *st;
    int32_t *sv;
    uint8_t *lt;
} JitDeopt;

typedef struct {
    size_t pc;
    int state;                  /* 0 chưa, 1 ok, 2 hỏng */
    void *entry;
    uint8_t *lty;               /* kiểu locals tại đầu vòng lặp */
    uint8_t retty;
    JitUnit unit;
    int misses;
    JitDeopt *deopts;
    int ndeopt;
    int nsave;                  /* số qword frame cần chép khi deopt */
} JitOsr;

typedef struct TJitFn {
    int estate;                 /* entry: 0 chưa, 1 đang biên dịch, 2 ok, 3 hỏng */
    uint8_t argty[256];
    int nargs;
    uint8_t retty;
    void *entry;
    JitUnit unit;
    int aborts;
    JitOsr osr[JIT_MAX_OSR];
    int nosr;
    TFn **callees;              /* giữ ref: code native nhảy thẳng vào chúng */
    int ncallees, capcallees;
} TJitFn;

static TJitFn *jit_of(TFn *fn) {
    if (!fn->jit) {
        fn->jit = calloc(1, sizeof(TJitFn));
    }
    return (TJitFn *)fn->jit;
}

/* ================= phân tích bytecode ================= */

/* micro-op: superop được trải ra (INC_L s -> LOAD s, PUSH_INT 1, ADD, STORE s) */
typedef struct {
    uint8_t op;
    int64_t k;          /* imm (PUSH_INT/PUSH_FLOAT bits/PUSH_BOOL), slot, argc */
    size_t pc;          /* pc bytecode của lệnh gốc */
    int32_t target;     /* chỉ số micro-op đích (nhảy) */
    int head;           /* là đầu lệnh gốc (đích nhảy hợp lệ) */
} MOp;

typedef struct {
    int valid;
    int depth;
    uint8_t *st;        /* kiểu stack ảo [depth] */
    int32_t *sv;        /* giá trị kèm (global idx cho JT_FN / NAT idx) */
    uint8_t *lt;        /* kiểu locals */
} AState;

typedef struct {
    TVM *vm;
    TFn *fn;
    TJitFn *jf;
    MOp *ops; int nops, capops;
    int nl;             /* nlocals */
    int maxd;
    int maxd_seen;      /* độ sâu stack ảo lớn nhất thực dùng (kích thước frame) */
    AState *in;         /* trạng thái vào mỗi micro-op */
    int self_retty_guess;
    int used_guess;
    int nest;
    /* codegen */
    uint8_t *b; size_t n, cap;
    size_t *label;      /* offset native mỗi micro-op */
    struct { size_t at; int32_t to; } *fix; int nfix, capfix;
    struct { size_t at; } *afix; int nafix, capafix;   /* nhảy tới abort */
    struct { size_t at; } *efix; int nefix, capefix;   /* nhảy tới epilogue */
    int32_t frame;
    int fail;
    const char *why;
    int osr;            /* biên dịch vùng OSR: guard hỏng -> deopt có trạng thái */
    int cur_op;         /* micro-op đang sinh mã (gắn vào điểm deopt) */
    int cur_head;       /* micro-op đầu của lệnh gốc đang sinh mã */
    /* điểm deopt: op (đầu lệnh) hoặc lối ra vùng main (pc + độ sâu riêng) */
    struct { size_t at; int op; size_t xpc; int xdepth; } *dfix; int ndfix, capdfix;
    /* chế độ main: vòng lặp top-level; global dùng như "local giả" ghi thẳng bộ nhớ */
    int mainmode;
    int32_t *gpseudo;   /* global -> chỉ số local giả (-1 nếu không) */
    int32_t *pglob;     /* local giả -> global */
} JC;

static void jc_fail(JC *j, const char *why) { if (!j->fail) { j->fail = 1; j->why = why; } }

static void add_op(JC *j, uint8_t op, int64_t k, size_t pc, int head) {
    if (j->nops >= j->capops) {
        j->capops = j->capops ? j->capops * 2 : 64;
        j->ops = (MOp *)realloc(j->ops, (size_t)j->capops * sizeof(MOp));
    }
    MOp *m = &j->ops[j->nops++];
    m->op = op; m->k = k; m->pc = pc; m->target = -1; m->head = head;
}

static uint16_t rd16(const uint8_t *c) { uint16_t v; memcpy(&v, c, 2); return v; }
static int32_t rd32(const uint8_t *c) { int32_t v; memcpy(&v, c, 4); return v; }
static int64_t rd64(const uint8_t *c) { int64_t v; memcpy(&v, c, 8); return v; }

/* thân hàm: [entry, end) với end = đích của JMP ngay trước entry */
static int body_range(TProg *p, TFn *fn, size_t *beg, size_t *end) {
    size_t e = (size_t)fn->entry;
    if (e < 5 || p->code[e - 5] != OP_JMP) return 0;
    int32_t rel = rd32(p->code + e - 4);
    if (rel <= 0) return 0;
    *beg = e; *end = e + (size_t)rel;
    return *end <= p->len;
}

static int decode(JC *j, TProg *p, size_t beg, size_t end) {
    const uint8_t *c = p->code;
    size_t pc = beg;
    while (pc < end) {
        uint8_t op = c[pc];
        size_t L = tvm_oplen(c, p->len, pc);
        switch (op) {
        case OP_PUSH_INT: add_op(j, OP_PUSH_INT, rd64(c + pc + 1), pc, 1); break;
        case OP_PUSH_FLOAT: add_op(j, OP_PUSH_FLOAT, rd64(c + pc + 1), pc, 1); break;
        case OP_PUSH_BOOL: add_op(j, OP_PUSH_BOOL, c[pc + 1] ? 1 : 0, pc, 1); break;
        case OP_LOAD: case OP_STORE: case OP_DECL: case OP_BIND: case OP_LOADG:
        case OP_STOREG: case OP_BINDG: case OP_DECLG:
            add_op(j, op == OP_DECLG ? OP_STOREG : op, rd16(c + pc + 1), pc, 1); break;
        case OP_INC_G: case OP_DEC_G: case OP_INCB_G: {
            uint16_t s = rd16(c + pc + 1);
            add_op(j, OP_LOADG, s, pc, 1);
            add_op(j, OP_PUSH_INT, 1, pc, 0);
            add_op(j, op == OP_DEC_G ? OP_SUB : OP_ADD, 0, pc, 0);
            add_op(j, op == OP_INCB_G ? OP_BINDG : OP_STOREG, s, pc, 0);
            break;
        }
        case OP_ADDI_G: {
            uint16_t s = rd16(c + pc + 1);
            add_op(j, OP_LOADG, s, pc, 1);
            add_op(j, OP_PUSH_INT, rd64(c + pc + 3), pc, 0);
            add_op(j, OP_ADD, 0, pc, 0);
            add_op(j, OP_STOREG, s, pc, 0);
            break;
        }
        case OP_ADDS_G: case OP_SUBS_G: {
            uint16_t d = rd16(c + pc + 1), s = rd16(c + pc + 3);
            add_op(j, OP_LOADG, d, pc, 1);
            add_op(j, OP_LOADG, s, pc, 0);
            add_op(j, op == OP_ADDS_G ? OP_ADD : OP_SUB, 0, pc, 0);
            add_op(j, OP_STOREG, d, pc, 0);
            break;
        }
        case OP_ADD: case OP_SUB: case OP_MUL: case OP_DIV: case OP_MOD:
        case OP_NEG: case OP_NOT:
        case OP_EQ: case OP_NE: case OP_LT: case OP_GT: case OP_LE: case OP_GE:
        case OP_RET: case OP_POP: case OP_DUP: case OP_NOP:
        case OP_IDX: case OP_SETIDX: case OP_LEN:
            add_op(j, op, 0, pc, 1); break;
        case OP_CALL: add_op(j, op, c[pc + 1], pc, 1); break;
        case OP_ATTR:
            /* xs.len trên list == LEN; thuộc tính khác: ngoài phạm vi */
            if (rd32(c + pc + 1) == 3 && memcmp(c + pc + 5, "len", 3) == 0) add_op(j, OP_LEN, 0, pc, 1);
            else add_op(j, OP_BAD, op, pc, 1);
            break;
        case OP_JMP: case OP_JMPF: case OP_JMPT:
            add_op(j, op, (int64_t)(pc + 5 + rd32(c + pc + 1)), pc, 1); break;
        case OP_INC_L: case OP_DEC_L: case OP_INCB_L: {
            uint16_t s = rd16(c + pc + 1);
            add_op(j, OP_LOAD, s, pc, 1);
            add_op(j, OP_PUSH_INT, 1, pc, 0);
            add_op(j, op == OP_DEC_L ? OP_SUB : OP_ADD, 0, pc, 0);
            add_op(j, op == OP_INCB_L ? OP_BIND : OP_STORE, s, pc, 0);
            break;
        }
        case OP_ADDI_L: {
            uint16_t s = rd16(c + pc + 1);
            add_op(j, OP_LOAD, s, pc, 1);
            add_op(j, OP_PUSH_INT, rd64(c + pc + 3), pc, 0);
            add_op(j, OP_ADD, 0, pc, 0);
            add_op(j, OP_STORE, s, pc, 0);
            break;
        }
        case OP_ADDS_L: case OP_SUBS_L: {
            uint16_t d = rd16(c + pc + 1), s = rd16(c + pc + 3);
            add_op(j, OP_LOAD, d, pc, 1);
            add_op(j, OP_LOAD, s, pc, 0);
            add_op(j, op == OP_ADDS_L ? OP_ADD : OP_SUB, 0, pc, 0);
            add_op(j, OP_STORE, d, pc, 0);
            break;
        }
        default:
            /* lệnh ngoài phạm vi: chỉ hỏng nếu phân tích đi tới (vd PUSH_NIL; RET
               cuối thân hàm thường không tới được) */
            add_op(j, OP_BAD, op, pc, 1);
            break;
        }
        pc += L;
    }
    /* đích nhảy -> chỉ số micro-op */
    for (int i = 0; i < j->nops; i++) {
        MOp *m = &j->ops[i];
        if (m->op == OP_JMP || m->op == OP_JMPF || m->op == OP_JMPT) {
            size_t t = (size_t)m->k;
            int found = -1;
            for (int q = 0; q < j->nops; q++)
                if (j->ops[q].head && j->ops[q].pc == t) { found = q; break; }
            if (found < 0) {
                if (j->mainmode) { m->target = -1; continue; }   /* lối ra vòng lặp */
                jc_fail(j, "nhảy ra ngoài thân hàm"); return 0;
            }
            m->target = found;
        }
    }
    return 1;
}

static void st_alloc(JC *j, AState *s) {
    s->st = (uint8_t *)calloc((size_t)j->maxd + 1, 1);
    s->sv = (int32_t *)calloc((size_t)j->maxd + 1, sizeof(int32_t));
    s->lt = (uint8_t *)calloc((size_t)j->nl + 1, 1);
}

/* gộp trạng thái vào micro-op i; trả 1 nếu đổi */
static int merge_in(JC *j, int i, int depth, const uint8_t *st, const int32_t *sv, const uint8_t *lt) {
    AState *s = &j->in[i];
    if (depth > j->maxd) { jc_fail(j, "stack quá sâu"); return 0; }
    if (!s->valid) {
        s->valid = 1; s->depth = depth;
        memcpy(s->st, st, (size_t)depth);
        memcpy(s->sv, sv, (size_t)depth * sizeof(int32_t));
        memcpy(s->lt, lt, (size_t)j->nl);
        return 1;
    }
    if (s->depth != depth) { jc_fail(j, "độ sâu stack lệch ở điểm gộp"); return 0; }
    int ch = 0;
    for (int d = 0; d < depth; d++) {
        if (s->st[d] != st[d] || s->sv[d] != sv[d]) { jc_fail(j, "kiểu stack lệch ở điểm gộp"); return 0; }
    }
    for (int l = 0; l < j->nl; l++) {
        if (s->lt[l] == lt[l]) continue;
        if (s->lt[l] == JT_UNK) continue;               /* đã UNK: giữ */
        if (lt[l] == JT_UNK) { s->lt[l] = JT_UNK; ch = 1; continue; }
        jc_fail(j, "local đổi kiểu qua vòng lặp/nhánh");
        return 0;
    }
    return ch;
}

static int compile_entry(TVM *vm, TFn *fn, const uint8_t *argty, int nest);

/* kiểu trả về + kiểu đối số của callee F (biên dịch nếu cần) */
static int callee_info(JC *j, TFn *F, const uint8_t *at, int k, uint8_t *retty) {
    if (j->fn && F == j->fn && j->jf->estate == 1) {
        /* đệ quy chính nó trong lúc đang biên dịch entry */
        for (int i = 0; i < k; i++) if (at[i] != j->jf->argty[i]) return 0;
        *retty = (uint8_t)j->self_retty_guess;
        j->used_guess = 1;
        return 1;
    }
    if (F->nargs != k) return 0;
    TJitFn *fj = jit_of(F);
    if (fj->estate == 0) {
        if (j->nest >= JIT_MAX_NEST) return 0;
        if (!compile_entry(j->vm, F, at, j->nest + 1)) return 0;
    }
    if (fj->estate != 2) return 0;
    for (int i = 0; i < k; i++) if (at[i] != fj->argty[i]) return 0;
    *retty = fj->retty;
    return 1;
}

static int is_num(uint8_t t) { return t == JT_I || t == JT_F; }
/* kiểu trừu tượng của list theo phần tử hiện có (đoán; mỗi lần đọc đều có guard) */
static uint8_t list_ty(const TList *l) {
    TType t0;
    if (l->len == 0) return JT_LI;
    t0 = l->items[0].t;
    if (t0 != T_INT && t0 != T_FLOAT && t0 != T_BOOL) return JT_LX;
    for (size_t i = 1; i < l->len; i++) if (l->items[i].t != t0) return JT_LX;
    return t0 == T_INT ? JT_LI : t0 == T_FLOAT ? JT_LF : JT_LB;
}
static int is_nat_ok(int64_t k);

/* thông dịch trừu tượng từ micro-op start với locals lt0 */
static int analyze(JC *j, int start, const uint8_t *lt0, uint8_t *retty_out) {
    int *work = (int *)malloc((size_t)(j->nops + 1) * sizeof(int));
    char *inq = (char *)calloc((size_t)j->nops + 1, 1);
    int nw = 0;
    uint8_t *st = (uint8_t *)calloc((size_t)j->maxd + 2, 1);
    int32_t *sv = (int32_t *)calloc((size_t)j->maxd + 2, sizeof(int32_t));
    uint8_t *lt = (uint8_t *)calloc((size_t)j->nl + 1, 1);
    int retty = JT_UNK;
    TVal *G = j->vm->prog->globals;
    size_t NG = j->vm->prog->nglobals;
    merge_in(j, start, 0, st, sv, lt0);
    work[nw++] = start; inq[start] = 1;
    while (nw > 0 && !j->fail) {
        int i = work[--nw];
        inq[i] = 0;
        AState *s = &j->in[i];
        int d = s->depth;
        memcpy(st, s->st, (size_t)d);
        memcpy(sv, s->sv, (size_t)d * sizeof(int32_t));
        memcpy(lt, s->lt, (size_t)j->nl);
        /* chạy tuyến tính tới lệnh rẽ nhánh */
        for (;;) {
            MOp *m = &j->ops[i];
            int next = i + 1, br = -1, fall = 1;
            if (j->fail) break;
            switch (m->op) {
            case OP_PUSH_INT: st[d] = JT_I; sv[d] = 0; d++; break;
            case OP_PUSH_FLOAT: st[d] = JT_F; sv[d] = 0; d++; break;
            case OP_PUSH_BOOL: st[d] = JT_B; sv[d] = 0; d++; break;
            case OP_LOAD:
                if (m->k >= j->nl || lt[m->k] == JT_UNK) { jc_fail(j, "đọc local chưa gán / kiểu lạ"); break; }
                st[d] = lt[m->k]; sv[d] = 0; d++; break;
            case OP_STORE: case OP_DECL: case OP_BIND:
                if (d < 1) { jc_fail(j, "stack rỗng"); break; }
                d--;
                if (m->k >= j->nl || !(st[d] == JT_I || st[d] == JT_F || st[d] == JT_B)) {
                    jc_fail(j, "gán local kiểu ngoài phạm vi"); break;
                }
                lt[m->k] = st[d]; break;
            case OP_STOREG: case OP_BINDG: {
                int pi = (j->mainmode && (size_t)m->k < NG) ? j->gpseudo[m->k] : -1;
                if (pi < 0) { jc_fail(j, "ghi global ngoài vòng lặp top-level"); break; }
                if (d < 1) { jc_fail(j, "stack rỗng"); break; }
                d--;
                if (!(st[d] == JT_I || st[d] == JT_F || st[d] == JT_B)) { jc_fail(j, "gán global kiểu ngoài phạm vi"); break; }
                lt[pi] = st[d];
                break;
            }
            case OP_LOADG: {
                if ((size_t)m->k >= NG) { jc_fail(j, "global lạ"); break; }
                TVal *g = &G[m->k];
                if (j->mainmode && j->gpseudo[m->k] >= 0) {
                    uint8_t t = lt[j->gpseudo[m->k]];
                    if (t == JT_UNK) { jc_fail(j, "đọc global chưa gán / kiểu lạ"); break; }
                    st[d] = t; sv[d] = 0; d++; break;
                }
                if (is_nat_ok(m->k)) {
                    st[d] = JT_NAT; sv[d] = (int32_t)m->k; d++; break;
                }
                if (g->t == T_INT) { st[d] = JT_I; sv[d] = 0; d++; break; }
                if (g->t == T_FLOAT) { st[d] = JT_F; sv[d] = 0; d++; break; }
                if (g->t == T_BOOL) { st[d] = JT_B; sv[d] = 0; d++; break; }
                if (g->t == T_FN) { st[d] = JT_FN; sv[d] = (int32_t)m->k; d++; break; }
                if (g->t == T_LIST) { st[d] = list_ty(g->u.list); sv[d] = 0; d++; break; }
                jc_fail(j, "global không phải số/hàm/list");
                break;
            }
            case OP_IDX: {
                if (d < 2 || list_elem(st[d - 2]) == JT_UNK || st[d - 1] != JT_I) {
                    jc_fail(j, "index ngoài phạm vi JIT"); break;
                }
                uint8_t e = list_elem(st[d - 2]);
                d--; st[d - 1] = e; sv[d - 1] = 0;
                break;
            }
            case OP_SETIDX: {
                if (d < 3 || list_elem(st[d - 3]) == JT_UNK || st[d - 2] != JT_I ||
                    st[d - 1] != list_elem(st[d - 3])) {
                    jc_fail(j, "gán phần tử ngoài phạm vi JIT"); break;
                }
                if (!j->osr) { jc_fail(j, "ghi list (side effect) chỉ JIT trong vùng OSR"); break; }
                d -= 3;
                break;
            }
            case OP_LEN:
                if (d < 1 || !is_list_t(st[d - 1])) { jc_fail(j, "len ngoài phạm vi JIT"); break; }
                st[d - 1] = JT_I; sv[d - 1] = 0;
                break;
            case OP_ADD: case OP_SUB: case OP_MUL: case OP_DIV: case OP_MOD: {
                if (d < 2 || !is_num(st[d - 2]) || !is_num(st[d - 1])) { jc_fail(j, "số học trên kiểu lạ"); break; }
                uint8_t a = st[d - 2], b = st[d - 1];
                d--;
                st[d - 1] = (m->op == OP_DIV || a == JT_F || b == JT_F) ? JT_F : JT_I;
                sv[d - 1] = 0;
                break;
            }
            case OP_NEG:
                if (d < 1 || !is_num(st[d - 1])) { jc_fail(j, "neg kiểu lạ"); break; }
                break;
            case OP_NOT:
                if (d < 1 || !(is_num(st[d - 1]) || st[d - 1] == JT_B)) { jc_fail(j, "not kiểu lạ"); break; }
                st[d - 1] = JT_B; break;
            case OP_EQ: case OP_NE: case OP_LT: case OP_GT: case OP_LE: case OP_GE: {
                if (d < 2) { jc_fail(j, "stack rỗng"); break; }
                uint8_t a = st[d - 2], b = st[d - 1];
                int ok = (is_num(a) && is_num(b)) ||
                         ((m->op == OP_EQ || m->op == OP_NE) &&
                          (a == JT_B || b == JT_B) && (is_num(a) || a == JT_B) && (is_num(b) || b == JT_B));
                if (!ok) { jc_fail(j, "so sánh kiểu lạ"); break; }
                d--; st[d - 1] = JT_B; sv[d - 1] = 0;
                break;
            }
            case OP_POP:
                if (d < 1) { jc_fail(j, "stack rỗng"); break; }
                d--; break;
            case OP_DUP:
                if (d < 1) { jc_fail(j, "stack rỗng"); break; }
                st[d] = st[d - 1]; sv[d] = sv[d - 1]; d++; break;
            case OP_NOP: break;
            case OP_JMP: br = m->target; fall = 0; break;   /* target < 0: lối ra (main) */
            case OP_JMPF: case OP_JMPT:
                if (d < 1 || !(is_num(st[d - 1]) || st[d - 1] == JT_B)) { jc_fail(j, "điều kiện kiểu lạ"); break; }
                d--; br = m->target; break;
            case OP_RET:
                if (d < 1 || !(st[d - 1] == JT_I || st[d - 1] == JT_F || st[d - 1] == JT_B)) {
                    jc_fail(j, "trả kiểu ngoài phạm vi"); break;
                }
                if (retty == JT_UNK) retty = st[d - 1];
                else if (retty != st[d - 1]) { jc_fail(j, "return nhiều kiểu"); break; }
                fall = 0;
                break;
            case OP_CALL: {
                int k = (int)m->k;
                if (d < k + 1) { jc_fail(j, "stack rỗng"); break; }
                uint8_t ft = st[d - k - 1];
                if (ft == JT_NAT) {
                    int nat = sv[d - k - 1];
                    uint8_t r;
                    if (nat == NAT_LEN) {
                        if (k != 1 || !is_list_t(st[d - 1])) { jc_fail(j, "len() ngoài phạm vi JIT"); break; }
                        r = JT_I;
                    } else if (nat == NAT_INT || nat == NAT_FLOAT || nat == NAT_ABS) {
                        if (k != 1 || !(is_num(st[d - 1]) || (nat == NAT_INT && st[d - 1] == JT_B) ||
                                        (nat == NAT_FLOAT && st[d - 1] == JT_B))) {
                            jc_fail(j, "builtin kiểu lạ"); break;
                        }
                        r = nat == NAT_INT ? JT_I : nat == NAT_FLOAT ? JT_F : st[d - 1];
                        if (nat == NAT_ABS && st[d - 1] == JT_B) { jc_fail(j, "abs(bool)"); break; }
                    } else {
                        if (k != 2 || !is_num(st[d - 1]) || st[d - 1] != st[d - 2]) {
                            jc_fail(j, "min/max cần 2 số cùng kiểu"); break;
                        }
                        r = st[d - 1];
                    }
                    d -= k;
                    st[d - 1] = r; sv[d - 1] = 0;
                    break;
                }
                if (ft != JT_FN) { jc_fail(j, "gọi giá trị không phải hàm"); break; }
                {
                    TVal *g = &G[sv[d - k - 1]];
                    uint8_t at[256], r = JT_UNK;
                    for (int q = 0; q < k; q++) {
                        at[q] = st[d - k + q];
                        if (!(at[q] == JT_I || at[q] == JT_F || at[q] == JT_B || is_list_t(at[q]))) {
                            jc_fail(j, "đối số kiểu lạ"); break;
                        }
                    }
                    if (j->fail) break;
                    if (g->t != T_FN || !callee_info(j, g->u.fn, at, k, &r) || r == JT_UNK) {
                        jc_fail(j, "callee không JIT được"); break;
                    }
                    d -= k;
                    st[d - 1] = r; sv[d - 1] = 0;
                }
                break;
            }
            case OP_BAD:
                jc_fail(j, tvm_opname((uint8_t)m->k));
                break;
            default:
                jc_fail(j, "op lạ");
            }
            if (j->fail) break;
            if (d > j->maxd) { jc_fail(j, "stack quá sâu"); break; }
            if (d > j->maxd_seen) j->maxd_seen = d;
            if (br >= 0) {
                if (merge_in(j, br, d, st, sv, lt) && !inq[br]) { work[nw++] = br; inq[br] = 1; }
                if (j->fail) break;
            }
            if (!fall) break;
            if (next >= j->nops) { jc_fail(j, "chạy quá cuối thân hàm"); break; }
            if (j->ops[next].head) {
                /* lệnh kế là đích nhảy tiềm năng: gộp và xử lý qua worklist */
                int changed = merge_in(j, next, d, st, sv, lt);
                if (j->fail) break;
                if (changed && !inq[next]) { work[nw++] = next; inq[next] = 1; }
                break;
            }
            /* micro-op giữa phần trải superop: chỉ tới từ lệnh trước -> ghi đè trạng thái */
            {
                AState *ns = &j->in[next];
                ns->valid = 1; ns->depth = d;
                memcpy(ns->st, st, (size_t)d);
                memcpy(ns->sv, sv, (size_t)d * sizeof(int32_t));
                memcpy(ns->lt, lt, (size_t)j->nl);
            }
            i = next;
        }
    }
    free(work); free(inq); free(st); free(sv); free(lt);
    if (j->fail) return 0;
    if (retty == JT_UNK) { jc_fail(j, "không có return"); return 0; }
    *retty_out = (uint8_t)retty;
    return 1;
}

/* ================= x86-64 emitter ================= */
static void eb(JC *j, uint8_t x) {
    if (j->n >= j->cap) {
        j->cap = j->cap ? j->cap * 2 : 4096;
        j->b = (uint8_t *)realloc(j->b, j->cap);
    }
    j->b[j->n++] = x;
}
static void eb2(JC *j, uint8_t a, uint8_t b) { eb(j, a); eb(j, b); }
static void eb3(JC *j, uint8_t a, uint8_t b, uint8_t c) { eb(j, a); eb(j, b); eb(j, c); }
static void e32(JC *j, int32_t v) { for (int i = 0; i < 4; i++) eb(j, (uint8_t)((uint32_t)v >> (8 * i))); }
static void e64(JC *j, int64_t v) { for (int i = 0; i < 8; i++) eb(j, (uint8_t)((uint64_t)v >> (8 * i))); }

#define LOC(i) (32 + 8 * (int32_t)(i))
#define SLOT(jc, d) (32 + 8 * ((int32_t)(jc)->nl + (int32_t)(d)))

/* [rsp + disp32] với reg (0..7) */
static void m_rsp(JC *j, int reg, int32_t disp) { eb(j, (uint8_t)(0x84 | (reg << 3))); eb(j, 0x24); e32(j, disp); }
static void ld_rax(JC *j, int32_t disp) { eb2(j, 0x48, 0x8B); m_rsp(j, 0, disp); }
static void ld_rcx(JC *j, int32_t disp) { eb2(j, 0x48, 0x8B); m_rsp(j, 1, disp); }
static void st_rax(JC *j, int32_t disp) { eb2(j, 0x48, 0x89); m_rsp(j, 0, disp); }
static void ld_xmm(JC *j, int x, int32_t disp) { eb3(j, 0xF2, 0x0F, 0x10); m_rsp(j, x, disp); }
static void st_xmm0(JC *j, int32_t disp) { eb3(j, 0xF2, 0x0F, 0x11); m_rsp(j, 0, disp); }
static void cvt_xmm(JC *j, int x, int32_t disp) { eb(j, 0xF2); eb(j, 0x48); eb2(j, 0x0F, 0x2A); m_rsp(j, x, disp); }
/* nạp số ở slot thành double vào xmm x */
static void ld_f(JC *j, int x, int32_t disp, uint8_t t) { if (t == JT_F) ld_xmm(j, x, disp); else cvt_xmm(j, x, disp); }
static void mov_rax_imm(JC *j, int64_t v) { eb2(j, 0x48, 0xB8); e64(j, v); }
static void mov_rcx_imm(JC *j, int64_t v) { eb2(j, 0x48, 0xB9); e64(j, v); }

static void jmp_fix(JC *j, uint8_t *pre, int npre, int32_t to) {
    for (int i = 0; i < npre; i++) eb(j, pre[i]);
    if (j->nfix >= j->capfix) {
        j->capfix = j->capfix ? j->capfix * 2 : 64;
        j->fix = realloc(j->fix, (size_t)j->capfix * sizeof(*j->fix));
    }
    j->fix[j->nfix].at = j->n; j->fix[j->nfix].to = to; j->nfix++;
    e32(j, 0);
}
static void jabort(JC *j, uint8_t cc) {   /* cc=0: jmp, khác: 0F cc */
    if (cc) eb2(j, 0x0F, cc); else eb(j, 0xE9);
    if (j->nafix >= j->capafix) {
        j->capafix = j->capafix ? j->capafix * 2 : 64;
        j->afix = realloc(j->afix, (size_t)j->capafix * sizeof(*j->afix));
    }
    j->afix[j->nafix++].at = j->n;
    e32(j, 0);
}
/* guard hỏng: entry -> huỷ cả lời gọi (hàm thuần, chạy lại); OSR -> deopt tại
   micro-op hiện tại (interpreter tiếp tục đúng lệnh này với trạng thái native) */
static void jdeopt_rec(JC *j, uint8_t cc, int op, size_t xpc, int xdepth) {
    if (cc) eb2(j, 0x0F, cc); else eb(j, 0xE9);
    if (j->ndfix >= j->capdfix) {
        j->capdfix = j->capdfix ? j->capdfix * 2 : 64;
        j->dfix = realloc(j->dfix, (size_t)j->capdfix * sizeof(*j->dfix));
    }
    j->dfix[j->ndfix].at = j->n;
    j->dfix[j->ndfix].op = op;
    j->dfix[j->ndfix].xpc = xpc;
    j->dfix[j->ndfix].xdepth = xdepth;
    j->ndfix++;
    e32(j, 0);
}
static void jguard(JC *j, uint8_t cc) {
    if (!j->osr) { jabort(j, cc); return; }
    /* deopt về ĐẦU lệnh gốc: các micro-op trước đó trong cùng lệnh chỉ đẩy giá
       trị (không side effect) nên interpreter chạy lại cả lệnh là đúng */
    jdeopt_rec(j, cc, j->cur_head, 0, -1);
}
/* nhảy ra khỏi vùng main: interpreter tiếp tục tại xpc với stack sâu xdepth */
static void jexit(JC *j, uint8_t cc, size_t xpc, int xdepth) {
    jdeopt_rec(j, cc, j->cur_head, xpc, xdepth);
}
/* nhảy có điều kiện / không điều kiện tới micro-op đích hoặc lối ra */
static void jto(JC *j, uint8_t *pre, int npre, MOp *m, int depth_after) {
    if (m->target >= 0) { jmp_fix(j, pre, npre, m->target); return; }
    jexit(j, npre == 2 ? pre[1] : 0, (size_t)m->k, depth_after);
}
static void jepi(JC *j, uint8_t cc) {
    if (cc) eb2(j, 0x0F, cc); else eb(j, 0xE9);
    if (j->nefix >= j->capefix) {
        j->capefix = j->capefix ? j->capefix * 2 : 64;
        j->efix = realloc(j->efix, (size_t)j->capefix * sizeof(*j->efix));
    }
    j->efix[j->nefix++].at = j->n;
    e32(j, 0);
}
static void patch32(JC *j, size_t at, size_t target) {
    int32_t rel = (int32_t)((int64_t)target - (int64_t)(at + 4));
    memcpy(j->b + at, &rel, 4);
}

/* prologue chung; copy n giá trị từ in[] vào locals theo mask kiểu */
static void prologue(JC *j, const uint8_t *copy_ty, int ncopy) {
    eb(j, 0x53);                         /* push rbx */
    eb2(j, 0x41, 0x54);                  /* push r12 */
    eb3(j, 0x48, 0x81, 0xEC); e32(j, j->frame);   /* sub rsp, frame */
    if (JIT_SYSV) { eb3(j, 0x48, 0x89, 0xF3); eb3(j, 0x49, 0x89, 0xFC); }  /* rbx=rsi r12=rdi */
    else { eb3(j, 0x48, 0x89, 0xD3); eb3(j, 0x49, 0x89, 0xCC); }           /* rbx=rdx r12=rcx */
    /* hết ngân sách stack C (đệ quy quá sâu) -> abort */
    eb(j, 0x48); eb3(j, 0x3B, 0x63, 0x08);   /* cmp rsp,[rbx+8] */
    jabort(j, 0x82);                         /* jb abort */
    for (int i = 0; i < ncopy; i++) {
        if (copy_ty[i] == JT_UNK) continue;
        eb3(j, 0x49, 0x8B, 0x84); eb(j, 0x24); e32(j, 8 * i);   /* mov rax,[r12+8i] */
        st_rax(j, LOC(i));
    }
}

static int jcc_of(uint8_t op, int fl) {
    /* cc cho setcc/jcc: int (fl=0) hoặc sau ucomisd đã xếp toán hạng (fl=1) */
    if (!fl) {
        switch (op) {
        case OP_EQ: return 0x94; case OP_NE: return 0x95;
        case OP_LT: return 0x9C; case OP_GT: return 0x9F;
        case OP_LE: return 0x9E; default: return 0x9D;
        }
    }
    switch (op) {
    case OP_LT: case OP_GT: return 0x97;       /* seta */
    default: return 0x93;                      /* setae (LE, GE) */
    }
}

static int is_nat_ok(int64_t k) {
    return k == NAT_INT || k == NAT_FLOAT || k == NAT_ABS || k == NAT_MIN || k == NAT_MAX ||
           k == NAT_LEN;
}

static int gen_body(JC *j) {
    TVal *G = j->vm->prog->globals;
    for (int i = 0; i < j->nops; i++) {
        MOp *m = &j->ops[i];
        AState *s = &j->in[i];
        j->label[i] = j->n;
        j->cur_op = i;
        if (m->head) j->cur_head = i;
        if (!s->valid) continue;               /* không tới được */
        int d = s->depth;
        uint8_t *st = s->st;
        switch (m->op) {
        case OP_PUSH_INT: case OP_PUSH_FLOAT:
            mov_rax_imm(j, m->k); st_rax(j, SLOT(j, d)); break;
        case OP_PUSH_BOOL:
            eb(j, 0xB8); e32(j, (int32_t)m->k); st_rax(j, SLOT(j, d)); break;
        case OP_LOAD:
            ld_rax(j, LOC(m->k)); st_rax(j, SLOT(j, d)); break;
        case OP_STORE: case OP_DECL: case OP_BIND:
            ld_rax(j, SLOT(j, d - 1)); st_rax(j, LOC(m->k)); break;
        case OP_STOREG: case OP_BINDG: {
            /* global giả (main): ghi thẳng TVal (tag + giá trị); ô cũ phải vô hướng */
            TVal *g = &G[m->k];
            uint8_t t = st[d - 1];
            int tag = t == JT_I ? T_INT : t == JT_F ? T_FLOAT : T_BOOL;
            mov_rax_imm(j, (int64_t)(intptr_t)g);
            if (m->head) { eb3(j, 0x83, 0x38, (uint8_t)T_FLOAT); jguard(j, 0x87); }  /* trong superop: LOADG đầu lệnh đã guard */
            eb2(j, 0xC7, 0x00); e32(j, tag);                       /* mov dword [rax],tag */
            eb2(j, 0x48, 0x8B); m_rsp(j, 1, SLOT(j, d - 1));       /* mov rcx,[slot] */
            eb(j, 0x48); eb3(j, 0x89, 0x48, 0x08);                 /* mov [rax+8],rcx */
            break;
        }
        case OP_LOADG: {
            /* builtin / hàm: xử lý ở CALL; số: guard tag rồi đọc giá trị */
            TVal *g = &G[m->k];
            if (j->mainmode && j->gpseudo[m->k] >= 0) {
                uint8_t t = s->lt[j->gpseudo[m->k]];
                int tag = t == JT_I ? T_INT : t == JT_F ? T_FLOAT : t == JT_B ? T_BOOL : T_LIST;
                mov_rax_imm(j, (int64_t)(intptr_t)g);
                eb3(j, 0x83, 0x38, (uint8_t)tag); jguard(j, 0x85);
                eb(j, 0x48); eb3(j, 0x8B, 0x40, 0x08);
                if (t == JT_B) eb3(j, 0x83, 0xE0, 0x01);
                st_rax(j, SLOT(j, d));
                break;
            }
            if (is_nat_ok(m->k) || g->t == T_FN) break;
            mov_rax_imm(j, (int64_t)(intptr_t)g);
            eb3(j, 0x83, 0x38, (uint8_t)g->t);  /* cmp dword [rax], tag */
            jguard(j, 0x85);                    /* jne -> guard */
            eb(j, 0x48); eb3(j, 0x8B, 0x40, 0x08);  /* mov rax,[rax+8] */
            if (g->t == T_BOOL) { eb3(j, 0x83, 0xE0, 0x01); }  /* and eax,1 (bool là int) */
            st_rax(j, SLOT(j, d));
            break;
        }
        case OP_ADD: case OP_SUB: case OP_MUL: case OP_DIV: case OP_MOD: {
            uint8_t a = st[d - 2], b = st[d - 1];
            int32_t A = SLOT(j, d - 2), B = SLOT(j, d - 1);
            if (m->op != OP_DIV && a == JT_I && b == JT_I) {
                if (m->op == OP_MOD) {
                    /* % kiểu floor: r = a % b; nếu r != 0 và dấu r khác dấu b: r += b */
                    size_t j1, j2, j3, j4;
                    ld_rcx(j, B);
                    eb3(j, 0x48, 0x85, 0xC9); jguard(j, 0x84);           /* test rcx; jz guard */
                    eb(j, 0x48); eb3(j, 0x83, 0xF9, 0xFF);                /* cmp rcx,-1 */
                    eb2(j, 0x75, 0x00); j1 = j->n - 1;                    /* jne Ldiv */
                    eb2(j, 0x31, 0xD2);                                   /* xor edx,edx */
                    eb2(j, 0xEB, 0x00); j2 = j->n - 1;                    /* jmp Ldone */
                    j->b[j1] = (uint8_t)(j->n - (j1 + 1));                /* Ldiv: */
                    ld_rax(j, A);
                    eb2(j, 0x48, 0x99);                                   /* cqo */
                    eb3(j, 0x48, 0xF7, 0xF9);                             /* idiv rcx */
                    eb3(j, 0x48, 0x85, 0xD2);                             /* test rdx,rdx */
                    eb2(j, 0x74, 0x00); j3 = j->n - 1;                    /* jz Ldone */
                    eb3(j, 0x48, 0x89, 0xD0);                             /* mov rax,rdx */
                    eb3(j, 0x48, 0x31, 0xC8);                             /* xor rax,rcx */
                    eb2(j, 0x79, 0x00); j4 = j->n - 1;                    /* jns Ldone */
                    eb3(j, 0x48, 0x01, 0xCA);                             /* add rdx,rcx */
                    j->b[j2] = (uint8_t)(j->n - (j2 + 1));                /* Ldone: */
                    j->b[j3] = (uint8_t)(j->n - (j3 + 1));
                    j->b[j4] = (uint8_t)(j->n - (j4 + 1));
                    eb3(j, 0x48, 0x89, 0xD0);                             /* mov rax,rdx */
                    st_rax(j, A);
                    break;
                }
                ld_rax(j, A);
                if (m->op == OP_ADD) { eb2(j, 0x48, 0x03); m_rsp(j, 0, B); }
                else if (m->op == OP_SUB) { eb2(j, 0x48, 0x2B); m_rsp(j, 0, B); }
                else { eb3(j, 0x48, 0x0F, 0xAF); m_rsp(j, 0, B); }
                st_rax(j, A);
                break;
            }
            ld_f(j, 0, A, a);
            ld_f(j, 1, B, b);
            if (m->op == OP_DIV || m->op == OP_MOD) {
                eb3(j, 0x66, 0x0F, 0x57); eb(j, 0xD2);      /* xorpd xmm2,xmm2 */
                eb3(j, 0x66, 0x0F, 0x2E); eb(j, 0xCA);      /* ucomisd xmm1,xmm2 */
                jguard(j, 0x84);                            /* je guard (0 hoặc NaN) */
            }
            if (m->op == OP_MOD) {
                mov_rax_imm(j, (int64_t)(intptr_t)&fmod_floor);
                eb2(j, 0xFF, 0xD0);                         /* call rax */
            } else {
                uint8_t o = m->op == OP_ADD ? 0x58 : m->op == OP_SUB ? 0x5C : m->op == OP_MUL ? 0x59 : 0x5E;
                eb3(j, 0xF2, 0x0F, o); eb(j, 0xC1);         /* opsd xmm0,xmm1 */
            }
            st_xmm0(j, A);
            break;
        }
        case OP_NEG: {
            int32_t A = SLOT(j, d - 1);
            ld_rax(j, A);
            if (st[d - 1] == JT_I) eb3(j, 0x48, 0xF7, 0xD8);                  /* neg rax */
            else { eb(j, 0x48); eb3(j, 0x0F, 0xBA, 0xF8); eb(j, 0x3F); }       /* btc rax,63 */
            st_rax(j, A);
            break;
        }
        case OP_NOT: {
            int32_t A = SLOT(j, d - 1);
            if (st[d - 1] == JT_F) {
                ld_xmm(j, 0, A);
                eb3(j, 0x66, 0x0F, 0x57); eb(j, 0xC9);      /* xorpd xmm1,xmm1 */
                eb3(j, 0x66, 0x0F, 0x2E); eb(j, 0xC1);      /* ucomisd xmm0,xmm1 */
                eb3(j, 0x0F, 0x94, 0xC0);                   /* sete al */
                eb3(j, 0x0F, 0x9B, 0xC1);                   /* setnp cl */
                eb2(j, 0x20, 0xC8);                         /* and al,cl */
                eb3(j, 0x0F, 0xB6, 0xC0);                   /* movzx eax,al */
            } else {
                ld_rax(j, A);
                eb3(j, 0x48, 0x85, 0xC0);                   /* test rax,rax */
                eb3(j, 0x0F, 0x94, 0xC0);                   /* sete al */
                eb3(j, 0x0F, 0xB6, 0xC0);
            }
            st_rax(j, A);
            break;
        }
        case OP_EQ: case OP_NE: case OP_LT: case OP_GT: case OP_LE: case OP_GE: {
            uint8_t a = st[d - 2], b = st[d - 1];
            int32_t A = SLOT(j, d - 2), B = SLOT(j, d - 1);
            if ((a == JT_B) != (b == JT_B)) {
                /* bool so với số: luôn khác (khớp tval_eq) */
                eb(j, 0xB8); e32(j, m->op == OP_NE ? 1 : 0);
                st_rax(j, A);
                break;
            }
            if ((a == JT_I || a == JT_B) && (b == JT_I || b == JT_B)) {
                ld_rax(j, A);
                eb2(j, 0x48, 0x3B); m_rsp(j, 0, B);         /* cmp rax,[B] */
                eb3(j, 0x0F, (uint8_t)jcc_of(m->op, 0), 0xC0);
                eb3(j, 0x0F, 0xB6, 0xC0);
                st_rax(j, A);
                break;
            }
            ld_f(j, 0, A, a);
            ld_f(j, 1, B, b);
            if (m->op == OP_EQ || m->op == OP_NE) {
                eb3(j, 0x66, 0x0F, 0x2E); eb(j, 0xC1);      /* ucomisd xmm0,xmm1 */
                if (m->op == OP_EQ) {
                    eb3(j, 0x0F, 0x94, 0xC0); eb3(j, 0x0F, 0x9B, 0xC1); eb2(j, 0x20, 0xC8);  /* sete al; setnp cl; and */
                } else {
                    eb3(j, 0x0F, 0x95, 0xC0); eb3(j, 0x0F, 0x9A, 0xC1); eb2(j, 0x08, 0xC8);  /* setne al; setp cl; or */
                }
            } else {
                if (m->op == OP_LT || m->op == OP_LE) { eb3(j, 0x66, 0x0F, 0x2E); eb(j, 0xC8); }  /* ucomisd xmm1,xmm0 */
                else { eb3(j, 0x66, 0x0F, 0x2E); eb(j, 0xC1); }                                   /* ucomisd xmm0,xmm1 */
                eb3(j, 0x0F, (uint8_t)jcc_of(m->op, 1), 0xC0);
            }
            eb3(j, 0x0F, 0xB6, 0xC0);
            st_rax(j, A);
            break;
        }
        case OP_POP: case OP_NOP: break;
        case OP_DUP:
            ld_rax(j, SLOT(j, d - 1)); st_rax(j, SLOT(j, d)); break;
        case OP_JMP: {
            uint8_t pre[1] = {0xE9};
            jto(j, pre, 1, m, d);
            break;
        }
        case OP_JMPF: case OP_JMPT: {
            int32_t A = SLOT(j, d - 1);
            if (st[d - 1] == JT_F) {
                ld_xmm(j, 0, A);
                eb3(j, 0x66, 0x0F, 0x57); eb(j, 0xC9);      /* xorpd xmm1,xmm1 */
                eb3(j, 0x66, 0x0F, 0x2E); eb(j, 0xC1);      /* ucomisd xmm0,xmm1 */
                if (m->op == OP_JMPF) {
                    /* falsy <=> ZF=1 && PF=0 : jp skip ; je target */
                    eb2(j, 0x7A, 0x06);
                    { uint8_t pre[2] = {0x0F, 0x84}; jto(j, pre, 2, m, d - 1); }
                } else {
                    { uint8_t pre[2] = {0x0F, 0x85}; jto(j, pre, 2, m, d - 1); }
                    { uint8_t pre[2] = {0x0F, 0x8A}; jto(j, pre, 2, m, d - 1); }
                }
            } else {
                ld_rax(j, A);
                eb3(j, 0x48, 0x85, 0xC0);
                uint8_t pre[2] = {0x0F, (uint8_t)(m->op == OP_JMPF ? 0x84 : 0x85)};
                jto(j, pre, 2, m, d - 1);
            }
            break;
        }
        case OP_IDX: case OP_SETIDX: {
            /* list ở slot L, chỉ số ở slot X; (SETIDX: giá trị ở slot V) */
            int set = m->op == OP_SETIDX;
            int32_t L = SLOT(j, d - (set ? 3 : 2)), X = SLOT(j, d - (set ? 2 : 1)), V = SLOT(j, d - 1);
            uint8_t et = list_elem(st[d - (set ? 3 : 2)]);
            int ttag = et == JT_I ? T_INT : et == JT_F ? T_FLOAT : T_BOOL;
            ld_rax(j, L);
            ld_rcx(j, X);
            eb(j, 0x48); eb2(j, 0x3B, 0x88); e32(j, (int32_t)offsetof(TList, len));   /* cmp rcx,[rax+len] */
            jguard(j, 0x83);                                                         /* jae (âm/ngoài) */
            eb(j, 0x48); eb2(j, 0x8B, 0x80); e32(j, (int32_t)offsetof(TList, items)); /* mov rax,[rax+items] */
            eb(j, 0x48); eb3(j, 0xC1, 0xE1, 0x05);                                   /* shl rcx,5 (32 byte/TVal) */
            eb3(j, 0x48, 0x01, 0xC8);                                                /* add rax,rcx */
            if (!set) {
                eb3(j, 0x83, 0x38, (uint8_t)ttag); jguard(j, 0x85);   /* phần tử đúng kiểu */
                eb(j, 0x48); eb3(j, 0x8B, 0x40, 0x08);                /* mov rax,[rax+8] */
                if (et == JT_B) eb3(j, 0x83, 0xE0, 0x01);              /* and eax,1 */
                st_rax(j, L);
            } else {
                eb3(j, 0x83, 0x38, (uint8_t)T_FLOAT); jguard(j, 0x87);  /* phần tử cũ phải là vô hướng (nil..float) */
                eb2(j, 0xC7, 0x00); e32(j, ttag);                       /* mov dword [rax],tag */
                eb2(j, 0x48, 0x8B); m_rsp(j, 1, V);                     /* mov rcx,[V] */
                eb(j, 0x48); eb3(j, 0x89, 0x48, 0x08);                  /* mov [rax+8],rcx */
            }
            break;
        }
        case OP_LEN:
            ld_rax(j, SLOT(j, d - 1));
            eb(j, 0x48); eb2(j, 0x8B, 0x80); e32(j, (int32_t)offsetof(TList, len));
            st_rax(j, SLOT(j, d - 1));
            break;
        case OP_RET:
            ld_rax(j, SLOT(j, d - 1));
            jepi(j, 0);
            break;
        case OP_CALL: {
            int k = (int)m->k;
            int fd = d - k - 1;
            if (st[fd] == JT_NAT) {
                int nat = s->sv[fd];
                int32_t A = SLOT(j, fd), X = SLOT(j, d - 1);
                uint8_t xt = st[d - 1];
                if (nat == NAT_LEN) {
                    ld_rax(j, X);
                    eb(j, 0x48); eb2(j, 0x8B, 0x80); e32(j, (int32_t)offsetof(TList, len));  /* mov rax,[rax+len] */
                    st_rax(j, A);
                } else if (nat == NAT_INT) {
                    if (xt == JT_F) {
                        ld_xmm(j, 0, X);
                        eb(j, 0xF2); eb(j, 0x48); eb3(j, 0x0F, 0x2C, 0xC0);   /* cvttsd2si rax,xmm0 */
                    } else ld_rax(j, X);
                    st_rax(j, A);
                } else if (nat == NAT_FLOAT) {
                    ld_f(j, 0, X, xt == JT_B ? JT_I : xt);
                    st_xmm0(j, A);
                } else if (nat == NAT_ABS) {
                    ld_rax(j, X);
                    if (xt == JT_I) {
                        eb3(j, 0x48, 0x85, 0xC0);           /* test rax,rax */
                        eb2(j, 0x79, 0x03);                 /* jns +3 */
                        eb3(j, 0x48, 0xF7, 0xD8);           /* neg rax */
                    } else { eb(j, 0x48); eb3(j, 0x0F, 0xBA, 0xF0); eb(j, 0x3F); }  /* btr rax,63 */
                    st_rax(j, A);
                } else {   /* min/max 2 đối cùng kiểu */
                    int32_t P = SLOT(j, d - 2), Q = SLOT(j, d - 1);
                    int ismax = (nat == NAT_MAX);
                    if (xt == JT_I) {
                        ld_rax(j, P); ld_rcx(j, Q);
                        eb3(j, 0x48, 0x39, 0xC8);           /* cmp rax,rcx */
                        /* max: a>=b ? a : b  -> cmovl rax,rcx ; min: a<=b ? a : b -> cmovg */
                        eb(j, 0x48); eb3(j, 0x0F, (uint8_t)(ismax ? 0x4C : 0x4F), 0xC1);
                        st_rax(j, A);
                    } else {
                        ld_xmm(j, 0, P); ld_xmm(j, 1, Q);
                        /* max: nếu a < b thật sự -> b ; min: nếu a > b thật sự -> b */
                        if (ismax) { eb3(j, 0x66, 0x0F, 0x2E); eb(j, 0xC8); }   /* ucomisd xmm1,xmm0 */
                        else { eb3(j, 0x66, 0x0F, 0x2E); eb(j, 0xC1); }         /* ucomisd xmm0,xmm1 */
                        ld_rax(j, P);
                        eb2(j, 0x76, 0x08);                 /* jbe +8 (không thật sự) */
                        eb2(j, 0x48, 0x8B); m_rsp(j, 0, Q); /* mov rax,[Q] (8 byte) */
                        st_rax(j, A);
                    }
                }
                break;
            }
            {
                TVal *g = &G[s->sv[fd]];
                TFn *F = g->u.fn;
                /* guard: global vẫn là đúng hàm F */
                mov_rax_imm(j, (int64_t)(intptr_t)g);
                eb3(j, 0x83, 0x38, (uint8_t)T_FN); jguard(j, 0x85);
                mov_rcx_imm(j, (int64_t)(intptr_t)F);
                eb(j, 0x48); eb3(j, 0x39, 0x48, 0x08); jguard(j, 0x85);   /* cmp [rax+8],rcx */
                /* arg1 = &slot(fd+1), arg2 = ctx */
                if (JIT_SYSV) { eb3(j, 0x48, 0x8D, 0xBC); eb(j, 0x24); e32(j, SLOT(j, fd + 1)); eb3(j, 0x48, 0x89, 0xDE); }
                else { eb3(j, 0x48, 0x8D, 0x8C); eb(j, 0x24); e32(j, SLOT(j, fd + 1)); eb3(j, 0x48, 0x89, 0xDA); }
                if (j->fn && F == j->fn && j->jf->estate == 1) {
                    eb(j, 0xE8);                    /* call rel32 -> đầu entry (offset 0) */
                    e32(j, (int32_t)(0 - (int64_t)(j->n + 4)));
                } else {
                    TJitFn *fj = (TJitFn *)F->jit;
                    mov_rax_imm(j, (int64_t)(intptr_t)fj->entry);
                    eb2(j, 0xFF, 0xD0);
                    /* giữ ref callee (code này nhảy thẳng vào nó) */
                    if (F != j->fn) {
                        int seen = 0;
                        for (int q = 0; q < j->jf->ncallees; q++) if (j->jf->callees[q] == F) seen = 1;
                        if (!seen) {
                            if (j->jf->ncallees >= j->jf->capcallees) {
                                j->jf->capcallees = j->jf->capcallees ? j->jf->capcallees * 2 : 4;
                                j->jf->callees = realloc(j->jf->callees, (size_t)j->jf->capcallees * sizeof(TFn *));
                            }
                            j->jf->callees[j->jf->ncallees++] = F;
                            F->refc++;
                        }
                    }
                }
                eb3(j, 0x83, 0x3B, 0x00);           /* cmp dword [rbx],0 */
                if (j->osr) jguard(j, 0x85);        /* callee huỷ: deopt tại CALL (callee thuần -> gọi lại được) */
                else jepi(j, 0x85);                 /* entry: truyền huỷ lên */
                st_rax(j, SLOT(j, fd));
            }
            break;
        }
        default:
            jc_fail(j, "codegen op lạ");
            return 0;
        }
    }
    return !j->fail;
}

/* epilogue + abort; vá mọi nhảy; chép vào bộ nhớ thực thi */
static int finish(JC *j, JitUnit *u) {
    size_t abort_at, epi_at;
    abort_at = j->n;
    eb2(j, 0xC7, 0x03); e32(j, 1);           /* mov dword [rbx],1 */
    epi_at = j->n;
    eb3(j, 0x48, 0x81, 0xC4); e32(j, j->frame);   /* add rsp,frame */
    eb2(j, 0x41, 0x5C);                      /* pop r12 */
    eb(j, 0x5B);                             /* pop rbx */
    eb(j, 0xC3);
    if (j->ndfix) {
        /* deopt: [rbx+4] = mã điểm; [rbx] = 2; chép nsave qword frame ra ctx->save */
        int32_t nsave = j->nl + j->maxd_seen + 2;
        size_t common = 0;
        size_t *stub = (size_t *)malloc(sizeof(size_t) * (size_t)j->ndfix);
        for (int q = 0; q < j->ndfix; q++) {
            stub[q] = j->n;
            eb3(j, 0xC7, 0x43, 0x04); e32(j, q);       /* mov dword [rbx+4], q */
            eb(j, 0xE9); e32(j, 0);                     /* jmp common (vá dưới) */
        }
        common = j->n;
        eb2(j, 0xC7, 0x03); e32(j, 2);                  /* mov dword [rbx],2 */
        eb3(j, 0x48, 0xC7, 0xC1); e32(j, nsave);        /* mov rcx,nsave */
        eb(j, 0x48); eb3(j, 0x8D, 0x44, 0x24); eb(j, 0x20);   /* lea rax,[rsp+32] */
        eb(j, 0x48); eb3(j, 0x8B, 0x53, 0x10);          /* mov rdx,[rbx+16] */
        eb3(j, 0x4C, 0x8B, 0x00);                       /* L: mov r8,[rax] */
        eb3(j, 0x4C, 0x89, 0x02);                       /* mov [rdx],r8 */
        eb(j, 0x48); eb3(j, 0x83, 0xC0, 0x08);          /* add rax,8 */
        eb(j, 0x48); eb3(j, 0x83, 0xC2, 0x08);          /* add rdx,8 */
        eb3(j, 0x48, 0xFF, 0xC9);                       /* dec rcx */
        eb2(j, 0x75, 0xED);                             /* jnz L (-19) */
        eb(j, 0xE9); e32(j, 0);                         /* jmp epi */
        patch32(j, j->n - 4, epi_at);
        for (int q = 0; q < j->ndfix; q++) {
            patch32(j, j->dfix[q].at, stub[q]);
            patch32(j, stub[q] + 8, common);
        }
        free(stub);
    }
    for (int i = 0; i < j->nfix; i++) patch32(j, j->fix[i].at, j->label[j->fix[i].to]);
    for (int i = 0; i < j->nafix; i++) patch32(j, j->afix[i].at, abort_at);
    for (int i = 0; i < j->nefix; i++) patch32(j, j->efix[i].at, epi_at);
    {
        /* debug: TVM_JITDUMP=thư_mục -> ghi code máy (xem bằng objdump -b binary) */
        const char *dd = getenv("TVM_JITDUMP");
        static int dumpno = 0;
        if (dd && *dd) {
            char path[512];
            snprintf(path, sizeof(path), "%s/jit_%d_%d.bin", dd, j->fn->entry, dumpno++);
            FILE *f = fopen(path, "wb");
            if (f) { fwrite(j->b, 1, j->n, f); fclose(f); }
        }
    }
    size_t sz = (j->n + 4095) & ~(size_t)4095;
    uint8_t *mem = (uint8_t *)jit_alloc_exec(sz);
    if (!mem) return 0;
    memcpy(mem, j->b, j->n);
    if (!jit_seal(mem, sz)) { jit_free_exec(mem, sz); return 0; }
    u->mem = mem; u->size = sz;
    return 1;
}

static void jc_free(JC *j) {
    if (j->in) {
        for (int i = 0; i < j->nops; i++) { free(j->in[i].st); free(j->in[i].sv); free(j->in[i].lt); }
        free(j->in);
    }
    free(j->ops); free(j->b); free(j->label); free(j->fix); free(j->afix); free(j->efix);
    free(j->dfix);
}

static int jc_setup(JC *j, TVM *vm, TFn *fn, int nest) {
    size_t beg, end;
    memset(j, 0, sizeof(*j));
    j->vm = vm; j->fn = fn; j->jf = jit_of(fn); j->nest = nest;
    j->nl = fn->nlocals;
    if (!body_range(vm->prog, fn, &beg, &end)) { jc_fail(j, "không xác định thân hàm"); return 0; }
    if (!decode(j, vm->prog, beg, end)) return 0;
    j->maxd = 64;
    j->in = (AState *)calloc((size_t)j->nops + 1, sizeof(AState));
    for (int i = 0; i < j->nops; i++) st_alloc(j, &j->in[i]);
    j->label = (size_t *)calloc((size_t)j->nops + 1, sizeof(size_t));
    return 1;
}

/* bảng deopt: trạng thái (stack + locals) tại từng điểm guard / lối ra */
static void build_deopts(JC *j, JitOsr *o) {
    int nl = j->nl;
    o->nsave = j->nl + j->maxd_seen + 2;
    o->ndeopt = j->ndfix;
    o->deopts = (JitDeopt *)calloc((size_t)j->ndfix + 1, sizeof(JitDeopt));
    for (int q = 0; q < j->ndfix; q++) {
        AState *as = &j->in[j->dfix[q].op];
        JitDeopt *dp = &o->deopts[q];
        int depth = j->dfix[q].xdepth >= 0 ? j->dfix[q].xdepth : as->depth;
        dp->pc = j->dfix[q].xdepth >= 0 ? j->dfix[q].xpc : j->ops[j->dfix[q].op].pc;
        dp->depth = depth;
        dp->st = (uint8_t *)malloc((size_t)depth + 1);
        dp->sv = (int32_t *)malloc(((size_t)depth + 1) * sizeof(int32_t));
        dp->lt = (uint8_t *)malloc((size_t)nl + 1);
        memcpy(dp->st, as->st, (size_t)depth);
        memcpy(dp->sv, as->sv, (size_t)depth * sizeof(int32_t));
        memcpy(dp->lt, as->lt, (size_t)nl);
    }
}

static void set_frame(JC *j) {
    int32_t f = 32 + 8 * (j->nl + j->maxd_seen + 2);
    if ((f & 15) != 8) f += 8;              /* sau 2 push + ret: rsp chia hết 16 khi call */
    j->frame = f;
}

/* biên dịch entry đặc hoá theo argty */
static int compile_entry(TVM *vm, TFn *fn, const uint8_t *argty, int nest) {
    TJitFn *jf = jit_of(fn);
    JC j;
    if (jf->estate != 0) return jf->estate == 2;
    if (getenv("TVM_NOJIT")) { jf->estate = 3; fn->jit_off |= 3; return 0; }
    jf->estate = 1;
    jf->nargs = fn->nargs;
    memcpy(jf->argty, argty, (size_t)fn->nargs);
    int ok = jc_setup(&j, vm, fn, nest);
    uint8_t *lt = (uint8_t *)calloc((size_t)fn->nlocals + 1, 1);
    for (int i = 0; i < fn->nargs && i < fn->nlocals; i++) lt[i] = argty[i];
    uint8_t retty = JT_UNK;
    if (ok) {
        /* đoán kiểu trả về cho lời gọi đệ quy: thử I, F, B */
        static const uint8_t guesses[3] = {JT_I, JT_F, JT_B};
        ok = 0;
        for (int gi = 0; gi < 3 && !ok; gi++) {
            j.self_retty_guess = guesses[gi];
            j.used_guess = 0;
            j.fail = 0; j.why = NULL;
            for (int i = 0; i < j.nops; i++) j.in[i].valid = 0;
            if (analyze(&j, 0, lt, &retty) && (!j.used_guess || retty == guesses[gi])) ok = 1;
            if (!ok && !j.used_guess) break;     /* thất bại không do đoán kiểu */
        }
    }
    if (ok) {
        jf->retty = retty;
        set_frame(&j);
        prologue(&j, argty, fn->nargs);
        ok = gen_body(&j) && finish(&j, &jf->unit);
    }
    if (!ok && getenv("TVM_JITLOG"))
        fprintf(stderr, "[jit] entry fail @%d: %s\n", fn->entry, j.why ? j.why : "?");
    jc_free(&j);
    free(lt);
    if (!ok) { jf->estate = 3; fn->jit_off |= 1; return 0; }
    jf->entry = jf->unit.mem;
    jf->estate = 2;
    fn->native = jf->entry;
    if (getenv("TVM_JITLOG")) fprintf(stderr, "[jit] entry ok @%d (%zu bytes)\n", fn->entry, jf->unit.size);
    return 1;
}

static uint8_t tag_ty(TType t) {
    return t == T_INT ? JT_I : t == T_FLOAT ? JT_F : t == T_BOOL ? JT_B : JT_UNK;
}
static uint8_t val_ty(const TVal *v) {
    if (v->t == T_LIST) return list_ty(v->u.list);
    return tag_ty(v->t);
}
/* guard lúc vào native: list chỉ cần đúng tag (phần tử được guard từng lần đọc) */
static int ty_ok(const TVal *v, uint8_t want) {
    if (is_list_t(want)) return v->t == T_LIST;
    return tag_ty(v->t) == want;
}

void t_jit_maybe(TVM *vm, TFn *fn, TVal *args) {
    uint8_t at[256];
    if (fn->native || (fn->jit_off & 1)) return;
    if (++fn->ncalls < JIT_THRESHOLD) return;
    for (int i = 0; i < fn->nargs; i++) {
        at[i] = val_ty(&args[i]);
        if (at[i] == JT_UNK) { fn->jit_off |= 1; return; }
    }
    compile_entry(vm, fn, at, 0);
}

static int to_ret(uint8_t ty, int64_t r, TVal *ret) {
    if (ty == JT_I) { ret->t = T_INT; ret->u.i = r; }
    else if (ty == JT_F) { ret->t = T_FLOAT; memcpy(&ret->u.f, &r, 8); }
    else { ret->t = T_BOOL; ret->u.b = (int)(r & 1); }
    return 1;
}

static int64_t val_bits(const TVal *v) {
    int64_t x = 0;
    if (v->t == T_INT) x = v->u.i;
    else if (v->t == T_FLOAT) memcpy(&x, &v->u.f, 8);
    else if (v->t == T_BOOL) x = v->u.b ? 1 : 0;
    else if (v->t == T_LIST) x = (int64_t)(intptr_t)v->u.list;   /* mượn: local/arg giữ ref */
    return x;
}
/* giá trị native (theo kiểu trừu tượng) -> TVal sở hữu được */
static TVal bits_val(TVM *vm, uint8_t ty, int64_t x, int32_t sv) {
    TVal v;
    v.t = T_NIL;
    if (ty == JT_I) { v.t = T_INT; v.u.i = x; }
    else if (ty == JT_F) { v.t = T_FLOAT; memcpy(&v.u.f, &x, 8); }
    else if (ty == JT_B) { v.t = T_BOOL; v.u.b = (int)(x & 1); }
    else if (is_list_t(ty)) { v.t = T_LIST; v.u.list = (TList *)(intptr_t)x; tval_retain(&v); }
    else if (ty == JT_FN || ty == JT_NAT) { v = vm->prog->globals[sv]; tval_retain(&v); }
    return v;
}

int t_jit_call(TVM *vm, TFn *fn, TVal *args, int argc, TVal *ret) {
    TJitFn *jf = (TJitFn *)fn->jit;
    int64_t in[256];
    (void)vm;
    if (!fn->native || !jf || jf->estate != 2 || argc != jf->nargs) return 0;
    for (int i = 0; i < argc; i++) {
        if (!ty_ok(&args[i], jf->argty[i])) return 0;
        in[i] = val_bits(&args[i]);
    }
    JitCtx ctx;
    ctx.abort = 0; ctx.deopt_id = 0; ctx.save = NULL;
    ctx.stack_limit = (char *)((uintptr_t)&ctx - JIT_STACK_BUDGET);
    int64_t r = ((JitEntry)jf->entry)(in, &ctx);
    if (ctx.abort) {
        if (++jf->aborts > JIT_MAX_ABORTS) { fn->native = NULL; fn->jit_off |= 1; }
        return 0;
    }
    return to_ret(jf->retty, r, ret);
}

int t_jit_osr(TVM *vm, TFn *fn, TVal *locals, size_t pc, TVal *ret) {
    TJitFn *jf = jit_of(fn);
    JitOsr *o = NULL;
    int nl = fn->nlocals;
    for (int i = 0; i < jf->nosr; i++) if (jf->osr[i].pc == pc) { o = &jf->osr[i]; break; }
    if (!o) {
        if (jf->nosr >= JIT_MAX_OSR || getenv("TVM_NOJIT")) { fn->jit_off |= 2; return 0; }
        o = &jf->osr[jf->nosr++];
        memset(o, 0, sizeof(*o));
        o->pc = pc;
        o->lty = (uint8_t *)calloc((size_t)nl + 1, 1);
        for (int i = 0; i < nl; i++) {
            /* local đã bị closure bắt (cell): ghi đè lúc deopt sẽ phá cell -> không OSR */
            if (locals[i].t == T_CELL) { o->state = 2; fn->jit_off |= 2; return 0; }
            o->lty[i] = val_ty(&locals[i]);
        }
        JC j;
        int ok = jc_setup(&j, vm, fn, 0);
        j.osr = 1;
        int start = -1;
        if (ok) {
            for (int i = 0; i < j.nops; i++) if (j.ops[i].head && j.ops[i].pc == pc) { start = i; break; }
            ok = start >= 0;
        }
        uint8_t retty = JT_UNK;
        if (ok) ok = analyze(&j, start, o->lty, &retty);
        if (ok) {
            /* locals UNK sau phân tích tại đầu vòng lặp thì không cần chép */
            for (int i = 0; i < nl; i++) if (j.in[start].lt[i] == JT_UNK) o->lty[i] = JT_UNK;
            o->retty = retty;
            set_frame(&j);
            prologue(&j, o->lty, nl);
            /* nhảy vào đầu vòng lặp */
            { uint8_t pre[1] = {0xE9}; jmp_fix(&j, pre, 1, start); }
            ok = gen_body(&j) && finish(&j, &o->unit);
            if (ok) build_deopts(&j, o);
        }
        if (!ok && getenv("TVM_JITLOG"))
            fprintf(stderr, "[jit] osr fail @%d pc=%zu: %s\n", fn->entry, pc, j.why ? j.why : "?");
        jc_free(&j);
        if (!ok) { o->state = 2; fn->jit_off |= 2; return 0; }
        o->entry = o->unit.mem;
        o->state = 1;
        if (getenv("TVM_JITLOG")) fprintf(stderr, "[jit] osr ok @%d pc=%zu\n", fn->entry, pc);
    }
    if (o->state != 1) return 0;
    {
        int64_t stackbuf[64];
        int64_t *in = nl <= 64 ? stackbuf : (int64_t *)malloc((size_t)nl * sizeof(int64_t));
        for (int i = 0; i < nl; i++) {
            if (o->lty[i] == JT_UNK) { in[i] = 0; continue; }
            if (!ty_ok(&locals[i], o->lty[i])) {
                if (in != stackbuf) free(in);
                if (++o->misses > 16) fn->jit_off |= 2;
                return 0;
            }
            in[i] = val_bits(&locals[i]);
        }
        int64_t savebuf[128];
        int64_t *save = o->nsave <= 128 ? savebuf : (int64_t *)malloc((size_t)o->nsave * sizeof(int64_t));
        JitCtx ctx;
        ctx.abort = 0; ctx.deopt_id = 0; ctx.save = save;
        ctx.stack_limit = (char *)((uintptr_t)&ctx - JIT_STACK_BUDGET);
        int64_t r = ((JitEntry)o->entry)(in, &ctx);
        if (in != stackbuf) free(in);
        if (ctx.abort == 2 && ctx.deopt_id >= 0 && ctx.deopt_id < o->ndeopt) {
            /* deopt có trạng thái: ghi locals + đẩy stack ảo, interpreter chạy tiếp tại pc */
            JitDeopt *dp = &o->deopts[ctx.deopt_id];
            for (int i = 0; i < nl; i++) {
                uint8_t t = dp->lt[i];
                if (t == JT_I || t == JT_F || t == JT_B) {
                    tval_free(&locals[i]);
                    locals[i] = bits_val(vm, t, save[i], 0);
                }
            }
            for (int q = 0; q < dp->depth; q++)
                tvm_push(vm, bits_val(vm, dp->st[q], save[nl + q], dp->sv[q]));
            vm->pc = dp->pc;
            if (save != savebuf) free(save);
            if (++jf->aborts > JIT_MAX_ABORTS * 4) fn->jit_off |= 2;
            return 2;
        }
        if (save != savebuf) free(save);
        if (ctx.abort) {
            if (++jf->aborts > JIT_MAX_ABORTS) fn->jit_off |= 2;
            return 0;
        }
        return to_ret(o->retty, r, ret);
    }
}

int t_jit_callees(TFn *fn, TFn ***out) {
    TJitFn *jf = (TJitFn *)fn->jit;
    if (!jf) return 0;
    *out = jf->callees;
    return jf->ncallees;
}

/* ================= vòng lặp top-level (main) ================= */
#define JIT_MAX_MAIN 32
typedef struct {
    TProg *prog;
    size_t header;
    int state;                  /* 1 ok, 2 hỏng */
    JitOsr u;                   /* entry + bảng deopt (lty: kiểu global giả) */
    int32_t *pglob;
    int npg;
} JitMain;
static JitMain jmains[JIT_MAX_MAIN];
static int njmain;
static TJitFn jmain_fn;         /* giữ ref các hàm mà code main gọi thẳng */

int t_jit_osr_main(TVM *vm, size_t header, size_t end) {
    JitMain *mu = NULL;
    TProg *P = vm->prog;
    TVal *G = P->globals;
    for (int i = 0; i < njmain; i++)
        if (jmains[i].prog == P && jmains[i].header == header) { mu = &jmains[i]; break; }
    if (!mu) {
        if (njmain >= JIT_MAX_MAIN || getenv("TVM_NOJIT")) return -1;
        mu = &jmains[njmain++];
        memset(mu, 0, sizeof(*mu));
        mu->prog = P; mu->header = header;
        JC j;
        memset(&j, 0, sizeof(j));
        j.vm = vm; j.fn = NULL; j.jf = &jmain_fn; j.osr = 1; j.mainmode = 1;
        j.gpseudo = (int32_t *)malloc(sizeof(int32_t) * (P->nglobals ? P->nglobals : 1));
        for (size_t g = 0; g < P->nglobals; g++) j.gpseudo[g] = -1;
        int ok = decode(&j, P, header, end);
        if (ok) {
            /* global được đọc/ghi trong vùng (trừ builtin và hàm) -> local giả */
            int np = 0;
            j.pglob = (int32_t *)malloc(sizeof(int32_t) * ((size_t)j.nops + 1));
            for (int i = 0; i < j.nops; i++) {
                MOp *m = &j.ops[i];
                if (m->op != OP_LOADG && m->op != OP_STOREG && m->op != OP_BINDG) continue;
                if ((size_t)m->k >= P->nglobals || is_nat_ok(m->k)) continue;
                if (G[m->k].t == T_FN && m->op == OP_LOADG) continue;
                if (j.gpseudo[m->k] < 0) { j.gpseudo[m->k] = np; j.pglob[np++] = (int32_t)m->k; }
            }
            j.nl = np;
            j.maxd = 64;
            j.in = (AState *)calloc((size_t)j.nops + 1, sizeof(AState));
            for (int i = 0; i < j.nops; i++) st_alloc(&j, &j.in[i]);
            j.label = (size_t *)calloc((size_t)j.nops + 1, sizeof(size_t));
            uint8_t *lt = (uint8_t *)calloc((size_t)np + 1, 1);
            for (int q = 0; q < np; q++) {
                TVal *gv = &G[j.pglob[q]];
                lt[q] = gv->t == T_CELL ? JT_UNK : val_ty(gv);
            }
            uint8_t retty = JT_UNK;
            ok = analyze(&j, 0, lt, &retty) || (!j.fail);
            /* main không có RET: analyze báo "không có return" khi mọi thứ khác ổn */
            if (j.fail && j.why && strcmp(j.why, "không có return") == 0) { ok = 1; j.fail = 0; j.why = NULL; }
            else if (j.fail) ok = 0;
            if (ok) {
                for (int q = 0; q < np; q++) if (j.in[0].lt[q] == JT_UNK) lt[q] = JT_UNK;
                mu->u.lty = lt;
                set_frame(&j);
                { uint8_t none[1] = {0}; prologue(&j, none, 0); }
                ok = gen_body(&j) && finish(&j, &mu->u.unit);
                if (ok) build_deopts(&j, &mu->u);
            } else free(lt);
            mu->pglob = j.pglob; j.pglob = NULL;
            mu->npg = np;
        }
        if (!ok && getenv("TVM_JITLOG"))
            fprintf(stderr, "[jit] main fail pc=%zu: %s\n", header, j.why ? j.why : "?");
        jc_free(&j);
        free(j.gpseudo); free(j.pglob);
        if (!ok) { mu->state = 2; return -1; }
        mu->u.entry = mu->u.unit.mem;
        mu->state = 1;
        if (getenv("TVM_JITLOG")) fprintf(stderr, "[jit] main ok pc=%zu\n", header);
    }
    if (mu->state != 1) return -1;
    for (int q = 0; q < mu->npg; q++) {
        uint8_t t = mu->u.lty[q];
        if (t != JT_UNK && !ty_ok(&G[mu->pglob[q]], t)) {
            if (++mu->u.misses > 16) { mu->state = 2; return -1; }
            return 0;
        }
    }
    {
        int64_t savebuf[128];
        int64_t *save = mu->u.nsave <= 128 ? savebuf : (int64_t *)malloc((size_t)mu->u.nsave * sizeof(int64_t));
        JitCtx ctx;
        ctx.abort = 0; ctx.deopt_id = 0; ctx.save = save;
        ctx.stack_limit = (char *)((uintptr_t)&ctx - JIT_STACK_BUDGET);
        ((JitEntry)mu->u.entry)(NULL, &ctx);
        if (ctx.abort == 2 && ctx.deopt_id >= 0 && ctx.deopt_id < mu->u.ndeopt) {
            /* global đã ghi thẳng bộ nhớ: chỉ cần đẩy stack ảo + đặt pc */
            JitDeopt *dp = &mu->u.deopts[ctx.deopt_id];
            for (int q = 0; q < dp->depth; q++)
                tvm_push(vm, bits_val(vm, dp->st[q], save[mu->npg + q], dp->sv[q]));
            vm->pc = dp->pc;
            if (save != savebuf) free(save);
            return 2;
        }
        if (save != savebuf) free(save);
        return 0;
    }
}

int t_jit_root_fns(TFn ***out) {
    *out = jmain_fn.callees;
    return jmain_fn.ncallees;
}

void t_jit_free_fn(TFn *fn) {
    TJitFn *jf = (TJitFn *)fn->jit;
    fn->native = NULL;
    if (!jf) return;
    fn->jit = NULL;
    if (jf->unit.mem) jit_free_exec(jf->unit.mem, jf->unit.size);
    for (int i = 0; i < jf->nosr; i++) {
        if (jf->osr[i].unit.mem) jit_free_exec(jf->osr[i].unit.mem, jf->osr[i].unit.size);
        free(jf->osr[i].lty);
        for (int q = 0; q < jf->osr[i].ndeopt; q++) {
            free(jf->osr[i].deopts[q].st); free(jf->osr[i].deopts[q].sv); free(jf->osr[i].deopts[q].lt);
        }
        free(jf->osr[i].deopts);
    }
    for (int i = 0; i < jf->ncallees; i++) {
        TVal v; v.t = T_FN; v.u.fn = jf->callees[i];
        tval_free(&v);
    }
    free(jf->callees);
    free(jf);
}
