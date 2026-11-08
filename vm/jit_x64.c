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
enum { JT_UNK = 0, JT_I, JT_F, JT_B, JT_FN, JT_NAT, JT_LI, JT_LF, JT_LB, JT_LX,
       JT_S = 10, JT_TYP = 11, JT_D = 12, JT_DN = 13, JT_SN = 14, JT_NIL = 15 };
/* chuỗi hằng: JT_S, slot giữ TVal* mượn (hằng cache trong ic, sống suốt native).
   JT_SN: chuỗi sở hữu được (wrapper TVal* malloc, giữ 1 ref) do str()/nối tạo ra.
   JT_D: dict mượn (local/global giữ ref); JT_DN: dict sở hữu được (NEW_DICT trong
   native, refc=1 + gc_track). Owned (SN/DN) thì deopt/to_ret move (không retain),
   borrowed thì retain; ghi đè/POP owned phải dec (ngược với record: chuỗi wrapper
   KHÔNG track GC nên orphan là leak vĩnh viễn, không chờ sweep được). */
typedef char tjit_ty_check[(JT_I == 1 && JT_F == 2 && JT_B == 3) ? 1 : -1];
static int is_list_t(uint8_t t) { return t >= JT_LI && t <= JT_LX; }
static int is_dict_t(uint8_t t) { return t == JT_D || t == JT_DN; }
static int is_str_t(uint8_t t) { return t == JT_S || t == JT_SN; }
static int is_own_rec_t(uint8_t t);
/* record (struct) mượn: kiểu JT_R0 + chỉ số shape. Shape = (kiểu struct, kiểu đoán của
   từng field). Native chỉ ĐỌC field (guard tag mỗi lần đọc) và gọi method tĩnh; record
   không được gán vào local / trả về (không cần đếm tham chiếu trong code máy). */
#define JT_R0 32
#define JIT_MAX_SHAPES 64
#define JIT_SHAPE_MAXF 32
/* record sở hữu được (tạo trong native qua __new, refc=1): JT_RN0 + shape.
   Borrowed (mượn từ caller/interpreter) là JT_R0 + shape. Mã khác nhau để deopt
   biết move (owned) hay retain (borrowed); merge cấm lẫn lộn trên stack, local
   cho nâng borrowed -> owned cùng shape (gán record mới vào biến cũ). */
#define JT_RN0 96
typedef struct { TDict *tp; int nf; uint8_t fty[JIT_SHAPE_MAXF]; const TVal *keys; } JShape;
static JShape jshapes[JIT_MAX_SHAPES];
static int njshapes;
static int is_rec_t(uint8_t t) { return t >= JT_R0 && t < JT_R0 + njshapes; }
static int is_own_rec_t(uint8_t t) { return t >= JT_RN0 && t < JT_RN0 + njshapes; }
static int is_any_rec_t(uint8_t t) { return is_rec_t(t) || is_own_rec_t(t); }
static int rec_shape(uint8_t t) { return is_own_rec_t(t) ? t - JT_RN0 : t - JT_R0; }
static JShape *shape_of(uint8_t t) { return &jshapes[t - JT_R0]; }
static JShape *shape_of_any(uint8_t t) {
    /* callsite đã kiểm is_any_rec_t; NULL chỉ để gcc chứng minh được bound */
    int s = rec_shape(t);
    return (s < 0 || s >= JIT_MAX_SHAPES) ? NULL : &jshapes[s];
}
static uint8_t bor_of_shape(int s) { return (uint8_t)(JT_R0 + s); }
static uint8_t own_of_shape(int s) { return (uint8_t)(JT_RN0 + s); }
/* callee luôn mượn đối số: chuẩn hoá owned -> borrowed khi tra/biên dịch spec */
static uint8_t norm_ty(uint8_t t) {
    if (is_own_rec_t(t)) return bor_of_shape(rec_shape(t));
    if (t == JT_DN) return JT_D;
    if (t == JT_SN) return JT_S;
    return t;
}
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
    int own_base;               /* offset của mảng cờ OWN trong save[] */
} JitOsr;

/* một bản đặc hoá entry theo bộ kiểu đối số */
#define JIT_MAX_SPEC 4
typedef struct {
    int estate;                 /* 1 đang biên dịch, 2 ok, 3 hỏng / đã tắt */
    uint8_t argty[256];
    uint8_t retty;
    void *entry;
    JitUnit unit;
    int aborts;
} JitSpec;

typedef struct TJitFn {
    JitSpec spec[JIT_MAX_SPEC];
    int nspec;
    int nargs;
    int aborts;                 /* OSR */
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
    TProg *prog;        /* prog chứa code đang biên dịch (fn->prog / prog main) */
    TFn *fn;
    TJitFn *jf;
    JitSpec *sp;        /* bản đặc hoá entry đang biên dịch (NULL: OSR / main) */
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
    /* nodec[l]: local từng được nâng borrowed -> owned tại điểm gộp. Lần ghi đầu
       ở runtime vẫn giữ giá trị borrowed của interpreter nên native KHÔNG được
       nhả ref cũ (giá trị cũ orphan chờ GC); chỉ dec khi chắc chắn native-owned */
    uint8_t *nodec;
    uint8_t retty;        /* kiểu trả về của hàm/vùng (cho epilogue RET dec SN) */
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
        case OP_PUSH_NIL: add_op(j, OP_PUSH_NIL, 0, pc, 1); break;
        case OP_PUSH_STR: add_op(j, OP_PUSH_STR, 0, pc, 1); break;   /* hằng chuỗi: JT_S mượn */
        case OP_SETATTR: add_op(j, OP_SETATTR, 0, pc, 1); break;     /* gán field số: tên đọc lại từ pc */
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
        case OP_BAND: case OP_BOR: case OP_BXOR: case OP_BNOT: case OP_SHL: case OP_SHR:
        case OP_EQ: case OP_NE: case OP_LT: case OP_GT: case OP_LE: case OP_GE:
        case OP_RET: case OP_POP: case OP_DUP: case OP_NOP:
        case OP_IDX: case OP_SETIDX: case OP_LEN:
            add_op(j, op, 0, pc, 1); break;
        case OP_NEW_DICT: add_op(j, OP_NEW_DICT, rd16(c + pc + 1), pc, 1); break;
        case OP_CALL: add_op(j, op, c[pc + 1], pc, 1); break;
        case OP_ATTR:
            /* xs.len trên list == LEN; thuộc tính khác: ngoài phạm vi */
            if (rd32(c + pc + 1) == 3 && memcmp(c + pc + 5, "len", 3) == 0) add_op(j, OP_LEN, 0, pc, 1);
            else add_op(j, OP_ATTR, 0, pc, 1);       /* field record (tên đọc lại từ pc) */
            break;
        case OP_CALLM:
            add_op(j, OP_CALLM, c[pc + 1], pc, 1);   /* k = số đối; tên đọc lại từ pc */
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
        /* nâng borrowed -> owned cùng shape (gán record mới vào biến giữ record
           cũ, vd acc = acc.add(d)): native giữ ref mới, interpreter giữ ref cũ */
        if (is_rec_t(s->lt[l]) && is_own_rec_t(lt[l]) &&
            rec_shape(s->lt[l]) == rec_shape(lt[l])) {
            s->lt[l] = lt[l]; ch = 1;
            if (j->nodec) j->nodec[l] = 1;
            continue;
        }
        /* nâng chuỗi/dict mượn -> sở hữu được (vd k = "w" + str(i)): deopt/codegen
           dùng cờ runtime OWN(j, l) để biết lúc nào giữ giá trị owned thật */
        if ((s->lt[l] == JT_S && lt[l] == JT_SN) ||
            (s->lt[l] == JT_D && lt[l] == JT_DN)) {
            s->lt[l] = lt[l]; ch = 1;
            if (j->nodec) j->nodec[l] = 1;
            continue;
        }
        if ((s->lt[l] == JT_SN && lt[l] == JT_S) ||
            (s->lt[l] == JT_DN && lt[l] == JT_D) ||
            (is_own_rec_t(s->lt[l]) && is_rec_t(lt[l]) &&
             rec_shape(s->lt[l]) == rec_shape(lt[l]))) {
            continue;
        }
        jc_fail(j, "local đổi kiểu qua vòng lặp/nhánh");
        return 0;
    }
    return ch;
}

static int compile_entry(TVM *vm, TFn *fn, const uint8_t *argty, int nest);

/* bản đặc hoá của fj đúng bộ kiểu at (mọi trạng thái), NULL nếu chưa có */
static JitSpec *spec_find(TJitFn *fj, const uint8_t *at, int k) {
    for (int q = 0; q < fj->nspec; q++)
        if (memcmp(fj->spec[q].argty, at, (size_t)k) == 0) return &fj->spec[q];
    return NULL;
}
/* lời gọi F(at) có phải là đệ quy vào chính bản đang biên dịch? */
static int is_self_spec(JC *j, TFn *F, const uint8_t *at, int k) {
    return j->fn && F == j->fn && j->sp && j->sp->estate == 1 &&
           memcmp(j->sp->argty, at, (size_t)k) == 0;
}

/* kiểu trả về + kiểu đối số của callee F (biên dịch nếu cần).
   Đối số record owned được chuẩn hoá thành borrowed (callee chỉ mượn). */
static int callee_info(JC *j, TFn *F, const uint8_t *at, int k, uint8_t *retty) {
    uint8_t nat[256];
    for (int q = 0; q < k; q++) nat[q] = norm_ty(at[q]);
    if (is_self_spec(j, F, nat, k)) {
        /* đệ quy chính nó trong lúc đang biên dịch entry */
        *retty = (uint8_t)j->self_retty_guess;
        j->used_guess = 1;
        return 1;
    }
    if (F->nargs != k) return 0;
    if (F->prog != j->prog) return 0;   /* hàm của module khác (load): code + global khác */
    TJitFn *fj = jit_of(F);
    JitSpec *sp = spec_find(fj, nat, k);
    if (!sp) {
        if (j->nest >= JIT_MAX_NEST) return 0;
        if (!compile_entry(j->vm, F, nat, j->nest + 1)) return 0;
        sp = spec_find(fj, nat, k);
    }
    if (!sp || sp->estate != 2) return 0;
    *retty = sp->retty;
    return 1;
}
/* entry native của callee F(at) đã biên dịch (codegen) */
static void *callee_entry(TFn *F, const uint8_t *at, int k) {
    JitSpec *sp = spec_find((TJitFn *)F->jit, at, k);
    return sp ? sp->entry : NULL;
}

static int is_num(uint8_t t) { return t == JT_I || t == JT_F; }
/* chỉ số field `name` trong shape (-1 nếu không có) */
static int shape_field(const JShape *sh, const uint8_t *name, uint32_t len) {
    for (int i = 0; i < sh->nf; i++) {
        const TVal *k = &sh->keys[i];
        if (k->t == T_STR && k->u.str.n == len && memcmp(k->u.str.s, name, len) == 0) return i;
    }
    return -1;
}
/* method `name` của kiểu struct (NULL nếu không có / không phải hàm T) */
static TFn *shape_method(const JShape *sh, const uint8_t *name, uint32_t len) {
    const TDict *tp = sh->tp;
    for (size_t i = 0; i < tp->len; i++) {
        const TVal *k = &tp->keys[i];
        if (k->t == T_STR && k->u.str.n == len && memcmp(k->u.str.s, name, len) == 0)
            return tp->vals[i].t == T_FN ? tp->vals[i].u.fn : NULL;
    }
    return NULL;
}
static int arg_ok_t(uint8_t t) {
    return t == JT_I || t == JT_F || t == JT_B || is_str_t(t) ||
           is_dict_t(t) || is_list_t(t) || is_any_rec_t(t);
}
/* kiểm tra từ sau micro-op idx tới cuối hàm không còn lệnh nào có thể abort/guard hỏng */
static int no_abort_after(JC *j, int idx) {
    for (int q = 0; q < j->nops; q++) {
        MOp *m = &j->ops[q];
        if ((m->op == OP_JMP || m->op == OP_JMPF || m->op == OP_JMPT) && m->target <= q)
            return 0; /* có vòng lặp trong hàm */
    }
    for (int q = idx + 1; q < j->nops; q++) {
        uint8_t op = j->ops[q].op;
        switch (op) {
        case OP_PUSH_INT: case OP_PUSH_FLOAT: case OP_PUSH_BOOL:
        case OP_PUSH_NIL: case OP_PUSH_STR:
        case OP_LOAD: case OP_STORE: case OP_DECL: case OP_BIND:
        case OP_ADD: case OP_SUB: case OP_MUL:
        case OP_NEG: case OP_NOT:
        case OP_BAND: case OP_BOR: case OP_BXOR: case OP_BNOT: case OP_SHL: case OP_SHR:
        case OP_EQ: case OP_NE: case OP_LT: case OP_GT: case OP_LE: case OP_GE:
        case OP_POP: case OP_DUP: case OP_NOP: case OP_LEN:
        case OP_ATTR: case OP_SETATTR:
        case OP_JMP: case OP_JMPF: case OP_JMPT:
        case OP_RET:
            break;
        default:
            return 0;
        }
    }
    return 1;
}
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
static int shape_reg(TDict *tp, int nf, const uint8_t *fty);

/* thông dịch trừu tượng từ micro-op start với locals lt0 */
static int analyze(JC *j, int start, const uint8_t *lt0, uint8_t *retty_out) {
    int *work = (int *)malloc((size_t)(j->nops + 1) * sizeof(int));
    char *inq = (char *)calloc((size_t)j->nops + 1, 1);
    int nw = 0;
    uint8_t *st = (uint8_t *)calloc((size_t)j->maxd + 2, 1);
    int32_t *sv = (int32_t *)calloc((size_t)j->maxd + 2, sizeof(int32_t));
    uint8_t *lt = (uint8_t *)calloc((size_t)j->nl + 1, 1);
    int retty = JT_UNK;
    TVal *G = j->prog->globals;
    size_t NG = j->prog->nglobals;
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
            case OP_PUSH_NIL: st[d] = JT_NIL; sv[d] = 0; d++; break;
            case OP_PUSH_STR: {
                /* hằng chuỗi mượn cache ic (khởi tạo ở lần chạy interpreter đầu) */
                if (!j->prog->ic || j->prog->ic[m->pc].kind != 4) {
                    jc_fail(j, "chuỗi chưa cache"); break;
                }
                st[d] = JT_S; sv[d] = 0; d++; break;
            }
            case OP_LOAD:
                if (m->k >= j->nl || lt[m->k] == JT_UNK) { jc_fail(j, "đọc local chưa gán / kiểu lạ"); break; }
                st[d] = lt[m->k];
                sv[d] = 0;
                if (is_own_rec_t(st[d])) {
                    st[d] = bor_of_shape(rec_shape(st[d]));  /* mượn xem */
                    if (!j->nodec[m->k]) sv[d] = 1;          /* G8: record tạo mới trong cùng hàm */
                }
                else if (st[d] == JT_DN) st[d] = JT_D;    /* dict mượn xem */
                else if (st[d] == JT_SN) st[d] = JT_S;    /* chuỗi mượn xem */
                d++; break;
            case OP_STORE: case OP_DECL: case OP_BIND:
                if (d < 1) { jc_fail(j, "stack rỗng"); break; }
                d--;
                if (m->k >= j->nl) { jc_fail(j, "slot lạ"); break; }
                if (st[d] == JT_I || st[d] == JT_F || st[d] == JT_B) { lt[m->k] = st[d]; break; }
                if (is_own_rec_t(st[d])) {
                    /* record mới: gán đè mọi kiểu cũ (OWN(j, k) kiểm ở runtime) */
                    lt[m->k] = st[d]; break;
                }
                if (is_rec_t(st[d])) {
                    /* record mượn xem (vd kết quả method trả về tham số): gán đè */
                    if (is_own_rec_t(lt[m->k]) && rec_shape(lt[m->k]) == rec_shape(st[d])) break;
                    lt[m->k] = st[d]; break;
                }
                if (st[d] == JT_SN || st[d] == JT_DN) {
                    /* chuỗi/dict mới: gán đè; codegen kiểm OWN(j, k) để nhả ref cũ */
                    lt[m->k] = st[d]; break;
                }
                if (st[d] == JT_S) {
                    /* G12 + G10: chuỗi mượn gán vào local; OWN(j, k) = 0 ở runtime */
                    if (j->mainmode) { jc_fail(j, "gán chuỗi mượn vào local main"); break; }
                    if (lt[m->k] != JT_SN) lt[m->k] = JT_S;
                    break;
                }
                if (st[d] == JT_D) {
                    /* dict mượn xem: gán đè */
                    if (lt[m->k] != JT_DN) lt[m->k] = JT_D;
                    break;
                }
                jc_fail(j, "gán local kiểu ngoài phạm vi"); break;
            case OP_STOREG: case OP_BINDG: {
                int pi = (j->mainmode && (size_t)m->k < NG) ? j->gpseudo[m->k] : -1;
                if (pi < 0) { jc_fail(j, "ghi global ngoài vòng lặp top-level"); break; }
                if (d < 1) { jc_fail(j, "stack rỗng"); break; }
                d--;
                /* record: chỉ khi global đã giữ đúng record đó (vd biến tạm của inliner
                   gán lại receiver mỗi vòng) -> guard cùng con trỏ, không cần refcount */
                if (is_own_rec_t(st[d])) {
                    /* record mới ghi thẳng global (write-through, move ref): pseudo giữ
                       dạng borrowed để lần đọc sau chỉ mượn xem */
                    lt[pi] = bor_of_shape(rec_shape(st[d]));
                    break;
                }
                if (st[d] == JT_SN) {
                    /* chuỗi mới: dec-old + move 32B, pseudo giữ JT_S mượn */
                    lt[pi] = JT_S;
                    break;
                }
                if (st[d] == JT_DN) {
                    /* dict mới: dec-old + move tag+ptr, pseudo giữ JT_D mượn */
                    lt[pi] = JT_D;
                    break;
                }
                if (st[d] == JT_D) {
                    /* dict mượn: retain + move tag+ptr (global sở hữu ref mới) */
                    lt[pi] = JT_D;
                    break;
                }
                if (!(st[d] == JT_I || st[d] == JT_F || st[d] == JT_B || is_rec_t(st[d]))) {
                    jc_fail(j, "gán global kiểu ngoài phạm vi"); break;
                }
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
                if (g->t == T_DICT && g->u.dict->kind == DK_TYPE) {
                    st[d] = JT_TYP; sv[d] = (int32_t)m->k; d++; break;
                }
                if (g->t == T_DICT && g->u.dict->kind == DK_DICT) {
                    st[d] = JT_D; sv[d] = 0; d++; break;
                }
                if (g->t == T_STR) { st[d] = JT_S; sv[d] = 0; d++; break; }
                jc_fail(j, "global không phải số/hàm/list");
                break;
            }
            case OP_IDX: {
                if (d >= 2 && is_dict_t(st[d - 2]) &&
                    (st[d - 1] == JT_I || is_str_t(st[d - 1]))) {
                    /* đọc dict đoán int (thiếu key/khác int -> deopt, interpreter xử lý) */
                    d--; st[d - 1] = JT_I; sv[d - 1] = 0;
                    break;
                }
                if (d >= 2 && is_str_t(st[d - 2]) && st[d - 1] == JT_I) {
                    /* s[i]: chuỗi 1 ký tự từ str1_cache (JT_S mượn; ngoài biên -> deopt) */
                    d--; st[d - 1] = JT_S; sv[d - 1] = 0;
                    break;
                }
                if (d < 2 || list_elem(st[d - 2]) == JT_UNK || st[d - 1] != JT_I) {
                    jc_fail(j, "index ngoài phạm vi JIT"); break;
                }
                uint8_t e = list_elem(st[d - 2]);
                d--; st[d - 1] = e; sv[d - 1] = 0;
                break;
            }
            case OP_SETIDX: {
                if (d >= 3 && is_dict_t(st[d - 3]) &&
                    (st[d - 2] == JT_I || st[d - 2] == JT_S || st[d - 2] == JT_SN) &&
                    (st[d - 1] == JT_I || st[d - 1] == JT_F || st[d - 1] == JT_B ||
                     st[d - 1] == JT_S || st[d - 1] == JT_SN || st[d - 1] == JT_D ||
                     st[d - 1] == JT_DN || is_any_rec_t(st[d - 1]))) {
                    /* gán dict (OSR only): helper giữ key + val, codegen dec temp owned */
                    if (!j->osr) { jc_fail(j, "ghi dict (side effect) chỉ JIT trong vùng OSR"); break; }
                    d -= 3;
                    break;
                }
                if (d < 3 || list_elem(st[d - 3]) == JT_UNK || st[d - 2] != JT_I ||
                    st[d - 1] != list_elem(st[d - 3])) {
                    jc_fail(j, "gán phần tử ngoài phạm vi JIT"); break;
                }
                if (!j->osr) { jc_fail(j, "ghi list (side effect) chỉ JIT trong vùng OSR"); break; }
                d -= 3;
                break;
            }
            case OP_ATTR: {
                const uint8_t *c = j->prog->code + m->pc;
                JShape *sh;
                int fi;
                if (d < 1 || !is_any_rec_t(st[d - 1])) { jc_fail(j, "thuộc tính ngoài phạm vi JIT"); break; }
                sh = shape_of_any(st[d - 1]);
                if (!sh) { jc_fail(j, "shape lạ"); break; }
                fi = shape_field(sh, c + 5, (uint32_t)rd32(c + 1));
                if (fi < 0 || sh->fty[fi] == JT_UNK) { jc_fail(j, "field không phải số"); break; }
                st[d - 1] = sh->fty[fi]; sv[d - 1] = 0;
                break;
            }
            case OP_SETATTR: {
                /* gán field số lên record: OSR, hoặc G8 trong entry khi receiver là
                   record mới tạo trong cùng hàm / không còn lệnh nào sau đó có thể abort */
                const uint8_t *c = j->prog->code + m->pc;
                JShape *sh;
                int fi;
                if (d < 2 || !is_any_rec_t(st[d - 2])) { jc_fail(j, "gán field ngoài phạm vi JIT"); break; }
                if (!j->osr && !(is_own_rec_t(st[d - 2]) || sv[d - 2] == 1 || no_abort_after(j, i))) {
                    jc_fail(j, "gán field (side effect) chỉ JIT trong vùng OSR"); break;
                }
                sh = shape_of_any(st[d - 2]);
                if (!sh) { jc_fail(j, "shape lạ"); break; }
                fi = shape_field(sh, c + 5, (uint32_t)rd32(c + 1));
                if (fi < 0) { jc_fail(j, "field lạ"); break; }
                if (!(st[d - 1] == JT_I || st[d - 1] == JT_F || st[d - 1] == JT_B) ||
                    st[d - 1] != sh->fty[fi]) {
                    jc_fail(j, "gán field đổi kiểu"); break;
                }
                d -= 2;
                break;
            }
            case OP_CALLM: {
                int k = (int)m->k;
                const uint8_t *c = j->prog->code + m->pc;
                uint8_t at[256], r = JT_UNK;
                JShape *sh;
                TFn *F;
                uint32_t mlen = (uint32_t)rd32(c + 2);
                const uint8_t *mnm = c + 6;
                if (d >= k + 1 && is_dict_t(st[d - k - 1])) {
                    if (k == 2 && mlen == 3 && memcmp(mnm, "get", 3) == 0 &&
                        (st[d - 2] == JT_I || is_str_t(st[d - 2])) &&
                        st[d - 1] == JT_I) {
                        /* d.get(k, dflt-int) đoán int */
                        d -= k;
                        st[d - 1] = JT_I; sv[d - 1] = 0;
                        break;
                    }
                    if (k == 1 && mlen == 3 && memcmp(mnm, "has", 3) == 0 &&
                        (st[d - 1] == JT_I || is_str_t(st[d - 1]))) {
                        d -= k;
                        st[d - 1] = JT_B; sv[d - 1] = 0;
                        break;
                    }
                    if (k == 1 && mlen == 6 && memcmp(mnm, "remove", 6) == 0 &&
                        (st[d - 1] == JT_I || is_str_t(st[d - 1]))) {
                        if (!j->osr) { jc_fail(j, "remove dict chỉ JIT trong vùng OSR"); break; }
                        d -= k;
                        st[d - 1] = JT_NIL; sv[d - 1] = 0;
                        break;
                    }
                }
                if (d < k + 1 || !is_any_rec_t(st[d - k - 1])) { jc_fail(j, "method ngoài phạm vi JIT"); break; }
                sh = shape_of_any(st[d - k - 1]);
                if (!sh) { jc_fail(j, "shape lạ"); break; }
                F = shape_method(sh, mnm, mlen);
                int has_sn_arg = 0;
                for (int q = 0; q <= k; q++) {
                    at[q] = st[d - k - 1 + q];
                    if (at[q] == JT_SN) has_sn_arg = 1;
                    if (!arg_ok_t(at[q])) { jc_fail(j, "đối số kiểu lạ"); break; }
                }
                if (j->fail) break;
                if (!F || !callee_info(j, F, at, k + 1, &r) || r == JT_UNK) { jc_fail(j, "method không JIT được"); break; }
                if (has_sn_arg && r == JT_S) { jc_fail(j, "method trả chuỗi mượn từ đối số mới"); break; }
                d -= k;
                st[d - 1] = r; sv[d - 1] = 0;
                break;
            }
            case OP_LEN:
                if (d < 1 || (!is_list_t(st[d - 1]) && st[d - 1] != JT_S && st[d - 1] != JT_SN &&
                              st[d - 1] != JT_D && st[d - 1] != JT_DN)) {
                    jc_fail(j, "len ngoài phạm vi JIT"); break;
                }
                st[d - 1] = JT_I; sv[d - 1] = 0;
                break;
            case OP_NEW_DICT: {
                int k = (int)m->k;
                if (d < 2 * k) { jc_fail(j, "stack rỗng"); break; }
                if (k > 0 && !j->osr && !no_abort_after(j, i)) {
                    /* dict literal có phần tử trong entry: chỉ khi không thể abort rò temp */
                }
                for (int q = 0; q < k; q++) {
                    uint8_t kt = st[d - 2 * k + 2 * q];
                    uint8_t vt = st[d - 2 * k + 2 * q + 1];
                    if (!(kt == JT_I || is_str_t(kt))) { jc_fail(j, "key dict literal ngoài phạm vi"); break; }
                    if (!(vt == JT_I || vt == JT_F || vt == JT_B || is_str_t(vt) ||
                          is_dict_t(vt) || is_any_rec_t(vt))) {
                        jc_fail(j, "val dict literal ngoài phạm vi"); break;
                    }
                }
                if (j->fail) break;
                d -= 2 * k;
                st[d] = JT_DN; sv[d] = 0; d++;
                break;
            }
            case OP_ADD: case OP_SUB: case OP_MUL: case OP_DIV: case OP_MOD: {
                if (m->op == OP_ADD && d >= 2 && is_str_t(st[d - 2]) && is_str_t(st[d - 1])) {
                    /* nối chuỗi -> owned (helper cấp mới; codegen dec toán hạng owned) */
                    d--;
                    st[d - 1] = JT_SN; sv[d - 1] = 0;
                    break;
                }
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
            case OP_BNOT:
                if (d < 1 || st[d - 1] != JT_I) { jc_fail(j, "bnot kiểu lạ"); break; }
                sv[d - 1] = 0;
                break;
            case OP_BAND: case OP_BOR: case OP_BXOR: case OP_SHL: case OP_SHR:
                if (d < 2 || st[d - 2] != JT_I || st[d - 1] != JT_I) { jc_fail(j, "bitop kiểu lạ"); break; }
                d--;
                st[d - 1] = JT_I;
                sv[d - 1] = 0;
                break;
            case OP_NOT:
                if (d < 1 || !(is_num(st[d - 1]) || st[d - 1] == JT_B)) { jc_fail(j, "not kiểu lạ"); break; }
                st[d - 1] = JT_B; break;
            case OP_EQ: case OP_NE: case OP_LT: case OP_GT: case OP_LE: case OP_GE: {
                if (d < 2) { jc_fail(j, "stack rỗng"); break; }
                uint8_t a = st[d - 2], b = st[d - 1];
                int ok = (is_num(a) && is_num(b)) ||
                         (is_str_t(a) && is_str_t(b) &&
                          (m->op == OP_EQ || m->op == OP_NE)) ||
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
                if (is_own_rec_t(st[d - 1]) || st[d - 1] == JT_SN || st[d - 1] == JT_DN) {
                    jc_fail(j, "dup giá trị mới"); break;
                }
                st[d] = st[d - 1]; sv[d] = sv[d - 1]; d++; break;
            case OP_NOP: break;
            case OP_JMP: br = m->target; fall = 0; break;   /* target < 0: lối ra (main) */
            case OP_JMPF: case OP_JMPT:
                if (d < 1 || !(is_num(st[d - 1]) || st[d - 1] == JT_B || st[d - 1] == JT_S ||
                               st[d - 1] == JT_SN || is_dict_t(st[d - 1]))) {
                    jc_fail(j, "điều kiện kiểu lạ"); break;
                }
                d--; br = m->target; break;
            case OP_RET: {
                uint8_t rt;
                if (d < 1) { jc_fail(j, "stack rỗng"); break; }
                rt = st[d - 1];
                if (!(rt == JT_I || rt == JT_F || rt == JT_B || rt == JT_NIL || is_str_t(rt) ||
                      is_dict_t(rt) ||
                      is_own_rec_t(rt) || is_rec_t(rt))) {
                    jc_fail(j, "trả kiểu ngoài phạm vi"); break;
                }
                if (retty == JT_UNK) retty = rt;
                else if (retty != rt) { jc_fail(j, "return nhiều kiểu"); break; }
                fall = 0;
                break;
            }
            case OP_CALL: {
                int k = (int)m->k;
                if (d < k + 1) { jc_fail(j, "stack rỗng"); break; }
                uint8_t ft = st[d - k - 1];
                if (ft == JT_NAT) {
                    int nat = sv[d - k - 1];
                    uint8_t r;
                    if (nat == NAT_LEN) {
                        if (k != 1 || (!is_list_t(st[d - 1]) && st[d - 1] != JT_S &&
                                       st[d - 1] != JT_SN && st[d - 1] != JT_D &&
                                       st[d - 1] != JT_DN)) {
                            jc_fail(j, "len() ngoài phạm vi JIT"); break;
                        }
                        r = JT_I;
                    } else if (nat == NAT_STR) {
                        /* str(x) số/chuỗi -> chuỗi mới (helper dùng fmt_str, không lỗi) */
                        if (k != 1 || !(is_num(st[d - 1]) || st[d - 1] == JT_B ||
                                        st[d - 1] == JT_S || st[d - 1] == JT_SN)) {
                            jc_fail(j, "str() ngoài phạm vi JIT"); break;
                        }
                        r = JT_SN;
                    } else if (nat == NAT___NEW) {
                        /* __new(Kiểu, v1..vn): tạo record field số */
                        uint8_t at2[JIT_SHAPE_MAXF];
                        TDict *tp;
                        int nf, sh;
                        if (k < 1 || st[d - k] != JT_TYP) { jc_fail(j, "__new cần kiểu struct"); break; }
                        tp = G[sv[d - k]].u.dict;
                        if (tp->kind != DK_TYPE || tp->mod) { jc_fail(j, "__new cần kiểu struct"); break; }
                        nf = (int)tp->fields->len;
                        if (k - 1 != nf || nf > JIT_SHAPE_MAXF) { jc_fail(j, "__new arity"); break; }
                        for (int q = 0; q < nf; q++) {
                            at2[q] = st[d - k + 1 + q];
                            if (at2[q] != JT_I && at2[q] != JT_F && at2[q] != JT_B) {
                                jc_fail(j, "field __new không phải số"); break;
                            }
                        }
                        if (j->fail) break;
                        sh = shape_reg(tp, nf, at2);
                        if (sh < 0) { jc_fail(j, "quá nhiều shape"); break; }
                        r = own_of_shape(sh);
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
                    int has_sn_arg = 0;
                    for (int q = 0; q < k; q++) {
                        at[q] = st[d - k + q];
                        if (at[q] == JT_SN) has_sn_arg = 1;
                        if (!arg_ok_t(at[q])) {
                            jc_fail(j, "đối số kiểu lạ"); break;
                        }
                    }
                    if (j->fail) break;
                    if (g->t != T_FN || !callee_info(j, g->u.fn, at, k, &r) || r == JT_UNK) {
                        jc_fail(j, "callee không JIT được"); break;
                    }
                    if (has_sn_arg && r == JT_S) {
                        jc_fail(j, "callee trả chuỗi mượn từ đối số mới"); break;
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
#define OWN(jc, i) (32 + 8 * ((int32_t)(jc)->nl + (int32_t)(jc)->maxd_seen + 2 + (int32_t)(i)))

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
static void mov_rcx_imm32(JC *j, int32_t v) { eb3(j, 0x48, 0xC7, 0xC1); e32(j, v); }  /* mov rcx,imm32 */
static void mov_r8_rax(JC *j) { eb3(j, 0x49, 0x89, 0xC0); }   /* mov r8,rax */
static void mov_r8_imm32(JC *j, int32_t v) { eb3(j, 0x49, 0xC7, 0xC0); e32(j, v); }   /* mov r8,imm32 */
static void mov_r9_imm32(JC *j, int32_t v) { eb3(j, 0x49, 0xC7, 0xC1); e32(j, v); }   /* mov r9,imm32 */
static void lea_rcx_rsp(JC *j, int32_t d) { eb3(j, 0x48, 0x8D, 0x8C); eb(j, 0x24); e32(j, d); }
static void lea_r9_rsp(JC *j, int32_t d) { eb3(j, 0x4C, 0x8D, 0x8C); eb(j, 0x24); e32(j, d); }
/* chuyển đối số cho helper C (SYSV: rdi rsi rdx / Win64: rcx rdx r8) */
static void mov_rdi_rax(JC *j) { eb3(j, 0x48, 0x89, 0xC7); }
static void mov_rsi_rax(JC *j) { eb3(j, 0x48, 0x89, 0xC6); }
static void mov_rdx_rax(JC *j) { eb3(j, 0x48, 0x89, 0xC2); }
static void mov_rcx_rax(JC *j) { eb3(j, 0x48, 0x89, 0xC1); }
static void mov_rdi_rcx(JC *j) { eb3(j, 0x48, 0x89, 0xCF); }
static void mov_rsi_imm(JC *j, int32_t v) { eb3(j, 0x48, 0xC7, 0xC6); e32(j, v); }
static void mov_rdx_imm(JC *j, int32_t v) { eb3(j, 0x48, 0xC7, 0xC2); e32(j, v); }
static void lea_rdx_rsp(JC *j, int32_t d) { eb3(j, 0x48, 0x8D, 0x94); eb(j, 0x24); e32(j, d); }
static void lea_r8_rsp(JC *j, int32_t d) { eb3(j, 0x4C, 0x8D, 0x84); eb(j, 0x24); e32(j, d); }
static void call_rax(JC *j) { eb2(j, 0xFF, 0xD0); }
/* nhả temp sở hữu được đã tiêu thụ xong (SN -> dec_sval, DN -> dec_dict;
   mượn/RN thì thôi: mượn không giữ ref, record chờ sweep) */
static void dec_temp(JC *j, int32_t slot, uint8_t ty) {
    void *fn = ty == JT_SN ? (void*)&tjit_dec_sval : (void*)&tjit_dec_dict;
    if (ty != JT_SN && ty != JT_DN) return;
    ld_rax(j, slot);
    if (JIT_SYSV) mov_rdi_rax(j); else mov_rcx_rax(j);
    mov_rax_imm(j, (int64_t)(intptr_t)fn);
    call_rax(j);
}
/* STORE local với cờ runtime OWN(j, local_i) (G10):
   - OWN(j, local_i) == 1 khi local đang giữ giá trị native-owned (RN/SN/DN).
   - Khi ghi đè: nếu mới != cũ và OWN == 1 thì nhả giá trị cũ (SN -> dec_sval,
     DN/RN -> dec_dict), rồi cập nhật OWN = (mới là owned ? 1 : 0). */
static void emit_store_local(JC *j, int local_i, int32_t src_slot, uint8_t old_t, uint8_t new_t) {
    int old_may_own = (!j->mainmode) && (is_own_rec_t(old_t) || old_t == JT_SN || old_t == JT_DN);
    int new_is_own = (!j->mainmode) && (is_own_rec_t(new_t) || new_t == JT_SN || new_t == JT_DN);
    if (!old_may_own) {
        ld_rax(j, src_slot);
        st_rax(j, LOC(local_i));
        if (new_is_own) {
            eb(j, 0xB8); e32(j, 1);
            st_rax(j, OWN(j, local_i));
        }
        return;
    }
    {
        size_t j_same, j_not_own;
        ld_rcx(j, src_slot);                         /* rcx = mới */
        ld_rax(j, LOC(local_i));                     /* rax = cũ */
        eb3(j, 0x48, 0x39, 0xC8);                    /* cmp rax,rcx */
        eb2(j, 0x74, 0x00); j_same = j->n - 1;       /* je same (tự gán x = x) */
        ld_rcx(j, OWN(j, local_i));                  /* rcx = cờ OWN */
        eb3(j, 0x48, 0x85, 0xC9);                    /* test rcx,rcx */
        eb2(j, 0x74, 0x00); j_not_own = j->n - 1;    /* jz not_own */
        if (JIT_SYSV) mov_rdi_rax(j); else mov_rcx_rax(j);
        if (old_t == JT_SN) mov_rax_imm(j, (int64_t)(intptr_t)&tjit_dec_sval);
        else mov_rax_imm(j, (int64_t)(intptr_t)&tjit_dec_dict);
        call_rax(j);
        j->b[j_not_own] = (uint8_t)(j->n - (j_not_own + 1)); /* not_own: */
        ld_rax(j, src_slot);
        st_rax(j, LOC(local_i));
        eb(j, 0xB8); e32(j, new_is_own ? 1 : 0);
        st_rax(j, OWN(j, local_i));
        j->b[j_same] = (uint8_t)(j->n - (j_same + 1));       /* same: */
    }
}

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
    if (j->nl > 0) {
        eb2(j, 0x31, 0xC0);                  /* xor eax,eax */
        for (int i = 0; i < j->nl; i++) st_rax(j, OWN(j, i));
    }
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
           k == NAT_LEN || k == NAT___NEW || k == NAT_STR;
}

/* giữ ref callee (code này nhảy thẳng vào nó) */
static void keep_callee(JC *j, TFn *F) {
    if (F == j->fn) return;
    for (int q = 0; q < j->jf->ncallees; q++) if (j->jf->callees[q] == F) return;
    if (j->jf->ncallees >= j->jf->capcallees) {
        j->jf->capcallees = j->jf->capcallees ? j->jf->capcallees * 2 : 4;
        j->jf->callees = realloc(j->jf->callees, (size_t)j->jf->capcallees * sizeof(TFn *));
    }
    j->jf->callees[j->jf->ncallees++] = F;
    F->refc++;
}

static int gen_body(JC *j) {
    TVal *G = j->prog->globals;
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
        case OP_PUSH_NIL:
            eb2(j, 0x31, 0xC0); st_rax(j, SLOT(j, d)); break;
        case OP_PUSH_STR: {
            /* TVal* mượn hằng cache (sống suốt native, không cần inc) */
            TVal *keep = (TVal *)j->prog->ic[m->pc].key;
            mov_rax_imm(j, (int64_t)(intptr_t)keep);
            st_rax(j, SLOT(j, d));
            break;
        }
        case OP_LOAD:
            ld_rax(j, LOC(m->k)); st_rax(j, SLOT(j, d)); break;
        case OP_STORE: case OP_DECL: case OP_BIND: {
            emit_store_local(j, (int)m->k, SLOT(j, d - 1), s->lt[m->k], st[d - 1]);
            break;
        }
        case OP_STOREG: case OP_BINDG: {
            /* global giả (main): ghi thẳng TVal (tag + giá trị); ô cũ phải vô hướng */
            TVal *g = &G[m->k];
            uint8_t t = st[d - 1];
            int tag = t == JT_I ? T_INT : t == JT_F ? T_FLOAT : T_BOOL;
            mov_rax_imm(j, (int64_t)(intptr_t)g);
            if (is_own_rec_t(t)) {
                /* record mới move vào global (không nhả cũ: cũ orphan chờ GC) */
                eb2(j, 0xC7, 0x00); e32(j, (int32_t)T_DICT);
                eb2(j, 0x48, 0x8B); m_rsp(j, 1, SLOT(j, d - 1));
                eb(j, 0x48); eb3(j, 0x89, 0x48, 0x08);
                break;
            }
            if (is_rec_t(t)) {
                eb3(j, 0x83, 0x38, (uint8_t)T_DICT); jguard(j, 0x85);   /* global đang là dict */
                eb2(j, 0x48, 0x8B); m_rsp(j, 1, SLOT(j, d - 1));       /* mov rcx,[slot] */
                eb(j, 0x48); eb3(j, 0x39, 0x48, 0x08); jguard(j, 0x85);  /* cmp [rax+8],rcx: cùng record */
                break;
            }
            if (t == JT_SN || t == JT_DN || t == JT_D) {
                /* chuỗi/dict ghi thẳng global: helper(g, bits, ty) nhả cũ + move/retain */
                int ty = t == JT_SN ? 0 : t == JT_DN ? 1 : 2;
                ld_rax(j, SLOT(j, d - 1));                   /* rax = bits */
                if (JIT_SYSV) mov_rsi_rax(j); else mov_rdx_rax(j);  /* arg2 = bits */
                mov_rax_imm(j, (int64_t)(intptr_t)g);        /* rax = g */
                if (JIT_SYSV) mov_rdi_rax(j); else mov_rcx_rax(j);  /* arg1 = g */
                if (JIT_SYSV) mov_rdx_imm(j, ty); else mov_r8_imm32(j, ty);
                mov_rax_imm(j, (int64_t)(intptr_t)&tjit_store_go);
                call_rax(j);
                break;
            }
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
                int tag = t == JT_I ? T_INT : t == JT_F ? T_FLOAT : t == JT_B ? T_BOOL :
                          (is_rec_t(t) || t == JT_D) ? T_DICT : t == JT_S ? T_STR : T_LIST;
                mov_rax_imm(j, (int64_t)(intptr_t)g);
                eb3(j, 0x83, 0x38, (uint8_t)tag); jguard(j, 0x85);
                if (t == JT_S) {
                    /* chuỗi: slot giữ TVal* (địa chỉ global, giữ ref) */
                    mov_rax_imm(j, (int64_t)(intptr_t)g);
                } else {
                    eb(j, 0x48); eb3(j, 0x8B, 0x40, 0x08);
                }
                if (t == JT_B) eb3(j, 0x83, 0xE0, 0x01);
                st_rax(j, SLOT(j, d));
                break;
            }
            if (is_nat_ok(m->k) || g->t == T_FN) break;
            if (g->t == T_DICT && g->u.dict->kind == DK_TYPE) break;  /* JT_TYP: deopt dựng lại từ sv */
            if (g->t == T_STR) {
                /* TVal* mượn global (global giữ ref suốt native) */
                mov_rax_imm(j, (int64_t)(intptr_t)g);
                eb3(j, 0x83, 0x38, (uint8_t)T_STR); jguard(j, 0x85);
                mov_rax_imm(j, (int64_t)(intptr_t)g);
                st_rax(j, SLOT(j, d));
                break;
            }
            if (g->t == T_DICT && g->u.dict->kind == DK_DICT) {
                /* TDict* mượn global; kind không đổi suốt native nên guard 1 lần */
                mov_rax_imm(j, (int64_t)(intptr_t)g);
                eb3(j, 0x83, 0x38, (uint8_t)T_DICT); jguard(j, 0x85);
                eb2(j, 0x83, 0xB8); e32(j, (int32_t)offsetof(TDict, kind));
                eb(j, (uint8_t)DK_DICT); jguard(j, 0x85);   /* cmp dword [rax+kind],DK_DICT */
                eb(j, 0x48); eb3(j, 0x8B, 0x40, 0x08);  /* mov rax,[rax+8] */
                st_rax(j, SLOT(j, d));
                break;
            }
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
            if (m->op == OP_ADD && (a == JT_S || a == JT_SN) && (b == JT_S || b == JT_SN)) {
                /* nối chuỗi: helper cấp wrapper mới, tự nhả toán hạng owned (m) */
                int ms = (a == JT_SN ? 1 : 0) | (b == JT_SN ? 2 : 0);
                ld_rax(j, A);
                if (JIT_SYSV) mov_rdi_rax(j); else mov_rcx_rax(j);
                ld_rax(j, B);
                if (JIT_SYSV) mov_rsi_rax(j); else mov_rdx_rax(j);
                if (JIT_SYSV) mov_rdx_imm(j, ms); else mov_r8_imm32(j, ms);
                mov_rax_imm(j, (int64_t)(intptr_t)&tjit_str_cat);
                call_rax(j);
                st_rax(j, A);
                break;
            }
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
        case OP_BNOT: {
            int32_t A = SLOT(j, d - 1);
            ld_rax(j, A);
            eb3(j, 0x48, 0xF7, 0xD0);                                          /* not rax */
            st_rax(j, A);
            break;
        }
        case OP_BAND: case OP_BOR: case OP_BXOR: {
            int32_t A = SLOT(j, d - 2), B = SLOT(j, d - 1);
            ld_rax(j, A);
            uint8_t o = m->op == OP_BAND ? 0x23 : m->op == OP_BOR ? 0x0B : 0x33;
            eb2(j, 0x48, o); m_rsp(j, 0, B);                                   /* and/or/xor rax,[rsp+B] */
            st_rax(j, A);
            break;
        }
        case OP_SHL: case OP_SHR: {
            int32_t A = SLOT(j, d - 2), B = SLOT(j, d - 1);
            ld_rax(j, A);
            ld_rcx(j, B);
            eb(j, 0x48); eb3(j, 0x83, 0xF9, 0x3F);                             /* cmp rcx,63 */
            jguard(j, 0x87);                                                   /* ja guard */
            eb3(j, 0x48, 0xD3, m->op == OP_SHL ? 0xE0 : 0xF8);                 /* shl/sar rax,cl */
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
            if ((a == JT_S || a == JT_SN) && (b == JT_S || b == JT_SN)) {
                /* so chuỗi qua helper; ms nhả wrapper owned đã tiêu thụ (mượn thì thôi) */
                int ms = (a == JT_SN ? 1 : 0) | (b == JT_SN ? 2 : 0);
                ld_rax(j, A);
                if (JIT_SYSV) mov_rdi_rax(j); else mov_rcx_rax(j);
                ld_rax(j, B);
                if (JIT_SYSV) mov_rsi_rax(j); else mov_rdx_rax(j);
                if (JIT_SYSV) mov_rdx_imm(j, ms); else mov_r8_imm32(j, ms);
                mov_rax_imm(j, (int64_t)(intptr_t)&tjit_str_eq);
                call_rax(j);
                if (m->op == OP_NE) { eb(j, 0x48); eb3(j, 0x83, 0xF0, 0x01); }   /* xor rax,1 */
                st_rax(j, A);
                break;
            }
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
        case OP_POP: {
            /* giá trị mới bị bỏ (vd gọi constructor làm câu lệnh): nhả ref */
            if (is_own_rec_t(st[d - 1]) || st[d - 1] == JT_DN) {
                ld_rax(j, SLOT(j, d - 1));
                if (JIT_SYSV) mov_rdi_rax(j); else mov_rcx_rax(j);
                mov_rax_imm(j, (int64_t)(intptr_t)&tjit_dec_dict);
                call_rax(j);
            } else if (st[d - 1] == JT_SN) {
                dec_temp(j, SLOT(j, d - 1), JT_SN);
            }
            break;
        }
        case OP_NOP: break;
        case OP_DUP:
            ld_rax(j, SLOT(j, d - 1)); st_rax(j, SLOT(j, d)); break;
        case OP_JMP: {
            uint8_t pre[1] = {0xE9};
            jto(j, pre, 1, m, d);
            break;
        }
        case OP_JMPF: case OP_JMPT: {
            int32_t A = SLOT(j, d - 1);
            if (st[d - 1] == JT_S || st[d - 1] == JT_SN) {
                /* chuỗi rỗng là falsy (đọc u.str.n ở offset 16 của TVal) */
                ld_rax(j, A);
                eb(j, 0x48); eb3(j, 0x83, 0x78, 0x10); eb(j, 0x00);   /* cmp qword [rax+16],0 */
                if (st[d - 1] == JT_SN) {
                    /* owned: tính bool trước rồi mới nhả (cả 2 đường đều chết giá trị) */
                    eb3(j, 0x0F, 0x95, 0xC0);                   /* setne al */
                    eb3(j, 0x0F, 0xB6, 0xC0);                   /* movzx eax,al */
                    dec_temp(j, A, JT_SN);
                    eb2(j, 0x85, 0xC0);                         /* test eax,eax */
                }
                uint8_t pre[2] = {0x0F, (uint8_t)(m->op == OP_JMPF ? 0x84 : 0x85)};
                jto(j, pre, 2, m, d - 1);
            } else if (is_dict_t(st[d - 1])) {
                /* dict rỗng là falsy (đọc len ở offsetof(TDict, len)) */
                ld_rax(j, A);
                eb(j, 0x48); eb2(j, 0x83, 0xB8); e32(j, (int32_t)offsetof(TDict, len)); eb(j, 0x00);
                if (st[d - 1] == JT_DN) {
                    eb3(j, 0x0F, 0x95, 0xC0);
                    eb3(j, 0x0F, 0xB6, 0xC0);
                    dec_temp(j, A, JT_DN);
                    eb2(j, 0x85, 0xC0);
                }
                uint8_t pre[2] = {0x0F, (uint8_t)(m->op == OP_JMPF ? 0x84 : 0x85)};
                jto(j, pre, 2, m, d - 1);
            } else if (st[d - 1] == JT_F) {
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
            uint8_t ct = st[d - (set ? 3 : 2)];
            if (is_dict_t(ct)) {
                /* dict đoán int: 1 lần tìm; thiếu/khác int -> deopt */
                uint8_t kt = st[d - (set ? 2 : 1)];
                int kstr = is_str_t(kt) ? 1 : 0;
                if (set && !j->osr) { jc_fail(j, "ghi dict chỉ JIT trong vùng OSR"); return 0; }
                if (set) {
                    /* SETIDX: luôn ghi (tạo mới nếu thiếu) qua set helper */
                    uint8_t vt = st[d - 1];
                    int vty = vt == JT_F ? 1 : vt == JT_B ? 2 : vt == JT_S ? 3 :
                              vt == JT_SN ? 4 : vt == JT_D ? 5 : vt == JT_DN ? 6 :
                              is_rec_t(vt) ? 7 : is_own_rec_t(vt) ? 8 : 0;
                    int kv = (kstr ? 1 : 0) | (vty << 1) | 32;   /* bit5: hint index (counter fusion) */
                    ld_rax(j, L);
                    if (JIT_SYSV) mov_rdi_rax(j); else mov_rcx_rax(j);
                    ld_rax(j, X);
                    if (JIT_SYSV) mov_rsi_rax(j); else mov_rdx_rax(j);
                    ld_rax(j, V);
                    if (JIT_SYSV) mov_rdx_rax(j); else mov_r8_rax(j);
                    if (JIT_SYSV) mov_rcx_imm32(j, kv); else mov_r9_imm32(j, kv);
                    mov_rax_imm(j, (int64_t)(intptr_t)&tjit_dict_set);
                    call_rax(j);
                    dec_temp(j, X, kt); dec_temp(j, V, vt); dec_temp(j, L, ct);
                    break;
                }
                /* IDX: getD(d, key, &L): 0 = int thấy, khác 0 -> deopt
                   (thiếu key: interpreter báo lỗi; khác int: interpreter trả) */
                ld_rax(j, L);
                if (JIT_SYSV) mov_rdi_rax(j); else mov_rcx_rax(j);
                ld_rax(j, X);
                if (JIT_SYSV) mov_rsi_rax(j); else mov_rdx_rax(j);
                if (JIT_SYSV) mov_rdx_imm(j, kstr); else mov_r8_imm32(j, kstr);
                if (JIT_SYSV) lea_rcx_rsp(j, L); else lea_r9_rsp(j, L);
                mov_rax_imm(j, (int64_t)(intptr_t)&tjit_dict_getD);
                call_rax(j);
                eb3(j, 0x48, 0x85, 0xC0);                             /* test rax,rax */
                jguard(j, 0x85);                                      /* jnz deopt */
                dec_temp(j, X, kt);                                   /* keytemp chết */
                break;
            }
            if (is_str_t(ct)) {
                /* s[i]: helper trả &str1_cache[ch] (JT_S mượn), NULL nếu ngoài biên */
                ld_rax(j, L);
                if (JIT_SYSV) mov_rdi_rax(j); else mov_rcx_rax(j);
                ld_rax(j, X);
                if (JIT_SYSV) mov_rsi_rax(j); else mov_rdx_rax(j);
                mov_rax_imm(j, (int64_t)(intptr_t)&tjit_str_idx);
                call_rax(j);
                eb3(j, 0x48, 0x85, 0xC0);                             /* test rax,rax */
                jguard(j, 0x84);                                      /* jz deopt */
                st_rax(j, X);                                         /* giữ tạm ở X */
                dec_temp(j, L, ct);                                   /* nhả s nếu JT_SN */
                ld_rax(j, X);
                st_rax(j, L);
                break;
            }
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
        case OP_ATTR: {
            const uint8_t *c = j->prog->code + m->pc;
            JShape *sh = shape_of_any(st[d - 1]);
            int fi;
            uint8_t ft;
            int ttag;
            if (!sh) { jc_fail(j, "shape lạ"); return 0; }
            fi = shape_field(sh, c + 5, (uint32_t)rd32(c + 1));
            ft = sh->fty[fi];
            ttag = ft == JT_I ? T_INT : ft == JT_F ? T_FLOAT : T_BOOL;
            ld_rax(j, SLOT(j, d - 1));
            eb(j, 0x48); eb2(j, 0x8B, 0x80); e32(j, (int32_t)offsetof(TDict, vals));   /* mov rax,[rax+vals] */
            eb2(j, 0x48, 0x05); e32(j, 32 * fi);                                      /* add rax,32*fi */
            eb3(j, 0x83, 0x38, (uint8_t)ttag); jguard(j, 0x85);                       /* field đúng kiểu */
            eb(j, 0x48); eb3(j, 0x8B, 0x40, 0x08);                                    /* mov rax,[rax+8] */
            if (ft == JT_B) eb3(j, 0x83, 0xE0, 0x01);
            st_rax(j, SLOT(j, d - 1));
            break;
        }
        case OP_SETATTR: {
            /* gán field số: guard field cũ cùng tag (không refcount), ghi tag + bit mới */
            const uint8_t *c = j->prog->code + m->pc;
            JShape *sh = shape_of_any(st[d - 2]);
            int fi;
            uint8_t vt;
            int32_t L, V;
            int vtag;
            if (!sh) { jc_fail(j, "shape lạ"); return 0; }
            fi = shape_field(sh, c + 5, (uint32_t)rd32(c + 1));
            vt = st[d - 1];
            L = SLOT(j, d - 2); V = SLOT(j, d - 1);
            vtag = vt == JT_I ? T_INT : vt == JT_F ? T_FLOAT : T_BOOL;
            ld_rax(j, L);
            eb(j, 0x48); eb2(j, 0x8B, 0x80); e32(j, (int32_t)offsetof(TDict, vals));
            if (!j->osr && !is_own_rec_t(st[d - 2]) && s->sv[d - 2] != 1) {
                /* G8: trước khi ghi vào tham số record trong entry, guard tất cả field
                   của record đó để các ATTR/SETATTR kế tiếp không bao giờ abort nửa chừng */
                for (int qf = 0; qf < sh->nf; qf++) {
                    int qtag = sh->fty[qf] == JT_I ? T_INT : sh->fty[qf] == JT_F ? T_FLOAT : T_BOOL;
                    eb2(j, 0x83, 0xB8); e32(j, 32 * qf); eb(j, (uint8_t)qtag);
                    jguard(j, 0x85);
                }
            }
            eb2(j, 0x48, 0x05); e32(j, 32 * fi);
            eb3(j, 0x83, 0x38, (uint8_t)vtag); jguard(j, 0x85);   /* field cũ phải cùng kiểu số */
            eb2(j, 0xC7, 0x00); e32(j, vtag);
            eb2(j, 0x48, 0x8B); m_rsp(j, 1, V);
            eb(j, 0x48); eb3(j, 0x89, 0x48, 0x08);
            break;
        }
        case OP_CALLM: {
            /* method tĩnh: kiểu struct của receiver đã cố định (guard lúc vào / shape) */
            int k = (int)m->k;
            int fd = d - k - 1;
            const uint8_t *c = j->prog->code + m->pc;
            uint8_t nat2[256];
            if (k == 2 && is_dict_t(st[fd]) &&
                rd32(c + 2) == 3 && memcmp(c + 6, "get", 3) == 0) {
                /* d.get(k, dflt-int) đoán int: getD 1 lần tìm; 0 = int thấy,
                   1 = deopt (khác int), 2 = thiếu key (ghi default, không deopt) */
                uint8_t kt = st[fd + 1];
                int kstr = is_str_t(kt) ? 1 : 0;
                int32_t R = SLOT(j, fd), K = SLOT(j, fd + 1), D = SLOT(j, fd + 2);
                size_t j1;
                ld_rax(j, R);
                if (JIT_SYSV) mov_rdi_rax(j); else mov_rcx_rax(j);
                ld_rax(j, K);
                if (JIT_SYSV) mov_rsi_rax(j); else mov_rdx_rax(j);
                if (JIT_SYSV) mov_rdx_imm(j, kstr); else mov_r8_imm32(j, kstr);
                if (JIT_SYSV) lea_rcx_rsp(j, R); else lea_r9_rsp(j, R);
                mov_rax_imm(j, (int64_t)(intptr_t)&tjit_dict_getD);
                call_rax(j);
                eb3(j, 0x83, 0xF8, 0x01);                             /* cmp eax,1 */
                jguard(j, 0x84);                                      /* je deopt (khác int) */
                eb3(j, 0x83, 0xF8, 0x02);                             /* cmp eax,2 */
                eb2(j, 0x75, 0x00); j1 = j->n - 1;                     /* jne DONE */
                ld_rax(j, D);                                         /* thiếu: dùng default */
                st_rax(j, R);
                j->b[j1] = (uint8_t)(j->n - (j1 + 1));                 /* DONE: */
                dec_temp(j, K, kt);                                   /* keytemp chết */
                break;
            }
            if (k == 1 && is_dict_t(st[fd])) {
                uint32_t mlen = (uint32_t)rd32(c + 2);
                const uint8_t *mnm = c + 6;
                uint8_t kt = st[fd + 1];
                int kstr = is_str_t(kt) ? 1 : 0;
                int32_t R = SLOT(j, fd), K = SLOT(j, fd + 1);
                if (mlen == 3 && memcmp(mnm, "has", 3) == 0) {
                    ld_rax(j, R);
                    if (JIT_SYSV) mov_rdi_rax(j); else mov_rcx_rax(j);
                    ld_rax(j, K);
                    if (JIT_SYSV) mov_rsi_rax(j); else mov_rdx_rax(j);
                    if (JIT_SYSV) mov_rdx_imm(j, kstr); else mov_r8_imm32(j, kstr);
                    mov_rax_imm(j, (int64_t)(intptr_t)&tjit_dict_has);
                    call_rax(j);
                    st_rax(j, SLOT(j, d));                            /* giữ tạm kết quả */
                    dec_temp(j, K, kt);
                    dec_temp(j, R, st[fd]);
                    ld_rax(j, SLOT(j, d));
                    st_rax(j, R);
                    break;
                }
                if (mlen == 6 && memcmp(mnm, "remove", 6) == 0) {
                    ld_rax(j, R);
                    if (JIT_SYSV) mov_rdi_rax(j); else mov_rcx_rax(j);
                    ld_rax(j, K);
                    if (JIT_SYSV) mov_rsi_rax(j); else mov_rdx_rax(j);
                    if (JIT_SYSV) mov_rdx_imm(j, kstr); else mov_r8_imm32(j, kstr);
                    mov_rax_imm(j, (int64_t)(intptr_t)&tjit_dict_remove);
                    call_rax(j);
                    dec_temp(j, K, kt);
                    dec_temp(j, R, st[fd]);
                    eb2(j, 0x31, 0xC0);
                    st_rax(j, R);
                    break;
                }
            }
            JShape *sh2 = shape_of_any(st[fd]);
            TFn *F;
            if (!sh2) { jc_fail(j, "shape lạ"); return 0; }
            F = shape_method(sh2, c + 6, (uint32_t)rd32(c + 2));
            for (int q = 0; q <= k; q++) nat2[q] = norm_ty(st[fd + q]);
            if (JIT_SYSV) { eb3(j, 0x48, 0x8D, 0xBC); eb(j, 0x24); e32(j, SLOT(j, fd)); eb3(j, 0x48, 0x89, 0xDE); }
            else { eb3(j, 0x48, 0x8D, 0x8C); eb(j, 0x24); e32(j, SLOT(j, fd)); eb3(j, 0x48, 0x89, 0xDA); }
            if (is_self_spec(j, F, nat2, k + 1)) {
                eb(j, 0xE8);
                e32(j, (int32_t)(0 - (int64_t)(j->n + 4)));
            } else {
                mov_rax_imm(j, (int64_t)(intptr_t)callee_entry(F, nat2, k + 1));
                eb2(j, 0xFF, 0xD0);
                keep_callee(j, F);
            }
            st_rax(j, SLOT(j, fd));
            if (j->osr) {
                eb3(j, 0x83, 0x3B, 0x00);       /* cmp dword [rbx],0 */
                jguard(j, 0x85);
                for (int q = 1; q <= k; q++)
                    if (st[fd + q] == JT_SN) dec_temp(j, SLOT(j, fd + q), JT_SN);
            } else {
                for (int q = 1; q <= k; q++)
                    if (st[fd + q] == JT_SN) dec_temp(j, SLOT(j, fd + q), JT_SN);
                ld_rax(j, SLOT(j, fd));
                eb3(j, 0x83, 0x3B, 0x00);       /* cmp dword [rbx],0 */
                jepi(j, 0x85);
            }
            break;
        }
        case OP_LEN:
            if (is_str_t(st[d - 1])) {
                /* len chuỗi = u.str.n (offset 16 trong TVal, trùng TList.len) */
                ld_rax(j, SLOT(j, d - 1));
                eb(j, 0x48); eb3(j, 0x8B, 0x40, 0x10);
                st_rax(j, SLOT(j, d - 1));
                break;
            }
            if (is_dict_t(st[d - 1])) {
                ld_rax(j, SLOT(j, d - 1));
                eb(j, 0x48); eb2(j, 0x8B, 0x80); e32(j, (int32_t)offsetof(TDict, len));
                st_rax(j, SLOT(j, d - 1));
                break;
            }
            ld_rax(j, SLOT(j, d - 1));
            eb(j, 0x48); eb2(j, 0x8B, 0x80); e32(j, (int32_t)offsetof(TList, len));
            st_rax(j, SLOT(j, d - 1));
            break;
        case OP_NEW_DICT: {
            int k = (int)m->k;
            int32_t D = SLOT(j, d);
            mov_rax_imm(j, (int64_t)(intptr_t)&tjit_dict_new0);
            call_rax(j);
            if (k == 0) {
                st_rax(j, D);
                break;
            }
            st_rax(j, D);
            for (int q = 0; q < k; q++) {
                int kd = d - 2 * k + 2 * q;
                uint8_t kt = st[kd], vt = st[kd + 1];
                int32_t K = SLOT(j, kd), V = SLOT(j, kd + 1);
                int kstr = is_str_t(kt) ? 1 : 0;
                int vty = vt == JT_F ? 1 : vt == JT_B ? 2 : vt == JT_S ? 3 :
                          vt == JT_SN ? 4 : vt == JT_D ? 5 : vt == JT_DN ? 6 :
                          is_rec_t(vt) ? 7 : is_own_rec_t(vt) ? 8 : 0;
                int kv = (kstr ? 1 : 0) | (vty << 1);
                ld_rax(j, D);
                if (JIT_SYSV) mov_rdi_rax(j); else mov_rcx_rax(j);
                ld_rax(j, K);
                if (JIT_SYSV) mov_rsi_rax(j); else mov_rdx_rax(j);
                ld_rax(j, V);
                if (JIT_SYSV) mov_rdx_rax(j); else mov_r8_rax(j);
                if (JIT_SYSV) mov_rcx_imm32(j, kv); else mov_r9_imm32(j, kv);
                mov_rax_imm(j, (int64_t)(intptr_t)&tjit_dict_set);
                call_rax(j);
                dec_temp(j, K, kt);
                dec_temp(j, V, vt);
            }
            ld_rax(j, D);
            st_rax(j, SLOT(j, d - 2 * k));
            break;
        }
        case OP_RET: {
            /* local chuỗi mới đã chết (không phải giá trị trả): nhả theo OWN(j, q) */
            if (!j->mainmode) {
                for (int q = 0; q < j->nl; q++) {
                    if (s->lt[q] == JT_SN) {
                        size_t j_not_own, j_alias = 0;
                        ld_rcx(j, OWN(j, q));
                        eb3(j, 0x48, 0x85, 0xC9);
                        eb2(j, 0x74, 0x00); j_not_own = j->n - 1;
                        ld_rax(j, LOC(q));
                        if (j->retty == JT_S) {
                            ld_rcx(j, SLOT(j, d - 1));
                            eb3(j, 0x48, 0x39, 0xC8);
                            eb2(j, 0x74, 0x00); j_alias = j->n - 1;
                        }
                        if (JIT_SYSV) mov_rdi_rax(j); else mov_rcx_rax(j);
                        mov_rax_imm(j, (int64_t)(intptr_t)&tjit_dec_sval);
                        call_rax(j);
                        if (j_alias) j->b[j_alias] = (uint8_t)(j->n - (j_alias + 1));
                        j->b[j_not_own] = (uint8_t)(j->n - (j_not_own + 1));
                    }
                }
            }
            ld_rax(j, SLOT(j, d - 1));
            jepi(j, 0);
            break;
        }
        case OP_CALL: {
            int k = (int)m->k;
            int fd = d - k - 1;
            if (st[fd] == JT_NAT) {
                int nat = s->sv[fd];
                int32_t A = SLOT(j, fd), X = SLOT(j, d - 1);
                uint8_t xt = st[d - 1];
                if (nat == NAT_LEN) {
                    /* [ptr+16] vừa là TList.len vừa là TVal.u.str.n (trùng layout);
                       dict đọc len ở offset riêng */
                    if (is_dict_t(xt)) {
                        ld_rax(j, X);
                        eb(j, 0x48); eb2(j, 0x8B, 0x80); e32(j, (int32_t)offsetof(TDict, len));
                        st_rax(j, A);
                        break;
                    }
                    ld_rax(j, X);
                    eb(j, 0x48); eb2(j, 0x8B, 0x80); e32(j, (int32_t)offsetof(TList, len));  /* mov rax,[rax+len] */
                    st_rax(j, A);
                } else if (nat == NAT_STR) {
                    /* str(x): helper dùng fmt_str (không lỗi), ty 0=I,1=F,2=B,3=S,4=SN */
                    int ty = xt == JT_I ? 0 : xt == JT_F ? 1 : xt == JT_B ? 2 :
                             xt == JT_S ? 3 : 4;
                    ld_rax(j, X);
                    if (JIT_SYSV) mov_rdi_rax(j); else mov_rcx_rax(j);
                    if (JIT_SYSV) mov_rsi_imm(j, ty); else mov_rdx_imm(j, ty);
                    mov_rax_imm(j, (int64_t)(intptr_t)&tjit_str_of);
                    call_rax(j);
                    st_rax(j, A);
                } else if (nat == NAT___NEW) {
                    /* __new(Kiểu, v...): guard kiểu rồi gọi helper cấp record mới */
                    int typg = s->sv[fd + 1];
                    TDict *tp = G[typg].u.dict;
                    uint8_t at2[JIT_SHAPE_MAXF];
                    int nf = (int)tp->fields->len, sh;
                    for (int q = 0; q < nf; q++) at2[q] = st[fd + 2 + q];
                    sh = shape_reg(tp, nf, at2);
                    mov_rax_imm(j, (int64_t)(intptr_t)&G[typg]);
                    eb3(j, 0x83, 0x38, (uint8_t)T_DICT); jguard(j, 0x85);
                    eb(j, 0x48); eb3(j, 0x8B, 0x48, 0x08);   /* mov rcx,[rax+8] */
                    mov_rax_imm(j, (int64_t)(intptr_t)tp);
                    eb3(j, 0x48, 0x39, 0xC1); jguard(j, 0x85); /* cmp rcx,rax */
                    if (JIT_SYSV) {
                        mov_rdi_rcx(j);
                        mov_rsi_imm(j, sh);
                        lea_rdx_rsp(j, SLOT(j, fd + 2));
                    } else {
                        mov_rdx_imm(j, sh);
                        lea_r8_rsp(j, SLOT(j, fd + 2));
                    }
                    mov_rax_imm(j, (int64_t)(intptr_t)&tjit_rec_new);
                    call_rax(j);
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
                uint8_t nat3[256];
                for (int q = 0; q < k; q++) nat3[q] = norm_ty(st[fd + 1 + q]);
                /* guard: global vẫn là đúng hàm F */
                mov_rax_imm(j, (int64_t)(intptr_t)g);
                eb3(j, 0x83, 0x38, (uint8_t)T_FN); jguard(j, 0x85);
                mov_rcx_imm(j, (int64_t)(intptr_t)F);
                eb(j, 0x48); eb3(j, 0x39, 0x48, 0x08); jguard(j, 0x85);   /* cmp [rax+8],rcx */
                /* arg1 = &slot(fd+1), arg2 = ctx */
                if (JIT_SYSV) { eb3(j, 0x48, 0x8D, 0xBC); eb(j, 0x24); e32(j, SLOT(j, fd + 1)); eb3(j, 0x48, 0x89, 0xDE); }
                else { eb3(j, 0x48, 0x8D, 0x8C); eb(j, 0x24); e32(j, SLOT(j, fd + 1)); eb3(j, 0x48, 0x89, 0xDA); }
                if (is_self_spec(j, F, nat3, k)) {
                    eb(j, 0xE8);                    /* call rel32 -> đầu entry (offset 0) */
                    e32(j, (int32_t)(0 - (int64_t)(j->n + 4)));
                } else {
                    mov_rax_imm(j, (int64_t)(intptr_t)callee_entry(F, nat3, k));
                    eb2(j, 0xFF, 0xD0);
                    keep_callee(j, F);
                }
                st_rax(j, SLOT(j, fd));
                if (j->osr) {
                    eb3(j, 0x83, 0x3B, 0x00);       /* cmp dword [rbx],0 */
                    jguard(j, 0x85);                /* callee huỷ: deopt tại CALL */
                    for (int q = 0; q < k; q++)
                        if (st[fd + 1 + q] == JT_SN) dec_temp(j, SLOT(j, fd + 1 + q), JT_SN);
                } else {
                    for (int q = 0; q < k; q++)
                        if (st[fd + 1 + q] == JT_SN) dec_temp(j, SLOT(j, fd + 1 + q), JT_SN);
                    ld_rax(j, SLOT(j, fd));
                    eb3(j, 0x83, 0x3B, 0x00);       /* cmp dword [rbx],0 */
                    jepi(j, 0x85);                  /* entry: truyền huỷ lên */
                }
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
        int32_t nsave = 2 * j->nl + j->maxd_seen + 2;
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
    free(j->dfix); free(j->nodec);
}

static int jc_setup(JC *j, TVM *vm, TFn *fn, int nest) {
    size_t beg, end;
    memset(j, 0, sizeof(*j));
    j->vm = vm; j->fn = fn; j->jf = jit_of(fn); j->nest = nest;
    j->prog = fn->prog;
    j->nl = fn->nlocals;
    if (!body_range(j->prog, fn, &beg, &end)) { jc_fail(j, "không xác định thân hàm"); return 0; }
    if (!decode(j, j->prog, beg, end)) return 0;
    j->maxd = 64;
    j->in = (AState *)calloc((size_t)j->nops + 1, sizeof(AState));
    for (int i = 0; i < j->nops; i++) st_alloc(j, &j->in[i]);
    j->label = (size_t *)calloc((size_t)j->nops + 1, sizeof(size_t));
    j->nodec = (uint8_t *)calloc((size_t)j->nl + 1, 1);
    return 1;
}

/* bảng deopt: trạng thái (stack + locals) tại từng điểm guard / lối ra */
static void build_deopts(JC *j, JitOsr *o) {
    int nl = j->nl;
    o->own_base = j->nl + j->maxd_seen + 2;
    o->nsave = 2 * j->nl + j->maxd_seen + 2;
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
    int32_t f = 32 + 8 * (2 * j->nl + j->maxd_seen + 2);
    if ((f & 15) != 8) f += 8;              /* sau 2 push + ret: rsp chia hết 16 khi call */
    j->frame = f;
}

/* biên dịch entry đặc hoá theo argty (đã chuẩn hoá owned -> borrowed) */
static int compile_entry(TVM *vm, TFn *fn, const uint8_t *argty, int nest) {
    TJitFn *jf = jit_of(fn);
    uint8_t nat[256];
    JitSpec *sp;
    JC j;
    for (int i = 0; i < fn->nargs; i++) nat[i] = norm_ty(argty[i]);
    sp = spec_find(jf, nat, fn->nargs);
    if (sp) return sp->estate == 2;
    if (getenv("TVM_NOJIT")) { fn->jit_off |= 3; return 0; }
    if (jf->nspec >= JIT_MAX_SPEC) { fn->jit_off |= 1; return 0; }
    sp = &jf->spec[jf->nspec++];
    memset(sp, 0, sizeof(*sp));
    sp->estate = 1;
    jf->nargs = fn->nargs;
    memcpy(sp->argty, nat, (size_t)fn->nargs);
    int ok = jc_setup(&j, vm, fn, nest);
    j.sp = sp;
    uint8_t *lt = (uint8_t *)calloc((size_t)fn->nlocals + 1, 1);
    for (int i = 0; i < fn->nargs && i < fn->nlocals; i++) lt[i] = nat[i];
    uint8_t retty = JT_UNK;
    if (ok) {
        /* G9: đăng ký trước shape cho mọi kiểu struct xuất hiện trong thân hàm
           để hàm đệ quy trả record (vd build(n)) đoán đúng retty ngay cả khi
           lời gọi đệ quy đứng trước lệnh tạo record */
        for (int qi = 0; qi < j.nops; qi++) {
            if (j.ops[qi].op == OP_LOADG && (size_t)j.ops[qi].k < j.prog->nglobals) {
                TVal *gv = &j.prog->globals[j.ops[qi].k];
                if (gv->t == T_DICT && gv->u.dict->kind == DK_TYPE && !gv->u.dict->mod) {
                    TDict *tp = gv->u.dict;
                    int nf = (int)tp->fields->len;
                    if (nf > 0 && nf <= JIT_SHAPE_MAXF) {
                        uint8_t ft_i[JIT_SHAPE_MAXF], ft_f[JIT_SHAPE_MAXF];
                        for (int f = 0; f < nf; f++) { ft_i[f] = JT_I; ft_f[f] = JT_F; }
                        shape_reg(tp, nf, ft_i);
                        shape_reg(tp, nf, ft_f);
                    }
                }
            }
        }
        uint8_t guesses[256];
        int ng = 0;
        guesses[ng++] = JT_I;
        guesses[ng++] = JT_F;
        guesses[ng++] = JT_B;
        guesses[ng++] = JT_SN;
        guesses[ng++] = JT_S;
        guesses[ng++] = JT_DN;
        guesses[ng++] = JT_D;
        guesses[ng++] = JT_NIL;
        for (int s = 0; s < njshapes && ng + 2 < 256; s++) {
            guesses[ng++] = own_of_shape(s);
            guesses[ng++] = bor_of_shape(s);
        }
        ok = 0;
        for (int gi = 0; gi < ng && !ok; gi++) {
            j.self_retty_guess = guesses[gi];
            j.used_guess = 0;
            j.fail = 0; j.why = NULL;
            memset(j.nodec, 0, (size_t)j.nl + 1);
            for (int i = 0; i < j.nops; i++) j.in[i].valid = 0;
            if (analyze(&j, 0, lt, &retty) && (!j.used_guess || retty == guesses[gi])) ok = 1;
            if (!ok && !j.used_guess) break;     /* thất bại không do đoán kiểu */
        }
    }
    if (ok) {
        sp->retty = retty;
        set_frame(&j);
        j.retty = retty;
        prologue(&j, nat, fn->nargs);
        ok = gen_body(&j) && finish(&j, &sp->unit);
    }
    if (!ok && getenv("TVM_JITLOG"))
        fprintf(stderr, "[jit] entry fail @%d: %s\n", fn->entry, j.why ? j.why : "?");
    jc_free(&j);
    free(lt);
    if (!ok) {
        sp->estate = 3;             /* nhớ bộ kiểu hỏng (không thử lại) */
        if (jf->nspec >= JIT_MAX_SPEC) fn->jit_off |= 1;
        return 0;
    }
    sp->entry = sp->unit.mem;
    sp->estate = 2;
    if (!fn->native) fn->native = sp->entry;    /* cờ "có bản native" */
    if (getenv("TVM_JITLOG")) fprintf(stderr, "[jit] entry ok @%d (%zu bytes)\n", fn->entry, sp->unit.size);
    return 1;
}

static uint8_t tag_ty(TType t) {
    return t == T_INT ? JT_I : t == T_FLOAT ? JT_F : t == T_BOOL ? JT_B : JT_UNK;
}
/* shape của record d (đăng ký nếu mới; giữ ref kiểu struct để địa chỉ không bị dùng lại) */
static int shape_reg(TDict *tp, int nf, const uint8_t *fty) {
    for (int q = 0; q < njshapes; q++) {
        JShape *sh = &jshapes[q];
        if (sh->tp == tp && sh->nf == nf && memcmp(sh->fty, fty, (size_t)nf) == 0)
            return q;
    }
    if (njshapes >= JIT_MAX_SHAPES) return -1;
    {
        JShape *sh = &jshapes[njshapes];
        sh->tp = tp;
        tp->refc++;
        sh->nf = nf;
        memcpy(sh->fty, fty, (size_t)nf);
        sh->keys = tp->fields->items;   /* mảng tên field của kiểu (kiểu đã giữ ref) */
        return njshapes++;
    }
}
static uint8_t rec_ty(const TDict *d) {
    uint8_t ft[JIT_SHAPE_MAXF];
    int s;
    if (d->kind != DK_REC || !d->meths || d->meths->mod || d->len > JIT_SHAPE_MAXF) return JT_UNK;
    for (size_t i = 0; i < d->len; i++) ft[i] = tag_ty(d->vals[i].t);
    s = shape_reg(d->meths, (int)d->len, ft);
    return s < 0 ? JT_UNK : bor_of_shape(s);
}
/* bảng kiểu field cho tjit_rec_new (tvm.c) */
const uint8_t *tjit_shape_fty(int shape) { return jshapes[shape].fty; }
static uint8_t val_ty(const TVal *v) {
    if (v->t == T_LIST) return list_ty(v->u.list);
    if (v->t == T_DICT) {
        if (v->u.dict->kind == DK_DICT) return JT_D;
        return rec_ty(v->u.dict);
    }
    if (v->t == T_STR) return JT_S;
    return tag_ty(v->t);
}
/* guard lúc vào native: list chỉ cần đúng tag (phần tử được guard từng lần đọc);
   record: đúng kiểu struct (field guard từng lần đọc) */
static int ty_ok(const TVal *v, uint8_t want) {
    if (is_list_t(want)) return v->t == T_LIST;
    if (want == JT_S) return v->t == T_STR;
    if (want == JT_D) return v->t == T_DICT && v->u.dict->kind == DK_DICT;
    if (is_rec_t(want)) {
        JShape *sh = shape_of(want);
        return v->t == T_DICT && v->u.dict->kind == DK_REC && v->u.dict->meths == sh->tp &&
               v->u.dict->len == (size_t)sh->nf;
    }
    return tag_ty(v->t) == want;
}

/* đếm lời gọi qua interpreter; đủ nóng thì biên dịch bản đặc hoá cho bộ kiểu hiện
   tại. Đã có bản native (kiểu lần này không khớp): chỉ xét lại mỗi 256 lần (tính
   kiểu list phải quét phần tử). */
void t_jit_maybe(TVM *vm, TFn *fn, TVal *args) {
    uint8_t at[256];
#if !defined(_M_X64) && !defined(__x86_64__)
#if defined(_M_ARM64) || defined(__aarch64__)
    t_arm64_jit_maybe(vm, fn, args);
    return;
#else
    (void)vm; (void)args;
    fn->jit_off |= 1;    /* JIT x64/arm64 không có trên kiến trúc này -> interpreter */
    return;
#endif
#endif
    if (fn->jit_off & 1) return;
    if (++fn->ncalls < JIT_THRESHOLD) return;
    if (fn->native && (fn->ncalls & 255)) return;
    for (int i = 0; i < fn->nargs; i++) {
        at[i] = val_ty(&args[i]);
        if (at[i] == JT_UNK) { if (!fn->native) fn->jit_off |= 1; return; }
    }
    compile_entry(vm, fn, at, 0);
}

static int to_ret(uint8_t ty, int64_t r, TVal *ret) {
    if (ty == JT_I) { ret->t = T_INT; ret->u.i = r; }
    else if (ty == JT_F) { ret->t = T_FLOAT; memcpy(&ret->u.f, &r, 8); }
    else if (ty == JT_NIL) { ret->t = T_NIL; }
    else if (ty == JT_S) { *ret = *(const TVal *)(intptr_t)r; tval_retain(ret); }
    else if (ty == JT_SN) { *ret = *(TVal *)(intptr_t)r; tjit_wrap_free((TVal *)(intptr_t)r); }
    else if (is_rec_t(ty)) {
        ret->t = T_DICT; ret->u.dict = (TDict *)(intptr_t)r; tval_retain(ret);
    }
    else if (is_own_rec_t(ty)) { ret->t = T_DICT; ret->u.dict = (TDict *)(intptr_t)r; }
    else if (ty == JT_D) {
        ret->t = T_DICT; ret->u.dict = (TDict *)(intptr_t)r; tval_retain(ret);
    }
    else if (ty == JT_DN) { ret->t = T_DICT; ret->u.dict = (TDict *)(intptr_t)r; }
    else { ret->t = T_BOOL; ret->u.b = (int)(r & 1); }
    return 1;
}

static int64_t val_bits(const TVal *v) {
    int64_t x = 0;
    if (v->t == T_INT) x = v->u.i;
    else if (v->t == T_FLOAT) memcpy(&x, &v->u.f, 8);
    else if (v->t == T_BOOL) x = v->u.b ? 1 : 0;
    else if (v->t == T_STR) x = (int64_t)(intptr_t)v;   /* mượn TVal (global/arg/local giữ ref) */
    else if (v->t == T_LIST) x = (int64_t)(intptr_t)v->u.list;   /* mượn: local/arg giữ ref */
    else if (v->t == T_DICT) x = (int64_t)(intptr_t)v->u.dict;    /* record mượn */
    return x;
}
/* giá trị native (theo kiểu trừu tượng) -> TVal sở hữu được.
   borrowed (list/record mượn, chuỗi, FN/NAT/TYP): retain; owned: move (không retain). */
static TVal bits_val(TVM *vm, uint8_t ty, int64_t x, int32_t sv) {
    TVal v;
    v.t = T_NIL;
    if (ty == JT_I) { v.t = T_INT; v.u.i = x; }
    else if (ty == JT_F) { v.t = T_FLOAT; memcpy(&v.u.f, &x, 8); }
    else if (ty == JT_B) { v.t = T_BOOL; v.u.b = (int)(x & 1); }
    else if (ty == JT_S) { v = *(const TVal *)(intptr_t)x; tval_retain(&v); }
    else if (ty == JT_SN) { v = *(TVal *)(intptr_t)x; tjit_wrap_free((TVal *)(intptr_t)x); }
    else if (is_list_t(ty)) { v.t = T_LIST; v.u.list = (TList *)(intptr_t)x; tval_retain(&v); }
    else if (is_rec_t(ty)) { v.t = T_DICT; v.u.dict = (TDict *)(intptr_t)x; tval_retain(&v); }
    else if (is_own_rec_t(ty)) { v.t = T_DICT; v.u.dict = (TDict *)(intptr_t)x; }
    else if (ty == JT_D) { v.t = T_DICT; v.u.dict = (TDict *)(intptr_t)x; tval_retain(&v); }
    else if (ty == JT_DN) { v.t = T_DICT; v.u.dict = (TDict *)(intptr_t)x; }
    else if (ty == JT_FN || ty == JT_NAT || ty == JT_TYP) { v = vm->prog->globals[sv]; tval_retain(&v); }
    return v;
}

int t_jit_call(TVM *vm, TFn *fn, TVal *args, int argc, TVal *ret) {
#if defined(_M_ARM64) || defined(__aarch64__)
    return t_arm64_jit_call(vm, fn, args, argc, ret);
#endif
    TJitFn *jf = (TJitFn *)fn->jit;
    int64_t in[256];
    JitSpec *sp = NULL;
    (void)vm;
    if (!fn->native || !jf || argc != jf->nargs) return 0;
    for (int q = 0; q < jf->nspec && !sp; q++) {
        JitSpec *c = &jf->spec[q];
        int i;
        if (c->estate != 2) continue;
        for (i = 0; i < argc; i++) if (!ty_ok(&args[i], c->argty[i])) break;
        if (i == argc) sp = c;
    }
    if (!sp) return 0;
    for (int i = 0; i < argc; i++) in[i] = val_bits(&args[i]);
    JitCtx ctx;
    ctx.abort = 0; ctx.deopt_id = 0; ctx.save = NULL;
    ctx.stack_limit = (char *)((uintptr_t)&ctx - JIT_STACK_BUDGET);
    int64_t r = ((JitEntry)sp->entry)(in, &ctx);
    if (ctx.abort) {
        if (++sp->aborts > JIT_MAX_ABORTS) {
            /* bản này huỷ quá nhiều: tắt nó (code vẫn giữ: callee khác có thể nhảy vào) */
            int live = 0;
            sp->estate = 3;
            for (int q = 0; q < jf->nspec; q++) if (jf->spec[q].estate == 2) live = 1;
            if (!live) { fn->native = NULL; fn->jit_off |= 1; }
        }
        return 0;
    }
    return to_ret(sp->retty, r, ret);
}

int t_jit_osr(TVM *vm, TFn *fn, TVal *locals, size_t pc, TVal *ret) {
#if !defined(_M_X64) && !defined(__x86_64__)
#if defined(_M_ARM64) || defined(__aarch64__)
    return t_arm64_jit_osr(vm, fn, locals, pc, ret);
#else
    (void)vm; (void)locals; (void)pc; (void)ret;
    fn->jit_off |= 2;    /* không có JIT x64/arm64 trên kiến trúc này -> interpreter */
    return 0;
#endif
#endif
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
            j.retty = retty;
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
                int is_own = (o->own_base + i < o->nsave) ? (save[o->own_base + i] != 0) : 1;
                if (t == JT_I || t == JT_F || t == JT_B) {
                    TVal nv = bits_val(vm, t, save[i], 0);
                    tval_free(&locals[i]);
                    locals[i] = nv;
                } else if (is_own_rec_t(t)) {
                    uint8_t rt = is_own ? t : bor_of_shape(rec_shape(t));
                    TVal nv = bits_val(vm, rt, save[i], 0);
                    tval_free(&locals[i]);
                    locals[i] = nv;
                } else if (is_rec_t(t)) {
                    TVal nv = bits_val(vm, t, save[i], 0);
                    tval_free(&locals[i]);
                    locals[i] = nv;
                } else if (t == JT_SN || t == JT_S) {
                    uint8_t rt = (t == JT_SN && is_own) ? JT_SN : JT_S;
                    TVal nv = bits_val(vm, rt, save[i], 0);
                    tval_free(&locals[i]);
                    locals[i] = nv;
                } else if (is_dict_t(t)) {
                    uint8_t rt = (t == JT_DN && is_own) ? JT_DN : JT_D;
                    TVal nv = bits_val(vm, rt, save[i], 0);
                    tval_free(&locals[i]);
                    locals[i] = nv;
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
#if !defined(_M_X64) && !defined(__x86_64__)
    (void)vm; (void)header; (void)end;
    return -1;           /* không JIT được trên kiến trúc này */
#endif
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
        j.vm = vm; j.prog = P; j.fn = NULL; j.jf = &jmain_fn; j.osr = 1; j.mainmode = 1;
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
                /* kiểu struct đọc trực tiếp kèm guard (không làm local giả) */
                if (G[m->k].t == T_DICT && G[m->k].u.dict->kind == DK_TYPE) continue;
                if (j.gpseudo[m->k] < 0) { j.gpseudo[m->k] = np; j.pglob[np++] = (int32_t)m->k; }
            }
            j.nl = np;
            j.maxd = 64;
            j.in = (AState *)calloc((size_t)j.nops + 1, sizeof(AState));
            for (int i = 0; i < j.nops; i++) st_alloc(&j, &j.in[i]);
            j.label = (size_t *)calloc((size_t)j.nops + 1, sizeof(size_t));
            j.nodec = (uint8_t *)calloc((size_t)np + 1, 1);
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
    for (int q = 0; q < jf->nspec; q++)
        if (jf->spec[q].unit.mem) jit_free_exec(jf->spec[q].unit.mem, jf->spec[q].unit.size);
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
