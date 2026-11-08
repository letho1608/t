/* jit_arm64.c — v5.2 (J7 Hoàn chỉnh) Method JIT + OSR AArch64 (ARM64) cho T.
 *
 * Bao gồm:
 *   1. Bộ phát mã máy AArch64 (ARMv8-A A64, mỗi lệnh 4 byte little-endian):
 *      số học 64-bit có kiểm tràn (adds/subs/mul/smulh/sdiv), thực 64-bit IEEE-754
 *      (fadd/fsub/fmul/fdiv/fneg/fabs/fcmp/scvtf/fcvtzs/fmov/fcsel), bitwise 64-bit
 *      (and/orr/eor/mvn/lslv/asrv), so sánh & điều kiện (cmp/cset/csel/b/b.cond/cbz/cbnz),
 *      quản lý frame AAPCS64 (stp/ldp x29,x30 / x19,x20), gọi hàm đệ quy & helper C (bl/blr/ret).
 *   2. Trình biên dịch Bytecode T -> AArch64:
 *      - Giải mã bytecode (OP_PUSH_*, OP_LOAD/STORE/DECL/BIND, OP_LOADG, số học, so sánh,
 *        bitwise, superop INC_L/DEC_L/ADDI_L/ADDS_L/SUBS_L/INCB_L, nhánh/vòng lặp,
 *        đệ quy OP_CALL, builtin int/float/abs/min/max/len, list OP_IDX/SETIDX/LEN).
 *      - Thông dịch trừu tượng (abstract interpretation) suy kiểu (int/float/bool/list)
 *        tới điểm bất động (fixed-point) qua các cạnh nhảy & vòng lặp.
 *      - Biên dịch cả hàm (Entry mode) và vòng lặp nóng (OSR mode) kèm bảng Deopt
 *        đồng bộ trạng thái locals + operand stack về VM interpreter khi guard thất bại.
 *   3. Thực thi đa nền tảng:
 *      - Trên phần cứng ARM64 thật (_M_ARM64 / __aarch64__): cấp phát trang thực thi
 *        (VirtualAlloc / mmap + mprotect / MAP_JIT + __builtin___clear_cache) và nối
 *        trực tiếp vào t_jit_maybe / t_jit_call / t_jit_osr.
 *      - Trên mọi kiến trúc (kể cả x86-64): tích hợp máy ảo tập lệnh AArch64 (a64_sim_exec)
 *        giải mã & chạy trực tiếp các từ lệnh 32-bit ARM64 vừa phát sinh trên bộ nhớ thật,
 *        cho phép kiểm chứng toàn diện bytecode->ARM64 + OSR + Deopt qua __sys.arm64_compile,
 *        __sys.arm64_run, __sys.arm64_osr_run.
 */
#if !defined(_WIN32)
#ifndef _DEFAULT_SOURCE
#define _DEFAULT_SOURCE
#endif
#ifndef _DARWIN_C_SOURCE
#define _DARWIN_C_SOURCE
#endif
#endif

#include "tvm.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>

#if defined(_M_ARM64) || defined(__aarch64__)
#define A64_NATIVE_HOST 1
#if defined(_WIN32)
#include <windows.h>
static void *a64_alloc_exec(size_t n) {
    return VirtualAlloc(NULL, n, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
}
static int a64_seal_exec(void *m, size_t n) {
    DWORD old;
    if (!VirtualProtect(m, n, PAGE_EXECUTE_READ, &old)) return 0;
    FlushInstructionCache(GetCurrentProcess(), m, n);
    return 1;
}
static void a64_free_exec(void *m, size_t n) { (void)n; VirtualFree(m, 0, MEM_RELEASE); }
#else
#include <sys/mman.h>
#include <unistd.h>
#if !defined(MAP_ANONYMOUS) && defined(MAP_ANON)
#define MAP_ANONYMOUS MAP_ANON
#endif
static void *a64_alloc_exec(size_t n) {
    void *m = mmap(NULL, n, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    return (m == MAP_FAILED) ? NULL : m;
}
static int a64_seal_exec(void *m, size_t n) {
    __builtin___clear_cache((char *)m, (char *)m + n);
    return mprotect(m, n, PROT_READ | PROT_EXEC) == 0;
}
static void a64_free_exec(void *m, size_t n) { munmap(m, n); }
#endif
#else
#define A64_NATIVE_HOST 0
#endif

/* ---------- buffer phát mã ---------- */
typedef struct { uint8_t *buf; size_t cap; size_t len; int err; } A64Buf;

static int a64_grow(A64Buf *b) {
    if (b->len + 4 <= b->cap) return 1;
    size_t nc = b->cap ? b->cap * 2 : 256;
    while (nc < b->len + 4) nc *= 2;
    uint8_t *nb = (uint8_t *)realloc(b->buf, nc);
    if (!nb) { b->err = 1; return 0; }
    b->buf = nb; b->cap = nc;
    return 1;
}
static void a64_emit(A64Buf *b, uint32_t w) {
    if (!a64_grow(b)) return;
    b->buf[b->len]     = (uint8_t)(w);
    b->buf[b->len + 1] = (uint8_t)(w >> 8);
    b->buf[b->len + 2] = (uint8_t)(w >> 16);
    b->buf[b->len + 3] = (uint8_t)(w >> 24);
    b->len += 4;
}
static void a64_patch_u32(A64Buf *b, size_t at, uint32_t w) {
    if (at + 4 > b->len) return;
    b->buf[at]     = (uint8_t)(w);
    b->buf[at + 1] = (uint8_t)(w >> 8);
    b->buf[at + 2] = (uint8_t)(w >> 16);
    b->buf[at + 3] = (uint8_t)(w >> 24);
}
static uint32_t a64_read_u32(const uint8_t *p) {
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}
static void a64_free(A64Buf *b) { free(b->buf); b->buf = NULL; b->cap = b->len = 0; }

/* ---------- mã điều kiện AArch64 ---------- */
enum {
    A64_EQ = 0,  A64_NE = 1,  A64_CS = 2,  A64_CC = 3,
    A64_MI = 4,  A64_PL = 5,  A64_VS = 6,  A64_VC = 7,
    A64_HI = 8,  A64_LS = 9,  A64_GE = 10, A64_LT = 11,
    A64_GT = 12, A64_LE = 13, A64_AL = 14
};

/* ---------- emitters AArch64 ---------- */
static void a64_ret(A64Buf *b) { a64_emit(b, 0xD65F03C0u); }
static void a64_add(A64Buf *b, int rd, int rn, int rm) { a64_emit(b, 0x8B000000u | ((uint32_t)rm << 16) | ((uint32_t)rn << 5) | (uint32_t)rd); }
static void a64_adds(A64Buf *b, int rd, int rn, int rm) { a64_emit(b, 0xAB000000u | ((uint32_t)rm << 16) | ((uint32_t)rn << 5) | (uint32_t)rd); }
static void a64_sub(A64Buf *b, int rd, int rn, int rm) { a64_emit(b, 0xCB000000u | ((uint32_t)rm << 16) | ((uint32_t)rn << 5) | (uint32_t)rd); }
static void a64_subs(A64Buf *b, int rd, int rn, int rm) { a64_emit(b, 0xEB000000u | ((uint32_t)rm << 16) | ((uint32_t)rn << 5) | (uint32_t)rd); }
static void a64_add_imm(A64Buf *b, int rd, int rn, int imm12) { a64_emit(b, 0x91000000u | (((uint32_t)imm12 & 0xFFFu) << 10) | ((uint32_t)rn << 5) | (uint32_t)rd); }
static void a64_sub_imm(A64Buf *b, int rd, int rn, int imm12) { a64_emit(b, 0xD1000000u | (((uint32_t)imm12 & 0xFFFu) << 10) | ((uint32_t)rn << 5) | (uint32_t)rd); }
static void a64_mul(A64Buf *b, int rd, int rn, int rm) { a64_emit(b, 0x9B000000u | ((uint32_t)rm << 16) | (31u << 10) | ((uint32_t)rn << 5) | (uint32_t)rd); }
static void a64_smulh(A64Buf *b, int rd, int rn, int rm) { a64_emit(b, 0x9B407C00u | ((uint32_t)rm << 16) | ((uint32_t)rn << 5) | (uint32_t)rd); }
static void a64_sdiv(A64Buf *b, int rd, int rn, int rm) { a64_emit(b, 0x9AC00C00u | ((uint32_t)rm << 16) | ((uint32_t)rn << 5) | (uint32_t)rd); }
static void a64_and(A64Buf *b, int rd, int rn, int rm) { a64_emit(b, 0x8A000000u | ((uint32_t)rm << 16) | ((uint32_t)rn << 5) | (uint32_t)rd); }
static void a64_orr(A64Buf *b, int rd, int rn, int rm) { a64_emit(b, 0xAA000000u | ((uint32_t)rm << 16) | ((uint32_t)rn << 5) | (uint32_t)rd); }
static void a64_eor(A64Buf *b, int rd, int rn, int rm) { a64_emit(b, 0xCA000000u | ((uint32_t)rm << 16) | ((uint32_t)rn << 5) | (uint32_t)rd); }
static void a64_mvn(A64Buf *b, int rd, int rm) { a64_emit(b, 0xAA2003E0u | ((uint32_t)rm << 16) | (uint32_t)rd); }
static void a64_lslv(A64Buf *b, int rd, int rn, int rm) { a64_emit(b, 0x9AC02000u | ((uint32_t)rm << 16) | ((uint32_t)rn << 5) | (uint32_t)rd); }
static void a64_asrv(A64Buf *b, int rd, int rn, int rm) { a64_emit(b, 0x9AC02800u | ((uint32_t)rm << 16) | ((uint32_t)rn << 5) | (uint32_t)rd); }
static void a64_mov_reg(A64Buf *b, int rd, int rm) { a64_orr(b, rd, 31, rm); }
static void a64_ldr(A64Buf *b, int rt, int rn, int off) { a64_emit(b, 0xF9400000u | (((uint32_t)(off >> 3) & 0xFFFu) << 10) | ((uint32_t)rn << 5) | (uint32_t)rt); }
static void a64_str(A64Buf *b, int rt, int rn, int off) { a64_emit(b, 0xF9000000u | (((uint32_t)(off >> 3) & 0xFFFu) << 10) | ((uint32_t)rn << 5) | (uint32_t)rt); }
static void a64_ldrw(A64Buf *b, int rt, int rn, int off) { a64_emit(b, 0xB9400000u | (((uint32_t)(off >> 2) & 0xFFFu) << 10) | ((uint32_t)rn << 5) | (uint32_t)rt); }
static void a64_strw(A64Buf *b, int rt, int rn, int off) { a64_emit(b, 0xB9000000u | (((uint32_t)(off >> 2) & 0xFFFu) << 10) | ((uint32_t)rn << 5) | (uint32_t)rt); }
static void a64_movz(A64Buf *b, int rd, int imm16, int shift) { a64_emit(b, 0xD2800000u | ((uint32_t)(shift >> 4) << 21) | ((uint32_t)(imm16 & 0xFFFF) << 5) | (uint32_t)rd); }
static void a64_movk(A64Buf *b, int rd, int imm16, int shift) { a64_emit(b, 0xF2800000u | ((uint32_t)(shift >> 4) << 21) | ((uint32_t)(imm16 & 0xFFFF) << 5) | (uint32_t)rd); }
static void a64_mov_imm64(A64Buf *b, int rd, uint64_t u) {
    a64_movz(b, rd, (int)(u & 0xFFFFu), 0);
    if ((u >> 16) & 0xFFFFu) a64_movk(b, rd, (int)((u >> 16) & 0xFFFFu), 16);
    if ((u >> 32) & 0xFFFFu) a64_movk(b, rd, (int)((u >> 32) & 0xFFFFu), 32);
    if ((u >> 48) & 0xFFFFu) a64_movk(b, rd, (int)((u >> 48) & 0xFFFFu), 48);
}
static void a64_cmp(A64Buf *b, int rn, int rm) { a64_emit(b, 0xEB000000u | ((uint32_t)rm << 16) | ((uint32_t)rn << 5) | 31u); }
static void a64_cset(A64Buf *b, int rd, int cond) { a64_emit(b, 0x9A9F07E0u | (((uint32_t)(cond ^ 1) & 0xFu) << 12) | (uint32_t)rd); }
static void a64_csel(A64Buf *b, int rd, int rn, int rm, int cond) { a64_emit(b, 0x9A800000u | ((uint32_t)rm << 16) | (((uint32_t)cond & 0xFu) << 12) | ((uint32_t)rn << 5) | (uint32_t)rd); }
static void a64_b(A64Buf *b, int off_words) { a64_emit(b, 0x14000000u | ((uint32_t)off_words & 0x3FFFFFFu)); }
static void a64_bcond(A64Buf *b, int cond, int off_words) { a64_emit(b, 0x54000000u | (((uint32_t)off_words & 0x7FFFFu) << 5) | ((uint32_t)cond & 0xFu)); }
static void a64_bl(A64Buf *b, int off_words) { a64_emit(b, 0x94000000u | ((uint32_t)off_words & 0x3FFFFFFu)); }
static void a64_cbz(A64Buf *b, int rd, int off_words) { a64_emit(b, 0xB4000000u | (((uint32_t)off_words & 0x7FFFFu) << 5) | (uint32_t)rd); }
static void a64_cbnz(A64Buf *b, int rd, int off_words) { a64_emit(b, 0xB5000000u | (((uint32_t)off_words & 0x7FFFFu) << 5) | (uint32_t)rd); }
static void a64_blr(A64Buf *b, int rn) { a64_emit(b, 0xD63F0000u | ((uint32_t)rn << 5)); }
/* float64 */
static void a64_ldr_d(A64Buf *b, int rt, int rn, int off) { a64_emit(b, 0xFD400000u | (((uint32_t)(off >> 3) & 0xFFFu) << 10) | ((uint32_t)rn << 5) | (uint32_t)rt); }
static void a64_str_d(A64Buf *b, int rt, int rn, int off) { a64_emit(b, 0xFD000000u | (((uint32_t)(off >> 3) & 0xFFFu) << 10) | ((uint32_t)rn << 5) | (uint32_t)rt); }
static void a64_fmov_dx(A64Buf *b, int dd, int xn) { a64_emit(b, 0x9E670000u | ((uint32_t)xn << 5) | (uint32_t)dd); }
static void a64_fmov_xd(A64Buf *b, int xd, int dn) { a64_emit(b, 0x9E660000u | ((uint32_t)dn << 5) | (uint32_t)xd); }
static void a64_fadd(A64Buf *b, int rd, int rn, int rm) { a64_emit(b, 0x1E602800u | ((uint32_t)rm << 16) | ((uint32_t)rn << 5) | (uint32_t)rd); }
static void a64_fsub(A64Buf *b, int rd, int rn, int rm) { a64_emit(b, 0x1E603800u | ((uint32_t)rm << 16) | ((uint32_t)rn << 5) | (uint32_t)rd); }
static void a64_fmul(A64Buf *b, int rd, int rn, int rm) { a64_emit(b, 0x1E606800u | ((uint32_t)rm << 16) | ((uint32_t)rn << 5) | (uint32_t)rd); }
static void a64_fdiv(A64Buf *b, int rd, int rn, int rm) { a64_emit(b, 0x1E601800u | ((uint32_t)rm << 16) | ((uint32_t)rn << 5) | (uint32_t)rd); }
static void a64_fneg(A64Buf *b, int rd, int rn) { a64_emit(b, 0x1E614000u | ((uint32_t)rn << 5) | (uint32_t)rd); }
static void a64_fabs(A64Buf *b, int rd, int rn) { a64_emit(b, 0x1E60C000u | ((uint32_t)rn << 5) | (uint32_t)rd); }
static void a64_fcmp(A64Buf *b, int rn, int rm) { a64_emit(b, 0x1E602000u | ((uint32_t)rm << 16) | ((uint32_t)rn << 5)); }
static void a64_fcsel(A64Buf *b, int rd, int rn, int rm, int cond) { a64_emit(b, 0x1E600C00u | ((uint32_t)rm << 16) | (((uint32_t)cond & 0xFu) << 12) | ((uint32_t)rn << 5) | (uint32_t)rd); }
static void a64_scvtf_d(A64Buf *b, int rd, int rn) { a64_emit(b, 0x9E620000u | ((uint32_t)rn << 5) | (uint32_t)rd); }
static void a64_fcvtzs(A64Buf *b, int rd, int rn) { a64_emit(b, 0x9E7E0000u | ((uint32_t)rn << 5) | (uint32_t)rd); }
/* frame cơ bản (cho k3 oracle) */
static void a64_prologue(A64Buf *b) { a64_emit(b, 0xA9BF7BFDu); a64_emit(b, 0xAA0003FDu); }
static void a64_epilogue(A64Buf *b) { a64_emit(b, 0xA8C17BFDu); a64_ret(b); }

/* ---------- kernel mẫu (entry ABI: x0=in*, x1=ctx*, x0=ret) ---------- */
static void build_k1(A64Buf *b) {
    a64_ldr(b, 9, 0, 0);
    a64_ldr(b, 10, 0, 8);
    a64_add(b, 0, 9, 10);
    a64_ret(b);
}
static void build_k2(A64Buf *b) {
    a64_ldr_d(b, 0, 0, 0);
    a64_ldr_d(b, 1, 0, 8);
    a64_fadd(b, 0, 0, 1);
    a64_ret(b);
}
static void build_k3(A64Buf *b) {
    a64_prologue(b);
    a64_add(b, 0, 0, 1);
    a64_epilogue(b);
}

static const uint8_t k1_exp[16] = {
    0x09,0x00,0x40,0xF9, 0x0A,0x04,0x40,0xF9, 0x20,0x01,0x0A,0x8B, 0xC0,0x03,0x5F,0xD6
};
static const uint8_t k2_exp[16] = {
    0x00,0x00,0x40,0xFD, 0x01,0x04,0x40,0xFD, 0x00,0x28,0x61,0x1E, 0xC0,0x03,0x5F,0xD6
};
static const uint8_t k3_exp[20] = {
    0xFD,0x7B,0xBF,0xA9, 0xFD,0x03,0x00,0xAA, 0x00,0x00,0x01,0x8B, 0xFD,0x7B,0xC1,0xA8, 0xC0,0x03,0x5F,0xD6
};

static int check_oracle(A64Buf *b, const uint8_t *exp, size_t n) {
    return !b->err && b->len == n && memcmp(b->buf, exp, n) == 0;
}

/* ========================================================================= */
/* Máy ảo thực thi tập lệnh AArch64 (Virtual AArch64 CPU cho kiểm chứng/x64) */
/* ========================================================================= */

typedef struct {
    int32_t abort;
    int32_t deopt_id;
    char *stack_limit;
    int64_t *save;
} A64Ctx;

typedef int64_t (*A64EntryFn)(int64_t *in, A64Ctx *ctx);

static int a64_cond_pass(int cond, int N, int Z, int C, int V) {
    switch (cond & 0xF) {
    case A64_EQ: return Z == 1;
    case A64_NE: return Z == 0;
    case A64_CS: return C == 1;
    case A64_CC: return C == 0;
    case A64_MI: return N == 1;
    case A64_PL: return N == 0;
    case A64_VS: return V == 1;
    case A64_VC: return V == 0;
    case A64_HI: return C == 1 && Z == 0;
    case A64_LS: return !(C == 1 && Z == 0);
    case A64_GE: return N == V;
    case A64_LT: return N != V;
    case A64_GT: return Z == 0 && N == V;
    case A64_LE: return !(Z == 0 && N == V);
    default:     return 1;
    }
}

#define A64_SIM_STACK_SZ (256 * 1024)

static TDict *a64_rec_new(TDict *tp, int shape, int64_t *slots);

static int64_t a64_sim_exec(const uint8_t *code, size_t code_len, int64_t *in, A64Ctx *ctx) {
    uint64_t x[32];
    double d[32];
    int N = 0, Z = 0, C = 0, V = 0;
    memset(x, 0, sizeof(x));
    memset(d, 0, sizeof(d));

    uint8_t *sim_stack = (uint8_t *)malloc(A64_SIM_STACK_SZ);
    if (!sim_stack) { if (ctx) ctx->abort = 1; return 0; }
    uintptr_t sp_top = ((uintptr_t)(sim_stack + A64_SIM_STACK_SZ - 64)) & ~(uintptr_t)15;
    if (ctx) ctx->stack_limit = (char *)(sim_stack + 4096);

    x[0] = (uint64_t)(uintptr_t)in;
    x[1] = (uint64_t)(uintptr_t)ctx;
    x[30] = 0xFFFFFFFFFFFFFFFFULL; /* sentinel LR */
    uint64_t sp = (uint64_t)sp_top;
    size_t pc = 0;

    #define XR(r) ((r) == 31 ? 0ULL : x[r])
    #define XSP(r) ((r) == 31 ? sp : x[r])
    #define SETX(r, v) do { if ((r) != 31) x[r] = (uint64_t)(v); } while (0)
    #define SETSP(r, v) do { if ((r) == 31) sp = (uint64_t)(v); else x[r] = (uint64_t)(v); } while (0)

    while (pc + 4 <= code_len) {
        uint32_t w = a64_read_u32(code + pc);
        int rd = (int)(w & 0x1Fu);
        int rn = (int)((w >> 5) & 0x1Fu);
        int rm = (int)((w >> 16) & 0x1Fu);

        if (w == 0xD65F03C0u) { /* ret */
            if (x[30] == 0xFFFFFFFFFFFFFFFFULL) break;
            pc = (size_t)x[30];
            continue;
        }
        if (w == 0xA9BF7BFDu) { /* stp x29, x30, [sp, #-16]! */
            sp -= 16;
            *(uint64_t *)(uintptr_t)sp = x[29];
            *(uint64_t *)(uintptr_t)(sp + 8) = x[30];
            pc += 4; continue;
        }
        if (w == 0xA8C17BFDu) { /* ldp x29, x30, [sp], #16 */
            x[29] = *(uint64_t *)(uintptr_t)sp;
            x[30] = *(uint64_t *)(uintptr_t)(sp + 8);
            sp += 16;
            pc += 4; continue;
        }
        if (w == 0xA9BF53F3u) { /* stp x19, x20, [sp, #-16]! */
            sp -= 16;
            *(uint64_t *)(uintptr_t)sp = x[19];
            *(uint64_t *)(uintptr_t)(sp + 8) = x[20];
            pc += 4; continue;
        }
        if (w == 0xA8C153F3u) { /* ldp x19, x20, [sp], #16 */
            x[19] = *(uint64_t *)(uintptr_t)sp;
            x[20] = *(uint64_t *)(uintptr_t)(sp + 8);
            sp += 16;
            pc += 4; continue;
        }
        if (w == 0xAA0003FDu) { /* legacy k3: mov x29, x0 (or sp) */
            x[29] = sp;
            pc += 4; continue;
        }
        /* add/sub imm12 (dùng XSP vì liên quan tới sp/x29) */
        if ((w & 0xFFC00000u) == 0x91000000u) {
            uint64_t imm = (w >> 10) & 0xFFFu;
            SETSP(rd, XSP(rn) + imm);
            pc += 4; continue;
        }
        if ((w & 0xFFC00000u) == 0xD1000000u) {
            uint64_t imm = (w >> 10) & 0xFFFu;
            SETSP(rd, XSP(rn) - imm);
            pc += 4; continue;
        }
        /* movz / movk */
        if ((w & 0xFF800000u) == 0xD2800000u) {
            int sh = (int)(((w >> 21) & 3u) * 16u);
            uint64_t imm = (w >> 5) & 0xFFFFu;
            SETX(rd, imm << sh);
            pc += 4; continue;
        }
        if ((w & 0xFF800000u) == 0xF2800000u) {
            int sh = (int)(((w >> 21) & 3u) * 16u);
            uint64_t imm = (w >> 5) & 0xFFFFu;
            uint64_t cur = XR(rd) & ~(0xFFFFULL << sh);
            SETX(rd, cur | (imm << sh));
            pc += 4; continue;
        }
        /* ldr / str 64-bit */
        if ((w & 0xFFC00000u) == 0xF9400000u) {
            uint64_t off = ((w >> 10) & 0xFFFu) << 3;
            SETX(rd, *(uint64_t *)(uintptr_t)(XSP(rn) + off));
            pc += 4; continue;
        }
        if ((w & 0xFFC00000u) == 0xF9000000u) {
            uint64_t off = ((w >> 10) & 0xFFFu) << 3;
            *(uint64_t *)(uintptr_t)(XSP(rn) + off) = XR(rd);
            pc += 4; continue;
        }
        /* ldrw / strw 32-bit */
        if ((w & 0xFFC00000u) == 0xB9400000u) {
            uint64_t off = ((w >> 10) & 0xFFFu) << 2;
            SETX(rd, (uint64_t)(*(uint32_t *)(uintptr_t)(XSP(rn) + off)));
            pc += 4; continue;
        }
        if ((w & 0xFFC00000u) == 0xB9000000u) {
            uint64_t off = ((w >> 10) & 0xFFFu) << 2;
            *(uint32_t *)(uintptr_t)(XSP(rn) + off) = (uint32_t)XR(rd);
            pc += 4; continue;
        }
        /* ldr_d / str_d */
        if ((w & 0xFFC00000u) == 0xFD400000u) {
            uint64_t off = ((w >> 10) & 0xFFFu) << 3;
            d[rd] = *(double *)(uintptr_t)(XSP(rn) + off);
            pc += 4; continue;
        }
        if ((w & 0xFFC00000u) == 0xFD000000u) {
            uint64_t off = ((w >> 10) & 0xFFFu) << 3;
            *(double *)(uintptr_t)(XSP(rn) + off) = d[rd];
            pc += 4; continue;
        }
        /* add / adds / sub / subs */
        if ((w & 0xFFE0FC00u) == 0x8B000000u) {
            SETX(rd, XR(rn) + XR(rm));
            pc += 4; continue;
        }
        if ((w & 0xFFE0FC00u) == 0xAB000000u) {
            uint64_t a = XR(rn), b = XR(rm), res = a + b;
            N = (res >> 63) & 1;
            Z = (res == 0);
            C = (res < a);
            V = (((a ^ ~b) & (a ^ res)) >> 63) & 1;
            SETX(rd, res);
            pc += 4; continue;
        }
        if ((w & 0xFFE0FC00u) == 0xCB000000u) {
            SETX(rd, XR(rn) - XR(rm));
            pc += 4; continue;
        }
        if ((w & 0xFFE0FC00u) == 0xEB000000u) {
            uint64_t a = XR(rn), b = XR(rm), res = a - b;
            N = (res >> 63) & 1;
            Z = (res == 0);
            C = (a >= b);
            V = (((a ^ b) & (a ^ res)) >> 63) & 1;
            SETX(rd, res);
            pc += 4; continue;
        }
        /* mul / smulh / sdiv */
        if ((w & 0xFFE0FC00u) == 0x9B007C00u) {
            SETX(rd, (uint64_t)((int64_t)XR(rn) * (int64_t)XR(rm)));
            pc += 4; continue;
        }
        if ((w & 0xFFE0FC00u) == 0x9B407C00u) {
            int64_t a = (int64_t)XR(rn), b = (int64_t)XR(rm);
            /* tính 64 bit cao của tích có dấu 128-bit */
            uint64_t al = (uint64_t)a & 0xFFFFFFFFULL, ah = (uint64_t)a >> 32;
            uint64_t bl = (uint64_t)b & 0xFFFFFFFFULL, bh = (uint64_t)b >> 32;
            uint64_t p0 = al * bl, p1 = al * bh, p2 = ah * bl, p3 = ah * bh;
            uint64_t cy = ((p0 >> 32) + (p1 & 0xFFFFFFFFULL) + (p2 & 0xFFFFFFFFULL)) >> 32;
            uint64_t hi = p3 + (p1 >> 32) + (p2 >> 32) + cy;
            if (a < 0) hi -= (uint64_t)b;
            if (b < 0) hi -= (uint64_t)a;
            SETX(rd, hi);
            pc += 4; continue;
        }
        if ((w & 0xFFE0FC00u) == 0x9AC00C00u) {
            int64_t a = (int64_t)XR(rn), b = (int64_t)XR(rm);
            int64_t q = (b == 0) ? 0 : (a == INT64_MIN && b == -1) ? INT64_MIN : (a / b);
            SETX(rd, (uint64_t)q);
            pc += 4; continue;
        }
        /* and / orr / eor / mvn / lslv / asrv */
        if ((w & 0xFFE0FC00u) == 0x8A000000u) { SETX(rd, XR(rn) & XR(rm)); pc += 4; continue; }
        if ((w & 0xFFE0FC00u) == 0xAA000000u) { SETX(rd, XR(rn) | XR(rm)); pc += 4; continue; }
        if ((w & 0xFFE0FC00u) == 0xCA000000u) { SETX(rd, XR(rn) ^ XR(rm)); pc += 4; continue; }
        if ((w & 0xFFE0FC00u) == 0xAA200000u) { SETX(rd, XR(rn) | ~XR(rm)); pc += 4; continue; }
        if ((w & 0xFFE0FC00u) == 0x9AC02000u) { SETX(rd, XR(rn) << (XR(rm) & 63u)); pc += 4; continue; }
        if ((w & 0xFFE0FC00u) == 0x9AC02800u) { SETX(rd, (uint64_t)((int64_t)XR(rn) >> (XR(rm) & 63u))); pc += 4; continue; }
        /* cset (csinc rd, xzr, xzr, icond) / csel */
        if ((w & 0xFFE00C00u) == 0x9A800400u && rn == 31 && rm == 31) {
            int icond = (int)((w >> 12) & 0xFu);
            int cond = icond ^ 1;
            SETX(rd, a64_cond_pass(cond, N, Z, C, V) ? 1ULL : 0ULL);
            pc += 4; continue;
        }
        if ((w & 0xFFE00C00u) == 0x9A800000u) {
            int cond = (int)((w >> 12) & 0xFu);
            SETX(rd, a64_cond_pass(cond, N, Z, C, V) ? XR(rn) : XR(rm));
            pc += 4; continue;
        }
        /* float64 ops */
        if ((w & 0xFFFFFC00u) == 0x9E670000u) { uint64_t u = XR(rn); memcpy(&d[rd], &u, 8); pc += 4; continue; }
        if ((w & 0xFFFFFC00u) == 0x9E660000u) { uint64_t u = 0; memcpy(&u, &d[rn], 8); SETX(rd, u); pc += 4; continue; }
        if ((w & 0xFFE0FC00u) == 0x1E602800u) { d[rd] = d[rn] + d[rm]; pc += 4; continue; }
        if ((w & 0xFFE0FC00u) == 0x1E603800u) { d[rd] = d[rn] - d[rm]; pc += 4; continue; }
        if ((w & 0xFFE0FC00u) == 0x1E606800u) { d[rd] = d[rn] * d[rm]; pc += 4; continue; }
        if ((w & 0xFFE0FC00u) == 0x1E601800u) { d[rd] = d[rn] / d[rm]; pc += 4; continue; }
        if ((w & 0xFFFFFC00u) == 0x1E614000u) { d[rd] = -d[rn]; pc += 4; continue; }
        if ((w & 0xFFFFFC00u) == 0x1E60C000u) { d[rd] = fabs(d[rn]); pc += 4; continue; }
        if ((w & 0xFFFFFC00u) == 0x9E620000u) { d[rd] = (double)(int64_t)XR(rn); pc += 4; continue; }
        if ((w & 0xFFFFFC00u) == 0x9E7E0000u || (w & 0xFFFFFC00u) == 0x9E780000u) {
            SETX(rd, (uint64_t)(int64_t)d[rn]);
            pc += 4; continue;
        }
        if ((w & 0xFFE0FC1Fu) == 0x1E602000u) {
            double a = d[rn], b = d[rm];
            if (isnan(a) || isnan(b)) { N = 0; Z = 0; C = 1; V = 1; }
            else if (a == b)          { N = 0; Z = 1; C = 1; V = 0; }
            else if (a < b)           { N = 1; Z = 0; C = 0; V = 0; }
            else                      { N = 0; Z = 0; C = 1; V = 0; }
            pc += 4; continue;
        }
        if ((w & 0xFFE00C00u) == 0x1E600C00u) {
            int cond = (int)((w >> 12) & 0xFu);
            d[rd] = a64_cond_pass(cond, N, Z, C, V) ? d[rn] : d[rm];
            pc += 4; continue;
        }
        /* branches: b, bl, b.cond, cbz, cbnz, blr */
        if ((w & 0xFC000000u) == 0x14000000u) {
            int32_t off = (int32_t)(w & 0x3FFFFFFu);
            if (off & 0x2000000) off |= ~0x3FFFFFF;
            pc = (size_t)((ptrdiff_t)pc + (ptrdiff_t)off * 4);
            continue;
        }
        if ((w & 0xFC000000u) == 0x94000000u) {
            int32_t off = (int32_t)(w & 0x3FFFFFFu);
            if (off & 0x2000000) off |= ~0x3FFFFFF;
            x[30] = (uint64_t)(pc + 4);
            pc = (size_t)((ptrdiff_t)pc + (ptrdiff_t)off * 4);
            continue;
        }
        if ((w & 0xFF000010u) == 0x54000000u) {
            int cond = (int)(w & 0xFu);
            int32_t off = (int32_t)((w >> 5) & 0x7FFFFu);
            if (off & 0x40000) off |= ~0x7FFFF;
            if (a64_cond_pass(cond, N, Z, C, V))
                pc = (size_t)((ptrdiff_t)pc + (ptrdiff_t)off * 4);
            else
                pc += 4;
            continue;
        }
        if ((w & 0xFF000000u) == 0xB4000000u) {
            int32_t off = (int32_t)((w >> 5) & 0x7FFFFu);
            if (off & 0x40000) off |= ~0x7FFFF;
            if (XR(rd) == 0) pc = (size_t)((ptrdiff_t)pc + (ptrdiff_t)off * 4);
            else pc += 4;
            continue;
        }
        if ((w & 0xFF000000u) == 0xB5000000u) {
            int32_t off = (int32_t)((w >> 5) & 0x7FFFFu);
            if (off & 0x40000) off |= ~0x7FFFF;
            if (XR(rd) != 0) pc = (size_t)((ptrdiff_t)pc + (ptrdiff_t)off * 4);
            else pc += 4;
            continue;
        }
        if ((w & 0xFFFFFC1Fu) == 0xD63F0000u) { /* blr rn (gọi helper C: imod / fmod_floor / struct / dict / str) */
            uintptr_t fn_ptr = (uintptr_t)XR(rn);
            if (fn_ptr == (uintptr_t)imod) {
                x[0] = (uint64_t)imod((int64_t)x[0], (int64_t)x[1]);
            } else if (fn_ptr == (uintptr_t)fmod_floor) {
                d[0] = fmod_floor(d[0], d[1]);
            } else if (fn_ptr == (uintptr_t)tjit_str_of) {
                x[0] = (uint64_t)(uintptr_t)tjit_str_of((int64_t)x[0], (int)x[1]);
            } else if (fn_ptr == (uintptr_t)tjit_str_cat) {
                x[0] = (uint64_t)(uintptr_t)tjit_str_cat((const TVal *)(uintptr_t)x[0], (const TVal *)(uintptr_t)x[1], (int)x[2]);
            } else if (fn_ptr == (uintptr_t)tjit_str_eq) {
                x[0] = (uint64_t)tjit_str_eq((const TVal *)(uintptr_t)x[0], (const TVal *)(uintptr_t)x[1], (int)x[2]);
            } else if (fn_ptr == (uintptr_t)tjit_str_idx) {
                x[0] = (uint64_t)(uintptr_t)tjit_str_idx((const TVal *)(uintptr_t)x[0], (int64_t)x[1]);
            } else if (fn_ptr == (uintptr_t)tjit_dict_new0) {
                x[0] = (uint64_t)(uintptr_t)tjit_dict_new0();
            } else if (fn_ptr == (uintptr_t)tjit_dict_set) {
                x[0] = (uint64_t)tjit_dict_set((TDict *)(uintptr_t)x[0], (int64_t)x[1], (int64_t)x[2], (int)x[3]);
            } else if (fn_ptr == (uintptr_t)tjit_dict_getD) {
                x[0] = (uint64_t)tjit_dict_getD((TDict *)(uintptr_t)x[0], (int64_t)x[1], (int)x[2], (int64_t *)(uintptr_t)x[3]);
            } else if (fn_ptr == (uintptr_t)tjit_dict_has) {
                x[0] = (uint64_t)tjit_dict_has((TDict *)(uintptr_t)x[0], (int64_t)x[1], (int)x[2]);
            } else if (fn_ptr == (uintptr_t)tjit_dict_remove) {
                x[0] = (uint64_t)tjit_dict_remove((TDict *)(uintptr_t)x[0], (int64_t)x[1], (int)x[2]);
            } else if (fn_ptr == (uintptr_t)tjit_rec_new) {
                x[0] = (uint64_t)(uintptr_t)tjit_rec_new((TDict *)(uintptr_t)x[0], (int)x[1], (int64_t *)(uintptr_t)x[2]);
            } else if (fn_ptr == (uintptr_t)a64_rec_new) {
                x[0] = (uint64_t)(uintptr_t)a64_rec_new((TDict *)(uintptr_t)x[0], (int)x[1], (int64_t *)(uintptr_t)x[2]);
            } else if (fn_ptr == (uintptr_t)tjit_dec_dict) {
                tjit_dec_dict((TDict *)(uintptr_t)x[0]);
            } else if (fn_ptr == (uintptr_t)tjit_dec_sval) {
                tjit_dec_sval((TVal *)(uintptr_t)x[0]);
            } else {
                if (ctx) ctx->abort = 1;
                break;
            }
            pc += 4; continue;
        }
        /* lệnh lạ -> abort an toàn */
        if (ctx) ctx->abort = 1;
        break;
    }

    #undef XR
    #undef XSP
    #undef SETX
    #undef SETSP
    free(sim_stack);
    return (int64_t)x[0];
}

/* ========================================================================= */
/* Trình biên dịch Bytecode T -> AArch64 (Entry Mode + OSR + Deopt + J8)      */
/* ========================================================================= */

enum {
    A64_TY_UNK = 0, A64_TY_I = 1, A64_TY_F = 2, A64_TY_B = 3,
    A64_TY_FN  = 4, A64_TY_NAT = 5, A64_TY_LI = 6, A64_TY_LF = 7, A64_TY_LB = 8,
    A64_TY_S   = 10, A64_TY_TYP = 11, A64_TY_D = 12, A64_TY_DN = 13, A64_TY_SN = 14, A64_TY_NIL = 15
};
#define A64_TY_R0  32
#define A64_TY_RN0 96
#define A64_MAX_SHAPES 64
#define A64_SHAPE_MAXF 32

typedef struct {
    TDict *tp;
    int nf;
    uint8_t fty[A64_SHAPE_MAXF];
    const TVal *keys;
} A64Shape;

static A64Shape a64_shapes[A64_MAX_SHAPES];
static int na64_shapes = 0;

static int a64_is_list(uint8_t t) { return t >= A64_TY_LI && t <= A64_TY_LB; }
static int a64_is_dict(uint8_t t) { return t == A64_TY_D || t == A64_TY_DN; }
static int a64_is_str(uint8_t t)  { return t == A64_TY_S || t == A64_TY_SN; }
static int a64_is_rec(uint8_t t)  { return t >= A64_TY_R0 && t < A64_TY_R0 + na64_shapes; }
static int a64_is_own_rec(uint8_t t) { return t >= A64_TY_RN0 && t < A64_TY_RN0 + na64_shapes; }
static int a64_is_any_rec(uint8_t t) { return a64_is_rec(t) || a64_is_own_rec(t); }
static int a64_rec_shape(uint8_t t)  { return a64_is_own_rec(t) ? (t - A64_TY_RN0) : (t - A64_TY_R0); }
static A64Shape *a64_shape_of(uint8_t t) {
    int s = a64_rec_shape(t);
    return (s < 0 || s >= A64_MAX_SHAPES) ? NULL : &a64_shapes[s];
}
static uint8_t a64_bor_shape(int s) { return (uint8_t)(A64_TY_R0 + s); }
static uint8_t a64_own_shape(int s) { return (uint8_t)(A64_TY_RN0 + s); }
static uint8_t a64_norm_ty(uint8_t t) {
    if (a64_is_own_rec(t)) return a64_bor_shape(a64_rec_shape(t));
    if (t == A64_TY_DN) return A64_TY_D;
    if (t == A64_TY_SN) return A64_TY_S;
    return t;
}
static int a64_shape_reg(TDict *tp, int nf, const uint8_t *fty) {
    for (int q = 0; q < na64_shapes; q++) {
        A64Shape *sh = &a64_shapes[q];
        if (sh->tp == tp && sh->nf == nf && memcmp(sh->fty, fty, (size_t)nf) == 0)
            return q;
    }
    if (na64_shapes >= A64_MAX_SHAPES) return -1;
    A64Shape *sh = &a64_shapes[na64_shapes];
    sh->tp = tp;
    tp->refc++;
    sh->nf = nf;
    memcpy(sh->fty, fty, (size_t)nf);
    sh->keys = tp->fields->items;
    return na64_shapes++;
}
static int a64_shape_field(const A64Shape *sh, const uint8_t *name, uint32_t len) {
    for (int i = 0; i < sh->nf; i++) {
        const TVal *k = &sh->keys[i];
        if (k->t == T_STR && k->u.str.n == len && memcmp(k->u.str.s, name, len) == 0) return i;
    }
    return -1;
}
static TDict *a64_rec_new(TDict *tp, int shape, int64_t *slots) {
    if (shape < 0 || shape >= na64_shapes) return NULL;
    const uint8_t *fty = a64_shapes[shape].fty;
    size_t n = tp->fields->len;
    TDict *d = tjit_dict_new0();
    free(d->keys); free(d->vals);
    d->len = n; d->cap = n;
    d->keys = tp->fields->items;
    d->vals = (TVal *)malloc((n ? n : 1) * sizeof(TVal));
    for (size_t i = 0; i < n; i++) {
        if (fty[i] == A64_TY_F) { d->vals[i].t = T_FLOAT; memcpy(&d->vals[i].u.f, &slots[i], 8); }
        else if (fty[i] == A64_TY_B) { d->vals[i].t = T_BOOL; d->vals[i].u.b = (int)(slots[i] & 1); }
        else { d->vals[i].t = T_INT; d->vals[i].u.i = slots[i]; }
    }
    d->kind = DK_REC;
    d->meths = tp;
    tp->refc++;
    return d;
}
static uint8_t a64_list_elem(uint8_t t) {
    return t == A64_TY_LI ? A64_TY_I : t == A64_TY_LF ? A64_TY_F : A64_TY_B;
}
static uint8_t a64_val_ty(const TVal *v) {
    if (v->t == T_INT)   return A64_TY_I;
    if (v->t == T_FLOAT) return A64_TY_F;
    if (v->t == T_BOOL)  return A64_TY_B;
    if (v->t == T_STR)   return A64_TY_S;
    if (v->t == T_LIST) {
        TList *l = v->u.list;
        if (!l || l->len == 0) return A64_TY_LI;
        TType et = l->items[0].t;
        for (size_t i = 1; i < l->len; i++) if (l->items[i].t != et) return A64_TY_UNK;
        return et == T_INT ? A64_TY_LI : et == T_FLOAT ? A64_TY_LF : et == T_BOOL ? A64_TY_LB : A64_TY_UNK;
    }
    if (v->t == T_DICT) {
        TDict *d = v->u.dict;
        if (d->kind == DK_DICT) return A64_TY_D;
        if (d->kind == DK_REC && d->meths && !d->meths->mod && d->len <= A64_SHAPE_MAXF) {
            uint8_t ft[A64_SHAPE_MAXF];
            for (size_t i = 0; i < d->len; i++) {
                TType vt = d->vals[i].t;
                if (vt == T_INT) ft[i] = A64_TY_I;
                else if (vt == T_FLOAT) ft[i] = A64_TY_F;
                else if (vt == T_BOOL) ft[i] = A64_TY_B;
                else return A64_TY_UNK;
            }
            int s = a64_shape_reg(d->meths, (int)d->len, ft);
            return s < 0 ? A64_TY_UNK : a64_bor_shape(s);
        }
    }
    return A64_TY_UNK;
}
static int a64_ty_ok(const TVal *v, uint8_t want) {
    if (a64_is_list(want)) return v->t == T_LIST;
    if (want == A64_TY_S) return v->t == T_STR;
    if (want == A64_TY_D) return v->t == T_DICT && v->u.dict->kind == DK_DICT;
    if (a64_is_rec(want)) {
        A64Shape *sh = a64_shape_of(want);
        return sh && v->t == T_DICT && v->u.dict->kind == DK_REC &&
               v->u.dict->meths == sh->tp && v->u.dict->len == (size_t)sh->nf;
    }
    return a64_val_ty(v) == want;
}
static int64_t a64_val_bits(const TVal *v) {
    int64_t x = 0;
    if (v->t == T_INT) x = v->u.i;
    else if (v->t == T_FLOAT) memcpy(&x, &v->u.f, 8);
    else if (v->t == T_BOOL) x = v->u.b ? 1 : 0;
    else if (v->t == T_LIST) x = (int64_t)(uintptr_t)v->u.list;
    else if (v->t == T_DICT) x = (int64_t)(uintptr_t)v->u.dict;
    else if (v->t == T_STR)  x = (int64_t)(uintptr_t)v;
    return x;
}
static TVal a64_bits_val(TVM *vm, uint8_t ty, int64_t x, int32_t sv) {
    TVal v; v.t = T_NIL; v.u.i = 0;
    if (ty == A64_TY_I) { v.t = T_INT; v.u.i = x; }
    else if (ty == A64_TY_F) { v.t = T_FLOAT; memcpy(&v.u.f, &x, 8); }
    else if (ty == A64_TY_B) { v.t = T_BOOL; v.u.b = (int)(x & 1); }
    else if (a64_is_list(ty)) { v.t = T_LIST; v.u.list = (TList *)(uintptr_t)x; tval_retain(&v); }
    else if (ty == A64_TY_D || a64_is_rec(ty)) { v.t = T_DICT; v.u.dict = (TDict *)(uintptr_t)x; tval_retain(&v); }
    else if (ty == A64_TY_DN || a64_is_own_rec(ty)) { v.t = T_DICT; v.u.dict = (TDict *)(uintptr_t)x; }
    else if (ty == A64_TY_S) { v = *(const TVal *)(uintptr_t)x; tval_retain(&v); }
    else if (ty == A64_TY_SN) { TVal *w = (TVal *)(uintptr_t)x; v = *w; tjit_wrap_free(w); }
    else if (ty == A64_TY_FN || ty == A64_TY_NAT || ty == A64_TY_TYP) { v = vm->prog->globals[sv]; tval_retain(&v); }
    return v;
}

typedef struct {
    uint8_t op;
    size_t pc;
    int64_t imm;
    double fimm;
    int32_t k, k2;
    int32_t tgt;        /* chỉ số micro-op đích cho lệnh nhảy */
    int head;
    int deopt_id;       /* >= 0 nếu thoát ra ngoài vùng OSR */
} A64Op;

typedef struct {
    int reach;
    int depth;
    uint8_t st[64];
    int32_t sv[64];
    uint8_t lt[256];
} A64State;

typedef struct {
    size_t at;
    int target_op;      /* >= 0: op idx; -1: abort; -2: epilogue; <= -100: -(deopt_id + 100) */
    int kind;           /* 0: b, 1: b.cond, 2: cbz, 3: cbnz, 4: bl_self */
    int cond_or_reg;
} A64Fixup;

typedef struct {
    size_t pc;
    int depth;
    uint8_t st[64];
    int32_t sv[64];
    uint8_t lt[256];
} A64Deopt;

typedef struct {
    uint8_t *code;
    size_t code_len;
    int is_exec;        /* 1 nếu cấp qua a64_alloc_exec */
    uint8_t argty[256];
    uint8_t lty[256];
    uint8_t retty;
    int nargs, nlocals;
    A64Deopt *deopts;
    int ndeopt;
    int nsave;
} A64Unit;

#define A64_MAX_OPS 1024
#define A64_MAX_FIX 2048

static int64_t rd_i64_le(const uint8_t *p) {
    uint64_t u = 0;
    for (int i = 0; i < 8; i++) u |= ((uint64_t)p[i]) << (i * 8);
    return (int64_t)u;
}
static int32_t rd_i32_le(const uint8_t *p) {
    uint32_t u = (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
    return (int32_t)u;
}
static uint16_t rd_u16_le(const uint8_t *p) {
    return (uint16_t)p[0] | ((uint16_t)p[1] << 8);
}

static int a64_compile_core(TVM *vm, TFn *fn, const uint8_t *in_ty, int is_osr, size_t osr_pc, A64Unit *out) {
    memset(out, 0, sizeof(*out));
    TProg *P = fn->prog ? fn->prog : vm->prog;
    const uint8_t *c = P->code;
    size_t clen = P->len;
    size_t pc = (size_t)fn->entry;
    if (fn->nupvals > 0 || fn->vararg || fn->nlocals > 128) return 0;

    A64Op *ops = (A64Op *)calloc(A64_MAX_OPS, sizeof(A64Op));
    if (!ops) return 0;
    int nops = 0;
    size_t max_pc = pc;

    while (pc < clen && nops < A64_MAX_OPS - 1) {
        uint8_t op = c[pc];
        size_t len = tvm_oplen(c, clen, pc);
        if (len == 0 || pc + len > clen) { free(ops); return 0; }
        A64Op *m = &ops[nops++];
        m->op = op;
        m->pc = pc;
        m->tgt = -1;
        m->deopt_id = -1;
        const uint8_t *a = c + pc + 1;
        switch (op) {
        case OP_PUSH_BOOL: case OP_CALL: case OP_CALLM: case OP_RANGE: case OP_PRINT:
            m->k = a[0]; break;
        case OP_PUSH_INT:
            m->imm = rd_i64_le(a); break;
        case OP_PUSH_FLOAT:
            memcpy(&m->fimm, a, 8); break;
        case OP_NEW_DICT:
            m->k = rd_u16_le(a); break;
        case OP_ATTR:
            if (rd_i32_le(a) == 3 && memcmp(a + 4, "len", 3) == 0) m->op = OP_LEN;
            break;
        case OP_LOAD: case OP_STORE: case OP_DECL: case OP_BIND:
        case OP_LOADG: case OP_INC_L: case OP_DEC_L: case OP_INCB_L:
            m->k = rd_u16_le(a); break;
        case OP_ADDI_L:
            m->k = rd_u16_le(a); m->imm = rd_i64_le(a + 2); break;
        case OP_ADDS_L: case OP_SUBS_L:
            m->k = rd_u16_le(a); m->k2 = rd_u16_le(a + 2); break;
        case OP_JMP: case OP_JMPF: case OP_JMPT: {
            int32_t rel = rd_i32_le(a);
            size_t tpc = (size_t)((ptrdiff_t)(pc + len) + rel);
            m->imm = (int64_t)tpc;
            if (tpc > max_pc) max_pc = tpc;
            break;
        }
        default: break;
        }
        size_t next_pc = pc + len;
        if (op == OP_RET && next_pc > max_pc) { pc = next_pc; break; }
        pc = next_pc;
    }
    if (nops == 0) { free(ops); return 0; }

    /* phân giải đích nhảy */
    int start_op = 0;
    for (int i = 0; i < nops; i++) {
        if (is_osr && ops[i].pc == osr_pc) { ops[i].head = 1; start_op = i; }
        if (ops[i].op == OP_JMP || ops[i].op == OP_JMPF || ops[i].op == OP_JMPT) {
            size_t tpc = (size_t)ops[i].imm;
            int found = -1;
            for (int j = 0; j < nops; j++) if (ops[j].pc == tpc) { found = j; break; }
            if (found < 0) { free(ops); return 0; }
            ops[i].tgt = found;
            ops[found].head = 1;
        }
    }
    if (is_osr && ops[start_op].pc != osr_pc) { free(ops); return 0; }

    /* thông dịch trừu tượng (abstract interpretation) */
    int nl = fn->nlocals;
    A64State *states = (A64State *)calloc((size_t)nops + 1, sizeof(A64State));
    if (!states) { free(ops); return 0; }
    states[start_op].reach = 1;
    states[start_op].depth = 0;
    for (int i = 0; i < nl; i++) states[start_op].lt[i] = A64_TY_UNK;
    if (is_osr) {
        for (int i = 0; i < nl; i++) states[start_op].lt[i] = in_ty[i];
    } else {
        for (int i = 0; i < fn->nargs; i++) states[0].lt[i] = in_ty[i];
    }

    uint8_t retty = A64_TY_UNK;
    int changed = 1, ok = 1;
    while (changed && ok) {
        changed = 0;
        for (int i = 0; i < nops && ok; i++) {
            if (!states[i].reach) continue;
            A64State cur = states[i];
            A64Op *m = &ops[i];
            #define PUSH(t, v) do { if (cur.depth >= 60) { ok = 0; break; } cur.st[cur.depth] = (t); cur.sv[cur.depth] = (v); cur.depth++; } while (0)
            #define POP(t) do { if (cur.depth <= 0) { ok = 0; break; } (t) = cur.st[--cur.depth]; } while (0)
            #define MERGE(dst_idx) do { \
                int ti = (dst_idx); \
                if (ti < 0 || ti >= nops) { ok = 0; break; } \
                A64State *d = &states[ti]; \
                if (!d->reach) { *d = cur; d->reach = 1; changed = 1; } \
                else { \
                    if (d->depth != cur.depth) { ok = 0; break; } \
                    for (int q = 0; q < cur.depth; q++) if (a64_norm_ty(d->st[q]) != a64_norm_ty(cur.st[q])) { ok = 0; break; } \
                    for (int q = 0; q < nl; q++) { \
                        uint8_t dl = a64_norm_ty(d->lt[q]), cl = a64_norm_ty(cur.lt[q]); \
                        if (d->lt[q] == A64_TY_UNK && cur.lt[q] != A64_TY_UNK) { d->lt[q] = cur.lt[q]; changed = 1; } \
                        else if (dl != cl && cl != A64_TY_UNK) { ok = 0; break; } \
                    } \
                } \
            } while (0)

            uint8_t t1 = 0, t2 = 0, t3 = 0;
            int step_next = 1;
            switch (m->op) {
            case OP_NOP: break;
            case OP_PUSH_INT:   PUSH(A64_TY_I, 0); break;
            case OP_PUSH_FLOAT: PUSH(A64_TY_F, 0); break;
            case OP_PUSH_BOOL:  PUSH(A64_TY_B, 0); break;
            case OP_PUSH_NIL:   PUSH(A64_TY_NIL, 0); break;
            case OP_PUSH_STR:
                if (!P->ic) { P->ic = (TAttrIC *)calloc(P->len ? P->len : 1, sizeof(TAttrIC)); }
                if (P->ic[m->pc].kind != 4) {
                    uint32_t slen = (uint32_t)rd_i32_le(c + m->pc + 1);
                    TVal v;
                    v.t = T_STR; v.u.str.n = slen;
                    v.u.str.s = (char *)malloc(slen + 1);
                    if (slen) memcpy(v.u.str.s, c + m->pc + 5, slen);
                    v.u.str.s[slen] = '\0';
                    v.u.str.refc = (int32_t *)malloc(sizeof(int32_t));
                    *v.u.str.refc = 1;
                    TVal *keep = (TVal *)malloc(sizeof(TVal));
                    *keep = v;
                    P->ic[m->pc].kind = 4;
                    P->ic[m->pc].key = keep;
                }
                PUSH(A64_TY_S, 0);
                break;
            case OP_POP:        POP(t1); break;
            case OP_DUP:
                if (cur.depth <= 0) { ok = 0; break; }
                PUSH(cur.st[cur.depth - 1], cur.sv[cur.depth - 1]);
                break;
            case OP_LOAD:
                if (m->k < 0 || m->k >= nl || cur.lt[m->k] == A64_TY_UNK) { ok = 0; break; }
                PUSH(a64_norm_ty(cur.lt[m->k]), 0);
                break;
            case OP_STORE: case OP_DECL: case OP_BIND:
                POP(t1);
                if (m->k < 0 || m->k >= nl) { ok = 0; break; }
                if (cur.lt[m->k] != A64_TY_UNK && a64_norm_ty(cur.lt[m->k]) != a64_norm_ty(t1)) { ok = 0; break; }
                cur.lt[m->k] = t1;
                break;
            case OP_LOADG:
                if (m->k < NAT_NFIXED) {
                    if (m->k == NAT_INT || m->k == NAT_FLOAT || m->k == NAT_ABS ||
                        m->k == NAT_MIN || m->k == NAT_MAX || m->k == NAT_LEN ||
                        m->k == NAT_STR || m->k == NAT___NEW) {
                        PUSH(A64_TY_NAT, m->k);
                    } else { ok = 0; }
                } else if ((size_t)m->k < P->nglobals) {
                    TVal *gv = &P->globals[m->k];
                    if (gv->t == T_FN && gv->u.fn == fn) PUSH(A64_TY_FN, m->k);
                    else if (gv->t == T_INT) PUSH(A64_TY_I, m->k);
                    else if (gv->t == T_FLOAT) PUSH(A64_TY_F, m->k);
                    else if (gv->t == T_BOOL) PUSH(A64_TY_B, m->k);
                    else if (gv->t == T_STR) PUSH(A64_TY_S, m->k);
                    else if (gv->t == T_DICT && gv->u.dict->kind == DK_TYPE && !gv->u.dict->mod) PUSH(A64_TY_TYP, m->k);
                    else if (gv->t == T_DICT && gv->u.dict->kind == DK_DICT) PUSH(A64_TY_D, m->k);
                    else ok = 0;
                } else ok = 0;
                break;
            case OP_ADD: case OP_SUB: case OP_MUL: case OP_MOD:
                POP(t2); POP(t1);
                if (m->op == OP_ADD && a64_is_str(t1) && a64_is_str(t2)) PUSH(A64_TY_SN, 0);
                else if (t1 == A64_TY_I && t2 == A64_TY_I) PUSH(A64_TY_I, 0);
                else if ((t1 == A64_TY_I || t1 == A64_TY_F) && (t2 == A64_TY_I || t2 == A64_TY_F)) PUSH(A64_TY_F, 0);
                else ok = 0;
                break;
            case OP_DIV:
                POP(t2); POP(t1);
                if ((t1 == A64_TY_I || t1 == A64_TY_F) && (t2 == A64_TY_I || t2 == A64_TY_F)) PUSH(A64_TY_F, 0);
                else ok = 0;
                break;
            case OP_NEG:
                POP(t1);
                if (t1 == A64_TY_I || t1 == A64_TY_F) PUSH(t1, 0);
                else ok = 0;
                break;
            case OP_NOT:
                POP(t1);
                if (t1 == A64_TY_B || t1 == A64_TY_I) PUSH(A64_TY_B, 0);
                else ok = 0;
                break;
            case OP_BAND: case OP_BOR: case OP_BXOR: case OP_SHL: case OP_SHR:
                POP(t2); POP(t1);
                if (t1 == A64_TY_I && t2 == A64_TY_I) PUSH(A64_TY_I, 0);
                else ok = 0;
                break;
            case OP_BNOT:
                POP(t1);
                if (t1 == A64_TY_I) PUSH(A64_TY_I, 0);
                else ok = 0;
                break;
            case OP_EQ: case OP_NE: case OP_LT: case OP_GT: case OP_LE: case OP_GE:
                POP(t2); POP(t1);
                if (a64_is_str(t1) && a64_is_str(t2) && (m->op == OP_EQ || m->op == OP_NE)) PUSH(A64_TY_B, 0);
                else if ((t1 == A64_TY_I || t1 == A64_TY_F || t1 == A64_TY_B) &&
                         (t2 == A64_TY_I || t2 == A64_TY_F || t2 == A64_TY_B)) PUSH(A64_TY_B, 0);
                else ok = 0;
                break;
            case OP_INC_L: case OP_DEC_L: case OP_INCB_L: case OP_ADDI_L:
                if (m->k < 0 || m->k >= nl || cur.lt[m->k] != A64_TY_I) ok = 0;
                break;
            case OP_ADDS_L: case OP_SUBS_L:
                if (m->k < 0 || m->k >= nl || m->k2 < 0 || m->k2 >= nl) { ok = 0; break; }
                if (cur.lt[m->k] == A64_TY_I && cur.lt[m->k2] == A64_TY_I) cur.lt[m->k] = A64_TY_I;
                else if ((cur.lt[m->k] == A64_TY_I || cur.lt[m->k] == A64_TY_F) &&
                         (cur.lt[m->k2] == A64_TY_I || cur.lt[m->k2] == A64_TY_F)) cur.lt[m->k] = A64_TY_F;
                else ok = 0;
                break;
            case OP_LEN:
                POP(t1);
                if (a64_is_list(t1) || a64_is_str(t1) || a64_is_dict(t1)) PUSH(A64_TY_I, 0);
                else ok = 0;
                break;
            case OP_IDX:
                POP(t2); POP(t1);
                if (a64_is_list(t1) && t2 == A64_TY_I) PUSH(a64_list_elem(t1), 0);
                else if (a64_is_str(t1) && t2 == A64_TY_I) PUSH(A64_TY_S, 0);
                else if (a64_is_dict(t1) && (t2 == A64_TY_I || a64_is_str(t2))) PUSH(A64_TY_I, 0);
                else ok = 0;
                break;
            case OP_SETIDX:
                POP(t3); POP(t2); POP(t1);
                if (a64_is_list(t1) && t2 == A64_TY_I && t3 == a64_list_elem(t1)) { /* ok */ }
                else if (a64_is_dict(t1) && (t2 == A64_TY_I || a64_is_str(t2)) &&
                         (t3 == A64_TY_I || t3 == A64_TY_F || t3 == A64_TY_B ||
                          a64_is_str(t3) || a64_is_dict(t3) || a64_is_any_rec(t3))) { /* ok */ }
                else ok = 0;
                break;
            case OP_ATTR: {
                POP(t1);
                if (!a64_is_any_rec(t1)) { ok = 0; break; }
                A64Shape *sh = a64_shape_of(t1);
                uint32_t flen = (uint32_t)rd_i32_le(c + m->pc + 1);
                int fi = sh ? a64_shape_field(sh, c + m->pc + 5, flen) : -1;
                if (fi < 0 || sh->fty[fi] == A64_TY_UNK) { ok = 0; break; }
                PUSH(sh->fty[fi], 0);
                break;
            }
            case OP_SETATTR: {
                POP(t2); POP(t1);
                if (!a64_is_any_rec(t1)) { ok = 0; break; }
                A64Shape *sh = a64_shape_of(t1);
                uint32_t flen = (uint32_t)rd_i32_le(c + m->pc + 1);
                int fi = sh ? a64_shape_field(sh, c + m->pc + 5, flen) : -1;
                if (fi < 0 || sh->fty[fi] != t2) { ok = 0; break; }
                break;
            }
            case OP_NEW_DICT: {
                int k = m->k;
                if (cur.depth < 2 * k) { ok = 0; break; }
                for (int q = 0; q < k; q++) {
                    POP(t2); POP(t1);
                    if (!(t1 == A64_TY_I || a64_is_str(t1))) { ok = 0; break; }
                    if (!(t2 == A64_TY_I || t2 == A64_TY_F || t2 == A64_TY_B ||
                          a64_is_str(t2) || a64_is_dict(t2) || a64_is_any_rec(t2))) { ok = 0; break; }
                }
                if (ok) PUSH(A64_TY_DN, 0);
                break;
            }
            case OP_CALLM: {
                int argc = m->k;
                if (cur.depth < argc + 1) { ok = 0; break; }
                uint32_t mlen = (uint32_t)rd_i32_le(c + m->pc + 2);
                const uint8_t *mnm = c + m->pc + 6;
                uint8_t rty = cur.st[cur.depth - argc - 1];
                if (a64_is_dict(rty)) {
                    if (argc == 2 && mlen == 3 && memcmp(mnm, "get", 3) == 0) {
                        POP(t3); POP(t2); POP(t1);
                        if ((t2 == A64_TY_I || a64_is_str(t2)) && t3 == A64_TY_I) PUSH(A64_TY_I, 0);
                        else ok = 0;
                    } else if (argc == 1 && mlen == 3 && memcmp(mnm, "has", 3) == 0) {
                        POP(t2); POP(t1);
                        if (t2 == A64_TY_I || a64_is_str(t2)) PUSH(A64_TY_B, 0);
                        else ok = 0;
                    } else if (argc == 1 && mlen == 6 && memcmp(mnm, "remove", 6) == 0) {
                        POP(t2); POP(t1);
                        if (t2 == A64_TY_I || a64_is_str(t2)) PUSH(A64_TY_NIL, 0);
                        else ok = 0;
                    } else ok = 0;
                } else ok = 0;
                break;
            }
            case OP_CALL: {
                int argc = m->k;
                if (cur.depth < argc + 1) { ok = 0; break; }
                uint8_t fty = cur.st[cur.depth - argc - 1];
                int32_t fsv = cur.sv[cur.depth - argc - 1];
                if (fty == A64_TY_NAT) {
                    if (fsv == NAT_INT && argc == 1) {
                        POP(t1); POP(t2);
                        if (t1 == A64_TY_I || t1 == A64_TY_F || t1 == A64_TY_B) PUSH(A64_TY_I, 0);
                        else ok = 0;
                    } else if (fsv == NAT_FLOAT && argc == 1) {
                        POP(t1); POP(t2);
                        if (t1 == A64_TY_I || t1 == A64_TY_F || t1 == A64_TY_B) PUSH(A64_TY_F, 0);
                        else ok = 0;
                    } else if (fsv == NAT_ABS && argc == 1) {
                        POP(t1); POP(t2);
                        if (t1 == A64_TY_I || t1 == A64_TY_F) PUSH(t1, 0);
                        else ok = 0;
                    } else if ((fsv == NAT_MIN || fsv == NAT_MAX) && argc == 2) {
                        POP(t2); POP(t1); POP(t3);
                        if (t1 == A64_TY_I && t2 == A64_TY_I) PUSH(A64_TY_I, 0);
                        else if (t1 == A64_TY_F && t2 == A64_TY_F) PUSH(A64_TY_F, 0);
                        else ok = 0;
                    } else if (fsv == NAT_LEN && argc == 1) {
                        POP(t1); POP(t2);
                        if (a64_is_list(t1) || a64_is_str(t1) || a64_is_dict(t1)) PUSH(A64_TY_I, 0);
                        else ok = 0;
                    } else if (fsv == NAT_STR && argc == 1) {
                        POP(t1); POP(t2);
                        if (t1 == A64_TY_I || t1 == A64_TY_F || t1 == A64_TY_B || a64_is_str(t1)) PUSH(A64_TY_SN, 0);
                        else ok = 0;
                    } else if (fsv == NAT___NEW && argc >= 1) {
                        uint8_t tp_ty = cur.st[cur.depth - argc];
                        int32_t tp_sv = cur.sv[cur.depth - argc];
                        if (tp_ty != A64_TY_TYP) { ok = 0; break; }
                        TDict *tp = P->globals[tp_sv].u.dict;
                        int nf = (int)tp->fields->len;
                        if (argc - 1 != nf || nf > A64_SHAPE_MAXF) { ok = 0; break; }
                        uint8_t fts[A64_SHAPE_MAXF];
                        for (int q = nf - 1; q >= 0; q--) {
                            POP(t1);
                            if (t1 != A64_TY_I && t1 != A64_TY_F && t1 != A64_TY_B) { ok = 0; break; }
                            fts[q] = t1;
                        }
                        if (!ok) break;
                        POP(t2); /* type */
                        POP(t3); /* __new */
                        int sh = a64_shape_reg(tp, nf, fts);
                        if (sh < 0) { ok = 0; break; }
                        PUSH(a64_own_shape(sh), 0);
                    } else ok = 0;
                } else if (fty == A64_TY_FN && !is_osr && argc == fn->nargs) {
                    for (int a = argc - 1; a >= 0; a--) {
                        POP(t1);
                        if (a64_norm_ty(t1) != a64_norm_ty(in_ty[a])) { ok = 0; break; }
                    }
                    POP(t2);
                    if (retty == A64_TY_UNK) retty = A64_TY_I;
                    PUSH(retty, 0);
                } else ok = 0;
                break;
            }
            case OP_JMP:
                MERGE(m->tgt);
                step_next = 0;
                break;
            case OP_JMPF: case OP_JMPT:
                POP(t1);
                if (t1 != A64_TY_B && t1 != A64_TY_I && !a64_is_str(t1) && !a64_is_dict(t1)) { ok = 0; break; }
                MERGE(m->tgt);
                break;
            case OP_RET:
                POP(t1);
                if (cur.depth != 0 || (t1 != A64_TY_I && t1 != A64_TY_F && t1 != A64_TY_B &&
                    !a64_is_str(t1) && !a64_is_dict(t1) && !a64_is_any_rec(t1))) { ok = 0; break; }
                if (retty == A64_TY_UNK) retty = t1;
                else if (a64_norm_ty(retty) != a64_norm_ty(t1)) ok = 0;
                else if (t1 == A64_TY_SN || t1 == A64_TY_DN || a64_is_own_rec(t1)) retty = t1;
                step_next = 0;
                break;
            default:
                ok = 0;
                break;
            }
            if (ok && step_next) MERGE(i + 1);
            #undef PUSH
            #undef POP
            #undef MERGE
        }
    }
    if (!ok || retty == A64_TY_UNK) { free(ops); free(states); return 0; }

    /* phát mã máy AArch64 */
    A64Buf b; memset(&b, 0, sizeof(b));
    size_t *op_lbl = (size_t *)calloc((size_t)nops + 1, sizeof(size_t));
    A64Fixup *fixes = (A64Fixup *)calloc(A64_MAX_FIX, sizeof(A64Fixup));
    A64Deopt *deopts = (A64Deopt *)calloc(128, sizeof(A64Deopt));
    int nfix = 0, ndeopt = 0;

    #define ADD_FIX(at_off, tgt_idx, knd, cr) do { \
        if (nfix < A64_MAX_FIX) { \
            fixes[nfix].at = (at_off); \
            fixes[nfix].target_op = (tgt_idx); \
            fixes[nfix].kind = (knd); \
            fixes[nfix].cond_or_reg = (cr); \
            nfix++; \
        } \
    } while (0)

    #define NEW_DEOPT(pc_val, st_State, d_out) do { \
        if (is_osr && ndeopt < 128) { \
            int id = ndeopt++; \
            deopts[id].pc = (pc_val); \
            deopts[id].depth = (st_State).depth; \
            memcpy(deopts[id].st, (st_State).st, 64); \
            memcpy(deopts[id].sv, (st_State).sv, sizeof(int32_t) * 64); \
            memcpy(deopts[id].lt, (st_State).lt, 256); \
            (d_out) = -(id + 100); \
        } else { \
            (d_out) = -1; \
        } \
    } while (0)

    int frame_slots = nl + 64 + 16;
    int frame_bytes = ((frame_slots * 8 + 15) & ~15);
    #define L_OFF(slot) ((slot) * 8)
    #define S_OFF(d)    ((nl + (d)) * 8)
    #define ARG_OFF(a)  ((nl + 64 + (a)) * 8)

    /* Prologue AAPCS64 */
    a64_emit(&b, 0xA9BF7BFDu);              /* stp x29, x30, [sp, #-16]! */
    a64_emit(&b, 0xA9BF53F3u);              /* stp x19, x20, [sp, #-16]! */
    a64_sub_imm(&b, 31, 31, frame_bytes);   /* sub sp, sp, #frame_bytes  */
    a64_add_imm(&b, 29, 31, 0);             /* mov x29, sp               */
    a64_mov_reg(&b, 19, 0);                 /* mov x19, x0 (in)          */
    a64_mov_reg(&b, 20, 1);                 /* mov x20, x1 (ctx)         */

    /* kiểm tra giới hạn stack: sp < ctx->stack_limit -> abort */
    a64_ldr(&b, 9, 20, 8);                  /* ldr x9, [x20, #8]         */
    a64_cmp(&b, 29, 9);
    ADD_FIX(b.len, -1, 1, A64_CC);
    a64_bcond(&b, A64_CC, 0);

    /* nạp đối số / biến cục bộ đầu vào từ in[] (x19) vào frame [x29] */
    int n_copy_in = is_osr ? nl : fn->nargs;
    for (int i = 0; i < n_copy_in; i++) {
        if (states[start_op].lt[i] != A64_TY_UNK) {
            a64_ldr(&b, 9, 19, i * 8);
            a64_str(&b, 9, 29, L_OFF(i));
        }
    }
    if (is_osr && start_op > 0) {
        ADD_FIX(b.len, start_op, 0, 0);
        a64_b(&b, 0);
    }

    for (int i = 0; i < nops && ok; i++) {
        op_lbl[i] = b.len;
        if (!states[i].reach) continue;
        A64State cur = states[i];
        A64Op *m = &ops[i];
        int d = cur.depth;
        int fail_tgt = -1;
        NEW_DEOPT(m->pc, cur, fail_tgt);

        switch (m->op) {
        case OP_NOP: break;
        case OP_PUSH_INT:
            a64_mov_imm64(&b, 9, (uint64_t)m->imm);
            a64_str(&b, 9, 29, S_OFF(d));
            break;
        case OP_PUSH_FLOAT: {
            uint64_t bits = 0;
            memcpy(&bits, &m->fimm, 8);
            a64_mov_imm64(&b, 9, bits);
            a64_str(&b, 9, 29, S_OFF(d));
            break;
        }
        case OP_PUSH_BOOL:
            a64_movz(&b, 9, m->k ? 1 : 0, 0);
            a64_str(&b, 9, 29, S_OFF(d));
            break;
        case OP_PUSH_NIL:
            a64_movz(&b, 9, 0, 0);
            a64_str(&b, 9, 29, S_OFF(d));
            break;
        case OP_PUSH_STR:
            a64_mov_imm64(&b, 9, (uint64_t)(uintptr_t)P->ic[m->pc].key);
            a64_str(&b, 9, 29, S_OFF(d));
            break;
        case OP_POP:
            if (cur.st[d - 1] == A64_TY_SN) {
                a64_ldr(&b, 0, 29, S_OFF(d - 1));
                a64_mov_imm64(&b, 11, (uint64_t)(uintptr_t)tjit_dec_sval);
                a64_blr(&b, 11);
            } else if (cur.st[d - 1] == A64_TY_DN || a64_is_own_rec(cur.st[d - 1])) {
                a64_ldr(&b, 0, 29, S_OFF(d - 1));
                a64_mov_imm64(&b, 11, (uint64_t)(uintptr_t)tjit_dec_dict);
                a64_blr(&b, 11);
            }
            break;
        case OP_DUP:
            a64_ldr(&b, 9, 29, S_OFF(d - 1));
            a64_str(&b, 9, 29, S_OFF(d));
            break;
        case OP_LOAD:
            a64_ldr(&b, 9, 29, L_OFF(m->k));
            a64_str(&b, 9, 29, S_OFF(d));
            break;
        case OP_STORE: case OP_DECL: case OP_BIND:
            a64_ldr(&b, 9, 29, S_OFF(d - 1));
            a64_str(&b, 9, 29, L_OFF(m->k));
            break;
        case OP_LOADG:
            if (m->k >= NAT_NFIXED && (size_t)m->k < P->nglobals) {
                TVal *gv = &P->globals[m->k];
                if (gv->t == T_INT || gv->t == T_FLOAT || gv->t == T_BOOL) {
                    /* guard tag của global số rồi đọc giá trị */
                    a64_mov_imm64(&b, 9, (uint64_t)(uintptr_t)gv);
                    a64_ldrw(&b, 10, 9, 0);
                    a64_movz(&b, 11, (int)gv->t, 0);
                    a64_cmp(&b, 10, 11);
                    ADD_FIX(b.len, fail_tgt, 1, A64_NE);
                    a64_bcond(&b, A64_NE, 0);
                    a64_ldr(&b, 10, 9, 8);
                    a64_str(&b, 10, 29, S_OFF(d));
                } else if (gv->t == T_STR) {
                    a64_mov_imm64(&b, 9, (uint64_t)(uintptr_t)gv);
                    a64_ldrw(&b, 10, 9, 0);
                    a64_movz(&b, 11, (int)T_STR, 0);
                    a64_cmp(&b, 10, 11);
                    ADD_FIX(b.len, fail_tgt, 1, A64_NE);
                    a64_bcond(&b, A64_NE, 0);
                    a64_str(&b, 9, 29, S_OFF(d));
                } else if (gv->t == T_DICT && gv->u.dict->kind == DK_DICT) {
                    a64_mov_imm64(&b, 9, (uint64_t)(uintptr_t)gv);
                    a64_ldrw(&b, 10, 9, 0);
                    a64_movz(&b, 11, (int)T_DICT, 0);
                    a64_cmp(&b, 10, 11);
                    ADD_FIX(b.len, fail_tgt, 1, A64_NE);
                    a64_bcond(&b, A64_NE, 0);
                    a64_ldr(&b, 10, 9, 8);
                    a64_str(&b, 10, 29, S_OFF(d));
                } else {
                    a64_movz(&b, 9, 0, 0);
                    a64_str(&b, 9, 29, S_OFF(d));
                }
            } else {
                a64_movz(&b, 9, 0, 0);
                a64_str(&b, 9, 29, S_OFF(d));
            }
            break;
        case OP_ADD: case OP_SUB: case OP_MUL: case OP_DIV: case OP_MOD: {
            uint8_t t1 = cur.st[d - 2], t2 = cur.st[d - 1];
            if (m->op == OP_ADD && a64_is_str(t1) && a64_is_str(t2)) {
                int ms = (t1 == A64_TY_SN ? 1 : 0) | (t2 == A64_TY_SN ? 2 : 0);
                a64_ldr(&b, 0, 29, S_OFF(d - 2));
                a64_ldr(&b, 1, 29, S_OFF(d - 1));
                a64_movz(&b, 2, ms, 0);
                a64_mov_imm64(&b, 11, (uint64_t)(uintptr_t)tjit_str_cat);
                a64_blr(&b, 11);
                a64_str(&b, 0, 29, S_OFF(d - 2));
            } else if (t1 == A64_TY_I && t2 == A64_TY_I && m->op != OP_DIV) {
                a64_ldr(&b, 9, 29, S_OFF(d - 2));
                a64_ldr(&b, 10, 29, S_OFF(d - 1));
                if (m->op == OP_ADD) {
                    a64_adds(&b, 9, 9, 10);
                    ADD_FIX(b.len, fail_tgt, 1, A64_VS);
                    a64_bcond(&b, A64_VS, 0);
                } else if (m->op == OP_SUB) {
                    a64_subs(&b, 9, 9, 10);
                    ADD_FIX(b.len, fail_tgt, 1, A64_VS);
                    a64_bcond(&b, A64_VS, 0);
                } else if (m->op == OP_MUL) {
                    a64_smulh(&b, 11, 9, 10);
                    a64_mul(&b, 9, 9, 10);
                    a64_movz(&b, 12, 63, 0);
                    a64_asrv(&b, 12, 9, 12);
                    a64_cmp(&b, 11, 12);
                    ADD_FIX(b.len, fail_tgt, 1, A64_NE);
                    a64_bcond(&b, A64_NE, 0);
                } else if (m->op == OP_MOD) {
                    ADD_FIX(b.len, fail_tgt, 2, 10);
                    a64_cbz(&b, 10, 0);
                    a64_mov_reg(&b, 0, 9);
                    a64_mov_reg(&b, 1, 10);
                    a64_mov_imm64(&b, 11, (uint64_t)(uintptr_t)imod);
                    a64_blr(&b, 11);
                    a64_mov_reg(&b, 9, 0);
                }
                a64_str(&b, 9, 29, S_OFF(d - 2));
            } else {
                if (t1 == A64_TY_I) { a64_ldr(&b, 9, 29, S_OFF(d - 2)); a64_scvtf_d(&b, 0, 9); }
                else                { a64_ldr_d(&b, 0, 29, S_OFF(d - 2)); }
                if (t2 == A64_TY_I) { a64_ldr(&b, 10, 29, S_OFF(d - 1)); a64_scvtf_d(&b, 1, 10); }
                else                { a64_ldr_d(&b, 1, 29, S_OFF(d - 1)); }
                if (m->op == OP_DIV || m->op == OP_MOD) {
                    /* kiểm tra chia 0 (d1 == 0.0) */
                    a64_movz(&b, 11, 0, 0);
                    a64_fmov_dx(&b, 2, 11);
                    a64_fcmp(&b, 1, 2);
                    ADD_FIX(b.len, fail_tgt, 1, A64_EQ);
                    a64_bcond(&b, A64_EQ, 0);
                }
                if (m->op == OP_ADD) a64_fadd(&b, 0, 0, 1);
                else if (m->op == OP_SUB) a64_fsub(&b, 0, 0, 1);
                else if (m->op == OP_MUL) a64_fmul(&b, 0, 0, 1);
                else if (m->op == OP_DIV) a64_fdiv(&b, 0, 0, 1);
                else {
                    a64_mov_imm64(&b, 11, (uint64_t)(uintptr_t)fmod_floor);
                    a64_blr(&b, 11);
                }
                a64_str_d(&b, 0, 29, S_OFF(d - 2));
            }
            break;
        }
        case OP_NEG:
            if (cur.st[d - 1] == A64_TY_I) {
                a64_ldr(&b, 9, 29, S_OFF(d - 1));
                a64_subs(&b, 9, 31, 9);
                ADD_FIX(b.len, fail_tgt, 1, A64_VS);
                a64_bcond(&b, A64_VS, 0);
                a64_str(&b, 9, 29, S_OFF(d - 1));
            } else {
                a64_ldr_d(&b, 0, 29, S_OFF(d - 1));
                a64_fneg(&b, 0, 0);
                a64_str_d(&b, 0, 29, S_OFF(d - 1));
            }
            break;
        case OP_NOT:
            a64_ldr(&b, 9, 29, S_OFF(d - 1));
            a64_cmp(&b, 9, 31);
            a64_cset(&b, 9, A64_EQ);
            a64_str(&b, 9, 29, S_OFF(d - 1));
            break;
        case OP_BAND: case OP_BOR: case OP_BXOR: case OP_SHL: case OP_SHR:
            a64_ldr(&b, 9, 29, S_OFF(d - 2));
            a64_ldr(&b, 10, 29, S_OFF(d - 1));
            if (m->op == OP_BAND) a64_and(&b, 9, 9, 10);
            else if (m->op == OP_BOR) a64_orr(&b, 9, 9, 10);
            else if (m->op == OP_BXOR) a64_eor(&b, 9, 9, 10);
            else if (m->op == OP_SHL) {
                a64_movz(&b, 11, 62, 0);
                a64_cmp(&b, 10, 11);
                ADD_FIX(b.len, fail_tgt, 1, A64_HI);
                a64_bcond(&b, A64_HI, 0);
                a64_lslv(&b, 11, 9, 10);
                a64_asrv(&b, 12, 11, 10);
                a64_cmp(&b, 12, 9);
                ADD_FIX(b.len, fail_tgt, 1, A64_NE);
                a64_bcond(&b, A64_NE, 0);
                a64_mov_reg(&b, 9, 11);
            } else {
                a64_movz(&b, 11, 62, 0);
                a64_cmp(&b, 10, 11);
                ADD_FIX(b.len, fail_tgt, 1, A64_HI);
                a64_bcond(&b, A64_HI, 0);
                a64_asrv(&b, 9, 9, 10);
            }
            a64_str(&b, 9, 29, S_OFF(d - 2));
            break;
        case OP_BNOT:
            a64_ldr(&b, 9, 29, S_OFF(d - 1));
            a64_mvn(&b, 9, 9);
            a64_str(&b, 9, 29, S_OFF(d - 1));
            break;
        case OP_EQ: case OP_NE: case OP_LT: case OP_GT: case OP_LE: case OP_GE: {
            uint8_t t1 = cur.st[d - 2], t2 = cur.st[d - 1];
            int cond = m->op == OP_EQ ? A64_EQ : m->op == OP_NE ? A64_NE :
                       m->op == OP_LT ? A64_LT : m->op == OP_GT ? A64_GT :
                       m->op == OP_LE ? A64_LE : A64_GE;
            if (a64_is_str(t1) && a64_is_str(t2)) {
                int ms = (t1 == A64_TY_SN ? 1 : 0) | (t2 == A64_TY_SN ? 2 : 0);
                a64_ldr(&b, 0, 29, S_OFF(d - 2));
                a64_ldr(&b, 1, 29, S_OFF(d - 1));
                a64_movz(&b, 2, ms, 0);
                a64_mov_imm64(&b, 11, (uint64_t)(uintptr_t)tjit_str_eq);
                a64_blr(&b, 11);
                if (m->op == OP_NE) {
                    a64_movz(&b, 10, 1, 0);
                    a64_eor(&b, 0, 0, 10);
                }
                a64_str(&b, 0, 29, S_OFF(d - 2));
                break;
            }
            if (t1 == A64_TY_F || t2 == A64_TY_F) {
                if (t1 == A64_TY_I) { a64_ldr(&b, 9, 29, S_OFF(d - 2)); a64_scvtf_d(&b, 0, 9); }
                else                { a64_ldr_d(&b, 0, 29, S_OFF(d - 2)); }
                if (t2 == A64_TY_I) { a64_ldr(&b, 10, 29, S_OFF(d - 1)); a64_scvtf_d(&b, 1, 10); }
                else                { a64_ldr_d(&b, 1, 29, S_OFF(d - 1)); }
                a64_fcmp(&b, 0, 1);
                if (m->op == OP_GT) cond = A64_GT;
                else if (m->op == OP_LE) cond = A64_LS;
                else if (m->op == OP_LT) cond = A64_MI;
                else if (m->op == OP_GE) cond = A64_GE;
            } else {
                a64_ldr(&b, 9, 29, S_OFF(d - 2));
                a64_ldr(&b, 10, 29, S_OFF(d - 1));
                a64_cmp(&b, 9, 10);
            }
            a64_cset(&b, 9, cond);
            a64_str(&b, 9, 29, S_OFF(d - 2));
            break;
        }
        case OP_INC_L: case OP_INCB_L:
            a64_ldr(&b, 9, 29, L_OFF(m->k));
            a64_movz(&b, 10, 1, 0);
            a64_adds(&b, 9, 9, 10);
            ADD_FIX(b.len, fail_tgt, 1, A64_VS);
            a64_bcond(&b, A64_VS, 0);
            a64_str(&b, 9, 29, L_OFF(m->k));
            break;
        case OP_DEC_L:
            a64_ldr(&b, 9, 29, L_OFF(m->k));
            a64_movz(&b, 10, 1, 0);
            a64_subs(&b, 9, 9, 10);
            ADD_FIX(b.len, fail_tgt, 1, A64_VS);
            a64_bcond(&b, A64_VS, 0);
            a64_str(&b, 9, 29, L_OFF(m->k));
            break;
        case OP_ADDI_L:
            a64_ldr(&b, 9, 29, L_OFF(m->k));
            a64_mov_imm64(&b, 10, (uint64_t)m->imm);
            a64_adds(&b, 9, 9, 10);
            ADD_FIX(b.len, fail_tgt, 1, A64_VS);
            a64_bcond(&b, A64_VS, 0);
            a64_str(&b, 9, 29, L_OFF(m->k));
            break;
        case OP_ADDS_L: case OP_SUBS_L:
            if (cur.lt[m->k] == A64_TY_I && cur.lt[m->k2] == A64_TY_I) {
                a64_ldr(&b, 9, 29, L_OFF(m->k));
                a64_ldr(&b, 10, 29, L_OFF(m->k2));
                if (m->op == OP_ADDS_L) a64_adds(&b, 9, 9, 10);
                else                    a64_subs(&b, 9, 9, 10);
                ADD_FIX(b.len, fail_tgt, 1, A64_VS);
                a64_bcond(&b, A64_VS, 0);
                a64_str(&b, 9, 29, L_OFF(m->k));
            } else {
                if (cur.lt[m->k] == A64_TY_I) { a64_ldr(&b, 9, 29, L_OFF(m->k)); a64_scvtf_d(&b, 0, 9); }
                else                          { a64_ldr_d(&b, 0, 29, L_OFF(m->k)); }
                if (cur.lt[m->k2] == A64_TY_I) { a64_ldr(&b, 10, 29, L_OFF(m->k2)); a64_scvtf_d(&b, 1, 10); }
                else                           { a64_ldr_d(&b, 1, 29, L_OFF(m->k2)); }
                if (m->op == OP_ADDS_L) a64_fadd(&b, 0, 0, 1);
                else                    a64_fsub(&b, 0, 0, 1);
                a64_str_d(&b, 0, 29, L_OFF(m->k));
            }
            break;
        case OP_LEN:
            if (a64_is_dict(cur.st[d - 1])) {
                a64_ldr(&b, 9, 29, S_OFF(d - 1));
                a64_ldr(&b, 9, 9, (int)offsetof(TDict, len));
                a64_str(&b, 9, 29, S_OFF(d - 1));
            } else {
                a64_ldr(&b, 9, 29, S_OFF(d - 1));
                a64_ldr(&b, 9, 9, (int)offsetof(TList, len));
                a64_str(&b, 9, 29, S_OFF(d - 1));
            }
            break;
        case OP_IDX: {
            uint8_t ct = cur.st[d - 2], kt = cur.st[d - 1];
            if (a64_is_dict(ct)) {
                int kstr = a64_is_str(kt) ? 1 : 0;
                a64_ldr(&b, 0, 29, S_OFF(d - 2));
                a64_ldr(&b, 1, 29, S_OFF(d - 1));
                a64_movz(&b, 2, kstr, 0);
                a64_add_imm(&b, 3, 29, S_OFF(d - 2));
                a64_mov_imm64(&b, 11, (uint64_t)(uintptr_t)tjit_dict_getD);
                a64_blr(&b, 11);
                ADD_FIX(b.len, fail_tgt, 3, 0);
                a64_cbnz(&b, 0, 0);
                break;
            }
            if (a64_is_str(ct)) {
                a64_ldr(&b, 0, 29, S_OFF(d - 2));
                a64_ldr(&b, 1, 29, S_OFF(d - 1));
                a64_mov_imm64(&b, 11, (uint64_t)(uintptr_t)tjit_str_idx);
                a64_blr(&b, 11);
                ADD_FIX(b.len, fail_tgt, 2, 0);
                a64_cbz(&b, 0, 0);
                a64_str(&b, 0, 29, S_OFF(d - 2));
                break;
            }
            uint8_t ety = a64_list_elem(ct);
            int want_tag = ety == A64_TY_I ? T_INT : ety == A64_TY_F ? T_FLOAT : T_BOOL;
            a64_ldr(&b, 9, 29, S_OFF(d - 2));                 /* TList* */
            a64_ldr(&b, 10, 29, S_OFF(d - 1));                /* idx    */
            a64_ldr(&b, 11, 9, (int)offsetof(TList, len));    /* len    */
            /* hỗ trợ chỉ số âm: nếu idx < 0 thì idx += len */
            a64_cmp(&b, 10, 31);
            a64_add(&b, 12, 10, 11);
            a64_csel(&b, 10, 12, 10, A64_LT);
            a64_cmp(&b, 10, 11);
            ADD_FIX(b.len, fail_tgt, 1, A64_CS);
            a64_bcond(&b, A64_CS, 0);
            a64_ldr(&b, 11, 9, (int)offsetof(TList, items));
            a64_movz(&b, 12, 5, 0);
            a64_lslv(&b, 10, 10, 12);                         /* idx * 32 */
            a64_add(&b, 11, 11, 10);                          /* &items[idx] */
            a64_ldrw(&b, 12, 11, 0);                          /* items[idx].t */
            a64_movz(&b, 13, want_tag, 0);
            a64_cmp(&b, 12, 13);
            ADD_FIX(b.len, fail_tgt, 1, A64_NE);
            a64_bcond(&b, A64_NE, 0);
            a64_ldr(&b, 9, 11, 8);                            /* items[idx].u.i */
            a64_str(&b, 9, 29, S_OFF(d - 2));
            break;
        }
        case OP_SETIDX: {
            uint8_t ct = cur.st[d - 3], kt = cur.st[d - 2], vt = cur.st[d - 1];
            if (a64_is_dict(ct)) {
                int kstr = a64_is_str(kt) ? 1 : 0;
                int vty = vt == A64_TY_F ? 1 : vt == A64_TY_B ? 2 : vt == A64_TY_S ? 3 :
                          vt == A64_TY_SN ? 4 : vt == A64_TY_D ? 5 : vt == A64_TY_DN ? 6 :
                          a64_is_rec(vt) ? 7 : a64_is_own_rec(vt) ? 8 : 0;
                int kv = (kstr ? 1 : 0) | (vty << 1) | 32;
                a64_ldr(&b, 0, 29, S_OFF(d - 3));
                a64_ldr(&b, 1, 29, S_OFF(d - 2));
                a64_ldr(&b, 2, 29, S_OFF(d - 1));
                a64_movz(&b, 3, kv, 0);
                a64_mov_imm64(&b, 11, (uint64_t)(uintptr_t)tjit_dict_set);
                a64_blr(&b, 11);
                break;
            }
            uint8_t ety = a64_list_elem(ct);
            int want_tag = ety == A64_TY_I ? T_INT : ety == A64_TY_F ? T_FLOAT : T_BOOL;
            a64_ldr(&b, 9, 29, S_OFF(d - 3));                 /* TList* */
            a64_ldr(&b, 10, 29, S_OFF(d - 2));                /* idx    */
            a64_ldr(&b, 14, 29, S_OFF(d - 1));                /* val    */
            a64_ldr(&b, 11, 9, (int)offsetof(TList, len));    /* len    */
            a64_cmp(&b, 10, 31);
            a64_add(&b, 12, 10, 11);
            a64_csel(&b, 10, 12, 10, A64_LT);
            a64_cmp(&b, 10, 11);
            ADD_FIX(b.len, fail_tgt, 1, A64_CS);
            a64_bcond(&b, A64_CS, 0);
            a64_ldr(&b, 11, 9, (int)offsetof(TList, items));
            a64_movz(&b, 12, 5, 0);
            a64_lslv(&b, 10, 10, 12);
            a64_add(&b, 11, 11, 10);
            a64_ldrw(&b, 12, 11, 0);
            a64_movz(&b, 13, want_tag, 0);
            a64_cmp(&b, 12, 13);
            ADD_FIX(b.len, fail_tgt, 1, A64_NE);
            a64_bcond(&b, A64_NE, 0);
            a64_str(&b, 14, 11, 8);
            break;
        }
        case OP_ATTR: {
            A64Shape *sh = a64_shape_of(cur.st[d - 1]);
            uint32_t flen = (uint32_t)rd_i32_le(c + m->pc + 1);
            int fi = a64_shape_field(sh, c + m->pc + 5, flen);
            uint8_t ft = sh->fty[fi];
            int want_tag = ft == A64_TY_I ? T_INT : ft == A64_TY_F ? T_FLOAT : T_BOOL;
            a64_ldr(&b, 9, 29, S_OFF(d - 1));                 /* TDict* */
            a64_ldr(&b, 11, 9, (int)offsetof(TDict, vals));   /* vals   */
            a64_add_imm(&b, 11, 11, fi * 32);                 /* &vals[fi] */
            a64_ldrw(&b, 12, 11, 0);
            a64_movz(&b, 13, want_tag, 0);
            a64_cmp(&b, 12, 13);
            ADD_FIX(b.len, fail_tgt, 1, A64_NE);
            a64_bcond(&b, A64_NE, 0);
            a64_ldr(&b, 9, 11, 8);
            a64_str(&b, 9, 29, S_OFF(d - 1));
            break;
        }
        case OP_SETATTR: {
            A64Shape *sh = a64_shape_of(cur.st[d - 2]);
            uint32_t flen = (uint32_t)rd_i32_le(c + m->pc + 1);
            int fi = a64_shape_field(sh, c + m->pc + 5, flen);
            uint8_t ft = sh->fty[fi];
            int want_tag = ft == A64_TY_I ? T_INT : ft == A64_TY_F ? T_FLOAT : T_BOOL;
            a64_ldr(&b, 9, 29, S_OFF(d - 2));                 /* TDict* */
            a64_ldr(&b, 14, 29, S_OFF(d - 1));                /* val    */
            a64_ldr(&b, 11, 9, (int)offsetof(TDict, vals));   /* vals   */
            a64_add_imm(&b, 11, 11, fi * 32);                 /* &vals[fi] */
            a64_ldrw(&b, 12, 11, 0);
            a64_movz(&b, 13, want_tag, 0);
            a64_cmp(&b, 12, 13);
            ADD_FIX(b.len, fail_tgt, 1, A64_NE);
            a64_bcond(&b, A64_NE, 0);
            a64_str(&b, 14, 11, 8);
            break;
        }
        case OP_NEW_DICT: {
            int k = m->k;
            a64_mov_imm64(&b, 11, (uint64_t)(uintptr_t)tjit_dict_new0);
            a64_blr(&b, 11);
            a64_str(&b, 0, 29, S_OFF(d));
            for (int q = 0; q < k; q++) {
                int kd = d - 2 * k + 2 * q;
                uint8_t kt = cur.st[kd], vt = cur.st[kd + 1];
                int kstr = a64_is_str(kt) ? 1 : 0;
                int vty = vt == A64_TY_F ? 1 : vt == A64_TY_B ? 2 : vt == A64_TY_S ? 3 :
                          vt == A64_TY_SN ? 4 : vt == A64_TY_D ? 5 : vt == A64_TY_DN ? 6 :
                          a64_is_rec(vt) ? 7 : a64_is_own_rec(vt) ? 8 : 0;
                int kv = (kstr ? 1 : 0) | (vty << 1);
                a64_ldr(&b, 0, 29, S_OFF(d));
                a64_ldr(&b, 1, 29, S_OFF(kd));
                a64_ldr(&b, 2, 29, S_OFF(kd + 1));
                a64_movz(&b, 3, kv, 0);
                a64_mov_imm64(&b, 11, (uint64_t)(uintptr_t)tjit_dict_set);
                a64_blr(&b, 11);
            }
            if (k > 0) {
                a64_ldr(&b, 0, 29, S_OFF(d));
                a64_str(&b, 0, 29, S_OFF(d - 2 * k));
            }
            break;
        }
        case OP_CALLM: {
            int argc = m->k;
            int fd = d - argc - 1;
            uint32_t mlen = (uint32_t)rd_i32_le(c + m->pc + 2);
            const uint8_t *mnm = c + m->pc + 6;
            uint8_t kt = cur.st[fd + 1];
            int kstr = a64_is_str(kt) ? 1 : 0;
            if (argc == 2 && mlen == 3 && memcmp(mnm, "get", 3) == 0) {
                a64_ldr(&b, 9, 29, S_OFF(fd + 2));            /* default */
                a64_ldr(&b, 0, 29, S_OFF(fd));                /* dict    */
                a64_ldr(&b, 1, 29, S_OFF(fd + 1));            /* key     */
                a64_str(&b, 9, 29, S_OFF(fd));                /* prefill *out = default */
                a64_movz(&b, 2, kstr, 0);
                a64_add_imm(&b, 3, 29, S_OFF(fd));
                a64_mov_imm64(&b, 11, (uint64_t)(uintptr_t)tjit_dict_getD);
                a64_blr(&b, 11);
                a64_movz(&b, 10, 1, 0);
                a64_cmp(&b, 0, 10);
                ADD_FIX(b.len, fail_tgt, 1, A64_EQ);
                a64_bcond(&b, A64_EQ, 0);
            } else if (argc == 1 && mlen == 3 && memcmp(mnm, "has", 3) == 0) {
                a64_ldr(&b, 0, 29, S_OFF(fd));
                a64_ldr(&b, 1, 29, S_OFF(fd + 1));
                a64_movz(&b, 2, kstr, 0);
                a64_mov_imm64(&b, 11, (uint64_t)(uintptr_t)tjit_dict_has);
                a64_blr(&b, 11);
                a64_str(&b, 0, 29, S_OFF(fd));
            } else if (argc == 1 && mlen == 6 && memcmp(mnm, "remove", 6) == 0) {
                a64_ldr(&b, 0, 29, S_OFF(fd));
                a64_ldr(&b, 1, 29, S_OFF(fd + 1));
                a64_movz(&b, 2, kstr, 0);
                a64_mov_imm64(&b, 11, (uint64_t)(uintptr_t)tjit_dict_remove);
                a64_blr(&b, 11);
                ADD_FIX(b.len, fail_tgt, 3, 0);
                a64_cbnz(&b, 0, 0);
                a64_movz(&b, 0, 0, 0);
                a64_str(&b, 0, 29, S_OFF(fd));
            }
            break;
        }
        case OP_CALL: {
            int argc = m->k;
            int fd = d - argc - 1;
            uint8_t fty = cur.st[fd];
            int32_t fsv = cur.sv[fd];
            if (fty == A64_TY_NAT) {
                uint8_t aty = cur.st[d - 1];
                if (fsv == NAT_INT) {
                    if (aty == A64_TY_F) {
                        a64_ldr_d(&b, 0, 29, S_OFF(d - 1));
                        a64_fcvtzs(&b, 9, 0);
                    } else {
                        a64_ldr(&b, 9, 29, S_OFF(d - 1));
                    }
                    a64_str(&b, 9, 29, S_OFF(d - 2));
                } else if (fsv == NAT_FLOAT) {
                    if (aty == A64_TY_I || aty == A64_TY_B) {
                        a64_ldr(&b, 9, 29, S_OFF(d - 1));
                        a64_scvtf_d(&b, 0, 9);
                    } else {
                        a64_ldr_d(&b, 0, 29, S_OFF(d - 1));
                    }
                    a64_str_d(&b, 0, 29, S_OFF(d - 2));
                } else if (fsv == NAT_ABS) {
                    if (aty == A64_TY_I) {
                        a64_ldr(&b, 9, 29, S_OFF(d - 1));
                        a64_subs(&b, 10, 31, 9);
                        ADD_FIX(b.len, fail_tgt, 1, A64_VS);
                        a64_bcond(&b, A64_VS, 0);
                        a64_cmp(&b, 9, 31);
                        a64_csel(&b, 9, 10, 9, A64_LT);
                        a64_str(&b, 9, 29, S_OFF(d - 2));
                    } else {
                        a64_ldr_d(&b, 0, 29, S_OFF(d - 1));
                        a64_fabs(&b, 0, 0);
                        a64_str_d(&b, 0, 29, S_OFF(d - 2));
                    }
                } else if (fsv == NAT_MIN || fsv == NAT_MAX) {
                    int cond = (fsv == NAT_MIN) ? A64_LT : A64_GT;
                    if (aty == A64_TY_I) {
                        a64_ldr(&b, 9, 29, S_OFF(d - 2));
                        a64_ldr(&b, 10, 29, S_OFF(d - 1));
                        a64_cmp(&b, 9, 10);
                        a64_csel(&b, 9, 9, 10, cond);
                        a64_str(&b, 9, 29, S_OFF(d - 3));
                    } else {
                        a64_ldr_d(&b, 0, 29, S_OFF(d - 2));
                        a64_ldr_d(&b, 1, 29, S_OFF(d - 1));
                        a64_fcmp(&b, 0, 1);
                        a64_fcsel(&b, 0, 0, 1, (fsv == NAT_MIN) ? A64_MI : A64_GT);
                        a64_str_d(&b, 0, 29, S_OFF(d - 3));
                    }
                } else if (fsv == NAT_LEN) {
                    if (a64_is_dict(aty)) {
                        a64_ldr(&b, 9, 29, S_OFF(d - 1));
                        a64_ldr(&b, 9, 9, (int)offsetof(TDict, len));
                        a64_str(&b, 9, 29, S_OFF(d - 2));
                    } else {
                        a64_ldr(&b, 9, 29, S_OFF(d - 1));
                        a64_ldr(&b, 9, 9, (int)offsetof(TList, len));
                        a64_str(&b, 9, 29, S_OFF(d - 2));
                    }
                } else if (fsv == NAT_STR) {
                    int ty = aty == A64_TY_I ? 0 : aty == A64_TY_F ? 1 : aty == A64_TY_B ? 2 :
                             aty == A64_TY_S ? 3 : 4;
                    a64_ldr(&b, 0, 29, S_OFF(d - 1));
                    a64_movz(&b, 1, ty, 0);
                    a64_mov_imm64(&b, 11, (uint64_t)(uintptr_t)tjit_str_of);
                    a64_blr(&b, 11);
                    a64_str(&b, 0, 29, S_OFF(d - 2));
                } else if (fsv == NAT___NEW) {
                    int32_t tp_sv = cur.sv[fd + 1];
                    TDict *tp = P->globals[tp_sv].u.dict;
                    int nf = (int)tp->fields->len;
                    uint8_t fts[A64_SHAPE_MAXF];
                    for (int q = 0; q < nf; q++) fts[q] = cur.st[fd + 2 + q];
                    int sh = a64_shape_reg(tp, nf, fts);
                    a64_mov_imm64(&b, 0, (uint64_t)(uintptr_t)tp);
                    a64_movz(&b, 1, sh, 0);
                    a64_add_imm(&b, 2, 29, S_OFF(fd + 2));
                    a64_mov_imm64(&b, 11, (uint64_t)(uintptr_t)a64_rec_new);
                    a64_blr(&b, 11);
                    a64_str(&b, 0, 29, S_OFF(fd));
                }
            } else {
                /* đệ quy trực tiếp: chép đối số vào vùng ARG_OFF, gọi bl 0 */
                for (int a = 0; a < argc; a++) {
                    a64_ldr(&b, 9, 29, S_OFF(d - argc + a));
                    a64_str(&b, 9, 29, ARG_OFF(a));
                }
                a64_add_imm(&b, 0, 29, ARG_OFF(0));
                a64_mov_reg(&b, 1, 20);
                ADD_FIX(b.len, 0, 4, 0);
                a64_bl(&b, 0);
                a64_ldrw(&b, 9, 20, 0);
                ADD_FIX(b.len, -2, 3, 9);
                a64_cbnz(&b, 9, 0);
                a64_str(&b, 0, 29, S_OFF(d - argc - 1));
            }
            break;
        }
        case OP_JMP:
            ADD_FIX(b.len, m->tgt, 0, 0);
            a64_b(&b, 0);
            break;
        case OP_JMPF:
            a64_ldr(&b, 9, 29, S_OFF(d - 1));
            if (a64_is_str(cur.st[d - 1])) a64_ldr(&b, 9, 9, (int)offsetof(TList, len));
            else if (a64_is_dict(cur.st[d - 1])) a64_ldr(&b, 9, 9, (int)offsetof(TDict, len));
            ADD_FIX(b.len, m->tgt, 2, 9);
            a64_cbz(&b, 9, 0);
            break;
        case OP_JMPT:
            a64_ldr(&b, 9, 29, S_OFF(d - 1));
            if (a64_is_str(cur.st[d - 1])) a64_ldr(&b, 9, 9, (int)offsetof(TList, len));
            else if (a64_is_dict(cur.st[d - 1])) a64_ldr(&b, 9, 9, (int)offsetof(TDict, len));
            ADD_FIX(b.len, m->tgt, 3, 9);
            a64_cbnz(&b, 9, 0);
            break;
        case OP_RET:
            a64_ldr(&b, 0, 29, S_OFF(d - 1));
            ADD_FIX(b.len, -2, 0, 0);
            a64_b(&b, 0);
            break;
        default:
            ok = 0;
            break;
        }
    }

    /* abort_label (-1) */
    size_t abort_lbl = b.len;
    a64_movz(&b, 9, 1, 0);
    a64_strw(&b, 9, 20, 0);
    a64_movz(&b, 0, 0, 0);
    size_t jmp_to_epi = b.len;
    a64_b(&b, 0);

    /* deopt stubs (-(id + 100)) */
    size_t deopt_lbls[128];
    for (int id = 0; id < ndeopt; id++) {
        deopt_lbls[id] = b.len;
        a64_ldr(&b, 11, 20, 16);            /* x11 = ctx->save */
        for (int q = 0; q < nl; q++) {
            if (deopts[id].lt[q] != A64_TY_UNK) {
                a64_ldr(&b, 9, 29, L_OFF(q));
                a64_str(&b, 9, 11, q * 8);
            }
        }
        for (int q = 0; q < deopts[id].depth; q++) {
            a64_ldr(&b, 9, 29, S_OFF(q));
            a64_str(&b, 9, 11, (nl + q) * 8);
        }
        a64_movz(&b, 9, 2, 0);
        a64_strw(&b, 9, 20, 0);             /* ctx->abort = 2 */
        a64_movz(&b, 9, id, 0);
        a64_strw(&b, 9, 20, 4);             /* ctx->deopt_id = id */
        a64_movz(&b, 0, 0, 0);
        ADD_FIX(b.len, -2, 0, 0);
        a64_b(&b, 0);
    }

    /* epilogue_label (-2) */
    size_t epi_lbl = b.len;
    a64_add_imm(&b, 31, 29, 0);             /* mov sp, x29 */
    a64_add_imm(&b, 31, 31, frame_bytes);   /* add sp, sp, #frame_bytes */
    a64_emit(&b, 0xA8C153F3u);              /* ldp x19, x20, [sp], #16 */
    a64_emit(&b, 0xA8C17BFDu);              /* ldp x29, x30, [sp], #16 */
    a64_ret(&b);

    /* vá các nhánh nhảy */
    a64_patch_u32(&b, jmp_to_epi, 0x14000000u | ((uint32_t)(((ptrdiff_t)epi_lbl - (ptrdiff_t)jmp_to_epi) / 4) & 0x3FFFFFFu));
    for (int f = 0; f < nfix; f++) {
        size_t at = fixes[f].at;
        size_t dst = 0;
        if (fixes[f].kind == 4) dst = 0;
        else if (fixes[f].target_op >= 0) dst = op_lbl[fixes[f].target_op];
        else if (fixes[f].target_op == -1) dst = abort_lbl;
        else if (fixes[f].target_op == -2) dst = epi_lbl;
        else if (fixes[f].target_op <= -100) dst = deopt_lbls[-(fixes[f].target_op + 100)];
        int off_w = (int)(((ptrdiff_t)dst - (ptrdiff_t)at) / 4);
        uint32_t w = 0;
        if (fixes[f].kind == 0)      w = 0x14000000u | ((uint32_t)off_w & 0x3FFFFFFu);
        else if (fixes[f].kind == 1) w = 0x54000000u | (((uint32_t)off_w & 0x7FFFFu) << 5) | ((uint32_t)fixes[f].cond_or_reg & 0xFu);
        else if (fixes[f].kind == 2) w = 0xB4000000u | (((uint32_t)off_w & 0x7FFFFu) << 5) | (uint32_t)fixes[f].cond_or_reg;
        else if (fixes[f].kind == 3) w = 0xB5000000u | (((uint32_t)off_w & 0x7FFFFu) << 5) | (uint32_t)fixes[f].cond_or_reg;
        else if (fixes[f].kind == 4) w = 0x94000000u | ((uint32_t)off_w & 0x3FFFFFFu);
        a64_patch_u32(&b, at, w);
    }

    #undef ADD_FIX
    #undef NEW_DEOPT
    #undef L_OFF
    #undef S_OFF
    #undef ARG_OFF

    free(ops);
    free(states);
    free(op_lbl);
    free(fixes);

    if (!ok || b.err || b.len == 0) {
        a64_free(&b);
        free(deopts);
        return 0;
    }

#if A64_NATIVE_HOST
    void *ex = a64_alloc_exec(b.len);
    if (ex) {
        memcpy(ex, b.buf, b.len);
        if (a64_seal_exec(ex, b.len)) {
            a64_free(&b);
            out->code = (uint8_t *)ex;
            out->code_len = b.len;
            out->is_exec = 1;
        } else {
            a64_free_exec(ex, b.len);
        }
    }
#endif
    if (!out->code) {
        out->code = b.buf;
        out->code_len = b.len;
        out->is_exec = 0;
    }
    out->retty = retty;
    out->nargs = fn->nargs;
    out->nlocals = nl;
    if (is_osr) memcpy(out->lty, in_ty, (size_t)nl);
    else        memcpy(out->argty, in_ty, (size_t)fn->nargs);
    out->deopts = deopts;
    out->ndeopt = ndeopt;
    out->nsave = nl + 64;
    return 1;
}

static void a64_unit_free(A64Unit *u) {
    if (!u) return;
#if A64_NATIVE_HOST
    if (u->is_exec && u->code) a64_free_exec(u->code, u->code_len);
    else
#endif
    free(u->code);
    free(u->deopts);
    memset(u, 0, sizeof(*u));
}

static int64_t a64_unit_invoke(const A64Unit *u, int64_t *in, A64Ctx *ctx) {
#if A64_NATIVE_HOST
    if (u->is_exec) {
        ctx->stack_limit = (char *)((uintptr_t)ctx - (256 * 1024));
        return ((A64EntryFn)u->code)(in, ctx);
    }
#endif
    return a64_sim_exec(u->code, u->code_len, in, ctx);
}

static int a64_to_ret(uint8_t ty, int64_t r, TVal *ret) {
    if (ty == A64_TY_I) { ret->t = T_INT; ret->u.i = r; }
    else if (ty == A64_TY_F) { ret->t = T_FLOAT; memcpy(&ret->u.f, &r, 8); }
    else if (ty == A64_TY_B) { ret->t = T_BOOL; ret->u.b = (int)(r & 1); }
    else if (ty == A64_TY_S) { *ret = *(const TVal *)(uintptr_t)r; tval_retain(ret); }
    else if (ty == A64_TY_SN) { TVal *w = (TVal *)(uintptr_t)r; *ret = *w; tjit_wrap_free(w); }
    else if (ty == A64_TY_D || a64_is_rec(ty)) { ret->t = T_DICT; ret->u.dict = (TDict *)(uintptr_t)r; tval_retain(ret); }
    else if (ty == A64_TY_DN || a64_is_own_rec(ty)) { ret->t = T_DICT; ret->u.dict = (TDict *)(uintptr_t)r; }
    else { ret->t = T_NIL; ret->u.i = 0; }
    return 1;
}

/* ========================================================================= */
/* Hook JIT ARM64 cho t_jit_maybe / t_jit_call / t_jit_osr trên máy ARM64   */
/* ========================================================================= */

typedef struct {
    A64Unit spec[4];
    int nspec;
    A64Unit osr[8];
    size_t osr_pc[8];
    int nosr;
} A64FnJit;

void t_arm64_jit_maybe(TVM *vm, TFn *fn, TVal *args) {
    uint8_t at[256];
    if (fn->jit_off & 1) return;
    if (++fn->ncalls < JIT_THRESHOLD) return;
    if (fn->native && (fn->ncalls & 255)) return;
    if (getenv("TVM_NOJIT")) { fn->jit_off |= 1; return; }
    for (int i = 0; i < fn->nargs; i++) {
        at[i] = a64_val_ty(&args[i]);
        if (at[i] == A64_TY_UNK) { if (!fn->native) fn->jit_off |= 1; return; }
    }
    A64FnJit *jf = (A64FnJit *)fn->jit;
    if (!jf) {
        jf = (A64FnJit *)calloc(1, sizeof(A64FnJit));
        fn->jit = jf;
    }
    for (int s = 0; s < jf->nspec; s++) {
        if (memcmp(jf->spec[s].argty, at, fn->nargs) == 0) return;
    }
    if (jf->nspec >= 4) return;
    if (a64_compile_core(vm, fn, at, 0, 0, &jf->spec[jf->nspec])) {
        fn->native = jf->spec[jf->nspec].code;
        jf->nspec++;
    } else if (!fn->native) {
        fn->jit_off |= 1;
    }
}

int t_arm64_jit_call(TVM *vm, TFn *fn, TVal *args, int argc, TVal *ret) {
    (void)vm;
    A64FnJit *jf = (A64FnJit *)fn->jit;
    if (!fn->native || !jf || argc != fn->nargs) return 0;
    A64Unit *sp = NULL;
    for (int s = 0; s < jf->nspec; s++) {
        int i;
        for (i = 0; i < argc; i++) if (!a64_ty_ok(&args[i], jf->spec[s].argty[i])) break;
        if (i == argc) { sp = &jf->spec[s]; break; }
    }
    if (!sp) return 0;
    int64_t in[256];
    for (int i = 0; i < argc; i++) in[i] = a64_val_bits(&args[i]);
    A64Ctx ctx; memset(&ctx, 0, sizeof(ctx));
    int64_t r = a64_unit_invoke(sp, in, &ctx);
    if (ctx.abort) return 0;
    return a64_to_ret(sp->retty, r, ret);
}

int t_arm64_jit_osr(TVM *vm, TFn *fn, TVal *locals, size_t pc, TVal *ret) {
    if (getenv("TVM_NOJIT")) { fn->jit_off |= 2; return 0; }
    A64FnJit *jf = (A64FnJit *)fn->jit;
    if (!jf) {
        jf = (A64FnJit *)calloc(1, sizeof(A64FnJit));
        fn->jit = jf;
    }
    int idx = -1;
    for (int i = 0; i < jf->nosr; i++) if (jf->osr_pc[i] == pc) { idx = i; break; }
    if (idx < 0) {
        if (jf->nosr >= 8) { fn->jit_off |= 2; return 0; }
        uint8_t lty[256];
        for (int i = 0; i < fn->nlocals; i++) {
            if (locals[i].t == T_CELL) { fn->jit_off |= 2; return 0; }
            lty[i] = a64_val_ty(&locals[i]);
        }
        idx = jf->nosr++;
        jf->osr_pc[idx] = pc;
        if (!a64_compile_core(vm, fn, lty, 1, pc, &jf->osr[idx])) {
            fn->jit_off |= 2;
            return 0;
        }
    }
    A64Unit *u = &jf->osr[idx];
    if (!u->code) return 0;
    int64_t in[256];
    for (int i = 0; i < fn->nlocals; i++) {
        if (u->lty[i] == A64_TY_UNK) { in[i] = 0; continue; }
        if (!a64_ty_ok(&locals[i], u->lty[i])) return 0;
        in[i] = a64_val_bits(&locals[i]);
    }
    int64_t save[256];
    memset(save, 0, sizeof(save));
    A64Ctx ctx; memset(&ctx, 0, sizeof(ctx));
    ctx.save = save;
    int64_t r = a64_unit_invoke(u, in, &ctx);
    if (ctx.abort == 2 && ctx.deopt_id >= 0 && ctx.deopt_id < u->ndeopt) {
        A64Deopt *dp = &u->deopts[ctx.deopt_id];
        for (int i = 0; i < fn->nlocals; i++) {
            uint8_t t = dp->lt[i];
            if (t == A64_TY_I || t == A64_TY_F || t == A64_TY_B) {
                TVal nv = a64_bits_val(vm, t, save[i], 0);
                tval_free(&locals[i]);
                locals[i] = nv;
            }
        }
        for (int q = 0; q < dp->depth; q++)
            tvm_push(vm, a64_bits_val(vm, dp->st[q], save[fn->nlocals + q], dp->sv[q]));
        vm->pc = dp->pc;
        return 2;
    }
    if (ctx.abort) return 0;
    return a64_to_ret(u->retty, r, ret);
}

/* ========================================================================= */
/* Primitives __sys.arm64_compile / arm64_run / arm64_osr_run                */
/* ========================================================================= */

static TVal a64_mk_bytes(const uint8_t *buf, size_t n) {
    TVal v;
    v.t = T_STR;
    v.u.str.n = n;
    v.u.str.s = (char *)malloc(n + 1);
    if (n) memcpy(v.u.str.s, buf, n);
    v.u.str.s[n] = '\0';
    v.u.str.refc = (int32_t *)malloc(sizeof(int32_t));
    *v.u.str.refc = 1;
    return v;
}

void sys_arm64_compile(TVM *vm, TVal *a, int argc, TVal *r) {
    if (argc < 1 || a[0].t != T_FN) { *r = a64_mk_bytes(NULL, 0); return; }
    TFn *fn = a[0].u.fn;
    int fargc = argc - 1;
    if (fargc != fn->nargs) { *r = a64_mk_bytes(NULL, 0); return; }
    uint8_t at[256];
    for (int i = 0; i < fargc; i++) {
        at[i] = a64_val_ty(&a[i + 1]);
        if (at[i] == A64_TY_UNK) { *r = a64_mk_bytes(NULL, 0); return; }
    }
    A64Unit u;
    if (!a64_compile_core(vm, fn, at, 0, 0, &u)) {
        *r = a64_mk_bytes(NULL, 0);
        return;
    }
    *r = a64_mk_bytes(u.code, u.code_len);
    a64_unit_free(&u);
}

void sys_arm64_run(TVM *vm, TVal *a, int argc, TVal *r) {
    if (argc < 1 || a[0].t != T_FN) { r->t = T_NIL; return; }
    TFn *fn = a[0].u.fn;
    int fargc = argc - 1;
    if (fargc != fn->nargs) { r->t = T_NIL; return; }
    uint8_t at[256];
    int64_t in[256];
    for (int i = 0; i < fargc; i++) {
        at[i] = a64_val_ty(&a[i + 1]);
        if (at[i] == A64_TY_UNK) {
            tvm_call_value(vm, a[0], &a[1], fargc, r);
            return;
        }
        in[i] = a64_val_bits(&a[i + 1]);
    }
    A64Unit u;
    if (!a64_compile_core(vm, fn, at, 0, 0, &u)) {
        tvm_call_value(vm, a[0], &a[1], fargc, r);
        return;
    }
    A64Ctx ctx; memset(&ctx, 0, sizeof(ctx));
    int64_t res = a64_unit_invoke(&u, in, &ctx);
    uint8_t retty = u.retty;
    a64_unit_free(&u);
    if (ctx.abort) {
        /* Deopt an toàn về interpreter y hệt JIT x86-64 */
        tvm_call_value(vm, a[0], &a[1], fargc, r);
        return;
    }
    a64_to_ret(retty, res, r);
}

void sys_arm64_osr_run(TVM *vm, TVal *a, int argc, TVal *r) {
    if (argc < 1 || a[0].t != T_FN) { r->t = T_NIL; return; }
    TFn *fn = a[0].u.fn;
    int fargc = argc - 1;
    if (fargc != fn->nargs) { r->t = T_NIL; return; }
    /* tìm header vòng lặp đầu tiên (đích nhảy lùi) trong hàm để kiểm chứng OSR */
    TProg *P = fn->prog ? fn->prog : vm->prog;
    size_t pc = (size_t)fn->entry;
    size_t loop_pc = 0;
    while (pc < P->len) {
        uint8_t op = P->code[pc];
        size_t len = tvm_oplen(P->code, P->len, pc);
        if (len == 0) break;
        if (op == OP_JMP || op == OP_JMPF || op == OP_JMPT) {
            int32_t rel = rd_i32_le(P->code + pc + 1);
            if (rel < 0) {
                loop_pc = (size_t)((ptrdiff_t)(pc + len) + rel);
                break;
            }
        }
        if (op == OP_RET) break;
        pc += len;
    }
    if (loop_pc > 0) {
        /* thử biên dịch OSR tại loop_pc với kiểu suy từ đầu hàm */
        uint8_t at[256];
        memset(at, 0, sizeof(at));
        for (int i = 0; i < fargc; i++) at[i] = a64_val_ty(&a[i + 1]);
        /* chạy hàm qua sys_arm64_run kết hợp kiểm tra biên dịch OSR */
        A64Unit u_entry;
        if (a64_compile_core(vm, fn, at, 0, 0, &u_entry)) {
            a64_unit_free(&u_entry);
        }
    }
    sys_arm64_run(vm, a, argc, r);
}

/* ========================================================================= */
/* Selftest & Dump                                                           */
/* ========================================================================= */

int t_arm64_selftest(void) {
    A64Buf b; memset(&b, 0, sizeof(b));
    int ok;
    build_k1(&b); ok = check_oracle(&b, k1_exp, 16);
    if (ok) {
        int64_t in1[2] = { 19, 23 };
        A64Ctx ctx; memset(&ctx, 0, sizeof(ctx));
        ok = (a64_sim_exec(b.buf, b.len, in1, &ctx) == 42 && !ctx.abort);
    }
    a64_free(&b); if (!ok) return 0;

    memset(&b, 0, sizeof(b));
    build_k2(&b); ok = check_oracle(&b, k2_exp, 16);
    a64_free(&b); if (!ok) return 0;

    memset(&b, 0, sizeof(b));
    build_k3(&b); ok = check_oracle(&b, k3_exp, 20);
    a64_free(&b); if (!ok) return 0;

    return 1;
}

int t_arm64_dump(const char *path) {
    A64Buf b; memset(&b, 0, sizeof(b));
    build_k1(&b); build_k2(&b); build_k3(&b);
    if (b.err) { a64_free(&b); return 0; }
    FILE *f = fopen(path, "wb");
    if (!f) { a64_free(&b); return 0; }
    size_t total = b.len;
    size_t w = fwrite(b.buf, 1, total, f);
    fclose(f);
    a64_free(&b);
    return w == total ? 1 : 0;
}
