/* jit_arm64.c — v5.1 (M7/J7) trình phát mã AArch64 cho T.
 *
 * PHẠM VI (skeleton, CHƯA phải JIT hoàn chỉnh): emitter sinh byte ARM64
 * 4-byte cho một tập lệnh nhỏ (add/sub/mul, ldr/str, movz/movk, fadd/fsub/
 * fmul/fdiv, scvtf/fcvtzs, cmp + b.cond/cbz/cbnz/bl/blr, stp/ldp frame,
 * ret), ABI int64 entry(in, ctx) giống jit_x64.c.
 *
 * Chưa nối vào t_jit_maybe: code chưa đủ dữ liệu để chạy thật (cần abstract
 * interpretation bytecode + deopt như jit_x64.c). Hiện tại dùng để:
 *   - t_arm64_selftest(): kiểm chứng byte phát ra theo oracle viết tay
 *     (chạy được trên mọi kiến trúc, kể cả x86-64);
 *   - t_arm64_dump(path): ghi byte ra file để kiểm bằng
 *     `objdump -D -b binary -m aarch64 file` trên máy có binutils ARM.
 * Trên kiến trúc không phải x86-64, jit_x64.c bị vô hiệu hoá ở t_jit_maybe/
 * t_jit_osr/t_jit_osr_main nên sẽ KHÔNG phát mã x86 mà chạy interpreter.
 *
 * Nguồn encoding theo ARM ARM (little-endian, mỗi lệnh 1 uint32).
 */
#include "tvm.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

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
/* phát 1 lệnh (uint32, little-endian) */
static void a64_emit(A64Buf *b, uint32_t w) {
    if (!a64_grow(b)) return;
    b->buf[b->len]     = (uint8_t)(w);
    b->buf[b->len + 1] = (uint8_t)(w >> 8);
    b->buf[b->len + 2] = (uint8_t)(w >> 16);
    b->buf[b->len + 3] = (uint8_t)(w >> 24);
    b->len += 4;
}
static void a64_free(A64Buf *b) { free(b->buf); b->buf = NULL; b->cap = b->len = 0; }

/* ---------- emitters ---------- */
static void a64_ret(A64Buf *b) { a64_emit(b, 0xD65F03C0u); }                       /* ret */
static void a64_add(A64Buf *b, int rd, int rn, int rm) { a64_emit(b, 0x8B000000u | ((uint32_t)rm << 16) | ((uint32_t)rn << 5) | (uint32_t)rd); }
static void a64_sub(A64Buf *b, int rd, int rn, int rm) { a64_emit(b, 0xCB000000u | ((uint32_t)rm << 16) | ((uint32_t)rn << 5) | (uint32_t)rd); }
static void a64_mul(A64Buf *b, int rd, int rn, int rm) { a64_emit(b, 0x9B000000u | ((uint32_t)rm << 16) | (31u << 10) | ((uint32_t)rn << 5) | (uint32_t)rd); } /* madd rd,rm,rn,xzr */
static void a64_ldr(A64Buf *b, int rt, int rn, int off) { a64_emit(b, 0xF9400000u | ((uint32_t)(off >> 3) << 10) | ((uint32_t)rn << 5) | (uint32_t)rt); }
static void a64_str(A64Buf *b, int rt, int rn, int off) { a64_emit(b, 0xF9000000u | ((uint32_t)(off >> 3) << 10) | ((uint32_t)rn << 5) | (uint32_t)rt); }
static void a64_movz(A64Buf *b, int rd, int imm16, int shift) { a64_emit(b, 0xD2800000u | ((uint32_t)(shift >> 4) << 21) | ((uint32_t)(imm16 & 0xFFFF) << 5) | (uint32_t)rd); }
static void a64_movk(A64Buf *b, int rd, int imm16, int shift) { a64_emit(b, 0xF2800000u | ((uint32_t)(shift >> 4) << 21) | ((uint32_t)(imm16 & 0xFFFF) << 5) | (uint32_t)rd); }
static void a64_cmp(A64Buf *b, int rn, int rm) { a64_emit(b, 0xEB000000u | ((uint32_t)rm << 16) | ((uint32_t)rn << 5) | 31u); } /* subs xzr, rn, rm */
static void a64_bcond(A64Buf *b, int cond, int off_words) { a64_emit(b, 0x54000000u | (((uint32_t)off_words & 0x7FFFFu) << 5) | (uint32_t)cond); }
static void a64_bl(A64Buf *b, int off_words) { a64_emit(b, 0x94000000u | ((uint32_t)off_words & 0x3FFFFFFu)); }
static void a64_cbz(A64Buf *b, int rd, int off_words) { a64_emit(b, 0xB4000000u | (((uint32_t)off_words & 0x7FFFFu) << 5) | (uint32_t)rd); }
static void a64_cbnz(A64Buf *b, int rd, int off_words) { a64_emit(b, 0xB5000000u | (((uint32_t)off_words & 0x7FFFFu) << 5) | (uint32_t)rd); }
static void a64_blr(A64Buf *b, int rn) { a64_emit(b, 0xD63F0000u | ((uint32_t)rn << 5)); }
/* float64 */
static void a64_ldr_d(A64Buf *b, int rt, int rn, int off) { a64_emit(b, 0xFD400000u | ((uint32_t)(off >> 3) << 10) | ((uint32_t)rn << 5) | (uint32_t)rt); }
static void a64_str_d(A64Buf *b, int rt, int rn, int off) { a64_emit(b, 0xFD000000u | ((uint32_t)(off >> 3) << 10) | ((uint32_t)rn << 5) | (uint32_t)rt); }
static void a64_fadd(A64Buf *b, int rd, int rn, int rm) { a64_emit(b, 0x1E602800u | ((uint32_t)rm << 16) | ((uint32_t)rn << 5) | (uint32_t)rd); }
static void a64_fsub(A64Buf *b, int rd, int rn, int rm) { a64_emit(b, 0x1E603800u | ((uint32_t)rm << 16) | ((uint32_t)rn << 5) | (uint32_t)rd); }
static void a64_fmul(A64Buf *b, int rd, int rn, int rm) { a64_emit(b, 0x1E606800u | ((uint32_t)rm << 16) | ((uint32_t)rn << 5) | (uint32_t)rd); }
static void a64_fdiv(A64Buf *b, int rd, int rn, int rm) { a64_emit(b, 0x1E601800u | ((uint32_t)rm << 16) | ((uint32_t)rn << 5) | (uint32_t)rd); }
static void a64_scvtf_d(A64Buf *b, int rd, int rn) { a64_emit(b, 0x9E620000u | ((uint32_t)rn << 5) | (uint32_t)rd); }
static void a64_fcvtzs(A64Buf *b, int rd, int rn) { a64_emit(b, 0x9E7E0000u | ((uint32_t)rn << 5) | (uint32_t)rd); }
/* frame */
static void a64_prologue(A64Buf *b) { a64_emit(b, 0xA9BF7BFD); a64_emit(b, 0xAA0003FD); } /* stp x29,x30,[sp,#-16]! ; mov x29,sp */
static void a64_epilogue(A64Buf *b) { a64_emit(b, 0xA8C17BFD); a64_ret(b); }                /* ldp x29,x30,[sp],#16 ; ret */

/* ---------- kernel mẫu (entry ABI: x0=in*, x1=ctx*, x0=ret) ---------- */
/* k1: return in[0] + in[1] (int64) */
static void build_k1(A64Buf *b) {
    a64_ldr(b, 9, 0, 0);        /* ldr x9, [x0]      */
    a64_ldr(b, 10, 0, 8);       /* ldr x10, [x0,#8]  */
    a64_add(b, 0, 9, 10);       /* add x0, x9, x10   */
    a64_ret(b);
}
/* k2: return (double)in[0] + (double)in[1] */
static void build_k2(A64Buf *b) {
    a64_ldr_d(b, 0, 0, 0);      /* ldr d0, [x0]      */
    a64_ldr_d(b, 1, 0, 8);      /* ldr d1, [x0,#8]   */
    a64_fadd(b, 0, 0, 1);       /* fadd d0, d0, d1   */
    a64_ret(b);
}
/* k3: add có frame (prologue/body/epilogue) */
static void build_k3(A64Buf *b) {
    a64_prologue(b);
    a64_add(b, 0, 0, 1);        /* add x0, x0, x1    */
    a64_epilogue(b);
}

/* ---------- oracle byte (viết tay từ ARM ARM) ---------- */
static const uint8_t k1_exp[16] = {
    /* 0xF9400009 */ 0x09,0x00,0x40,0xF9,
    /* 0xF940040A */ 0x0A,0x04,0x40,0xF9,
    /* 0x8B0A0120 */ 0x20,0x01,0x0A,0x8B,
    /* 0xD65F03C0 */ 0xC0,0x03,0x5F,0xD6
};
static const uint8_t k2_exp[16] = {
    /* 0xFD400000 */ 0x00,0x00,0x40,0xFD,
    /* 0xFD400401 */ 0x01,0x04,0x40,0xFD,
    /* 0x1E612800 */ 0x00,0x28,0x61,0x1E,
    /* 0xD65F03C0 */ 0xC0,0x03,0x5F,0xD6
};
static const uint8_t k3_exp[20] = {
    /* 0xA9BF7BFD */ 0xFD,0x7B,0xBF,0xA9,
    /* 0xAA0003FD */ 0xFD,0x03,0x00,0xAA,
    /* 0x8B010000 */ 0x00,0x00,0x01,0x8B,
    /* 0xA8C17BFD */ 0xFD,0x7B,0xC1,0xA8,
    /* 0xD65F03C0 */ 0xC0,0x03,0x5F,0xD6
};

static int check(A64Buf *b, const uint8_t *exp, size_t n) {
    return !b->err && b->len == n && memcmp(b->buf, exp, n) == 0;
}

/* tự kiểm chứng emitter: 1 = khớp toàn bộ oracle, 0 = lỗi */
int t_arm64_selftest(void) {
    A64Buf b; memset(&b, 0, sizeof(b));
    int ok;
    build_k1(&b); ok = check(&b, k1_exp, 16); a64_free(&b); if (!ok) return 0;
    memset(&b, 0, sizeof(b));
    build_k2(&b); ok = check(&b, k2_exp, 16); a64_free(&b); if (!ok) return 0;
    memset(&b, 0, sizeof(b));
    build_k3(&b); ok = check(&b, k3_exp, 20); a64_free(&b); if (!ok) return 0;
    return 1;
}

/* ghi 3 kernel mẫu (48 byte) ra file để kiểm bằng objdump -m aarch64; trả 1/0 */
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
