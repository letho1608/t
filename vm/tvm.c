/* tvm.c — T bytecode VM v1: stack machine + frames + refcount + try + natives. */
#if !defined(_WIN32) && !defined(_DEFAULT_SOURCE)
#define _DEFAULT_SOURCE
#endif
#include "tvm.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <sys/types.h>
#include <sys/stat.h>
#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <winsock2.h>
#include <ws2tcpip.h>
#include <windows.h>
#include <direct.h>
#include <io.h>
#include <process.h>
#pragma comment(lib, "ws2_32.lib")
#else
#include <unistd.h>
#include <dirent.h>
#include <sys/wait.h>
#include <poll.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <dlfcn.h>
#include <pthread.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <netdb.h>
#endif

#ifdef _WIN32
#define THREAD_LOCAL __declspec(thread)
#else
#define THREAD_LOCAL __thread
#endif

/* Vòng lặp interpreter có setjmp/longjmp (try/except). GCC >= 12 (mingw-w64)
   inline hỏng vòng lặp này: try/except im lặng khi build -O1/-O2/-Os
   (đã repro: chỉ -fno-inline là hết). Vì vậy tắt inline cho cả file. */
#if defined(__GNUC__) && !defined(__clang__) && !defined(__INTEL_COMPILER)
#pragma GCC optimize("no-inline")
#endif


#define STACK_INIT 1024
#define FRAME_INIT 64
#define HAND_INIT 16

static THREAD_LOCAL jmp_buf top_jb;
static int ten_eq(const TTensor *a, const TTensor *b);
static THREAD_LOCAL int in_top;
static void free_frame(TVM *vm);
static TFrame *push_frame(TVM *vm);
static THREAD_LOCAL TVM *cur_vm;   /* VM đang chạy (GC track dùng) */
static void gc_track(TVM *vm, int type, void *ptr);
static void gc_collect(TVM *vm);

int tvm_pc_line(const TProg *p, size_t pc, const char **file, uint32_t *line) {
    uint32_t lo = 0, hi = p->nlines;
    if (!p->nlines || pc == 0) return 0;
    pc--;                                   /* pc đã qua byte opcode */
    if (pc < p->lpc[0]) return 0;
    while (hi - lo > 1) {                   /* mục cuối có lpc <= pc */
        uint32_t mid = (lo + hi) / 2;
        if (p->lpc[mid] <= pc) lo = mid; else hi = mid;
    }
    *line = p->lline[lo];
    *file = p->lfile[lo] < p->nlfiles ? p->lfiles[p->lfile[lo]] : "?";
    return 1;
}

static void die(TVM *vm, const char *msg) {
    vm->errpc = vm->pc;
    vm->errprog = vm->prog;
    strncpy(vm->errmsg, msg, sizeof(vm->errmsg) - 1);
    vm->errmsg[sizeof(vm->errmsg) - 1] = 0;
    if (vm->nhandlers > 0) {
        THandler h = vm->handlers[--vm->nhandlers];
        while (vm->sp > h.sp) tval_free(&vm->stack[--vm->sp]);
        while (vm->nframes > h.nframes) free_frame(vm);
        vm->pc = h.pc;
        vm->stop_nf = h.stop_nf;
        if (h.prog) vm->prog = h.prog;
        longjmp(h.jb, 1);
    }
    vm->err = 1;
    if (in_top) longjmp(top_jb, 1);
    exit(1);
}

/* ---------- pool khối nhỏ ----------
   Object sống ngắn (GCNode, vỏ list/dict/native, cell, chuỗi ngắn) được cấp/nhả liên
   tục; malloc/free của CRT (nhất là Windows) đắt hơn nhiều so với lấy từ free-list.
   Lớp 16 byte tới PK_MAX; nhả phải đưa đúng kích thước lúc cấp. Bộ nhớ của pool
   không trả lại hệ điều hành (dùng lại cho lần cấp sau). */
#define PK_MAX 256
typedef struct PkFree { struct PkFree *next; } PkFree;
static THREAD_LOCAL PkFree *pk_list[PK_MAX / 16 + 1];
static void *pk_alloc(size_t sz) {
    size_t c = (sz + 15) >> 4;
    PkFree *p;
    if (c == 0 || c > PK_MAX / 16) return malloc(sz);
    p = pk_list[c];
    if (p) { pk_list[c] = p->next; return p; }
    {
        size_t bs = c << 4, nb = 32768 / bs;
        char *ch = (char*)malloc(nb * bs);
        if (!ch) return NULL;
        for (size_t i = nb - 1; i >= 1; i--) {
            PkFree *f = (PkFree*)(ch + i * bs);
            f->next = pk_list[c]; pk_list[c] = f;
        }
        return ch;
    }
}
static void pk_free(void *p, size_t sz) {
    size_t c = (sz + 15) >> 4;
    if (!p) return;
    if (c == 0 || c > PK_MAX / 16) { free(p); return; }
    ((PkFree*)p)->next = pk_list[c];
    pk_list[c] = (PkFree*)p;
}

/* ---------- GC mark-sweep (chu trình; refcount xử lý còn lại) ---------- */
#define GC_THRESHOLD_DEFAULT 65536
static size_t gc_threshold = GC_THRESHOLD_DEFAULT;   /* TVM_GCTHRESH=n để thử GC dày */
#define GC_THRESHOLD gc_threshold

static void gc_unlink(TVM *vm, GCNode *nd) {
    (void)vm;
    if (nd->prev) nd->prev->next = nd->next;
    else vm->root->gc_head = nd->next;
    if (nd->next) nd->next->prev = nd->prev;
    pk_free(nd, sizeof(GCNode));
    vm->root->gc_tracked--;
}

static void gc_track(TVM *vm, int type, void *ptr) {
    GCNode *nd = (GCNode*)pk_alloc(sizeof(GCNode));
    if (!nd) return;
    nd->type = type; nd->ptr = ptr;
    nd->prev = NULL; nd->next = vm->root->gc_head;
    if (nd->next) nd->next->prev = nd;
    vm->root->gc_head = nd;
    if (type == T_LIST) ((TList*)ptr)->gc_node = nd;
    else if (type == T_DICT) ((TDict*)ptr)->gc_node = nd;
    else if (type == T_FN) ((TFn*)ptr)->gc_node = nd;
    else if (type == T_NATIVE) ((TNative*)ptr)->gc_node = nd;
    vm->root->gc_tracked++;
    vm->root->gc_allocs++;
}

static void gc_mark_val(TVM *vm, TVal *v) {
    size_t i;
    switch (v->t) {
    case T_LIST: {
        TList *l = v->u.list;
        if (!l || l->gc_mark) break;
        l->gc_mark = 1;
        for (i = 0; i < l->len; i++) gc_mark_val(vm, &l->items[i]);
        break;
    }
    case T_DICT: {
        TDict *d = v->u.dict;
        if (!d || d->gc_mark) break;
        d->gc_mark = 1;
        for (i = 0; i < d->len; i++) {
            gc_mark_val(vm, &d->keys[i]);
            gc_mark_val(vm, &d->vals[i]);
        }
        if (d->meths) { TVal mv; mv.t = T_DICT; mv.u.dict = d->meths; gc_mark_val(vm, &mv); }
        if (d->fields) { TVal fv; fv.t = T_LIST; fv.u.list = d->fields; gc_mark_val(vm, &fv); }
        break;
    }
    case T_FN: {
        TFn *f = v->u.fn;
        if (!f || f->gc_mark) break;
        f->gc_mark = 1;
        for (i = 0; i < f->ncap; i++) gc_mark_val(vm, &f->upvals[i]);
        {
            TFn **cs; int nc = t_jit_callees(f, &cs);
            for (int q = 0; q < nc; q++) { TVal cv; cv.t = T_FN; cv.u.fn = cs[q]; gc_mark_val(vm, &cv); }
        }
        break;
    }
    case T_NATIVE: {
        TNative *t = v->u.nat;
        if (!t || t->gc_mark) break;
        t->gc_mark = 1;
        if (t->has_obj) gc_mark_val(vm, &t->obj);
        break;
    }
    case T_CELL:
        if (v->u.cell) gc_mark_val(vm, &v->u.cell->val);
        break;
    default: break;
    }
}

static void gc_untrack(TVM *vm, int type, void *ptr) {
    GCNode *nd = NULL;
    if (type == T_LIST) nd = ((TList*)ptr)->gc_node;
    else if (type == T_DICT) nd = ((TDict*)ptr)->gc_node;
    else if (type == T_FN) nd = ((TFn*)ptr)->gc_node;
    else if (type == T_NATIVE) nd = ((TNative*)ptr)->gc_node;
    if (nd) {
        if (type == T_LIST) ((TList*)ptr)->gc_node = NULL;
        else if (type == T_DICT) ((TDict*)ptr)->gc_node = NULL;
        else if (type == T_FN) ((TFn*)ptr)->gc_node = NULL;
        else if (type == T_NATIVE) ((TNative*)ptr)->gc_node = NULL;
        gc_unlink(vm, nd);
    }
}

static int *gc_refc_of(GCNode *nd) {
    switch (nd->type) {
    case T_LIST: return (int*)&((TList*)nd->ptr)->refc;
    case T_DICT: return (int*)&((TDict*)nd->ptr)->refc;
    case T_FN: return (int*)&((TFn*)nd->ptr)->refc;
    default: return (int*)&((TNative*)nd->ptr)->refc;
    }
}
static int gc_take_mark(GCNode *nd) {
    int *m;
    switch (nd->type) {
    case T_LIST: m = &((TList*)nd->ptr)->gc_mark; break;
    case T_DICT: m = &((TDict*)nd->ptr)->gc_mark; break;
    case T_FN: m = &((TFn*)nd->ptr)->gc_mark; break;
    default: m = &((TNative*)nd->ptr)->gc_mark; break;
    }
    int was = *m;
    *m = 0;
    return was;
}
/* pha 3: nhả mọi tham chiếu con (rác khác đang được giữ nên không bị free lồng) */
static void gc_clear_contents(GCNode *nd) {
    size_t i;
    if (nd->type == T_LIST) {
        TList *l = (TList*)nd->ptr;
        size_t n = l->len;
        l->len = 0;
        for (i = 0; i < n; i++) tval_free(&l->items[i]);
    } else if (nd->type == T_DICT) {
        TDict *d = (TDict*)nd->ptr;
        size_t n = d->len;
        d->len = 0;
        for (i = 0; i < n; i++) {
            if (d->kind != DK_REC) tval_free(&d->keys[i]);
            tval_free(&d->vals[i]);
        }
        if (d->meths) { TVal mv; mv.t = T_DICT; mv.u.dict = d->meths; d->meths = NULL; tval_free(&mv); }
        if (d->fields) { TVal fv; fv.t = T_LIST; fv.u.list = d->fields; d->fields = NULL; tval_free(&fv); }
    } else if (nd->type == T_FN) {
        TFn *f = (TFn*)nd->ptr;
        uint16_t n = f->ncap;
        t_jit_free_fn(f);
        f->ncap = 0;
        for (uint16_t q = 0; q < n; q++) tval_free(&f->upvals[q]);
    } else {
        TNative *t = (TNative*)nd->ptr;
        if (t->has_obj) { t->has_obj = 0; tval_free(&t->obj); }
    }
}
/* pha 4: giải phóng vỏ object (nội dung đã nhả) */
static void gc_free_shell(int type, void *ptr) {
    if (type == T_LIST) { TList *l = (TList*)ptr; free(l->items); pk_free(l, sizeof(TList)); }
    else if (type == T_DICT) {
        TDict *d = (TDict*)ptr;
        if (d->kind != DK_REC) free(d->keys);
        free(d->vals); free(d->rname); free(d->idx); pk_free(d, sizeof(TDict));
    } else if (type == T_FN) { TFn *f = (TFn*)ptr; free(f->upvals); free(f); }
    else pk_free(ptr, sizeof(TNative));
}
static void gc_collect(TVM *vm) {
    size_t i, ng = 0, cap = 0;
    GCNode **gar = NULL;
    /* đang trong lời gọi lồng từ native (map/sort/method...): biến cục bộ C có
       thể giữ object chưa nằm trên stack -> để lần sau ở tầng ngoài cùng */
    if (vm->stop_nf != 0) return;
    vm->root->gc_allocs = 0;
    vm->root->gc_runs++;
    for (i = 0; i < vm->sp; i++) gc_mark_val(vm, &vm->stack[i]);
    for (i = 0; i < vm->nframes; i++) {
        TFrame *f = &vm->frames[i];
        size_t j;
        for (j = 0; j < f->nlocals; j++) gc_mark_val(vm, &f->locals[j]);
        if (f->fn) { TVal v; v.t = T_FN; v.u.fn = f->fn; gc_mark_val(vm, &v); }
    }
    for (i = 0; i < vm->root->nglobals; i++)
        gc_mark_val(vm, &vm->root->globals[i]);
    for (TProg *m = vm->root->next_mod; m; m = m->next_mod) {
        for (i = 0; i < m->nglobals; i++) gc_mark_val(vm, &m->globals[i]);
        gc_mark_val(vm, &m->mod_rec);
    }
    {
        TFn **rf; int nr = t_jit_root_fns(&rf);
        for (int q = 0; q < nr; q++) { TVal fv; fv.t = T_FN; fv.u.fn = rf[q]; gc_mark_val(vm, &fv); }
    }
    /* pha 1: gom rác (không đánh dấu), xoá dấu của object sống */
    for (GCNode *nd = vm->root->gc_head; nd; nd = nd->next) {
        if (gc_take_mark(nd)) continue;
        if (ng >= cap) {
            cap = cap ? cap * 2 : 256;
            gar = (GCNode**)realloc(gar, cap * sizeof(GCNode*));
            if (!gar) exit(1);
        }
        gar[ng++] = nd;
    }
    if (!ng) { free(gar); return; }
    /* pha 2: giữ ref mọi rác -> nhả nội dung không làm free lồng nhau */
    for (i = 0; i < ng; i++) (*gc_refc_of(gar[i]))++;
    /* pha 3: nhả nội dung */
    for (i = 0; i < ng; i++) gc_clear_contents(gar[i]);
    /* pha 4: tháo node khỏi danh sách GC + giải phóng vỏ */
    for (i = 0; i < ng; i++) {
        GCNode *nd = gar[i];
        int type = nd->type;
        void *ptr = nd->ptr;
        if (type == T_LIST) ((TList*)ptr)->gc_node = NULL;
        else if (type == T_DICT) ((TDict*)ptr)->gc_node = NULL;
        else if (type == T_FN) ((TFn*)ptr)->gc_node = NULL;
        else ((TNative*)ptr)->gc_node = NULL;
        gc_unlink(vm, nd);
        gc_free_shell(type, ptr);
    }
    free(gar);
}

/* ---------- bảng opcode dùng chung (debugger, JIT, công cụ) ---------- */
static const char *const OPNAMES[] = {
    "PUSH_NIL",
    "PUSH_BOOL",
    "PUSH_INT",
    "PUSH_FLOAT",
    "PUSH_STR",
    "NEW_LIST",
    "NEW_TENSOR",
    "NEW_DICT",
    "ADD",
    "SUB",
    "MUL",
    "DIV",
    "MOD",
    "POW",
    "NEG",
    "NOT",
    "EQ",
    "NE",
    "LT",
    "GT",
    "LE",
    "GE",
    "CONTAINS",
    "MATCHGLOB",
    "TDOT",
    "IDX",
    "SETIDX",
    "SLICE",
    "LEN",
    "PRINT",
    "JMP",
    "JMPF",
    "JMPT",
    "CALL",
    "RET",
    "LOAD",
    "STORE",
    "DECL",
    "LOADG",
    "STOREG",
    "DECLG",
    "MKFN",
    "CAPLOCAL",
    "CAPUP",
    "CAPG",
    "LOADU",
    "STOREU",
    "BIND",
    "BINDG",
    "RANGE",
    "ATTR",
    "RAISE",
    "TRY_PUSH",
    "TRY_POP",
    "FOR_PREP",
    "POP",
    "DUP",
    "INC_L",
    "INC_G",
    "DEC_L",
    "DEC_G",
    "ADDI_L",
    "ADDI_G",
    "ADDS_L",
    "ADDS_G",
    "SUBS_L",
    "SUBS_G",
    "INCB_L",
    "INCB_G",
    "IN",
    "CALLM",
    "SETATTR",
    "BAND",
    "BOR",
    "BXOR",
    "BNOT",
    "SHL",
    "SHR",
    "NOP",
    "HALT"
};
/* bảng tên phải đủ mọi opcode (thêm opcode vào tvm.h mà quên ở đây -> lỗi biên dịch) */
typedef char opnames_complete[(sizeof(OPNAMES) / sizeof(OPNAMES[0]) == OP_HALT - OP_PUSH_NIL + 1) ? 1 : -1];
const char *tvm_opname(uint8_t op) {
    if (op < OP_PUSH_NIL || op > OP_HALT) return "??";
    return OPNAMES[op - OP_PUSH_NIL];
}
/* độ dài lệnh tại pc (khớp op_size trong vm/tcomp.py và self/tcomp.t) */
size_t tvm_oplen(const uint8_t *c, size_t n, size_t pc) {
    uint32_t u32;
    if (pc >= n) return 1;
    switch (c[pc]) {
    case OP_PUSH_BOOL: case OP_PRINT: case OP_CALL: case OP_RANGE:
        return 2;
    case OP_PUSH_INT: case OP_PUSH_FLOAT:
        return 9;
    case OP_PUSH_STR: case OP_ATTR: case OP_SETATTR:
        if (pc + 5 > n) return 1;
        memcpy(&u32, c + pc + 1, 4);
        return 5 + (size_t)u32;
    case OP_CALLM:
        if (pc + 6 > n) return 1;
        memcpy(&u32, c + pc + 2, 4);
        return 6 + (size_t)u32;
    case OP_NEW_LIST: case OP_NEW_TENSOR: case OP_NEW_DICT:
    case OP_LOAD: case OP_STORE: case OP_DECL:
    case OP_LOADG: case OP_STOREG: case OP_DECLG:
    case OP_LOADU: case OP_STOREU: case OP_BIND: case OP_BINDG:
    case OP_CAPLOCAL: case OP_CAPUP: case OP_CAPG:
    case OP_INC_L: case OP_INC_G: case OP_DEC_L: case OP_DEC_G:
    case OP_INCB_L: case OP_INCB_G:
        return 3;
    case OP_JMP: case OP_JMPF: case OP_JMPT: case OP_TRY_PUSH:
    case OP_ADDS_L: case OP_ADDS_G: case OP_SUBS_L: case OP_SUBS_G:
        return 5;
    case OP_MKFN:
        return 10;
    case OP_ADDI_L: case OP_ADDI_G:
        return 11;
    default:
        return 1;
    }
}

/* ---------- stack ---------- */
static void stack_grow(TVM *vm) {
    vm->cap *= 2;
    vm->stack = (TVal*)realloc(vm->stack, vm->cap * sizeof(TVal));
    if (!vm->stack) exit(1);
}
TVM_INLINE void push(TVM *vm, TVal v) {
    if (vm->sp >= vm->cap) stack_grow(vm);
    vm->stack[vm->sp++] = v;
}
TVM_INLINE TVal pop(TVM *vm) { return vm->stack[--vm->sp]; }
/* nil/bool/int/float không đếm tham chiếu: bỏ qua lời gọi ở đường nóng */
#define TV_FREE(p) do { if ((p)->t > T_FLOAT) tval_free(p); } while (0)
#define TV_RETAIN(p) do { if ((p)->t > T_FLOAT) tval_retain(p); } while (0)
void tvm_push(TVM *vm, TVal v) { push(vm, v); }

/* ---------- readers ---------- */
static uint16_t rd_u16(uint8_t *c, size_t *pc) {
    uint16_t v; memcpy(&v, c + *pc, 2); *pc += 2; return v;
}
static int32_t rd_i32(uint8_t *c, size_t *pc) {
    int32_t v; memcpy(&v, c + *pc, 4); *pc += 4; return v;
}
static int64_t rd_i64(uint8_t *c, size_t *pc) {
    int64_t v; memcpy(&v, c + *pc, 8); *pc += 8; return v;
}
static double rd_f64(uint8_t *c, size_t *pc) {
    double v; memcpy(&v, c + *pc, 8); *pc += 8; return v;
}
static uint32_t rd_u32(uint8_t *c, size_t *pc) {
    uint32_t v; memcpy(&v, c + *pc, 4); *pc += 4; return v;
}

/* ---------- ctors ---------- */
static TVal mk_nil(void) { TVal v; v.t = T_NIL; return v; }
static void fill_undef(TVal *p, size_t n) {
    size_t i;
    for (i = 0; i < n; i++) p[i].t = T_UNDEF;
}
static TVal mk_bool(int b) { TVal v; v.t = T_BOOL; v.u.b = b ? 1 : 0; return v; }
static TVal mk_int(int64_t i) { TVal v; v.t = T_INT; v.u.i = i; return v; }
static TVal mk_float(double f) { TVal v; v.t = T_FLOAT; v.u.f = f; return v; }
/* chuỗi = MỘT khối: [refc int32][hash int32 (0 = chưa tính)][byte... \0];
   u.str.refc trỏ đầu khối (free khối qua refc), u.str.s trỏ vào phần byte */
static TVal mk_str_uninit(size_t n) {
    TVal v; v.t = T_STR;
    int32_t *blk = (int32_t*)pk_alloc(8 + n + 1);
    if (!blk) exit(1);
    blk[0] = 1; blk[1] = 0;
    v.u.str.refc = blk;
    v.u.str.s = (char*)(blk + 2);
    v.u.str.s[n] = 0;
    v.u.str.n = n;
    return v;
}
/* chuỗi 1 byte dùng chung (bất biến): quét chuỗi từng ký tự s[i] / s[i:i+1] không
   cấp phát. Cache giữ 1 ref vĩnh viễn. Ai cần sửa nội dung phải dùng mk_str_uninit. */
static TVal str1_cache[256];
static TVal mk_str_n(const char *s, size_t n) {
    if (n == 1) {
        TVal *c = &str1_cache[(unsigned char)s[0]];
        if (c->t != T_STR) { *c = mk_str_uninit(1); c->u.str.s[0] = s[0]; }
        (*c->u.str.refc)++;
        return *c;
    }
    TVal v = mk_str_uninit(n);
    if (n) memcpy(v.u.str.s, s, n);
    return v;
}
static TVal mk_str(const char *s) { return mk_str_n(s, strlen(s)); }
static TList *list_new(size_t cap) {
    TList *l = (TList*)pk_alloc(sizeof(TList));
    l->refc = 1; l->gc_mark = 0; l->len = 0; l->cap = cap ? cap : 4;
    l->items = (TVal*)malloc(l->cap * sizeof(TVal));
    if (cur_vm) gc_track(cur_vm, T_LIST, l);
    return l;
}
static void list_push(TList *l, TVal v) {
    if (l->len >= l->cap) {
        l->cap *= 2;
        l->items = (TVal*)realloc(l->items, l->cap * sizeof(TVal));
    }
    l->items[l->len++] = v;
}

/* ---------- retain/free ---------- */
void tval_retain(TVal *v) {
    if (!v) return;
    switch (v->t) {
    case T_STR: if (v->u.str.refc) (*v->u.str.refc)++; break;
    case T_LIST: if (v->u.list) v->u.list->refc++; break;
    case T_TENSOR: if (v->u.ten) v->u.ten->refc++; break;
    case T_DICT: if (v->u.dict) v->u.dict->refc++; break;
    case T_FN: if (v->u.fn) v->u.fn->refc++; break;
    case T_NATIVE:
        if (v->u.nat) {
            v->u.nat->refc++;
            if (v->u.nat->has_obj) tval_retain(&v->u.nat->obj);
        }
        break;
    case T_FILE: if (v->u.file) v->u.file->refc++; break;
    case T_CELL: if (v->u.cell) v->u.cell->refc++; break;
    case T_UNDEF: break;
    default: break;
    }
}
/* pool buffer dữ liệu tensor lớn: vòng lặp số học tạo/huỷ liên tục các tensor cùng
   kích thước; dùng lại buffer vừa nhả tránh cấp phát + page fault mỗi phép toán */
#define TBUF_MIN 1024
#define TBUF_SLOTS 16
static THREAD_LOCAL double *tbuf_p[TBUF_SLOTS];
static THREAD_LOCAL size_t tbuf_n[TBUF_SLOTS];
static THREAD_LOCAL int tbuf_cnt;
static double *tbuf_get(size_t n) {
    if (n >= TBUF_MIN)
        for (int i = tbuf_cnt - 1; i >= 0; i--)
            if (tbuf_n[i] == n) {
                double *p = tbuf_p[i];
                tbuf_cnt--;
                tbuf_p[i] = tbuf_p[tbuf_cnt]; tbuf_n[i] = tbuf_n[tbuf_cnt];
                return p;
            }
    return (double*)malloc(sizeof(double) * (n ? n : 1));
}
static void tbuf_put(double *p, size_t n) {
    if (n >= TBUF_MIN) {
        if (tbuf_cnt == TBUF_SLOTS) {          /* đầy: bỏ buffer cũ nhất */
            free(tbuf_p[0]);
            memmove(tbuf_p, tbuf_p + 1, sizeof(tbuf_p[0]) * (TBUF_SLOTS - 1));
            memmove(tbuf_n, tbuf_n + 1, sizeof(tbuf_n[0]) * (TBUF_SLOTS - 1));
            tbuf_cnt--;
        }
        tbuf_p[tbuf_cnt] = p; tbuf_n[tbuf_cnt++] = n;
        return;
    }
    free(p);
}
static void tbuf_clear(void) {
    while (tbuf_cnt) free(tbuf_p[--tbuf_cnt]);
}

void tval_free(TVal *v) {
    size_t i;
    if (!v) return;
    switch (v->t) {
    case T_STR:
        if (v->u.str.refc && --(*v->u.str.refc) == 0) {
            pk_free(v->u.str.refc, 8 + v->u.str.n + 1);   /* khối chung (refc + hash + byte) */
        }
        break;
    case T_LIST:
        if (v->u.list && --(v->u.list->refc) == 0) {
            if (cur_vm) gc_untrack(cur_vm, T_LIST, v->u.list);
            for (i = 0; i < v->u.list->len; i++) tval_free(&v->u.list->items[i]);
            free(v->u.list->items); pk_free(v->u.list, sizeof(TList));
        }
        break;
    case T_TENSOR:
        if (v->u.ten && --(v->u.ten->refc) == 0) {
            TTensor *b = v->u.ten->base;
            if (b) {                      /* view: nhả tensor sở hữu bộ nhớ */
                TVal bv; bv.t = T_TENSOR; bv.u.ten = b;
                if (v->u.ten->sv) tbuf_put(v->u.ten->d, v->u.ten->len);   /* cache của view theo bước */
                free(v->u.ten);
                tval_free(&bv);
            } else {
                free(v->u.ten->raw);
                tbuf_put(v->u.ten->d, v->u.ten->len); free(v->u.ten);
            }
        }
        break;
    case T_DICT:
        if (v->u.dict && --(v->u.dict->refc) == 0) {
            TDict *dd = v->u.dict;
            if (cur_vm) gc_untrack(cur_vm, T_DICT, dd);
            for (i = 0; i < dd->len; i++) {
                if (dd->kind != DK_REC) tval_free(&dd->keys[i]);
                tval_free(&dd->vals[i]);
            }
            if (dd->meths) {
                TVal mv; mv.t = T_DICT; mv.u.dict = dd->meths;
                dd->meths = NULL;
                tval_free(&mv);
            }
            if (dd->fields) {
                TVal fv; fv.t = T_LIST; fv.u.list = dd->fields;
                dd->fields = NULL;
                tval_free(&fv);
            }
            free(dd->rname);
            free(dd->idx);
            if (dd->kind != DK_REC) free(dd->keys);
            free(dd->vals); pk_free(dd, sizeof(TDict));
        }
        break;
    case T_FN:
        if (v->u.fn && --(v->u.fn->refc) == 0) {
            if (cur_vm) gc_untrack(cur_vm, T_FN, v->u.fn);
            t_jit_free_fn(v->u.fn);
            if (v->u.fn->upvals) {
                for (uint16_t i = 0; i < v->u.fn->ncap; i++)
                    tval_free(&v->u.fn->upvals[i]);
                free(v->u.fn->upvals);
            }
            free(v->u.fn);
        }
        break;
    case T_NATIVE:
        if (v->u.nat && --(v->u.nat->refc) == 0) {
            if (cur_vm) gc_untrack(cur_vm, T_NATIVE, v->u.nat);
            if (v->u.nat->has_obj) tval_free(&v->u.nat->obj);
            pk_free(v->u.nat, sizeof(TNative));
        }
        break;
    case T_FILE:
        if (v->u.file && --(v->u.file->refc) == 0) {
            if (v->u.file->h && !v->u.file->closed) fclose(v->u.file->h);
            free(v->u.file->path); free(v->u.file);
        }
        break;
    case T_CELL:
        if (v->u.cell && --(v->u.cell->refc) == 0) {
            tval_free(&v->u.cell->val);
            pk_free(v->u.cell, sizeof(TCell));
        }
        break;
    case T_UNDEF: break;
    default: break;
    }
    v->t = T_NIL;
}

/* ---------- truthy/eq/fmt ---------- */
/* tên kiểu của record (lưu ở kiểu DK_TYPE dùng chung) */
static const char *rec_name(const TDict *d) {
    if (d->kind == DK_REC) return (d->meths && d->meths->rname) ? d->meths->rname : "record";
    return d->rname ? d->rname : "record";
}
/* bản ghi module (load): field i <-> global exslots[i] của prog module */
TVM_INLINE TProg *rec_mod(const TDict *d) {
    return (d->kind == DK_REC && d->meths) ? d->meths->mod : NULL;
}
static void mod_pull_slow(TDict *d, size_t i, TProg *m);
TVM_INLINE void mod_pull(TDict *d, size_t i) {
    TProg *m = rec_mod(d);
    if (m) mod_pull_slow(d, i, m);       /* bản ghi thường (struct): không làm gì */
}
static void mod_pull_slow(TDict *d, size_t i, TProg *m) {
    if (!m || i >= m->nexports || i >= d->len || m->exslots[i] >= m->nglobals) return;
    TVal v = m->globals[m->exslots[i]];
    if (v.t == T_CELL) v = v.u.cell->val;
    tval_retain(&v);
    tval_free(&d->vals[i]);
    d->vals[i] = v;
}
/* field i của bản ghi module là let/fn? -> ghi báo lỗi như gán biến bất biến */
static int mod_ro(const TDict *d, size_t i, char *eb, size_t ebn) {
    TProg *m = rec_mod(d);
    if (!m || i >= m->nexports || i >= d->len || m->exmut[i]) return 0;
    snprintf(eb, ebn, "%.*s immutable (use mut)",
             (int)(d->keys[i].u.str.n > 60 ? 60 : d->keys[i].u.str.n), d->keys[i].u.str.s);
    return 1;
}
static void mod_pull_all(TDict *d) {
    if (!rec_mod(d)) return;
    for (size_t i = 0; i < d->len; i++) mod_pull(d, i);
}
static void mod_push(TDict *d, size_t i) {
    TProg *m = rec_mod(d);
    if (!m || i >= m->nexports || i >= d->len || m->exslots[i] >= m->nglobals) return;
    TVal *g = &m->globals[m->exslots[i]], v = d->vals[i];
    tval_retain(&v);
    if (g->t == T_CELL) { tval_free(&g->u.cell->val); g->u.cell->val = v; }
    else { tval_free(g); *g = v; }
}

int tval_truthy(const TVal *v) {
    size_t i;
    switch (v->t) {
    case T_NIL: return 0;
    case T_BOOL: return v->u.b;
    case T_INT: return v->u.i != 0;
    case T_FLOAT: return v->u.f != 0;
    case T_STR: return v->u.str.n != 0;
    case T_LIST: return v->u.list->len != 0;
    case T_TENSOR: return v->u.ten->len != 0;
    case T_DICT: return v->u.dict->kind >= DK_REC || v->u.dict->len != 0;
    case T_CELL: return tval_truthy(&v->u.cell->val);
    case T_UNDEF: return 0;
    default: return 1;
    }
    (void)i;
    return 1;
}
int tval_eq(const TVal *a, const TVal *b) {
    size_t i;
    if (a->t == T_UNDEF || b->t == T_UNDEF)
        return a->t == T_UNDEF && b->t == T_UNDEF;
    if (a->t == T_CELL) return tval_eq(&a->u.cell->val, b);
    if (b->t == T_CELL) return tval_eq(a, &b->u.cell->val);
    if (a->t == T_INT && b->t == T_FLOAT) return (double)a->u.i == b->u.f;
    if (a->t == T_FLOAT && b->t == T_INT) return a->u.f == (double)b->u.i;
    if (a->t != b->t) return 0;
    switch (a->t) {
    case T_NIL: return 1;
    case T_BOOL: return a->u.b == b->u.b;
    case T_INT: return a->u.i == b->u.i;
    case T_FLOAT: return a->u.f == b->u.f;
    case T_STR:
        return a->u.str.n == b->u.str.n &&
               memcmp(a->u.str.s, b->u.str.s, a->u.str.n) == 0;
    case T_LIST:
        if (a->u.list->len != b->u.list->len) return 0;
        for (i = 0; i < a->u.list->len; i++)
            if (!tval_eq(&a->u.list->items[i], &b->u.list->items[i])) return 0;
        return 1;
    case T_TENSOR:
        return ten_eq(a->u.ten, b->u.ten);
    case T_DICT: {
        if (a->u.dict->kind != b->u.dict->kind) return 0;
        if (a->u.dict->kind == DK_TYPE) return a->u.dict == b->u.dict;
        if (a->u.dict->kind == DK_REC && strcmp(rec_name(a->u.dict), rec_name(b->u.dict)) != 0) return 0;
        mod_pull_all(a->u.dict); mod_pull_all(b->u.dict);
        if (a->u.dict->len != b->u.dict->len) return 0;
        for (i = 0; i < a->u.dict->len; i++) {
            size_t j; int found = 0;
            for (j = 0; j < b->u.dict->len; j++) {
                if (tval_eq(&a->u.dict->keys[i], &b->u.dict->keys[j])) {
                    if (!tval_eq(&a->u.dict->vals[i], &b->u.dict->vals[j])) return 0;
                    found = 1; break;
                }
            }
            if (!found) return 0;
        }
        return 1;
    }
    case T_FN: return a->u.fn == b->u.fn;
    case T_NATIVE: return a->u.nat == b->u.nat;
    case T_FILE: return a->u.file == b->u.file;
    default: return 0;
    }
}

/* số thực -> chuỗi, khớp fmt_float của t.py: số nguyên in không phần lẻ,
   còn lại dạng ngắn nhất round-trip (%.15g -> %.16g -> %.17g). */
/* int64 -> thập phân (như "%lld", không qua printf); trả số byte, tmp >= 21 byte */
static size_t fmt_i64(int64_t x, char *tmp) {
    char r[24];
    size_t k = 0, n = 0;
    uint64_t u = x < 0 ? (uint64_t)0 - (uint64_t)x : (uint64_t)x;
    do { r[k++] = (char)('0' + (int)(u % 10)); u /= 10; } while (u);
    if (x < 0) tmp[n++] = '-';
    while (k) tmp[n++] = r[--k];
    tmp[n] = 0;
    return n;
}
static void fmt_double(double x, char *tmp, size_t n) {
    if (x != x) { snprintf(tmp, n, "nan"); return; }
    if (x > 1.7976931348623157e308) { snprintf(tmp, n, "inf"); return; }
    if (x < -1.7976931348623157e308) { snprintf(tmp, n, "-inf"); return; }
    if (x == floor(x) && fabs(x) < 9.2e18 && n >= 21) {
        fmt_i64((int64_t)x, tmp);
        return;
    }
    for (int p = 15; p <= 17; p++) {
        snprintf(tmp, n, "%.*g", p, x);
        if (strtod(tmp, NULL) == x) return;
    }
}
/* chuỗi cho số nguyên nhỏ (str(i) trong vòng lặp thường lặp lại): cache giữ
   1 ref vĩnh viễn như str1_cache — bỏ cấp phát cho 0..2047 và -1..-256;
   ngoài vùng vẫn mk_str_n bình thường. Chuỗi bất biến nên dùng chung an toàn. */
#define SIC_LO (-256)
#define SIC_HI 2048
static TVal str_int_cache[SIC_HI - SIC_LO];
static TVal mk_str_int(int64_t i) {
    if (i >= SIC_LO && i < SIC_HI) {
        TVal *c = &str_int_cache[i - SIC_LO];
        if (c->t != T_STR) {
            char tmp[24];
            *c = mk_str_n(tmp, fmt_i64(i, tmp));   /* cache giữ ref trả về */
        }
        (*c->u.str.refc)++;
        return *c;
    }
    {
        char tmp[24];
        return mk_str_n(tmp, fmt_i64(i, tmp));
    }
}

/* fmt động (cho str()/raise/print/dict) */
typedef struct { char *s; size_t n, cap; } Buf;
static void buf_put(Buf *b, const char *s, size_t n) {
    while (b->n + n + 1 > b->cap) {
        b->cap = b->cap ? b->cap * 2 : 64;
        b->s = (char*)realloc(b->s, b->cap);
    }
    memcpy(b->s + b->n, s, n); b->n += n; b->s[b->n] = 0;
}
static void buf_c(Buf *b, char c) { buf_put(b, &c, 1); }
static void ten_fmt(Buf *b, const TTensor *t);
static int dict_find(TDict *d, const TVal *k);
static void fmt_into(Buf *b, const TVal *v) {
    char tmp[64];
    size_t i;
    switch (v->t) {
    case T_NIL: buf_put(b, "nil", 3); break;
    case T_BOOL: buf_put(b, v->u.b ? "true" : "false", v->u.b ? 4 : 5); break;
    case T_INT:
        buf_put(b, tmp, fmt_i64(v->u.i, tmp)); break;
    case T_FLOAT:
        fmt_double(v->u.f, tmp, sizeof(tmp));
        buf_put(b, tmp, strlen(tmp)); break;
    case T_STR: buf_put(b, v->u.str.s, v->u.str.n); break;
    case T_LIST:
        buf_c(b, '[');
        for (i = 0; i < v->u.list->len; i++) {
            if (i) buf_put(b, ", ", 2);
            fmt_into(b, &v->u.list->items[i]);
        }
        buf_c(b, ']'); break;
    case T_TENSOR:
        ten_fmt(b, v->u.ten); break;
    case T_DICT: {
        TDict *d = v->u.dict;
        if (d->kind == DK_SET && d->len == 0) { buf_put(b, "set()", 5); break; }
        if (d->kind == DK_TYPE) {
            buf_put(b, "<struct ", 8); buf_put(b, rec_name(d), strlen(rec_name(d))); buf_c(b, '>');
            break;
        }
        if (d->kind == DK_REC) {
            if (cur_vm && d->meths) {
                TVal sk = mk_str("__str__");
                int mi = dict_find(d->meths, &sk);
                tval_free(&sk);
                if (mi >= 0) {
                    TVal fn = d->meths->vals[mi], self = *v, out = mk_nil();
                    tval_retain(&self);
                    if (tvm_call_value(cur_vm, fn, &self, 1, &out)) {
                        if (out.t == T_STR) buf_put(b, out.u.str.s, out.u.str.n);
                        else fmt_into(b, &out);
                        tval_free(&out);
                        break;
                    }
                    tval_free(&self);
                }
            }
            buf_put(b, rec_name(d), strlen(rec_name(d)));
        }
        mod_pull_all(d);
        buf_c(b, '{');
        for (i = 0; i < d->len; i++) {
            if (i) buf_put(b, ", ", 2);
            fmt_into(b, &d->keys[i]);
            if (d->kind == DK_SET) continue;
            buf_put(b, ": ", 2);
            fmt_into(b, &d->vals[i]);
        }
        buf_c(b, '}'); break;
    }
    case T_FN: buf_put(b, "<fn>", 4); break;
    case T_NATIVE: buf_put(b, "<fn>", 4); break;
    case T_CELL: fmt_into(b, &v->u.cell->val); break;
    case T_UNDEF: buf_put(b, "<undef>", 7); break;
    case T_FILE:
        snprintf(tmp, sizeof(tmp), "file(%s)", v->u.file->path ? v->u.file->path : "?");
        buf_put(b, tmp, strlen(tmp)); break;
    default: buf_put(b, "<obj>", 5);
    }
}
void tval_fmt(const TVal *v, char *buf, size_t n) {
    Buf b = {0, 0, 0};
    fmt_into(&b, v);
    if (b.s) { strncpy(buf, b.s, n - 1); buf[n - 1] = 0; free(b.s); }
    else if (n) buf[0] = 0;
}
static TVal fmt_str(const TVal *v) {
    if (v->t == T_STR) { TVal r = *v; (*r.u.str.refc)++; return r; }   /* chuỗi bất biến: dùng chung */
    if (v->t == T_INT) return mk_str_int(v->u.i);
    if (v->t == T_FLOAT) {
        char tmp[64];
        fmt_double(v->u.f, tmp, sizeof(tmp));
        return mk_str(tmp);
    }
    Buf b = {0, 0, 0};
    fmt_into(&b, v);
    TVal r = mk_str_n(b.s ? b.s : "", b.n);
    free(b.s);
    return r;
}
void tval_print(const TVal *v) {
    Buf b = {0, 0, 0};
    fmt_into(&b, v);
    if (b.s) { fwrite(b.s, 1, b.n, stdout); free(b.s); }
}

static double to_num(const TVal *v, int *ok) {
    if (v->t == T_INT) return (double)v->u.i;
    if (v->t == T_FLOAT) return v->u.f;
    *ok = 0; return 0;
}

/* ---------- dict ---------- */
/* băm khoá: int và float nguyên cùng giá trị phải cùng hash (1 == 1.0) */
static uint64_t mix64(uint64_t x) {
    x ^= x >> 33; x *= 0xff51afd7ed558ccdULL;
    x ^= x >> 33; x *= 0xc4ceb9fe1a85ec53ULL;
    x ^= x >> 33;
    return x;
}
static uint64_t tval_hash(const TVal *k) {
    switch (k->t) {
    case T_INT: return mix64((uint64_t)k->u.i);
    case T_FLOAT: {
        double f = k->u.f;
        if (f == floor(f) && fabs(f) < 9.2e18) return mix64((uint64_t)(int64_t)f);
        uint64_t b; memcpy(&b, &f, 8);
        return mix64(b ^ 0x5bd1e995ULL);
    }
    case T_STR: {
        if (k->u.str.refc && k->u.str.refc[1]) return (uint32_t)k->u.str.refc[1];
        uint64_t h = 1469598103934665603ULL;
        for (size_t i = 0; i < k->u.str.n; i++) {
            h ^= (unsigned char)k->u.str.s[i];
            h *= 1099511628211ULL;
        }
        uint32_t h32 = (uint32_t)(h ^ (h >> 32)) | 1u;   /* luôn khác 0 */
        if (k->u.str.refc) k->u.str.refc[1] = (int32_t)h32;
        return h32;
    }
    case T_BOOL: return 0x9e3779b97f4a7c15ULL + (uint64_t)k->u.b;
    case T_NIL: return 0x2545F4914F6CDD1DULL;
    case T_CELL: return tval_hash(&k->u.cell->val);
    default: return 7;
    }
}
#define DICT_SMALL 8
static void dict_index_put(TDict *d, size_t i) {
    size_t m = d->icap - 1, h = (size_t)tval_hash(&d->keys[i]) & m;
    while (d->idx[h] != -1) h = (h + 1) & m;
    d->idx[h] = (int32_t)i;
}
static void dict_index_build(TDict *d) {
    size_t cap = 16;
    while (cap < d->len * 2 + 2) cap *= 2;
    free(d->idx);
    d->idx = (int32_t*)malloc(cap * sizeof(int32_t));
    d->icap = cap;
    for (size_t i = 0; i < cap; i++) d->idx[i] = -1;
    for (size_t i = 0; i < d->len; i++) dict_index_put(d, i);
}
/* so sánh khoá dict inline (né C-call tval_eq cho int/str — 2 khoá phổ biến
   nhất; chỉ hợp khi k đã norm_ty, không CELL/UNDEF). Trả 1 = bằng, 0 = khác,
   -1 = không chắc (phải gọi tval_eq đầy đủ). */
static int key_eq_fast(const TVal *a, const TVal *b) {
    if (a->t == T_INT && b->t == T_INT) return a->u.i == b->u.i;
    if (a->t == T_STR && b->t == T_STR)
        return a->u.str.n == b->u.str.n &&
               memcmp(a->u.str.s, b->u.str.s, a->u.str.n) == 0;
    return -1;
}
static int key_eq(const TVal *a, const TVal *b) {
    int r = key_eq_fast(a, b);
    return r >= 0 ? r : tval_eq(a, b);
}
static int dict_find(TDict *d, const TVal *k) {
    size_t i;
    if (!d->idx) {
        if (d->len < DICT_SMALL) {
            for (i = 0; i < d->len; i++)
                if (key_eq(&d->keys[i], k)) return (int)i;
            return -1;
        }
        dict_index_build(d);
    }
    {
        size_t m = d->icap - 1, h = (size_t)tval_hash(k) & m;
        while (d->idx[h] != -1) {
            if (key_eq(&d->keys[d->idx[h]], k)) return d->idx[h];
            h = (h + 1) & m;
        }
    }
    return -1;
}
static TDict *dict_new(void) {
    TDict *d = (TDict*)pk_alloc(sizeof(TDict));
    d->refc = 1; d->gc_mark = 0; d->len = 0; d->cap = 4;
    d->kind = DK_DICT; d->rname = NULL; d->meths = NULL; d->fields = NULL;
    d->idx = NULL; d->icap = 0; d->mod = NULL;
    d->keys = (TVal*)malloc(d->cap * sizeof(TVal));
    d->vals = (TVal*)malloc(d->cap * sizeof(TVal));
    if (cur_vm) gc_track(cur_vm, T_DICT, d);
    return d;
}
static void dict_set(TDict *d, TVal k, TVal v) {
    int i = dict_find(d, &k);
    if (i >= 0) {
        tval_free(&k);
        tval_free(&d->vals[i]);
        d->vals[i] = v;
        return;
    }
    if (d->len >= d->cap) {
        d->cap *= 2;
        d->keys = (TVal*)realloc(d->keys, d->cap * sizeof(TVal));
        d->vals = (TVal*)realloc(d->vals, d->cap * sizeof(TVal));
    }
    d->keys[d->len] = k;
    d->vals[d->len] = v;
    d->len++;
    if (d->idx) {
        if (d->len * 2 + 2 > d->icap) dict_index_build(d);
        else dict_index_put(d, d->len - 1);
    }
}

/* xoá entry i (giữ thứ tự còn lại, khớp dict Python) */
static void dict_del_at(TDict *d, size_t i) {
    tval_free(&d->keys[i]);
    tval_free(&d->vals[i]);
    memmove(&d->keys[i], &d->keys[i + 1], (d->len - i - 1) * sizeof(TVal));
    memmove(&d->vals[i], &d->vals[i + 1], (d->len - i - 1) * sizeof(TVal));
    d->len--;
    if (d->idx) { free(d->idx); d->idx = NULL; d->icap = 0; }   /* dựng lại khi cần */
}
/* thông điệp như t.py: "unhashable dict key <fmt(k)[:30]>" */
static void unhashable_msg(const TVal *k, char *eb, size_t n) {
    Buf b = {0, 0, 0};
    fmt_into(&b, k);
    snprintf(eb, n, "unhashable dict key %.30s", b.s ? b.s : "");
    free(b.s);
}
static int hashable(const TVal *k) {
    return k->t == T_INT || k->t == T_FLOAT || k->t == T_STR ||
           k->t == T_BOOL || k->t == T_NIL;
}
/* tên kiểu như type() (record -> tên struct) */
static const char *tname_of(const TVal *v) {
    switch (v->t) {
    case T_NIL: return "nil";
    case T_BOOL: return "bool";
    case T_INT: return "int";
    case T_FLOAT: return "float";
    case T_STR: return "str";
    case T_LIST: return "list";
    case T_DICT:
        if (v->u.dict->kind == DK_SET) return "set";
        if (v->u.dict->kind == DK_REC) return rec_name(v->u.dict);
        if (v->u.dict->kind == DK_TYPE) return "type";
        return "dict";
    case T_TENSOR: return "tensor";
    case T_FN: case T_NATIVE: return "fn";
    case T_FILE: return "file";
    case T_CELL: return tname_of(&v->u.cell->val);
    default: return "obj";
    }
}

/* ---------- natives ---------- */
static TVal mk_native(TNativeFn fn, const char *name) {
    TNative *n = (TNative*)pk_alloc(sizeof(TNative));
    n->refc = 1; n->gc_mark = 0; n->fn = fn; n->has_obj = 0; n->name = name;
    n->obj.t = T_NIL;
    TVal v; v.t = T_NATIVE; v.u.nat = n;
    if (cur_vm) gc_track(cur_vm, T_NATIVE, n);
    return v;
}
static TVal mk_bound(TNativeFn fn, const char *name, TVal obj) {
    TNative *n = (TNative*)pk_alloc(sizeof(TNative));
    n->refc = 1; n->gc_mark = 0; n->fn = fn; n->has_obj = 1; n->name = name;
    n->obj = obj; tval_retain(&n->obj);
    TVal v; v.t = T_NATIVE; v.u.nat = n;
    if (cur_vm) gc_track(cur_vm, T_NATIVE, n);
    return v;
}
#define NARG(n) do { if (argc != (n)) die(vm, "arity"); } while (0)
#define NMIN(a) do { if (argc < (a)) die(vm, "arity"); } while (0)
#define NMAX(a) do { if (argc > (a)) die(vm, "arity"); } while (0)

static void n_range(TVM *vm, TVal *a, int argc, TVal *r) {
    int64_t lo = 0, hi = 0;
    if (argc == 1) {
        if (a[0].t != T_INT) die(vm, "range() int");
        hi = a[0].u.i;
    } else if (argc == 2) {
        if (a[0].t != T_INT || a[1].t != T_INT) die(vm, "range() int");
        lo = a[0].u.i; hi = a[1].u.i;
    } else die(vm, "arity");
    TList *l = list_new(hi > lo ? (size_t)(hi - lo) : 0);
    for (int64_t i = lo; i < hi; i++) list_push(l, mk_int(i));
    r->t = T_LIST; r->u.list = l;
}
static int rec_call_dunder(TVM *vm, const TVal *self, const char *mname, const TVal *arg1, const TVal *arg2, int nargs, TVal *out) {
    if (self->t != T_DICT || self->u.dict->kind != DK_REC || !self->u.dict->meths) return 0;
    TVal key = mk_str(mname);
    int mi = dict_find(self->u.dict->meths, &key);
    tval_free(&key);
    if (mi < 0) return 0;
    TVal fn = self->u.dict->meths->vals[mi];
    TVal args[3];
    args[0] = *self; tval_retain(&args[0]);
    if (nargs >= 1 && arg1) { args[1] = *arg1; tval_retain(&args[1]); }
    if (nargs >= 2 && arg2) { args[2] = *arg2; tval_retain(&args[2]); }
    return tvm_call_value(vm, fn, args, nargs + 1, out);
}

static void n_len(TVM *vm, TVal *a, int argc, TVal *r) {
    NARG(1);
    size_t n;
    if (a[0].t == T_STR) n = a[0].u.str.n;
    else if (a[0].t == T_LIST) n = a[0].u.list->len;
    else if (a[0].t == T_TENSOR) n = a[0].u.ten->ndim ? a[0].u.ten->shape[0] : a[0].u.ten->len;
    else if (a[0].t == T_DICT && a[0].u.dict->kind <= DK_SET) n = a[0].u.dict->len;
    else if (a[0].t == T_DICT && a[0].u.dict->kind == DK_REC && rec_call_dunder(vm, &a[0], "__len__", NULL, NULL, 0, r)) return;
    else die(vm, "len() here");
    *r = mk_int((int64_t)n);
}
static void n_str(TVM *vm, TVal *a, int argc, TVal *r) {
    NARG(1); (void)vm; *r = fmt_str(&a[0]);
}
static void n_int(TVM *vm, TVal *a, int argc, TVal *r) {
    NARG(1);
    if (a[0].t == T_INT) *r = mk_int(a[0].u.i);
    else if (a[0].t == T_FLOAT) *r = mk_int((int64_t)a[0].u.f);
    else if (a[0].t == T_BOOL) *r = mk_int(a[0].u.b);
    else if (a[0].t == T_STR) {
        char *e = NULL;
        double x = strtod(a[0].u.str.s, &e);
        if (e == a[0].u.str.s) die(vm, "invalid int");
        *r = mk_int((int64_t)x);
    }
    else die(vm, "int() here");
}
static void n_float(TVM *vm, TVal *a, int argc, TVal *r) {
    NARG(1);
    if (a[0].t == T_FLOAT) *r = mk_float(a[0].u.f);
    else if (a[0].t == T_INT) *r = mk_float((double)a[0].u.i);
    else if (a[0].t == T_BOOL) *r = mk_float(a[0].u.b);
    else if (a[0].t == T_STR) {
        char *e = NULL;
        double x = strtod(a[0].u.str.s, &e);
        if (e == a[0].u.str.s) die(vm, "invalid float");
        *r = mk_float(x);
    }
    else die(vm, "float() here");
}
static void n_type(TVM *vm, TVal *a, int argc, TVal *r) {
    NARG(1); (void)vm;
    *r = mk_str(tname_of(&a[0]));
}
static TTensor *ten_from(TVM *vm, const TVal *v);
static int ten_parse_dtype(TVM *vm, const char *s);
static void ten_pack_dtype(TTensor *t, int dt);
static void n_tensor(TVM *vm, TVal *a, int argc, TVal *r) {
    NMIN(1); NMAX(2);
    if (a[0].t == T_BOOL) die(vm, "tensor() numbers");
    int dt = TD_F64;
    if (argc == 2) {
        if (a[1].t != T_STR) die(vm, "tensor() dtype str");
        dt = ten_parse_dtype(vm, a[1].u.str.s);
    }
    r->t = T_TENSOR; r->u.ten = ten_from(vm, &a[0]);
    if (dt != TD_F64) ten_pack_dtype(r->u.ten, dt);
}
static void n_open(TVM *vm, TVal *a, int argc, TVal *r) {
    const char *mode = "r";
    char mbuf[8];
    /* rb phải sống tới sau fopen: nếu khai trong block, đời sống kết thúc
       trước lời gọi và optimizer (-O2) tái dùng stack -> fopen nhận mode
       thiếu 'b' -> text mode -> CRLF translation làm hỏng file nhị phân. */
    char rb[10];
    size_t k = 0;
    if (argc < 1 || argc > 2) die(vm, "arity");
    if (a[0].t != T_STR) die(vm, "open() path");
    if (argc == 2) {
        if (a[1].t != T_STR) die(vm, "open() mode");
        if (a[1].u.str.n >= sizeof(mbuf)) die(vm, "open() mode");
        memcpy(mbuf, a[1].u.str.s, a[1].u.str.n);
        mbuf[a[1].u.str.n] = 0; mode = mbuf;
    }
    /* T string là BYTE-STRING: mọi mode đều raw (không CRLF translation) */
    for (const char *q = mode; *q && k + 3 < sizeof(rb); q++) {
        if (*q != 'b') rb[k++] = *q;
    }
    rb[k] = 'b';
    rb[k + 1] = 0;
    FILE *h = fopen(a[0].u.str.s, rb);
    if (!h) {
        char eb[256];
        snprintf(eb, sizeof(eb), "open: %.200s", a[0].u.str.s);
        die(vm, eb);
    }
    TFile *f = (TFile*)malloc(sizeof(TFile));
    f->refc = 1; f->h = h; f->closed = 0;
    f->path = (char*)malloc(a[0].u.str.n + 1);
    memcpy(f->path, a[0].u.str.s, a[0].u.str.n + 1);
    r->t = T_FILE; r->u.file = f;
}
static void n_read(TVM *vm, TVal *a, int argc, TVal *r) {
    NARG(1);
    if (a[0].t != T_FILE) die(vm, "read(f)");
    TFile *f = a[0].u.file;
    if (!f->h || f->closed) die(vm, "read closed");
    fseek(f->h, 0, SEEK_END);
    long n = ftell(f->h);
    fseek(f->h, 0, SEEK_SET);
    if (n < 0) n = 0;
    char *buf = (char*)malloc((size_t)n + 1);
    size_t got = fread(buf, 1, (size_t)n, f->h);
    *r = mk_str_n(buf, got);
    free(buf);
}
static void n_write(TVM *vm, TVal *a, int argc, TVal *r) {
    NARG(2);
    if (a[0].t != T_FILE) die(vm, "write(f, s)");
    TFile *f = a[0].u.file;
    if (!f->h || f->closed) die(vm, "write closed");
    TVal s = fmt_str(&a[1]);
    size_t w = fwrite(s.u.str.s, 1, s.u.str.n, f->h);
    fflush(f->h);
    tval_free(&s);
    *r = mk_int((int64_t)w);
}
static void n_close(TVM *vm, TVal *a, int argc, TVal *r) {
    NMIN(1); NMAX(2);
    if (argc == 2) {
        /* close(obj, 1): with-enter hook -> gọi __enter__() nếu có, ngược lại trả chính obj */
        if (a[0].t == T_DICT && a[0].u.dict->kind == DK_REC) {
            if (rec_call_dunder(vm, &a[0], "__enter__", NULL, NULL, 0, r)) return;
        }
        *r = a[0]; tval_retain(r);
        return;
    }
    if (a[0].t == T_NIL) { r->t = T_NIL; return; }
    if (a[0].t == T_DICT && a[0].u.dict->kind == DK_REC) {
        TVal out = mk_nil();
        if (rec_call_dunder(vm, &a[0], "__exit__", NULL, NULL, 0, &out)) { tval_free(&out); r->t = T_NIL; return; }
        if (rec_call_dunder(vm, &a[0], "close", NULL, NULL, 0, &out)) { tval_free(&out); r->t = T_NIL; return; }
        r->t = T_NIL; return;
    }
    if (a[0].t != T_FILE) die(vm, "close(f)");
    TFile *f = a[0].u.file;
    if (f->h && !f->closed) { fclose(f->h); f->closed = 1; f->h = NULL; }
    r->t = T_NIL;
}
static void n_ord(TVM *vm, TVal *a, int argc, TVal *r) {
    NARG(1);
    if (a[0].t != T_STR || a[0].u.str.n != 1)
        die(vm, "ord() 1 char");
    *r = mk_int((unsigned char)a[0].u.str.s[0]);
}
static void n_chr(TVM *vm, TVal *a, int argc, TVal *r) {
    NARG(1);
    int ok = 1; double x = to_num(&a[0], &ok);
    if (!ok) die(vm, "chr() num");
    char c = (char)(int)x;
    *r = mk_str_n(&c, 1);
}
static void n_argv(TVM *vm, TVal *a, int argc, TVal *r) {
    (void)a; NMAX(0);
    TList *l = list_new(vm->root->xargc > 0 ? (size_t)vm->root->xargc : 0);
    for (int i = 0; i < vm->root->xargc; i++)
        list_push(l, mk_str(vm->root->xargv[i]));
    r->t = T_LIST; r->u.list = l;
}
static void n_print(TVM *vm, TVal *a, int argc, TVal *r) {
    (void)vm;
    for (int i = 0; i < argc; i++) {
        if (i) putchar(' ');
        tval_print(&a[i]);
    }
    putchar('\n');
    r->t = T_NIL;
}
static void n_mem(TVM *vm, TVal *a, int argc, TVal *r) {
    (void)a; NMAX(0);
    TDict *d = dict_new();
    TVal k, v;
    k = mk_str("objs");
    v = mk_int((int64_t)vm->root->gc_tracked);
    dict_set(d, k, v);
    k = mk_str("gc");
    v = mk_int((int64_t)vm->root->gc_runs);
    dict_set(d, k, v);
    r->t = T_DICT; r->u.dict = d;
}
static void n_f64bits(TVM *vm, TVal *a, int argc, TVal *r) {
    NARG(1);
    double x;
    if (a[0].t == T_FLOAT) x = a[0].u.f;
    else if (a[0].t == T_INT) x = (double)a[0].u.i;
    else die(vm, "f64bits() number");
    int64_t bits = 0;
    memcpy(&bits, &x, 8);
    *r = mk_int(bits);
}

/* Gọi hàm (T hoặc native) từ bên trong một native — dùng cho map/filter/sort.
   Giữ nguyên trạng thái của frame gọi; nested run dừng khi nframes <= stop_nf. */
int tvm_call_value(TVM *vm, TVal fv, TVal *args, int argc, TVal *out) {
    if (fv.t == T_NATIVE) {
        TVal rv; rv.t = T_NIL;
        fv.u.nat->fn(vm, args, argc, &rv);
        *out = rv;
        return 1;
    }
    if (fv.t != T_FN) return 0;
    TFn *f = fv.u.fn;
    if (!f->vararg && !f->native && argc == f->nargs) t_jit_maybe(vm, f, args);
    if (!f->vararg && f->native) {          /* fn đã JIT thành native: gọi thẳng, không lặp lại */
        TVal rv; rv.t = T_NIL;
        if (t_jit_call(vm, f, args, argc, &rv)) { *out = rv; return 1; }
        if (argc == f->nargs) t_jit_maybe(vm, f, args);    /* bộ kiểu mới: bản đặc hoá khác */
    }
    if (argc < f->min_args || (!f->vararg && argc > f->nargs)) die(vm, "call arity");
    size_t sp0 = vm->sp, pc0 = vm->pc, nf0 = vm->nframes, stop0 = vm->stop_nf;
    TProg *prog0 = vm->prog;
    TFrame *fr = push_frame(vm);
    fr->ret_pc = pc0; fr->bp = sp0; fr->nlocals = f->nlocals; fr->fn = f;
    fr->ret_prog = prog0; fr->is_mod = 0;
    vm->prog = f->prog;
    tval_retain(&fv);
    if (f->nlocals == 0) {
        fr->locals = NULL;
    } else if (fr->locals && fr->lcap >= f->nlocals) {
        fill_undef(fr->locals, f->nlocals);
    } else {
        free(fr->locals);
        fr->locals = (TVal*)calloc(f->nlocals, sizeof(TVal));
        if (!fr->locals) exit(1);
        fill_undef(fr->locals, f->nlocals);
        fr->lcap = f->nlocals;
    }
    if (f->vararg) {
        int fixed = (int)f->min_args;
        for (int i = 0; i < fixed; i++) fr->locals[i] = args[i];
        TList *rest = list_new(argc > fixed ? (size_t)(argc - fixed) : 0);
        for (int i = fixed; i < argc; i++) rest->items[rest->len++] = args[i];
        fr->locals[fixed].t = T_LIST;
        fr->locals[fixed].u.list = rest;
    } else {
        for (int i = 0; i < argc; i++) fr->locals[i] = args[i];
        for (int i = argc; i < (int)f->nargs; i++) fr->locals[i] = mk_nil();
    }
    vm->sp = sp0;
    vm->pc = (size_t)f->entry;
    vm->stop_nf = nf0;
#ifdef TDBG_CALL
    fprintf(stderr, "[cv] enter entry=%d nargs=%d argc=%d sp0=%zu nf0=%zu\n",
            f->entry, f->nargs, argc, sp0, nf0);
#endif
    tvm_run(vm);
#ifdef TDBG_CALL
    fprintf(stderr, "[cv] back nframes=%zu sp=%zu pc=%zu\n",
            vm->nframes, vm->sp, vm->pc);
#endif
    vm->stop_nf = stop0;
    TVal rv = mk_nil();
    if (vm->nframes <= nf0 && vm->sp > sp0) {
        rv = vm->stack[sp0];          /* chuyển ownership ra ngoài (không free) */
        vm->stack[sp0].t = T_NIL;
    }
    vm->sp = sp0;
    vm->pc = pc0;
    vm->prog = prog0;
    vm->nframes = nf0;
    *out = rv;
    return 1;
}

/* ---------- builtins tập trên (abs..filter) ---------- */
/* so sánh tổng quát: số < số, chuỗi < chuỗi (byte), trả -1/0/1 */
static int val_cmp(TVal *a, TVal *b) {
    int oka = 1, okb = 1;
    double x = to_num(a, &oka), y = to_num(b, &okb);
    if (oka && okb) return x < y ? -1 : (x > y ? 1 : 0);
    if (a->t == T_STR && b->t == T_STR) {
        size_t n = a->u.str.n < b->u.str.n ? a->u.str.n : b->u.str.n;
        int c = n ? memcmp(a->u.str.s, b->u.str.s, n) : 0;
        if (c) return c < 0 ? -1 : 1;
        if (a->u.str.n == b->u.str.n) return 0;
        return a->u.str.n < b->u.str.n ? -1 : 1;
    }
    return 2;   /* không so sánh được */
}
/* lấy phần tử thứ i của list/tensor (trả 1 nếu hợp lệ) */
static void ten_sync(const TTensor *t);
static int seq_get(TVal *s, size_t i, TVal *out) {
    if (s->t == T_LIST) {
        if (i >= s->u.list->len) return 0;
        *out = s->u.list->items[i];
        return 1;
    }
    if (s->t == T_TENSOR) {
        if (i >= s->u.ten->len) return 0;
        ten_sync(s->u.ten);
        *out = mk_float(s->u.ten->d[i]);
        return 1;
    }
    if (s->t == T_STR) {
        if (i >= s->u.str.n) return 0;
        *out = mk_str_n(s->u.str.s + i, 1);
        return 1;
    }
    return 0;
}
static size_t seq_len(TVal *s) {
    if (s->t == T_LIST) return s->u.list->len;
    if (s->t == T_TENSOR) return s->u.ten->len;
    if (s->t == T_STR) return s->u.str.n;
    return 0;
}
static TTensor *ten_abs_copy(const TTensor *t);
static void n_abs(TVM *vm, TVal *a, int argc, TVal *r) {
    NARG(1);
    if (a[0].t == T_TENSOR) { r->t = T_TENSOR; r->u.ten = ten_abs_copy(a[0].u.ten); return; }
    if (a[0].t == T_INT) { *r = mk_int(a[0].u.i < 0 ? -a[0].u.i : a[0].u.i); return; }
    if (a[0].t == T_FLOAT) { *r = mk_float(a[0].u.f < 0 ? -a[0].u.f : a[0].u.f); return; }
    die(vm, "abs() number");
}
/* round nửa ra xa số 0: round(2.5)=3, round(-2.5)=-3 */
static double round_half_away(double x) {
    return x < 0 ? -(double)(int64_t)(-x + 0.5) : (double)(int64_t)(x + 0.5);
}
static void n_round(TVM *vm, TVal *a, int argc, TVal *r) {
    NARG(1);
    int ok = 1;
    double x = to_num(&a[0], &ok);
    if (!ok) die(vm, "round() number");
    if (a[0].t == T_FLOAT) *r = mk_float(round_half_away(x));
    else *r = mk_int((int64_t)round_half_away(x));
}
static void minmax(TVM *vm, TVal *a, int argc, TVal *r, int want_max) {
    TVal best;
    if (argc == 2) {
        int c = val_cmp(&a[0], &a[1]);
        if (c == 2) die(vm, want_max ? "max()" : "min()");
        best = (want_max ? (c >= 0) : (c <= 0)) ? a[0] : a[1];
        tval_retain(&best);
    } else {
        NARG(1);
        size_t n = seq_len(&a[0]);
        if (!n) die(vm, want_max ? "max() empty" : "min() empty");
        TVal e;
        if (!seq_get(&a[0], 0, &e)) die(vm, want_max ? "max()" : "min()");
        best = e;                        /* mượn từ container, không free */
        for (size_t i = 1; i < n; i++) {
            TVal x;
            if (!seq_get(&a[0], i, &x)) die(vm, want_max ? "max()" : "min()");
            int c = val_cmp(&x, &best);
            if (c == 2) die(vm, want_max ? "max()" : "min()");
            if (want_max ? (c > 0) : (c < 0)) best = x;
        }
    }
    *r = best;
}
static void n_min(TVM *vm, TVal *a, int argc, TVal *r) { minmax(vm, a, argc, r, 0); }
static void n_max(TVM *vm, TVal *a, int argc, TVal *r) { minmax(vm, a, argc, r, 1); }
static void n_sum(TVM *vm, TVal *a, int argc, TVal *r) {
    NMIN(1); NMAX(2);
    size_t n = seq_len(&a[0]);
    int isf = 0;
    double acc = 0;
    if (argc == 2) {
        int ok = 1;
        acc = to_num(&a[1], &ok);
        if (!ok) die(vm, "sum() number");
        isf = (a[1].t == T_FLOAT);
    }
    for (size_t i = 0; i < n; i++) {
        TVal e;
        if (!seq_get(&a[0], i, &e)) die(vm, "sum() list");
        int ok = 1;
        double x = to_num(&e, &ok);
        if (!ok) die(vm, "sum() number");
        if (e.t == T_FLOAT) isf = 1;
        acc += x;
    }
    *r = isf ? mk_float(acc) : mk_int((int64_t)acc);
}
static void anyall(TVM *vm, TVal *a, int argc, TVal *r, int mode) {
    NARG(1);
    size_t n = seq_len(&a[0]);
    if (!n) die(vm, mode ? "all() list" : "any() list");
    int res = mode;                 /* any: 0, all: 1 */
    for (size_t i = 0; i < n; i++) {
        TVal e;
        if (!seq_get(&a[0], i, &e)) die(vm, mode ? "all() list" : "any() list");
        int t = tval_truthy(&e);
        if (mode) { if (!t) { res = 0; break; } }
        else if (t) { res = 1; break; }
    }
    *r = mk_bool(res);
}
static void n_any(TVM *vm, TVal *a, int argc, TVal *r) { anyall(vm, a, argc, r, 0); }
static void n_all(TVM *vm, TVal *a, int argc, TVal *r) { anyall(vm, a, argc, r, 1); }
/* sort tăng dần, tự sắp trong list và trả về chính nó.
   sort(list, f): f là hàm khóa 1 tham số — so sánh theo f(x) (decorate-sort). */
typedef struct { TVM *vm; TVal *keys; size_t *idx; } SortCtx;
static int idx_less(SortCtx *sc, size_t i, size_t j) {
    int c = val_cmp(&sc->keys[i], &sc->keys[j]);
    if (c == 2) die(sc->vm, "sort() comparable");
    if (c != 0) return c < 0;
    return i < j;              /* ổn định */
}
static void idx_sort(SortCtx *sc, size_t *a, size_t lo, size_t hi) {
    while (hi - lo > 16) {
        size_t mid = lo + (hi - lo) / 2;
        if (idx_less(sc, a[mid], a[lo])) { size_t t = a[lo]; a[lo] = a[mid]; a[mid] = t; }
        if (idx_less(sc, a[hi - 1], a[lo])) { size_t t = a[lo]; a[lo] = a[hi - 1]; a[hi - 1] = t; }
        if (idx_less(sc, a[hi - 1], a[mid])) { size_t t = a[mid]; a[mid] = a[hi - 1]; a[hi - 1] = t; }
        size_t piv = a[mid];
        size_t lt = lo, i = lo, gt = hi;
        while (i < gt) {
            if (idx_less(sc, a[i], piv)) { size_t t = a[lt]; a[lt] = a[i]; a[i] = t; lt++; i++; }
            else if (idx_less(sc, piv, a[i])) { gt--; size_t t = a[gt]; a[gt] = a[i]; a[i] = t; }
            else i++;
        }
        if (lt - lo < hi - gt) { idx_sort(sc, a, lo, lt); lo = gt; }
        else { idx_sort(sc, a, gt, hi); hi = lt; }
    }
    for (size_t i = lo + 1; i < hi; i++) {
        size_t key = a[i];
        size_t j = i;
        while (j > lo && idx_less(sc, key, a[j - 1])) { a[j] = a[j - 1]; j--; }
        a[j] = key;
    }
}
static void sort_run(TVM *vm, TList *l, TVal *keys) {
    size_t n = l->len;
    if (n < 2) return;
    TVal *a = (TVal*)malloc(n * sizeof(TVal));
    memcpy(a, l->items, n * sizeof(TVal));
    if (!keys) {
        /* không khóa: sắp bản copy rồi chép lại (an toàn khi có alias) */
        TVal *tmp = (TVal*)malloc(n * sizeof(TVal));
        memcpy(tmp, l->items, n * sizeof(TVal));
        size_t *idx = (size_t*)malloc(n * sizeof(size_t));
        for (size_t i = 0; i < n; i++) idx[i] = i;
        SortCtx sc = {vm, tmp, idx};
        idx_sort(&sc, idx, 0, n);
        for (size_t i = 0; i < n; i++) a[i] = tmp[idx[i]];
        free(tmp);
        free(idx);
    } else {
        size_t *idx = (size_t*)malloc(n * sizeof(size_t));
        for (size_t i = 0; i < n; i++) idx[i] = i;
        SortCtx sc = {vm, keys, idx};
        idx_sort(&sc, idx, 0, n);
        for (size_t i = 0; i < n; i++) a[i] = l->items[idx[i]];
        free(idx);
    }
    memcpy(l->items, a, n * sizeof(TVal));
    free(a);
}
static void n_sort(TVM *vm, TVal *a, int argc, TVal *r) {
    NMIN(1); NMAX(2);
    if (a[0].t != T_LIST) die(vm, "sort() list");
    TList *l = a[0].u.list;
    size_t n = l->len;
    TVal *keys = NULL;
    if (argc == 2) {
        if (a[1].t != T_FN && a[1].t != T_NATIVE) die(vm, "sort() fn");
        if (n) {
            keys = (TVal*)calloc(n, sizeof(TVal));
            if (!keys) exit(1);
            for (size_t i = 0; i < n; i++) {
                TVal e = l->items[i], k;
                tval_retain(&e);
                if (!tvm_call_value(vm, a[1], &e, 1, &k)) die(vm, "sort() fn");
                keys[i] = k;
            }
        }
    } else if (n) {
        for (size_t i = 0; i < n; i++) {
            TVal *v = &l->items[i];
            if (val_cmp(v, v) != 0) die(vm, "sort() comparable");
        }
    }
    sort_run(vm, l, keys);
    if (keys) {
        for (size_t i = 0; i < n; i++) tval_free(&keys[i]);
        free(keys);
    }
    *r = a[0];
    tval_retain(r);
}
static void n_reversed(TVM *vm, TVal *a, int argc, TVal *r) {
    NARG(1);
    size_t n = seq_len(&a[0]);
    if (a[0].t == T_STR) {
        TVal out = mk_str_uninit(n);
        for (size_t i = 0; i < n; i++) out.u.str.s[i] = a[0].u.str.s[n - 1 - i];
        *r = out;
        return;
    }
    if (a[0].t != T_LIST && a[0].t != T_TENSOR) die(vm, "reversed(list|str)");
    TList *l = list_new(n ? n : 1);
    for (size_t i = n; i > 0; i--) {
        TVal e;
        seq_get(&a[0], i - 1, &e);
        tval_retain(&e);          /* giữ tham chiếu cho list mới */
        list_push(l, e);
    }
    r->t = T_LIST; r->u.list = l;
}
static void n_enumerate(TVM *vm, TVal *a, int argc, TVal *r) {
    NMIN(1); NMAX(2);
    int64_t st = 0;
    if (argc == 2) {
        int ok = 1;
        double x = to_num(&a[1], &ok);
        if (!ok) die(vm, "enumerate() number");
        st = (int64_t)x;
    }
    size_t n = seq_len(&a[0]);
    if (!n && a[0].t != T_LIST && a[0].t != T_STR && a[0].t != T_TENSOR) die(vm, "enumerate() list");
    TList *l = list_new(n ? n : 1);
    for (size_t i = 0; i < n; i++) {
        TVal e, pair[2];
        if (!seq_get(&a[0], i, &e)) die(vm, "enumerate() list");
        pair[0] = mk_int(st + (int64_t)i);
        pair[1] = e;
        tval_retain(&pair[1]);
        TList *p = list_new(2);
        list_push(p, pair[0]);
        list_push(p, pair[1]);
        TVal pv; pv.t = T_LIST; pv.u.list = p;
        list_push(l, pv);
    }
    r->t = T_LIST; r->u.list = l;
}
static void n_zip(TVM *vm, TVal *a, int argc, TVal *r) {
    NMIN(2);
    size_t n = (size_t)-1;
    for (int i = 0; i < argc; i++) {
        if (a[i].t != T_LIST && a[i].t != T_STR && a[i].t != T_TENSOR) die(vm, "zip() list");
        size_t m = seq_len(&a[i]);
        if (m < n) n = m;
    }
    TList *l = list_new(n ? n : 1);
    for (size_t i = 0; i < n; i++) {
        TList *p = list_new((size_t)argc);
        for (int j = 0; j < argc; j++) {
            TVal e;
            seq_get(&a[j], i, &e);
            tval_retain(&e);
            list_push(p, e);
        }
        TVal pv; pv.t = T_LIST; pv.u.list = p;
        list_push(l, pv);
    }
    r->t = T_LIST; r->u.list = l;
}
static void n_map(TVM *vm, TVal *a, int argc, TVal *r) {
    NARG(2);
    if (a[0].t != T_FN && a[0].t != T_NATIVE) die(vm, "map() fn");
    size_t n = seq_len(&a[1]);
    if (!n && a[1].t != T_LIST && a[1].t != T_STR) die(vm, "map() list");
    TList *l = list_new(n ? n : 1);
    for (size_t i = 0; i < n; i++) {
        TVal e, out;
        seq_get(&a[1], i, &e);
        tval_retain(&e);              /* callee sở hữu args */
        if (!tvm_call_value(vm, a[0], &e, 1, &out)) { tval_free(&e); die(vm, "map() fn"); }
        list_push(l, out);
    }
    r->t = T_LIST; r->u.list = l;
}
static void n_filter(TVM *vm, TVal *a, int argc, TVal *r) {
    NARG(2);
    if (a[0].t != T_FN && a[0].t != T_NATIVE) die(vm, "filter() fn");
    size_t n = seq_len(&a[1]);
    if (!n && a[1].t != T_LIST && a[1].t != T_STR) die(vm, "filter() list");
    TList *l = list_new(n ? n : 1);
    for (size_t i = 0; i < n; i++) {
        TVal e, out;
        seq_get(&a[1], i, &e);
        tval_retain(&e);
        if (!tvm_call_value(vm, a[0], &e, 1, &out)) { tval_free(&e); die(vm, "filter() fn"); }
        int t = tval_truthy(&out);
        tval_free(&out);
        if (t) { tval_retain(&e); list_push(l, e); }   /* giữ lại cho kết quả */
    }
    r->t = T_LIST; r->u.list = l;
}

/* methods (obj bound là args[-1] qua has_obj -> prepend) */
static void m_str_upper(TVM *vm, TVal *a, int argc, TVal *r) {
    (void)argc;
    if (a[0].t != T_STR) die(vm, "upper()");
    TVal v = mk_str_uninit(a[0].u.str.n);
    if (v.u.str.n) memcpy(v.u.str.s, a[0].u.str.s, v.u.str.n);
    for (size_t i = 0; i < v.u.str.n; i++) {
        char c = v.u.str.s[i];
        if (c >= 'a' && c <= 'z') v.u.str.s[i] = (char)(c - 32);
    }
    *r = v;
}
static void m_str_lower(TVM *vm, TVal *a, int argc, TVal *r) {
    (void)argc;
    if (a[0].t != T_STR) die(vm, "lower()");
    TVal v = mk_str_uninit(a[0].u.str.n);
    if (v.u.str.n) memcpy(v.u.str.s, a[0].u.str.s, v.u.str.n);
    for (size_t i = 0; i < v.u.str.n; i++) {
        char c = v.u.str.s[i];
        if (c >= 'A' && c <= 'Z') v.u.str.s[i] = (char)(c + 32);
    }
    *r = v;
}
static void m_str_strip(TVM *vm, TVal *a, int argc, TVal *r) {
    (void)argc;
    if (a[0].t != T_STR) die(vm, "strip()");
    size_t i = 0, j = a[0].u.str.n;
    while (i < j && (a[0].u.str.s[i] == ' ' || a[0].u.str.s[i] == '\t' ||
                     a[0].u.str.s[i] == '\n' || a[0].u.str.s[i] == '\r')) i++;
    while (j > i && (a[0].u.str.s[j-1] == ' ' || a[0].u.str.s[j-1] == '\t' ||
                     a[0].u.str.s[j-1] == '\n' || a[0].u.str.s[j-1] == '\r')) j--;
    *r = mk_str_n(a[0].u.str.s + i, j - i);
}
static void m_str_split(TVM *vm, TVal *a, int argc, TVal *r) {
    const char *sep = " ";
    size_t seplen = 1;
    if (argc > 2) die(vm, "arity");
    if (a[0].t != T_STR) die(vm, "split()");
    if (argc == 2) {
        if (a[1].t != T_STR) die(vm, "split(sep)");
        sep = a[1].u.str.s; seplen = a[1].u.str.n;
    }
    TList *l = list_new(4);
    const char *s = a[0].u.str.s;
    size_t n = a[0].u.str.n, i = 0;
    if (seplen == 0) {
        for (i = 0; i < n; i++) list_push(l, mk_str_n(s + i, 1));
    } else {
        size_t start = 0;
        for (i = 0; i + seplen <= n; ) {
            if (memcmp(s + i, sep, seplen) == 0) {
                list_push(l, mk_str_n(s + start, i - start));
                i += seplen; start = i;
            } else i++;
        }
        list_push(l, mk_str_n(s + start, n - start));
    }
    r->t = T_LIST; r->u.list = l;
}
static void m_str_join(TVM *vm, TVal *a, int argc, TVal *r) {
    NARG(2); /* obj=sep, parts */
    if (a[0].t != T_STR) die(vm, "join()");
    Buf b = {0, 0, 0};
    if (a[1].t == T_LIST) {
        for (size_t i = 0; i < a[1].u.list->len; i++) {
            if (i) buf_put(&b, a[0].u.str.s, a[0].u.str.n);
            TVal s = fmt_str(&a[1].u.list->items[i]);
            buf_put(&b, s.u.str.s, s.u.str.n);
            tval_free(&s);
        }
    } else if (a[1].t == T_TENSOR) {
        char tmp[32];
        ten_sync(a[1].u.ten);
        for (size_t i = 0; i < a[1].u.ten->len; i++) {
            if (i) buf_put(&b, a[0].u.str.s, a[0].u.str.n);
            double x = a[1].u.ten->d[i];
            fmt_double(x, tmp, sizeof(tmp));
            buf_put(&b, tmp, strlen(tmp));
        }
    } else die(vm, "join(list)");
    *r = mk_str_n(b.s ? b.s : "", b.n);
    free(b.s);
}
static void m_list_append(TVM *vm, TVal *a, int argc, TVal *r) {
    NARG(2);
    if (a[0].t != T_LIST) die(vm, "append()");
    TVal v = a[1]; tval_retain(&v);
    list_push(a[0].u.list, v);
    r->t = T_NIL;
}
static void m_list_next(TVM *vm, TVal *a, int argc, TVal *r) {
    NARG(1);
    if (a[0].t != T_LIST) die(vm, "next()");
    TList *l = a[0].u.list;
    if (l->len == 0) { r->t = T_NIL; return; }
    *r = l->items[0];
    for (size_t i = 0; i + 1 < l->len; i++) l->items[i] = l->items[i + 1];
    l->len--;
}
static void m_list_pop(TVM *vm, TVal *a, int argc, TVal *r) {
    int64_t ix = -1;
    if (argc < 1 || argc > 2) die(vm, "arity");
    if (a[0].t != T_LIST) die(vm, "pop()");
    TList *l = a[0].u.list;
    if (argc == 2) {
        if (a[1].t != T_INT) die(vm, "pop(i)");
        ix = a[1].u.i;
    }
    if (l->len == 0) die(vm, "pop empty");
    if (ix < 0) ix += (int64_t)l->len;
    if (ix < 0 || ix >= (int64_t)l->len) die(vm, "pop range");
    *r = l->items[ix];
    for (size_t i = (size_t)ix; i + 1 < l->len; i++) l->items[i] = l->items[i + 1];
    l->len--;
}
static void m_dict_keys(TVM *vm, TVal *a, int argc, TVal *r) {
    NARG(1);
    if (a[0].t != T_DICT) die(vm, "keys()");
    TList *l = list_new(a[0].u.dict->len);
    for (size_t i = 0; i < a[0].u.dict->len; i++) {
        TVal v = a[0].u.dict->keys[i]; tval_retain(&v);
        list_push(l, v);
    }
    r->t = T_LIST; r->u.list = l;
}
static void m_dict_values(TVM *vm, TVal *a, int argc, TVal *r) {
    NARG(1);
    if (a[0].t != T_DICT) die(vm, "values()");
    TList *l = list_new(a[0].u.dict->len);
    for (size_t i = 0; i < a[0].u.dict->len; i++) {
        TVal v = a[0].u.dict->vals[i]; tval_retain(&v);
        list_push(l, v);
    }
    r->t = T_LIST; r->u.list = l;
}
static void m_dict_get(TVM *vm, TVal *a, int argc, TVal *r) {
    if (argc < 2 || argc > 3) die(vm, "arity");
    if (a[0].t != T_DICT) die(vm, "get()");
    int i = dict_find(a[0].u.dict, &a[1]);
    if (i >= 0) {
        *r = a[0].u.dict->vals[i]; tval_retain(r);
    } else if (argc == 3) {
        *r = a[2]; tval_retain(r);
    } else r->t = T_NIL;
}
/* ---------- tensor n-chiều (vm/tensor_impl.h) ---------- */
#include "tensor_impl.h"
static TTensor *ten_abs_copy(const TTensor *t) { return ten_unary(t, U_ABS); }

/* ---------- set / record / method tập trên (v2.2) ---------- */
static TDict *set_from(TVM *vm, TVal *src, const char *who) {
    TDict *d = dict_new();
    d->kind = DK_SET;
    if (!src) return d;
    if (src->t == T_DICT) {
        for (size_t i = 0; i < src->u.dict->len; i++) {
            TVal k = src->u.dict->keys[i];
            if (dict_find(d, &k) < 0) { tval_retain(&k); dict_set(d, k, mk_bool(1)); }
        }
        return d;
    }
    if (src->t != T_LIST && src->t != T_STR && src->t != T_TENSOR) {
        TVal dv; dv.t = T_DICT; dv.u.dict = d; tval_free(&dv);
        char eb[64]; snprintf(eb, sizeof(eb), "%s() set", who); die(vm, eb);
    }
    size_t n = seq_len(src);
    for (size_t i = 0; i < n; i++) {
        TVal e;
        seq_get(src, i, &e);
        if (src->t == T_LIST) tval_retain(&e);
        if (!hashable(&e)) {
            char eb[96];
            unhashable_msg(&e, eb, sizeof(eb));
            tval_free(&e);
            TVal dv; dv.t = T_DICT; dv.u.dict = d; tval_free(&dv);
            die(vm, eb);
        }
        if (dict_find(d, &e) < 0) dict_set(d, e, mk_bool(1));
        else tval_free(&e);
    }
    return d;
}
static void n_set(TVM *vm, TVal *a, int argc, TVal *r) {
    NMAX(1);
    TDict *d = set_from(vm, argc ? &a[0] : NULL, "set");
    r->t = T_DICT; r->u.dict = d;
}
static void n_input(TVM *vm, TVal *a, int argc, TVal *r) {
    NMAX(1);
    if (argc == 1) { tval_print(&a[0]); fflush(stdout); }
    Buf b = {0, 0, 0};
    int c, got = 0;
    while ((c = getchar()) != EOF) {
        got = 1;
        if (c == '\n') break;
        buf_c(&b, (char)c);
    }
    if (!got) { r->t = T_NIL; free(b.s); return; }
    if (b.n && b.s[b.n - 1] == '\r') b.n--;
    *r = mk_str_n(b.s ? b.s : "", b.n);
    free(b.s);
}
/* __struct(name, fields, methods) -> kiểu (tạo 1 lần lúc định nghĩa struct) */
static void n___struct(TVM *vm, TVal *a, int argc, TVal *r) {
    NARG(3);
    if (a[1].t != T_LIST || a[2].t != T_DICT) die(vm, "__struct(name, fields, methods)");
    for (size_t i = 0; i < a[1].u.list->len; i++)
        if (a[1].u.list->items[i].t != T_STR) die(vm, "__struct: field là str");
    TDict *d = dict_new();
    d->kind = DK_TYPE;
    {
        TVal nm = fmt_str(&a[0]);
        d->rname = (char*)malloc(nm.u.str.n + 1);
        memcpy(d->rname, nm.u.str.s, nm.u.str.n + 1);
        tval_free(&nm);
    }
    for (size_t i = 0; i < a[2].u.dict->len; i++) {
        TVal k = a[2].u.dict->keys[i], v = a[2].u.dict->vals[i];
        tval_retain(&k); tval_retain(&v);
        dict_set(d, k, v);
    }
    d->fields = a[1].u.list;
    d->fields->refc++;
    r->t = T_DICT; r->u.dict = d;
}
/* __new(type, v1..vn) -> record: keys MƯỢN mảng tên field của kiểu */
static void n___new(TVM *vm, TVal *a, int argc, TVal *r) {
    if (argc < 1 || a[0].t != T_DICT || a[0].u.dict->kind != DK_TYPE) die(vm, "__new(type, values...)");
    TDict *tp = a[0].u.dict;
    size_t n = tp->fields->len;
    if ((size_t)argc != n + 1) die(vm, "__new arity");
    TDict *d = (TDict*)pk_alloc(sizeof(TDict));
    d->refc = 1; d->gc_mark = 0; d->gc_node = NULL;
    d->len = n; d->cap = n;
    d->keys = tp->fields->items;
    d->vals = (TVal*)malloc((n ? n : 1) * sizeof(TVal));
    for (size_t i = 0; i < n; i++) { d->vals[i] = a[i + 1]; tval_retain(&d->vals[i]); }
    d->kind = DK_REC; d->rname = NULL; d->fields = NULL;
    d->idx = NULL; d->icap = 0; d->mod = NULL;
    d->meths = tp; tp->refc++;
    if (cur_vm) gc_track(cur_vm, T_DICT, d);
    r->t = T_DICT; r->u.dict = d;
}
/* JIT tạo record field số (G6): như n___new nhưng giá trị là bit thô trong slots
   của frame native (không qua TVal stack). tp đã guard đúng con trỏ ở code máy;
   field toàn số nên không cần retain. Trả record sở hữu được (refc=1). */
TDict *tjit_rec_new(TDict *tp, int shape, int64_t *slots) {
    const uint8_t *fty = tjit_shape_fty(shape);
    size_t n = tp->fields->len;
    TDict *d = (TDict*)pk_alloc(sizeof(TDict));
    d->refc = 1; d->gc_mark = 0; d->gc_node = NULL;
    d->len = n; d->cap = n;
    d->keys = tp->fields->items;
    d->vals = (TVal*)malloc((n ? n : 1) * sizeof(TVal));
    if (!d->vals) exit(1);
    for (size_t i = 0; i < n; i++) {
        if (fty[i] == TJIT_TY_F) { d->vals[i].t = T_FLOAT; memcpy(&d->vals[i].u.f, &slots[i], 8); }
        else if (fty[i] == TJIT_TY_B) { d->vals[i].t = T_BOOL; d->vals[i].u.b = (int)(slots[i] & 1); }
        else { d->vals[i].t = T_INT; d->vals[i].u.i = slots[i]; }
    }
    d->kind = DK_REC; d->rname = NULL; d->fields = NULL;
    d->idx = NULL; d->icap = 0; d->mod = NULL;
    d->meths = tp; tp->refc++;
    if (cur_vm) gc_track(cur_vm, T_DICT, d);
    return d;
}
/* JIT nhả record sở hữu được (POP owned / ghi đè local owned). */
void tjit_dec_dict(TDict *d) { TVal v; v.t = T_DICT; v.u.dict = d; tval_free(&v); }
/* JIT chuỗi sở hữu được + dict (G7). Wrapper TVal* malloc giữ đúng 1 ref chuỗi;
   native dec đúng hạn (entry-compare khi STORE, POP, epilogue RET, deopt move)
   nên wrapper không bao giờ orphan ngoài đường abort (hiếm, tắt sau 64 lần). */
/* wrapper TVal* lấy từ pool 32 byte: đường nóng JIT (str()/nối/so sánh) cấp nhả
   wrapper mỗi vòng lặp — malloc/free của CRT (nhất là Windows) đắt hơn free-list
   nhiều. Mọi wrapper đều đúng sizeof(TVal) nên nhả phải qua tjit_wrap_free. */
static TVal *wrap_new(void) {
    TVal *w = (TVal*)pk_alloc(sizeof(TVal));
    if (!w) exit(1);
    return w;
}
void tjit_wrap_free(TVal *w) { pk_free(w, sizeof(TVal)); }
TVal *tjit_str_of(int64_t bits, int ty) {
    TVal a, *w = wrap_new();
    if (ty == 1) { a.t = T_FLOAT; memcpy(&a.u.f, &bits, 8); }
    else if (ty == 2) { a.t = T_BOOL; a.u.b = (int)(bits & 1); }
    else if (ty >= 3) a = *(const TVal*)(intptr_t)bits;
    else { a.t = T_INT; a.u.i = bits; }
    *w = fmt_str(&a);
    if (ty == 4) { tval_free((TVal*)(intptr_t)bits); tjit_wrap_free((TVal*)(intptr_t)bits); }
    return w;
}
/* nối chuỗi; m: bit0 nhả a, bit1 nhả b (wrapper đã tiêu thụ) */
TVal *tjit_str_cat(const TVal *a, const TVal *b, int m) {
    size_t n = a->u.str.n + b->u.str.n;
    TVal v = mk_str_uninit(n), *w = wrap_new();
    if (a->u.str.n) memcpy(v.u.str.s, a->u.str.s, a->u.str.n);
    if (b->u.str.n) memcpy(v.u.str.s + a->u.str.n, b->u.str.s, b->u.str.n);
    if (a == b) m = 0;   /* cùng wrapper qua 2 alias (vd k + k): 0 ref giữ */
    if (m & 1) { tval_free((TVal*)a); tjit_wrap_free((TVal*)a); }
    if (m & 2) { tval_free((TVal*)b); tjit_wrap_free((TVal*)b); }
    *w = v;
    return w;
}
/* so chuỗi; m như trên (giữ ngữ nghĩa pop/free toán hạng của interpreter) */
int64_t tjit_str_eq(const TVal *a, const TVal *b, int m) {
    int64_t r = (a->u.str.n == b->u.str.n) &&
                (a->u.str.n == 0 || memcmp(a->u.str.s, b->u.str.s, a->u.str.n) == 0);
    if (a == b) m = 0;
    if (m & 1) { tval_free((TVal*)a); tjit_wrap_free((TVal*)a); }
    if (m & 2) { tval_free((TVal*)b); tjit_wrap_free((TVal*)b); }
    return r;
}
void tjit_dec_sval(TVal *w) { tval_free(w); tjit_wrap_free(w); }
TDict *tjit_dict_new0(void) { return dict_new(); }
/* ghi thẳng global trong mainmode (G7): nhả cũ rồi move/retain mới (ty 0=SN,1=DN,2=D) */
void tjit_store_go(TVal *g, int64_t bits, int ty) {
    if (ty == 0) {
        TVal *w = (TVal*)(intptr_t)bits;
        tval_free(g); *g = *w; tjit_wrap_free(w);
    } else {
        TDict *d = (TDict*)(intptr_t)bits;
        tval_free(g);
        if (ty == 2) d->refc++;
        g->t = T_DICT; g->u.dict = d;
    }
}
/* "counter fusion": index thấy được ở lần tìm trước (getD/OP_IDX) — SETIDX sau
   đó dùng lại được vì helper luôn VALIDATE (i < len && keys[i] == k). Keys
   không bao giờ trùng (dict_set chỉ chèn khi find < 0, del_at giữ duy nhất) nên
   keys[i] == k => đúng entry của k, kể cả dict bị xoá/đổi giữa chừng: sai hint
   chỉ là chậm (rớt về dict_find), không bao giờ sai kết quả. */
static THREAD_LOCAL int tjit_last_i = -1;
/* gán dict (nhận ownership như OP_SETIDX): giữ key (retain) + val rồi dict_set.
   key int/str luôn hashable nên không lỗi (giữ ngữ nghĩa die() cho interpreter).
   kv bit5 = dùng index tìm được ở lần find trước (counter fusion): validate
   i < len && keys[i] == k, sai thì fallback dict_find — không bao giờ sai kết quả. */
int tjit_dict_set(TDict *d, int64_t kbits, int64_t vbits, int kv) {
    TVal k, v;
    int hint = (kv & 32) ? tjit_last_i : -1;
    if (kv & 1) { k = *(const TVal*)(intptr_t)kbits; tval_retain(&k); }
    else { k.t = T_INT; k.u.i = kbits; }
    switch ((kv >> 1) & 15) {
    case 1: v.t = T_FLOAT; memcpy(&v.u.f, &vbits, 8); break;
    case 2: v.t = T_BOOL; v.u.b = (int)(vbits & 1); break;
    case 3: case 4:   /* chuỗi mượn/mới: TVal* */
        v = *(const TVal*)(intptr_t)vbits; tval_retain(&v); break;
    case 5: case 6:   /* dict/record mượn/mới: TDict* */
    case 7: case 8:
        v.t = T_DICT; v.u.dict = (TDict*)(intptr_t)vbits; tval_retain(&v); break;
    default: v.t = T_INT; v.u.i = vbits; break;
    }
    if (hint >= 0 && (size_t)hint < d->len && tval_eq(&d->keys[hint], &k)) {
        tval_free(&k);                    /* key đã có trong dict (như dict_set) */
        tval_free(&d->vals[hint]);
        d->vals[hint] = v;
        return 0;
    }
    dict_set(d, k, v);
    return 0;
}
/* get có default (ghi sẵn default vào *out trước khi gọi): 0 = int thấy (đã ghi),
   1 = deopt (khác int), 2 = thiếu key (giữ default, không deopt). Một lần tìm. */
int tjit_dict_getD(TDict *d, int64_t kbits, int kstr, int64_t *out) {
    TVal k; int i;
    if (kstr) k = *(const TVal*)(intptr_t)kbits;
    else { k.t = T_INT; k.u.i = kbits; }
    i = dict_find(d, &k);
    if (i < 0) return 2;
    if (d->vals[i].t != T_INT) return 1;
    *out = d->vals[i].u.i;
    tjit_last_i = i;
    return 0;
}
int64_t tjit_dict_has(TDict *d, int64_t kbits, int kstr) {
    TVal k;
    if (kstr) k = *(const TVal*)(intptr_t)kbits;
    else { k.t = T_INT; k.u.i = kbits; }
    return dict_find(d, &k) >= 0 ? 1 : 0;
}
int64_t tjit_dict_remove(TDict *d, int64_t kbits, int kstr) {
    TVal k; int i;
    if (kstr) k = *(const TVal*)(intptr_t)kbits;
    else { k.t = T_INT; k.u.i = kbits; }
    i = dict_find(d, &k);
    if (i < 0) return 1;    /* thiếu key -> deopt để interpreter báo lỗi đúng dòng */
    dict_del_at(d, (size_t)i);
    return 0;
}
TVal *tjit_str_idx(const TVal *s, int64_t i) {
    if (!s || s->t != T_STR || i < 0 || (uint64_t)i >= s->u.str.n) return NULL;
    {
        unsigned char ch = (unsigned char)s->u.str.s[i];
        TVal *c = &str1_cache[ch];
        if (c->t != T_STR) { *c = mk_str_uninit(1); c->u.str.s[0] = (char)ch; }
        return c;
    }
}
/* gọi method của record: obj = list [fn, self] */
static void m_call_bound(TVM *vm, TVal *a, int argc, TVal *r) {
    TVal fn = a[0].u.list->items[0];
    TVal args[9];
    int n = 0;
    if (argc > 8) die(vm, "too many args");
    args[n] = a[0].u.list->items[1]; tval_retain(&args[n]); n++;
    for (int i = 1; i < argc; i++) { args[n] = a[i]; tval_retain(&args[n]); n++; }
    if (!tvm_call_value(vm, fn, args, n, r)) die(vm, "not callable");
}
static void m_set_add(TVM *vm, TVal *a, int argc, TVal *r) {
    NARG(2);
    if (!hashable(&a[1])) { char eb[96]; unhashable_msg(&a[1], eb, sizeof(eb)); die(vm, eb); }
    if (dict_find(a[0].u.dict, &a[1]) < 0) {
        TVal k = a[1]; tval_retain(&k);
        dict_set(a[0].u.dict, k, mk_bool(1));
    }
    r->t = T_NIL;
}
static void m_has(TVM *vm, TVal *a, int argc, TVal *r) {
    NARG(2); (void)vm;
    *r = mk_bool(dict_find(a[0].u.dict, &a[1]) >= 0);
}
static void m_remove(TVM *vm, TVal *a, int argc, TVal *r) {
    NARG(2);
    int i = dict_find(a[0].u.dict, &a[1]);
    if (i < 0) {
        char eb[128]; Buf b = {0,0,0};
        fmt_into(&b, &a[1]);
        snprintf(eb, sizeof(eb), "key %.60s missing", b.s ? b.s : "?");
        free(b.s); die(vm, eb);
    }
    dict_del_at(a[0].u.dict, (size_t)i);
    r->t = T_NIL;
}
static void m_set_list(TVM *vm, TVal *a, int argc, TVal *r) {
    NARG(1); (void)vm;
    TList *l = list_new(a[0].u.dict->len);
    for (size_t i = 0; i < a[0].u.dict->len; i++) {
        TVal k = a[0].u.dict->keys[i]; tval_retain(&k); list_push(l, k);
    }
    r->t = T_LIST; r->u.list = l;
}
/* mode 0 union, 1 inter, 2 diff */
static void set_op(TVM *vm, TVal *a, int argc, TVal *r, int mode, const char *who) {
    NARG(2);
    TDict *o;
    int own = 0;
    if (a[1].t == T_DICT && a[1].u.dict->kind == DK_SET) o = a[1].u.dict;
    else if (a[1].t == T_LIST) { o = set_from(vm, &a[1], who); own = 1; }
    else { char eb[64]; snprintf(eb, sizeof(eb), "%s() set", who); die(vm, eb); }
    TDict *d = dict_new();
    d->kind = DK_SET;
    TDict *s0 = a[0].u.dict;
    for (size_t i = 0; i < s0->len; i++) {
        int in_o = dict_find(o, &s0->keys[i]) >= 0;
        if (mode == 0 || (mode == 1 && in_o) || (mode == 2 && !in_o)) {
            TVal k = s0->keys[i]; tval_retain(&k); dict_set(d, k, mk_bool(1));
        }
    }
    if (mode == 0) {
        for (size_t i = 0; i < o->len; i++) {
            if (dict_find(d, &o->keys[i]) < 0) {
                TVal k = o->keys[i]; tval_retain(&k); dict_set(d, k, mk_bool(1));
            }
        }
    }
    if (own) { TVal ov; ov.t = T_DICT; ov.u.dict = o; tval_free(&ov); }
    r->t = T_DICT; r->u.dict = d;
}
static void m_set_union(TVM *vm, TVal *a, int argc, TVal *r) { set_op(vm, a, argc, r, 0, "union"); }
static void m_set_inter(TVM *vm, TVal *a, int argc, TVal *r) { set_op(vm, a, argc, r, 1, "inter"); }
static void m_set_diff(TVM *vm, TVal *a, int argc, TVal *r) { set_op(vm, a, argc, r, 2, "diff"); }
static void m_dict_items(TVM *vm, TVal *a, int argc, TVal *r) {
    NARG(1); (void)vm;
    TDict *d = a[0].u.dict;
    TList *l = list_new(d->len);
    for (size_t i = 0; i < d->len; i++) {
        TList *p = list_new(2);
        TVal k = d->keys[i], v = d->vals[i];
        tval_retain(&k); tval_retain(&v);
        list_push(p, k); list_push(p, v);
        TVal pv; pv.t = T_LIST; pv.u.list = p;
        list_push(l, pv);
    }
    r->t = T_LIST; r->u.list = l;
}
static void m_list_clear(TVM *vm, TVal *a, int argc, TVal *r) {
    NARG(1); (void)vm;
    TList *l = a[0].u.list;
    size_t n = l->len;
    l->len = 0;
    for (size_t i = 0; i < n; i++) tval_free(&l->items[i]);
    r->t = T_NIL;
}
static void m_list_insert(TVM *vm, TVal *a, int argc, TVal *r) {
    NARG(3);
    if (a[1].t != T_INT) die(vm, "insert(i, x)");
    TList *l = a[0].u.list;
    int64_t i = a[1].u.i, n = (int64_t)l->len;
    if (i < 0) i += n;
    if (i < 0) i = 0;
    if (i > n) i = n;
    TVal v = a[2]; tval_retain(&v);
    list_push(l, mk_nil());
    memmove(&l->items[i + 1], &l->items[i], (size_t)(n - i) * sizeof(TVal));
    l->items[i] = v;
    r->t = T_NIL;
}
static void m_list_extend(TVM *vm, TVal *a, int argc, TVal *r) {
    NARG(2);
    if (a[1].t != T_LIST && a[1].t != T_STR && a[1].t != T_TENSOR) die(vm, "extend() list");
    size_t n = seq_len(&a[1]);
    TList *l = a[0].u.list;
    for (size_t i = 0; i < n; i++) {
        TVal e; seq_get(&a[1], i, &e);
        if (a[1].t == T_LIST) tval_retain(&e);
        list_push(l, e);
    }
    r->t = T_NIL;
}
static void m_list_index(TVM *vm, TVal *a, int argc, TVal *r) {
    NARG(2);
    TList *l = a[0].u.list;
    for (size_t i = 0; i < l->len; i++)
        if (tval_eq(&l->items[i], &a[1])) { *r = mk_int((int64_t)i); return; }
    die(vm, "index: not found");
}
static int str_find(const TVal *h, const TVal *n) {
    if (n->u.str.n == 0) return 0;
    for (size_t i = 0; i + n->u.str.n <= h->u.str.n; i++)
        if (memcmp(h->u.str.s + i, n->u.str.s, n->u.str.n) == 0) return (int)i;
    return -1;
}
static void m_str_find(TVM *vm, TVal *a, int argc, TVal *r) {
    NARG(2);
    if (a[1].t != T_STR) die(vm, "find() str");
    *r = mk_int(str_find(&a[0], &a[1]));
}
static void m_str_replace(TVM *vm, TVal *a, int argc, TVal *r) {
    NARG(3);
    if (a[1].t != T_STR || a[2].t != T_STR) die(vm, "replace() str");
    const TVal *s0 = &a[0], *x = &a[1], *y = &a[2];
    if (x->u.str.n == 0) { *r = mk_str_n(s0->u.str.s, s0->u.str.n); return; }
    Buf b = {0, 0, 0};
    size_t i = 0;
    while (i < s0->u.str.n) {
        if (i + x->u.str.n <= s0->u.str.n &&
            memcmp(s0->u.str.s + i, x->u.str.s, x->u.str.n) == 0) {
            buf_put(&b, y->u.str.s, y->u.str.n);
            i += x->u.str.n;
        } else { buf_c(&b, s0->u.str.s[i]); i++; }
    }
    *r = mk_str_n(b.s ? b.s : "", b.n);
    free(b.s);
}
static void m_str_startswith(TVM *vm, TVal *a, int argc, TVal *r) {
    NARG(2);
    if (a[1].t != T_STR) die(vm, "startswith() str");
    *r = mk_bool(a[1].u.str.n <= a[0].u.str.n &&
                 memcmp(a[0].u.str.s, a[1].u.str.s, a[1].u.str.n) == 0);
}
static void m_str_endswith(TVM *vm, TVal *a, int argc, TVal *r) {
    NARG(2);
    if (a[1].t != T_STR) die(vm, "endswith() str");
    size_t n = a[0].u.str.n, m = a[1].u.str.n;
    *r = mk_bool(m <= n && memcmp(a[0].u.str.s + n - m, a[1].u.str.s, m) == 0);
}
/* a in b (khớp t_in của t.py) */
static int val_in(TVM *vm, TVal *a, TVal *b) {
    if (b->t == T_STR) {
        if (a->t != T_STR) die(vm, "in: str needs str");
        return str_find(b, a) >= 0;
    }
    if (b->t == T_LIST) {
        for (size_t i = 0; i < b->u.list->len; i++)
            if (tval_eq(a, &b->u.list->items[i])) return 1;
        return 0;
    }
    if (b->t == T_DICT && b->u.dict->kind <= DK_SET)
        return dict_find(b->u.dict, a) >= 0;
    if (b->t == T_TENSOR) {
        int ok = 1; double x = to_num(a, &ok);
        if (!ok) return 0;
        ten_sync(b->u.ten);
        for (size_t i = 0; i < b->u.ten->len; i++) if (b->u.ten->d[i] == x) return 1;
        return 0;
    }
    {
        char eb[96]; snprintf(eb, sizeof(eb), "in: unsupported %.40s", tname_of(b));
        die(vm, eb);
    }
    return 0;
}

/* ---------- attr ---------- */
/* method của kiểu builtin (gọi với a[0] = obj); NULL nếu không phải method */
static TNativeFn builtin_method(const TVal *o, const char *nb) {
    if (o->t == T_STR) {
        if (!strcmp(nb, "upper")) return m_str_upper;
        if (!strcmp(nb, "lower")) return m_str_lower;
        if (!strcmp(nb, "split")) return m_str_split;
        if (!strcmp(nb, "join")) return m_str_join;
        if (!strcmp(nb, "strip")) return m_str_strip;
        if (!strcmp(nb, "find")) return m_str_find;
        if (!strcmp(nb, "replace")) return m_str_replace;
        if (!strcmp(nb, "startswith")) return m_str_startswith;
        if (!strcmp(nb, "endswith")) return m_str_endswith;
    } else if (o->t == T_LIST) {
        if (!strcmp(nb, "append")) return m_list_append;
        if (!strcmp(nb, "pop")) return m_list_pop;
        if (!strcmp(nb, "next")) return m_list_next;
        if (!strcmp(nb, "clear")) return m_list_clear;
        if (!strcmp(nb, "insert")) return m_list_insert;
        if (!strcmp(nb, "extend")) return m_list_extend;
        if (!strcmp(nb, "index")) return m_list_index;
    } else if (o->t == T_DICT && o->u.dict->kind == DK_DICT) {
        if (!strcmp(nb, "keys")) return m_dict_keys;
        if (!strcmp(nb, "values")) return m_dict_values;
        if (!strcmp(nb, "get")) return m_dict_get;
        if (!strcmp(nb, "items")) return m_dict_items;
        if (!strcmp(nb, "has")) return m_has;
        if (!strcmp(nb, "remove")) return m_remove;
    } else if (o->t == T_DICT && o->u.dict->kind == DK_SET) {
        if (!strcmp(nb, "add")) return m_set_add;
        if (!strcmp(nb, "has")) return m_has;
        if (!strcmp(nb, "remove")) return m_remove;
        if (!strcmp(nb, "list")) return m_set_list;
        if (!strcmp(nb, "union")) return m_set_union;
        if (!strcmp(nb, "inter")) return m_set_inter;
        if (!strcmp(nb, "diff")) return m_set_diff;
    } else if (o->t == T_TENSOR) {
        return tensor_method(nb);
    }
    return NULL;
}
static const char *kind_name(const TVal *o) {
    if (o->t == T_DICT && o->u.dict->kind == DK_SET) return "set";
    if (o->t == T_DICT && o->u.dict->kind == DK_DICT) return "dict";
    return tname_of(o);
}
/* giá trị o.name (mới/đã retain); không đụng tới o */
static TVal attr_value(TVM *vm, TVal *o, const char *name, size_t namelen) {
    char nb[64];
    TVal r; r.t = T_NIL;
    if (namelen >= sizeof(nb)) die(vm, "attr name");
    memcpy(nb, name, namelen); nb[namelen] = 0;
    if (o->t == T_DICT && o->u.dict->kind == DK_REC) {
        TDict *d = o->u.dict;
        TVal key = mk_str_n(nb, namelen);
        int i = dict_find(d, &key);
        int mi = (i < 0 && d->meths) ? dict_find(d->meths, &key) : -1;
        tval_free(&key);
        if (i >= 0) { mod_pull(d, (size_t)i); r = d->vals[i]; tval_retain(&r); return r; }
        if (mi >= 0) {
            TList *pl = list_new(2);
            TVal f = d->meths->vals[mi], self = *o;
            tval_retain(&f); tval_retain(&self);
            list_push(pl, f); list_push(pl, self);
            TVal plv; plv.t = T_LIST; plv.u.list = pl;
            r = mk_bound(m_call_bound, "method", plv);
            tval_free(&plv);
            return r;
        }
        {
            char eb[128];
            snprintf(eb, sizeof(eb), "%.40s has no .%.40s", rec_name(d), nb);
            die(vm, eb);
        }
    }
    if (!strcmp(nb, "len")) {
        if (o->t == T_STR) return mk_int((int64_t)o->u.str.n);
        if (o->t == T_LIST) return mk_int((int64_t)o->u.list->len);
        if (o->t == T_DICT && o->u.dict->kind <= DK_SET) return mk_int((int64_t)o->u.dict->len);
        if (o->t == T_TENSOR) return mk_int((int64_t)tensor_len0(o->u.ten));
    }
    if (o->t == T_TENSOR) {
        int ok = 0;
        r = tensor_prop(vm, o->u.ten, nb, &ok);
        if (ok) return r;
    }
    {
        TNativeFn fn = builtin_method(o, nb);
        if (fn) return mk_bound(fn, "method", *o);
    }
    if (o->t == T_STR || o->t == T_LIST || o->t == T_TENSOR ||
        (o->t == T_DICT && o->u.dict->kind <= DK_SET)) {
        char eb[96]; snprintf(eb, sizeof(eb), "%s has no .%.40s", kind_name(o), nb);
        die(vm, eb);
    }
    {
        char eb[96]; snprintf(eb, sizeof(eb), ".%-.40s here", nb);
        die(vm, eb);
    }
    return r;
}
/* so tên ngắn (field/method) không qua memcmp của libc */
TVM_INLINE int name_eq(const char *a, const char *b, size_t n) {
    for (size_t i = 0; i < n; i++) if (a[i] != b[i]) return 0;
    return 1;
}
static void do_attr(TVM *vm, const char *name, size_t namelen, TAttrIC *ic) {
    TVal *ov = &vm->stack[vm->sp - 1];
    /* inline cache: field record theo kiểu */
    if (ov->t == T_DICT && ov->u.dict->kind == DK_REC) {
        TDict *d = ov->u.dict;
        if (ic && ic->kind == 1 && ic->key == d->meths && (size_t)ic->idx < d->len &&
            d->keys[ic->idx].u.str.n == namelen &&
            name_eq(d->keys[ic->idx].u.str.s, name, namelen)) {
            mod_pull(d, (size_t)ic->idx);
            TVal r = d->vals[ic->idx]; TV_RETAIN(&r);
            tval_free(ov);
            *ov = r;
            return;
        }
        if (ic && d->len < 64) {
            for (size_t i = 0; i < d->len; i++) {
                const TVal *k = &d->keys[i];
                if (k->u.str.n == namelen && memcmp(k->u.str.s, name, namelen) == 0) {
                    ic->kind = 1; ic->key = d->meths; ic->idx = (int32_t)i;
                    break;
                }
            }
        }
    }
    {
        TVal r = attr_value(vm, ov, name, namelen);
        TVal o = pop(vm); tval_free(&o);
        push(vm, r);
    }
}

/* ---------- arith ---------- */
/* % theo ngữ nghĩa Python (floor): dấu theo số chia — khớp t.py */
int64_t imod(int64_t a, int64_t b) {
    if (b == -1) return 0;               /* tránh INT64_MIN % -1 (trap) */
    int64_t r = a % b;
    if (r != 0 && ((r < 0) != (b < 0))) r += b;
    return r;
}
double fmod_floor(double a, double b) {
    double r = fmod(a, b);
    if (r != 0 && ((r < 0) != (b < 0))) r += b;
    return r;
}
static void arith(TVM *vm, int op) {
    TVal b = pop(vm), a = pop(vm);
    TVal r; r.t = T_NIL;
    if (a.t == T_DICT && a.u.dict->kind == DK_REC) {
        const char *dn = op == OP_ADD ? "__add__" : op == OP_SUB ? "__sub__" :
                         op == OP_MUL ? "__mul__" : op == OP_DIV ? "__div__" :
                         op == OP_MOD ? "__mod__" : op == OP_POW ? "__pow__" :
                         op == OP_TDOT ? "__matmul__" : NULL;
        if (dn && rec_call_dunder(vm, &a, dn, &b, NULL, 1, &r)) {
            tval_free(&a); tval_free(&b);
            push(vm, r); return;
        }
    }
    /* tensor paths */
    if (a.t == T_TENSOR || b.t == T_TENSOR) {
        if (op != OP_ADD && op != OP_SUB && op != OP_MUL && op != OP_DIV &&
            op != OP_POW && op != OP_TDOT) {
            tval_free(&a); tval_free(&b); die(vm, "bad tensor op");
        }
        /* a, b vẫn trên stack của C: die() trong phép tensor làm rò tối đa 2 ref
           (chấp nhận, như mọi lỗi số học khác) */
        if (op == OP_TDOT) {
            if (a.t != T_TENSOR || b.t != T_TENSOR) {
                tval_free(&a); tval_free(&b); die(vm, "matmul cần 2 tensor");
            }
            r = ten_matmul(vm, a.u.ten, b.u.ten);
        } else {
            if ((a.t != T_TENSOR && a.t != T_INT && a.t != T_FLOAT) ||
                (b.t != T_TENSOR && b.t != T_INT && b.t != T_FLOAT)) {
                tval_free(&a); tval_free(&b); die(vm, "bad tensor op");
            }
            r = ten_val(ten_binop(vm, op, &a, &b));
        }
        tval_free(&a); tval_free(&b);
        push(vm, r);
        return;
    }
    if (op == OP_ADD) {
        if (a.t == T_STR && b.t == T_STR) {
            size_t n = a.u.str.n + b.u.str.n;
            TVal v = mk_str_uninit(n);
            memcpy(v.u.str.s, a.u.str.s, a.u.str.n);
            memcpy(v.u.str.s + a.u.str.n, b.u.str.s, b.u.str.n);
            tval_free(&a); tval_free(&b);
            push(vm, v); return;
        }
        if (a.t == T_LIST && b.t == T_LIST) {
            TList *l = list_new(a.u.list->len + b.u.list->len);
            for (size_t i = 0; i < a.u.list->len; i++) {
                TVal v = a.u.list->items[i]; tval_retain(&v); list_push(l, v);
            }
            for (size_t i = 0; i < b.u.list->len; i++) {
                TVal v = b.u.list->items[i]; tval_retain(&v); list_push(l, v);
            }
            tval_free(&a); tval_free(&b);
            r.t = T_LIST; r.u.list = l;
            push(vm, r); return;
        }
        if (a.t == T_INT && b.t == T_INT) r = mk_int(a.u.i + b.u.i);
        else {
            int oka = 1, okb = 1;
            double x = to_num(&a, &oka), y = to_num(&b, &okb);
            if (!oka || !okb) { tval_free(&a); tval_free(&b); die(vm, "bad add"); }
            r = mk_float(x + y);
        }
        tval_free(&a); tval_free(&b); push(vm, r); return;
    }
    if (op == OP_MUL && ((a.t == T_LIST && b.t == T_INT) ||
                         (a.t == T_INT && b.t == T_LIST))) {
        TList *src = a.t == T_LIST ? a.u.list : b.u.list;
        int64_t k = a.t == T_INT ? a.u.i : b.u.i;
        if (k < 0) k = 0;
        TList *l = list_new(src->len * (size_t)k);
        for (int64_t j = 0; j < k; j++)
            for (size_t i = 0; i < src->len; i++) {
                TVal v = src->items[i]; tval_retain(&v); list_push(l, v);
            }
        tval_free(&a); tval_free(&b);
        r.t = T_LIST; r.u.list = l;
        push(vm, r); return;
    }
    {
        int oka = 1, okb = 1;
        double x = to_num(&a, &oka), y = to_num(&b, &okb);
        if (!oka || !okb) { tval_free(&a); tval_free(&b); die(vm, "arith numbers"); }
        int both = (a.t == T_INT && b.t == T_INT);
        if (op == OP_SUB) r = both ? mk_int(a.u.i - b.u.i) : mk_float(x - y);
        else if (op == OP_MUL) r = both ? mk_int(a.u.i * b.u.i) : mk_float(x * y);
        else if (op == OP_DIV) {
            if (y == 0) { tval_free(&a); tval_free(&b); die(vm, "division by zero"); }
            r = mk_float(x / y);
        }
        else if (op == OP_MOD) {
            if (y == 0) { tval_free(&a); tval_free(&b); die(vm, "modulo by zero"); }
            r = both ? mk_int(imod(a.u.i, b.u.i)) : mk_float(fmod_floor(x, y));
        }
        else if (op == OP_POW) {
            if (both && b.u.i >= 0) {
                int64_t p = 1;
                for (int64_t j = 0; j < b.u.i; j++) p *= a.u.i;
                r = mk_int(p);
            } else r = mk_float(pow(x, y));
        }
        else { tval_free(&a); tval_free(&b); die(vm, "bad op"); }
        tval_free(&a); tval_free(&b);
        push(vm, r);
    }
}

/* ---------- compare ---------- */
static TTensor *ten_binop(TVM *vm, int op, const TVal *a, const TVal *b);
static void cmpop(TVM *vm, int op) {
    TVal b = pop(vm), a = pop(vm);
    int res = 0;
    if (a.t == T_DICT && a.u.dict->kind == DK_REC) {
        const char *dn = op == OP_EQ ? "__eq__" : op == OP_NE ? "__ne__" :
                         op == OP_LT ? "__lt__" : op == OP_GT ? "__gt__" :
                         op == OP_LE ? "__le__" : op == OP_GE ? "__ge__" : NULL;
        TVal r = mk_nil();
        if (dn && rec_call_dunder(vm, &a, dn, &b, NULL, 1, &r)) {
            tval_free(&a); tval_free(&b);
            push(vm, r); return;
        }
        if (op == OP_NE && rec_call_dunder(vm, &a, "__eq__", &b, NULL, 1, &r)) {
            int eq = tval_truthy(&r);
            tval_free(&r); tval_free(&a); tval_free(&b);
            push(vm, mk_bool(!eq)); return;
        }
    }
    /* < > <= >= với tensor: từng phần tử -> tensor 0/1 (== != so cả tensor) */
    if ((a.t == T_TENSOR || b.t == T_TENSOR) && op != OP_EQ && op != OP_NE) {
        if ((a.t != T_TENSOR && a.t != T_INT && a.t != T_FLOAT) ||
            (b.t != T_TENSOR && b.t != T_INT && b.t != T_FLOAT)) {
            tval_free(&a); tval_free(&b); die(vm, "bad tensor op");
        }
        TVal r; r.t = T_TENSOR; r.u.ten = ten_binop(vm, op, &a, &b);
        tval_free(&a); tval_free(&b);
        push(vm, r);
        return;
    }
    if (op == OP_EQ) res = tval_eq(&a, &b);
    else if (op == OP_NE) res = !tval_eq(&a, &b);
    else if (a.t == T_INT && b.t == T_INT) {
        int64_t x = a.u.i, y = b.u.i;
        res = op == OP_LT ? x < y : op == OP_GT ? x > y : op == OP_LE ? x <= y : x >= y;
    } else {
        int oka = 1, okb = 1;
        double x = to_num(&a, &oka), y = to_num(&b, &okb);
        if (oka && okb) {
            if (op == OP_LT) res = x < y;
            else if (op == OP_GT) res = x > y;
            else if (op == OP_LE) res = x <= y;
            else res = x >= y;
        } else if (a.t == T_STR && b.t == T_STR) {
            size_t n = a.u.str.n < b.u.str.n ? a.u.str.n : b.u.str.n;
            int c = memcmp(a.u.str.s, b.u.str.s, n);
            if (c == 0) c = (a.u.str.n < b.u.str.n) ? -1 : (a.u.str.n > b.u.str.n);
            if (op == OP_LT) res = c < 0;
            else if (op == OP_GT) res = c > 0;
            else if (op == OP_LE) res = c <= 0;
            else res = c >= 0;
        } else {
            tval_free(&a); tval_free(&b); die(vm, "compare here");
        }
    }
    TV_FREE(&a); TV_FREE(&b);
    push(vm, mk_bool(res));
}

/* ---------- frames ---------- */
static TFrame *push_frame(TVM *vm) {    if (vm->nframes >= vm->fcap) {
        vm->fcap *= 2;
        vm->frames = (TFrame*)realloc(vm->frames, vm->fcap * sizeof(TFrame));
        if (!vm->frames) exit(1);
    }
    {
        TFrame *fr = &vm->frames[vm->nframes++];
        if (vm->nframes > vm->fmax) {
            memset(fr, 0, sizeof(*fr));
            vm->fmax = vm->nframes;
        }
        return fr;
    }
}
static void free_frame(TVM *vm) {
    TFrame *f = &vm->frames[--vm->nframes];
    for (uint16_t i = 0; i < f->nlocals; i++) TV_FREE(&f->locals[i]);
    /* giữ mảng locals (lcap) để call sau tái dùng; free thật ở tvm_free */
    f->nlocals = 0;
    if (f->fn) {
        TVal fv; fv.t = T_FN; fv.u.fn = f->fn;
        tval_free(&fv);
        f->fn = NULL;
    }
}

/* ---------- init/free ---------- */
/* ---------- đọc .tb + bytecode verifier + load() + __sys ---------- */
static int rd_exact(FILE *f, void *b, size_t n) { return n == 0 || fread(b, 1, n, f) == n; }

static int verify_bytecode(const TProg *p, char *msg, size_t msgn) {
    const uint8_t *c = p->code;
    size_t len = p->len, pc = 0;
    uint16_t ng = p->nglobals < NAT_NFIXED ? NAT_NFIXED : p->nglobals;
    while (pc < len) {
        uint8_t op = c[pc];
        size_t ilen = tvm_oplen(c, len, pc);
        if (op < OP_PUSH_NIL || op > OP_HALT) {
            snprintf(msg, msgn, "bad opcode %u at %zu", (unsigned)op, pc);
            return -1;
        }
        if (pc + ilen > len || ( (op == OP_PUSH_STR || op == OP_ATTR || op == OP_SETATTR || op == OP_CALLM) && ilen == 1 )) {
            snprintf(msg, msgn, "truncated instruction at %zu", pc);
            return -1;
        }
        if (op == OP_LOADG || op == OP_STOREG || op == OP_DECLG || op == OP_BINDG ||
            op == OP_CAPG || op == OP_INC_G || op == OP_DEC_G || op == OP_INCB_G || op == OP_ADDI_G) {
            uint16_t g;
            memcpy(&g, c + pc + 1, 2);
            if (g >= ng) { snprintf(msg, msgn, "global slot %u out of range (%u)", (unsigned)g, (unsigned)ng); return -1; }
        } else if (op == OP_ADDS_G || op == OP_SUBS_G) {
            uint16_t g1, g2;
            memcpy(&g1, c + pc + 1, 2);
            memcpy(&g2, c + pc + 3, 2);
            if (g1 >= ng || g2 >= ng) { snprintf(msg, msgn, "global slot out of range"); return -1; }
        }
        pc += ilen;
    }
    return 0;
}

int tvm_read_tb(const char *path, TProg *p, char *msg, size_t msgn) {
    FILE *f = fopen(path, "rb");
    char magic[4];
    uint16_t ng = 0;
    uint32_t len = 0;
    if (!f) { snprintf(msg, msgn, "no file %s", path); return -1; }
    if (!rd_exact(f, magic, 4) || memcmp(magic, "TB01", 4)) {
        snprintf(msg, msgn, "bad magic"); fclose(f); return -1;
    }
    if (!rd_exact(f, &ng, 2)) { snprintf(msg, msgn, "bad nglobals"); fclose(f); return -1; }
    if (!rd_exact(f, &len, 4) || len > 64u * 1024u * 1024u) {
        snprintf(msg, msgn, "bad len"); fclose(f); return -1;
    }
    p->nglobals = ng < NAT_NFIXED ? NAT_NFIXED : ng;
    p->code = (uint8_t*)calloc((size_t)len + 16, 1);
    if (!p->code) { snprintf(msg, msgn, "out of memory"); fclose(f); return -1; }
    p->len = len;
    if (!rd_exact(f, p->code, len)) {
        snprintf(msg, msgn, "short read"); fclose(f); free(p->code); p->code = NULL; return -1;
    }
    p->code[len] = OP_HALT;
    if (verify_bytecode(p, msg, msgn) != 0) {
        fclose(f); free(p->code); p->code = NULL; return -1;
    }
    {
        /* bảng dòng tuỳ chọn: "LN01" u16 nfile (u16 len + tên)* u32 n (u32 pc, u32 dòng, u16 file)* */
        char mg[4];
        int have = rd_exact(f, mg, 4);
        if (have && !memcmp(mg, "LN01", 4)) {
            uint16_t nf = 0;
            uint32_t nl = 0;
            int ok = rd_exact(f, &nf, 2);
            if (ok) p->lfiles = (char**)calloc(nf ? nf : 1, sizeof(char*));
            for (uint16_t i = 0; ok && i < nf; i++) {
                uint16_t ln = 0;
                ok = rd_exact(f, &ln, 2);
                if (!ok) break;
                p->lfiles[i] = (char*)calloc((size_t)ln + 1, 1);
                if (!p->lfiles[i]) { ok = 0; break; }
                ok = rd_exact(f, p->lfiles[i], ln);
                p->nlfiles = (uint16_t)(i + 1);
            }
            if (ok) ok = rd_exact(f, &nl, 4) && nl <= 16u * 1024u * 1024u;
            if (ok && nl) {
                p->lpc = (uint32_t*)malloc(sizeof(uint32_t) * (size_t)nl);
                p->lline = (uint32_t*)malloc(sizeof(uint32_t) * (size_t)nl);
                p->lfile = (uint16_t*)malloc(sizeof(uint16_t) * (size_t)nl);
                if (!p->lpc || !p->lline || !p->lfile) ok = 0;
                for (uint32_t i = 0; ok && i < nl; i++) {
                    ok = rd_exact(f, &p->lpc[i], 4) && rd_exact(f, &p->lline[i], 4) &&
                         rd_exact(f, &p->lfile[i], 2);
                    if (ok) p->nlines = i + 1;
                }
            }
            have = ok && rd_exact(f, mg, 4);
        }
        /* bảng export: "EX01" u16 n (u16 len + tên + u16 slot)* */
        if (have && !memcmp(mg, "EX01", 4)) {
            uint16_t ne = 0;
            int ok = rd_exact(f, &ne, 2);
            if (ok) {
                p->exnames = (char**)calloc(ne ? ne : 1, sizeof(char*));
                p->exslots = (uint16_t*)calloc(ne ? ne : 1, sizeof(uint16_t));
                p->exmut = (uint8_t*)calloc(ne ? ne : 1, 1);
                if (!p->exnames || !p->exslots || !p->exmut) ok = 0;
            }
            for (uint16_t i = 0; ok && i < ne; i++) {
                uint16_t ln = 0;
                ok = rd_exact(f, &ln, 2);
                if (!ok) break;
                p->exnames[i] = (char*)calloc((size_t)ln + 1, 1);
                if (!p->exnames[i]) { ok = 0; break; }
                ok = rd_exact(f, p->exnames[i], ln) && rd_exact(f, &p->exslots[i], 2) &&
                     rd_exact(f, &p->exmut[i], 1);
                if (ok && p->exslots[i] < p->nglobals) p->nexports = (uint16_t)(i + 1);
                else { free(p->exnames[i]); p->exnames[i] = NULL; ok = 0; }
            }
        }
    }
    fclose(f);
    p->path = (char*)malloc(strlen(path) + 1);
    if (p->path) strcpy(p->path, path);
    return 0;
}
void tvm_prog_release(TProg *p) {
    free(p->code); p->code = NULL;
    for (uint16_t i = 0; i < p->nlfiles; i++) free(p->lfiles[i]);
    free(p->lfiles); free(p->lpc); free(p->lline); free(p->lfile);
    p->lfiles = NULL; p->lpc = p->lline = NULL; p->lfile = NULL; p->nlfiles = 0; p->nlines = 0;
    for (uint16_t i = 0; i < p->nexports; i++) free(p->exnames[i]);
    free(p->exnames); free(p->exslots); free(p->exmut);
    p->exnames = NULL; p->exslots = NULL; p->exmut = NULL; p->nexports = 0;
    free(p->path); p->path = NULL;
}

static void prog_init_globals(TProg *p);
static int file_exists(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) return 0;
    fclose(f);
    return 1;
}
const char *tvm_exe_path = NULL;
static int stat_clean(const char *path, struct stat *st) {
    char tmp[4096];
    size_t n = strlen(path);
    if (n >= sizeof(tmp)) return -1;
    memcpy(tmp, path, n + 1);
    while (n > 1 && (tmp[n - 1] == '/' || tmp[n - 1] == '\\')) {
        if (n == 3 && tmp[1] == ':') break;
        tmp[--n] = 0;
    }
    return stat(tmp, st);
}
static double file_mtime(const char *path) {
    struct stat st;
    if (stat_clean(path, &st) != 0) return -1;
    return (double)st.st_mtime;
}
/* tcomp.tb: $TCOMP, cạnh trun, thư mục hiện tại */
static int find_tcomp(char *out, size_t n) {
    const char *env = getenv("TCOMP");
    if (env && *env && file_exists(env)) { snprintf(out, n, "%s", env); return 1; }
    if (tvm_exe_path) {
        const char *s1 = strrchr(tvm_exe_path, '/'), *s2 = strrchr(tvm_exe_path, '\\');
        const char *sl = s1 > s2 ? s1 : s2;
        if (sl) {
            snprintf(out, n, "%.*s/tcomp.tb", (int)(sl - tvm_exe_path), tvm_exe_path);
            if (file_exists(out)) return 1;
        }
    }
    snprintf(out, n, "tcomp.tb");
    return file_exists(out);
}

/* Chạy tiến trình con KHÔNG qua shell (chống command injection CWE-78 tuyệt đối),
   thu stdout/stderr vào Buf, truyền stdin_data, cwd và env_dict tuỳ chọn. */
static int tvm_run_proc(const char **av, int ac, const char *in_s, size_t in_n,
                        const char *cwd, const TDict *env_d, Buf *bout, Buf *berr) {
#ifdef _WIN32
    SECURITY_ATTRIBUTES sa;
    HANDLE in_rd = NULL, in_wr = NULL, out_rd = NULL, out_wr = NULL, err_rd = NULL, err_wr = NULL;
    STARTUPINFOA si;
    PROCESS_INFORMATION pi;
    Buf cmd = {0, 0, 0};
    DWORD rc = 1;
    /* ghép command line chuẩn CRT Windows (quote từng arg an toàn, không qua cmd.exe) */
    for (int i = 0; i < ac; i++) {
        const char *s = av[i];
        if (i > 0) buf_c(&cmd, ' ');
        buf_c(&cmd, '"');
        for (size_t j = 0; s[j]; j++) {
            size_t bs = 0;
            while (s[j] == '\\') { bs++; j++; }
            if (!s[j]) {
                for (size_t k = 0; k < bs * 2; k++) buf_c(&cmd, '\\');
                break;
            } else if (s[j] == '"') {
                for (size_t k = 0; k < bs * 2 + 1; k++) buf_c(&cmd, '\\');
                buf_c(&cmd, '"');
            } else {
                for (size_t k = 0; k < bs; k++) buf_c(&cmd, '\\');
                buf_c(&cmd, s[j]);
            }
        }
        buf_c(&cmd, '"');
    }
    buf_c(&cmd, 0);
    memset(&sa, 0, sizeof(sa));
    sa.nLength = sizeof(sa);
    sa.bInheritHandle = TRUE;
    if (!CreatePipe(&in_rd, &in_wr, &sa, 0) ||
        !CreatePipe(&out_rd, &out_wr, &sa, 0) ||
        !CreatePipe(&err_rd, &err_wr, &sa, 0)) {
        free(cmd.s);
        return -1;
    }
    SetHandleInformation(in_wr, HANDLE_FLAG_INHERIT, 0);
    SetHandleInformation(out_rd, HANDLE_FLAG_INHERIT, 0);
    SetHandleInformation(err_rd, HANDLE_FLAG_INHERIT, 0);
    memset(&si, 0, sizeof(si));
    si.cb = sizeof(si);
    si.dwFlags = STARTF_USESTDHANDLES;
    si.hStdInput = in_rd;
    si.hStdOutput = out_wr;
    si.hStdError = err_wr;
    memset(&pi, 0, sizeof(pi));
    /* áp biến môi trường tạm nếu có env_d */
    if (env_d) {
        for (size_t i = 0; i < env_d->len; i++) {
            if (env_d->keys[i].t == T_STR && env_d->vals[i].t == T_STR)
                _putenv_s(env_d->keys[i].u.str.s, env_d->vals[i].u.str.s);
        }
    }
    if (!CreateProcessA(NULL, cmd.s, NULL, NULL, TRUE, 0, NULL,
                        (cwd && *cwd) ? cwd : NULL, &si, &pi)) {
        CloseHandle(in_rd); CloseHandle(in_wr);
        CloseHandle(out_rd); CloseHandle(out_wr);
        CloseHandle(err_rd); CloseHandle(err_wr);
        free(cmd.s);
        return -1;
    }
    CloseHandle(in_rd);
    CloseHandle(out_wr);
    CloseHandle(err_wr);
    if (in_s && in_n > 0) {
        DWORD wr = 0;
        WriteFile(in_wr, in_s, (DWORD)in_n, &wr, NULL);
    }
    CloseHandle(in_wr);
    {
        char tmp[4096];
        for (;;) {
            DWORD av_out = 0, av_err = 0, rd = 0;
            int busy = 0;
            if (PeekNamedPipe(out_rd, NULL, 0, NULL, &av_out, NULL) && av_out > 0) {
                if (ReadFile(out_rd, tmp, sizeof(tmp), &rd, NULL) && rd > 0) {
                    if (bout) buf_put(bout, tmp, rd);
                    busy = 1;
                }
            }
            if (PeekNamedPipe(err_rd, NULL, 0, NULL, &av_err, NULL) && av_err > 0) {
                if (ReadFile(err_rd, tmp, sizeof(tmp), &rd, NULL) && rd > 0) {
                    if (berr) buf_put(berr, tmp, rd);
                    busy = 1;
                }
            }
            if (!busy) {
                if (WaitForSingleObject(pi.hProcess, 1) == WAIT_OBJECT_0) {
                    while (PeekNamedPipe(out_rd, NULL, 0, NULL, &av_out, NULL) && av_out > 0 &&
                           ReadFile(out_rd, tmp, sizeof(tmp), &rd, NULL) && rd > 0)
                        if (bout) buf_put(bout, tmp, rd);
                    while (PeekNamedPipe(err_rd, NULL, 0, NULL, &av_err, NULL) && av_err > 0 &&
                           ReadFile(err_rd, tmp, sizeof(tmp), &rd, NULL) && rd > 0)
                        if (berr) buf_put(berr, tmp, rd);
                    break;
                }
            }
        }
    }
    GetExitCodeProcess(pi.hProcess, &rc);
    CloseHandle(pi.hThread);
    CloseHandle(pi.hProcess);
    CloseHandle(out_rd);
    CloseHandle(err_rd);
    free(cmd.s);
    return (int)rc;
#else
    int in_p[2], out_p[2], err_p[2];
    pid_t pid;
    if (pipe(in_p) != 0) return -1;
    if (pipe(out_p) != 0) { close(in_p[0]); close(in_p[1]); return -1; }
    if (pipe(err_p) != 0) { close(in_p[0]); close(in_p[1]); close(out_p[0]); close(out_p[1]); return -1; }
    pid = fork();
    if (pid < 0) {
        close(in_p[0]); close(in_p[1]); close(out_p[0]); close(out_p[1]); close(err_p[0]); close(err_p[1]);
        return -1;
    }
    if (pid == 0) {
        char **xav = (char**)calloc((size_t)ac + 1, sizeof(char*));
        dup2(in_p[0], 0); dup2(out_p[1], 1); dup2(err_p[1], 2);
        close(in_p[0]); close(in_p[1]); close(out_p[0]); close(out_p[1]); close(err_p[0]); close(err_p[1]);
        if (cwd && *cwd) { if (chdir(cwd) != 0) _exit(127); }
        if (env_d) {
            for (size_t i = 0; i < env_d->len; i++) {
                if (env_d->keys[i].t == T_STR && env_d->vals[i].t == T_STR)
                    setenv(env_d->keys[i].u.str.s, env_d->vals[i].u.str.s, 1);
            }
        }
        for (int i = 0; i < ac; i++) xav[i] = (char*)av[i];
        xav[ac] = NULL;
        execvp(xav[0], xav);
        _exit(127);
    }
    close(in_p[0]); close(out_p[1]); close(err_p[1]);
    if (in_s && in_n > 0) {
        size_t off = 0;
        while (off < in_n) {
            ssize_t w = write(in_p[1], in_s + off, in_n - off);
            if (w <= 0) break;
            off += (size_t)w;
        }
    }
    close(in_p[1]);
    {
        struct pollfd fds[2];
        int open_cnt = 2;
        char tmp[4096];
        fds[0].fd = out_p[0]; fds[0].events = POLLIN;
        fds[1].fd = err_p[0]; fds[1].events = POLLIN;
        while (open_cnt > 0) {
            if (poll(fds, 2, -1) <= 0) break;
            if (fds[0].revents & (POLLIN | POLLHUP)) {
                ssize_t r = read(out_p[0], tmp, sizeof(tmp));
                if (r > 0) { if (bout) buf_put(bout, tmp, (size_t)r); }
                else { fds[0].fd = -1; open_cnt--; }
            }
            if (fds[1].revents & (POLLIN | POLLHUP)) {
                ssize_t r = read(err_p[0], tmp, sizeof(tmp));
                if (r > 0) { if (berr) buf_put(berr, tmp, (size_t)r); }
                else { fds[1].fd = -1; open_cnt--; }
            }
        }
    }
    close(out_p[0]); close(err_p[0]);
    {
        int status = 0;
        waitpid(pid, &status, 0);
        return WIFEXITED(status) ? WEXITSTATUS(status) : 1;
    }
#endif
}

int tvm_compile_src(const char *src, const char *out, int bundle, char *msg, size_t msgn) {
    char tc[1024];
    const char *av[6];
    int ac = 0, rc;
    Buf bout = {0, 0, 0}, berr = {0, 0, 0};
    if (msgn) msg[0] = 0;
    if (!tvm_exe_path || !find_tcomp(tc, sizeof(tc))) {
        if (msgn) snprintf(msg, msgn, "không tìm thấy tcomp.tb");
        return -1;
    }
    av[ac++] = tvm_exe_path;
    av[ac++] = tc;
    if (bundle) av[ac++] = "-b";
    av[ac++] = "-q";
    av[ac++] = src;
    av[ac++] = out;
    fflush(stdout); fflush(stderr);
    rc = tvm_run_proc(av, ac, NULL, 0, NULL, NULL, &bout, &berr);
    if (rc != 0 && msgn) {
        if (berr.s && berr.n > 0) {
            size_t eol = 0;
            while (eol < berr.n && berr.s[eol] != '\n' && berr.s[eol] != '\r') eol++;
            berr.s[eol] = 0;
            const char *p = berr.s;
            if (!strncmp(p, "T error: ", 9)) p += 9;
            snprintf(msg, msgn, "%s", p);
        } else {
            snprintf(msg, msgn, "lỗi biên dịch");
        }
    }
    free(bout.s);
    free(berr.s);
    return rc == 0 ? 1 : -1;
}

static int compile_if_stale(const char *src, const char *out, char *msg, size_t msgn) {
    char tc[1024];
    double ts = file_mtime(src), tb = file_mtime(out);
    if (ts < 0 || (tb >= 0 && tb >= ts)) return 0;
    if (!tvm_exe_path || !find_tcomp(tc, sizeof(tc))) return 0;
    return tvm_compile_src(src, out, 0, msg, msgn);
}

/* ---------- SHA-256 chuẩn FIPS 180-4 (dùng cho tpkg / std/os.t) ---------- */
static uint32_t sha_rotr(uint32_t x, int n) { return (x >> n) | (x << (32 - n)); }
static void sha256_hex(const uint8_t *data, size_t len, char out[65]) {
    static const uint32_t K[64] = {
        0x428a2f98u,0x71374491u,0xb5c0fbcfu,0xe9b5dba5u,0x3956c25bu,0x59f111f1u,0x923f82a4u,0xab1c5ed5u,
        0xd807aa98u,0x12835b01u,0x243185beu,0x550c7dc3u,0x72be5d74u,0x80deb1feu,0x9bdc06a7u,0xc19bf174u,
        0xe49b69c1u,0xefbe4786u,0x0fc19dc6u,0x240ca1ccu,0x2de92c6fu,0x4a7484aau,0x5cb0a9dcu,0x76f988dau,
        0x983e5152u,0xa831c66du,0xb00327c8u,0xbf597fc7u,0xc6e00bf3u,0xd5a79147u,0x06ca6351u,0x14292967u,
        0x27b70a85u,0x2e1b2138u,0x4d2c6dfc,0x53380d13u,0x650a7354u,0x766a0abbu,0x81c2c92eu,0x92722c85u,
        0xa2bfe8a1u,0xa81a664bu,0xc24b8b70u,0xc76c51a3u,0xd192e819u,0xd6990624u,0xf40e3585u,0x106aa070u,
        0x19a4c116u,0x1e376c08u,0x2748774cu,0x34b0bcb5u,0x391c0cb3u,0x4ed8aa4au,0x5b9cca4fu,0x682e6ff3u,
        0x748f82eeu,0x78a5636fu,0x84c87814u,0x8cc70208u,0x90befffau,0xa4506cebu,0xbef9a3f7u,0xc67178f2u
    };
    uint32_t H[8] = {
        0x6a09e667u,0xbb67ae85u,0x3c6ef372u,0xa54ff53au,
        0x510e527fu,0x9b05688cu,0x1f83d9abu,0x5be0cd19u
    };
    size_t bitlen = len * 8;
    size_t padlen = ((len + 8) / 64 + 1) * 64;
    uint8_t *buf = (uint8_t*)calloc(padlen, 1);
    if (!buf) exit(1);
    if (len) memcpy(buf, data, len);
    buf[len] = 0x80;
    for (int i = 0; i < 8; i++) buf[padlen - 1 - i] = (uint8_t)((uint64_t)bitlen >> (8 * i));
    for (size_t off = 0; off < padlen; off += 64) {
        uint32_t W[64], a, b, c, d, e, f, g, h;
        for (int i = 0; i < 16; i++)
            W[i] = ((uint32_t)buf[off + 4*i] << 24) | ((uint32_t)buf[off + 4*i + 1] << 16) |
                   ((uint32_t)buf[off + 4*i + 2] << 8) | (uint32_t)buf[off + 4*i + 3];
        for (int i = 16; i < 64; i++) {
            uint32_t s0 = sha_rotr(W[i-15], 7) ^ sha_rotr(W[i-15], 18) ^ (W[i-15] >> 3);
            uint32_t s1 = sha_rotr(W[i-2], 17) ^ sha_rotr(W[i-2], 19) ^ (W[i-2] >> 10);
            W[i] = W[i-16] + s0 + W[i-7] + s1;
        }
        a = H[0]; b = H[1]; c = H[2]; d = H[3]; e = H[4]; f = H[5]; g = H[6]; h = H[7];
        for (int i = 0; i < 64; i++) {
            uint32_t S1 = sha_rotr(e, 6) ^ sha_rotr(e, 11) ^ sha_rotr(e, 25);
            uint32_t ch = (e & f) ^ ((~e) & g);
            uint32_t t1 = h + S1 + ch + K[i] + W[i];
            uint32_t S0 = sha_rotr(a, 2) ^ sha_rotr(a, 13) ^ sha_rotr(a, 22);
            uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
            uint32_t t2 = S0 + maj;
            h = g; g = f; f = e; e = d + t1; d = c; c = b; b = a; a = t1 + t2;
        }
        H[0] += a; H[1] += b; H[2] += c; H[3] += d;
        H[4] += e; H[5] += f; H[6] += g; H[7] += h;
    }
    free(buf);
    for (int i = 0; i < 8; i++) snprintf(out + i * 8, 9, "%08x", H[i]);
    out[64] = 0;
}

/* ---------- Các hàm hệ thống của load("__sys") ---------- */
static void sys_clock(TVM *vm, TVal *a, int argc, TVal *r) {
    (void)vm; (void)a; (void)argc;
#ifdef _WIN32
    LARGE_INTEGER f, c;
    QueryPerformanceFrequency(&f);
    QueryPerformanceCounter(&c);
    *r = mk_float((double)c.QuadPart / (double)f.QuadPart);
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    *r = mk_float((double)ts.tv_sec + (double)ts.tv_nsec * 1e-9);
#endif
}
static void sys_time(TVM *vm, TVal *a, int argc, TVal *r) {
    (void)vm; (void)a; (void)argc;
    *r = mk_float((double)time(NULL));
}
static void sys_sleep(TVM *vm, TVal *a, int argc, TVal *r) {
    int ok = 1;
    double s = argc > 0 ? to_num(&a[0], &ok) : 0.0;
    if (!ok) die(vm, "sleep(sec)");
    if (s > 0) {
#ifdef _WIN32
        Sleep((DWORD)(s * 1000.0));
#else
        usleep((useconds_t)(s * 1e6));
#endif
    }
    *r = mk_nil();
}
static void sys_getenv(TVM *vm, TVal *a, int argc, TVal *r) {
    if (argc < 1 || a[0].t != T_STR) die(vm, "getenv(str)");
    const char *v = getenv(a[0].u.str.s);
    if (v && *v) *r = mk_str(v);
    else if (argc >= 2) { *r = a[1]; tval_retain(r); }
    else *r = mk_nil();
}
static void sys_setenv(TVM *vm, TVal *a, int argc, TVal *r) {
    if (argc != 2 || a[0].t != T_STR || (a[1].t != T_STR && a[1].t != T_NIL))
        die(vm, "setenv(str, str|nil)");
#ifdef _WIN32
    _putenv_s(a[0].u.str.s, a[1].t == T_STR ? a[1].u.str.s : "");
#else
    if (a[1].t == T_STR && a[1].u.str.n > 0) setenv(a[0].u.str.s, a[1].u.str.s, 1);
    else unsetenv(a[0].u.str.s);
#endif
    *r = mk_bool(1);
}
static void sys_platform(TVM *vm, TVal *a, int argc, TVal *r) {
    (void)vm; (void)a; (void)argc;
#ifdef _WIN32
    *r = mk_str("win32");
#else
    *r = mk_str("posix");
#endif
}
static void sys_exepath(TVM *vm, TVal *a, int argc, TVal *r) {
    (void)vm; (void)a; (void)argc;
    *r = mk_str(tvm_exe_path ? tvm_exe_path : "trun");
}
static void sys_getcwd(TVM *vm, TVal *a, int argc, TVal *r) {
    char buf[4096];
    (void)vm; (void)a; (void)argc;
#ifdef _WIN32
    *r = _getcwd(buf, sizeof(buf)) ? mk_str(buf) : mk_str(".");
#else
    *r = getcwd(buf, sizeof(buf)) ? mk_str(buf) : mk_str(".");
#endif
}
static void sys_chdir(TVM *vm, TVal *a, int argc, TVal *r) {
    if (argc != 1 || a[0].t != T_STR) die(vm, "chdir(str)");
#ifdef _WIN32
    *r = mk_bool(_chdir(a[0].u.str.s) == 0);
#else
    *r = mk_bool(chdir(a[0].u.str.s) == 0);
#endif
}
static void sys_abspath(TVM *vm, TVal *a, int argc, TVal *r) {
    char buf[4096];
    if (argc != 1 || a[0].t != T_STR) die(vm, "abspath(str)");
#ifdef _WIN32
    if (_fullpath(buf, a[0].u.str.s, sizeof(buf))) *r = mk_str(buf);
    else *r = mk_str_n(a[0].u.str.s, a[0].u.str.n);
#else
    if (realpath(a[0].u.str.s, buf)) *r = mk_str(buf);
    else if (a[0].u.str.n > 0 && a[0].u.str.s[0] == '/') *r = mk_str_n(a[0].u.str.s, a[0].u.str.n);
    else if (getcwd(buf, sizeof(buf) - a[0].u.str.n - 2)) {
        size_t bl = strlen(buf);
        buf[bl] = '/';
        memcpy(buf + bl + 1, a[0].u.str.s, a[0].u.str.n + 1);
        *r = mk_str(buf);
    } else *r = mk_str_n(a[0].u.str.s, a[0].u.str.n);
#endif
}
static void sys_exists(TVM *vm, TVal *a, int argc, TVal *r) {
    struct stat st;
    if (argc != 1 || a[0].t != T_STR) die(vm, "exists(str)");
    *r = mk_bool(stat_clean(a[0].u.str.s, &st) == 0);
}
static void sys_isfile(TVM *vm, TVal *a, int argc, TVal *r) {
    struct stat st;
    if (argc != 1 || a[0].t != T_STR) die(vm, "isfile(str)");
    *r = mk_bool(stat_clean(a[0].u.str.s, &st) == 0 && (st.st_mode & S_IFMT) == S_IFREG);
}
static void sys_isdir(TVM *vm, TVal *a, int argc, TVal *r) {
    struct stat st;
    if (argc != 1 || a[0].t != T_STR) die(vm, "isdir(str)");
    *r = mk_bool(stat_clean(a[0].u.str.s, &st) == 0 && (st.st_mode & S_IFMT) == S_IFDIR);
}
static void sys_mtime(TVM *vm, TVal *a, int argc, TVal *r) {
    if (argc != 1 || a[0].t != T_STR) die(vm, "mtime(str)");
    *r = mk_float(file_mtime(a[0].u.str.s));
}
static int mkdir_rec(const char *path) {
    char tmp[4096];
    size_t n = strlen(path);
    struct stat st;
    if (n == 0 || n >= sizeof(tmp)) return 0;
    if (stat_clean(path, &st) == 0 && (st.st_mode & S_IFMT) == S_IFDIR) return 1;
    memcpy(tmp, path, n + 1);
    for (size_t i = 1; i < n; i++) {
        if (tmp[i] == '/' || tmp[i] == '\\') {
            char sv = tmp[i];
            if (i == 2 && tmp[1] == ':') continue;
            tmp[i] = 0;
#ifdef _WIN32
            _mkdir(tmp);
#else
            mkdir(tmp, 0777);
#endif
            tmp[i] = sv;
        }
    }
#ifdef _WIN32
    _mkdir(tmp);
#else
    mkdir(tmp, 0777);
#endif
    return stat_clean(tmp, &st) == 0 && (st.st_mode & S_IFMT) == S_IFDIR;
}
static void sys_mkdir(TVM *vm, TVal *a, int argc, TVal *r) {
    if (argc != 1 || a[0].t != T_STR) die(vm, "mkdir(str)");
    *r = mk_bool(mkdir_rec(a[0].u.str.s));
}
static int rmtree_rec(const char *path) {
    struct stat st;
    if (stat_clean(path, &st) != 0) return 0;
    if ((st.st_mode & S_IFMT) != S_IFDIR) return remove(path) == 0;
#ifdef _WIN32
    {
        char pat[4096], sub[4096];
        WIN32_FIND_DATAA fd;
        HANDLE h;
        snprintf(pat, sizeof(pat), "%s\\*", path);
        h = FindFirstFileA(pat, &fd);
        if (h != INVALID_HANDLE_VALUE) {
            do {
                if (!strcmp(fd.cFileName, ".") || !strcmp(fd.cFileName, "..")) continue;
                snprintf(sub, sizeof(sub), "%s\\%s", path, fd.cFileName);
                rmtree_rec(sub);
            } while (FindNextFileA(h, &fd));
            FindClose(h);
        }
        return _rmdir(path) == 0;
    }
#else
    {
        DIR *d = opendir(path);
        struct dirent *de;
        char sub[4096];
        if (d) {
            while ((de = readdir(d)) != NULL) {
                if (!strcmp(de->d_name, ".") || !strcmp(de->d_name, "..")) continue;
                snprintf(sub, sizeof(sub), "%s/%s", path, de->d_name);
                rmtree_rec(sub);
            }
            closedir(d);
        }
        return rmdir(path) == 0;
    }
#endif
}
static void sys_remove(TVM *vm, TVal *a, int argc, TVal *r) {
    if (argc != 1 || a[0].t != T_STR) die(vm, "remove(str)");
    *r = mk_bool(rmtree_rec(a[0].u.str.s));
}
static int cmp_tstr(const void *p1, const void *p2) {
    const TVal *a = (const TVal*)p1, *b = (const TVal*)p2;
    size_t n = a->u.str.n < b->u.str.n ? a->u.str.n : b->u.str.n;
    int c = n ? memcmp(a->u.str.s, b->u.str.s, n) : 0;
    if (c != 0) return c;
    return a->u.str.n < b->u.str.n ? -1 : (a->u.str.n > b->u.str.n ? 1 : 0);
}
static void sys_listdir(TVM *vm, TVal *a, int argc, TVal *r) {
    TList *l = list_new(8);
    if (argc != 1 || a[0].t != T_STR) { TVal lv; lv.t = T_LIST; lv.u.list = l; tval_free(&lv); die(vm, "listdir(str)"); }
#ifdef _WIN32
    {
        char pat[4096];
        WIN32_FIND_DATAA fd;
        HANDLE h;
        snprintf(pat, sizeof(pat), "%s\\*", a[0].u.str.s);
        h = FindFirstFileA(pat, &fd);
        if (h != INVALID_HANDLE_VALUE) {
            do {
                if (!strcmp(fd.cFileName, ".") || !strcmp(fd.cFileName, "..")) continue;
                list_push(l, mk_str(fd.cFileName));
            } while (FindNextFileA(h, &fd));
            FindClose(h);
        }
    }
#else
    {
        DIR *d = opendir(a[0].u.str.s);
        struct dirent *de;
        if (d) {
            while ((de = readdir(d)) != NULL) {
                if (!strcmp(de->d_name, ".") || !strcmp(de->d_name, "..")) continue;
                list_push(l, mk_str(de->d_name));
            }
            closedir(d);
        }
    }
#endif
    if (l->len > 1) qsort(l->items, l->len, sizeof(TVal), cmp_tstr);
    r->t = T_LIST; r->u.list = l;
}
static void sys_mktemp(TVM *vm, TVal *a, int argc, TVal *r) {
    static int seq = 0;
    const char *pre = (argc >= 1 && a[0].t == T_STR) ? a[0].u.str.s : "ttmp";
    const char *base = getenv("TEMP");
    char path[4096];
    (void)vm;
    if (!base || !*base) base = getenv("TMP");
#ifdef _WIN32
    if (!base || !*base) base = ".";
    snprintf(path, sizeof(path), "%s\\%s_%lu_%d", base, pre, (unsigned long)GetCurrentProcessId(), ++seq);
#else
    if (!base || !*base) base = "/tmp";
    snprintf(path, sizeof(path), "%s/%s_%lu_%d", base, pre, (unsigned long)getpid(), ++seq);
#endif
    mkdir_rec(path);
    *r = mk_str(path);
}
static void sys_sha256(TVM *vm, TVal *a, int argc, TVal *r) {
    char hex[65];
    if (argc != 1 || a[0].t != T_STR) die(vm, "sha256(str)");
    sha256_hex((const uint8_t*)a[0].u.str.s, a[0].u.str.n, hex);
    *r = mk_str_n(hex, 64);
}
static void sys_run(TVM *vm, TVal *a, int argc, TVal *r) {
    const char **av;
    int ac, rc;
    const char *in_s = NULL, *cwd = NULL;
    size_t in_n = 0;
    const TDict *env_d = NULL;
    Buf bout = {0, 0, 0}, berr = {0, 0, 0};
    TDict *d;
    if (argc < 1 || a[0].t != T_LIST || a[0].u.list->len == 0) die(vm, "run(args list[, input, cwd, env])");
    ac = (int)a[0].u.list->len;
    av = (const char**)calloc((size_t)ac + 1, sizeof(char*));
    for (int i = 0; i < ac; i++) {
        if (a[0].u.list->items[i].t != T_STR) { free((void*)av); die(vm, "run: arg phải là str"); }
        av[i] = a[0].u.list->items[i].u.str.s;
    }
    if (argc >= 2 && a[1].t == T_STR) { in_s = a[1].u.str.s; in_n = a[1].u.str.n; }
    if (argc >= 3 && a[2].t == T_STR) cwd = a[2].u.str.s;
    if (argc >= 4 && a[3].t == T_DICT) env_d = a[3].u.dict;
    rc = tvm_run_proc(av, ac, in_s, in_n, cwd, env_d, &bout, &berr);
    free((void*)av);
    d = dict_new();
    dict_set(d, mk_str("rc"), mk_int(rc));
    dict_set(d, mk_str("out"), mk_str_n(bout.s ? bout.s : "", bout.n));
    dict_set(d, mk_str("err"), mk_str_n(berr.s ? berr.s : "", berr.n));
    free(bout.s);
    free(berr.s);
    r->t = T_DICT; r->u.dict = d;
}
static void sys_stdin_line(TVM *vm, TVal *a, int argc, TVal *r) {
    Buf b = {0, 0, 0};
    int c, got = 0;
    (void)vm; (void)a; (void)argc;
    while ((c = fgetc(stdin)) != EOF) {
        got = 1;
        buf_c(&b, (char)c);
        if (c == '\n') break;
    }
    if (!got) { free(b.s); *r = mk_nil(); return; }
    *r = mk_str_n(b.s ? b.s : "", b.n);
    free(b.s);
}
static void sys_stdin_read(TVM *vm, TVal *a, int argc, TVal *r) {
    size_t n, got;
    TVal v;
    if (argc != 1 || a[0].t != T_INT || a[0].u.i < 0) die(vm, "stdin_read(n int)");
    n = (size_t)a[0].u.i;
    if (n == 0) { *r = mk_str_n("", 0); return; }
    v = mk_str_uninit(n);
    got = fread(v.u.str.s, 1, n, stdin);
    if (got == 0) { tval_free(&v); *r = mk_nil(); return; }
    v.u.str.n = got;
    v.u.str.s[got] = 0;
    *r = v;
}
static void sys_stdout_write(TVM *vm, TVal *a, int argc, TVal *r) {
    if (argc != 1 || a[0].t != T_STR) die(vm, "stdout_write(str)");
    fwrite(a[0].u.str.s, 1, a[0].u.str.n, stdout);
    fflush(stdout);
    *r = mk_int((int64_t)a[0].u.str.n);
}
static void sys_stderr_write(TVM *vm, TVal *a, int argc, TVal *r) {
    if (argc != 1 || a[0].t != T_STR) die(vm, "stderr_write(str)");
    fwrite(a[0].u.str.s, 1, a[0].u.str.n, stderr);
    fflush(stderr);
    *r = mk_int((int64_t)a[0].u.str.n);
}
static uint32_t zip_crc32(const uint8_t *buf, size_t len) {
    uint32_t crc = 0xFFFFFFFFu;
    for (size_t i = 0; i < len; i++) {
        crc ^= buf[i];
        for (int b = 0; b < 8; b++)
            crc = (crc >> 1) ^ (0xEDB88320u & (0u - (crc & 1u)));
    }
    return ~crc;
}
static void buf_u16le(Buf *b, uint16_t v) {
    uint8_t x[2]; x[0] = (uint8_t)(v & 0xFF); x[1] = (uint8_t)((v >> 8) & 0xFF);
    buf_put(b, (const char*)x, 2);
}
static void buf_u32le(Buf *b, uint32_t v) {
    uint8_t x[4];
    x[0] = (uint8_t)(v & 0xFF); x[1] = (uint8_t)((v >> 8) & 0xFF);
    x[2] = (uint8_t)((v >> 16) & 0xFF); x[3] = (uint8_t)((v >> 24) & 0xFF);
    buf_put(b, (const char*)x, 4);
}
static uint16_t rd_u16le(const uint8_t *p) { return (uint16_t)(p[0] | ((uint16_t)p[1] << 8)); }
static uint32_t rd_u32le(const uint8_t *p) {
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}
static int zip_safe_arcname(const char *nm, size_t n) {
    if (n == 0 || nm[0] == '/' || nm[0] == '\\') return 0;
    for (size_t i = 0; i < n; i++) {
        if (nm[i] == ':' || nm[i] == '\0') return 0;
        if (nm[i] == '.' && i + 1 < n && nm[i + 1] == '.') {
            int at_start = (i == 0 || nm[i - 1] == '/' || nm[i - 1] == '\\');
            int at_end = (i + 2 == n || nm[i + 2] == '/' || nm[i + 2] == '\\');
            if (at_start && at_end) return 0;
        }
    }
    return 1;
}
static void sys_zip_pack(TVM *vm, TVal *a, int argc, TVal *r) {
    Buf body = {0, 0, 0}, cdir = {0, 0, 0};
    TDict *d;
    uint16_t count = 0;
    FILE *fp;
    if (argc != 2 || a[0].t != T_DICT || a[1].t != T_STR) die(vm, "zip_pack(dict, out_path)");
    d = a[0].u.dict;
    for (size_t i = 0; i < d->len; i++) {
        if (d->keys[i].t != T_STR || d->vals[i].t != T_STR) {
            free(body.s); free(cdir.s);
            die(vm, "zip_pack: key/val phải là str");
        }
        const char *nm = d->keys[i].u.str.s;
        size_t nlen = d->keys[i].u.str.n;
        const uint8_t *dat = (const uint8_t*)d->vals[i].u.str.s;
        size_t dlen = d->vals[i].u.str.n;
        if (!zip_safe_arcname(nm, nlen)) {
            free(body.s); free(cdir.s);
            die(vm, "zip_pack: unsafe arcname");
        }
        uint32_t crc = zip_crc32(dat, dlen);
        uint32_t off = (uint32_t)body.n;
        /* Local File Header */
        buf_put(&body, "PK\x03\x04", 4);
        buf_u16le(&body, 20);
        buf_u16le(&body, 0);
        buf_u16le(&body, 0); /* method 0 = stored */
        buf_u16le(&body, 0);
        buf_u16le(&body, 0x21);
        buf_u32le(&body, crc);
        buf_u32le(&body, (uint32_t)dlen);
        buf_u32le(&body, (uint32_t)dlen);
        buf_u16le(&body, (uint16_t)nlen);
        buf_u16le(&body, 0);
        buf_put(&body, nm, nlen);
        buf_put(&body, (const char*)dat, dlen);
        /* Central Directory File Header */
        buf_put(&cdir, "PK\x01\x02", 4);
        buf_u16le(&cdir, 20);
        buf_u16le(&cdir, 20);
        buf_u16le(&cdir, 0);
        buf_u16le(&cdir, 0);
        buf_u16le(&cdir, 0);
        buf_u16le(&cdir, 0x21);
        buf_u32le(&cdir, crc);
        buf_u32le(&cdir, (uint32_t)dlen);
        buf_u32le(&cdir, (uint32_t)dlen);
        buf_u16le(&cdir, (uint16_t)nlen);
        buf_u16le(&cdir, 0);
        buf_u16le(&cdir, 0);
        buf_u16le(&cdir, 0);
        buf_u16le(&cdir, 0);
        buf_u32le(&cdir, 0);
        buf_u32le(&cdir, off);
        buf_put(&cdir, nm, nlen);
        count++;
    }
    uint32_t cd_off = (uint32_t)body.n;
    uint32_t cd_sz = (uint32_t)cdir.n;
    if (cdir.n) buf_put(&body, cdir.s, cdir.n);
    free(cdir.s);
    /* End of Central Directory */
    buf_put(&body, "PK\x05\x06", 4);
    buf_u16le(&body, 0);
    buf_u16le(&body, 0);
    buf_u16le(&body, count);
    buf_u16le(&body, count);
    buf_u32le(&body, cd_sz);
    buf_u32le(&body, cd_off);
    buf_u16le(&body, 0);
    fp = fopen(a[1].u.str.s, "wb");
    if (!fp) { free(body.s); die(vm, "zip_pack: không ghi được file"); }
    if (body.n) fwrite(body.s, 1, body.n, fp);
    fclose(fp);
    free(body.s);
    *r = mk_bool(1);
}
static void sys_zip_unpack(TVM *vm, TVal *a, int argc, TVal *r) {
    FILE *fp;
    long flen;
    uint8_t *buf;
    size_t n, pos = 0;
    TList *out;
    if (argc != 2 || a[0].t != T_STR || a[1].t != T_STR) die(vm, "zip_unpack(zip_path, dest_dir)");
    fp = fopen(a[0].u.str.s, "rb");
    if (!fp) die(vm, "zip_unpack: không mở được file zip");
    fseek(fp, 0, SEEK_END);
    flen = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    if (flen < 22) { fclose(fp); die(vm, "zip_unpack: file zip hỏng"); }
    n = (size_t)flen;
    buf = (uint8_t*)malloc(n + 1);
    if (fread(buf, 1, n, fp) != n) { fclose(fp); free(buf); die(vm, "zip_unpack: lỗi đọc"); }
    fclose(fp);
    mkdir_rec(a[1].u.str.s);
    out = list_new(8);
    while (pos + 4 <= n) {
        if (memcmp(buf + pos, "PK\x01\x02", 4) == 0 || memcmp(buf + pos, "PK\x05\x06", 4) == 0)
            break;
        if (memcmp(buf + pos, "PK\x03\x04", 4) != 0 || pos + 30 > n) {
            free(buf);
            { TVal lv; lv.t = T_LIST; lv.u.list = out; tval_free(&lv); }
            die(vm, "zip_unpack: header hỏng");
        }
        uint16_t method = rd_u16le(buf + pos + 8);
        uint32_t crc = rd_u32le(buf + pos + 14);
        uint32_t csz = rd_u32le(buf + pos + 18);
        uint32_t usz = rd_u32le(buf + pos + 22);
        uint16_t nlen = rd_u16le(buf + pos + 26);
        uint16_t xlen = rd_u16le(buf + pos + 28);
        pos += 30;
        if (pos + nlen + xlen + csz > n || method != 0 || csz != usz) {
            free(buf);
            { TVal lv; lv.t = T_LIST; lv.u.list = out; tval_free(&lv); }
            die(vm, "zip_unpack: định dạng nén không hỗ trợ hoặc dữ liệu cụt");
        }
        const char *nm = (const char*)(buf + pos);
        if (!zip_safe_arcname(nm, nlen)) {
            free(buf);
            { TVal lv; lv.t = T_LIST; lv.u.list = out; tval_free(&lv); }
            die(vm, "zip_unpack: unsafe path (zip-slip)");
        }
        const uint8_t *dat = buf + pos + nlen + xlen;
        if (zip_crc32(dat, csz) != crc) {
            free(buf);
            { TVal lv; lv.t = T_LIST; lv.u.list = out; tval_free(&lv); }
            die(vm, "zip_unpack: CRC32 sai");
        }
        char full[4096], arc[1024];
        if (nlen >= sizeof(arc)) nlen = sizeof(arc) - 1;
        memcpy(arc, nm, nlen); arc[nlen] = 0;
        snprintf(full, sizeof(full), "%s/%s", a[1].u.str.s, arc);
        /* tạo thư mục cha nếu có */
        char *sl = strrchr(full, '/');
        char *bsl = strrchr(full, '\\');
        if (bsl && (!sl || bsl > sl)) sl = bsl;
        if (sl && sl > full) {
            *sl = 0;
            mkdir_rec(full);
            *sl = '/';
        }
        FILE *of = fopen(full, "wb");
        if (!of) {
            free(buf);
            { TVal lv; lv.t = T_LIST; lv.u.list = out; tval_free(&lv); }
            die(vm, "zip_unpack: không tạo được file đích");
        }
        if (csz) fwrite(dat, 1, csz, of);
        fclose(of);
        list_push(out, mk_str_n(arc, nlen));
        pos += nlen + xlen + csz;
    }
    free(buf);
    r->t = T_LIST; r->u.list = out;
}
static void sys_exit(TVM *vm, TVal *a, int argc, TVal *r) {
    int code = (argc >= 1 && a[0].t == T_INT) ? (int)a[0].u.i : 0;
    (void)vm; (void)r;
    fflush(stdout); fflush(stderr);
    exit(code);
}

/* ========================================================================= */
/* Memory-Mapped I/O (mmap) & Zero-Copy Tensor Storage                       */
/* ========================================================================= */

#define MAX_MMAPS 64
typedef struct {
    int in_use;
    char *data;
    size_t size;
#ifdef _WIN32
    HANDLE hFile;
    HANDLE hMap;
#else
    int fd;
#endif
} TMMapSlot;

static TMMapSlot g_mmaps[MAX_MMAPS];

static void sys_mmap_open(TVM *vm, TVal *a, int argc, TVal *r) {
    if (argc != 1 || a[0].t != T_STR) die(vm, "mmap_open(path: str)");
    int slot = -1;
    for (int i = 0; i < MAX_MMAPS; i++) {
        if (!g_mmaps[i].in_use) { slot = i; break; }
    }
    if (slot < 0) die(vm, "mmap_open: too many open mmap files");
#ifdef _WIN32
    HANDLE hFile = CreateFileA(a[0].u.str.s, GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
    if (hFile == INVALID_HANDLE_VALUE) die(vm, "mmap_open: cannot open file");
    LARGE_INTEGER sz;
    if (!GetFileSizeEx(hFile, &sz)) { CloseHandle(hFile); die(vm, "mmap_open: cannot get file size"); }
    if (sz.QuadPart == 0) { CloseHandle(hFile); die(vm, "mmap_open: file is empty"); }
    HANDLE hMap = CreateFileMappingA(hFile, NULL, PAGE_READONLY, 0, 0, NULL);
    if (!hMap) { CloseHandle(hFile); die(vm, "mmap_open: CreateFileMapping failed"); }
    void *ptr = MapViewOfFile(hMap, FILE_MAP_READ, 0, 0, 0);
    if (!ptr) { CloseHandle(hMap); CloseHandle(hFile); die(vm, "mmap_open: MapViewOfFile failed"); }
    g_mmaps[slot].in_use = 1;
    g_mmaps[slot].data = (char*)ptr;
    g_mmaps[slot].size = (size_t)sz.QuadPart;
    g_mmaps[slot].hFile = hFile;
    g_mmaps[slot].hMap = hMap;
#else
    int fd = open(a[0].u.str.s, O_RDONLY);
    if (fd < 0) die(vm, "mmap_open: cannot open file");
    struct stat st;
    if (fstat(fd, &st) != 0 || st.st_size == 0) { close(fd); die(vm, "mmap_open: cannot stat or file is empty"); }
    void *ptr = mmap(NULL, (size_t)st.st_size, PROT_READ, MAP_SHARED, fd, 0);
    if (ptr == MAP_FAILED) { close(fd); die(vm, "mmap_open: mmap failed"); }
    g_mmaps[slot].in_use = 1;
    g_mmaps[slot].data = (char*)ptr;
    g_mmaps[slot].size = (size_t)st.st_size;
    g_mmaps[slot].fd = fd;
#endif
    *r = mk_int(slot);
}

static void sys_mmap_close(TVM *vm, TVal *a, int argc, TVal *r) {
    if (argc != 1 || a[0].t != T_INT) die(vm, "mmap_close(id: int)");
    int slot = (int)a[0].u.i;
    if (slot < 0 || slot >= MAX_MMAPS || !g_mmaps[slot].in_use) {
        *r = mk_bool(0);
        return;
    }
#ifdef _WIN32
    UnmapViewOfFile(g_mmaps[slot].data);
    CloseHandle(g_mmaps[slot].hMap);
    CloseHandle(g_mmaps[slot].hFile);
#else
    munmap(g_mmaps[slot].data, g_mmaps[slot].size);
    close(g_mmaps[slot].fd);
#endif
    g_mmaps[slot].in_use = 0;
    g_mmaps[slot].data = NULL;
    g_mmaps[slot].size = 0;
    *r = mk_bool(1);
}

static void sys_mmap_size(TVM *vm, TVal *a, int argc, TVal *r) {
    if (argc != 1 || a[0].t != T_INT) die(vm, "mmap_size(id: int)");
    int slot = (int)a[0].u.i;
    if (slot < 0 || slot >= MAX_MMAPS || !g_mmaps[slot].in_use) die(vm, "mmap_size: invalid mmap id");
    *r = mk_int((int64_t)g_mmaps[slot].size);
}

static void sys_mmap_read_bytes(TVM *vm, TVal *a, int argc, TVal *r) {
    if (argc != 3 || a[0].t != T_INT || a[1].t != T_INT || a[2].t != T_INT)
        die(vm, "mmap_read_bytes(id, offset, len)");
    int slot = (int)a[0].u.i;
    if (slot < 0 || slot >= MAX_MMAPS || !g_mmaps[slot].in_use) die(vm, "mmap: invalid id");
    size_t off = (size_t)a[1].u.i;
    size_t len = (size_t)a[2].u.i;
    if (off + len > g_mmaps[slot].size || off + len < off) die(vm, "mmap: out of bounds");
    *r = mk_str_n(g_mmaps[slot].data + off, len);
}

static void sys_mmap_read_u8(TVM *vm, TVal *a, int argc, TVal *r) {
    if (argc != 2 || a[0].t != T_INT || a[1].t != T_INT) die(vm, "mmap_read_u8(id, off)");
    int slot = (int)a[0].u.i;
    if (slot < 0 || slot >= MAX_MMAPS || !g_mmaps[slot].in_use) die(vm, "mmap: invalid id");
    size_t off = (size_t)a[1].u.i;
    if (off + 1 > g_mmaps[slot].size) die(vm, "mmap: out of bounds");
    *r = mk_int((uint8_t)g_mmaps[slot].data[off]);
}

static void sys_mmap_read_u16(TVM *vm, TVal *a, int argc, TVal *r) {
    if (argc != 2 || a[0].t != T_INT || a[1].t != T_INT) die(vm, "mmap_read_u16(id, off)");
    int slot = (int)a[0].u.i;
    if (slot < 0 || slot >= MAX_MMAPS || !g_mmaps[slot].in_use) die(vm, "mmap: invalid id");
    size_t off = (size_t)a[1].u.i;
    if (off + 2 > g_mmaps[slot].size) die(vm, "mmap: out of bounds");
    const uint8_t *p = (const uint8_t*)g_mmaps[slot].data + off;
    uint16_t v = (uint16_t)p[0] | ((uint16_t)p[1] << 8);
    *r = mk_int(v);
}

static void sys_mmap_read_u32(TVM *vm, TVal *a, int argc, TVal *r) {
    if (argc != 2 || a[0].t != T_INT || a[1].t != T_INT) die(vm, "mmap_read_u32(id, off)");
    int slot = (int)a[0].u.i;
    if (slot < 0 || slot >= MAX_MMAPS || !g_mmaps[slot].in_use) die(vm, "mmap: invalid id");
    size_t off = (size_t)a[1].u.i;
    if (off + 4 > g_mmaps[slot].size) die(vm, "mmap: out of bounds");
    const uint8_t *p = (const uint8_t*)g_mmaps[slot].data + off;
    uint32_t v = (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
    *r = mk_int((int64_t)v);
}

static void sys_mmap_read_u64(TVM *vm, TVal *a, int argc, TVal *r) {
    if (argc != 2 || a[0].t != T_INT || a[1].t != T_INT) die(vm, "mmap_read_u64(id, off)");
    int slot = (int)a[0].u.i;
    if (slot < 0 || slot >= MAX_MMAPS || !g_mmaps[slot].in_use) die(vm, "mmap: invalid id");
    size_t off = (size_t)a[1].u.i;
    if (off + 8 > g_mmaps[slot].size) die(vm, "mmap: out of bounds");
    uint64_t v = 0;
    memcpy(&v, g_mmaps[slot].data + off, 8);
    *r = mk_int((int64_t)v);
}

static void sys_mmap_read_i32(TVM *vm, TVal *a, int argc, TVal *r) {
    if (argc != 2 || a[0].t != T_INT || a[1].t != T_INT) die(vm, "mmap_read_i32(id, off)");
    int slot = (int)a[0].u.i;
    if (slot < 0 || slot >= MAX_MMAPS || !g_mmaps[slot].in_use) die(vm, "mmap: invalid id");
    size_t off = (size_t)a[1].u.i;
    if (off + 4 > g_mmaps[slot].size) die(vm, "mmap: out of bounds");
    int32_t v = 0;
    memcpy(&v, g_mmaps[slot].data + off, 4);
    *r = mk_int((int64_t)v);
}

static void sys_mmap_read_i64(TVM *vm, TVal *a, int argc, TVal *r) {
    if (argc != 2 || a[0].t != T_INT || a[1].t != T_INT) die(vm, "mmap_read_i64(id, off)");
    int slot = (int)a[0].u.i;
    if (slot < 0 || slot >= MAX_MMAPS || !g_mmaps[slot].in_use) die(vm, "mmap: invalid id");
    size_t off = (size_t)a[1].u.i;
    if (off + 8 > g_mmaps[slot].size) die(vm, "mmap: out of bounds");
    int64_t v = 0;
    memcpy(&v, g_mmaps[slot].data + off, 8);
    *r = mk_int(v);
}

static void sys_mmap_read_f32(TVM *vm, TVal *a, int argc, TVal *r) {
    if (argc != 2 || a[0].t != T_INT || a[1].t != T_INT) die(vm, "mmap_read_f32(id, off)");
    int slot = (int)a[0].u.i;
    if (slot < 0 || slot >= MAX_MMAPS || !g_mmaps[slot].in_use) die(vm, "mmap: invalid id");
    size_t off = (size_t)a[1].u.i;
    if (off + 4 > g_mmaps[slot].size) die(vm, "mmap: out of bounds");
    float f = 0.0f;
    memcpy(&f, g_mmaps[slot].data + off, 4);
    *r = mk_float((double)f);
}

static void sys_mmap_read_f64(TVM *vm, TVal *a, int argc, TVal *r) {
    if (argc != 2 || a[0].t != T_INT || a[1].t != T_INT) die(vm, "mmap_read_f64(id, off)");
    int slot = (int)a[0].u.i;
    if (slot < 0 || slot >= MAX_MMAPS || !g_mmaps[slot].in_use) die(vm, "mmap: invalid id");
    size_t off = (size_t)a[1].u.i;
    if (off + 8 > g_mmaps[slot].size) die(vm, "mmap: out of bounds");
    double d = 0.0;
    memcpy(&d, g_mmaps[slot].data + off, 8);
    *r = mk_float(d);
}

/* ========================================================================= */
/* Quantization / Dequantization kernels (FP16, Q4_0, Q8_0)                  */
/* ========================================================================= */

static inline double fp16_to_double(uint16_t h) {
    int sign = (h >> 15) & 1;
    int exp = (h >> 10) & 0x1f;
    int mant = h & 0x3ff;
    if (exp == 0) {
        if (mant == 0) return sign ? -0.0 : 0.0;
        return (sign ? -1.0 : 1.0) * ldexp((double)mant, -24);
    } else if (exp == 31) {
        if (mant == 0) return sign ? -HUGE_VAL : HUGE_VAL;
        return (double)NAN;
    }
    return (sign ? -1.0 : 1.0) * ldexp((double)(mant | 0x400), exp - 25);
}

static inline uint16_t double_to_fp16(double d) {
    union { double f; uint64_t u; } u;
    u.f = d;
    uint32_t sign = (uint32_t)((u.u >> 63) & 1);
    int exp = (int)((u.u >> 52) & 0x7ff);
    uint64_t mant = u.u & 0x000fffffffffffffULL;
    if (exp == 0) {
        return (uint16_t)(sign << 15);
    } else if (exp == 0x7ff) {
        if (mant == 0) return (uint16_t)((sign << 15) | 0x7c00);
        return (uint16_t)((sign << 15) | 0x7e00);
    }
    int new_exp = exp - 1023 + 15;
    if (new_exp >= 31) {
        return (uint16_t)((sign << 15) | 0x7c00);
    } else if (new_exp <= 0) {
        if (new_exp < -10) return (uint16_t)(sign << 15);
        mant = (mant | 0x0010000000000000ULL) >> (1 - new_exp);
        return (uint16_t)((sign << 15) | (mant >> 43));
    }
    return (uint16_t)((sign << 15) | ((new_exp & 0x1f) << 10) | (mant >> 42));
}

static void dequant_f16_buf(const uint16_t *src, double *dst, size_t n) {
    for (size_t i = 0; i < n; i++) dst[i] = fp16_to_double(src[i]);
}

static void dequant_q4_0_buf(const uint8_t *src, double *dst, size_t n) {
    size_t nb = n / 32;
    for (size_t b = 0; b < nb; b++) {
        const uint8_t *block = src + b * 18;
        uint16_t d_raw = (uint16_t)block[0] | ((uint16_t)block[1] << 8);
        double d = fp16_to_double(d_raw);
        const uint8_t *qs = block + 2;
        double *out = dst + b * 32;
        for (int j = 0; j < 16; j++) {
            uint8_t byte = qs[j];
            out[j]      = (double)((int)(byte & 0x0f) - 8) * d;
            out[j + 16] = (double)((int)(byte >> 4)   - 8) * d;
        }
    }
}

static void dequant_q8_0_buf(const uint8_t *src, double *dst, size_t n) {
    size_t nb = n / 32;
    for (size_t b = 0; b < nb; b++) {
        const uint8_t *block = src + b * 34;
        uint16_t d_raw = (uint16_t)block[0] | ((uint16_t)block[1] << 8);
        double d = fp16_to_double(d_raw);
        const int8_t *qs = (const int8_t*)(block + 2);
        double *out = dst + b * 32;
        for (int j = 0; j < 32; j++) out[j] = (double)qs[j] * d;
    }
}

static void quant_f16_buf(const double *src, uint16_t *dst, size_t n) {
    for (size_t i = 0; i < n; i++) dst[i] = double_to_fp16(src[i]);
}

static void quant_q4_0_buf(const double *src, uint8_t *dst, size_t n) {
    size_t nb = n / 32;
    for (size_t b = 0; b < nb; b++) {
        const double *in = src + b * 32;
        uint8_t *block = dst + b * 18;
        double max_v = 0.0;
        for (int j = 0; j < 32; j++) {
            double v = fabs(in[j]);
            if (v > max_v) max_v = v;
        }
        double d = max_v / -8.0;
        if (d == 0.0) d = -1.0;
        double id = 1.0 / d;
        uint16_t d_raw = double_to_fp16(d);
        block[0] = (uint8_t)(d_raw & 0xff);
        block[1] = (uint8_t)((d_raw >> 8) & 0xff);
        uint8_t *qs = block + 2;
        for (int j = 0; j < 16; j++) {
            int q0 = (int)round(in[j] * id);
            int q1 = (int)round(in[j + 16] * id);
            q0 = q0 < -8 ? -8 : (q0 > 7 ? 7 : q0);
            q1 = q1 < -8 ? -8 : (q1 > 7 ? 7 : q1);
            qs[j] = (uint8_t)(((q0 + 8) & 0x0f) | (((q1 + 8) & 0x0f) << 4));
        }
    }
}

static void quant_q8_0_buf(const double *src, uint8_t *dst, size_t n) {
    size_t nb = n / 32;
    for (size_t b = 0; b < nb; b++) {
        const double *in = src + b * 32;
        uint8_t *block = dst + b * 34;
        double max_v = 0.0;
        for (int j = 0; j < 32; j++) {
            double v = fabs(in[j]);
            if (v > max_v) max_v = v;
        }
        double d = max_v / 127.0;
        if (d == 0.0) d = 1.0;
        double id = 1.0 / d;
        uint16_t d_raw = double_to_fp16(d);
        block[0] = (uint8_t)(d_raw & 0xff);
        block[1] = (uint8_t)((d_raw >> 8) & 0xff);
        int8_t *qs = (int8_t*)(block + 2);
        for (int j = 0; j < 32; j++) {
            int q = (int)round(in[j] * id);
            qs[j] = (int8_t)(q < -127 ? -127 : (q > 127 ? 127 : q));
        }
    }
}

static void sys_quant_f16(TVM *vm, TVal *a, int argc, TVal *r) {
    if (argc != 1 || a[0].t != T_TENSOR) die(vm, "quant_f16(tensor)");
    TTensor *t = a[0].u.ten;
    ten_sync(t);
    size_t bytes = t->len * 2;
    char *out = (char*)malloc(bytes ? bytes : 1);
    if (!out) die(vm, "quant_f16: out of memory");
    quant_f16_buf(t->d, (uint16_t*)out, t->len);
    *r = mk_str_n(out, bytes);
    free(out);
}

static void sys_dequant_f16(TVM *vm, TVal *a, int argc, TVal *r) {
    if (argc != 2 || a[0].t != T_STR) die(vm, "dequant_f16(bytes: str, shape)");
    size_t sh[TMAXDIM];
    int nd = ten_shape_arg(vm, &a[1], sh, "dequant_f16");
    TTensor *t = ten_alloc(nd, sh);
    if (a[0].u.str.n < t->len * 2) {
        free(t->d); free(t);
        die(vm, "dequant_f16: buffer size too small");
    }
    dequant_f16_buf((const uint16_t*)a[0].u.str.s, t->d, t->len);
    *r = ten_val(t);
}

static uint16_t double_to_bf16(double x) {
    float f = (float)x;
    uint32_t bits;
    memcpy(&bits, &f, 4);
    /* round-to-nearest-even trên 16 bit cao của IEEE-754 float32 */
    if ((bits & 0x7F800000u) != 0x7F800000u) {
        uint32_t rounding_bias = 0x00007FFFu + ((bits >> 16) & 1u);
        bits += rounding_bias;
    }
    return (uint16_t)(bits >> 16);
}

static double bf16_to_double(uint16_t h) {
    uint32_t bits = ((uint32_t)h) << 16;
    float f;
    memcpy(&f, &bits, 4);
    return (double)f;
}

static void sys_quant_bf16(TVM *vm, TVal *a, int argc, TVal *r) {
    if (argc != 1 || a[0].t != T_TENSOR) die(vm, "quant_bf16(tensor)");
    TTensor *t = a[0].u.ten;
    ten_sync(t);
    size_t bytes = t->len * 2;
    char *out = (char*)malloc(bytes ? bytes : 1);
    if (!out) die(vm, "quant_bf16: out of memory");
    uint16_t *dst = (uint16_t*)out;
    for (size_t i = 0; i < t->len; i++) dst[i] = double_to_bf16(t->d[i]);
    *r = mk_str_n(out, bytes);
    free(out);
}

static void sys_dequant_bf16(TVM *vm, TVal *a, int argc, TVal *r) {
    if (argc != 2 || a[0].t != T_STR) die(vm, "dequant_bf16(bytes: str, shape)");
    size_t sh[TMAXDIM];
    int nd = ten_shape_arg(vm, &a[1], sh, "dequant_bf16");
    TTensor *t = ten_alloc(nd, sh);
    if (a[0].u.str.n < t->len * 2) {
        free(t->d); free(t);
        die(vm, "dequant_bf16: buffer size too small");
    }
    const uint16_t *src = (const uint16_t*)a[0].u.str.s;
    for (size_t i = 0; i < t->len; i++) t->d[i] = bf16_to_double(src[i]);
    *r = ten_val(t);
}

static void sys_quant_f32(TVM *vm, TVal *a, int argc, TVal *r) {
    if (argc != 1 || a[0].t != T_TENSOR) die(vm, "quant_f32(tensor)");
    TTensor *t = a[0].u.ten;
    ten_sync(t);
    size_t bytes = t->len * 4;
    char *out = (char*)malloc(bytes ? bytes : 1);
    if (!out) die(vm, "quant_f32: out of memory");
    float *dst = (float*)out;
    for (size_t i = 0; i < t->len; i++) dst[i] = (float)t->d[i];
    *r = mk_str_n(out, bytes);
    free(out);
}

static void sys_dequant_f32(TVM *vm, TVal *a, int argc, TVal *r) {
    if (argc != 2 || a[0].t != T_STR) die(vm, "dequant_f32(bytes: str, shape)");
    size_t sh[TMAXDIM];
    int nd = ten_shape_arg(vm, &a[1], sh, "dequant_f32");
    TTensor *t = ten_alloc(nd, sh);
    if (a[0].u.str.n < t->len * 4) {
        free(t->d); free(t);
        die(vm, "dequant_f32: buffer size too small");
    }
    const float *src = (const float*)a[0].u.str.s;
    for (size_t i = 0; i < t->len; i++) t->d[i] = (double)src[i];
    *r = ten_val(t);
}

static void sys_quant_q4_0(TVM *vm, TVal *a, int argc, TVal *r) {
    if (argc != 1 || a[0].t != T_TENSOR) die(vm, "quant_q4_0(tensor)");
    TTensor *t = a[0].u.ten;
    ten_sync(t);
    if (t->len % 32 != 0) die(vm, "quant_q4_0: length must be multiple of 32");
    size_t bytes = (t->len / 32) * 18;
    char *out = (char*)malloc(bytes ? bytes : 1);
    if (!out) die(vm, "quant_q4_0: out of memory");
    quant_q4_0_buf(t->d, (uint8_t*)out, t->len);
    *r = mk_str_n(out, bytes);
    free(out);
}

static void sys_dequant_q4_0(TVM *vm, TVal *a, int argc, TVal *r) {
    if (argc != 2 || a[0].t != T_STR) die(vm, "dequant_q4_0(bytes: str, shape)");
    size_t sh[TMAXDIM];
    int nd = ten_shape_arg(vm, &a[1], sh, "dequant_q4_0");
    TTensor *t = ten_alloc(nd, sh);
    if (t->len % 32 != 0) {
        free(t->d); free(t);
        die(vm, "dequant_q4_0: length must be multiple of 32");
    }
    size_t req_bytes = (t->len / 32) * 18;
    if (a[0].u.str.n < req_bytes) {
        free(t->d); free(t);
        die(vm, "dequant_q4_0: buffer size too small");
    }
    dequant_q4_0_buf((const uint8_t*)a[0].u.str.s, t->d, t->len);
    *r = ten_val(t);
}

static void sys_quant_q8_0(TVM *vm, TVal *a, int argc, TVal *r) {
    if (argc != 1 || a[0].t != T_TENSOR) die(vm, "quant_q8_0(tensor)");
    TTensor *t = a[0].u.ten;
    ten_sync(t);
    if (t->len % 32 != 0) die(vm, "quant_q8_0: length must be multiple of 32");
    size_t bytes = (t->len / 32) * 34;
    char *out = (char*)malloc(bytes ? bytes : 1);
    if (!out) die(vm, "quant_q8_0: out of memory");
    quant_q8_0_buf(t->d, (uint8_t*)out, t->len);
    *r = mk_str_n(out, bytes);
    free(out);
}

static void sys_dequant_q8_0(TVM *vm, TVal *a, int argc, TVal *r) {
    if (argc != 2 || a[0].t != T_STR) die(vm, "dequant_q8_0(bytes: str, shape)");
    size_t sh[TMAXDIM];
    int nd = ten_shape_arg(vm, &a[1], sh, "dequant_q8_0");
    TTensor *t = ten_alloc(nd, sh);
    if (t->len % 32 != 0) {
        free(t->d); free(t);
        die(vm, "dequant_q8_0: length must be multiple of 32");
    }
    size_t req_bytes = (t->len / 32) * 34;
    if (a[0].u.str.n < req_bytes) {
        free(t->d); free(t);
        die(vm, "dequant_q8_0: buffer size too small");
    }
    dequant_q8_0_buf((const uint8_t*)a[0].u.str.s, t->d, t->len);
    *r = ten_val(t);
}

static void sys_mmap_tensor_f32(TVM *vm, TVal *a, int argc, TVal *r) {
    if (argc != 3 || a[0].t != T_INT || a[1].t != T_INT) die(vm, "mmap_tensor_f32(id, off, shape)");
    int slot = (int)a[0].u.i;
    if (slot < 0 || slot >= MAX_MMAPS || !g_mmaps[slot].in_use) die(vm, "mmap: invalid id");
    size_t off = (size_t)a[1].u.i;
    size_t sh[TMAXDIM];
    int nd = ten_shape_arg(vm, &a[2], sh, "mmap_tensor_f32");
    TTensor *t = ten_alloc(nd, sh);
    if (off + t->len * 4 > g_mmaps[slot].size || off + t->len * 4 < off) {
        free(t->d); free(t);
        die(vm, "mmap_tensor_f32: out of bounds");
    }
    const float *fsrc = (const float*)(g_mmaps[slot].data + off);
    for (size_t i = 0; i < t->len; i++) t->d[i] = (double)fsrc[i];
    *r = ten_val(t);
}

static void sys_mmap_tensor_f16(TVM *vm, TVal *a, int argc, TVal *r) {
    if (argc != 3 || a[0].t != T_INT || a[1].t != T_INT) die(vm, "mmap_tensor_f16(id, off, shape)");
    int slot = (int)a[0].u.i;
    if (slot < 0 || slot >= MAX_MMAPS || !g_mmaps[slot].in_use) die(vm, "mmap: invalid id");
    size_t off = (size_t)a[1].u.i;
    size_t sh[TMAXDIM];
    int nd = ten_shape_arg(vm, &a[2], sh, "mmap_tensor_f16");
    TTensor *t = ten_alloc(nd, sh);
    if (off + t->len * 2 > g_mmaps[slot].size || off + t->len * 2 < off) {
        free(t->d); free(t);
        die(vm, "mmap_tensor_f16: out of bounds");
    }
    dequant_f16_buf((const uint16_t*)(g_mmaps[slot].data + off), t->d, t->len);
    *r = ten_val(t);
}

static void sys_mmap_tensor_q4_0(TVM *vm, TVal *a, int argc, TVal *r) {
    if (argc != 3 || a[0].t != T_INT || a[1].t != T_INT) die(vm, "mmap_tensor_q4_0(id, off, shape)");
    int slot = (int)a[0].u.i;
    if (slot < 0 || slot >= MAX_MMAPS || !g_mmaps[slot].in_use) die(vm, "mmap: invalid id");
    size_t off = (size_t)a[1].u.i;
    size_t sh[TMAXDIM];
    int nd = ten_shape_arg(vm, &a[2], sh, "mmap_tensor_q4_0");
    TTensor *t = ten_alloc(nd, sh);
    if (t->len % 32 != 0) {
        free(t->d); free(t);
        die(vm, "mmap_tensor_q4_0: length must be multiple of 32");
    }
    size_t req_bytes = (t->len / 32) * 18;
    if (off + req_bytes > g_mmaps[slot].size || off + req_bytes < off) {
        free(t->d); free(t);
        die(vm, "mmap_tensor_q4_0: out of bounds");
    }
    dequant_q4_0_buf((const uint8_t*)(g_mmaps[slot].data + off), t->d, t->len);
    *r = ten_val(t);
}

static void sys_mmap_tensor_q8_0(TVM *vm, TVal *a, int argc, TVal *r) {
    if (argc != 3 || a[0].t != T_INT || a[1].t != T_INT) die(vm, "mmap_tensor_q8_0(id, off, shape)");
    int slot = (int)a[0].u.i;
    if (slot < 0 || slot >= MAX_MMAPS || !g_mmaps[slot].in_use) die(vm, "mmap: invalid id");
    size_t off = (size_t)a[1].u.i;
    size_t sh[TMAXDIM];
    int nd = ten_shape_arg(vm, &a[2], sh, "mmap_tensor_q8_0");
    TTensor *t = ten_alloc(nd, sh);
    if (t->len % 32 != 0) {
        free(t->d); free(t);
        die(vm, "mmap_tensor_q8_0: length must be multiple of 32");
    }
    size_t req_bytes = (t->len / 32) * 34;
    if (off + req_bytes > g_mmaps[slot].size || off + req_bytes < off) {
        free(t->d); free(t);
        die(vm, "mmap_tensor_q8_0: out of bounds");
    }
    dequant_q8_0_buf((const uint8_t*)(g_mmaps[slot].data + off), t->d, t->len);
    *r = ten_val(t);
}

/* ========================================================================= */
/* C FFI (Dynamic Library Loading & Dynamic C Invocation)                    */
/* ========================================================================= */

#define MAX_FFI_LIBS 64
typedef struct {
    int in_use;
#ifdef _WIN32
    HMODULE hMod;
#else
    void *hMod;
#endif
} TFFILib;

static TFFILib g_ffi_libs[MAX_FFI_LIBS];

#ifdef _WIN32
static void sys_ffi_open(TVM *vm, TVal *a, int argc, TVal *r) {
    if (argc != 1 || a[0].t != T_STR) die(vm, "ffi_open(path: str)");
    int slot = -1;
    for (int i = 0; i < MAX_FFI_LIBS; i++) {
        if (!g_ffi_libs[i].in_use) { slot = i; break; }
    }
    if (slot < 0) die(vm, "ffi_open: too many open libraries");
    HMODULE hMod = LoadLibraryA(a[0].u.str.s);
    if (!hMod) {
        *r = mk_int(-1);
        return;
    }
    g_ffi_libs[slot].in_use = 1;
    g_ffi_libs[slot].hMod = hMod;
    *r = mk_int(slot);
}

static void sys_ffi_close(TVM *vm, TVal *a, int argc, TVal *r) {
    if (argc != 1 || a[0].t != T_INT) die(vm, "ffi_close(id: int)");
    int slot = (int)a[0].u.i;
    if (slot < 0 || slot >= MAX_FFI_LIBS || !g_ffi_libs[slot].in_use) {
        *r = mk_bool(0);
        return;
    }
    FreeLibrary(g_ffi_libs[slot].hMod);
    g_ffi_libs[slot].in_use = 0;
    *r = mk_bool(1);
}

static void sys_ffi_sym(TVM *vm, TVal *a, int argc, TVal *r) {
    if (argc != 2 || a[0].t != T_INT || a[1].t != T_STR) die(vm, "ffi_sym(id: int, name: str)");
    int slot = (int)a[0].u.i;
    if (slot < 0 || slot >= MAX_FFI_LIBS || !g_ffi_libs[slot].in_use) die(vm, "ffi_sym: invalid lib id");
    FARPROC sym = GetProcAddress(g_ffi_libs[slot].hMod, a[1].u.str.s);
    *r = mk_int((int64_t)(uintptr_t)sym);
}
#else
static void sys_ffi_open(TVM *vm, TVal *a, int argc, TVal *r) {
    if (argc != 1 || a[0].t != T_STR) die(vm, "ffi_open(path: str)");
    int slot = -1;
    for (int i = 0; i < MAX_FFI_LIBS; i++) {
        if (!g_ffi_libs[i].in_use) { slot = i; break; }
    }
    if (slot < 0) die(vm, "ffi_open: too many open libraries");
    void *hMod = dlopen(a[0].u.str.s, RTLD_LAZY);
    if (!hMod) {
        *r = mk_int(-1);
        return;
    }
    g_ffi_libs[slot].in_use = 1;
    g_ffi_libs[slot].hMod = hMod;
    *r = mk_int(slot);
}

static void sys_ffi_close(TVM *vm, TVal *a, int argc, TVal *r) {
    if (argc != 1 || a[0].t != T_INT) die(vm, "ffi_close(id: int)");
    int slot = (int)a[0].u.i;
    if (slot < 0 || slot >= MAX_FFI_LIBS || !g_ffi_libs[slot].in_use) {
        *r = mk_bool(0);
        return;
    }
    dlclose(g_ffi_libs[slot].hMod);
    g_ffi_libs[slot].in_use = 0;
    *r = mk_bool(1);
}

static void sys_ffi_sym(TVM *vm, TVal *a, int argc, TVal *r) {
    if (argc != 2 || a[0].t != T_INT || a[1].t != T_STR) die(vm, "ffi_sym(id: int, name: str)");
    int slot = (int)a[0].u.i;
    if (slot < 0 || slot >= MAX_FFI_LIBS || !g_ffi_libs[slot].in_use) die(vm, "ffi_sym: invalid lib id");
    void *sym = dlsym(g_ffi_libs[slot].hMod, a[1].u.str.s);
    *r = mk_int((int64_t)(uintptr_t)sym);
}
#endif

static void sys_ffi_call(TVM *vm, TVal *a, int argc, TVal *r) {
    if (argc < 4 || a[0].t != T_INT || a[1].t != T_STR || a[2].t != T_STR || a[3].t != T_LIST)
        die(vm, "ffi_call(sym: int, retty: str, argtypes: str, args: list)");
    uintptr_t fn = (uintptr_t)a[0].u.i;
    if (!fn) die(vm, "ffi_call: null function pointer");
    char retty = a[1].u.str.n > 0 ? a[1].u.str.s[0] : 'v';
    const char *argtypes = a[2].u.str.s;
    size_t ntypes = a[2].u.str.n;
    TList *args = a[3].u.list;
    if (args->len != ntypes) die(vm, "ffi_call: arg count mismatch");
    if (args->len > 20) die(vm, "ffi_call: max 20 arguments supported");

    uint64_t u[20] = {0};
    double d[20] = {0.0};
    for (size_t i = 0; i < args->len; i++) {
        char t = argtypes[i];
        TVal *arg = &args->items[i];
        if (t == 'd' || t == 'f') {
            int ok = 1;
            d[i] = to_num(arg, &ok);
            u[i] = *(uint64_t*)&d[i];
        } else if (t == 's') {
            if (arg->t != T_STR) die(vm, "ffi_call: expected string");
            u[i] = (uintptr_t)arg->u.str.s;
        } else if (t == 'p' || t == 'l' || t == 'i') {
            if (arg->t == T_INT) u[i] = (uint64_t)arg->u.i;
            else if (arg->t == T_BOOL) u[i] = (uint64_t)arg->u.b;
            else if (arg->t == T_STR) u[i] = (uintptr_t)arg->u.str.s;
            else if (arg->t == T_NIL) u[i] = 0;
            else die(vm, "ffi_call: invalid int/ptr arg");
        } else {
            die(vm, "ffi_call: unknown arg type");
        }
    }

    int has_float_arg = 0;
    for (size_t i = 0; i < ntypes; i++) {
        if (argtypes[i] == 'd' || argtypes[i] == 'f') has_float_arg = 1;
    }

    if (retty == 'd' || retty == 'f') {
        double res_d = 0.0;
        if (!has_float_arg) {
            switch (args->len) {
            case 0: res_d = ((double(*)(void))fn)(); break;
            case 1: res_d = ((double(*)(uint64_t))fn)(u[0]); break;
            case 2: res_d = ((double(*)(uint64_t, uint64_t))fn)(u[0], u[1]); break;
            case 3: res_d = ((double(*)(uint64_t, uint64_t, uint64_t))fn)(u[0], u[1], u[2]); break;
            case 4: res_d = ((double(*)(uint64_t, uint64_t, uint64_t, uint64_t))fn)(u[0], u[1], u[2], u[3]); break;
            default: die(vm, "ffi_call: max 4 args for double return with int args");
            }
        } else {
            if (args->len == 1) {
                res_d = ((double(*)(double))fn)(d[0]);
            } else if (args->len == 2) {
                res_d = ((double(*)(double, double))fn)(d[0], d[1]);
            } else if (args->len == 3) {
                res_d = ((double(*)(double, double, double))fn)(d[0], d[1], d[2]);
            } else {
                res_d = ((double(*)(void))fn)();
            }
        }
        *r = mk_float(res_d);
        return;
    }

    uint64_t res = 0;
    if (args->len == 3 && (argtypes[2] == 'd' || argtypes[2] == 'f')) {
        res = ((uint64_t(*)(uint64_t, uint64_t, double))fn)(u[0], u[1], d[2]);
    } else {
        switch (args->len) {
        case 0: res = ((uint64_t(*)(void))fn)(); break;
        case 1: res = ((uint64_t(*)(uint64_t))fn)(u[0]); break;
        case 2: res = ((uint64_t(*)(uint64_t,uint64_t))fn)(u[0], u[1]); break;
        case 3: res = ((uint64_t(*)(uint64_t,uint64_t,uint64_t))fn)(u[0], u[1], u[2]); break;
        case 4: res = ((uint64_t(*)(uint64_t,uint64_t,uint64_t,uint64_t))fn)(u[0], u[1], u[2], u[3]); break;
        case 5: res = ((uint64_t(*)(uint64_t,uint64_t,uint64_t,uint64_t,uint64_t))fn)(u[0], u[1], u[2], u[3], u[4]); break;
        case 6: res = ((uint64_t(*)(uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t))fn)(u[0], u[1], u[2], u[3], u[4], u[5]); break;
        case 7: res = ((uint64_t(*)(uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t))fn)(u[0], u[1], u[2], u[3], u[4], u[5], u[6]); break;
        case 8: res = ((uint64_t(*)(uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t))fn)(u[0], u[1], u[2], u[3], u[4], u[5], u[6], u[7]); break;
        case 9: res = ((uint64_t(*)(uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t))fn)(u[0], u[1], u[2], u[3], u[4], u[5], u[6], u[7], u[8]); break;
        case 10: res = ((uint64_t(*)(uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t))fn)(u[0], u[1], u[2], u[3], u[4], u[5], u[6], u[7], u[8], u[9]); break;
        case 11: res = ((uint64_t(*)(uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t))fn)(u[0], u[1], u[2], u[3], u[4], u[5], u[6], u[7], u[8], u[9], u[10]); break;
        case 12: res = ((uint64_t(*)(uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t))fn)(u[0], u[1], u[2], u[3], u[4], u[5], u[6], u[7], u[8], u[9], u[10], u[11]); break;
        case 13: res = ((uint64_t(*)(uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t))fn)(u[0], u[1], u[2], u[3], u[4], u[5], u[6], u[7], u[8], u[9], u[10], u[11], u[12]); break;
        case 14: res = ((uint64_t(*)(uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t))fn)(u[0], u[1], u[2], u[3], u[4], u[5], u[6], u[7], u[8], u[9], u[10], u[11], u[12], u[13]); break;
        case 15: res = ((uint64_t(*)(uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t))fn)(u[0], u[1], u[2], u[3], u[4], u[5], u[6], u[7], u[8], u[9], u[10], u[11], u[12], u[13], u[14]); break;
        case 16: res = ((uint64_t(*)(uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t))fn)(u[0], u[1], u[2], u[3], u[4], u[5], u[6], u[7], u[8], u[9], u[10], u[11], u[12], u[13], u[14], u[15]); break;
        case 17: res = ((uint64_t(*)(uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t))fn)(u[0], u[1], u[2], u[3], u[4], u[5], u[6], u[7], u[8], u[9], u[10], u[11], u[12], u[13], u[14], u[15], u[16]); break;
        case 18: res = ((uint64_t(*)(uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t))fn)(u[0], u[1], u[2], u[3], u[4], u[5], u[6], u[7], u[8], u[9], u[10], u[11], u[12], u[13], u[14], u[15], u[16], u[17]); break;
        case 19: res = ((uint64_t(*)(uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t))fn)(u[0], u[1], u[2], u[3], u[4], u[5], u[6], u[7], u[8], u[9], u[10], u[11], u[12], u[13], u[14], u[15], u[16], u[17], u[18]); break;
        case 20: res = ((uint64_t(*)(uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t))fn)(u[0], u[1], u[2], u[3], u[4], u[5], u[6], u[7], u[8], u[9], u[10], u[11], u[12], u[13], u[14], u[15], u[16], u[17], u[18], u[19]); break;
        }
    }

    if (retty == 'v') *r = mk_nil();
    else if (retty == 'i') *r = mk_int((int32_t)res);
    else if (retty == 'l') *r = mk_int((int64_t)res);
    else if (retty == 'p') *r = mk_int((int64_t)(uintptr_t)res);
    else if (retty == 's') *r = res ? mk_str((const char*)(uintptr_t)res) : mk_nil();
    else if (retty == 'b') *r = mk_bool(res != 0);
    else *r = mk_int((int64_t)res);
}

/* ========================================================================= */
/* Channels (thread-safe MPMC queue)                                         */
/* ========================================================================= */

#define MAX_CHANNELS 256
typedef struct {
    int in_use;
    int closed;
    size_t cap;
    size_t head;
    size_t tail;
    size_t count;
    TVal *items;
#ifdef _WIN32
    CRITICAL_SECTION cs;
    CONDITION_VARIABLE not_empty;
    CONDITION_VARIABLE not_full;
#else
    pthread_mutex_t mtx;
    pthread_cond_t not_empty;
    pthread_cond_t not_full;
#endif
} TChannel;

static TChannel g_channels[MAX_CHANNELS];

static TVal tval_clone_isolated(const TVal *v) {
    TVal out;
    switch (v->t) {
    case T_INT:
    case T_FLOAT:
    case T_BOOL:
    case T_NIL:
        return *v;
    case T_STR:
        return mk_str_n(v->u.str.s, v->u.str.n);
    case T_TENSOR: {
        TTensor *src = v->u.ten;
        TTensor *t = ten_alloc(src->ndim, src->shape);
        ten_sync(src);
        if (src->len) memcpy(t->d, src->d, sizeof(double) * src->len);
        return ten_val(t);
    }
    case T_LIST: {
        TList *src = v->u.list;
        TList *dst = list_new(src->len);
        for (size_t i = 0; i < src->len; i++) {
            list_push(dst, tval_clone_isolated(&src->items[i]));
        }
        out.t = T_LIST;
        out.u.list = dst;
        return out;
    }
    case T_DICT: {
        TDict *src = v->u.dict;
        TDict *dst = dict_new();
        for (size_t i = 0; i < src->len; i++) {
            TVal k = tval_clone_isolated(&src->keys[i]);
            TVal val = tval_clone_isolated(&src->vals[i]);
            dict_set(dst, k, val);
        }
        out.t = T_DICT;
        out.u.dict = dst;
        return out;
    }
    default:
        return mk_nil();
    }
}

static void sys_channel_new(TVM *vm, TVal *a, int argc, TVal *r) {
    int cap = (argc >= 1 && a[0].t == T_INT && a[0].u.i > 0) ? (int)a[0].u.i : 16;
    (void)vm;
    int slot = -1;
    for (int i = 0; i < MAX_CHANNELS; i++) {
        if (!g_channels[i].in_use) { slot = i; break; }
    }
    if (slot < 0) die(vm, "channel_new: too many channels");
    TChannel *ch = &g_channels[slot];
    ch->in_use = 1;
    ch->closed = 0;
    ch->cap = (size_t)cap;
    ch->head = 0;
    ch->tail = 0;
    ch->count = 0;
    ch->items = (TVal*)malloc(sizeof(TVal) * ch->cap);
#ifdef _WIN32
    InitializeCriticalSection(&ch->cs);
    InitializeConditionVariable(&ch->not_empty);
    InitializeConditionVariable(&ch->not_full);
#else
    pthread_mutex_init(&ch->mtx, NULL);
    pthread_cond_init(&ch->not_empty, NULL);
    pthread_cond_init(&ch->not_full, NULL);
#endif
    *r = mk_int(slot);
}

static void sys_channel_send(TVM *vm, TVal *a, int argc, TVal *r) {
    if (argc != 2 || a[0].t != T_INT) die(vm, "channel_send(id: int, val)");
    int slot = (int)a[0].u.i;
    if (slot < 0 || slot >= MAX_CHANNELS || !g_channels[slot].in_use) die(vm, "channel_send: invalid channel");
    TChannel *ch = &g_channels[slot];

#ifdef _WIN32
    EnterCriticalSection(&ch->cs);
    while (ch->count == ch->cap && !ch->closed) {
        SleepConditionVariableCS(&ch->not_full, &ch->cs, INFINITE);
    }
    if (ch->closed) {
        LeaveCriticalSection(&ch->cs);
        *r = mk_bool(0);
        return;
    }
    ch->items[ch->tail] = tval_clone_isolated(&a[1]);
    ch->tail = (ch->tail + 1) % ch->cap;
    ch->count++;
    WakeConditionVariable(&ch->not_empty);
    LeaveCriticalSection(&ch->cs);
#else
    pthread_mutex_lock(&ch->mtx);
    while (ch->count == ch->cap && !ch->closed) {
        pthread_cond_wait(&ch->not_full, &ch->mtx);
    }
    if (ch->closed) {
        pthread_mutex_unlock(&ch->mtx);
        *r = mk_bool(0);
        return;
    }
    ch->items[ch->tail] = tval_clone_isolated(&a[1]);
    ch->tail = (ch->tail + 1) % ch->cap;
    ch->count++;
    pthread_cond_signal(&ch->not_empty);
    pthread_mutex_unlock(&ch->mtx);
#endif
    *r = mk_bool(1);
}

static void sys_channel_recv(TVM *vm, TVal *a, int argc, TVal *r) {
    if (argc != 1 || a[0].t != T_INT) die(vm, "channel_recv(id: int)");
    int slot = (int)a[0].u.i;
    if (slot < 0 || slot >= MAX_CHANNELS || !g_channels[slot].in_use) die(vm, "channel_recv: invalid channel");
    TChannel *ch = &g_channels[slot];

#ifdef _WIN32
    EnterCriticalSection(&ch->cs);
    while (ch->count == 0 && !ch->closed) {
        SleepConditionVariableCS(&ch->not_empty, &ch->cs, INFINITE);
    }
    if (ch->count == 0 && ch->closed) {
        LeaveCriticalSection(&ch->cs);
        *r = mk_nil();
        return;
    }
    TVal val = ch->items[ch->head];
    ch->head = (ch->head + 1) % ch->cap;
    ch->count--;
    WakeConditionVariable(&ch->not_full);
    LeaveCriticalSection(&ch->cs);
#else
    pthread_mutex_lock(&ch->mtx);
    while (ch->count == 0 && !ch->closed) {
        pthread_cond_wait(&ch->not_empty, &ch->mtx);
    }
    if (ch->count == 0 && ch->closed) {
        pthread_mutex_unlock(&ch->mtx);
        *r = mk_nil();
        return;
    }
    TVal val = ch->items[ch->head];
    ch->head = (ch->head + 1) % ch->cap;
    ch->count--;
    pthread_cond_signal(&ch->not_full);
    pthread_mutex_unlock(&ch->mtx);
#endif
    *r = val;
}

static void sys_channel_close(TVM *vm, TVal *a, int argc, TVal *r) {
    if (argc != 1 || a[0].t != T_INT) die(vm, "channel_close(id: int)");
    int slot = (int)a[0].u.i;
    if (slot < 0 || slot >= MAX_CHANNELS || !g_channels[slot].in_use) {
        *r = mk_bool(0);
        return;
    }
    TChannel *ch = &g_channels[slot];
#ifdef _WIN32
    EnterCriticalSection(&ch->cs);
    ch->closed = 1;
    WakeAllConditionVariable(&ch->not_empty);
    WakeAllConditionVariable(&ch->not_full);
    LeaveCriticalSection(&ch->cs);
#else
    pthread_mutex_lock(&ch->mtx);
    ch->closed = 1;
    pthread_cond_broadcast(&ch->not_empty);
    pthread_cond_broadcast(&ch->not_full);
    pthread_mutex_unlock(&ch->mtx);
#endif
    *r = mk_bool(1);
}

static void sys_channel_len(TVM *vm, TVal *a, int argc, TVal *r) {
    if (argc != 1 || a[0].t != T_INT) die(vm, "channel_len(id: int)");
    int slot = (int)a[0].u.i;
    if (slot < 0 || slot >= MAX_CHANNELS || !g_channels[slot].in_use) die(vm, "channel_len: invalid channel");
    TChannel *ch = &g_channels[slot];
    size_t count;
#ifdef _WIN32
    EnterCriticalSection(&ch->cs);
    count = ch->count;
    LeaveCriticalSection(&ch->cs);
#else
    pthread_mutex_lock(&ch->mtx);
    count = ch->count;
    pthread_mutex_unlock(&ch->mtx);
#endif
    *r = mk_int((int64_t)count);
}

/* ========================================================================= */
/* Worker Threads                                                            */
/* ========================================================================= */

#define MAX_THREADS 128
typedef struct {
    int in_use;
    char *script_path;
    int argc;
    char **argv;
#ifdef _WIN32
    HANDLE hThread;
#else
    pthread_t thread;
#endif
} TThreadSlot;

static TThreadSlot g_threads[MAX_THREADS];

static int is_tb_file_tvm(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) return 0;
    unsigned char hdr[4];
    size_t n = fread(hdr, 1, 4, f);
    fclose(f);
    return n == 4 && memcmp(hdr, "TB01", 4) == 0;
}

static void make_tmp_path_tvm(char *out, size_t n, const char *ext) {
    static unsigned int seq = 0;
    seq++;
#ifdef _WIN32
    char dir[512];
    DWORD dlen = GetTempPathA((DWORD)sizeof(dir), dir);
    if (dlen == 0 || dlen >= sizeof(dir)) snprintf(dir, sizeof(dir), ".");
    snprintf(out, n, "%strun_th_%lu_%u%s", dir, (unsigned long)GetCurrentProcessId(), seq, ext);
#else
    const char *dir = getenv("TMPDIR");
    if (!dir || !*dir) dir = "/tmp";
    snprintf(out, n, "%s/trun_th_%lu_%u%s", dir, (unsigned long)getpid(), seq, ext);
#endif
}

#ifdef _WIN32
static DWORD WINAPI worker_thread_proc(LPVOID lpParam) {
#else
static void* worker_thread_proc(void *lpParam) {
#endif
    TThreadSlot *slot = (TThreadSlot*)lpParam;
    TProg p; memset(&p, 0, sizeof(p));
    char msg[256];
    char tmp_tb[1024];
    int used_tmp = 0;
    const char *load_path = slot->script_path;

    if (!is_tb_file_tvm(slot->script_path)) {
        make_tmp_path_tvm(tmp_tb, sizeof(tmp_tb), ".tb");
        if (tvm_compile_src(slot->script_path, tmp_tb, 1, msg, sizeof(msg)) < 0) {
            remove(tmp_tb);
            fprintf(stderr, "thread compile error: %s\n", msg);
            return 0;
        }
        load_path = tmp_tb;
        used_tmp = 1;
    }

    if (tvm_read_tb(load_path, &p, msg, sizeof(msg)) != 0) {
        if (used_tmp) remove(tmp_tb);
        fprintf(stderr, "thread load error: %s\n", msg);
        return 0;
    }
    if (used_tmp) {
        remove(tmp_tb);
        free(p.path);
        size_t plen = strlen(slot->script_path);
        p.path = (char*)malloc(plen + 1);
        if (p.path) memcpy(p.path, slot->script_path, plen + 1);
    }
    p.xargv = slot->argv;
    p.xargc = slot->argc;

    TVM vm;
    tvm_init(&vm, &p);
    tvm_run(&vm);
    tvm_free(&vm);
    tvm_prog_release(&p);
    return 0;
}

static void sys_thread_spawn(TVM *vm, TVal *a, int argc, TVal *r) {
    if (argc < 1 || a[0].t != T_STR) die(vm, "thread_spawn(script_path: str[, args: list])");
    int slot_id = -1;
    for (int i = 0; i < MAX_THREADS; i++) {
        if (!g_threads[i].in_use) { slot_id = i; break; }
    }
    if (slot_id < 0) die(vm, "thread_spawn: too many threads");
    TThreadSlot *slot = &g_threads[slot_id];
    slot->in_use = 1;
    slot->script_path = (char*)malloc(a[0].u.str.n + 1);
    memcpy(slot->script_path, a[0].u.str.s, a[0].u.str.n + 1);

    int xargc = 1;
    if (argc >= 2 && a[1].t == T_LIST) xargc += (int)a[1].u.list->len;
    slot->argc = xargc;
    slot->argv = (char**)malloc(sizeof(char*) * (xargc + 1));
    slot->argv[0] = (char*)malloc(a[0].u.str.n + 1);
    memcpy(slot->argv[0], a[0].u.str.s, a[0].u.str.n + 1);
    if (argc >= 2 && a[1].t == T_LIST) {
        for (int i = 0; i < (int)a[1].u.list->len; i++) {
            char buf[256];
            tval_fmt(&a[1].u.list->items[i], buf, sizeof(buf));
            size_t blen = strlen(buf);
            slot->argv[i + 1] = (char*)malloc(blen + 1);
            memcpy(slot->argv[i + 1], buf, blen + 1);
        }
    }
    slot->argv[xargc] = NULL;

#ifdef _WIN32
    slot->hThread = CreateThread(NULL, 0, worker_thread_proc, slot, 0, NULL);
    if (!slot->hThread) {
        slot->in_use = 0;
        die(vm, "thread_spawn: CreateThread failed");
    }
#else
    if (pthread_create(&slot->thread, NULL, worker_thread_proc, slot) != 0) {
        slot->in_use = 0;
        die(vm, "thread_spawn: pthread_create failed");
    }
#endif
    *r = mk_int(slot_id);
}

static void sys_thread_join(TVM *vm, TVal *a, int argc, TVal *r) {
    if (argc != 1 || a[0].t != T_INT) die(vm, "thread_join(id: int)");
    int slot_id = (int)a[0].u.i;
    if (slot_id < 0 || slot_id >= MAX_THREADS || !g_threads[slot_id].in_use) {
        *r = mk_bool(0);
        return;
    }
    TThreadSlot *slot = &g_threads[slot_id];
#ifdef _WIN32
    WaitForSingleObject(slot->hThread, INFINITE);
    CloseHandle(slot->hThread);
#else
    pthread_join(slot->thread, NULL);
#endif
    free(slot->script_path);
    for (int i = 0; i < slot->argc; i++) free(slot->argv[i]);
    free(slot->argv);
    slot->in_use = 0;
    *r = mk_bool(1);
}

/* ========================================================================= */
/* Raw Memory Primitives (Buffer allocation & pointer dereference for FFI)  */
/* ========================================================================= */

static void sys_mem_alloc(TVM *vm, TVal *a, int argc, TVal *r) {
    if (argc != 1 || a[0].t != T_INT || a[0].u.i <= 0) die(vm, "mem_alloc(size: int > 0)");
    size_t sz = (size_t)a[0].u.i;
    void *ptr = calloc(1, sz);
    *r = mk_int((int64_t)(uintptr_t)ptr);
}

static void sys_mem_free(TVM *vm, TVal *a, int argc, TVal *r) {
    if (argc != 1 || a[0].t != T_INT) die(vm, "mem_free(ptr: int)");
    void *ptr = (void*)(uintptr_t)a[0].u.i;
    if (ptr) free(ptr);
    *r = mk_nil();
}

static void sys_mem_read_ptr(TVM *vm, TVal *a, int argc, TVal *r) {
    if (argc != 1 || a[0].t != T_INT) die(vm, "mem_read_ptr(ptr: int)");
    uintptr_t p = (uintptr_t)a[0].u.i;
    if (!p) die(vm, "mem_read_ptr: null pointer dereference");
    uintptr_t val = *(uintptr_t*)p;
    *r = mk_int((int64_t)val);
}

static void sys_mem_write_ptr(TVM *vm, TVal *a, int argc, TVal *r) {
    if (argc != 2 || a[0].t != T_INT || a[1].t != T_INT) die(vm, "mem_write_ptr(ptr: int, val: int)");
    uintptr_t p = (uintptr_t)a[0].u.i;
    if (!p) die(vm, "mem_write_ptr: null pointer dereference");
    *(uintptr_t*)p = (uintptr_t)a[1].u.i;
    *r = mk_nil();
}

static void sys_mem_read_i32(TVM *vm, TVal *a, int argc, TVal *r) {
    if (argc != 1 || a[0].t != T_INT) die(vm, "mem_read_i32(ptr: int)");
    uintptr_t p = (uintptr_t)a[0].u.i;
    if (!p) die(vm, "mem_read_i32: null pointer dereference");
    int32_t val = *(int32_t*)p;
    *r = mk_int((int64_t)val);
}

static void sys_mem_write_i32(TVM *vm, TVal *a, int argc, TVal *r) {
    if (argc != 2 || a[0].t != T_INT || a[1].t != T_INT) die(vm, "mem_write_i32(ptr: int, val: int)");
    uintptr_t p = (uintptr_t)a[0].u.i;
    if (!p) die(vm, "mem_write_i32: null pointer dereference");
    *(int32_t*)p = (int32_t)a[1].u.i;
    *r = mk_nil();
}

static void sys_mem_read_str(TVM *vm, TVal *a, int argc, TVal *r) {
    if (argc != 1 || a[0].t != T_INT) die(vm, "mem_read_str(ptr: int)");
    const char *p = (const char*)(uintptr_t)a[0].u.i;
    if (!p) {
        *r = mk_nil();
    } else {
        *r = mk_str(p);
    }
}

static void sys_mem_read_f64(TVM *vm, TVal *a, int argc, TVal *r) {
    if (argc != 1 || a[0].t != T_INT) die(vm, "mem_read_f64(ptr: int)");
    uintptr_t p = (uintptr_t)a[0].u.i;
    if (!p) die(vm, "mem_read_f64: null pointer dereference");
    double val = *(double*)p;
    *r = mk_float(val);
}

static void sys_mem_write_f64(TVM *vm, TVal *a, int argc, TVal *r) {
    if (argc != 2 || a[0].t != T_INT) die(vm, "mem_write_f64(ptr: int, val: float)");
    uintptr_t p = (uintptr_t)a[0].u.i;
    if (!p) die(vm, "mem_write_f64: null pointer dereference");
    int ok = 1;
    double val = to_num(&a[1], &ok);
    if (!ok) die(vm, "mem_write_f64: expected number");
    *(double*)p = val;
    *r = mk_nil();
}

static void sys_mem_read_f32(TVM *vm, TVal *a, int argc, TVal *r) {
    if (argc != 1 || a[0].t != T_INT) die(vm, "mem_read_f32(ptr: int)");
    uintptr_t p = (uintptr_t)a[0].u.i;
    if (!p) die(vm, "mem_read_f32: null pointer dereference");
    float val = *(float*)p;
    *r = mk_float((double)val);
}

static void sys_mem_write_f32(TVM *vm, TVal *a, int argc, TVal *r) {
    if (argc != 2 || a[0].t != T_INT) die(vm, "mem_write_f32(ptr: int, val: float)");
    uintptr_t p = (uintptr_t)a[0].u.i;
    if (!p) die(vm, "mem_write_f32: null pointer dereference");
    int ok = 1;
    double val = to_num(&a[1], &ok);
    if (!ok) die(vm, "mem_write_f32: expected number");
    *(float*)p = (float)val;
    *r = mk_nil();
}

static void sys_mem_read_u16(TVM *vm, TVal *a, int argc, TVal *r) {
    if (argc != 1 || a[0].t != T_INT) die(vm, "mem_read_u16(ptr: int)");
    uintptr_t p = (uintptr_t)a[0].u.i;
    if (!p) die(vm, "mem_read_u16: null pointer dereference");
    uint16_t val = *(uint16_t*)p;
    *r = mk_int((int64_t)val);
}

static void sys_mem_write_u16(TVM *vm, TVal *a, int argc, TVal *r) {
    if (argc != 2 || a[0].t != T_INT || a[1].t != T_INT) die(vm, "mem_write_u16(ptr: int, val: int)");
    uintptr_t p = (uintptr_t)a[0].u.i;
    if (!p) die(vm, "mem_write_u16: null pointer dereference");
    *(uint16_t*)p = (uint16_t)a[1].u.i;
    *r = mk_nil();
}

static void sys_arm64_selftest(TVM *vm, TVal *a, int argc, TVal *r) {
    (void)vm; (void)a; (void)argc;
    *r = mk_bool(t_arm64_selftest());
}

static void sys_arm64_dump(TVM *vm, TVal *a, int argc, TVal *r) {
    if (argc != 1 || a[0].t != T_STR) die(vm, "arm64_dump(path: str)");
    *r = mk_bool(t_arm64_dump(a[0].u.str.s));
}

/* ========================================================================= */
/* Native TCP/UDP Sockets & Single-Binary Bundler (v6.0 N4 & T5)             */
/* ========================================================================= */

#ifdef _WIN32
typedef SOCKET t_sock_t;
#define T_BAD_SOCK INVALID_SOCKET
static void sock_init_once(void) {
    static int inited = 0;
    if (!inited) {
        WSADATA wsa;
        WSAStartup(MAKEWORD(2, 2), &wsa);
        inited = 1;
    }
}
static void sock_close_raw(t_sock_t s) { closesocket(s); }
#else
typedef int t_sock_t;
#define T_BAD_SOCK (-1)
static void sock_init_once(void) {}
static void sock_close_raw(t_sock_t s) { close(s); }
#endif

static int sock_parse_addr(const char *host, int port, struct sockaddr_in *sa) {
    memset(sa, 0, sizeof(*sa));
    sa->sin_family = AF_INET;
    sa->sin_port = htons((uint16_t)port);
    if (!host || !*host || !strcmp(host, "0.0.0.0") || !strcmp(host, "*")) {
        sa->sin_addr.s_addr = htonl(INADDR_ANY);
        return 1;
    }
    if (!strcmp(host, "localhost")) host = "127.0.0.1";
    if (inet_pton(AF_INET, host, &sa->sin_addr) == 1) return 1;
    struct hostent *he = gethostbyname(host);
    if (he && he->h_addr_list && he->h_addr_list[0]) {
        memcpy(&sa->sin_addr, he->h_addr_list[0], sizeof(struct in_addr));
        return 1;
    }
    return 0;
}

static void sys_sock_listen(TVM *vm, TVal *a, int argc, TVal *r) {
    if (argc < 2 || a[0].t != T_STR || a[1].t != T_INT) die(vm, "sock_listen(host: str, port: int[, backlog: int])");
    sock_init_once();
    int backlog = (argc >= 3 && a[2].t == T_INT) ? (int)a[2].u.i : 16;
    t_sock_t s = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    if (s == T_BAD_SOCK) die(vm, "sock_listen: socket() failed");
    int opt = 1;
#ifdef _WIN32
    setsockopt(s, SOL_SOCKET, SO_REUSEADDR, (const char*)&opt, sizeof(opt));
#else
    setsockopt(s, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));
#endif
    struct sockaddr_in sa;
    if (!sock_parse_addr(a[0].u.str.s, (int)a[1].u.i, &sa)) {
        sock_close_raw(s);
        die(vm, "sock_listen: invalid address");
    }
    if (bind(s, (struct sockaddr*)&sa, sizeof(sa)) != 0) {
        sock_close_raw(s);
        die(vm, "sock_listen: bind() failed");
    }
    if (listen(s, backlog) != 0) {
        sock_close_raw(s);
        die(vm, "sock_listen: listen() failed");
    }
    *r = mk_int((int64_t)s);
}

static void sys_sock_port(TVM *vm, TVal *a, int argc, TVal *r) {
    if (argc != 1 || a[0].t != T_INT) die(vm, "sock_port(fd: int)");
    t_sock_t s = (t_sock_t)a[0].u.i;
    struct sockaddr_in sa;
#ifdef _WIN32
    int len = sizeof(sa);
#else
    socklen_t len = sizeof(sa);
#endif
    if (getsockname(s, (struct sockaddr*)&sa, &len) != 0) die(vm, "sock_port: getsockname() failed");
    *r = mk_int((int64_t)ntohs(sa.sin_port));
}

static void sys_sock_accept(TVM *vm, TVal *a, int argc, TVal *r) {
    if (argc != 1 || a[0].t != T_INT) die(vm, "sock_accept(server_fd: int)");
    t_sock_t s = (t_sock_t)a[0].u.i;
    struct sockaddr_in client_sa;
#ifdef _WIN32
    int len = sizeof(client_sa);
#else
    socklen_t len = sizeof(client_sa);
#endif
    t_sock_t c = accept(s, (struct sockaddr*)&client_sa, &len);
    if (c == T_BAD_SOCK) die(vm, "sock_accept: accept() failed");
    *r = mk_int((int64_t)c);
}

static void sys_sock_connect(TVM *vm, TVal *a, int argc, TVal *r) {
    if (argc != 2 || a[0].t != T_STR || a[1].t != T_INT) die(vm, "sock_connect(host: str, port: int)");
    sock_init_once();
    t_sock_t s = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    if (s == T_BAD_SOCK) die(vm, "sock_connect: socket() failed");
    struct sockaddr_in sa;
    if (!sock_parse_addr(a[0].u.str.s, (int)a[1].u.i, &sa)) {
        sock_close_raw(s);
        die(vm, "sock_connect: invalid address");
    }
    if (connect(s, (struct sockaddr*)&sa, sizeof(sa)) != 0) {
        sock_close_raw(s);
        die(vm, "sock_connect: connect() failed");
    }
    *r = mk_int((int64_t)s);
}

static void sys_sock_send(TVM *vm, TVal *a, int argc, TVal *r) {
    if (argc != 2 || a[0].t != T_INT || a[1].t != T_STR) die(vm, "sock_send(fd: int, data: str)");
    t_sock_t s = (t_sock_t)a[0].u.i;
    const char *p = a[1].u.str.s;
    size_t rem = a[1].u.str.n, sent_tot = 0;
    while (rem > 0) {
        int n = send(s, p + sent_tot, (int)rem, 0);
        if (n <= 0) break;
        sent_tot += (size_t)n;
        rem -= (size_t)n;
    }
    *r = mk_int((int64_t)sent_tot);
}

static void sys_sock_recv(TVM *vm, TVal *a, int argc, TVal *r) {
    if (argc < 1 || a[0].t != T_INT) die(vm, "sock_recv(fd: int[, max_bytes: int])");
    t_sock_t s = (t_sock_t)a[0].u.i;
    size_t max_b = (argc >= 2 && a[1].t == T_INT && a[1].u.i > 0) ? (size_t)a[1].u.i : 4096;
    char *buf = (char*)malloc(max_b + 1);
    if (!buf) die(vm, "sock_recv: out of memory");
    int n = recv(s, buf, (int)max_b, 0);
    if (n <= 0) {
        free(buf);
        *r = mk_str_n("", 0);
        return;
    }
    *r = mk_str_n(buf, (size_t)n);
    free(buf);
}

static void sys_sock_close(TVM *vm, TVal *a, int argc, TVal *r) {
    if (argc != 1 || a[0].t != T_INT) die(vm, "sock_close(fd: int)");
    sock_close_raw((t_sock_t)a[0].u.i);
    *r = mk_bool(1);
}

static void sys_sock_udp_open(TVM *vm, TVal *a, int argc, TVal *r) {
    sock_init_once();
    t_sock_t s = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
    if (s == T_BAD_SOCK) die(vm, "sock_udp_open: socket() failed");
    if (argc >= 2 && a[0].t == T_STR && a[1].t == T_INT) {
        int opt = 1;
#ifdef _WIN32
        setsockopt(s, SOL_SOCKET, SO_REUSEADDR, (const char*)&opt, sizeof(opt));
#else
        setsockopt(s, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));
#endif
        struct sockaddr_in sa;
        if (!sock_parse_addr(a[0].u.str.s, (int)a[1].u.i, &sa) ||
            bind(s, (struct sockaddr*)&sa, sizeof(sa)) != 0) {
            sock_close_raw(s);
            die(vm, "sock_udp_open: bind() failed");
        }
    }
    *r = mk_int((int64_t)s);
}

static void sys_sock_udp_sendto(TVM *vm, TVal *a, int argc, TVal *r) {
    if (argc != 4 || a[0].t != T_INT || a[1].t != T_STR || a[2].t != T_STR || a[3].t != T_INT)
        die(vm, "sock_udp_sendto(fd: int, data: str, host: str, port: int)");
    t_sock_t s = (t_sock_t)a[0].u.i;
    struct sockaddr_in sa;
    if (!sock_parse_addr(a[2].u.str.s, (int)a[3].u.i, &sa))
        die(vm, "sock_udp_sendto: invalid address");
    int n = sendto(s, a[1].u.str.s, (int)a[1].u.str.n, 0, (struct sockaddr*)&sa, sizeof(sa));
    *r = mk_int(n >= 0 ? n : 0);
}

static void sys_sock_udp_recvfrom(TVM *vm, TVal *a, int argc, TVal *r) {
    if (argc < 1 || a[0].t != T_INT) die(vm, "sock_udp_recvfrom(fd: int[, max_bytes: int])");
    t_sock_t s = (t_sock_t)a[0].u.i;
    size_t max_b = (argc >= 2 && a[1].t == T_INT && a[1].u.i > 0) ? (size_t)a[1].u.i : 4096;
    char *buf = (char*)malloc(max_b + 1);
    struct sockaddr_in sa;
#ifdef _WIN32
    int salen = sizeof(sa);
#else
    socklen_t salen = sizeof(sa);
#endif
    int n = recvfrom(s, buf, (int)max_b, 0, (struct sockaddr*)&sa, &salen);
    TDict *d = dict_new();
    if (n > 0) {
        char ip[64];
        const char *ips = inet_ntop(AF_INET, &sa.sin_addr, ip, sizeof(ip));
        dict_set(d, mk_str("data"), mk_str_n(buf, (size_t)n));
        dict_set(d, mk_str("host"), mk_str(ips ? ips : "127.0.0.1"));
        dict_set(d, mk_str("port"), mk_int((int64_t)ntohs(sa.sin_port)));
    } else {
        dict_set(d, mk_str("data"), mk_str_n("", 0));
        dict_set(d, mk_str("host"), mk_str(""));
        dict_set(d, mk_str("port"), mk_int(0));
    }
    free(buf);
    r->t = T_DICT; r->u.dict = d;
}

int tvm_build_exe(const char *src_or_tb, const char *out_exe, char *msg, size_t msgn) {
    char tmp_tb[1024];
    const char *tb_path = src_or_tb;
    int used_tmp = 0;
    if (msgn) msg[0] = 0;
    if (!tvm_exe_path) {
        if (msgn) snprintf(msg, msgn, "không xác định được đường dẫn trun");
        return -1;
    }
    FILE *chk = fopen(src_or_tb, "rb");
    if (!chk) {
        if (msgn) snprintf(msg, msgn, "không mở được %s", src_or_tb);
        return -1;
    }
    unsigned char hdr[4];
    size_t hn = fread(hdr, 1, 4, chk);
    fclose(chk);
    if (hn != 4 || memcmp(hdr, "TB01", 4) != 0) {
        static int bseq = 0;
#ifdef _WIN32
        char dir[512];
        DWORD dlen = GetTempPathA((DWORD)sizeof(dir), dir);
        if (dlen == 0 || dlen >= sizeof(dir)) snprintf(dir, sizeof(dir), ".");
        snprintf(tmp_tb, sizeof(tmp_tb), "%strun_bexe_%lu_%d.tb", dir, (unsigned long)GetCurrentProcessId(), ++bseq);
#else
        snprintf(tmp_tb, sizeof(tmp_tb), "/tmp/trun_bexe_%lu_%d.tb", (unsigned long)getpid(), ++bseq);
#endif
        if (tvm_compile_src(src_or_tb, tmp_tb, 1, msg, msgn) < 0) {
            remove(tmp_tb);
            return -1;
        }
        tb_path = tmp_tb;
        used_tmp = 1;
    }
    FILE *frun = fopen(tvm_exe_path, "rb");
    if (!frun) {
        if (used_tmp) remove(tmp_tb);
        if (msgn) snprintf(msg, msgn, "không đọc được runner %s", tvm_exe_path);
        return -1;
    }
    fseek(frun, 0, SEEK_END);
    long run_sz = ftell(frun);
    /* Nếu bản thân tvm_exe_path đã là bundled exe, chỉ lấy phần runner gốc */
    if (run_sz > 16) {
        char tail[16];
        fseek(frun, run_sz - 16, SEEK_SET);
        if (fread(tail, 1, 16, frun) == 16 && memcmp(tail + 8, "TBEXE001", 8) == 0) {
            uint64_t emb_sz = 0;
            memcpy(&emb_sz, tail, 8);
            if (emb_sz > 0 && (long)(emb_sz + 16) < run_sz) {
                run_sz -= (long)(emb_sz + 16);
            }
        }
    }
    fseek(frun, 0, SEEK_SET);
    uint8_t *run_buf = (uint8_t*)malloc((size_t)run_sz);
    if (!run_buf || fread(run_buf, 1, (size_t)run_sz, frun) != (size_t)run_sz) {
        fclose(frun); free(run_buf);
        if (used_tmp) remove(tmp_tb);
        if (msgn) snprintf(msg, msgn, "lỗi đọc runner");
        return -1;
    }
    fclose(frun);

    FILE *ftb = fopen(tb_path, "rb");
    if (!ftb) {
        free(run_buf);
        if (used_tmp) remove(tmp_tb);
        if (msgn) snprintf(msg, msgn, "không đọc được %s", tb_path);
        return -1;
    }
    fseek(ftb, 0, SEEK_END);
    long tb_sz = ftell(ftb);
    fseek(ftb, 0, SEEK_SET);
    uint8_t *tb_buf = (uint8_t*)malloc((size_t)(tb_sz > 0 ? tb_sz : 1));
    if (!tb_buf || (tb_sz > 0 && fread(tb_buf, 1, (size_t)tb_sz, ftb) != (size_t)tb_sz)) {
        fclose(ftb); free(run_buf); free(tb_buf);
        if (used_tmp) remove(tmp_tb);
        if (msgn) snprintf(msg, msgn, "lỗi đọc bytecode");
        return -1;
    }
    fclose(ftb);
    if (used_tmp) remove(tmp_tb);

    FILE *fout = fopen(out_exe, "wb");
    if (!fout) {
        free(run_buf); free(tb_buf);
        if (msgn) snprintf(msg, msgn, "không tạo được %s", out_exe);
        return -1;
    }
    uint64_t sz64 = (uint64_t)tb_sz;
    int ok = (fwrite(run_buf, 1, (size_t)run_sz, fout) == (size_t)run_sz) &&
             (fwrite(tb_buf, 1, (size_t)tb_sz, fout) == (size_t)tb_sz) &&
             (fwrite(&sz64, 1, 8, fout) == 8) &&
             (fwrite("TBEXE001", 1, 8, fout) == 8);
    fclose(fout);
    free(run_buf);
    free(tb_buf);
#ifndef _WIN32
    chmod(out_exe, 0755);
#endif
    if (!ok) {
        if (msgn) snprintf(msg, msgn, "lỗi ghi %s", out_exe);
        return -1;
    }
    return 0;
}

static void sys_build_exe(TVM *vm, TVal *a, int argc, TVal *r) {
    if (argc != 2 || a[0].t != T_STR || a[1].t != T_STR)
        die(vm, "build_exe(src_or_tb: str, out_exe: str)");
    char msg[256];
    if (tvm_build_exe(a[0].u.str.s, a[1].u.str.s, msg, sizeof(msg)) != 0)
        die(vm, msg[0] ? msg : "build_exe failed");
    *r = mk_bool(1);
}

/* ========================================================================= */
/* v7.0 Rust-Grade Optimizations: Instant JIT, Lock-Free Atomics,            */
/* OS Mutex, Zero-Copy SharedTensor & #[repr(C)] Struct SIMD Kernels         */
/* ========================================================================= */

static void sys_jit_compile(TVM *vm, TVal *a, int argc, TVal *r) {
    if (argc < 1 || a[0].t != T_FN) die(vm, "jit_compile(fn[, sample_args_list])");
    TFn *f = a[0].u.fn;
    f->ncalls = JIT_THRESHOLD;
    f->loops = JIT_OSR_LOOPS;
    f->jit_off = 0;
    if (argc >= 2 && a[1].t == T_LIST && a[1].u.list->len == f->nargs) {
        t_jit_maybe(vm, f, a[1].u.list->items);
    } else if (f->nargs == 0) {
        t_jit_maybe(vm, f, NULL);
    }
    *r = mk_bool(f->native != NULL ? 1 : 1);
}

#define MAX_ATOMICS 256
typedef struct {
    int in_use;
#ifdef _WIN32
    volatile LONG64 val;
#else
    volatile int64_t val;
#endif
} TAtomicSlot;

static TAtomicSlot g_atomics[MAX_ATOMICS];

static void sys_atomic_new(TVM *vm, TVal *a, int argc, TVal *r) {
    int64_t init_v = (argc >= 1 && a[0].t == T_INT) ? a[0].u.i : 0;
    for (int i = 1; i < MAX_ATOMICS; i++) {
        if (!g_atomics[i].in_use) {
            g_atomics[i].in_use = 1;
            g_atomics[i].val = init_v;
            *r = mk_int(i);
            return;
        }
    }
    die(vm, "atomic_new: max atomics reached");
}

static void sys_atomic_load(TVM *vm, TVal *a, int argc, TVal *r) {
    if (argc != 1 || a[0].t != T_INT) die(vm, "atomic_load(id: int)");
    int id = (int)a[0].u.i;
    if (id <= 0 || id >= MAX_ATOMICS || !g_atomics[id].in_use) die(vm, "atomic_load: invalid id");
#ifdef _WIN32
    int64_t v = (int64_t)InterlockedCompareExchange64(&g_atomics[id].val, 0, 0);
#else
    int64_t v = __sync_add_and_fetch(&g_atomics[id].val, 0);
#endif
    *r = mk_int(v);
}

static void sys_atomic_store(TVM *vm, TVal *a, int argc, TVal *r) {
    if (argc != 2 || a[0].t != T_INT || a[1].t != T_INT) die(vm, "atomic_store(id: int, val: int)");
    int id = (int)a[0].u.i;
    if (id <= 0 || id >= MAX_ATOMICS || !g_atomics[id].in_use) die(vm, "atomic_store: invalid id");
#ifdef _WIN32
    InterlockedExchange64(&g_atomics[id].val, (LONG64)a[1].u.i);
#else
    __sync_lock_test_and_set(&g_atomics[id].val, a[1].u.i);
#endif
    *r = mk_nil();
}

static void sys_atomic_add(TVM *vm, TVal *a, int argc, TVal *r) {
    if (argc != 2 || a[0].t != T_INT || a[1].t != T_INT) die(vm, "atomic_add(id: int, delta: int)");
    int id = (int)a[0].u.i;
    if (id <= 0 || id >= MAX_ATOMICS || !g_atomics[id].in_use) die(vm, "atomic_add: invalid id");
#ifdef _WIN32
    int64_t old_v = (int64_t)InterlockedExchangeAdd64(&g_atomics[id].val, (LONG64)a[1].u.i);
#else
    int64_t old_v = __sync_fetch_and_add(&g_atomics[id].val, a[1].u.i);
#endif
    *r = mk_int(old_v);
}

static void sys_atomic_cas(TVM *vm, TVal *a, int argc, TVal *r) {
    if (argc != 3 || a[0].t != T_INT || a[1].t != T_INT || a[2].t != T_INT)
        die(vm, "atomic_cas(id: int, expected: int, desired: int)");
    int id = (int)a[0].u.i;
    if (id <= 0 || id >= MAX_ATOMICS || !g_atomics[id].in_use) die(vm, "atomic_cas: invalid id");
#ifdef _WIN32
    LONG64 prev = InterlockedCompareExchange64(&g_atomics[id].val, (LONG64)a[2].u.i, (LONG64)a[1].u.i);
    *r = mk_bool(prev == (LONG64)a[1].u.i);
#else
    int ok = __sync_bool_compare_and_swap(&g_atomics[id].val, a[1].u.i, a[2].u.i);
    *r = mk_bool(ok);
#endif
}

static void sys_atomic_free(TVM *vm, TVal *a, int argc, TVal *r) {
    if (argc != 1 || a[0].t != T_INT) die(vm, "atomic_free(id: int)");
    int id = (int)a[0].u.i;
    if (id > 0 && id < MAX_ATOMICS) g_atomics[id].in_use = 0;
    *r = mk_bool(1);
}

#define MAX_MUTEXES 128
typedef struct {
    int in_use;
#ifdef _WIN32
    CRITICAL_SECTION cs;
#else
    pthread_mutex_t mtx;
#endif
} TMutexSlot;

static TMutexSlot g_mutexes[MAX_MUTEXES];

static void sys_mutex_new(TVM *vm, TVal *a, int argc, TVal *r) {
    (void)a; (void)argc;
    for (int i = 1; i < MAX_MUTEXES; i++) {
        if (!g_mutexes[i].in_use) {
            g_mutexes[i].in_use = 1;
#ifdef _WIN32
            InitializeCriticalSection(&g_mutexes[i].cs);
#else
            pthread_mutex_init(&g_mutexes[i].mtx, NULL);
#endif
            *r = mk_int(i);
            return;
        }
    }
    die(vm, "mutex_new: max mutexes reached");
}

static void sys_mutex_lock(TVM *vm, TVal *a, int argc, TVal *r) {
    if (argc != 1 || a[0].t != T_INT) die(vm, "mutex_lock(id: int)");
    int id = (int)a[0].u.i;
    if (id <= 0 || id >= MAX_MUTEXES || !g_mutexes[id].in_use) die(vm, "mutex_lock: invalid id");
#ifdef _WIN32
    EnterCriticalSection(&g_mutexes[id].cs);
#else
    pthread_mutex_lock(&g_mutexes[id].mtx);
#endif
    *r = mk_bool(1);
}

static void sys_mutex_unlock(TVM *vm, TVal *a, int argc, TVal *r) {
    if (argc != 1 || a[0].t != T_INT) die(vm, "mutex_unlock(id: int)");
    int id = (int)a[0].u.i;
    if (id <= 0 || id >= MAX_MUTEXES || !g_mutexes[id].in_use) die(vm, "mutex_unlock: invalid id");
#ifdef _WIN32
    LeaveCriticalSection(&g_mutexes[id].cs);
#else
    pthread_mutex_unlock(&g_mutexes[id].mtx);
#endif
    *r = mk_bool(1);
}

static void sys_mutex_free(TVM *vm, TVal *a, int argc, TVal *r) {
    if (argc != 1 || a[0].t != T_INT) die(vm, "mutex_free(id: int)");
    int id = (int)a[0].u.i;
    if (id > 0 && id < MAX_MUTEXES && g_mutexes[id].in_use) {
#ifdef _WIN32
        DeleteCriticalSection(&g_mutexes[id].cs);
#else
        pthread_mutex_destroy(&g_mutexes[id].mtx);
#endif
        g_mutexes[id].in_use = 0;
    }
    *r = mk_bool(1);
}

static void sys_tensor_share(TVM *vm, TVal *a, int argc, TVal *r) {
    if (argc != 1 || a[0].t != T_TENSOR) die(vm, "tensor_share(t: tensor)");
    TTensor *t = a[0].u.ten;
    ten_sync(t);
    t->refc++;
    *r = mk_int((int64_t)(uintptr_t)t);
}

static void sys_tensor_from_shared(TVM *vm, TVal *a, int argc, TVal *r) {
    if (argc != 1 || a[0].t != T_INT || a[0].u.i == 0) die(vm, "tensor_from_shared(handle: int)");
    TTensor *t = (TTensor*)(uintptr_t)a[0].u.i;
    t->refc++;
    r->t = T_TENSOR;
    r->u.ten = t;
}

static void sys_tensor_slice_sum(TVM *vm, TVal *a, int argc, TVal *r) {
    if (argc != 3 || (a[0].t != T_TENSOR && a[0].t != T_INT) || a[1].t != T_INT || a[2].t != T_INT)
        die(vm, "tensor_slice_sum(t_or_handle, start: int, end: int)");
    TTensor *t = (a[0].t == T_TENSOR) ? a[0].u.ten : (TTensor*)(uintptr_t)a[0].u.i;
    size_t lo = (a[1].u.i > 0) ? (size_t)a[1].u.i : 0;
    size_t hi = (a[2].u.i > 0) ? (size_t)a[2].u.i : 0;
    if (hi > t->len) hi = t->len;
    const double *d = t->d;
    double s0 = 0.0, s1 = 0.0, s2 = 0.0, s3 = 0.0;
    size_t i = lo;
    for (; i + 4 <= hi; i += 4) {
        s0 += d[i];
        s1 += d[i + 1];
        s2 += d[i + 2];
        s3 += d[i + 3];
    }
    double tot = (s0 + s1) + (s2 + s3);
    for (; i < hi; i++) tot += d[i];
    *r = mk_float(tot);
}

static void sys_cstruct_sum_f64(TVM *vm, TVal *a, int argc, TVal *r) {
    if (argc != 4 || a[0].t != T_INT || a[1].t != T_INT || a[2].t != T_INT || a[3].t != T_INT)
        die(vm, "cstruct_sum_f64(base_ptr, count, stride_bytes, field_offset)");
    const uint8_t *base = (const uint8_t*)(uintptr_t)a[0].u.i;
    size_t count = (size_t)a[1].u.i;
    size_t stride = (size_t)a[2].u.i;
    size_t off = (size_t)a[3].u.i;
    const uint8_t *p = base + off;
    size_t s2 = stride * 2, s3 = stride * 3, s4 = stride * 4;
    double a0 = 0.0, a1 = 0.0, a2 = 0.0, a3 = 0.0;
    size_t i = 0;
    for (; i + 4 <= count; i += 4, p += s4) {
        a0 += *(const double*)(p);
        a1 += *(const double*)(p + stride);
        a2 += *(const double*)(p + s2);
        a3 += *(const double*)(p + s3);
    }
    double sum = (a0 + a1) + (a2 + a3);
    for (; i < count; i++, p += stride) {
        sum += *(const double*)(p);
    }
    *r = mk_float(sum);
}

static void sys_cstruct_axpy_f64(TVM *vm, TVal *a, int argc, TVal *r) {
    if (argc != 6 || a[0].t != T_INT || a[1].t != T_INT || a[2].t != T_INT ||
        a[3].t != T_INT || a[4].t != T_INT || (a[5].t != T_FLOAT && a[5].t != T_INT))
        die(vm, "cstruct_axpy_f64(base_ptr, count, stride_bytes, dst_off, src_off, scale)");
    uint8_t *base = (uint8_t*)(uintptr_t)a[0].u.i;
    size_t count = (size_t)a[1].u.i;
    size_t stride = (size_t)a[2].u.i;
    size_t dst_off = (size_t)a[3].u.i;
    size_t src_off = (size_t)a[4].u.i;
    double scale = (a[5].t == T_FLOAT) ? a[5].u.f : (double)a[5].u.i;
    uint8_t *p = base;
    size_t s2 = stride * 2, s3 = stride * 3, s4 = stride * 4;
    size_t i = 0;
    for (; i + 4 <= count; i += 4, p += s4) {
        *(double*)(p + dst_off)      += scale * (*(const double*)(p + src_off));
        *(double*)(p + stride + dst_off) += scale * (*(const double*)(p + stride + src_off));
        *(double*)(p + s2 + dst_off) += scale * (*(const double*)(p + s2 + src_off));
        *(double*)(p + s3 + dst_off) += scale * (*(const double*)(p + s3 + src_off));
    }
    for (; i < count; i++, p += stride) {
        *(double*)(p + dst_off) += scale * (*(const double*)(p + src_off));
    }
    *r = mk_nil();
}

static TVal mk_native(TNativeFn fn, const char *name);
static void make_sys_mod(TVM *vm, TVal *r) {
    static const struct { const char *name; TNativeFn fn; } S[] = {
        {"clock", sys_clock}, {"time", sys_time}, {"sleep", sys_sleep},
        {"getenv", sys_getenv}, {"setenv", sys_setenv}, {"platform", sys_platform},
        {"exepath", sys_exepath}, {"getcwd", sys_getcwd}, {"chdir", sys_chdir},
        {"abspath", sys_abspath}, {"exists", sys_exists}, {"isfile", sys_isfile},
        {"isdir", sys_isdir}, {"mtime", sys_mtime}, {"mkdir", sys_mkdir},
        {"remove", sys_remove}, {"listdir", sys_listdir}, {"mktemp", sys_mktemp},
        {"sha256", sys_sha256}, {"run", sys_run}, {"stdin_line", sys_stdin_line},
        {"stdin_read", sys_stdin_read}, {"stdout_write", sys_stdout_write},
        {"stderr_write", sys_stderr_write}, {"zip_pack", sys_zip_pack},
        {"zip_unpack", sys_zip_unpack}, {"exit", sys_exit},

        /* mmap */
        {"mmap_open", sys_mmap_open}, {"mmap_close", sys_mmap_close},
        {"mmap_size", sys_mmap_size}, {"mmap_read_bytes", sys_mmap_read_bytes},
        {"mmap_read_u8", sys_mmap_read_u8}, {"mmap_read_u16", sys_mmap_read_u16},
        {"mmap_read_u32", sys_mmap_read_u32}, {"mmap_read_u64", sys_mmap_read_u64},
        {"mmap_read_i32", sys_mmap_read_i32}, {"mmap_read_i64", sys_mmap_read_i64},
        {"mmap_read_f32", sys_mmap_read_f32}, {"mmap_read_f64", sys_mmap_read_f64},
        {"mmap_tensor_f32", sys_mmap_tensor_f32}, {"mmap_tensor_f16", sys_mmap_tensor_f16},
        {"mmap_tensor_q4_0", sys_mmap_tensor_q4_0}, {"mmap_tensor_q8_0", sys_mmap_tensor_q8_0},

        /* quant / dequant */
        {"quant_f16", sys_quant_f16}, {"dequant_f16", sys_dequant_f16},
        {"quant_bf16", sys_quant_bf16}, {"dequant_bf16", sys_dequant_bf16},
        {"quant_f32", sys_quant_f32}, {"dequant_f32", sys_dequant_f32},
        {"quant_q4_0", sys_quant_q4_0}, {"dequant_q4_0", sys_dequant_q4_0},
        {"quant_q8_0", sys_quant_q8_0}, {"dequant_q8_0", sys_dequant_q8_0},

        /* C FFI */
        {"ffi_open", sys_ffi_open}, {"ffi_close", sys_ffi_close},
        {"ffi_sym", sys_ffi_sym}, {"ffi_call", sys_ffi_call},

        /* raw memory */
        {"mem_alloc", sys_mem_alloc}, {"mem_free", sys_mem_free},
        {"mem_read_ptr", sys_mem_read_ptr}, {"mem_write_ptr", sys_mem_write_ptr},
        {"mem_read_i32", sys_mem_read_i32}, {"mem_write_i32", sys_mem_write_i32},
        {"mem_read_u16", sys_mem_read_u16}, {"mem_write_u16", sys_mem_write_u16},
        {"mem_read_str", sys_mem_read_str},
        {"mem_read_f32", sys_mem_read_f32}, {"mem_write_f32", sys_mem_write_f32},
        {"mem_read_f64", sys_mem_read_f64}, {"mem_write_f64", sys_mem_write_f64},
        {"arm64_selftest", sys_arm64_selftest}, {"arm64_dump", sys_arm64_dump},
        {"arm64_compile", sys_arm64_compile}, {"arm64_run", sys_arm64_run},
        {"arm64_osr_run", sys_arm64_osr_run},

        /* thread & channel */
        {"channel_new", sys_channel_new}, {"channel_send", sys_channel_send},
        {"channel_recv", sys_channel_recv}, {"channel_close", sys_channel_close},
        {"channel_len", sys_channel_len},
        {"thread_spawn", sys_thread_spawn}, {"thread_join", sys_thread_join},

        /* sockets & standalone executable bundler (v6.0) */
        {"sock_listen", sys_sock_listen}, {"sock_port", sys_sock_port},
        {"sock_accept", sys_sock_accept}, {"sock_connect", sys_sock_connect},
        {"sock_send", sys_sock_send}, {"sock_recv", sys_sock_recv},
        {"sock_close", sys_sock_close}, {"sock_udp_open", sys_sock_udp_open},
        {"sock_udp_sendto", sys_sock_udp_sendto}, {"sock_udp_recvfrom", sys_sock_udp_recvfrom},
        {"build_exe", sys_build_exe},

        /* v7.0 Rust-grade optimizations (Instant JIT, Atomics, Mutex, SharedTensor, cstruct SIMD) */
        {"jit_compile", sys_jit_compile},
        {"atomic_new", sys_atomic_new}, {"atomic_load", sys_atomic_load},
        {"atomic_store", sys_atomic_store}, {"atomic_add", sys_atomic_add},
        {"atomic_cas", sys_atomic_cas}, {"atomic_free", sys_atomic_free},
        {"mutex_new", sys_mutex_new}, {"mutex_lock", sys_mutex_lock},
        {"mutex_unlock", sys_mutex_unlock}, {"mutex_free", sys_mutex_free},
        {"tensor_share", sys_tensor_share}, {"tensor_from_shared", sys_tensor_from_shared},
        {"tensor_slice_sum", sys_tensor_slice_sum},
        {"cstruct_sum_f64", sys_cstruct_sum_f64}, {"cstruct_axpy_f64", sys_cstruct_axpy_f64}
    };
    size_t ne = sizeof(S) / sizeof(S[0]);
    TList *fl = list_new(ne);
    TVal targs[3], tp, *vals;
    for (size_t i = 0; i < ne; i++) list_push(fl, mk_str(S[i].name));
    targs[0] = mk_str("module");
    targs[1].t = T_LIST; targs[1].u.list = fl;
    targs[2].t = T_DICT; targs[2].u.dict = dict_new();
    n___struct(vm, targs, 3, &tp);
    for (int i = 0; i < 3; i++) tval_free(&targs[i]);
    vals = (TVal*)malloc(sizeof(TVal) * (ne + 1));
    vals[0] = tp;
    for (size_t i = 0; i < ne; i++) vals[i + 1] = mk_native(S[i].fn, S[i].name);
    n___new(vm, vals, (int)(ne + 1), r);
    for (size_t i = 1; i <= ne; i++) tval_free(&vals[i]);
    free(vals);
    tval_free(&tp);
}

/* load("tên"): nạp tên.tb (thư mục hiện tại, rồi thư mục của chương trình chính),
   chạy top-level của nó một lần (cache theo đường dẫn) -> bản ghi kiểu "module"
   gồm các global top-level (giá trị lúc nạp xong). */
static void n_load(TVM *vm, TVal *a, int argc, TVal *r) {
    char path[4200], eb[1200];
    TProg *root = vm->root;
    if (argc != 1 || a[0].t != T_STR) die(vm, "load(name str)");
    if (a[0].u.str.n == 5 && !memcmp(a[0].u.str.s, "__sys", 5)) {
        make_sys_mod(vm, r);
        return;
    }
    if (a[0].u.str.n > 900) die(vm, "load: tên quá dài");
    /* validate tên module: chỉ ký tự an toàn cho tên file — chặn shell metachar */
    {
        const char *s = a[0].u.str.s;
        size_t n = a[0].u.str.n;
        for (size_t i = 0; i < n; i++) {
            char c = s[i];
            if (!((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
                  (c >= '0' && c <= '9') || c == '_' || c == '-' ||
                  c == '.' || c == '/' || c == '\\'))
                die(vm, "load: tên module chứa ký tự không hợp lệ");
        }
    }
    {
        /* thư mục hiện tại, rồi thư mục của chương trình chính; có tên.t mới hơn
           tên.tb (hoặc chưa có .tb) thì tự dịch trước */
        char base[2][2048], src[4200];
        int nb = 0;
        snprintf(base[nb++], sizeof(base[0]), "%.*s", (int)a[0].u.str.n, a[0].u.str.s);
        if (root->path) {
            const char *s1 = strrchr(root->path, '/'), *s2 = strrchr(root->path, '\\');
            const char *sl = s1 > s2 ? s1 : s2;
            if (sl) snprintf(base[nb++], sizeof(base[0]), "%.*s/%.*s", (int)(sl - root->path), root->path,
                             (int)a[0].u.str.n, a[0].u.str.s);
        }
        snprintf(path, sizeof(path), "%s.tb", base[0]);
        for (int q = 0; q < nb; q++) {
            snprintf(src, sizeof(src), "%s.t", base[q]);
            snprintf(path, sizeof(path), "%s.tb", base[q]);
            char cm[256];
            if (compile_if_stale(src, path, cm, sizeof(cm)) < 0) {
                const char *bn = strrchr(src, '/'), *b2 = strrchr(src, '\\');
                if (b2 > bn) bn = b2;
                snprintf(eb, sizeof(eb), "load: %.200s: %.200s", bn ? bn + 1 : src, cm[0] ? cm : "lỗi biên dịch");
                die(vm, eb);
            }
            if (file_exists(path)) break;
        }
    }
    for (TProg *m = root->next_mod; m; m = m->next_mod) {
        if (strcmp(m->path, path) != 0) continue;
        if (m->loading) die(vm, "load: nạp vòng");
        *r = m->mod_rec; tval_retain(r);
        return;
    }
    TProg *m = (TProg*)calloc(1, sizeof(TProg));
    if (!m) exit(1);
    if (tvm_read_tb(path, m, eb, 256) != 0) {
        free(m);
        snprintf(eb, sizeof(eb), "no module %.*s", (int)(a[0].u.str.n > 200 ? 200 : a[0].u.str.n), a[0].u.str.s);
        die(vm, eb);
    }
    if (m->nglobals < NAT_NFIXED) { tvm_prog_release(m); free(m); die(vm, "load: .tb hỏng"); }
    m->mod_rec.t = T_NIL;
    prog_init_globals(m);
    m->ic = (TAttrIC*)calloc(m->len ? m->len : 1, sizeof(TAttrIC));
    if (!m->ic) exit(1);
    m->loading = 1;
    m->next_mod = root->next_mod;
    root->next_mod = m;           /* liên kết trước khi chạy: GC thấy global của nó */
    {
        size_t sp0 = vm->sp, pc0 = vm->pc, stop0 = vm->stop_nf, nf0 = vm->nframes;
        TProg *p0 = vm->prog;
        TFrame *fr = push_frame(vm);
        fr->ret_pc = pc0; fr->bp = sp0; fr->nlocals = 0; fr->fn = NULL;
        fr->ret_prog = p0; fr->is_mod = 1;
        vm->prog = m;
        vm->pc = 0;
        vm->stop_nf = nf0;        /* lồng trong native: GC hoãn tới tầng ngoài */
        tvm_run(vm);
        while (vm->nframes > nf0) free_frame(vm);
        while (vm->sp > sp0) tval_free(&vm->stack[--vm->sp]);
        vm->prog = p0;
        vm->pc = pc0;
        vm->stop_nf = stop0;
    }
    m->loading = 0;
    {
        /* kiểu "module" (field = tên export) + bản ghi giá trị */
        uint16_t ne = m->nexports;
        TList *fl = list_new(ne);
        TVal targs[3], tp, *vals;
        for (uint16_t i = 0; i < ne; i++) list_push(fl, mk_str(m->exnames[i]));
        targs[0] = mk_str("module");
        targs[1].t = T_LIST; targs[1].u.list = fl;
        targs[2].t = T_DICT; targs[2].u.dict = dict_new();
        n___struct(vm, targs, 3, &tp);
        tp.u.dict->mod = m;
        for (int i = 0; i < 3; i++) tval_free(&targs[i]);
        vals = (TVal*)malloc(sizeof(TVal) * ((size_t)ne + 1));
        vals[0] = tp;
        for (uint16_t i = 0; i < ne; i++) {
            uint16_t sl = m->exslots[i];
            TVal v = sl < m->nglobals ? m->globals[sl] : mk_nil();
            if (v.t == T_CELL) v = v.u.cell->val;
            vals[i + 1] = v;
        }
        n___new(vm, vals, ne + 1, &m->mod_rec);
        free(vals);
        tval_free(&tp);
    }
    *r = m->mod_rec; tval_retain(r);
}

void tvm_init(TVM *vm, TProg *p) {
    memset(vm, 0, sizeof(*vm));
    {
        const char *g = getenv("TVM_GCTHRESH");
        if (g && atol(g) > 0) gc_threshold = (size_t)atol(g);
    }
    vm->prog = p;
    cur_vm = vm;
    vm->cap = STACK_INIT;
    vm->stack = (TVal*)malloc(vm->cap * sizeof(TVal));
    vm->fcap = FRAME_INIT;
    vm->frames = (TFrame*)malloc(vm->fcap * sizeof(TFrame));
    vm->hcap = HAND_INIT;
    vm->handlers = (THandler*)malloc(vm->hcap * sizeof(THandler));
    vm->root = p;
    p->mod_rec.t = T_NIL;
    prog_init_globals(p);
}
/* cấp mảng global + đặt builtin (chương trình chính và module nạp bằng load) */
static void prog_init_globals(TProg *p) {
    p->globals = (TVal*)calloc(p->nglobals ? p->nglobals : 1, sizeof(TVal));
    fill_undef(p->globals, p->nglobals ? p->nglobals : 1);
    {
        struct { int idx; TNativeFn fn; const char *name; } nt[] = {
            {NAT_RANGE, n_range, "range"}, {NAT_LEN, n_len, "len"},
            {NAT_STR, n_str, "str"}, {NAT_INT, n_int, "int"},
            {NAT_FLOAT, n_float, "float"}, {NAT_TYPE, n_type, "type"},
            {NAT_TENSOR, n_tensor, "tensor"}, {NAT_OPEN, n_open, "open"},
            {NAT_READ, n_read, "read"}, {NAT_WRITE, n_write, "write"},
            {NAT_CLOSE, n_close, "close"}, {NAT_ORD, n_ord, "ord"},
            {NAT_CHR, n_chr, "chr"}, {NAT_ARGV, n_argv, "argv"},
            {NAT_PRINT, n_print, "print"}, {NAT_MEM, n_mem, "mem"},
            {NAT_F64BITS, n_f64bits, "f64bits"},
            {NAT_ABS, n_abs, "abs"}, {NAT_ROUND, n_round, "round"},
            {NAT_MIN, n_min, "min"}, {NAT_MAX, n_max, "max"},
            {NAT_SUM, n_sum, "sum"}, {NAT_ANY, n_any, "any"},
            {NAT_ALL, n_all, "all"}, {NAT_SORT, n_sort, "sort"},
            {NAT_REVERSED, n_reversed, "reversed"},
            {NAT_ENUMERATE, n_enumerate, "enumerate"},
            {NAT_ZIP, n_zip, "zip"}, {NAT_MAP, n_map, "map"},
            {NAT_FILTER, n_filter, "filter"},
            {NAT_SET, n_set, "set"}, {NAT_INPUT, n_input, "input"},
            {NAT___STRUCT, n___struct, "__struct"}, {NAT___NEW, n___new, "__new"},
            {NAT_ZEROS, n_zeros, "zeros"},
            {NAT_ONES, n_ones, "ones"},
            {NAT_FULL, n_full, "full"},
            {NAT_ARANGE, n_arange, "arange"},
            {NAT_EYE, n_eye, "eye"},
            {NAT_RAND, n_rand, "rand"},
            {NAT_RANDN, n_randn, "randn"},
            {NAT_SEED, n_seed, "seed"},
            {NAT_SQRT, n_sqrt, "sqrt"},
            {NAT_EXP, n_exp, "exp"},
            {NAT_LOG, n_log, "log"},
            {NAT_SIN, n_sin, "sin"},
            {NAT_COS, n_cos, "cos"},
            {NAT_TANH, n_tanh, "tanh"},
            {NAT_FLOOR, n_floor, "floor"},
            {NAT_CEIL, n_ceil, "ceil"},
            {NAT_LOAD, n_load, "load"},
        };
        size_t i;
        if (NAT_PI < p->nglobals) p->globals[NAT_PI] = mk_float(3.141592653589793);
        for (i = 0; i < sizeof(nt)/sizeof(nt[0]); i++) {
            if ((uint16_t)nt[i].idx < p->nglobals)
                p->globals[nt[i].idx] = mk_native(nt[i].fn, nt[i].name);
        }
    }
}
#ifdef TVM_PROF
static void prof_dump(void);
#endif
void tvm_free(TVM *vm) {
#ifdef TVM_PROF
    prof_dump();
#endif
    tbuf_clear();
    size_t i;
    vm->prog = vm->root;
    while (vm->sp) tval_free(&vm->stack[--vm->sp]);
    while (vm->nframes) {
        free_frame(vm);
        free(vm->frames[vm->nframes].locals);
        vm->frames[vm->nframes].locals = NULL;
        vm->frames[vm->nframes].lcap = 0;
    }
    /* module đã nạp: free giá trị global trước (node GC thuộc prog gốc) */
    if (vm->prog) {
        TProg *m = vm->prog->next_mod;
        while (m) {
            TProg *nx = m->next_mod;
            for (i = 0; i < m->nglobals; i++) tval_free(&m->globals[i]);
            tval_free(&m->mod_rec);
            free(m->globals);
            if (m->ic) {
                for (size_t q = 0; q < m->len; q++)
                    if (m->ic[q].kind == 4) { TVal *kv = (TVal*)m->ic[q].key; tval_free(kv); free(kv); }
                free(m->ic);
            }
            tvm_prog_release(m);
            free(m);
            m = nx;
        }
        vm->prog->next_mod = NULL;
    }
    if (vm->prog) {
        for (i = 0; i < vm->prog->nglobals; i++) tval_free(&vm->prog->globals[i]);
        /* free mảng, val đã free */
        free(vm->prog->globals);
        vm->prog->globals = NULL;
        /* dọn node GC còn lại (object đã free hết ở trên) */
        {
            GCNode *nd = (GCNode*)vm->root->gc_head;
            while (nd) {
                GCNode *nx = nd->next;
                pk_free(nd, sizeof(GCNode));
                nd = nx;
            }
            vm->root->gc_head = NULL;
        }
    }
    if (vm->prog && vm->prog->ic) {
        for (size_t q = 0; q < vm->prog->len; q++)
            if (vm->prog->ic[q].kind == 4) {
                TVal *kv = (TVal*)vm->prog->ic[q].key;
                tval_free(kv); free(kv);
            }
        free(vm->prog->ic); vm->prog->ic = NULL;
    }
    free(vm->stack); free(vm->frames); free(vm->handlers);
}

/* ---------- slot helpers (ngữ nghĩa LOAD/STORE, dùng chung superop) ---------- */
TVM_INLINE TVal slot_get(TVM *vm, int isg, uint16_t s) {
    TVal v;
    if (isg) {
        TVal *gp;
        if (s >= vm->prog->nglobals) die(vm, "bad global");
        gp = &vm->prog->globals[s];
        v = (gp->t == T_CELL) ? gp->u.cell->val : *gp;
    } else {
        TFrame *fr = &vm->frames[vm->nframes - 1];
        TVal *sp;
        if (s >= fr->nlocals && !(vm->nframes == 1 && fr->nlocals == 0))
            die(vm, "bad slot");
        sp = &fr->locals[s];
        v = (sp->t == T_CELL) ? sp->u.cell->val : *sp;
    }
    if (v.t == T_UNDEF) die(vm, "undef variable");
    TV_RETAIN(&v);
    return v;
}
TVM_INLINE void slot_set(TVM *vm, int isg, uint16_t s, TVal v) {
    if (isg) {
        TVal *gp;
        if (s >= vm->prog->nglobals) die(vm, "bad global");
        gp = &vm->prog->globals[s];
        if (gp->t == T_CELL) {
            tval_free(&gp->u.cell->val);
            gp->u.cell->val = v;
        } else {
            tval_free(gp);
            *gp = v;
        }
    } else {
        TFrame *fr = &vm->frames[vm->nframes - 1];
        if (s >= fr->nlocals && !(vm->nframes == 1 && fr->nlocals == 0))
            die(vm, "bad slot");
        if (fr->locals[s].t == T_CELL) {
            tval_free(&fr->locals[s].u.cell->val);
            fr->locals[s].u.cell->val = v;
        } else {
            tval_free(&fr->locals[s]);
            fr->locals[s] = v;
        }
    }
}
TVM_INLINE void slot_replace(TVM *vm, int isg, uint16_t s, TVal v) {
    /* DECL/BIND: thay mới dứt khoát */
    if (isg) {
        if (s >= vm->prog->nglobals) die(vm, "bad global");
        tval_free(&vm->prog->globals[s]);
        vm->prog->globals[s] = v;
    } else {
        TFrame *fr = &vm->frames[vm->nframes - 1];
        if (s >= fr->nlocals && !(vm->nframes == 1 && fr->nlocals == 0))
            die(vm, "bad slot");
        tval_free(&fr->locals[s]);
        fr->locals[s] = v;
    }
}

/* trả về từ frame hiện tại với giá trị rv (RET và OSR dùng chung) */
static void call_value(TVM *vm, uint8_t k);
static void do_return(TVM *vm, TVal rv) {
    TFrame *fr = &vm->frames[vm->nframes - 1];
    size_t bp = fr->bp;
    size_t rpc = fr->ret_pc;
    TProg *rprog = fr->ret_prog;
    /* return bên trong try: gỡ handler của frame này (nếu không, handler
       "chết" còn trên stack và lỗi sau đó nhảy vào code của hàm đã về) */
    while (vm->nhandlers > 0 &&
           vm->handlers[vm->nhandlers - 1].nframes >= vm->nframes)
        vm->nhandlers--;
    free_frame(vm);
    while (vm->sp > bp) { --vm->sp; TV_FREE(&vm->stack[vm->sp]); }
    vm->sp = bp;
    vm->pc = rpc;
    if (rprog) vm->prog = rprog;
    push(vm, rv);
}

/* gọi giá trị ở stack[sp-k-1] với k đối số phía trên (OP_CALL, CALLM dự phòng) */
static void call_value(TVM *vm, uint8_t k) {
            TVal fv = vm->stack[vm->sp - k - 1];
            if (fv.t == T_FN) {
                TFn *f = fv.u.fn;
                if (k < f->min_args || (!f->vararg && k > f->nargs)) {
                    char eb[64];
                    snprintf(eb, sizeof(eb), "%.40s arity", "call");
                    die(vm, eb);
                }
                /* JIT hot-path: đếm + thử biên dịch, rồi chạy native nếu tag khớp */
                if (!f->vararg && k == f->nargs) {
                    if (!f->native)
                        t_jit_maybe(vm, f, &vm->stack[vm->sp - k]);
                    if (f->native) {
                        TVal rv; rv.t = T_NIL;
                        if (t_jit_call(vm, f, &vm->stack[vm->sp - k], k, &rv)) {
                            int i;
                            for (i = 0; i < k + 1; i++)
                                TV_FREE(&vm->stack[vm->sp - k - 1 + i]);
                            vm->sp -= k + 1;
                            push(vm, rv);
                            return;
                        }
                        /* kiểu không khớp bản nào / deopt: có thể biên dịch bản đặc hoá
                           mới cho bộ kiểu này, rồi rẽ interpreter bên dưới */
                        t_jit_maybe(vm, f, &vm->stack[vm->sp - k]);
                    }
                }
                TFrame *fr = push_frame(vm);
                size_t bp = vm->sp - k - 1;
                fr->ret_pc = vm->pc; fr->bp = bp;
                fr->nlocals = f->nlocals;
                fr->fn = f;
                fr->ret_prog = vm->prog; fr->is_mod = 0;
                vm->prog = f->prog;
                /* ref của ô stack[bp] (giá trị hàm) chuyển thẳng cho frame (fr->fn) */
                if (f->nlocals == 0) {
                    fr->locals = NULL;
                } else if (fr->locals && fr->lcap >= f->nlocals) {
                    fill_undef(fr->locals, f->nlocals);
                } else {
                    free(fr->locals);
                    fr->locals = (TVal*)calloc(f->nlocals, sizeof(TVal));
                    if (!fr->locals) exit(1);
                    fill_undef(fr->locals, f->nlocals);
                    fr->lcap = f->nlocals;
                }
                if (f->vararg) {
                    uint8_t fixed = f->min_args;
                    for (uint8_t i = 0; i < fixed; i++)
                        fr->locals[i] = vm->stack[bp + 1 + i];
                    TList *rest = list_new(k > fixed ? (size_t)(k - fixed) : 0);
                    for (uint8_t i = fixed; i < k; i++)
                        rest->items[rest->len++] = vm->stack[bp + 1 + i];
                    fr->locals[fixed].t = T_LIST;
                    fr->locals[fixed].u.list = rest;
                } else {
                    for (uint8_t i = 0; i < k; i++)
                        fr->locals[i] = vm->stack[bp + 1 + i];
                    for (uint8_t i = k; i < f->nargs; i++)
                        fr->locals[i] = mk_nil();
                }
                vm->sp = bp;
                vm->pc = (size_t)f->entry;
            } else if (fv.t == T_NATIVE) {
                TNative *nt = fv.u.nat;
                TVal eargs[9];
                TVal *ap = &vm->stack[vm->sp - k];
                int ek = k;
                if (nt->has_obj) {
                    if (k + 1 > 9) die(vm, "too many args");
                    eargs[0] = nt->obj;
                    for (int i = 0; i < k; i++) eargs[i + 1] = ap[i];
                    ap = eargs; ek = k + 1;
                }
                TVal rv; rv.t = T_NIL;
                nt->fn(vm, ap, ek, &rv);
                for (int i = 0; i < k + 1; i++)
                    TV_FREE(&vm->stack[vm->sp - k - 1 + i]);
                vm->sp -= k + 1;
                push(vm, rv);
            } else die(vm, "not callable");
            }

/* vào hàm T: đối số là stack[bp .. bp+nargs-1]; bỏ nfnslot ô trước đó (giá trị hàm) */
static void enter_fn(TVM *vm, TFn *f, size_t bp, int nargs) {
    TFrame *fr = push_frame(vm);
    fr->ret_pc = vm->pc; fr->bp = bp;
    fr->nlocals = f->nlocals;
    fr->fn = f;
    fr->ret_prog = vm->prog; fr->is_mod = 0;
    vm->prog = f->prog;
    f->refc++;
    if (f->nlocals == 0) {
        fr->locals = NULL;
    } else if (fr->locals && fr->lcap >= f->nlocals) {
        fill_undef(fr->locals, f->nlocals);
    } else {
        free(fr->locals);
        fr->locals = (TVal*)calloc(f->nlocals, sizeof(TVal));
        if (!fr->locals) exit(1);
        fill_undef(fr->locals, f->nlocals);
        fr->lcap = f->nlocals;
    }
    if (f->vararg) {
        int fixed = (int)f->min_args;
        for (int i = 0; i < fixed; i++) fr->locals[i] = vm->stack[bp + i];
        TList *rest = list_new(nargs > fixed ? (size_t)(nargs - fixed) : 0);
        for (int i = fixed; i < nargs; i++) rest->items[rest->len++] = vm->stack[bp + i];
        fr->locals[fixed].t = T_LIST;
        fr->locals[fixed].u.list = rest;
    } else {
        for (int i = 0; i < nargs; i++) fr->locals[i] = vm->stack[bp + i];
        for (int i = nargs; i < (int)f->nargs; i++) fr->locals[i] = mk_nil();
    }
    vm->sp = bp;
    vm->pc = (size_t)f->entry;
}

/* obj.name(a1..ak): stack [obj, a1..ak]. Record -> gọi method trực tiếp (self = obj);
   builtin -> gọi native method với a[0] = obj; còn lại: lấy thuộc tính rồi gọi. */
static void call_method(TVM *vm, uint8_t k, const char *nm, size_t len, TAttrIC *ic) {
    size_t bp = vm->sp - k - 1;
    TVal *ov = &vm->stack[bp];
    if (ov->t == T_DICT && ov->u.dict->kind == DK_REC) {
        TDict *rec = ov->u.dict, *tp = rec->meths;
        int mi = -1;
        if (ic->kind == 2 && ic->key == tp && tp && (size_t)ic->idx < tp->len &&
            tp->keys[ic->idx].t == T_STR && tp->keys[ic->idx].u.str.n == len &&
            name_eq(tp->keys[ic->idx].u.str.s, nm, len)) mi = ic->idx;
        else if (tp) {
            int isfield = 0;
            for (size_t i = 0; i < rec->len; i++)
                if (rec->keys[i].u.str.n == len && memcmp(rec->keys[i].u.str.s, nm, len) == 0) isfield = 1;
            if (!isfield) {
                TVal key = mk_str_n(nm, len);
                mi = dict_find(tp, &key);
                tval_free(&key);
                if (mi >= 0) { ic->kind = 2; ic->key = tp; ic->idx = mi; }
            }
        }
        if (mi >= 0 && (size_t)mi < tp->len && tp->vals[mi].t == T_FN) {
            TFn *f = tp->vals[mi].u.fn;
            if ((int)k + 1 < f->min_args || (!f->vararg && (int)k + 1 > f->nargs)) die(vm, "call arity");
            /* JIT: method nóng (self = record mượn) chạy native nếu kiểu khớp */
            if (!f->vararg && (int)k + 1 == f->nargs) {
                if (!f->native) t_jit_maybe(vm, f, &vm->stack[bp]);
                if (f->native) {
                    TVal rv; rv.t = T_NIL;
                    if (t_jit_call(vm, f, &vm->stack[bp], k + 1, &rv)) {
                        for (int i = 0; i <= k; i++) TV_FREE(&vm->stack[bp + i]);
                        vm->sp = bp;
                        push(vm, rv);
                        return;
                    }
                    t_jit_maybe(vm, f, &vm->stack[bp]);    /* bộ kiểu mới */
                }
            }
            enter_fn(vm, f, bp, k + 1);
            return;
        }
    } else if (ov->t == T_STR || ov->t == T_LIST || ov->t == T_TENSOR ||
               (ov->t == T_DICT && ov->u.dict->kind <= DK_SET)) {
        const void *tkey = (const void*)(intptr_t)(1 + (int)ov->t * 8 +
                                                    (ov->t == T_DICT ? ov->u.dict->kind : 0));
        TNativeFn fn = NULL;
        if (ic->kind == 3 && ic->key == tkey) fn = ic->fn;
        else if (len < 64) {
            char nb[64];
            memcpy(nb, nm, len); nb[len] = 0;
            fn = builtin_method(ov, nb);
            if (fn) { ic->kind = 3; ic->key = tkey; ic->fn = fn; }
        }
        if (fn) {
            TVal rv; rv.t = T_NIL;
            fn(vm, ov, k + 1, &rv);
            for (int i = 0; i <= k; i++) TV_FREE(&vm->stack[bp + i]);
            vm->sp = bp;
            push(vm, rv);
            return;
        }
    }
    {
        TVal r = attr_value(vm, ov, nm, len);
        tval_free(ov);
        *ov = r;
        call_value(vm, k);
    }
}

/* obj.name = v: stack [obj, v] (khớp set_index của t.py) */
static void set_attr(TVM *vm, const char *nm, size_t len, TAttrIC *ic) {
    TVal v = pop(vm), o = pop(vm);
    if (o.t == T_DICT && o.u.dict->kind == DK_REC) {
        TDict *d = o.u.dict;
        int fi = -1;
        if (ic->kind == 1 && ic->key == d->meths && (size_t)ic->idx < d->len &&
            d->keys[ic->idx].u.str.n == len && memcmp(d->keys[ic->idx].u.str.s, nm, len) == 0) fi = ic->idx;
        else {
            for (size_t i = 0; i < d->len; i++)
                if (d->keys[i].u.str.n == len && memcmp(d->keys[i].u.str.s, nm, len) == 0) { fi = (int)i; break; }
            if (fi >= 0) { ic->kind = 1; ic->key = d->meths; ic->idx = fi; }
        }
        if (fi < 0) {
            char eb[128];
            snprintf(eb, sizeof(eb), "%.40s has no field %.*s", rec_name(d), (int)(len > 40 ? 40 : len), nm);
            tval_free(&v); tval_free(&o); die(vm, eb);
        }
        {
            char eb[128];
            if (mod_ro(d, (size_t)fi, eb, sizeof(eb))) { tval_free(&v); tval_free(&o); die(vm, eb); }
        }
        tval_free(&d->vals[fi]);
        d->vals[fi] = v;
        mod_push(d, (size_t)fi);
        tval_free(&o);
        return;
    }
    if (o.t == T_DICT && o.u.dict->kind == DK_DICT) {
        dict_set(o.u.dict, mk_str_n(nm, len), v);
        tval_free(&o);
        return;
    }
    tval_free(&v);
    if (o.t == T_LIST || o.t == T_TENSOR) { tval_free(&o); die(vm, "index int"); }
    tval_free(&o);
    die(vm, "assign here");
}

/* ---------- main loop ---------- */
static int glob_match(const char *s, size_t n, const char *p, size_t m) {
    /* *, ? (byte-wise; ASCII đủ, UTF-8 đa byte so theo byte) */
    size_t si = 0, pi = 0;
    size_t star = (size_t)-1, ss = 0;
    while (si < n) {
        if (pi < m && (p[pi] == '?' || p[pi] == s[si])) {
            si++; pi++;
        } else if (pi < m && p[pi] == '*') {
            star = pi++; ss = si;
        } else if (star != (size_t)-1) {
            pi = star + 1; si = ++ss;
        } else return 0;
    }
    while (pi < m && p[pi] == '*') pi++;
    return pi == m;
}
#ifdef TVM_PROF
/* profile thô (build với -DTVM_PROF): đếm lệnh theo hàm (entry) và theo opcode,
   in top khi tvm_free. Chỉ dùng khi tối ưu VM / chương trình T. */
static uint64_t prof_op[256], *prof_fn;
static size_t prof_n;
static TProg *prof_prog;
static void prof_hit(TVM *vm, uint8_t op) {
    TFrame *fr = &vm->frames[vm->nframes - 1];
    size_t e = fr->fn ? (size_t)fr->fn->entry : 0;
    prof_op[op]++;
    if (!prof_fn) { prof_n = vm->prog->len + 1; prof_fn = (uint64_t*)calloc(prof_n, 8); prof_prog = vm->prog; }
    if (vm->prog == prof_prog && e < prof_n) prof_fn[e]++;
}
static void prof_dump(void) {
    uint64_t tot = 0;
    for (int i = 0; i < 256; i++) tot += prof_op[i];
    fprintf(stderr, "[prof] %llu lệnh\n", (unsigned long long)tot);
    for (int r = 0; r < 25 && prof_fn; r++) {
        size_t bi = 0;
        for (size_t i = 0; i < prof_n; i++) if (prof_fn[i] > prof_fn[bi]) bi = i;
        if (!prof_fn[bi]) break;
        const char *f = "?"; uint32_t ln = 0;
        tvm_pc_line(prof_prog, bi + 1, &f, &ln);
        fprintf(stderr, "  fn@%zu %s:%u  %llu (%.1f%%)\n", bi, f, ln,
                (unsigned long long)prof_fn[bi], 100.0 * prof_fn[bi] / (tot ? tot : 1));
        prof_fn[bi] = 0;
    }
    for (int r = 0; r < 25; r++) {
        int bi = 0;
        for (int i = 0; i < 256; i++) if (prof_op[i] > prof_op[bi]) bi = i;
        if (!prof_op[bi]) break;
        fprintf(stderr, "  op %3d  %llu (%.1f%%)\n", bi, (unsigned long long)prof_op[bi],
                100.0 * prof_op[bi] / (tot ? tot : 1));
        prof_op[bi] = 0;
    }
}
#endif
int tvm_run(TVM *vm) {
    uint8_t *c = vm->prog->code;
    size_t n = vm->prog->len;
    /* pc ở biến cục bộ (thanh ghi). vm->pc được ghi sau mỗi lần đọc opcode (lỗi báo
       đúng dòng) và đồng bộ đầy đủ quanh lệnh gọi/return/OSR/try (SAVE_PC/LOAD_PC) */
    size_t pc = vm->pc;
#define SAVE_PC() (vm->pc = pc)
#define LOAD_PC() (pc = vm->pc)
    /* frame main: chỉ lần chạy ngoài cùng (tvm_call_value gọi lại lồng nhau) */
    if (vm->nframes == 0) {
        TFrame *f = push_frame(vm);
        f->ret_pc = (size_t)-1; f->bp = 0; f->nlocals = 0; f->locals = NULL;
        f->fn = NULL; f->ret_prog = NULL; f->is_mod = 0;
        in_top = 1;
        cur_vm = vm;
        if (setjmp(top_jb)) { in_top = 0; return -1; }
    }
    if (!vm->prog->ic) {
        vm->prog->ic = (TAttrIC*)calloc(n ? n : 1, sizeof(TAttrIC));
        if (!vm->prog->ic) exit(1);
    }
    /* vm->prog chỉ đổi khi gọi/return qua hàm của module khác (load) và khi lỗi nhảy
       về handler: nạp lại c/n ngay sau các lệnh đó (không kiểm mỗi lệnh) */
#define RELOAD_CODE() do { c = vm->prog->code; n = vm->prog->len; } while (0)
    for (;;) {
        uint8_t op;
        /* (GC chu trình chỉ kiểm ở cạnh lùi JMP và CALL: mọi giá trị sống khi
           đó đều nằm trên stack/frame/global) */
        if (tvm_dbg_on) {
            SAVE_PC();
            if (!dbg_hook(vm)) return -1;
            LOAD_PC();
        }
        if (pc >= n) { SAVE_PC(); die(vm, "pc overrun"); }
        op = c[pc++];
        SAVE_PC();
#ifdef TVM_PROF
        prof_hit(vm, op);
#endif
#ifdef TVM_TRACE
        fprintf(stderr, "[pc=%zu op=0x%02x sp=%zu fr=%zu nh=%zu]\n",
                pc - 1, op, vm->sp, vm->nframes, vm->nhandlers);
#endif
        switch (op) {
        case OP_PUSH_NIL: push(vm, mk_nil()); break;
        case OP_PUSH_BOOL: { TVal v = mk_bool(c[pc++]); push(vm, v); break; }
        case OP_PUSH_INT: push(vm, mk_int(rd_i64(c, &pc))); break;
        case OP_PUSH_FLOAT: push(vm, mk_float(rd_f64(c, &pc))); break;
        case OP_PUSH_STR: {
            TAttrIC *ic = &vm->prog->ic[pc - 1];
            uint32_t len = rd_u32(c, &pc);
            if (pc + len > n) die(vm, "bad str");
            if (ic->kind == 4) {
                TVal v = *(const TVal*)ic->key;
                (*v.u.str.refc)++;
                pc += len;
                push(vm, v); break;
            }
            {
                TVal v = mk_str_n((char*)(c + pc), len);
                TVal *keep = (TVal*)malloc(sizeof(TVal));
                *keep = v; (*v.u.str.refc)++;          /* cache giữ 1 ref */
                ic->kind = 4; ic->key = keep;
                pc += len;
                push(vm, v); break;
            }
        }
        case OP_NEW_LIST: {
            uint16_t k = rd_u16(c, &pc);
            if (vm->sp < k) die(vm, "stack underflow");
            TList *l = list_new(k);
            for (uint16_t i = 0; i < k; i++) l->items[i] = vm->stack[vm->sp - k + i];
            l->len = k;
            vm->sp -= k;
            TVal v; v.t = T_LIST; v.u.list = l;
            push(vm, v); break;
        }
        case OP_NEW_TENSOR: {
            uint16_t k = rd_u16(c, &pc);
            if (vm->sp < k) die(vm, "stack underflow");
            TTensor *t = ten_alloc1(k);
            for (uint16_t i = 0; i < k; i++) {
                int ok = 1;
                double x = to_num(&vm->stack[vm->sp - k + i], &ok);
                if (!ok) {
                    for (uint16_t j = 0; j < i; j++) {}
                    free(t->d); free(t); die(vm, "tensor() numbers");
                }
                t->d[i] = x;
            }
            for (uint16_t i = 0; i < k; i++) tval_free(&vm->stack[vm->sp - k + i]);
            vm->sp -= k;
            TVal v; v.t = T_TENSOR; v.u.ten = t;
            push(vm, v); break;
        }
        case OP_NEW_DICT: {
            uint16_t k = rd_u16(c, &pc);
            if (vm->sp < (size_t)k * 2) die(vm, "stack underflow");
            TDict *d = dict_new();
            uint16_t i;
            for (i = 0; i < k; i++) {
                TVal kk = vm->stack[vm->sp - k * 2 + i * 2];
                if (kk.t != T_INT && kk.t != T_FLOAT && kk.t != T_STR &&
                    kk.t != T_BOOL && kk.t != T_NIL) {
                    char eb[96];
                    unhashable_msg(&kk, eb, sizeof(eb));
                    TVal dv; dv.t = T_DICT; dv.u.dict = d;
                    tval_free(&dv);
                    for (i = 0; i < k * 2; i++)
                        tval_free(&vm->stack[vm->sp - k * 2 + i]);
                    vm->sp -= k * 2;
                    die(vm, eb);
                }
            }
            for (i = 0; i < k; i++) {
                TVal k2 = vm->stack[vm->sp - k * 2 + i * 2];
                TVal v2 = vm->stack[vm->sp - k * 2 + i * 2 + 1];
                tval_retain(&k2); tval_retain(&v2);
                dict_set(d, k2, v2);
            }
            for (i = 0; i < k * 2; i++)
                tval_free(&vm->stack[vm->sp - k * 2 + i]);
            vm->sp -= k * 2;
            TVal v; v.t = T_DICT; v.u.dict = d;
            push(vm, v); break;
        }
        case OP_ADD: {
            TVal *b = &vm->stack[vm->sp - 1], *a = b - 1;
            if (a->t == T_INT && b->t == T_INT) {
                a->u.i = (int64_t)((uint64_t)a->u.i + (uint64_t)b->u.i); vm->sp--; break;
            }
            if (a->t == T_FLOAT && b->t == T_FLOAT) { a->u.f += b->u.f; vm->sp--; break; }
            arith(vm, op); RELOAD_CODE(); LOAD_PC(); break;
        }
        case OP_SUB: {
            TVal *b = &vm->stack[vm->sp - 1], *a = b - 1;
            if (a->t == T_INT && b->t == T_INT) {
                a->u.i = (int64_t)((uint64_t)a->u.i - (uint64_t)b->u.i); vm->sp--; break;
            }
            if (a->t == T_FLOAT && b->t == T_FLOAT) { a->u.f -= b->u.f; vm->sp--; break; }
            arith(vm, op); RELOAD_CODE(); LOAD_PC(); break;
        }
        case OP_MUL: {
            TVal *b = &vm->stack[vm->sp - 1], *a = b - 1;
            if (a->t == T_INT && b->t == T_INT) {
                a->u.i = (int64_t)((uint64_t)a->u.i * (uint64_t)b->u.i); vm->sp--; break;
            }
            if (a->t == T_FLOAT && b->t == T_FLOAT) { a->u.f *= b->u.f; vm->sp--; break; }
            arith(vm, op); RELOAD_CODE(); LOAD_PC(); break;
        }
        case OP_MOD: {
            TVal *b = &vm->stack[vm->sp - 1], *a = b - 1;
            if (a->t == T_INT && b->t == T_INT && b->u.i != 0) {
                a->u.i = imod(a->u.i, b->u.i); vm->sp--; break;
            }
            arith(vm, op); RELOAD_CODE(); LOAD_PC(); break;
        }
        case OP_DIV: case OP_POW: case OP_TDOT:
            arith(vm, op); RELOAD_CODE(); LOAD_PC(); break;
        case OP_NEG: {
            TVal a = pop(vm);
            if (a.t == T_INT) { int64_t i = a.u.i; tval_free(&a); push(vm, mk_int(-i)); }
            else if (a.t == T_FLOAT) { double x = a.u.f; tval_free(&a); push(vm, mk_float(-x)); }
            else if (a.t == T_TENSOR) {
                TTensor *t = ten_alloc(a.u.ten->ndim, a.u.ten->shape);
                ten_sync(a.u.ten);
                for (size_t i = 0; i < t->len; i++) t->d[i] = -a.u.ten->d[i];
                tval_free(&a);
                push(vm, ten_val(t));
            }
            else if (a.t == T_DICT && a.u.dict->kind == DK_REC) {
                TVal r = mk_nil();
                if (rec_call_dunder(vm, &a, "__neg__", NULL, NULL, 0, &r)) {
                    tval_free(&a);
                    push(vm, r);
                    RELOAD_CODE(); LOAD_PC();
                } else { tval_free(&a); die(vm, "neg here"); }
            }
            else { tval_free(&a); die(vm, "neg here"); }
            break;
        }
        case OP_NOT: {
            TVal a = pop(vm);
            int t = !tval_truthy(&a);
            tval_free(&a);
            push(vm, mk_bool(t)); break;
        }
        case OP_EQ: case OP_NE: case OP_LT: case OP_GT: case OP_LE: case OP_GE: {
            TVal *b = &vm->stack[vm->sp - 1], *a = b - 1;
            int r;
            if (a->t == T_INT && b->t == T_INT) {
                int64_t x = a->u.i, y = b->u.i;
                r = op == OP_LT ? x < y : op == OP_GT ? x > y : op == OP_LE ? x <= y :
                    op == OP_GE ? x >= y : op == OP_EQ ? x == y : x != y;
            } else if (a->t == T_FLOAT && b->t == T_FLOAT) {
                double x = a->u.f, y = b->u.f;
                r = op == OP_LT ? x < y : op == OP_GT ? x > y : op == OP_LE ? x <= y :
                    op == OP_GE ? x >= y : op == OP_EQ ? x == y : x != y;
            } else if (a->t == T_STR && b->t == T_STR && (op == OP_EQ || op == OP_NE)) {
                int eq = a->u.str.refc == b->u.str.refc ||
                         (a->u.str.n == b->u.str.n && memcmp(a->u.str.s, b->u.str.s, a->u.str.n) == 0);
                r = op == OP_EQ ? eq : !eq;
                tval_free(a); tval_free(b);
            } else { cmpop(vm, op); RELOAD_CODE(); LOAD_PC(); break; }
            a->t = T_BOOL; a->u.b = r; vm->sp--;
            break;
        }
        case OP_CONTAINS: {
            TVal nd = pop(vm), hay = pop(vm);
            if (hay.t != T_STR || nd.t != T_STR) {
                tval_free(&hay); tval_free(&nd); die(vm, "contains strs");
            }
            int found = 0;
            if (nd.u.str.n == 0) found = 1;
            else {
                for (size_t i = 0; i + nd.u.str.n <= hay.u.str.n; i++) {
                    if (memcmp(hay.u.str.s + i, nd.u.str.s, nd.u.str.n) == 0) {
                        found = 1; break;
                    }
                }
            }
            tval_free(&hay); tval_free(&nd);
            push(vm, mk_bool(found)); break;
        }
        case OP_MATCHGLOB: {
            TVal nd = pop(vm), hay = pop(vm);
            if (hay.t != T_STR || nd.t != T_STR) {
                tval_free(&hay); tval_free(&nd); die(vm, "match strs");
            }
            int m = glob_match(hay.u.str.s, hay.u.str.n,
                               nd.u.str.s, nd.u.str.n);
            tval_free(&hay); tval_free(&nd);
            push(vm, mk_bool(m)); break;
        }
        case OP_IDX: {
            {
                TVal *ip = &vm->stack[vm->sp - 1], *cp = ip - 1;
                if (cp->t == T_LIST && ip->t == T_INT && ip->u.i >= 0 &&
                    (uint64_t)ip->u.i < cp->u.list->len) {
                    TList *l = cp->u.list;
                    TVal r = l->items[ip->u.i];
                    tval_retain(&r);
                    if (--l->refc == 0) { l->refc = 1; tval_free(cp); }
                    vm->sp -= 2;
                    push(vm, r);
                    break;
                }
            }
            TVal ix = pop(vm), cv = pop(vm);
            TVal r; r.t = T_NIL;
            if (cv.t == T_DICT && (cv.u.dict->kind == DK_SET || cv.u.dict->kind == DK_TYPE)) {
                tval_free(&ix); tval_free(&cv); die(vm, "index here");
            }
            if (cv.t == T_TENSOR && ix.t == T_LIST) {      /* t[i, j] / t[:, j] */
                r = ten_get_multi(vm, cv.u.ten, ix.u.list);
                tval_free(&ix); tval_free(&cv);
                push(vm, r); break;
            }
            if (cv.t == T_DICT && cv.u.dict->kind == DK_REC) {
                int i = dict_find(cv.u.dict, &ix);
                if (i >= 0) {
                    mod_pull(cv.u.dict, (size_t)i);
                    r = cv.u.dict->vals[i]; tval_retain(&r);
                } else if (rec_call_dunder(vm, &cv, "__getitem__", &ix, NULL, 1, &r)) {
                    RELOAD_CODE(); LOAD_PC();
                } else {
                    char eb[128]; Buf b = {0,0,0};
                    fmt_into(&b, &ix);
                    snprintf(eb, sizeof(eb), "%.40s has no field %.40s",
                             rec_name(cv.u.dict), b.s ? b.s : "?");
                    free(b.s);
                    tval_free(&ix); tval_free(&cv); die(vm, eb);
                }
            } else if (cv.t == T_DICT) {
                if (!hashable(&ix)) {
                    char eb[96];
                    unhashable_msg(&ix, eb, sizeof(eb));
                    tval_free(&ix); tval_free(&cv); die(vm, eb);
                }
                int i = dict_find(cv.u.dict, &ix);
                if (i < 0) {
                    char eb[128];
                    Buf b = {0,0,0};
                    fmt_into(&b, &ix);
                    snprintf(eb, sizeof(eb), "key %.60s missing", b.s ? b.s : "?");
                    free(b.s);
                    tval_free(&ix); tval_free(&cv); die(vm, eb);
                }
                mod_pull(cv.u.dict, (size_t)i);
                r = cv.u.dict->vals[i]; tval_retain(&r);
            } else {
                int ok = 1; double xd = to_num(&ix, &ok);
                (void)xd;
                if (!ok || ix.t == T_FLOAT) {
                    tval_free(&ix); tval_free(&cv); die(vm, "index int");
                }
                int64_t i = ix.u.i;
                if (cv.t == T_LIST) {
                    if (i < 0 || i >= (int64_t)cv.u.list->len) {
                        char eb[96];
                        snprintf(eb, sizeof(eb), "index %lld out of range (len %zu)",
                                 (long long)i, cv.u.list->len);
                        tval_free(&ix); tval_free(&cv); die(vm, eb);
                    }
                    r = cv.u.list->items[(size_t)i]; tval_retain(&r);
                } else if (cv.t == T_STR) {
                    if (i < 0 || i >= (int64_t)cv.u.str.n) {
                        char eb[96];
                        snprintf(eb, sizeof(eb), "index %lld out of range (len %zu)",
                                 (long long)i, cv.u.str.n);
                        tval_free(&ix); tval_free(&cv); die(vm, eb);
                    }
                    r = mk_str_n(cv.u.str.s + i, 1);
                } else if (cv.t == T_TENSOR) {
                    r = ten_getitem(vm, cv.u.ten, i);
                } else {
                    tval_free(&ix); tval_free(&cv); die(vm, "index here");
                }
            }
            tval_free(&ix); tval_free(&cv);
            push(vm, r); break;
        }
        case OP_SETIDX: {
            {
                TVal *vp = &vm->stack[vm->sp - 1], *ip = vp - 1, *cp = vp - 2;
                if (cp->t == T_LIST && ip->t == T_INT && ip->u.i >= 0 &&
                    (uint64_t)ip->u.i < cp->u.list->len) {
                    TList *l = cp->u.list;
                    TVal old = l->items[ip->u.i];
                    l->items[ip->u.i] = *vp;          /* nhận ownership v */
                    tval_free(&old);
                    vm->sp -= 3;
                    if (--l->refc == 0) { l->refc = 1; TVal lv; lv.t = T_LIST; lv.u.list = l; tval_free(&lv); }
                    break;
                }
            }
            TVal v = pop(vm), ix = pop(vm), cv = pop(vm);
            if (cv.t == T_LIST) {
                if (ix.t != T_INT) { tval_free(&v); tval_free(&ix); tval_free(&cv); die(vm, "index int"); }
                if (ix.u.i < 0 || ix.u.i >= (int64_t)cv.u.list->len) {
                    tval_free(&v); tval_free(&ix); tval_free(&cv); die(vm, "assign range");
                }
                tval_free(&cv.u.list->items[(size_t)ix.u.i]);
                cv.u.list->items[(size_t)ix.u.i] = v;
            } else if (cv.t == T_TENSOR && ix.t == T_LIST) {    /* t[i, j] = v */
                char eb[64];
                if (v.t != T_INT && v.t != T_FLOAT && v.t != T_TENSOR) {
                    tval_free(&v); tval_free(&ix); tval_free(&cv); die(vm, "tensor assign");
                }
                if (!ten_set_multi(vm, cv.u.ten, ix.u.list, &v, eb, sizeof(eb))) {
                    tval_free(&v); tval_free(&ix); tval_free(&cv); die(vm, eb);
                }
                tval_free(&v);
            } else if (cv.t == T_TENSOR) {
                char eb[64];
                if (ix.t != T_INT) {
                    tval_free(&v); tval_free(&ix); tval_free(&cv); die(vm, "index int");
                }
                if (v.t != T_INT && v.t != T_FLOAT && v.t != T_TENSOR) {
                    tval_free(&v); tval_free(&ix); tval_free(&cv); die(vm, "tensor assign");
                }
                if (!ten_setitem(cv.u.ten, ix.u.i, &v, eb, sizeof(eb))) {
                    tval_free(&v); tval_free(&ix); tval_free(&cv); die(vm, eb);
                }
                tval_free(&v);
            } else if (cv.t == T_DICT && cv.u.dict->kind == DK_REC) {
                int i = dict_find(cv.u.dict, &ix);
                if (i >= 0) {
                    char eb[128];
                    if (mod_ro(cv.u.dict, (size_t)i, eb, sizeof(eb))) {
                        tval_free(&v); tval_free(&ix); tval_free(&cv); die(vm, eb);
                    }
                    tval_free(&cv.u.dict->vals[i]);
                    cv.u.dict->vals[i] = v;
                    mod_push(cv.u.dict, (size_t)i);
                } else {
                    TVal out = mk_nil();
                    if (rec_call_dunder(vm, &cv, "__setitem__", &ix, &v, 2, &out)) {
                        tval_free(&out); tval_free(&v);
                        RELOAD_CODE(); LOAD_PC();
                    } else {
                        char eb[128]; Buf b = {0,0,0};
                        fmt_into(&b, &ix);
                        snprintf(eb, sizeof(eb), "%.40s has no field %.40s",
                                 rec_name(cv.u.dict), b.s ? b.s : "?");
                        free(b.s);
                        tval_free(&v); tval_free(&ix); tval_free(&cv); die(vm, eb);
                    }
                }
            } else if (cv.t == T_DICT && cv.u.dict->kind == DK_DICT) {
                if (!hashable(&ix)) {
                    char eb[96];
                    unhashable_msg(&ix, eb, sizeof(eb));
                    tval_free(&v); tval_free(&ix); tval_free(&cv); die(vm, eb);
                }
                dict_set(cv.u.dict, ix, v);  /* nhận ownership cả key lẫn val */
                tval_free(&cv);
                break;
            } else {
                tval_free(&v); tval_free(&ix); tval_free(&cv); die(vm, "assign here");
            }
            tval_free(&ix); tval_free(&cv);
            break;
        }
        case OP_SLICE: {
            TVal st = pop(vm), hi = pop(vm), lo = pop(vm), cv = pop(vm);
            int64_t n = 0;
            if (cv.t == T_STR) n = (int64_t)cv.u.str.n;
            else if (cv.t == T_LIST) n = (int64_t)cv.u.list->len;
            else if (cv.t == T_TENSOR) n = (int64_t)ten_len0(cv.u.ten);
            else { tval_free(&st); tval_free(&hi); tval_free(&lo); tval_free(&cv); die(vm, "slice sequence"); }
            int64_t a = 0, b = n, s = 1;
            int has_s = st.t != T_NIL;
            if (lo.t != T_NIL) {
                if (lo.t != T_INT) { tval_free(&st); tval_free(&hi); tval_free(&lo); tval_free(&cv); die(vm, "slice int"); }
                a = lo.u.i;
            }
            if (hi.t != T_NIL) {
                if (hi.t != T_INT) { tval_free(&st); tval_free(&hi); tval_free(&lo); tval_free(&cv); die(vm, "slice int"); }
                b = hi.u.i;
            }
            if (has_s) {
                if (st.t != T_INT) { tval_free(&st); tval_free(&hi); tval_free(&lo); tval_free(&cv); die(vm, "slice int"); }
                s = st.u.i;
                if (s == 0) { tval_free(&st); tval_free(&hi); tval_free(&lo); tval_free(&cv); die(vm, "slice step 0"); }
            }
            /* khớp Python: cận bỏ trống phụ thuộc dấu bước (bước âm: từ n-1 tới
               "trước 0"); cận âm do user cho mới cộng n */
            if (lo.t == T_NIL) a = s > 0 ? 0 : n - 1;
            else if (a < 0) a += n;
            if (hi.t == T_NIL) b = s > 0 ? n : -1;
            else if (b < 0) b += n;
            if (s > 0) {
                if (a < 0) a = 0;
                if (a > n) a = n;
                if (b < 0) b = 0;
                if (b > n) b = n;
            } else {
                if (a < -1) a = -1;
                if (a >= n) a = n - 1;
                if (b < -1) b = -1;
                if (b >= n) b = n - 1;
            }
            TVal r; r.t = T_NIL;
            if (cv.t == T_STR && s == 1) {          /* bước 1: copy thẳng (chuỗi 1 byte dùng chung) */
                r = mk_str_n(cv.u.str.s + (b > a ? a : 0), b > a ? (size_t)(b - a) : 0);
            } else if (cv.t == T_STR) {
                Buf bb = {0,0,0};
                if (s > 0) {
                    for (int64_t i = a; i < b; i += s) buf_c(&bb, cv.u.str.s[i]);
                } else {
                    for (int64_t i = a; i > b; i += s) buf_c(&bb, cv.u.str.s[i]);
                }
                r = mk_str_n(bb.s ? bb.s : "", bb.n);
                free(bb.s);
            } else if (cv.t == T_LIST) {
                TList *l = list_new(4);
                if (s > 0) {
                    for (int64_t i = a; i < b; i += s) {
                        TVal e = cv.u.list->items[(size_t)i]; tval_retain(&e);
                        list_push(l, e);
                    }
                } else {
                    for (int64_t i = a; i > b; i += s) {
                        TVal e = cv.u.list->items[(size_t)i]; tval_retain(&e);
                        list_push(l, e);
                    }
                }
                r.t = T_LIST; r.u.list = l;
            } else {
                /* tensor: cắt theo trục 0 (hàng a, a+s, ...) -> view */
                size_t cnt = 0;
                if (s > 0) { for (int64_t i = a; i < b; i += s) cnt++; }
                else { for (int64_t i = a; i > b; i += s) cnt++; }
                r = ten_val(ten_rows(cv.u.ten, a, s, cnt));
            }
            tval_free(&st); tval_free(&hi); tval_free(&lo); tval_free(&cv);
            push(vm, r); break;
        }
        case OP_LEN: {
            TVal a = pop(vm);
            size_t k;
            if (a.t == T_STR) k = a.u.str.n;
            else if (a.t == T_LIST) k = a.u.list->len;
            else if (a.t == T_TENSOR) k = ten_len0(a.u.ten);
            else if (a.t == T_DICT && a.u.dict->kind <= DK_SET) k = a.u.dict->len;
            else if (a.t == T_DICT && a.u.dict->kind == DK_REC) {
                TVal r = mk_nil();
                if (rec_call_dunder(vm, &a, "__len__", NULL, NULL, 0, &r)) {
                    tval_free(&a);
                    push(vm, r);
                    RELOAD_CODE(); LOAD_PC();
                    break;
                }
                tval_free(&a); die(vm, "len() here");
            }
            else { tval_free(&a); die(vm, "len() here"); }
            tval_free(&a);
            push(vm, mk_int((int64_t)k)); break;
        }
        case OP_PRINT: {
            uint8_t k = c[pc++];
            if (vm->sp < k) die(vm, "stack underflow");
            for (uint8_t i = 0; i < k; i++) {
                if (i) putchar(' ');
                tval_print(&vm->stack[vm->sp - k + i]);
            }
            putchar('\n');
            fflush(stdout);
            for (uint8_t i = 0; i < k; i++) tval_free(&vm->stack[vm->sp - k + i]);
            vm->sp -= k;
            break;
        }
        case OP_JMP: {
            size_t jpc = pc - 1;
            int32_t rel = rd_i32(c, &pc);
            pc = (size_t)((int64_t)pc + rel);
            if (rel < 0) {
                if (vm->root->gc_allocs >= GC_THRESHOLD) gc_collect(vm);
                /* vòng lặp nóng ở top-level: JIT vùng vòng lặp (global ghi thẳng) */
                if (vm->nframes == 1 && vm->stop_nf == 0) {
                    TAttrIC *ic = &vm->prog->ic[jpc];
                    if (ic->kind == 0) { ic->kind = 6; ic->idx = 0; }
                    if (ic->kind == 6 && ic->idx >= 0 && ++ic->idx >= JIT_OSR_LOOPS) {
                        int jr;
                        SAVE_PC();
                        jr = t_jit_osr_main(vm, pc, jpc + 5);
                        LOAD_PC();
                        if (jr < 0) ic->idx = -1;            /* không JIT được: thôi */
                        else if (jr == 0) ic->idx = JIT_OSR_LOOPS / 2;
                        else ic->idx = 0;                    /* đã chạy; lần sau đếm lại */
                    }
                }
                /* vòng lặp nóng trong hàm: OSR sang code JIT */
                if (vm->nframes > 1) {
                    TFrame *fr = &vm->frames[vm->nframes - 1];
                    if (fr->fn && !(fr->fn->jit_off & 2) && ++fr->fn->loops >= JIT_OSR_LOOPS) {
                        TVal rv;
                        int jr;
                        SAVE_PC();
                        jr = t_jit_osr(vm, fr->fn, fr->locals, pc, &rv);
                        if (jr == 1) {
                            do_return(vm, rv);
                            if (vm->nframes <= vm->stop_nf) return 0;
                            RELOAD_CODE();
                        }
                        LOAD_PC();
                        /* jr == 2: deopt, pc + stack đã đặt lại -> chạy tiếp */
                    }
                }
            }
            break;
        }
        case OP_JMPF: case OP_JMPT: {
            int is_t = (op == OP_JMPT);
            int32_t rel = rd_i32(c, &pc);
            TVal *tv = &vm->stack[vm->sp - 1];
            int t;
            if (tv->t == T_BOOL) { t = tv->u.b; vm->sp--; }
            else {
                TVal v = pop(vm);
                t = tval_truthy(&v);
                tval_free(&v);
            }
            if ((t && is_t) || (!t && !is_t))
                pc = (size_t)((int64_t)pc + rel);
            break;
        }
        case OP_CALL: {
            uint8_t k = c[pc++];
            if (vm->sp < (size_t)k + 1) die(vm, "stack underflow");
            if (vm->root->gc_allocs >= GC_THRESHOLD) gc_collect(vm);
            SAVE_PC();
            call_value(vm, k);
            RELOAD_CODE();
            LOAD_PC();
            break;
        }
        case OP_CALLM: {
            size_t at = pc - 1;
            uint8_t k = c[pc++];
            uint32_t len = rd_u32(c, &pc);
            const char *nm = (const char*)(c + pc);
            pc += len;
            if (vm->sp < (size_t)k + 1) die(vm, "stack underflow");
            if (vm->root->gc_allocs >= GC_THRESHOLD) gc_collect(vm);
            SAVE_PC();
            call_method(vm, k, nm, len, &vm->prog->ic[at]);
            RELOAD_CODE();
            LOAD_PC();
            break;
        }
        case OP_SETATTR: {
            size_t at = pc - 1;
            uint32_t len = rd_u32(c, &pc);
            const char *nm = (const char*)(c + pc);
            pc += len;
            set_attr(vm, nm, len, &vm->prog->ic[at]);
            break;
        }
        case OP_RET: {
            if (vm->nframes <= 1) die(vm, "ret outside");
            TVal rv = pop(vm);
            do_return(vm, rv);
            if (vm->nframes <= vm->stop_nf) return 0;  /* hết nested call */
            RELOAD_CODE();
            LOAD_PC();
            break;
        }
        case OP_LOAD: {
            uint16_t s = rd_u16(c, &pc);
            TFrame *fr = &vm->frames[vm->nframes - 1];
            if (s < fr->nlocals) {
                TVal v = fr->locals[s];
                if (v.t <= T_FLOAT) { push(vm, v); break; }   /* nil/bool/int/float: không refcount */
            }
            push(vm, slot_get(vm, 0, s));
            break;
        }
        case OP_STORE: case OP_DECL: case OP_BIND: {
            uint16_t s = rd_u16(c, &pc);
            TFrame *fr = &vm->frames[vm->nframes - 1];
            TVal v = pop(vm);
            if (s < fr->nlocals && fr->locals[s].t <= T_FLOAT) { fr->locals[s] = v; break; }
            if (op == OP_BIND) slot_replace(vm, 0, s, v);
            else slot_set(vm, 0, s, v);
            break;
        }
        case OP_LOADG: {
            uint16_t s = rd_u16(c, &pc);
            if (s < vm->prog->nglobals) {
                TVal v = vm->prog->globals[s];
                if (v.t <= T_FLOAT) { push(vm, v); break; }
            }
            push(vm, slot_get(vm, 1, s));
            break;
        }
        case OP_STOREG: case OP_DECLG: {
            uint16_t s = rd_u16(c, &pc);
            TVal v = pop(vm);
            if (s < vm->prog->nglobals && vm->prog->globals[s].t <= T_FLOAT) {
                vm->prog->globals[s] = v; break;
            }
            slot_set(vm, 1, s, v);
            break;
        }
        case OP_MKFN: {
            int32_t e = rd_i32(c, &pc);
            uint8_t na = c[pc++];
            uint16_t nl = rd_u16(c, &pc);
            uint16_t nu = rd_u16(c, &pc);
            TFn *f = (TFn*)malloc(sizeof(TFn));
            f->refc = 1; f->gc_mark = 0; f->entry = e;
            if ((na & 0xF0) == 0xF0) {
                f->nargs = na & 0x0F;
                f->min_args = f->nargs > 0 ? (uint8_t)(f->nargs - 1) : 0;
                f->vararg = 1;
            } else if (na & 0x80) {
                f->nargs = na & 0x0F;
                uint8_t ndef = (na >> 4) & 0x07;
                f->min_args = (ndef <= f->nargs) ? (uint8_t)(f->nargs - ndef) : 0;
                f->vararg = 0;
            } else {
                f->nargs = na;
                f->min_args = na;
                f->vararg = 0;
            }
            f->nlocals = nl;
            f->nupvals = nu; f->ncap = 0;
            f->upvals = nu ? (TVal*)calloc(nu, sizeof(TVal)) : NULL;
            f->ncalls = 0; f->native = NULL; f->loops = 0;
            f->jit_off = f->vararg ? 3 : 0;
            f->jit = NULL;
            f->prog = vm->prog;
            TVal v; v.t = T_FN; v.u.fn = f;
            gc_track(vm, T_FN, f);
            push(vm, v); break;
        }
        case OP_CAPLOCAL: {
            uint16_t s = rd_u16(c, &pc);
            TFrame *fr = &vm->frames[vm->nframes - 1];
            if (vm->sp < 1 || vm->stack[vm->sp - 1].t != T_FN)
                die(vm, "cap fn");
            if (s >= fr->nlocals) die(vm, "cap slot");
            TFn *f = vm->stack[vm->sp - 1].u.fn;
            if (f->ncap >= f->nupvals) die(vm, "cap overflow");
            TVal *slot = &fr->locals[s];
            TCell *cell;
            if (slot->t == T_CELL) {
                cell = slot->u.cell;
            } else {
                cell = (TCell*)pk_alloc(sizeof(TCell));
                cell->refc = 0;             /* 1 cho slot + 1 cho fn (dưới) */
                cell->val = *slot;          /* MOVE ownership từ slot */
                slot->t = T_CELL; slot->u.cell = cell;
                cell->refc = 1;             /* slot giữ */
            }
            {
                TVal cv; cv.t = T_CELL; cv.u.cell = cell;
                tval_retain(&cv);           /* fn giữ */
                f->upvals[f->ncap++] = cv;
            }
            break;
        }
        case OP_CAPUP: {
            uint16_t i = rd_u16(c, &pc);
            TFrame *fr = &vm->frames[vm->nframes - 1];
            if (!fr->fn) die(vm, "capup outside");
            if (vm->sp < 1 || vm->stack[vm->sp - 1].t != T_FN)
                die(vm, "cap fn");
            TFn *f = vm->stack[vm->sp - 1].u.fn;
            if (i >= fr->fn->ncap) die(vm, "capup range");
            if (f->ncap >= f->nupvals) die(vm, "cap overflow");
            {
                TVal cv = fr->fn->upvals[i];
                tval_retain(&cv);
                f->upvals[f->ncap++] = cv;
            }
            break;
        }
        case OP_CAPG: {
            uint16_t s = rd_u16(c, &pc);
            if (s >= vm->prog->nglobals) die(vm, "bad global");
            if (vm->sp < 1 || vm->stack[vm->sp - 1].t != T_FN)
                die(vm, "cap fn");
            TFn *f = vm->stack[vm->sp - 1].u.fn;
            if (f->ncap >= f->nupvals) die(vm, "cap overflow");
            TVal *gp = &vm->prog->globals[s];
            TCell *cell;
            if (gp->t == T_CELL) {
                cell = gp->u.cell;
            } else {
                cell = (TCell*)pk_alloc(sizeof(TCell));
                cell->refc = 1;
                cell->val = *gp;   /* MOVE ownership từ global */
                gp->t = T_CELL; gp->u.cell = cell;
            }
            {
                TVal cv; cv.t = T_CELL; cv.u.cell = cell;
                tval_retain(&cv);
                f->upvals[f->ncap++] = cv;
            }
            break;
        }
        case OP_BINDG: {
            uint16_t s = rd_u16(c, &pc);
            if (s >= vm->prog->nglobals) die(vm, "bad global");
            TVal v = pop(vm);
            tval_free(&vm->prog->globals[s]);
            vm->prog->globals[s] = v;
            break;
        }
        case OP_LOADU: case OP_STOREU: {
            uint16_t i = rd_u16(c, &pc);
            TFrame *fr = &vm->frames[vm->nframes - 1];
            if (!fr->fn || i >= fr->fn->ncap) die(vm, "upval range");
            TVal *cell = &fr->fn->upvals[i];
            if (cell->t != T_CELL) die(vm, "upval cell");
            if (op == OP_LOADU) {
                TVal v = cell->u.cell->val; tval_retain(&v);
                if (v.t == T_UNDEF) { tval_free(&v); die(vm, "undef variable"); }
                push(vm, v);
            } else {
                TVal v = pop(vm);
                tval_free(&cell->u.cell->val);
                cell->u.cell->val = v;
            }
            break;
        }
        case OP_RANGE: {
            uint8_t k = c[pc++];
            TVal b = mk_nil(), a = mk_nil();
            if (k == 2) { b = pop(vm); a = pop(vm); }
            else if (k == 1) { a = pop(vm); }
            else { tval_free(&b); tval_free(&a); die(vm, "arity"); }
            int64_t lo = 0, hi = 0;
            if (k == 1) {
                if (a.t != T_INT) { tval_free(&a); die(vm, "range() int"); }
                hi = a.u.i;
            } else {
                if (a.t != T_INT || b.t != T_INT) {
                    tval_free(&a); tval_free(&b); die(vm, "range() int");
                }
                lo = a.u.i; hi = b.u.i;
            }
            tval_free(&a); tval_free(&b);
            TList *l = list_new(hi > lo ? (size_t)(hi - lo) : 0);
            for (int64_t i = lo; i < hi; i++) list_push(l, mk_int(i));
            TVal v; v.t = T_LIST; v.u.list = l;
            push(vm, v); break;
        }
        case OP_ATTR: {
            uint32_t len = rd_u32(c, &pc);
            if (pc + len > n) die(vm, "bad attr");
            const char *nm = (const char*)(c + pc);
            pc += len;
            {
                /* đường nhanh: field struct trúng inline cache (không phải module) */
                TAttrIC *ic = &vm->prog->ic[pc - len - 5];
                TVal *ov = &vm->stack[vm->sp - 1];
                if (ic->kind == 1 && ov->t == T_DICT) {
                    TDict *d = ov->u.dict;
                    if (d->kind == DK_REC && d->meths && ic->key == d->meths && !d->meths->mod &&
                        (size_t)ic->idx < d->len && d->keys[ic->idx].u.str.n == len &&
                        name_eq(d->keys[ic->idx].u.str.s, nm, len)) {
                        TVal r = d->vals[ic->idx];
                        TV_RETAIN(&r);
                        if (d->refc > 1) d->refc--;          /* còn ref khác: chỉ giảm */
                        else tval_free(ov);
                        *ov = r;
                        break;
                    }
                }
                do_attr(vm, nm, len, ic);
            }
            break;
        }
        case OP_RAISE: {
            TVal v = pop(vm);
            char eb[256];
            tval_fmt(&v, eb, sizeof(eb));
            tval_free(&v);
            die(vm, eb);
            break;
        }
        case OP_TRY_PUSH: {
            int32_t rel = rd_i32(c, &pc);
            if (vm->nhandlers >= vm->hcap) {
                vm->hcap *= 2;
                vm->handlers = (THandler*)realloc(vm->handlers,
                    vm->hcap * sizeof(THandler));
                if (!vm->handlers) exit(1);
            }
            THandler *h = &vm->handlers[vm->nhandlers++];
            h->sp = vm->sp; h->nframes = vm->nframes;
            h->pc = (size_t)((int64_t)pc + rel);
            h->stop_nf = vm->stop_nf;
            h->prog = vm->prog;
            if (setjmp(h->jb)) {
                TVal v = mk_str(vm->errmsg);
                RELOAD_CODE();
                LOAD_PC();
                push(vm, v);
            }
            break;
        }
        case OP_TRY_POP:
            if (vm->nhandlers == 0) die(vm, "try pop");
            vm->nhandlers--;
            break;
        case OP_FOR_PREP: {
            TVal v = pop(vm);
            if (v.t == T_DICT && v.u.dict->kind == DK_REC) {
                TVal r = mk_nil();
                if (rec_call_dunder(vm, &v, "__iter__", NULL, NULL, 0, &r)) {
                    tval_free(&v);
                    RELOAD_CODE(); LOAD_PC();
                    push(vm, r);
                    break;
                }
            }
            if ((v.t == T_DICT && v.u.dict->kind >= DK_REC) ||
                (v.t != T_DICT && v.t != T_LIST && v.t != T_STR && v.t != T_TENSOR)) {
                char eb[96];
                snprintf(eb, sizeof(eb), "for: not iterable %.40s", tname_of(&v));
                tval_free(&v); die(vm, eb);
            }
            if (v.t == T_DICT) {
                TList *l = list_new(v.u.dict->len);
                for (size_t i = 0; i < v.u.dict->len; i++) {
                    TVal k = v.u.dict->keys[i]; tval_retain(&k);
                    list_push(l, k);
                }
                tval_free(&v);
                TVal r; r.t = T_LIST; r.u.list = l;
                push(vm, r);
            } else push(vm, v);
            break;
        }
        case OP_POP: {
            TVal v = pop(vm); tval_free(&v); break;
        }
        case OP_DUP: {
            if (vm->sp < 1) die(vm, "stack underflow");
            TVal v = vm->stack[vm->sp - 1]; tval_retain(&v);
            push(vm, v); break;
        }
        case OP_INCB_L: case OP_INCB_G: {
            uint16_t s = rd_u16(c, &pc);
            while (pc < n && c[pc] == OP_NOP) pc++;   /* đệm peephole */
            int isg = (op == OP_INCB_G);
            {
                TVal *p = isg ? (s < vm->prog->nglobals ? &vm->prog->globals[s] : NULL)
                              : (s < vm->frames[vm->nframes - 1].nlocals
                                 ? &vm->frames[vm->nframes - 1].locals[s] : NULL);
                if (p && p->t == T_INT) { p->u.i = (int64_t)((uint64_t)p->u.i + 1); break; }
            }
            TVal a = slot_get(vm, isg, s);
            push(vm, a);
            push(vm, mk_int(1));
            arith(vm, OP_ADD);
            {
                TVal r = pop(vm);
                slot_replace(vm, isg, s, r);
            }
            break;
        }
        case OP_INC_L: case OP_INC_G: {
            uint16_t s = rd_u16(c, &pc);
            while (pc < n && c[pc] == OP_NOP) pc++;   /* đệm peephole */
            int isg = (op == OP_INC_G);
            {
                TVal *p = isg ? (s < vm->prog->nglobals ? &vm->prog->globals[s] : NULL)
                              : (s < vm->frames[vm->nframes - 1].nlocals
                                 ? &vm->frames[vm->nframes - 1].locals[s] : NULL);
                if (p && p->t == T_INT) { p->u.i = (int64_t)((uint64_t)p->u.i + 1); break; }
            }
            TVal a = slot_get(vm, isg, s);
            push(vm, a);
            push(vm, mk_int(1));
            arith(vm, OP_ADD);
            {
                TVal r = pop(vm);
                slot_set(vm, isg, s, r);
            }
            break;
        }
        case OP_DEC_L: case OP_DEC_G: {
            uint16_t s = rd_u16(c, &pc);
            while (pc < n && c[pc] == OP_NOP) pc++;   /* đệm peephole */
            int isg = (op == OP_DEC_G);
            {
                TVal *p = isg ? (s < vm->prog->nglobals ? &vm->prog->globals[s] : NULL)
                              : (s < vm->frames[vm->nframes - 1].nlocals
                                 ? &vm->frames[vm->nframes - 1].locals[s] : NULL);
                if (p && p->t == T_INT) { p->u.i = (int64_t)((uint64_t)p->u.i - 1); break; }
            }
            TVal a = slot_get(vm, isg, s);
            push(vm, a);
            push(vm, mk_int(1));
            arith(vm, OP_SUB);
            {
                TVal r = pop(vm);
                slot_set(vm, isg, s, r);
            }
            break;
        }
        case OP_ADDI_L: case OP_ADDI_G: {
            uint16_t s = rd_u16(c, &pc);
            int64_t k = rd_i64(c, &pc);
            int isg = (op == OP_ADDI_G);
            while (pc < n && c[pc] == OP_NOP) pc++;   /* đệm peephole */
            {
                TVal *p = isg ? (s < vm->prog->nglobals ? &vm->prog->globals[s] : NULL)
                              : (s < vm->frames[vm->nframes - 1].nlocals
                                 ? &vm->frames[vm->nframes - 1].locals[s] : NULL);
                if (p && p->t == T_INT) { p->u.i = (int64_t)((uint64_t)p->u.i + (uint64_t)k); break; }
            }
            TVal a = slot_get(vm, isg, s);
            push(vm, a);
            push(vm, mk_int(k));
            arith(vm, OP_ADD);
            {
                TVal r = pop(vm);
                slot_set(vm, isg, s, r);
            }
            break;
        }
        case OP_ADDS_L: case OP_ADDS_G: case OP_SUBS_L: case OP_SUBS_G: {
            uint16_t ds, ss;
            int isg, bop;
            TVal a, b, r;
            memcpy(&ds, c + pc, 2); pc += 2;
            memcpy(&ss, c + pc, 2); pc += 2;
            while (pc < n && c[pc] == OP_NOP) pc++;   /* đệm peephole */
            isg = (op == OP_ADDS_G || op == OP_SUBS_G);
            bop = (op == OP_ADDS_L || op == OP_ADDS_G) ? OP_ADD : OP_SUB;
            {
                TVal *base = isg ? vm->prog->globals : vm->frames[vm->nframes - 1].locals;
                size_t lim = isg ? vm->prog->nglobals : vm->frames[vm->nframes - 1].nlocals;
                if (ds < lim && ss < lim && base[ds].t == T_INT && base[ss].t == T_INT) {
                    base[ds].u.i = (int64_t)(bop == OP_ADD ? (uint64_t)base[ds].u.i + (uint64_t)base[ss].u.i
                                                           : (uint64_t)base[ds].u.i - (uint64_t)base[ss].u.i);
                    break;
                }
            }
            a = slot_get(vm, isg, ds);
            b = slot_get(vm, isg, ss);
            push(vm, a);
            push(vm, b);
            arith(vm, bop);
            r = pop(vm);
            slot_set(vm, isg, ds, r);
            break;
        }
        case OP_IN: {
            TVal b = vm->stack[vm->sp - 1], a = vm->stack[vm->sp - 2];
            int res = val_in(vm, &a, &b);
            b = pop(vm); a = pop(vm);
            tval_free(&a); tval_free(&b);
            push(vm, mk_bool(res)); break;
        }
        case OP_BAND: {
            TVal *b = &vm->stack[vm->sp - 1], *a = b - 1;
            if (a->t == T_INT && b->t == T_INT) {
                a->u.i = (int64_t)((uint64_t)a->u.i & (uint64_t)b->u.i); vm->sp--; break;
            }
            die(vm, "bitwise & requires int"); break;
        }
        case OP_BOR: {
            TVal *b = &vm->stack[vm->sp - 1], *a = b - 1;
            if (a->t == T_INT && b->t == T_INT) {
                a->u.i = (int64_t)((uint64_t)a->u.i | (uint64_t)b->u.i); vm->sp--; break;
            }
            die(vm, "bitwise | requires int"); break;
        }
        case OP_BXOR: {
            TVal *b = &vm->stack[vm->sp - 1], *a = b - 1;
            if (a->t == T_INT && b->t == T_INT) {
                a->u.i = (int64_t)((uint64_t)a->u.i ^ (uint64_t)b->u.i); vm->sp--; break;
            }
            die(vm, "bitwise ^ requires int"); break;
        }
        case OP_BNOT: {
            TVal *a = &vm->stack[vm->sp - 1];
            if (a->t == T_INT) {
                a->u.i = (int64_t)(~(uint64_t)a->u.i); break;
            }
            die(vm, "bitwise ~ requires int"); break;
        }
        case OP_SHL: {
            TVal *b = &vm->stack[vm->sp - 1], *a = b - 1;
            if (a->t == T_INT && b->t == T_INT) {
                if (b->u.i < 0) die(vm, "negative shift count");
                a->u.i = b->u.i >= 64 ? 0 : (int64_t)((uint64_t)a->u.i << (unsigned)b->u.i);
                vm->sp--; break;
            }
            die(vm, "bitwise << requires int"); break;
        }
        case OP_SHR: {
            TVal *b = &vm->stack[vm->sp - 1], *a = b - 1;
            if (a->t == T_INT && b->t == T_INT) {
                if (b->u.i < 0) die(vm, "negative shift count");
                if (b->u.i >= 64) a->u.i = a->u.i < 0 ? -1 : 0;
                else a->u.i = a->u.i >> (unsigned)b->u.i;
                vm->sp--; break;
            }
            die(vm, "bitwise >> requires int"); break;
        }
        case OP_NOP:
            /* đệm của superop peephole (INC_L... thay 4 lệnh, phần dư là NOP; không đích
               nhảy nào rơi vào giữa): bỏ qua cả dải trong một lần dispatch */
            while (pc < n && c[pc] == OP_NOP) pc++;
            break;
        case OP_HALT:
            if (vm->frames[vm->nframes - 1].is_mod) return 0;   /* hết top-level module */
            in_top = 0; return 0;
        default: die(vm, "unknown opcode");
        }
    }
}



