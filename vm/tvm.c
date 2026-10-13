/* tvm.c — T bytecode VM v1: stack machine + frames + refcount + try + natives. */
#include "tvm.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/types.h>
#include <sys/stat.h>

/* Vòng lặp interpreter có setjmp/longjmp (try/except). GCC >= 12 (mingw-w64)
   inline hỏng vòng lặp này: try/except im lặng khi build -O1/-O2/-Os
   (đã repro: chỉ -fno-inline là hết). Vì vậy tắt inline cho cả file. */
#if defined(__GNUC__) && !defined(__clang__) && !defined(__INTEL_COMPILER)
#pragma GCC optimize("no-inline")
#endif


#define STACK_INIT 1024
#define FRAME_INIT 64
#define HAND_INIT 16

static jmp_buf top_jb;
static int ten_eq(const TTensor *a, const TTensor *b);
static int in_top;
static void free_frame(TVM *vm);
static TFrame *push_frame(TVM *vm);
static TVM *cur_vm;   /* VM đang chạy (GC track dùng; đơn luồng) */
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

/* ---------- GC mark-sweep (chu trình; refcount xử lý còn lại) ---------- */
#define GC_THRESHOLD_DEFAULT 65536
static size_t gc_threshold = GC_THRESHOLD_DEFAULT;   /* TVM_GCTHRESH=n để thử GC dày */
#define GC_THRESHOLD gc_threshold

static void gc_unlink(TVM *vm, GCNode *nd) {
    (void)vm;
    if (nd->prev) nd->prev->next = nd->next;
    else vm->root->gc_head = nd->next;
    if (nd->next) nd->next->prev = nd->prev;
    free(nd);
    vm->root->gc_tracked--;
}

static void gc_track(TVM *vm, int type, void *ptr) {
    GCNode *nd = (GCNode*)malloc(sizeof(GCNode));
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
    if (type == T_LIST) { TList *l = (TList*)ptr; free(l->items); free(l); }
    else if (type == T_DICT) {
        TDict *d = (TDict*)ptr;
        if (d->kind != DK_REC) free(d->keys);
        free(d->vals); free(d->rname); free(d->idx); free(d);
    } else if (type == T_FN) { TFn *f = (TFn*)ptr; free(f->upvals); free(f); }
    else free(ptr);
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
static void push(TVM *vm, TVal v) {
    if (vm->sp >= vm->cap) {
        vm->cap *= 2;
        vm->stack = (TVal*)realloc(vm->stack, vm->cap * sizeof(TVal));
        if (!vm->stack) exit(1);
    }
    vm->stack[vm->sp++] = v;
}
static TVal pop(TVM *vm) { return vm->stack[--vm->sp]; }
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
    int32_t *blk = (int32_t*)malloc(8 + n + 1);
    if (!blk) exit(1);
    blk[0] = 1; blk[1] = 0;
    v.u.str.refc = blk;
    v.u.str.s = (char*)(blk + 2);
    v.u.str.s[n] = 0;
    v.u.str.n = n;
    return v;
}
static TVal mk_str_n(const char *s, size_t n) {
    TVal v = mk_str_uninit(n);
    if (n) memcpy(v.u.str.s, s, n);
    return v;
}
static TVal mk_str(const char *s) { return mk_str_n(s, strlen(s)); }
static TList *list_new(size_t cap) {
    TList *l = (TList*)malloc(sizeof(TList));
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
static double *tbuf_p[TBUF_SLOTS];
static size_t tbuf_n[TBUF_SLOTS];
static int tbuf_cnt;
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
            free(v->u.str.refc);          /* khối chung (refc + hash + byte) */
        }
        break;
    case T_LIST:
        if (v->u.list && --(v->u.list->refc) == 0) {
            if (cur_vm) gc_untrack(cur_vm, T_LIST, v->u.list);
            for (i = 0; i < v->u.list->len; i++) tval_free(&v->u.list->items[i]);
            free(v->u.list->items); free(v->u.list);
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
            free(dd->vals); free(dd);
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
            free(v->u.nat);
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
            free(v->u.cell);
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
static TProg *rec_mod(const TDict *d) {
    return (d->kind == DK_REC && d->meths) ? d->meths->mod : NULL;
}
static void mod_pull(TDict *d, size_t i) {
    TProg *m = rec_mod(d);
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
static void fmt_double(double x, char *tmp, size_t n) {
    if (x != x) { snprintf(tmp, n, "nan"); return; }
    if (x > 1.7976931348623157e308) { snprintf(tmp, n, "inf"); return; }
    if (x < -1.7976931348623157e308) { snprintf(tmp, n, "-inf"); return; }
    if (x == floor(x) && fabs(x) < 9.2e18) {
        snprintf(tmp, n, "%lld", (long long)x);
        return;
    }
    for (int p = 15; p <= 17; p++) {
        snprintf(tmp, n, "%.*g", p, x);
        if (strtod(tmp, NULL) == x) return;
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
static void fmt_into(Buf *b, const TVal *v) {
    char tmp[64];
    size_t i;
    switch (v->t) {
    case T_NIL: buf_put(b, "nil", 3); break;
    case T_BOOL: buf_put(b, v->u.b ? "true" : "false", v->u.b ? 4 : 5); break;
    case T_INT:
        snprintf(tmp, sizeof(tmp), "%lld", (long long)v->u.i);
        buf_put(b, tmp, strlen(tmp)); break;
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
        if (d->kind == DK_REC) buf_put(b, rec_name(d), strlen(rec_name(d)));
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
    if (v->t == T_INT || v->t == T_FLOAT) {
        char tmp[64];
        if (v->t == T_INT) snprintf(tmp, sizeof(tmp), "%lld", (long long)v->u.i);
        else fmt_double(v->u.f, tmp, sizeof(tmp));
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
static int dict_find(TDict *d, const TVal *k) {
    size_t i;
    if (!d->idx) {
        if (d->len < DICT_SMALL) {
            for (i = 0; i < d->len; i++)
                if (tval_eq(&d->keys[i], k)) return (int)i;
            return -1;
        }
        dict_index_build(d);
    }
    {
        size_t m = d->icap - 1, h = (size_t)tval_hash(k) & m;
        while (d->idx[h] != -1) {
            if (tval_eq(&d->keys[d->idx[h]], k)) return d->idx[h];
            h = (h + 1) & m;
        }
    }
    return -1;
}
static TDict *dict_new(void) {
    TDict *d = (TDict*)malloc(sizeof(TDict));
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
    TNative *n = (TNative*)malloc(sizeof(TNative));
    n->refc = 1; n->gc_mark = 0; n->fn = fn; n->has_obj = 0; n->name = name;
    n->obj.t = T_NIL;
    TVal v; v.t = T_NATIVE; v.u.nat = n;
    if (cur_vm) gc_track(cur_vm, T_NATIVE, n);
    return v;
}
static TVal mk_bound(TNativeFn fn, const char *name, TVal obj) {
    TNative *n = (TNative*)malloc(sizeof(TNative));
    n->refc = 1; n->gc_mark = 0; n->fn = fn; n->has_obj = 1; n->name = name;
    n->obj = obj; tval_retain(&n->obj);
    TVal v; v.t = T_NATIVE; v.u.nat = n;
    if (cur_vm) gc_track(cur_vm, T_NATIVE, n);
    return v;
}
#define NERR(msg) do { die(vm, msg); } while (0)
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
static void n_len(TVM *vm, TVal *a, int argc, TVal *r) {
    NARG(1);
    size_t n;
    if (a[0].t == T_STR) n = a[0].u.str.n;
    else if (a[0].t == T_LIST) n = a[0].u.list->len;
    else if (a[0].t == T_TENSOR) n = a[0].u.ten->ndim ? a[0].u.ten->shape[0] : a[0].u.ten->len;
    else if (a[0].t == T_DICT && a[0].u.dict->kind <= DK_SET) n = a[0].u.dict->len;
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
static void n_tensor(TVM *vm, TVal *a, int argc, TVal *r) {
    NARG(1);
    if (a[0].t == T_BOOL) die(vm, "tensor() numbers");
    r->t = T_TENSOR; r->u.ten = ten_from(vm, &a[0]);
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
    NARG(1);
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
    if (!f->native && argc == f->nargs) t_jit_maybe(vm, f, args);
    if (f->native) {          /* fn đã JIT thành native: gọi thẳng, không lặp lại */
        TVal rv; rv.t = T_NIL;
        if (t_jit_call(vm, f, args, argc, &rv)) { *out = rv; return 1; }
    }
    if (argc != f->nargs) die(vm, "call arity");
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
    for (int i = 0; i < argc; i++) fr->locals[i] = args[i];
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
        TVal out = mk_str_n(a[0].u.str.s, n);
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
    TVal v = mk_str_n(a[0].u.str.s, a[0].u.str.n);
    for (size_t i = 0; i < v.u.str.n; i++) {
        char c = v.u.str.s[i];
        if (c >= 'a' && c <= 'z') v.u.str.s[i] = (char)(c - 32);
    }
    *r = v;
}
static void m_str_lower(TVM *vm, TVal *a, int argc, TVal *r) {
    (void)argc;
    if (a[0].t != T_STR) die(vm, "lower()");
    TVal v = mk_str_n(a[0].u.str.s, a[0].u.str.n);
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
    TDict *d = (TDict*)malloc(sizeof(TDict));
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
static void do_attr(TVM *vm, const char *name, size_t namelen, TAttrIC *ic) {
    TVal *ov = &vm->stack[vm->sp - 1];
    /* inline cache: field record theo kiểu */
    if (ov->t == T_DICT && ov->u.dict->kind == DK_REC) {
        TDict *d = ov->u.dict;
        if (ic && ic->kind == 1 && ic->key == d->meths && (size_t)ic->idx < d->len &&
            d->keys[ic->idx].u.str.n == namelen &&
            memcmp(d->keys[ic->idx].u.str.s, name, namelen) == 0) {
            mod_pull(d, (size_t)ic->idx);
            TVal r = d->vals[ic->idx]; tval_retain(&r);
            TVal o = pop(vm); tval_free(&o);
            push(vm, r);
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
    else {
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
    tval_free(&a); tval_free(&b);
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
    for (uint16_t i = 0; i < f->nlocals; i++) tval_free(&f->locals[i]);
    /* giữ mảng locals (lcap) để call sau tái dùng; free thật ở tvm_free */
    f->nlocals = 0;
    if (f->fn) {
        TVal fv; fv.t = T_FN; fv.u.fn = f->fn;
        tval_free(&fv);
        f->fn = NULL;
    }
}

/* ---------- init/free ---------- */
/* ---------- đọc .tb + load() (import động lúc chạy) ---------- */
static int rd_exact(FILE *f, void *b, size_t n) { return n == 0 || fread(b, 1, n, f) == n; }
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
    if (!rd_exact(f, &len, 4)) { snprintf(msg, msgn, "bad len"); fclose(f); return -1; }
    p->nglobals = ng;
    p->code = (uint8_t*)malloc(len ? len : 1); p->len = len;
    if (!rd_exact(f, p->code, len)) {
        snprintf(msg, msgn, "short read"); fclose(f); free(p->code); p->code = NULL; return -1;
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
                ok = rd_exact(f, p->lfiles[i], ln);
                p->nlfiles = (uint16_t)(i + 1);
            }
            if (ok) ok = rd_exact(f, &nl, 4);
            if (ok && nl) {
                p->lpc = (uint32_t*)malloc(sizeof(uint32_t) * nl);
                p->lline = (uint32_t*)malloc(sizeof(uint32_t) * nl);
                p->lfile = (uint16_t*)malloc(sizeof(uint16_t) * nl);
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
            }
            for (uint16_t i = 0; ok && i < ne; i++) {
                uint16_t ln = 0;
                ok = rd_exact(f, &ln, 2);
                if (!ok) break;
                p->exnames[i] = (char*)calloc((size_t)ln + 1, 1);
                ok = rd_exact(f, p->exnames[i], ln) && rd_exact(f, &p->exslots[i], 2) &&
                     rd_exact(f, &p->exmut[i], 1);
                if (ok) p->nexports = (uint16_t)(i + 1);
                else { free(p->exnames[i]); p->exnames[i] = NULL; }
            }
        }
    }
    fclose(f);
    p->path = (char*)malloc(strlen(path) + 1);
    strcpy(p->path, path);
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
static double file_mtime(const char *path) {
    struct stat st;
    if (stat(path, &st) != 0) return -1;
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
/* src.t mới hơn (hoặc chưa có) out.tb: dịch bằng "trun tcomp.tb src.t out.tb".
   0 = không cần/không dịch được (không có trun/tcomp), 1 = đã dịch, -1 = lỗi biên dịch */
static int compile_if_stale(const char *src, const char *out, char *msg, size_t msgn) {
    char tc[1024], cmd[4096], ef[1100];
    int rc;
    double ts = file_mtime(src), tb = file_mtime(out);
    if (ts < 0 || (tb >= 0 && tb >= ts)) return 0;
    if (!tvm_exe_path || !find_tcomp(tc, sizeof(tc))) return 0;
    snprintf(ef, sizeof(ef), "%s.err", out);
#ifdef _WIN32
    /* cmd /c: bọc cả lệnh trong một cặp nháy ngoài để giữ nháy bên trong */
    snprintf(cmd, sizeof(cmd), "\"\"%s\" \"%s\" \"%s\" \"%s\" >nul 2>\"%s\"\"", tvm_exe_path, tc, src, out, ef);
#else
    snprintf(cmd, sizeof(cmd), "\"%s\" \"%s\" \"%s\" \"%s\" >/dev/null 2>\"%s\"", tvm_exe_path, tc, src, out, ef);
#endif
    fflush(stdout); fflush(stderr);
    rc = system(cmd);
    msg[0] = 0;
    {
        /* dòng đầu stderr của compiler (bỏ tiền tố "T error: ") làm thông điệp lỗi */
        FILE *f = fopen(ef, "rb");
        if (f) {
            if (fgets(msg, (int)msgn, f)) {
                size_t k = strlen(msg);
                while (k && (msg[k - 1] == '\n' || msg[k - 1] == '\r')) msg[--k] = 0;
                if (!strncmp(msg, "T error: ", 9)) memmove(msg, msg + 9, k - 9 + 1);
            }
            fclose(f);
        }
        remove(ef);
    }
    return rc == 0 ? 1 : -1;
}
/* load("tên"): nạp tên.tb (thư mục hiện tại, rồi thư mục của chương trình chính),
   chạy top-level của nó một lần (cache theo đường dẫn) -> bản ghi kiểu "module"
   gồm các global top-level (giá trị lúc nạp xong). */
static void n_load(TVM *vm, TVal *a, int argc, TVal *r) {
    char path[4200], eb[1200];
    TProg *root = vm->root;
    if (argc != 1 || a[0].t != T_STR) die(vm, "load(name str)");
    if (a[0].u.str.n > 900) die(vm, "load: tên quá dài");
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
void tvm_free(TVM *vm) {
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
                free(nd);
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
    tval_retain(&v);
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
    while (vm->sp > bp) tval_free(&vm->stack[--vm->sp]);
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
                if (k != f->nargs) {
                    char eb[64];
                    snprintf(eb, sizeof(eb), "%.40s arity", "call");
                    die(vm, eb);
                }
                /* JIT hot-path: đếm + thử biên dịch, rồi chạy native nếu tag khớp */
                if (!f->native)
                    t_jit_maybe(vm, f, &vm->stack[vm->sp - k]);
                if (f->native) {
                    TVal rv; rv.t = T_NIL;
                    if (t_jit_call(vm, f, &vm->stack[vm->sp - k], k, &rv)) {
                        int i;
                        for (i = 0; i < k + 1; i++)
                            tval_free(&vm->stack[vm->sp - k - 1 + i]);
                        vm->sp -= k + 1;
                        push(vm, rv);
                        return;
                    }
                    /* deopt: rẽ interpreter bên dưới */
                }
                TFrame *fr = push_frame(vm);
                size_t bp = vm->sp - k - 1;
                fr->ret_pc = vm->pc; fr->bp = bp;
                fr->nlocals = f->nlocals;
                fr->fn = f;
                fr->ret_prog = vm->prog; fr->is_mod = 0;
                vm->prog = f->prog;
                { TVal fv2 = vm->stack[bp]; tval_retain(&fv2); }
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
                for (uint8_t i = 0; i < k; i++)
                    fr->locals[i] = vm->stack[bp + 1 + i];
                tval_free(&vm->stack[bp]);
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
                    tval_free(&vm->stack[vm->sp - k - 1 + i]);
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
    for (int i = 0; i < nargs; i++) fr->locals[i] = vm->stack[bp + i];
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
            memcmp(tp->keys[ic->idx].u.str.s, nm, len) == 0) mi = ic->idx;
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
            if ((int)k + 1 != f->nargs) die(vm, "call arity");
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
            for (int i = 0; i <= k; i++) tval_free(&vm->stack[bp + i]);
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
int tvm_run(TVM *vm) {
    uint8_t *c = vm->prog->code;
    size_t n = vm->prog->len;
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
    for (;;) {
        uint8_t op;
        /* (GC chu trình chỉ kiểm ở cạnh lùi JMP và CALL: mọi giá trị sống khi
           đó đều nằm trên stack/frame/global) */
        if (tvm_dbg_on && !dbg_hook(vm)) return -1;
        /* gọi/return qua hàm của module khác (load) đổi vm->prog */
        if (c != vm->prog->code) { c = vm->prog->code; n = vm->prog->len; }
        if (vm->pc >= n) { die(vm, "pc overrun"); }
        op = c[vm->pc++];
#ifdef TVM_TRACE
        fprintf(stderr, "[pc=%zu op=0x%02x sp=%zu fr=%zu nh=%zu]\n",
                vm->pc - 1, op, vm->sp, vm->nframes, vm->nhandlers);
#endif
        switch (op) {
        case OP_PUSH_NIL: push(vm, mk_nil()); break;
        case OP_PUSH_BOOL: { TVal v = mk_bool(c[vm->pc++]); push(vm, v); break; }
        case OP_PUSH_INT: push(vm, mk_int(rd_i64(c, &vm->pc))); break;
        case OP_PUSH_FLOAT: push(vm, mk_float(rd_f64(c, &vm->pc))); break;
        case OP_PUSH_STR: {
            TAttrIC *ic = &vm->prog->ic[vm->pc - 1];
            uint32_t len = rd_u32(c, &vm->pc);
            if (vm->pc + len > n) die(vm, "bad str");
            if (ic->kind == 4) {
                TVal v = *(const TVal*)ic->key;
                (*v.u.str.refc)++;
                vm->pc += len;
                push(vm, v); break;
            }
            {
                TVal v = mk_str_n((char*)(c + vm->pc), len);
                TVal *keep = (TVal*)malloc(sizeof(TVal));
                *keep = v; (*v.u.str.refc)++;          /* cache giữ 1 ref */
                ic->kind = 4; ic->key = keep;
                vm->pc += len;
                push(vm, v); break;
            }
        }
        case OP_NEW_LIST: {
            uint16_t k = rd_u16(c, &vm->pc);
            if (vm->sp < k) die(vm, "stack underflow");
            TList *l = list_new(k);
            for (uint16_t i = 0; i < k; i++) l->items[i] = vm->stack[vm->sp - k + i];
            l->len = k;
            vm->sp -= k;
            TVal v; v.t = T_LIST; v.u.list = l;
            push(vm, v); break;
        }
        case OP_NEW_TENSOR: {
            uint16_t k = rd_u16(c, &vm->pc);
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
            uint16_t k = rd_u16(c, &vm->pc);
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
            arith(vm, op); break;
        }
        case OP_SUB: {
            TVal *b = &vm->stack[vm->sp - 1], *a = b - 1;
            if (a->t == T_INT && b->t == T_INT) {
                a->u.i = (int64_t)((uint64_t)a->u.i - (uint64_t)b->u.i); vm->sp--; break;
            }
            if (a->t == T_FLOAT && b->t == T_FLOAT) { a->u.f -= b->u.f; vm->sp--; break; }
            arith(vm, op); break;
        }
        case OP_MUL: {
            TVal *b = &vm->stack[vm->sp - 1], *a = b - 1;
            if (a->t == T_INT && b->t == T_INT) {
                a->u.i = (int64_t)((uint64_t)a->u.i * (uint64_t)b->u.i); vm->sp--; break;
            }
            if (a->t == T_FLOAT && b->t == T_FLOAT) { a->u.f *= b->u.f; vm->sp--; break; }
            arith(vm, op); break;
        }
        case OP_MOD: {
            TVal *b = &vm->stack[vm->sp - 1], *a = b - 1;
            if (a->t == T_INT && b->t == T_INT && b->u.i != 0) {
                a->u.i = imod(a->u.i, b->u.i); vm->sp--; break;
            }
            arith(vm, op); break;
        }
        case OP_DIV: case OP_POW: case OP_TDOT:
            arith(vm, op); break;
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
            } else { cmpop(vm, op); break; }
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
                if (i < 0) {
                    char eb[128]; Buf b = {0,0,0};
                    fmt_into(&b, &ix);
                    snprintf(eb, sizeof(eb), "%.40s has no field %.40s",
                             rec_name(cv.u.dict), b.s ? b.s : "?");
                    free(b.s);
                    tval_free(&ix); tval_free(&cv); die(vm, eb);
                }
                mod_pull(cv.u.dict, (size_t)i);
                r = cv.u.dict->vals[i]; tval_retain(&r);
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
                if (i < 0) {
                    char eb[128]; Buf b = {0,0,0};
                    fmt_into(&b, &ix);
                    snprintf(eb, sizeof(eb), "%.40s has no field %.40s",
                             rec_name(cv.u.dict), b.s ? b.s : "?");
                    free(b.s);
                    tval_free(&v); tval_free(&ix); tval_free(&cv); die(vm, eb);
                }
                {
                    char eb[128];
                    if (mod_ro(cv.u.dict, (size_t)i, eb, sizeof(eb))) {
                        tval_free(&v); tval_free(&ix); tval_free(&cv); die(vm, eb);
                    }
                }
                tval_free(&cv.u.dict->vals[i]);
                cv.u.dict->vals[i] = v;
                mod_push(cv.u.dict, (size_t)i);
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
            if (cv.t == T_STR) {
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
            else { tval_free(&a); die(vm, "len() here"); }
            tval_free(&a);
            push(vm, mk_int((int64_t)k)); break;
        }
        case OP_PRINT: {
            uint8_t k = c[vm->pc++];
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
            size_t jpc = vm->pc - 1;
            int32_t rel = rd_i32(c, &vm->pc);
            vm->pc = (size_t)((int64_t)vm->pc + rel);
            if (rel < 0) {
                if (vm->root->gc_allocs >= GC_THRESHOLD) gc_collect(vm);
                /* vòng lặp nóng ở top-level: JIT vùng vòng lặp (global ghi thẳng) */
                if (vm->nframes == 1 && vm->stop_nf == 0) {
                    TAttrIC *ic = &vm->prog->ic[jpc];
                    if (ic->kind == 0) { ic->kind = 6; ic->idx = 0; }
                    if (ic->kind == 6 && ic->idx >= 0 && ++ic->idx >= JIT_OSR_LOOPS) {
                        int jr = t_jit_osr_main(vm, vm->pc, jpc + 5);
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
                        int jr = t_jit_osr(vm, fr->fn, fr->locals, vm->pc, &rv);
                        if (jr == 1) {
                            do_return(vm, rv);
                            if (vm->nframes <= vm->stop_nf) return 0;
                        }
                        /* jr == 2: deopt, vm->pc + stack đã đặt lại -> chạy tiếp */
                    }
                }
            }
            break;
        }
        case OP_JMPF: case OP_JMPT: {
            int is_t = (op == OP_JMPT);
            int32_t rel = rd_i32(c, &vm->pc);
            TVal *tv = &vm->stack[vm->sp - 1];
            int t;
            if (tv->t == T_BOOL) { t = tv->u.b; vm->sp--; }
            else {
                TVal v = pop(vm);
                t = tval_truthy(&v);
                tval_free(&v);
            }
            if ((t && is_t) || (!t && !is_t))
                vm->pc = (size_t)((int64_t)vm->pc + rel);
            break;
        }
        case OP_CALL: {
            uint8_t k = c[vm->pc++];
            if (vm->sp < (size_t)k + 1) die(vm, "stack underflow");
            if (vm->root->gc_allocs >= GC_THRESHOLD) gc_collect(vm);
            call_value(vm, k);
            break;
        }
        case OP_CALLM: {
            size_t at = vm->pc - 1;
            uint8_t k = c[vm->pc++];
            uint32_t len = rd_u32(c, &vm->pc);
            const char *nm = (const char*)(c + vm->pc);
            vm->pc += len;
            if (vm->sp < (size_t)k + 1) die(vm, "stack underflow");
            if (vm->root->gc_allocs >= GC_THRESHOLD) gc_collect(vm);
            call_method(vm, k, nm, len, &vm->prog->ic[at]);
            break;
        }
        case OP_SETATTR: {
            size_t at = vm->pc - 1;
            uint32_t len = rd_u32(c, &vm->pc);
            const char *nm = (const char*)(c + vm->pc);
            vm->pc += len;
            set_attr(vm, nm, len, &vm->prog->ic[at]);
            break;
        }
        case OP_RET: {
            if (vm->nframes <= 1) die(vm, "ret outside");
            TVal rv = pop(vm);
            do_return(vm, rv);
            if (vm->nframes <= vm->stop_nf) return 0;  /* hết nested call */
            break;
        }
        case OP_LOAD: {
            uint16_t s = rd_u16(c, &vm->pc);
            TFrame *fr = &vm->frames[vm->nframes - 1];
            if (s < fr->nlocals) {
                TVal v = fr->locals[s];
                if (v.t <= T_FLOAT) { push(vm, v); break; }   /* nil/bool/int/float: không refcount */
            }
            push(vm, slot_get(vm, 0, s));
            break;
        }
        case OP_STORE: case OP_DECL: case OP_BIND: {
            uint16_t s = rd_u16(c, &vm->pc);
            TFrame *fr = &vm->frames[vm->nframes - 1];
            TVal v = pop(vm);
            if (s < fr->nlocals && fr->locals[s].t <= T_FLOAT) { fr->locals[s] = v; break; }
            if (op == OP_BIND) slot_replace(vm, 0, s, v);
            else slot_set(vm, 0, s, v);
            break;
        }
        case OP_LOADG: {
            uint16_t s = rd_u16(c, &vm->pc);
            if (s < vm->prog->nglobals) {
                TVal v = vm->prog->globals[s];
                if (v.t <= T_FLOAT) { push(vm, v); break; }
            }
            push(vm, slot_get(vm, 1, s));
            break;
        }
        case OP_STOREG: case OP_DECLG: {
            uint16_t s = rd_u16(c, &vm->pc);
            TVal v = pop(vm);
            if (s < vm->prog->nglobals && vm->prog->globals[s].t <= T_FLOAT) {
                vm->prog->globals[s] = v; break;
            }
            slot_set(vm, 1, s, v);
            break;
        }
        case OP_MKFN: {
            int32_t e = rd_i32(c, &vm->pc);
            uint8_t na = c[vm->pc++];
            uint16_t nl = rd_u16(c, &vm->pc);
            uint16_t nu = rd_u16(c, &vm->pc);
            TFn *f = (TFn*)malloc(sizeof(TFn));
            f->refc = 1; f->gc_mark = 0; f->entry = e; f->nargs = na; f->nlocals = nl;
            f->nupvals = nu; f->ncap = 0;
            f->upvals = nu ? (TVal*)calloc(nu, sizeof(TVal)) : NULL;
            f->ncalls = 0; f->native = NULL; f->loops = 0; f->jit_off = 0; f->jit = NULL;
            f->prog = vm->prog;
            TVal v; v.t = T_FN; v.u.fn = f;
            gc_track(vm, T_FN, f);
            push(vm, v); break;
        }
        case OP_CAPLOCAL: {
            uint16_t s = rd_u16(c, &vm->pc);
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
                cell = (TCell*)malloc(sizeof(TCell));
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
            uint16_t i = rd_u16(c, &vm->pc);
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
            uint16_t s = rd_u16(c, &vm->pc);
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
                cell = (TCell*)malloc(sizeof(TCell));
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
            uint16_t s = rd_u16(c, &vm->pc);
            if (s >= vm->prog->nglobals) die(vm, "bad global");
            TVal v = pop(vm);
            tval_free(&vm->prog->globals[s]);
            vm->prog->globals[s] = v;
            break;
        }
        case OP_LOADU: case OP_STOREU: {
            uint16_t i = rd_u16(c, &vm->pc);
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
            uint8_t k = c[vm->pc++];
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
            uint32_t len = rd_u32(c, &vm->pc);
            if (vm->pc + len > n) die(vm, "bad attr");
            const char *nm = (const char*)(c + vm->pc);
            vm->pc += len;
            do_attr(vm, nm, len, &vm->prog->ic[vm->pc - len - 5]);
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
            int32_t rel = rd_i32(c, &vm->pc);
            if (vm->nhandlers >= vm->hcap) {
                vm->hcap *= 2;
                vm->handlers = (THandler*)realloc(vm->handlers,
                    vm->hcap * sizeof(THandler));
                if (!vm->handlers) exit(1);
            }
            THandler *h = &vm->handlers[vm->nhandlers++];
            h->sp = vm->sp; h->nframes = vm->nframes;
            h->pc = (size_t)((int64_t)vm->pc + rel);
            h->stop_nf = vm->stop_nf;
            h->prog = vm->prog;
            if (setjmp(h->jb)) {
                TVal v = mk_str(vm->errmsg);
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
            uint16_t s = rd_u16(c, &vm->pc);
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
            uint16_t s = rd_u16(c, &vm->pc);
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
            uint16_t s = rd_u16(c, &vm->pc);
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
            uint16_t s = rd_u16(c, &vm->pc);
            int64_t k = rd_i64(c, &vm->pc);
            int isg = (op == OP_ADDI_G);
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
            memcpy(&ds, c + vm->pc, 2); vm->pc += 2;
            memcpy(&ss, c + vm->pc, 2); vm->pc += 2;
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
        case OP_NOP: break;
        case OP_HALT:
            if (vm->frames[vm->nframes - 1].is_mod) return 0;   /* hết top-level module */
            in_top = 0; return 0;
        default: die(vm, "unknown opcode");
        }
    }
}



