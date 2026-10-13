/* tvm.c ??? T bytecode VM v1: stack machine + frames + refcount + try + natives. */
#include "tvm.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define STACK_INIT 1024
#define FRAME_INIT 64
#define HAND_INIT 16

static jmp_buf top_jb;
static int in_top;
static void free_frame(TVM *vm);
static TVM *cur_vm;   /* VM ??ang ch???y (GC track d??ng; ????n lu???ng) */
static void gc_track(TVM *vm, int type, void *ptr);
static void gc_collect(TVM *vm);

static void die(TVM *vm, const char *msg) {
    strncpy(vm->errmsg, msg, sizeof(vm->errmsg) - 1);
    vm->errmsg[sizeof(vm->errmsg) - 1] = 0;
    if (vm->nhandlers > 0) {
        THandler h = vm->handlers[--vm->nhandlers];
        while (vm->sp > h.sp) tval_free(&vm->stack[--vm->sp]);
        while (vm->nframes > h.nframes) free_frame(vm);
        vm->pc = h.pc;
        longjmp(h.jb, 1);
    }
    vm->err = 1;
    if (in_top) longjmp(top_jb, 1);
    exit(1);
}

/* ---------- GC mark-sweep (chu tr??nh; refcount x??? l?? c??n l???i) ---------- */
#define GC_THRESHOLD 1000000000

static void gc_unlink(TVM *vm, GCNode *nd) {
    (void)vm;
    if (nd->prev) nd->prev->next = nd->next;
    else vm->prog->gc_head = nd->next;
    if (nd->next) nd->next->prev = nd->prev;
    free(nd);
    vm->prog->gc_tracked--;
}

static void gc_track(TVM *vm, int type, void *ptr) {
    GCNode *nd = (GCNode*)malloc(sizeof(GCNode));
    if (!nd) return;
    nd->type = type; nd->ptr = ptr;
    nd->prev = NULL; nd->next = vm->prog->gc_head;
    if (nd->next) nd->next->prev = nd;
    vm->prog->gc_head = nd;
    if (type == T_LIST) ((TList*)ptr)->gc_node = nd;
    else if (type == T_DICT) ((TDict*)ptr)->gc_node = nd;
    else if (type == T_FN) ((TFn*)ptr)->gc_node = nd;
    else if (type == T_NATIVE) ((TNative*)ptr)->gc_node = nd;
    vm->prog->gc_tracked++;
    vm->prog->gc_allocs++;
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
        break;
    }
    case T_FN: {
        TFn *f = v->u.fn;
        if (!f || f->gc_mark) break;
        f->gc_mark = 1;
        for (i = 0; i < f->ncap; i++) gc_mark_val(vm, &f->upvals[i]);
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

static void gc_mark_obj(TVM *vm, int type, void *ptr) {
    (void)vm; (void)type; (void)ptr;
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

static void gc_destroy(TVM *vm, int type, void *ptr) {
    TVal w;
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
    if (type == T_LIST) {
        ((TList*)ptr)->refc = 1;
        w.t = T_LIST; w.u.list = (TList*)ptr; tval_free(&w);
    } else if (type == T_DICT) {
        ((TDict*)ptr)->refc = 1;
        w.t = T_DICT; w.u.dict = (TDict*)ptr; tval_free(&w);
    } else if (type == T_FN) {
        TFn *f = (TFn*)ptr;
        t_jit_free_fn(f);
        if (f->upvals) {
            for (uint16_t i = 0; i < f->ncap; i++) tval_free(&f->upvals[i]);
            free(f->upvals);
        }
        free(f);
    } else if (type == T_NATIVE) {
        ((TNative*)ptr)->refc = 1;
        w.t = T_NATIVE; w.u.nat = (TNative*)ptr; tval_free(&w);
    }
}

static void gc_collect(TVM *vm) {
    size_t i;
    GCNode **pp;
    vm->prog->gc_allocs = 0;
    vm->prog->gc_runs++;
    for (i = 0; i < vm->sp; i++) gc_mark_val(vm, &vm->stack[i]);
    for (i = 0; i < vm->nframes; i++) {
        TFrame *f = &vm->frames[i];
        size_t j;
        for (j = 0; j < f->nlocals; j++) gc_mark_val(vm, &f->locals[j]);
        if (f->fn) { TVal v; v.t = T_FN; v.u.fn = f->fn; gc_mark_val(vm, &v); }
    }
    for (i = 0; i < vm->prog->nglobals; i++)
        gc_mark_val(vm, &vm->prog->globals[i]);
    pp = (GCNode**)&vm->prog->gc_head;
    (void)pp;
    {
        GCNode *nd = vm->prog->gc_head;
        while (nd) {
            GCNode *nx = nd->next;
            int marked = 0;
            if (nd->type == T_LIST) marked = ((TList*)nd->ptr)->gc_mark;
            else if (nd->type == T_DICT) marked = ((TDict*)nd->ptr)->gc_mark;
            else if (nd->type == T_FN) marked = ((TFn*)nd->ptr)->gc_mark;
            else if (nd->type == T_NATIVE) marked = ((TNative*)nd->ptr)->gc_mark;
            if (marked) {
                if (nd->type == T_LIST) ((TList*)nd->ptr)->gc_mark = 0;
                else if (nd->type == T_DICT) ((TDict*)nd->ptr)->gc_mark = 0;
                else if (nd->type == T_FN) ((TFn*)nd->ptr)->gc_mark = 0;
                else if (nd->type == T_NATIVE) ((TNative*)nd->ptr)->gc_mark = 0;
            } else {
                /* gc_destroy t??? unlink node */
                gc_destroy(vm, nd->type, nd->ptr);
            }
            nd = nx;
        }
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
static TVal mk_undef(void) { TVal v; v.t = T_UNDEF; return v; }
static void fill_undef(TVal *p, size_t n) {
    size_t i;
    for (i = 0; i < n; i++) p[i].t = T_UNDEF;
}
static TVal mk_bool(int b) { TVal v; v.t = T_BOOL; v.u.b = b ? 1 : 0; return v; }
static TVal mk_int(int64_t i) { TVal v; v.t = T_INT; v.u.i = i; return v; }
static TVal mk_float(double f) { TVal v; v.t = T_FLOAT; v.u.f = f; return v; }
static TVal mk_str_n(const char *s, size_t n) {
    TVal v; v.t = T_STR;
    v.u.str.s = (char*)malloc(n + 1);
    memcpy(v.u.str.s, s, n); v.u.str.s[n] = 0;
    v.u.str.n = n;
    v.u.str.refc = (int32_t*)malloc(sizeof(int32_t));
    *v.u.str.refc = 1;
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
void tval_free(TVal *v) {
    size_t i;
    if (!v) return;
    switch (v->t) {
    case T_STR:
        if (v->u.str.refc && --(*v->u.str.refc) == 0) {
            free(v->u.str.s); free(v->u.str.refc);
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
            free(v->u.ten->d); free(v->u.ten);
        }
        break;
    case T_DICT:
        if (v->u.dict && --(v->u.dict->refc) == 0) {
            if (cur_vm) gc_untrack(cur_vm, T_DICT, v->u.dict);
            for (i = 0; i < v->u.dict->len; i++) {
                tval_free(&v->u.dict->keys[i]);
                tval_free(&v->u.dict->vals[i]);
            }
            free(v->u.dict->keys); free(v->u.dict->vals); free(v->u.dict);
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
    case T_DICT: return v->u.dict->len != 0;
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
        if (a->u.ten->len != b->u.ten->len) return 0;
        for (i = 0; i < a->u.ten->len; i++)
            if (a->u.ten->d[i] != b->u.ten->d[i]) return 0;
        return 1;
    case T_DICT: {
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

/* fmt ?????ng (cho str()/raise/print/dict) */
typedef struct { char *s; size_t n, cap; } Buf;
static void buf_put(Buf *b, const char *s, size_t n) {
    while (b->n + n + 1 > b->cap) {
        b->cap = b->cap ? b->cap * 2 : 64;
        b->s = (char*)realloc(b->s, b->cap);
    }
    memcpy(b->s + b->n, s, n); b->n += n; b->s[b->n] = 0;
}
static void buf_c(Buf *b, char c) { buf_put(b, &c, 1); }
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
        if (v->u.f == (long long)v->u.f)
            snprintf(tmp, sizeof(tmp), "%lld", (long long)v->u.f);
        else snprintf(tmp, sizeof(tmp), "%g", v->u.f);
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
        buf_c(b, '[');
        for (i = 0; i < v->u.ten->len; i++) {
            double x = v->u.ten->d[i];
            if (i) buf_put(b, ", ", 2);
            if (x == (long long)x) snprintf(tmp, sizeof(tmp), "%lld", (long long)x);
            else snprintf(tmp, sizeof(tmp), "%g", x);
            buf_put(b, tmp, strlen(tmp));
        }
        buf_c(b, ']'); break;
    case T_DICT:
        buf_c(b, '{');
        for (i = 0; i < v->u.dict->len; i++) {
            if (i) buf_put(b, ", ", 2);
            fmt_into(b, &v->u.dict->keys[i]);
            buf_put(b, ": ", 2);
            fmt_into(b, &v->u.dict->vals[i]);
        }
        buf_c(b, '}'); break;
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
static int dict_find(TDict *d, const TVal *k) {
    size_t i;
    for (i = 0; i < d->len; i++)
        if (tval_eq(&d->keys[i], k)) return (int)i;
    return -1;
}
static TDict *dict_new(void) {
    TDict *d = (TDict*)malloc(sizeof(TDict));
    d->refc = 1; d->gc_mark = 0; d->len = 0; d->cap = 4;
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
    else if (a[0].t == T_TENSOR) n = a[0].u.ten->len;
    else if (a[0].t == T_DICT) n = a[0].u.dict->len;
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
    const char *s = "?";
    switch (a[0].t) {
    case T_NIL: s = "nil"; break;
    case T_BOOL: s = "bool"; break;
    case T_INT: s = "int"; break;
    case T_FLOAT: s = "float"; break;
    case T_STR: s = "str"; break;
    case T_LIST: s = "list"; break;
    case T_DICT: s = "dict"; break;
    case T_TENSOR: s = "tensor"; break;
    case T_FN: case T_NATIVE: s = "fn"; break;
    case T_FILE: s = "file"; break;
    default: s = "obj";
    }
    *r = mk_str(s);
}
static void n_tensor(TVM *vm, TVal *a, int argc, TVal *r) {
    NARG(1);
    TTensor *t = (TTensor*)malloc(sizeof(TTensor));
    t->refc = 1;
    if (a[0].t == T_LIST) {
        t->len = a[0].u.list->len;
        t->d = (double*)malloc(sizeof(double) * (t->len ? t->len : 1));
        for (size_t i = 0; i < t->len; i++) {
            int ok = 1; double x = to_num(&a[0].u.list->items[i], &ok);
            if (!ok) { free(t->d); free(t); die(vm, "tensor() numbers"); }
            t->d[i] = x;
        }
    } else if (a[0].t == T_TENSOR) {
        t->len = a[0].u.ten->len;
        t->d = (double*)malloc(sizeof(double) * (t->len ? t->len : 1));
        memcpy(t->d, a[0].u.ten->d, sizeof(double) * t->len);
    } else {
        int ok = 1; double x = to_num(&a[0], &ok);
        if (!ok) die(vm, "tensor() here");
        t->len = 1; t->d = (double*)malloc(sizeof(double)); t->d[0] = x;
    }
    r->t = T_TENSOR; r->u.ten = t;
}
static void n_open(TVM *vm, TVal *a, int argc, TVal *r) {
    const char *mode = "r";
    char mbuf[8];
    if (argc < 1 || argc > 2) die(vm, "arity");
    if (a[0].t != T_STR) die(vm, "open() path");
    if (argc == 2) {
        if (a[1].t != T_STR) die(vm, "open() mode");
        if (a[1].u.str.n >= sizeof(mbuf)) die(vm, "open() mode");
        memcpy(mbuf, a[1].u.str.s, a[1].u.str.n);
        mbuf[a[1].u.str.n] = 0; mode = mbuf;
    }
    /* T string l?? BYTE-STRING: m???i mode ?????u raw (kh??ng CRLF translation) */
    {
        char rb[10];
        size_t k = 0;
        for (const char *q = mode; *q && k + 3 < sizeof(rb); q++) {
            if (*q != 'b') rb[k++] = *q;
        }
        rb[k] = 'b';
        rb[k + 1] = 0;
        mode = rb;
    }
    FILE *h = fopen(a[0].u.str.s, mode);
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
    if (a[0].t != T_STR || a[0].u.str.n != 1) {
        char eb[160];
        TFrame *fr = &vm->frames[vm->nframes - 1];
        snprintf(eb, sizeof(eb), "ord() 1 char (n=%zu fn=%p pc=%zu)",
                 a[0].t == T_STR ? a[0].u.str.n : 0,
                 fr->fn ? (void*)fr->fn : NULL, vm->pc);
        die(vm, eb);
    }
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
    TList *l = list_new(vm->prog->xargc > 0 ? (size_t)vm->prog->xargc : 0);
    for (int i = 0; i < vm->prog->xargc; i++)
        list_push(l, mk_str(vm->prog->xargv[i]));
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
    v = mk_int((int64_t)vm->prog->gc_tracked);
    dict_set(d, k, v);
    k = mk_str("gc");
    v = mk_int((int64_t)vm->prog->gc_runs);
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
/* methods (obj bound l?? args[-1] qua has_obj -> prepend) */
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
        for (size_t i = 0; i < a[1].u.ten->len; i++) {
            if (i) buf_put(&b, a[0].u.str.s, a[0].u.str.n);
            double x = a[1].u.ten->d[i];
            if (x == (long long)x) snprintf(tmp, sizeof(tmp), "%lld", (long long)x);
            else snprintf(tmp, sizeof(tmp), "%g", x);
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
static void m_ten_sum(TVM *vm, TVal *a, int argc, TVal *r) {
    NARG(1);
    if (a[0].t != T_TENSOR) die(vm, "sum()");
    double s = 0;
    for (size_t i = 0; i < a[0].u.ten->len; i++) s += a[0].u.ten->d[i];
    *r = mk_float(s);
}
static void m_ten_mean(TVM *vm, TVal *a, int argc, TVal *r) {
    NARG(1);
    if (a[0].t != T_TENSOR) die(vm, "mean()");
    if (a[0].u.ten->len == 0) { *r = mk_float(0); return; }
    double s = 0;
    for (size_t i = 0; i < a[0].u.ten->len; i++) s += a[0].u.ten->d[i];
    *r = mk_float(s / (double)a[0].u.ten->len);
}
static void m_ten_max(TVM *vm, TVal *a, int argc, TVal *r) {
    NARG(1);
    if (a[0].t != T_TENSOR || a[0].u.ten->len == 0) die(vm, "max()");
    double m = a[0].u.ten->d[0];
    for (size_t i = 1; i < a[0].u.ten->len; i++)
        if (a[0].u.ten->d[i] > m) m = a[0].u.ten->d[i];
    *r = mk_float(m);
}
static void m_ten_min(TVM *vm, TVal *a, int argc, TVal *r) {
    NARG(1);
    if (a[0].t != T_TENSOR || a[0].u.ten->len == 0) die(vm, "min()");
    double m = a[0].u.ten->d[0];
    for (size_t i = 1; i < a[0].u.ten->len; i++)
        if (a[0].u.ten->d[i] < m) m = a[0].u.ten->d[i];
    *r = mk_float(m);
}

/* ---------- attr ---------- */
static void do_attr(TVM *vm, const char *name, size_t namelen) {
    TVal o = pop(vm);
    char nb[64];
    if (namelen >= sizeof(nb)) die(vm, "attr name");
    memcpy(nb, name, namelen); nb[namelen] = 0;
    TVal r; r.t = T_NIL;
    if (o.t == T_STR) {
        if (!strcmp(nb, "len")) r = mk_int((int64_t)o.u.str.n);
        else if (!strcmp(nb, "upper")) r = mk_bound(m_str_upper, "upper", o);
        else if (!strcmp(nb, "lower")) r = mk_bound(m_str_lower, "lower", o);
        else if (!strcmp(nb, "split")) r = mk_bound(m_str_split, "split", o);
        else if (!strcmp(nb, "join")) r = mk_bound(m_str_join, "join", o);
        else if (!strcmp(nb, "strip")) r = mk_bound(m_str_strip, "strip", o);
        else { char eb[96]; snprintf(eb, sizeof(eb), "str has no .%.40s", nb); tval_free(&o); die(vm, eb); }
    } else if (o.t == T_LIST) {
        if (!strcmp(nb, "len")) r = mk_int((int64_t)o.u.list->len);
        else if (!strcmp(nb, "append")) r = mk_bound(m_list_append, "append", o);
        else if (!strcmp(nb, "pop")) r = mk_bound(m_list_pop, "pop", o);
        else { char eb[96]; snprintf(eb, sizeof(eb), "list has no .%.40s", nb); tval_free(&o); die(vm, eb); }
    } else if (o.t == T_DICT) {
        if (!strcmp(nb, "len")) r = mk_int((int64_t)o.u.dict->len);
        else if (!strcmp(nb, "keys")) r = mk_bound(m_dict_keys, "keys", o);
        else if (!strcmp(nb, "values")) r = mk_bound(m_dict_values, "values", o);
        else if (!strcmp(nb, "get")) r = mk_bound(m_dict_get, "get", o);
        else { char eb[96]; snprintf(eb, sizeof(eb), "dict has no .%.40s", nb); tval_free(&o); die(vm, eb); }
    } else if (o.t == T_TENSOR) {
        if (!strcmp(nb, "len")) r = mk_int((int64_t)o.u.ten->len);
        else if (!strcmp(nb, "sum")) r = mk_bound(m_ten_sum, "sum", o);
        else if (!strcmp(nb, "mean")) r = mk_bound(m_ten_mean, "mean", o);
        else if (!strcmp(nb, "max")) r = mk_bound(m_ten_max, "max", o);
        else if (!strcmp(nb, "min")) r = mk_bound(m_ten_min, "min", o);
        else { char eb[96]; snprintf(eb, sizeof(eb), "tensor has no .%.40s", nb); tval_free(&o); die(vm, eb); }
    } else {
        char eb[96]; snprintf(eb, sizeof(eb), ".%-.40s here", nb);
        tval_free(&o); die(vm, eb);
    }
    tval_free(&o);
    push(vm, r);
}

/* ---------- arith ---------- */
static void arith(TVM *vm, int op) {
    TVal b = pop(vm), a = pop(vm);
    TVal r; r.t = T_NIL;
    /* tensor paths */
    if (a.t == T_TENSOR || b.t == T_TENSOR) {
        if (op != OP_ADD && op != OP_SUB && op != OP_MUL && op != OP_DIV &&
            op != OP_POW && op != OP_TDOT) {
            tval_free(&a); tval_free(&b); die(vm, "bad tensor op");
        }
        if (op == OP_TDOT) {
            if (a.t != T_TENSOR || b.t != T_TENSOR ||
                a.u.ten->len != b.u.ten->len) {
                tval_free(&a); tval_free(&b); die(vm, "dot shape");
            }
            double s = 0;
            for (size_t i = 0; i < a.u.ten->len; i++)
                s += a.u.ten->d[i] * b.u.ten->d[i];
            tval_free(&a); tval_free(&b);
            push(vm, mk_float(s));
            return;
        }
        TTensor *ta = a.t == T_TENSOR ? a.u.ten : NULL;
        TTensor *tb = b.t == T_TENSOR ? b.u.ten : NULL;
        size_t n = ta ? ta->len : tb->len;
        double sb = 0, sa = 0; int oka = 1, okb = 1;
        if (!ta) sa = to_num(&a, &oka);
        if (!tb) sb = to_num(&b, &okb);
        if (ta && tb && ta->len != tb->len) {
            tval_free(&a); tval_free(&b); die(vm, "tensor len mismatch");
        }
        if ((ta == NULL && !oka) || (tb == NULL && !okb)) {
            tval_free(&a); tval_free(&b); die(vm, "bad tensor op");
        }
        TTensor *t = (TTensor*)malloc(sizeof(TTensor));
        t->refc = 1; t->len = n;
        t->d = (double*)malloc(sizeof(double) * (n ? n : 1));
        for (size_t i = 0; i < n; i++) {
            double x = ta ? ta->d[i] : sa;
            double y = tb ? tb->d[i] : sb;
            double z = 0;
            if (op == OP_ADD) z = x + y;
            else if (op == OP_SUB) z = x - y;
            else if (op == OP_MUL) z = x * y;
            else if (op == OP_DIV) z = x / y;
            else if (op == OP_POW) z = pow(x, y);
            t->d[i] = z;
        }
        tval_free(&a); tval_free(&b);
        r.t = T_TENSOR; r.u.ten = t;
        push(vm, r);
        return;
    }
    if (op == OP_ADD) {
        if (a.t == T_STR && b.t == T_STR) {
            size_t n = a.u.str.n + b.u.str.n;
            TVal v = mk_str_n("", 0);
            free(v.u.str.s);
            v.u.str.s = (char*)malloc(n + 1);
            memcpy(v.u.str.s, a.u.str.s, a.u.str.n);
            memcpy(v.u.str.s + a.u.str.n, b.u.str.s, b.u.str.n);
            v.u.str.s[n] = 0; v.u.str.n = n;
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
            r = both ? mk_int(a.u.i % b.u.i) : mk_float(fmod(x, y));
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
static void cmpop(TVM *vm, int op) {
    TVal b = pop(vm), a = pop(vm);
    int res = 0;
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
    /* gi??? m???ng locals (lcap) ????? call sau t??i d??ng; free th???t ??? tvm_free */
    f->nlocals = 0;
    if (f->fn) {
        TVal fv; fv.t = T_FN; fv.u.fn = f->fn;
        tval_free(&fv);
        f->fn = NULL;
    }
}

/* ---------- init/free ---------- */
void tvm_init(TVM *vm, TProg *p) {
    memset(vm, 0, sizeof(*vm));
    vm->prog = p;
    cur_vm = vm;
    vm->cap = STACK_INIT;
    vm->stack = (TVal*)malloc(vm->cap * sizeof(TVal));
    vm->fcap = FRAME_INIT;
    vm->frames = (TFrame*)malloc(vm->fcap * sizeof(TFrame));
    vm->hcap = HAND_INIT;
    vm->handlers = (THandler*)malloc(vm->hcap * sizeof(THandler));
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
        };
        size_t i;
        for (i = 0; i < sizeof(nt)/sizeof(nt[0]); i++) {
            if ((uint16_t)nt[i].idx < p->nglobals)
                p->globals[nt[i].idx] = mk_native(nt[i].fn, nt[i].name);
        }
    }
}
void tvm_free(TVM *vm) {
    size_t i;
    while (vm->sp) tval_free(&vm->stack[--vm->sp]);
    while (vm->nframes) {
        free_frame(vm);
        free(vm->frames[vm->nframes].locals);
        vm->frames[vm->nframes].locals = NULL;
        vm->frames[vm->nframes].lcap = 0;
    }
    if (vm->prog) {
        for (i = 0; i < vm->prog->nglobals; i++) tval_free(&vm->prog->globals[i]);
        /* free m???ng, val ???? free */
        free(vm->prog->globals);
        vm->prog->globals = NULL;
        /* d???n node GC c??n l???i (object ???? free h???t ??? tr??n) */
        {
            GCNode *nd = (GCNode*)vm->prog->gc_head;
            while (nd) {
                GCNode *nx = nd->next;
                free(nd);
                nd = nx;
            }
            vm->prog->gc_head = NULL;
        }
    }
    free(vm->stack); free(vm->frames); free(vm->handlers);
}

/* ---------- slot helpers (ng??? ngh??a LOAD/STORE, d??ng chung superop) ---------- */
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
    /* DECL/BIND: thay m???i d???t kho??t */
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

/* ---------- main loop ---------- */
static int glob_match(const char *s, size_t n, const char *p, size_t m) {
    /* *, ? (byte-wise; ASCII ?????, UTF-8 ??a byte so theo byte) */
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
    /* frame main */
    {
        TFrame *f = push_frame(vm);
        f->ret_pc = (size_t)-1; f->bp = 0; f->nlocals = 0; f->locals = NULL;
        f->fn = NULL;
    }
    in_top = 1;
    cur_vm = vm;
    if (setjmp(top_jb)) { in_top = 0; return -1; }
    for (;;) {
        uint8_t op;
        if (dbg_enabled() && !dbg_hook(vm)) return -1;
        if (vm->prog->gc_allocs >= GC_THRESHOLD) gc_collect(vm);
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
            uint32_t len = rd_u32(c, &vm->pc);
            if (vm->pc + len > n) die(vm, "bad str");
            TVal v = mk_str_n((char*)(c + vm->pc), len);
            vm->pc += len;
            push(vm, v); break;
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
            TTensor *t = (TTensor*)malloc(sizeof(TTensor));
            t->refc = 1; t->len = k;
            t->d = (double*)malloc(sizeof(double) * (k ? k : 1));
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
                    TVal dv; dv.t = T_DICT; dv.u.dict = d;
                    tval_free(&dv);
                    for (i = 0; i < k * 2; i++)
                        tval_free(&vm->stack[vm->sp - k * 2 + i]);
                    vm->sp -= k * 2;
                    die(vm, "unhashable dict key");
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
        case OP_ADD: case OP_SUB: case OP_MUL: case OP_DIV:
        case OP_MOD: case OP_POW: case OP_TDOT:
            arith(vm, op); break;
        case OP_NEG: {
            TVal a = pop(vm);
            if (a.t == T_INT) { int64_t i = a.u.i; tval_free(&a); push(vm, mk_int(-i)); }
            else if (a.t == T_FLOAT) { double x = a.u.f; tval_free(&a); push(vm, mk_float(-x)); }
            else if (a.t == T_TENSOR) {
                TTensor *t = (TTensor*)malloc(sizeof(TTensor));
                t->refc = 1; t->len = a.u.ten->len;
                t->d = (double*)malloc(sizeof(double) * (t->len ? t->len : 1));
                for (size_t i = 0; i < t->len; i++) t->d[i] = -a.u.ten->d[i];
                tval_free(&a);
                TVal v; v.t = T_TENSOR; v.u.ten = t;
                push(vm, v);
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
        case OP_EQ: case OP_NE: case OP_LT: case OP_GT: case OP_LE: case OP_GE:
            cmpop(vm, op); break;
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
            TVal ix = pop(vm), cv = pop(vm);
            TVal r; r.t = T_NIL;
            if (cv.t == T_DICT) {
                int i = dict_find(cv.u.dict, &ix);
                if (i < 0) {
                    char eb[128];
                    Buf b = {0,0,0};
                    fmt_into(&b, &ix);
                    snprintf(eb, sizeof(eb), "key %.60s missing", b.s ? b.s : "?");
                    free(b.s);
                    tval_free(&ix); tval_free(&cv); die(vm, eb);
                }
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
                    if (i < 0 || i >= (int64_t)cv.u.ten->len) {
                        char eb[96];
                        snprintf(eb, sizeof(eb), "index %lld out of range (len %zu)",
                                 (long long)i, cv.u.ten->len);
                        tval_free(&ix); tval_free(&cv); die(vm, eb);
                    }
                    r = mk_float(cv.u.ten->d[i]);
                } else {
                    tval_free(&ix); tval_free(&cv); die(vm, "index here");
                }
            }
            tval_free(&ix); tval_free(&cv);
            push(vm, r); break;
        }
        case OP_SETIDX: {
            TVal v = pop(vm), ix = pop(vm), cv = pop(vm);
            if (cv.t == T_LIST) {
                if (ix.t != T_INT) { tval_free(&v); tval_free(&ix); tval_free(&cv); die(vm, "index int"); }
                if (ix.u.i < 0 || ix.u.i >= (int64_t)cv.u.list->len) {
                    tval_free(&v); tval_free(&ix); tval_free(&cv); die(vm, "assign range");
                }
                tval_free(&cv.u.list->items[(size_t)ix.u.i]);
                cv.u.list->items[(size_t)ix.u.i] = v;
            } else if (cv.t == T_TENSOR) {
                int ok = 1; double x = to_num(&v, &ok);
                if (ix.t != T_INT || !ok) {
                    tval_free(&v); tval_free(&ix); tval_free(&cv); die(vm, "tensor assign");
                }
                if (ix.u.i < 0 || ix.u.i >= (int64_t)cv.u.ten->len) {
                    tval_free(&v); tval_free(&ix); tval_free(&cv); die(vm, "assign range");
                }
                cv.u.ten->d[(size_t)ix.u.i] = x;
                tval_free(&v);
            } else if (cv.t == T_DICT) {
                dict_set(cv.u.dict, ix, v);  /* nh???n ownership c??? key l???n val */
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
            else if (cv.t == T_TENSOR) n = (int64_t)cv.u.ten->len;
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
            if (a < 0) a += n;
            if (b < 0) b += n;
            if (s > 0) {
                if (a < 0) a = 0; if (a > n) a = n;
                if (b < 0) b = 0; if (b > n) b = n;
            } else {
                if (a < -1) a = -1; if (a >= n) a = n - 1;
                if (b < -1) b = -1; if (b >= n) b = n - 1;
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
                TTensor *t = (TTensor*)malloc(sizeof(TTensor));
                t->refc = 1;
                size_t cap = 4, len = 0;
                t->d = (double*)malloc(sizeof(double) * cap);
                if (s > 0) {
                    for (int64_t i = a; i < b; i += s) {
                        if (len >= cap) { cap *= 2; t->d = (double*)realloc(t->d, sizeof(double)*cap); }
                        t->d[len++] = cv.u.ten->d[(size_t)i];
                    }
                } else {
                    for (int64_t i = a; i > b; i += s) {
                        if (len >= cap) { cap *= 2; t->d = (double*)realloc(t->d, sizeof(double)*cap); }
                        t->d[len++] = cv.u.ten->d[(size_t)i];
                    }
                }
                t->len = len;
                r.t = T_TENSOR; r.u.ten = t;
            }
            tval_free(&st); tval_free(&hi); tval_free(&lo); tval_free(&cv);
            push(vm, r); break;
        }
        case OP_LEN: {
            TVal a = pop(vm);
            size_t k;
            if (a.t == T_STR) k = a.u.str.n;
            else if (a.t == T_LIST) k = a.u.list->len;
            else if (a.t == T_TENSOR) k = a.u.ten->len;
            else if (a.t == T_DICT) k = a.u.dict->len;
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
            int32_t rel = rd_i32(c, &vm->pc);
            vm->pc = (size_t)((int64_t)vm->pc + rel);
            break;
        }
        case OP_JMPF: case OP_JMPT: {
            int is_t = (op == OP_JMPT);
            int32_t rel = rd_i32(c, &vm->pc);
            TVal v = pop(vm);
            int t = tval_truthy(&v);
            tval_free(&v);
            if ((t && is_t) || (!t && !is_t))
                vm->pc = (size_t)((int64_t)vm->pc + rel);
            break;
        }
        case OP_CALL: {
            uint8_t k = c[vm->pc++];
            if (vm->sp < (size_t)k + 1) die(vm, "stack underflow");
            TVal fv = vm->stack[vm->sp - k - 1];
            if (fv.t == T_FN) {
                TFn *f = fv.u.fn;
                if (k != f->nargs) {
                    char eb[64];
                    snprintf(eb, sizeof(eb), "%.40s arity", "call");
                    die(vm, eb);
                }
                /* JIT hot-path: ?????m + th??? bi??n d???ch, r???i ch???y native n???u tag kh???p */
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
                        break;
                    }
                    /* deopt: r??? interpreter b??n d?????i */
                }
                TFrame *fr = push_frame(vm);
                size_t bp = vm->sp - k - 1;
                fr->ret_pc = vm->pc; fr->bp = bp;
                fr->nlocals = f->nlocals;
                fr->fn = f;
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
            break;
        }
        case OP_RET: {
            if (vm->nframes <= 1) die(vm, "ret outside");
            TVal rv = pop(vm);
            TFrame *fr = &vm->frames[vm->nframes - 1];
            size_t bp = fr->bp;
            size_t rpc = fr->ret_pc;
            free_frame(vm);
            vm->sp = bp;
            vm->pc = rpc;
            push(vm, rv);
            break;
        }
        case OP_LOAD: case OP_STORE: case OP_DECL: case OP_BIND: {
            uint16_t s = rd_u16(c, &vm->pc);
            if (op == OP_LOAD) {
                push(vm, slot_get(vm, 0, s));
                break;
            }
            {
                TVal v = pop(vm);
                if (op == OP_BIND) slot_replace(vm, 0, s, v);
                else slot_set(vm, 0, s, v);
            }
            break;
        }
        case OP_LOADG: case OP_STOREG: case OP_DECLG: {
            uint16_t s = rd_u16(c, &vm->pc);
            if (op == OP_LOADG) {
                push(vm, slot_get(vm, 1, s));
            } else {
                TVal v = pop(vm);
                slot_set(vm, 1, s, v);
            }
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
            f->ncalls = 0; f->tried = 0; f->nat_kind = 0;
            f->nat_op = 0; f->nat_t0 = 0; f->nat_t1 = 0; f->native = NULL;
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
                cell->refc = 0;             /* 1 cho slot + 1 cho fn (d?????i) */
                cell->val = *slot;          /* MOVE ownership t??? slot */
                slot->t = T_CELL; slot->u.cell = cell;
                cell->refc = 1;             /* slot gi??? */
            }
            {
                TVal cv; cv.t = T_CELL; cv.u.cell = cell;
                tval_retain(&cv);           /* fn gi??? */
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
                cell->val = *gp;   /* MOVE ownership t??? global */
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
            do_attr(vm, nm, len);
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
            a = slot_get(vm, isg, ds);
            b = slot_get(vm, isg, ss);
            push(vm, a);
            push(vm, b);
            arith(vm, bop);
            r = pop(vm);
            slot_set(vm, isg, ds, r);
            break;
        }
        case OP_NOP: break;
        case OP_HALT: in_top = 0; return 0;
        default: die(vm, "unknown opcode");
        }
    }
}




