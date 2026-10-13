/* tensor_impl.h — tensor n-chiều của VM C (include MỘT lần từ tvm.c).
 * Phải khớp từng bit với ttensor.py (hiện thực tham chiếu của t.py):
 *   float64 row-major, ndim 1..8; broadcast kiểu numpy; tổng/trung bình cộng
 *   tuần tự trái->phải; @ tích luỹ theo k tăng dần; ngẫu nhiên splitmix64
 *   (seed mặc định 42), randn Box-Muller một giá trị/lần.
 * Dùng các helper tĩnh của tvm.c: die, mk_*, list_new, list_push, to_num...
 */

#define TMAXDIM 8

static size_t sh_prod(const size_t *sh, int nd) {
    size_t n = 1;
    for (int i = 0; i < nd; i++) n *= sh[i];
    return n;
}

static TTensor *ten_alloc(int nd, const size_t *sh) {
    TTensor *t = (TTensor*)malloc(sizeof(TTensor));
    if (!t) exit(1);
    t->refc = 1;
    t->ndim = nd;
    for (int i = 0; i < nd; i++) t->shape[i] = sh[i];
    t->len = sh_prod(sh, nd);
    t->d = (double*)malloc(sizeof(double) * (t->len ? t->len : 1));
    if (!t->d) exit(1);
    return t;
}
static TTensor *ten_alloc1(size_t n) { return ten_alloc(1, &n); }
static TVal ten_val(TTensor *t) { TVal v; v.t = T_TENSOR; v.u.ten = t; return v; }
static TTensor *ten_copy(const TTensor *a) {
    TTensor *t = ten_alloc(a->ndim, a->shape);
    if (a->len) memcpy(t->d, a->d, sizeof(double) * a->len);
    return t;
}
static size_t ten_len0(const TTensor *t) { return t->ndim ? t->shape[0] : t->len; }

/* ---------- fmt lồng nhau ---------- */
static void fmt_double(double x, char *tmp, size_t n);
static void ten_fmt_rec(Buf *b, const TTensor *t, size_t off, int dim) {
    char tmp[64];
    size_t n = t->shape[dim];
    buf_c(b, '[');
    if (dim == t->ndim - 1) {
        for (size_t i = 0; i < n; i++) {
            if (i) buf_put(b, ", ", 2);
            fmt_double(t->d[off + i], tmp, sizeof(tmp));
            buf_put(b, tmp, strlen(tmp));
        }
    } else {
        size_t step = sh_prod(t->shape + dim + 1, t->ndim - dim - 1);
        for (size_t i = 0; i < n; i++) {
            if (i) buf_put(b, ", ", 2);
            ten_fmt_rec(b, t, off + i * step, dim + 1);
        }
    }
    buf_c(b, ']');
}
static void ten_fmt(Buf *b, const TTensor *t) {
    for (int i = 0; i + 1 < t->ndim; i++)
        if (t->shape[i] == 0) { buf_put(b, "[]", 2); return; }
    ten_fmt_rec(b, t, 0, 0);
}
static int ten_eq(const TTensor *a, const TTensor *b) {
    if (a->ndim != b->ndim) return 0;
    for (int i = 0; i < a->ndim; i++) if (a->shape[i] != b->shape[i]) return 0;
    for (size_t i = 0; i < a->len; i++) if (a->d[i] != b->d[i]) return 0;
    return 1;
}

/* ---------- tạo từ list lồng nhau ---------- */
static int ten_shape_of(const TVal *v, size_t *sh) {
    int nd = 0;
    const TVal *cur = v;
    while (cur->t == T_LIST) {
        if (nd >= TMAXDIM) return -1;
        sh[nd++] = cur->u.list->len;
        if (cur->u.list->len == 0) break;
        cur = &cur->u.list->items[0];
    }
    return nd;
}
static int ten_fill(const TVal *v, const size_t *sh, int nd, int dim, double *out, size_t *pos) {
    if (dim == nd) {
        int ok = 1;
        if (v->t == T_LIST) return 0;
        double x = to_num(v, &ok);
        if (!ok) return -1;
        out[(*pos)++] = x;
        return 1;
    }
    if (v->t != T_LIST || v->u.list->len != sh[dim]) return 0;
    for (size_t i = 0; i < v->u.list->len; i++) {
        int r = ten_fill(&v->u.list->items[i], sh, nd, dim + 1, out, pos);
        if (r <= 0) return r;
    }
    return 1;
}
static TTensor *ten_from(TVM *vm, const TVal *v) {
    if (v->t == T_TENSOR) return ten_copy(v->u.ten);
    if (v->t == T_LIST) {
        size_t sh[TMAXDIM];
        int nd = ten_shape_of(v, sh);
        if (nd < 0) die(vm, "tensor: quá 8 chiều");
        TTensor *t = ten_alloc(nd, sh);
        size_t pos = 0;
        int r = ten_fill(v, sh, nd, 0, t->d, &pos);
        if (r <= 0) {
            free(t->d); free(t);
            die(vm, r < 0 ? "tensor() numbers" : "tensor: ragged");
        }
        return t;
    }
    {
        int ok = 1;
        double x = to_num(v, &ok);
        if (!ok) die(vm, "tensor() numbers");
        TTensor *t = ten_alloc1(1);
        t->d[0] = x;
        return t;
    }
}

/* shape từ int hoặc list int */
static int ten_shape_arg(TVM *vm, const TVal *s, size_t *sh, const char *who) {
    char eb[64];
    snprintf(eb, sizeof(eb), "%s(shape)", who);
    if (s->t == T_INT) {
        if (s->u.i < 0) die(vm, eb);
        sh[0] = (size_t)s->u.i;
        return 1;
    }
    if (s->t != T_LIST || s->u.list->len == 0 || s->u.list->len > TMAXDIM) die(vm, eb);
    for (size_t i = 0; i < s->u.list->len; i++) {
        const TVal *x = &s->u.list->items[i];
        if (x->t != T_INT || x->u.i < 0) die(vm, eb);
        sh[i] = (size_t)x->u.i;
    }
    return (int)s->u.list->len;
}

/* ---------- broadcast ---------- */
static void ten_bshape(TVM *vm, const TTensor *a, const TTensor *b, int *ndo, size_t *out,
                       size_t *a2, size_t *b2) {
    int n = a->ndim > b->ndim ? a->ndim : b->ndim;
    for (int i = 0; i < n; i++) {
        int ia = i - (n - a->ndim), ib = i - (n - b->ndim);
        size_t x = ia >= 0 ? a->shape[ia] : 1, y = ib >= 0 ? b->shape[ib] : 1;
        a2[i] = x; b2[i] = y;
        if (x == y || y == 1) out[i] = x;
        else if (x == 1) out[i] = y;
        else die(vm, "tensor shape mismatch");
    }
    *ndo = n;
}
/* bảng chỉ số phẳng vào input (đã đệm 1) cho mỗi phần tử output */
static void ten_bindex(const size_t *out, int nd, const size_t *in, size_t *idx) {
    size_t st[TMAXDIM], eff[TMAXDIM], cnt[TMAXDIM];
    size_t acc = 1, n = sh_prod(out, nd), off = 0;
    for (int i = nd - 1; i >= 0; i--) { st[i] = acc; acc *= in[i]; }
    for (int i = 0; i < nd; i++) { eff[i] = in[i] == 1 ? 0 : st[i]; cnt[i] = 0; }
    for (size_t k = 0; k < n; k++) {
        idx[k] = off;
        for (int j = nd - 1; j >= 0; j--) {
            cnt[j]++;
            off += eff[j];
            if (cnt[j] < out[j]) break;
            off -= eff[j] * cnt[j];
            cnt[j] = 0;
        }
    }
}
#define TOP_EQ 1001   /* .eq() / .ne() từng phần tử (== của ngôn ngữ so cả tensor) */
#define TOP_NE 1002
static double ten_apply(int op, double x, double y) {
    switch (op) {
    case OP_ADD: return x + y;
    case OP_SUB: return x - y;
    case OP_MUL: return x * y;
    case OP_DIV: return x / y;
    case OP_LT: return x < y ? 1.0 : 0.0;
    case OP_GT: return x > y ? 1.0 : 0.0;
    case OP_LE: return x <= y ? 1.0 : 0.0;
    case OP_GE: return x >= y ? 1.0 : 0.0;
    case TOP_EQ: return x == y ? 1.0 : 0.0;
    case TOP_NE: return x != y ? 1.0 : 0.0;
    default: return pow(x, y);
    }
}
/* phép từng phần tử: tensor/số với tensor/số (ít nhất 1 tensor) */
static TTensor *ten_binop(TVM *vm, int op, const TVal *a, const TVal *b) {
    if (a->t != T_TENSOR || b->t != T_TENSOR) {
        int ok = 1;
        const TTensor *t = a->t == T_TENSOR ? a->u.ten : b->u.ten;
        double s = to_num(a->t == T_TENSOR ? b : a, &ok);
        if (!ok) die(vm, "bad tensor op");
        TTensor *r = ten_alloc(t->ndim, t->shape);
        if (a->t == T_TENSOR) for (size_t i = 0; i < t->len; i++) r->d[i] = ten_apply(op, t->d[i], s);
        else for (size_t i = 0; i < t->len; i++) r->d[i] = ten_apply(op, s, t->d[i]);
        return r;
    }
    {
        const TTensor *x = a->u.ten, *y = b->u.ten;
        int same = x->ndim == y->ndim;
        for (int i = 0; same && i < x->ndim; i++) same = x->shape[i] == y->shape[i];
        if (same) {
            TTensor *r = ten_alloc(x->ndim, x->shape);
            for (size_t i = 0; i < x->len; i++) r->d[i] = ten_apply(op, x->d[i], y->d[i]);
            return r;
        }
        {
            size_t out[TMAXDIM], a2[TMAXDIM], b2[TMAXDIM];
            int nd;
            ten_bshape(vm, x, y, &nd, out, a2, b2);
            TTensor *r = ten_alloc(nd, out);
            size_t *ia = (size_t*)malloc(sizeof(size_t) * (r->len ? r->len : 1));
            size_t *ib = (size_t*)malloc(sizeof(size_t) * (r->len ? r->len : 1));
            ten_bindex(out, nd, a2, ia);
            ten_bindex(out, nd, b2, ib);
            for (size_t k = 0; k < r->len; k++) r->d[k] = ten_apply(op, x->d[ia[k]], y->d[ib[k]]);
            free(ia); free(ib);
            return r;
        }
    }
}

/* ---------- @ ---------- */
static TVal ten_matmul(TVM *vm, const TTensor *a, const TTensor *b) {
    if (a->ndim == 1 && b->ndim == 1) {
        if (a->len != b->len) die(vm, "matmul shape");
        double acc = 0.0;
        for (size_t i = 0; i < a->len; i++) acc += a->d[i] * b->d[i];
        return mk_float(acc);
    }
    if (a->ndim > 2 || b->ndim > 2) die(vm, "matmul: tối đa 2 chiều");
    {
        size_t m = a->ndim == 1 ? 1 : a->shape[0], k = a->ndim == 1 ? a->shape[0] : a->shape[1];
        size_t k2 = b->shape[0], n = b->ndim == 1 ? 1 : b->shape[1];
        if (k != k2) die(vm, "matmul shape");
        size_t sh[2];
        int nd;
        if (a->ndim == 1) { sh[0] = n; nd = 1; }
        else if (b->ndim == 1) { sh[0] = m; nd = 1; }
        else { sh[0] = m; sh[1] = n; nd = 2; }
        TTensor *r = ten_alloc(nd, sh);
        double *C = r->d;
        const double *A = a->d, *B = b->d;
        for (size_t i = 0; i < m * n; i++) C[i] = 0.0;
        /* thứ tự i-k-j: mỗi C[i,j] cộng dồn theo k tăng dần (trùng bit với bản
           tham chiếu), vòng j trong cùng chạy liên tục bộ nhớ (vector hoá được) */
        for (size_t i = 0; i < m; i++) {
            double *Ci = C + i * n;
            const double *Ai = A + i * k;
            for (size_t q = 0; q < k; q++) {
                double aiq = Ai[q];
                const double *Bq = B + q * n;
                for (size_t j = 0; j < n; j++) Ci[j] += aiq * Bq[j];
            }
        }
        return ten_val(r);
    }
}

/* ---------- chỉ số theo trục 0 ---------- */
static TVal ten_getitem(TVM *vm, const TTensor *t, int64_t i) {
    size_t n = ten_len0(t);
    if (i < 0 || (uint64_t)i >= n) {
        char eb[96];
        snprintf(eb, sizeof(eb), "index %lld out of range (len %zu)", (long long)i, n);
        die(vm, eb);
    }
    if (t->ndim <= 1) return mk_float(t->d[i]);
    {
        size_t step = sh_prod(t->shape + 1, t->ndim - 1);
        TTensor *r = ten_alloc(t->ndim - 1, t->shape + 1);
        if (step) memcpy(r->d, t->d + (size_t)i * step, sizeof(double) * step);
        return ten_val(r);
    }
}
/* trả 0 nếu lỗi (thông điệp vào eb) */
static int ten_setitem(TTensor *t, int64_t i, const TVal *v, char *eb, size_t ebn) {
    size_t n = ten_len0(t);
    if (i < 0 || (uint64_t)i >= n) { snprintf(eb, ebn, "assign range"); return 0; }
    if (t->ndim <= 1) {
        int ok = 1;
        double x = to_num(v, &ok);
        if (!ok || v->t == T_BOOL) { snprintf(eb, ebn, "tensor assign"); return 0; }
        t->d[i] = x;
        return 1;
    }
    {
        size_t step = sh_prod(t->shape + 1, t->ndim - 1);
        if (v->t == T_TENSOR) {
            const TTensor *s = v->u.ten;
            int ok = s->ndim == t->ndim - 1;
            for (int q = 0; ok && q < s->ndim; q++) ok = s->shape[q] == t->shape[q + 1];
            if (!ok) { snprintf(eb, ebn, "tensor assign shape"); return 0; }
            if (step) memmove(t->d + (size_t)i * step, s->d, sizeof(double) * step);
            return 1;
        }
        {
            int ok = 1;
            double x = to_num(v, &ok);
            if (!ok || v->t == T_BOOL) { snprintf(eb, ebn, "tensor assign"); return 0; }
            for (size_t q = 0; q < step; q++) t->d[(size_t)i * step + q] = x;
            return 1;
        }
    }
}
/* chỉ số nhiều trục t[[i, nil, ...]]: int chọn (bỏ trục), nil giữ cả trục,
   thiếu trục cuối = giữ. Trả số vị trí (cấp *pos) + shape kết quả; lỗi -> die */
/* chỉ số của khoảng lo:hi:step trên trục dài n (ngữ nghĩa slice của Python) */
static size_t slice_ix(int64_t n, const TVal *lo, const TVal *hi, const TVal *sp, int64_t *out) {
    int64_t s = sp->t == T_INT ? sp->u.i : 1, a, b;
    size_t c = 0;
    if (lo->t == T_NIL) a = s > 0 ? 0 : n - 1;
    else { a = lo->u.i; if (a < 0) a += n; }
    if (hi->t == T_NIL) b = s > 0 ? n : -1;
    else { b = hi->u.i; if (b < 0) b += n; }
    if (s > 0) {
        if (a < 0) a = 0;
        if (a > n) a = n;
        if (b < 0) b = 0;
        if (b > n) b = n;
        for (int64_t i = a; i < b; i += s) out[c++] = i;
    } else {
        if (a < -1) a = -1;
        if (a >= n) a = n - 1;
        if (b < -1) b = -1;
        if (b >= n) b = n - 1;
        for (int64_t i = a; i > b; i += s) out[c++] = i;
    }
    return c;
}
static size_t ten_multi_pos(TVM *vm, const TTensor *t, const TList *key, int *rnd, size_t *rsh,
                            size_t **pos) {
    size_t st[TMAXDIM], acc = 1, base = 0, cnt[TMAXDIM];
    int kept[TMAXDIM], nk = 0;
    int64_t *ix[TMAXDIM];
    char eb[96];
    if ((int)key->len > t->ndim) die(vm, "too many indices");
    for (int i = t->ndim - 1; i >= 0; i--) { st[i] = acc; acc *= t->shape[i]; }
    for (int ax = 0; ax < t->ndim; ax++) {
        const TVal *k = (size_t)ax < key->len ? &key->items[ax] : NULL;
        int64_t n = (int64_t)t->shape[ax];
        if (!k || k->t == T_NIL || k->t == T_LIST) {
            int64_t *lst = (int64_t*)malloc(sizeof(int64_t) * (size_t)(n ? n : 1));
            size_t c;
            if (!k || k->t == T_NIL) {
                for (int64_t i = 0; i < n; i++) lst[i] = i;
                c = (size_t)n;
            } else {
                const TList *sl = k->u.list;
                if (sl->len != 3) { free(lst); for (int q = 0; q < nk; q++) free(ix[q]); die(vm, "slice [lo, hi, step]"); }
                for (int q = 0; q < 3; q++)
                    if (sl->items[q].t != T_NIL && sl->items[q].t != T_INT) {
                        free(lst); for (int z = 0; z < nk; z++) free(ix[z]); die(vm, "slice int");
                    }
                if (sl->items[2].t == T_INT && sl->items[2].u.i == 0) {
                    free(lst); for (int z = 0; z < nk; z++) free(ix[z]); die(vm, "slice step 0");
                }
                c = slice_ix(n, &sl->items[0], &sl->items[1], &sl->items[2], lst);
            }
            ix[nk] = lst;
            rsh[nk] = c;
            kept[nk++] = ax;
            continue;
        }
        if (k->t != T_INT) { for (int q = 0; q < nk; q++) free(ix[q]); die(vm, "index int"); }
        if (k->u.i < 0 || (uint64_t)k->u.i >= t->shape[ax]) {
            for (int q = 0; q < nk; q++) free(ix[q]);
            snprintf(eb, sizeof(eb), "index %lld out of range (len %zu)", (long long)k->u.i, t->shape[ax]);
            die(vm, eb);
        }
        base += (size_t)k->u.i * st[ax];
    }
    size_t n = 1;
    for (int q = 0; q < nk; q++) { n *= rsh[q]; cnt[q] = 0; }
    *rnd = nk;
    size_t *p = (size_t*)malloc(sizeof(size_t) * (n ? n : 1));
    for (size_t w = 0; w < n; w++) {
        size_t off = base;
        for (int q = 0; q < nk; q++) off += (size_t)ix[q][cnt[q]] * st[kept[q]];
        p[w] = off;
        for (int q = nk - 1; q >= 0; q--) {
            if (++cnt[q] < rsh[q]) break;
            cnt[q] = 0;
        }
    }
    for (int q = 0; q < nk; q++) free(ix[q]);
    *pos = p;
    return n;
}
static TVal ten_get_multi(TVM *vm, const TTensor *t, const TList *key) {
    size_t sh[TMAXDIM], *pos;
    int nd;
    size_t n = ten_multi_pos(vm, t, key, &nd, sh, &pos);
    if (nd == 0) { double x = t->d[pos[0]]; free(pos); return mk_float(x); }
    TTensor *r = ten_alloc(nd, sh);
    for (size_t i = 0; i < n; i++) r->d[i] = t->d[pos[i]];
    free(pos);
    return ten_val(r);
}
/* trả 0 + eb nếu lỗi kiểu/shape giá trị */
static int ten_set_multi(TVM *vm, TTensor *t, const TList *key, const TVal *v, char *eb, size_t ebn) {
    size_t sh[TMAXDIM], *pos;
    int nd;
    size_t n = ten_multi_pos(vm, t, key, &nd, sh, &pos);
    if (v->t == T_TENSOR) {
        const TTensor *s = v->u.ten;
        int ok = nd == 0 ? (s->ndim == 1 && s->shape[0] == 1) : s->ndim == nd;
        for (int q = 0; ok && nd && q < nd; q++) ok = s->shape[q] == sh[q];
        if (!ok) { free(pos); snprintf(eb, ebn, "tensor assign shape"); return 0; }
        for (size_t i = 0; i < n; i++) t->d[pos[i]] = s->d[i];
    } else {
        int ok = 1;
        double x = to_num(v, &ok);
        if (!ok || v->t == T_BOOL) { free(pos); snprintf(eb, ebn, "tensor assign"); return 0; }
        for (size_t i = 0; i < n; i++) t->d[pos[i]] = x;
    }
    free(pos);
    return 1;
}

/* lát cắt theo trục 0 với danh sách chỉ số */
static TTensor *ten_take0(const TTensor *t, const int64_t *ix, size_t cnt) {
    size_t step = t->ndim <= 1 ? 1 : sh_prod(t->shape + 1, t->ndim - 1);
    size_t sh[TMAXDIM];
    int nd = t->ndim ? t->ndim : 1;
    sh[0] = cnt;
    for (int q = 1; q < nd; q++) sh[q] = t->shape[q];
    TTensor *r = ten_alloc(nd, sh);
    for (size_t k = 0; k < cnt; k++)
        if (step) memcpy(r->d + k * step, t->d + (size_t)ix[k] * step, sizeof(double) * step);
    return r;
}

/* ---------- reshape / transpose / tolist ---------- */
static TTensor *ten_transpose(const TTensor *t) {
    if (t->ndim <= 1) return ten_copy(t);
    size_t sh[TMAXDIM], st0[TMAXDIM], st[TMAXDIM], cnt[TMAXDIM];
    int nd = t->ndim;
    size_t acc = 1;
    for (int i = nd - 1; i >= 0; i--) { st0[i] = acc; acc *= t->shape[i]; }
    for (int i = 0; i < nd; i++) { sh[i] = t->shape[nd - 1 - i]; st[i] = st0[nd - 1 - i]; cnt[i] = 0; }
    TTensor *r = ten_alloc(nd, sh);
    size_t off = 0;
    for (size_t k = 0; k < r->len; k++) {
        r->d[k] = t->d[off];
        for (int j = nd - 1; j >= 0; j--) {
            cnt[j]++;
            off += st[j];
            if (cnt[j] < sh[j]) break;
            off -= st[j] * cnt[j];
            cnt[j] = 0;
        }
    }
    return r;
}
static TVal ten_tolist_rec(const TTensor *t, size_t off, int dim) {
    size_t n = t->shape[dim];
    TList *l = list_new(n ? n : 1);
    if (dim == t->ndim - 1) {
        for (size_t i = 0; i < n; i++) list_push(l, mk_float(t->d[off + i]));
    } else {
        size_t step = sh_prod(t->shape + dim + 1, t->ndim - dim - 1);
        for (size_t i = 0; i < n; i++) list_push(l, ten_tolist_rec(t, off + i * step, dim + 1));
    }
    TVal v; v.t = T_LIST; v.u.list = l;
    return v;
}

/* ---------- rút gọn ---------- */
enum { R_SUM, R_MEAN, R_MAX, R_MIN, R_ARGMAX, R_ARGMIN };
static double red_run(int kind, const double *x, size_t n, size_t stride, int *err) {
    size_t i;
    double acc;
    *err = 0;
    switch (kind) {
    case R_SUM: case R_MEAN:
        acc = 0.0;
        for (i = 0; i < n; i++) acc += x[i * stride];
        if (kind == R_MEAN) return n ? acc / (double)n : 0.0;
        return acc;
    case R_MAX: case R_MIN: {
        if (!n) { *err = 1; return 0; }
        double m = x[0];
        for (i = 1; i < n; i++) {
            double v = x[i * stride];
            if (kind == R_MAX ? v > m : v < m) m = v;
        }
        return m;
    }
    default: {
        if (!n) { *err = 1; return 0; }
        size_t bi = 0;
        for (i = 1; i < n; i++) {
            double v = x[i * stride], b = x[bi * stride];
            if (kind == R_ARGMAX ? v > b : v < b) bi = i;
        }
        return (double)bi;
    }
    }
}
static const char *red_name(int kind) {
    static const char *nm[] = {"sum", "mean", "max", "min", "argmax", "argmin"};
    return nm[kind];
}
static void ten_reduce(TVM *vm, int kind, TVal *a, int argc, TVal *r) {
    const TTensor *t = a[0].u.ten;
    int err;
    char eb[64];
    if (argc == 1) {
        double v = red_run(kind, t->d, t->len, 1, &err);
        if (err) { snprintf(eb, sizeof(eb), "%s() empty", red_name(kind)); die(vm, eb); }
        *r = (kind == R_ARGMAX || kind == R_ARGMIN) ? mk_int((int64_t)v) : mk_float(v);
        return;
    }
    if (argc != 2) { snprintf(eb, sizeof(eb), "%s([axis])", red_name(kind)); die(vm, eb); }
    if (a[1].t != T_INT) die(vm, "axis int");
    {
        int64_t ax = a[1].u.i;
        if (ax < 0) ax += t->ndim;
        if (ax < 0 || ax >= t->ndim) die(vm, "axis out of range");
        size_t outer = sh_prod(t->shape, (int)ax), n = t->shape[ax];
        size_t inner = sh_prod(t->shape + ax + 1, t->ndim - (int)ax - 1);
        size_t osh[TMAXDIM];
        int ond = 0;
        for (int i = 0; i < t->ndim; i++) if (i != ax) osh[ond++] = t->shape[i];
        if (ond == 0) { osh[0] = 1; ond = 1; }
        TTensor *o = ten_alloc(ond, osh);
        size_t w = 0;
        for (size_t p = 0; p < outer; p++)
            for (size_t i = 0; i < inner; i++) {
                double v = red_run(kind, t->d + p * n * inner + i, n, inner, &err);
                if (err) {
                    free(o->d); free(o);
                    snprintf(eb, sizeof(eb), "%s() empty", red_name(kind)); die(vm, eb);
                }
                o->d[w++] = v;
            }
        *r = ten_val(o);
    }
}

/* ---------- hàm từng phần tử ---------- */
enum { U_EXP, U_LOG, U_SQRT, U_TANH, U_SIN, U_COS, U_ABS, U_RELU, U_SIGMOID, U_FLOOR, U_CEIL };
static double un_apply(int k, double x) {
    switch (k) {
    case U_EXP: return exp(x);
    case U_LOG: return x > 0 ? log(x) : (x == 0 ? -HUGE_VAL : NAN);
    case U_SQRT: return x >= 0 ? sqrt(x) : NAN;
    case U_TANH: return tanh(x);
    case U_SIN: return sin(x);
    case U_COS: return cos(x);
    case U_ABS: return fabs(x);
    case U_RELU: return x > 0 ? x : 0.0;
    case U_SIGMOID: return 1.0 / (1.0 + exp(-x));
    case U_FLOOR: return floor(x);
    default: return ceil(x);
    }
}
static TTensor *ten_unary(const TTensor *t, int k) {
    TTensor *r = ten_alloc(t->ndim, t->shape);
    for (size_t i = 0; i < t->len; i++) r->d[i] = un_apply(k, t->d[i]);
    return r;
}

/* ---------- ngẫu nhiên: splitmix64 ---------- */
static uint64_t rng_state = 42;
static uint64_t rng_next(void) {
    rng_state += 0x9E3779B97F4A7C15ULL;
    uint64_t z = rng_state;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}
static double rng_unif(void) { return (double)(rng_next() >> 11) * (1.0 / 9007199254740992.0); }
static double rng_gauss(void) {
    double u1 = 1.0 - rng_unif();
    double u2 = rng_unif();
    return sqrt(-2.0 * log(u1)) * cos(2.0 * 3.141592653589793 * u2);
}

/* ---------- method tensor (a[0] = tensor) ---------- */
#define TENM(name, kind) static void name(TVM *vm, TVal *a, int argc, TVal *r) { ten_reduce(vm, kind, a, argc, r); }
TENM(m_ten_sum, R_SUM)
TENM(m_ten_mean, R_MEAN)
TENM(m_ten_max, R_MAX)
TENM(m_ten_min, R_MIN)
TENM(m_ten_argmax, R_ARGMAX)
TENM(m_ten_argmin, R_ARGMIN)
#define TENU(name, kind) static void name(TVM *vm, TVal *a, int argc, TVal *r) { \
    NARG(1); *r = ten_val(ten_unary(a[0].u.ten, kind)); }
TENU(m_ten_exp, U_EXP)
TENU(m_ten_log, U_LOG)
TENU(m_ten_sqrt, U_SQRT)
TENU(m_ten_tanh, U_TANH)
TENU(m_ten_sin, U_SIN)
TENU(m_ten_cos, U_COS)
TENU(m_ten_abs, U_ABS)
TENU(m_ten_relu, U_RELU)
TENU(m_ten_sigmoid, U_SIGMOID)
TENU(m_ten_floor, U_FLOOR)
TENU(m_ten_ceil, U_CEIL)

static void m_ten_reshape(TVM *vm, TVal *a, int argc, TVal *r) {
    const TTensor *t = a[0].u.ten;
    TVal *xs = a + 1;
    int n = argc - 1;
    size_t sh[TMAXDIM];
    int neg = -1;
    if (n == 1 && xs[0].t == T_LIST) { n = (int)xs[0].u.list->len; xs = xs[0].u.list->items; }
    if (n < 1 || n > TMAXDIM) die(vm, "reshape shape");
    size_t rest = 1;
    for (int i = 0; i < n; i++) {
        if (xs[i].t != T_INT) die(vm, "reshape(int...)");
        if (xs[i].u.i == -1) {
            if (neg >= 0) die(vm, "reshape shape");
            neg = i; continue;
        }
        if (xs[i].u.i < 0) die(vm, "reshape shape");
        sh[i] = (size_t)xs[i].u.i;
        rest *= sh[i];
    }
    if (neg >= 0) {
        if (rest == 0 || t->len % rest) die(vm, "reshape shape");
        sh[neg] = t->len / rest;
        rest *= sh[neg];
    }
    if (rest != t->len) die(vm, "reshape shape");
    TTensor *o = ten_alloc(n, sh);
    if (t->len) memcpy(o->d, t->d, sizeof(double) * t->len);
    *r = ten_val(o);
}
static void m_ten_transpose(TVM *vm, TVal *a, int argc, TVal *r) { NARG(1); *r = ten_val(ten_transpose(a[0].u.ten)); }
static void m_ten_flatten(TVM *vm, TVal *a, int argc, TVal *r) {
    NARG(1);
    TTensor *o = ten_alloc1(a[0].u.ten->len);
    if (o->len) memcpy(o->d, a[0].u.ten->d, sizeof(double) * o->len);
    *r = ten_val(o);
}
static void m_ten_tolist(TVM *vm, TVal *a, int argc, TVal *r) { NARG(1); *r = ten_tolist_rec(a[0].u.ten, 0, 0); }
static void m_ten_copy(TVM *vm, TVal *a, int argc, TVal *r) { NARG(1); *r = ten_val(ten_copy(a[0].u.ten)); }
static void m_ten_dot(TVM *vm, TVal *a, int argc, TVal *r) {
    NARG(2);
    if (a[1].t != T_TENSOR) die(vm, "matmul cần 2 tensor");
    *r = ten_matmul(vm, a[0].u.ten, a[1].u.ten);
}
static void m_ten_clip(TVM *vm, TVal *a, int argc, TVal *r) {
    NARG(3);
    int ok1 = 1, ok2 = 1;
    double lo = to_num(&a[1], &ok1), hi = to_num(&a[2], &ok2);
    if (!ok1 || !ok2 || a[1].t == T_BOOL || a[2].t == T_BOOL) die(vm, "tensor() numbers");
    const TTensor *t = a[0].u.ten;
    TTensor *o = ten_alloc(t->ndim, t->shape);
    for (size_t i = 0; i < t->len; i++) {
        double x = t->d[i];
        o->d[i] = x < lo ? lo : (x > hi ? hi : x);
    }
    *r = ten_val(o);
}
static void m_ten_softmax(TVM *vm, TVal *a, int argc, TVal *r) {
    NARG(1);
    const TTensor *t = a[0].u.ten;
    size_t n = t->shape[t->ndim - 1];
    TTensor *o = ten_alloc(t->ndim, t->shape);
    if (n == 0) { *r = ten_val(o); return; }
    for (size_t off = 0; off < t->len; off += n) {
        double m = t->d[off];
        for (size_t i = 1; i < n; i++) if (t->d[off + i] > m) m = t->d[off + i];
        double s = 0.0;
        for (size_t i = 0; i < n; i++) { o->d[off + i] = exp(t->d[off + i] - m); s += o->d[off + i]; }
        for (size_t i = 0; i < n; i++) o->d[off + i] = o->d[off + i] / s;
    }
    *r = ten_val(o);
}
static void ten_cmp_m(TVM *vm, TVal *a, int argc, TVal *r, int op) {
    NARG(2);
    if (a[1].t != T_TENSOR && a[1].t != T_INT && a[1].t != T_FLOAT) die(vm, "bad tensor op");
    *r = ten_val(ten_binop(vm, op, &a[0], &a[1]));
}
static void m_ten_eq(TVM *vm, TVal *a, int argc, TVal *r) { ten_cmp_m(vm, a, argc, r, TOP_EQ); }
static void m_ten_ne(TVM *vm, TVal *a, int argc, TVal *r) { ten_cmp_m(vm, a, argc, r, TOP_NE); }
static void m_ten_item(TVM *vm, TVal *a, int argc, TVal *r) {
    NARG(1);
    if (a[0].u.ten->len != 1) die(vm, "item() cần tensor 1 phần tử");
    *r = mk_float(a[0].u.ten->d[0]);
}

static TNativeFn tensor_method(const char *nb) {
    static const struct { const char *n; TNativeFn f; } tb[] = {
        {"sum", m_ten_sum}, {"mean", m_ten_mean}, {"max", m_ten_max}, {"min", m_ten_min},
        {"argmax", m_ten_argmax}, {"argmin", m_ten_argmin},
        {"exp", m_ten_exp}, {"log", m_ten_log}, {"sqrt", m_ten_sqrt}, {"tanh", m_ten_tanh},
        {"sin", m_ten_sin}, {"cos", m_ten_cos}, {"abs", m_ten_abs}, {"relu", m_ten_relu},
        {"sigmoid", m_ten_sigmoid}, {"floor", m_ten_floor}, {"ceil", m_ten_ceil},
        {"reshape", m_ten_reshape}, {"transpose", m_ten_transpose}, {"flatten", m_ten_flatten},
        {"tolist", m_ten_tolist}, {"copy", m_ten_copy}, {"dot", m_ten_dot}, {"clip", m_ten_clip},
        {"softmax", m_ten_softmax}, {"item", m_ten_item}, {"eq", m_ten_eq}, {"ne", m_ten_ne},
    };
    for (size_t i = 0; i < sizeof(tb) / sizeof(tb[0]); i++) if (!strcmp(nb, tb[i].n)) return tb[i].f;
    return NULL;
}
static size_t tensor_len0(const TTensor *t) { return ten_len0(t); }
/* thuộc tính không phải method: shape, ndim, size, T */
static TVal tensor_prop(TVM *vm, TTensor *t, const char *nb, int *ok) {
    TVal r; r.t = T_NIL;
    (void)vm;
    *ok = 1;
    if (!strcmp(nb, "shape")) {
        TList *l = list_new((size_t)t->ndim);
        for (int i = 0; i < t->ndim; i++) list_push(l, mk_int((int64_t)t->shape[i]));
        r.t = T_LIST; r.u.list = l;
        return r;
    }
    if (!strcmp(nb, "ndim")) return mk_int(t->ndim);
    if (!strcmp(nb, "size")) return mk_int((int64_t)t->len);
    if (!strcmp(nb, "T")) return ten_val(ten_transpose(t));
    *ok = 0;
    return r;
}

/* ---------- builtin mềm: tạo tensor + toán học ---------- */
static void n_zeros(TVM *vm, TVal *a, int argc, TVal *r) {
    size_t sh[TMAXDIM];
    NARG(1);
    int nd = ten_shape_arg(vm, &a[0], sh, "zeros");
    TTensor *t = ten_alloc(nd, sh);
    for (size_t i = 0; i < t->len; i++) t->d[i] = 0.0;
    *r = ten_val(t);
}
static void n_ones(TVM *vm, TVal *a, int argc, TVal *r) {
    size_t sh[TMAXDIM];
    NARG(1);
    int nd = ten_shape_arg(vm, &a[0], sh, "ones");
    TTensor *t = ten_alloc(nd, sh);
    for (size_t i = 0; i < t->len; i++) t->d[i] = 1.0;
    *r = ten_val(t);
}
static void n_full(TVM *vm, TVal *a, int argc, TVal *r) {
    size_t sh[TMAXDIM];
    int ok = 1;
    NARG(2);
    int nd = ten_shape_arg(vm, &a[0], sh, "full");
    double v = to_num(&a[1], &ok);
    if (!ok || a[1].t == T_BOOL) die(vm, "tensor() numbers");
    TTensor *t = ten_alloc(nd, sh);
    for (size_t i = 0; i < t->len; i++) t->d[i] = v;
    *r = ten_val(t);
}
static void n_arange(TVM *vm, TVal *a, int argc, TVal *r) {
    double x[3];
    if (argc < 1 || argc > 3) die(vm, "arange(n) / arange(a, b[, step])");
    for (int i = 0; i < argc; i++) {
        int ok = 1;
        x[i] = to_num(&a[i], &ok);
        if (!ok || a[i].t == T_BOOL) die(vm, "tensor() numbers");
    }
    double lo = argc == 1 ? 0.0 : x[0], hi = argc == 1 ? x[0] : x[1], st = argc == 3 ? x[2] : 1.0;
    if (st == 0) die(vm, "arange step 0");
    double nn = ceil((hi - lo) / st);
    size_t n = nn > 0 ? (size_t)nn : 0;
    TTensor *t = ten_alloc1(n);
    for (size_t i = 0; i < n; i++) t->d[i] = lo + (double)i * st;
    *r = ten_val(t);
}
static void n_eye(TVM *vm, TVal *a, int argc, TVal *r) {
    NARG(1);
    if (a[0].t != T_INT || a[0].u.i < 0) die(vm, "eye(n)");
    size_t n = (size_t)a[0].u.i, sh[2] = {n, n};
    TTensor *t = ten_alloc(2, sh);
    for (size_t i = 0; i < n * n; i++) t->d[i] = 0.0;
    for (size_t i = 0; i < n; i++) t->d[i * n + i] = 1.0;
    *r = ten_val(t);
}
static void n_rand(TVM *vm, TVal *a, int argc, TVal *r) {
    size_t sh[TMAXDIM];
    NARG(1);
    int nd = ten_shape_arg(vm, &a[0], sh, "rand");
    TTensor *t = ten_alloc(nd, sh);
    for (size_t i = 0; i < t->len; i++) t->d[i] = rng_unif();
    *r = ten_val(t);
}
static void n_randn(TVM *vm, TVal *a, int argc, TVal *r) {
    size_t sh[TMAXDIM];
    NARG(1);
    int nd = ten_shape_arg(vm, &a[0], sh, "randn");
    TTensor *t = ten_alloc(nd, sh);
    for (size_t i = 0; i < t->len; i++) t->d[i] = rng_gauss();
    *r = ten_val(t);
}
static void n_seed(TVM *vm, TVal *a, int argc, TVal *r) {
    NARG(1);
    if (a[0].t != T_INT) die(vm, "seed(int)");
    rng_state = (uint64_t)a[0].u.i;
    r->t = T_NIL;
}
/* toán học: số -> float (inf/nan như C); tensor -> từng phần tử */
static void math1(TVM *vm, TVal *a, int argc, TVal *r, int k, const char *who) {
    char eb[48];
    if (argc != 1) { snprintf(eb, sizeof(eb), "%s(x)", who); die(vm, eb); }
    if (a[0].t == T_TENSOR) { *r = ten_val(ten_unary(a[0].u.ten, k)); return; }
    if (a[0].t != T_INT && a[0].t != T_FLOAT) { snprintf(eb, sizeof(eb), "%s() number", who); die(vm, eb); }
    {
        double x = a[0].t == T_INT ? (double)a[0].u.i : a[0].u.f;
        if (k == U_FLOOR || k == U_CEIL) {
            if (a[0].t == T_INT) { *r = a[0]; return; }
            if (x != x || x == HUGE_VAL || x == -HUGE_VAL) { snprintf(eb, sizeof(eb), "%s() finite", who); die(vm, eb); }
            *r = mk_int((int64_t)un_apply(k, x));
            return;
        }
        *r = mk_float(un_apply(k, x));
    }
}
#define MATHN(name, k, s) static void name(TVM *vm, TVal *a, int argc, TVal *r) { math1(vm, a, argc, r, k, s); }
MATHN(n_sqrt, U_SQRT, "sqrt")
MATHN(n_exp, U_EXP, "exp")
MATHN(n_log, U_LOG, "log")
MATHN(n_sin, U_SIN, "sin")
MATHN(n_cos, U_COS, "cos")
MATHN(n_tanh, U_TANH, "tanh")
MATHN(n_floor, U_FLOOR, "floor")
MATHN(n_ceil, U_CEIL, "ceil")
