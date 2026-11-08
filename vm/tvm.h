/* tvm.h — T bytecode VM v1: value model, opcodes, VM state.
 * Stack machine + call frames + refcount + try (setjmp) + natives.
 * File .tb: magic "TB01" + u16 nglobals + u32 len + code bytes.
 */
#ifndef TVM_H
#define TVM_H

#include <stdint.h>
#include <stddef.h>
#include <stdio.h>
#include <setjmp.h>

#ifdef _MSC_VER
#define TVM_INLINE static __forceinline
#else
#define TVM_INLINE static inline __attribute__((always_inline))
#endif

typedef enum {
    T_NIL, T_BOOL, T_INT, T_FLOAT, T_STR, T_LIST, T_TENSOR, T_DICT,
    T_FN, T_NATIVE, T_FILE, T_CELL, T_UNDEF
} TType;

typedef struct TVal TVal;
typedef struct TList TList;
typedef struct TTensor TTensor;
typedef struct TDict TDict;
typedef struct TFn TFn;
typedef struct TNative TNative;
typedef struct TFile TFile;
typedef struct TCell TCell;
typedef struct TVM TVM;
typedef struct GCNode GCNode;
struct GCNode {
    int type;
    void *ptr;
    GCNode *prev, *next;
};

struct TList {
    int32_t refc;
    int gc_mark;
    GCNode *gc_node;
    size_t len, cap;
    TVal *items;
};

enum {
    TD_F64  = 0,
    TD_F32  = 1,
    TD_F16  = 2,
    TD_BF16 = 3,
    TD_I32  = 4,
    TD_I8   = 5
};

struct TTensor {
    int32_t refc;
    size_t len;         /* tổng số phần tử */
    double *d;          /* row-major dày (view liền: trỏ vào bộ nhớ của base;
                           view theo bước: cache, xem ten_sync) */
    void *raw;          /* G14: buffer nén theo dtype (float32/fp16/bf16/int32/int8); NULL nếu TD_F64 */
    int dtype;          /* TD_F64 (mặc định = 0) .. TD_I8 */
    TTensor *base;      /* view: tensor sở hữu bộ nhớ (giữ ref); NULL = tự sở hữu */
    int ndim;           /* 1..8 */
    size_t shape[8];
    int sv;             /* 1 = view theo bước: phần tử ở base->d[off + Σ i_k*st[k]] */
    size_t off;
    ptrdiff_t st[8];
    uint64_t ver;       /* chủ bộ nhớ: tăng mỗi lần ghi; view theo bước: ver của cache */
};

#define DK_DICT 0
#define DK_SET  1
#define DK_REC  2   /* record: keys MƯỢN mảng tên field của kiểu; meths = kiểu (DK_TYPE) */
#define DK_TYPE 3   /* kiểu struct: keys/vals = bảng method; fields = list tên field */
struct TProg;
struct TDict {
    struct TProg *mod;  /* DK_TYPE của bản ghi module (load): field = global của prog này */
    int32_t refc;
    int gc_mark;
    GCNode *gc_node;
    size_t len, cap;
    TVal *keys;
    TVal *vals;
    int kind;           /* DK_* */
    char *rname;        /* DK_REC: tên struct (malloc) */
    TDict *meths;       /* DK_REC: bảng method (giữ ref) hoặc NULL */
    TList *fields;      /* DK_TYPE: tên field (giữ ref) */
    int32_t *idx;       /* bảng băm mở: chỉ số entry (-1 trống); NULL khi dict nhỏ */
    size_t icap;        /* kích thước idx (luỹ thừa 2) */
};

struct TProg;
struct TFn {
    struct TProg *prog; /* chương trình chứa code của hàm (module nạp bằng load()) */
    int32_t refc;
    int gc_mark;
    GCNode *gc_node;
    int32_t entry;
    uint8_t nargs;
    uint8_t min_args;
    uint8_t vararg;
    uint16_t nlocals;
    /* closure: mảng cell bắt từ scope ngoài (giữ ref) */
    uint16_t nupvals;
    uint16_t ncap;      /* đã capture (runtime) */
    TVal *upvals;
    /* JIT (jit_x64.c) */
    int64_t ncalls;   /* số lần gọi (kích hoạt biên dịch entry) */
    void *native;     /* entry native khi đã JIT (NULL nếu chưa/không) */
    int64_t loops;    /* số cạnh lùi đã chạy (kích hoạt OSR) */
    int jit_off;      /* bit 1: entry tắt, bit 2: OSR tắt */
    void *jit;        /* TJitFn* (jit_x64.c) */
};
#define JIT_THRESHOLD 50
#define JIT_OSR_LOOPS 2000

typedef void (*TNativeFn)(TVM *vm, TVal *args, int argc, TVal *ret);

struct TVal {
    TType t;
    union {
        int b;
        int64_t i;
        double f;
        struct { char *s; size_t n; int32_t *refc; } str;
        TList *list;
        TTensor *ten;
        TDict *dict;
        TFn *fn;
        TNative *nat;
        TFile *file;
        TCell *cell;
    } u;
};

struct TCell {
    int32_t refc;
    TVal val;
};

struct TNative {
    int32_t refc;
    int gc_mark;
    GCNode *gc_node;
    TNativeFn fn;
    TVal obj;      /* bound receiver (t=T_NIL nếu không có) */
    int has_obj;
    const char *name;
};

struct TFile {
    int32_t refc;
    FILE *h;
    char *path;
    int closed;
};

typedef struct {
    size_t ret_pc;
    size_t bp;          /* stack base lúc call */
    TVal *locals;
    uint16_t nlocals;
    uint16_t lcap;      /* capacity mảng locals (tái dùng qua các call) */
    TFn *fn;            /* hàm đang chạy (NULL ở main; retain) */
    struct TProg *ret_prog; /* prog của bên gọi (khôi phục khi return) */
    int is_mod;         /* frame top-level của module đang nạp (HALT = về load()) */
} TFrame;

typedef struct {
    jmp_buf jb;
    size_t sp;
    size_t nframes;
    size_t pc;
    size_t stop_nf;     /* khôi phục khi lỗi xuyên qua lời gọi lồng (native -> T) */
    struct TProg *prog; /* prog lúc vào try */
} THandler;

/* Opcodes v1 (tcomp.py dựng OP theo đúng thứ tự này — không sửa tay số) */
enum {
    OP_PUSH_NIL = 0x01,
    OP_PUSH_BOOL,     /* u8 0/1 */
    OP_PUSH_INT,      /* i64 le */
    OP_PUSH_FLOAT,    /* f64 le */
    OP_PUSH_STR,      /* u32 len + bytes */
    OP_NEW_LIST,      /* u16 n (pop n) */
    OP_NEW_TENSOR,    /* u16 n (pop n số) */
    OP_NEW_DICT,      /* u16 n (pop 2n: k0 v0 ...) */
    OP_ADD, OP_SUB, OP_MUL, OP_DIV, OP_MOD, OP_POW,
    OP_NEG, OP_NOT,
    OP_EQ, OP_NE, OP_LT, OP_GT, OP_LE, OP_GE,
    OP_CONTAINS,      /* hay needle -> bool (except matching) */
    OP_MATCHGLOB,     /* hay pattern(*,?) -> bool */
    OP_TDOT,          /* @ */
    OP_IDX,           /* c[i] */
    OP_SETIDX,        /* c[i]=v (pop v,i,c; không đẩy) */
    OP_SLICE,         /* c lo hi st (nil allowed) */
    OP_LEN,
    OP_PRINT,         /* u8 argc */
    OP_JMP,           /* i32 rel */
    OP_JMPF,          /* i32 rel (pop, nhảy nếu falsy) */
    OP_JMPT,          /* i32 rel (pop, nhảy nếu truthy) */
    OP_CALL,          /* u8 argc */
    OP_RET,
    OP_LOAD,          /* u16 slot (frame locals) */
    OP_STORE,         /* u16 */
    OP_DECL,          /* u16 (giống store, rõ nghĩa) */
    OP_LOADG,         /* u16 (globals) */
    OP_STOREG,        /* u16 */
    OP_DECLG,         /* u16 */
    OP_MKFN,          /* i32 entry, u8 nargs, u16 nlocals, u16 nupvals */
    OP_CAPLOCAL,      /* u16 slot: box frame-slot thành cell, capture vào fn mới */
    OP_CAPUP,         /* u16 idx: capture upval của frame hiện tại vào fn mới */
    OP_CAPG,          /* u16 gidx: box global thành cell, capture vào fn mới */
    OP_LOADU,         /* u16 idx: đọc cell upval của frame.fn */
    OP_STOREU,        /* u16 idx: ghi cell upval */
    OP_BIND,          /* u16 slot: gán mới frame (không xuyên cell; biến lặp) */
    OP_BINDG,         /* u16 gidx: gán mới global (không xuyên cell; biến lặp) */
    OP_RANGE,         /* u8 argc 1/2 */
    OP_ATTR,          /* u32 len + name */
    OP_RAISE,         /* pop v -> lỗi fmt(v) */
    OP_TRY_PUSH,      /* i32 handler_rel */
    OP_TRY_POP,
    OP_FOR_PREP,      /* dict -> keys-list, còn lại giữ nguyên */
    OP_POP,
    OP_DUP,           /* nhân đôi top */
    OP_INC_L,         /* u16: local += 1 (int/float) */
    OP_INC_G,         /* u16: global += 1 */
    OP_DEC_L,         /* u16: local -= 1 */
    OP_DEC_G,         /* u16: global -= 1 */
    OP_ADDI_L,        /* u16 slot + i64 k: local += k */
    OP_ADDI_G,        /* u16 + i64 */
    OP_ADDS_L,        /* u16 dst + u16 src: local = local + local */
    OP_ADDS_G,
    OP_SUBS_L,        /* local = local - local */
    OP_SUBS_G,
    OP_INCB_L,        /* u16: local += 1, gán mới (biến lặp) */
    OP_INCB_G,
    OP_IN,            /* a b -> bool (a in b) */
    OP_CALLM,         /* u8 argc + u32 len + name: obj a1..ak -> obj.name(a1..ak) */
    OP_SETATTR,       /* u32 len + name: obj v -> obj.name = v */
    OP_BAND, OP_BOR, OP_BXOR, OP_BNOT, OP_SHL, OP_SHR,
    OP_NOP,
    OP_HALT
};

/* Globals cố định cho natives (tcomp.py NGLOBNAT phải khớp, bắt đầu từ 0) */
#define NAT_RANGE  0
#define NAT_LEN    1
#define NAT_STR    2
#define NAT_INT    3
#define NAT_FLOAT  4
#define NAT_TYPE   5
#define NAT_TENSOR 6
#define NAT_OPEN   7
#define NAT_READ   8
#define NAT_WRITE  9
#define NAT_CLOSE  10
#define NAT_ORD    11
#define NAT_CHR    12
#define NAT_ARGV   13
#define NAT_PRINT  14
#define NAT_MEM    15
#define NAT_F64BITS 16
#define NAT_ABS    17
#define NAT_ROUND  18
#define NAT_MIN    19
#define NAT_MAX    20
#define NAT_SUM    21
#define NAT_ANY    22
#define NAT_ALL    23
#define NAT_SORT   24
#define NAT_REVERSED 25
#define NAT_ENUMERATE 26
#define NAT_ZIP    27
#define NAT_MAP    28
#define NAT_FILTER 29
#define NAT_SET    30
#define NAT_INPUT  31
#define NAT___STRUCT 32
#define NAT___NEW  33
#define NAT_ZEROS 34
#define NAT_ONES 35
#define NAT_FULL 36
#define NAT_ARANGE 37
#define NAT_EYE 38
#define NAT_RAND 39
#define NAT_RANDN 40
#define NAT_SEED 41
#define NAT_SQRT 42
#define NAT_EXP 43
#define NAT_LOG 44
#define NAT_SIN 45
#define NAT_COS 46
#define NAT_TANH 47
#define NAT_FLOOR 48
#define NAT_CEIL 49
#define NAT_PI 50
#define NAT_LOAD 51
#define NAT_NFIXED 52   /* user globals bắt đầu từ đây (tcomp.py) */

/* Gọi lại hàm T từ trong native (map/filter/sort cmp): dừng vòng lặp khi
   số frame rơi về <= stop_nf (0 = chạy tới HALT như bình thường). */
int tvm_call_value(TVM *vm, TVal fv, TVal *args, int argc, TVal *out);

typedef struct TProg {
    uint8_t *code;
    size_t len;
    TVal *globals;
    uint16_t nglobals;
    int64_t hot[256];
    /* argv cho bltin argv() */
    char **xargv;
    int xargc;
    /* GC mark-sweep thu hồi chu trình (refcount xử lý còn lại) */
    GCNode *gc_head;     /* doubly-linked */
    size_t gc_tracked;
    size_t gc_allocs;
    size_t gc_runs;
    struct TAttrIC *ic;  /* inline cache ATTR/CALLM/SETATTR theo pc (cấp lười) */
    /* bảng dòng (phần "LN01" của .tb, có thể không có): pc tăng dần */
    uint32_t nlines;
    uint32_t *lpc, *lline;
    uint16_t *lfile;
    uint16_t nlfiles;
    char **lfiles;
    /* bảng export (phần "EX01", có thể không có): tên global top-level -> slot */
    uint16_t nexports;
    char **exnames;
    uint16_t *exslots;
    uint8_t *exmut;         /* 1 = mut (ghi được qua m.X = v) */
    /* module nạp lúc chạy (load): đường dẫn, bản ghi export, danh sách (từ prog gốc) */
    char *path;
    TVal mod_rec;
    int loading;
    struct TProg *next_mod;
} TProg;
/* đường dẫn trun (tự biên dịch .t cho load); trun.c đặt lúc khởi động */
extern const char *tvm_exe_path;
/* dịch file .t sang .tb bằng tcomp.tb (bundle=1 gộp import); 1 ok, -1 lỗi (msg) */
int tvm_compile_src(const char *src, const char *out, int bundle, char *msg, size_t msgn);
/* đóng gói file .t hoặc .tb thành file thực thi độc lập (kèm footer TBEXE001); 0 ok, -1 lỗi */
int tvm_build_exe(const char *src_or_tb, const char *out_exe, char *msg, size_t msgn);
/* đọc file .tb (TB01 + LN01/EX01 tuỳ chọn) vào p (memset trước); 0 ok, -1 lỗi (msg) */
int tvm_read_tb(const char *path, TProg *p, char *msg, size_t msgn);
/* giải phóng phần do tvm_read_tb cấp (code, bảng dòng, export, path) */
void tvm_prog_release(TProg *p);
/* vị trí nguồn của pc: trả 1 + file, line nếu có bảng dòng */
int tvm_pc_line(const TProg *p, size_t pc, const char **file, uint32_t *line);

typedef struct TAttrIC {
    const void *key;     /* kiểu record (TDict* DK_TYPE) hoặc tag kiểu builtin + 1 */
    int32_t idx;         /* chỉ số field / method */
    int32_t kind;        /* 1 field record, 2 method record, 3 method builtin */
    TNativeFn fn;        /* kind 3 */
} TAttrIC;

struct TVM {
    TVal *stack;
    size_t sp, cap;
    TFrame *frames;
    size_t nframes, fcap, fmax;
    THandler *handlers;
    size_t nhandlers, hcap;
    TProg *prog;        /* prog đang chạy (đổi khi gọi hàm của module khác) */
    TProg *root;        /* chương trình chính: giữ GC + argv + danh sách module */
    TProg *errprog;     /* prog lúc lỗi (tra bảng dòng) */
    size_t pc;
    size_t stop_nf;   /* tvm_call_value: dừng nested run khi nframes <= mốc này */
    int err;
    char errmsg[256];
    size_t errpc;     /* pc lúc lỗi (tra bảng dòng) */
};

void tvm_init(TVM *vm, TProg *p);
void tvm_free(TVM *vm);
int  tvm_run(TVM *vm);            /* 0 ok, -1 error (errmsg) */
void tval_free(TVal *v);
void tval_retain(TVal *v);
void tval_print(const TVal *v);
void tval_fmt(const TVal *v, char *buf, size_t n);
int  tval_truthy(const TVal *v);
int  tval_eq(const TVal *a, const TVal *b);

/* JIT hot-path (jit_x64.c) */
void t_jit_maybe(TVM *vm, TFn *fn, TVal *args);
int  t_jit_call(TVM *vm, TFn *fn, TVal *args, int argc, TVal *ret);
/* OSR: đang ở đầu vòng lặp (pc) của fn với locals hiện tại -> chạy tiếp bằng
   code native tới hết hàm. Trả 1 + *ret khi đã chạy xong; 0 = tiếp tục thông dịch
   từ đầu vòng lặp; 2 = native đã deopt: locals/stack đã ghi lại, chạy tiếp tại vm->pc. */
void tvm_push(TVM *vm, TVal v);
int  t_jit_osr(TVM *vm, TFn *fn, TVal *locals, size_t pc, TVal *ret);
void t_jit_free_fn(TFn *fn);
int  t_jit_callees(TFn *fn, TFn ***out);   /* hàm mà code native giữ ref (GC đánh dấu) */
/* vòng lặp top-level nóng (header..end): 2 = đã chạy native rồi deopt/thoát (stack +
   vm->pc đã đặt), 0 = thử lại sau, -1 = không JIT được (đừng thử nữa) */
int  t_jit_osr_main(TVM *vm, size_t header, size_t end);
int  t_jit_root_fns(TFn ***out);           /* hàm mà code main giữ (gốc GC) */

/* Helper cho JIT tạo/gán record + chuỗi (G6): code máy gọi trực tiếp, tự lo
   refcount/GC (record mới refc=1 + gc_track; chuỗi so sánh không giữ ref). */
enum { TJIT_TY_F = 2, TJIT_TY_B = 3 };  /* khớp JT_F/B của jit_x64.c (JT_I = 1) */
TDict *tjit_rec_new(TDict *tp, int shape, int64_t *slots);
const uint8_t *tjit_shape_fty(int shape);   /* bảng kiểu field theo shape (jit_x64.c) */
int64_t tjit_str_eq(const TVal *a, const TVal *b, int m);
void tjit_dec_dict(TDict *d);
/* Helper cho JIT dict + chuỗi sở hữu được (G7): wrapper TVal* malloc (KHÔNG track
   GC) — native tự dec đúng hạn theo luật ownership (entry-compare, POP, epilogue
   RET, deopt move); orphan chờ GC chỉ đúng với dict (tracked). */
TVal *tjit_str_of(int64_t bits, int ty);  /* ty 0=I,1=F,2=B,3=S,4=SN(nhả arg) */
TVal *tjit_str_cat(const TVal *a, const TVal *b, int m);
void tjit_dec_sval(TVal *w);              /* tval_free + free wrapper */
void tjit_wrap_free(TVal *w);             /* nhả wrapper TVal* (pool 32 byte) */
void tjit_store_go(TVal *g, int64_t bits, int ty); /* write-through global mainmode */
TDict *tjit_dict_new0(void);
int tjit_dict_set(TDict *d, int64_t kbits, int64_t vbits, int kv); /* kv: b0=kstr, b[2:1]=vty */
int tjit_dict_getD(TDict *d, int64_t kbits, int kstr, int64_t *out); /* 0=int,1=deopt,2=giữ default */
int64_t tjit_dict_has(TDict *d, int64_t kbits, int kstr);
int64_t tjit_dict_remove(TDict *d, int64_t kbits, int kstr);
TVal *tjit_str_idx(const TVal *s, int64_t i); /* trả &str1_cache[ch] (mượn) hoặc NULL nếu ngoài biên */

/* JIT ARM64 (jit_arm64.c): emitter AArch64 + compiler bytecode->ARM64 + OSR + sim */
int  t_arm64_selftest(void);            /* 1 = khớp toàn bộ oracle byte + compiler selftest */
int  t_arm64_dump(const char *path);    /* ghi kernel mẫu ra file; 1/0 */
void t_arm64_jit_maybe(TVM *vm, TFn *fn, TVal *args);
int  t_arm64_jit_call(TVM *vm, TFn *fn, TVal *args, int argc, TVal *ret);
int  t_arm64_jit_osr(TVM *vm, TFn *fn, TVal *locals, size_t pc, TVal *ret);
void sys_arm64_compile(TVM *vm, TVal *a, int argc, TVal *r);
void sys_arm64_run(TVM *vm, TVal *a, int argc, TVal *r);
void sys_arm64_osr_run(TVM *vm, TVal *a, int argc, TVal *r);

/* Debugger (tdbg.c) */
extern int tvm_dbg_on;
void dbg_enable(void);
int  dbg_hook(TVM *vm);
void dbg_dis(TProg *p, size_t at, int count);

/* % kiểu Python (floor), dùng chung VM + JIT */
int64_t imod(int64_t a, int64_t b);
double fmod_floor(double a, double b);

/* Bảng opcode dùng chung */
const char *tvm_opname(uint8_t op);
size_t tvm_oplen(const uint8_t *c, size_t n, size_t pc);

#endif
