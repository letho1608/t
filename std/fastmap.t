# std/fastmap.t — Unboxed Fibonacci Open-Addressing HashMap (IntMap) & Zero-Copy String SIMD Kernels (v7.1 U3)

let _sys = load("__sys")

struct IntMap:
    h
    owns

    fn len(self) -> int:
        return _sys.fastmap_len(self.h)

    fn __len__(self) -> int:
        return _sys.fastmap_len(self.h)

    fn get(self, k: int, dflt = 0) -> int:
        return _sys.fastmap_get(self.h, k, dflt)

    fn __getitem__(self, k: int) -> int:
        return _sys.fastmap_get(self.h, k, 0)

    fn set(self, k: int, v: int):
        _sys.fastmap_set(self.h, k, v)

    fn __setitem__(self, k: int, v: int):
        _sys.fastmap_set(self.h, k, v)

    fn inc(self, k: int, delta = 1) -> int:
        return _sys.fastmap_inc(self.h, k, delta)

    fn sum_vals(self) -> int:
        return _sys.fastmap_sum_vals(self.h)

    fn batch_freq(self, vec_i64, mask = 65535) -> int:
        return _sys.fastmap_batch_freq(self.h, vec_i64.h, mask)

    fn free(self):
        if self.owns and self.h != 0:
            _sys.fastmap_free(self.h)
            self.h = 0
            self.owns = false

    fn close(self):
        self.free()

    fn __enter__(self):
        return self

    fn __exit__(self):
        self.free()

fn int_map(cap = 64):
    return IntMap(_sys.fastmap_new(cap), true)

# Zero-copy 8-way unrolled byte counting in string
fn str_count_byte(s: str, byte_val: int) -> int:
    return _sys.str_count_byte(s, byte_val)

# Zero-copy 64-bit FNV-1a string hash
fn str_fnv1a_64(s: str) -> int:
    return _sys.str_fnv1a_64(s)

# Zero-copy substring search (strstr)
fn str_find_fast(s: str, sub: str) -> int:
    return _sys.str_find_fast(s, sub)
