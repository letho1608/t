# std/vec.t — Unboxed Homogeneous Vectors (VecI64, VecF64), Zero-Cost Iterator Fusion & O(N) LSD Radix Sort (v7.1 U2)

let _sys = load("__sys")

struct VecI64:
    h
    owns

    fn len(self) -> int:
        return _sys.vec_i64_len(self.h)

    fn __len__(self) -> int:
        return _sys.vec_i64_len(self.h)

    fn ptr(self) -> int:
        return _sys.vec_i64_ptr(self.h)

    fn get(self, idx: int) -> int:
        return _sys.vec_i64_get(self.h, idx)

    fn __getitem__(self, idx: int) -> int:
        return _sys.vec_i64_get(self.h, idx)

    fn set(self, idx: int, val: int):
        _sys.vec_i64_set(self.h, idx, val)

    fn __setitem__(self, idx: int, val: int):
        _sys.vec_i64_set(self.h, idx, val)

    fn push(self, val: int):
        _sys.vec_i64_push(self.h, val)

    fn sum(self) -> int:
        return _sys.vec_i64_sum(self.h)

    fn fused_map_filter_sum(self, mul: int, add: int, min_thresh: int) -> int:
        return _sys.vec_i64_fused_map_filter_sum(self.h, mul, add, min_thresh)

    fn radix_sort(self):
        _sys.vec_i64_radix_sort(self.h)
        return self

    fn qsort(self):
        _sys.vec_i64_qsort(self.h)
        return self

    fn par_sum(self, nthreads = 4) -> int:
        return _sys.par_reduce_i64(self.h, nthreads)

    fn free(self):
        if self.owns and self.h != 0:
            _sys.vec_i64_free(self.h)
            self.h = 0
            self.owns = false

    fn close(self):
        self.free()

    fn __enter__(self):
        return self

    fn __exit__(self):
        self.free()

struct VecF64:
    h
    owns

    fn len(self) -> int:
        return _sys.vec_f64_len(self.h)

    fn __len__(self) -> int:
        return _sys.vec_f64_len(self.h)

    fn ptr(self) -> int:
        return _sys.vec_f64_ptr(self.h)

    fn get(self, idx: int) -> float:
        return _sys.vec_f64_get(self.h, idx)

    fn __getitem__(self, idx: int) -> float:
        return _sys.vec_f64_get(self.h, idx)

    fn set(self, idx: int, val):
        _sys.vec_f64_set(self.h, idx, float(val))

    fn __setitem__(self, idx: int, val):
        _sys.vec_f64_set(self.h, idx, float(val))

    fn push(self, val):
        _sys.vec_f64_push(self.h, float(val))

    fn dot(self, other) -> float:
        return _sys.vec_f64_dot(self.h, other.h)

    fn par_scale_bias(self, scale: float, bias: float, nthreads = 4):
        _sys.par_map_f64(self.h, scale, bias, nthreads)
        return self

    fn free(self):
        if self.owns and self.h != 0:
            _sys.vec_f64_free(self.h)
            self.h = 0
            self.owns = false

    fn close(self):
        self.free()

    fn __enter__(self):
        return self

    fn __exit__(self):
        self.free()

# Tao VecI64 toan so 0
fn i64_zeros(n = 0):
    return VecI64(_sys.vec_i64_new(n, 0), true)

# Tao VecI64 day [0, 1, ..., n-1]
fn i64_range(n: int):
    return VecI64(_sys.vec_i64_new(n, 1), true)

# Tao VecI64 ngau nhien (deterministic SplitMix64)
fn i64_rand(n: int):
    return VecI64(_sys.vec_i64_new(n, 2), true)

# Tao VecF64 voi gia tri khoi tao fill
fn f64_fill(n: int, fill = 0.0):
    return VecF64(_sys.vec_f64_new(n, float(fill)), true)
