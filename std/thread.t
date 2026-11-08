# std/thread.t — Worker Threads, Channels, Lock-Free AtomicInt, Mutex & Zero-Copy SharedTensor (v4.0 & v7.0 S1)

let _sys = load("__sys")

struct Channel:
    id

    fn send(self, val):
        return _sys.channel_send(self.id, val)

    fn recv(self):
        return _sys.channel_recv(self.id)

    fn close(self):
        return _sys.channel_close(self.id)

    fn len(self):
        return _sys.channel_len(self.id)

fn channel(capacity = 16):
    let id = _sys.channel_new(capacity)
    return Channel(id)

fn channel_from_id(id):
    return Channel(id)

struct AtomicInt:
    id

    fn load(self) -> int:
        return _sys.atomic_load(self.id)

    fn store(self, val: int):
        return _sys.atomic_store(self.id, val)

    fn add(self, delta: int) -> int:
        return _sys.atomic_add(self.id, delta)

    fn sub(self, delta: int) -> int:
        return _sys.atomic_add(self.id, -delta)

    fn cas(self, expected: int, desired: int) -> bool:
        return _sys.atomic_cas(self.id, expected, desired)

    fn free(self):
        return _sys.atomic_free(self.id)

    fn close(self):
        return _sys.atomic_free(self.id)

fn atomic_int(init_val = 0):
    return AtomicInt(_sys.atomic_new(init_val))

fn atomic_from_id(id):
    return AtomicInt(id)

struct Mutex:
    id

    fn lock(self):
        return _sys.mutex_lock(self.id)

    fn unlock(self):
        return _sys.mutex_unlock(self.id)

    fn free(self):
        return _sys.mutex_free(self.id)

    fn __enter__(self):
        _sys.mutex_lock(self.id)
        return self

    fn __exit__(self):
        _sys.mutex_unlock(self.id)

fn mutex():
    return Mutex(_sys.mutex_new())

fn mutex_from_id(id):
    return Mutex(id)

fn share_tensor(t):
    return _sys.tensor_share(t)

fn from_shared_tensor(handle):
    return _sys.tensor_from_shared(handle)

fn slice_sum(t_or_handle, start, end):
    return _sys.tensor_slice_sum(t_or_handle, start, end)

fn spawn(script_path, args = []):
    return _sys.thread_spawn(script_path, args)

fn join(handle):
    return _sys.thread_join(handle)
