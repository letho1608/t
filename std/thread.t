# std/thread.t — Worker Threads và Channels (Đa luồng không GIL) cho T

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

fn spawn(script_path, args = []):
    return _sys.thread_spawn(script_path, args)

fn join(handle):
    return _sys.thread_join(handle)
