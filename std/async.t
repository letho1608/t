# std/async.t — Event Loop bất đồng bộ & Cooperative Scheduler cho Ngôn ngữ T (v6.0 N4)
import time

struct Future:
    done
    val
    gen

    fn is_done(self):
        return self.done

    fn result(self):
        if not self.done and self.gen != nil:
            while true:
                v = self.gen.next()
                if v == nil:
                    break
                self.val = v
            self.done = true
        return self.val

struct EventLoop:
    queue
    results

    fn spawn(self, task_or_fn):
        mut g = task_or_fn
        if type(task_or_fn) == "fn":
            g = task_or_fn()
        if type(g) == "list":
            fut = Future(false, nil, g)
            self.queue.append(fut)
            return fut
        fut = Future(true, g, nil)
        self.queue.append(fut)
        return fut

    fn run_until_complete(self):
        mut active = true
        while active:
            active = false
            for fut in self.queue:
                if not fut.done:
                    if fut.gen != nil and type(fut.gen) == "list":
                        step = fut.gen.next()
                        if step != nil:
                            fut.val = step
                            active = true
                        else:
                            fut.done = true
                    else:
                        fut.done = true
        mut out = []
        for fut in self.queue:
            out.append(fut.val)
        return out

fn new_loop():
    return EventLoop([], [])

fn spawn(task_or_fn):
    if type(task_or_fn) == "fn":
        r = task_or_fn()
        if type(r) == "list":
            return Future(false, nil, r)
        return Future(true, r, nil)
    if type(task_or_fn) == "list":
        return Future(false, nil, task_or_fn)
    return Future(true, task_or_fn, nil)

fn sleep(ms):
    if ms > 0:
        time.sleep(ms)
    return ms

fn gather(tasks):
    loop = new_loop()
    for t in tasks:
        if type(t) == "Future":
            loop.queue.append(t)
        else:
            loop.spawn(t)
    return loop.run_until_complete()

fn run(coro):
    if type(coro) == "fn":
        res = coro()
        if type(res) == "Future":
            return res.result()
        return res
    if type(coro) == "Future":
        return coro.result()
    return coro
