# tests/thread_test.t — Kiểm thử Worker Threads và Channels không GIL

import thread
import os

print("=== 1. Kiểm thử Channel nội bộ ===")
let ch = thread.channel(10)
assert ch.len() == 0

ch.send(100)
ch.send("t-lang parallel")
ch.send([10, 20, 30])
ch.send({"key": "val", "num": 42})
assert ch.len() == 4

assert ch.recv() == 100
assert ch.recv() == "t-lang parallel"
let lst = ch.recv()
assert lst[0] == 10 and lst[1] == 20 and lst[2] == 30
let d = ch.recv()
assert d["key"] == "val" and d["num"] == 42
assert ch.len() == 0

ch.close()
print("Channel nội bộ OK")

print("=== 2. Kiểm thử Worker Thread song song qua Channel ===")
let ch_in = thread.channel(5)
let ch_out = thread.channel(5)

let worker_src = "tests/_worker.t"
let worker_bin = "tests/_worker.tb"
if not os.exists(worker_src):
    mut wf = open(worker_src, "w")
    write(wf, "# Worker\nimport thread\nlet args = argv()\nlet ch_in = thread.channel_from_id(int(args[1]))\nlet ch_out = thread.channel_from_id(int(args[2]))\nlet num = ch_in.recv()\nch_out.send(num * num)\n")
    close(wf)

let worker_target = os.abspath(worker_bin if os.exists(worker_bin) else worker_src)
let h = thread.spawn(worker_target, [ch_in.id, ch_out.id])
assert h >= 0

ch_in.send(9)
let result = ch_out.recv()
print("Worker computed 9 * 9 =", result)
assert result == 81

thread.join(h)
ch_in.close()
ch_out.close()

print("=== THREAD TEST PASSED ===")
