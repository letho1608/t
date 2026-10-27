# tests/_worker.t — Worker script cho tests/thread_test.t
import thread

let args = argv()
let in_id = int(args[1])
let out_id = int(args[2])

let ch_in = thread.channel_from_id(in_id)
let ch_out = thread.channel_from_id(out_id)

let num = ch_in.recv()
ch_out.send(num * num)
