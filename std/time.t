# std/time.t — thư viện thời gian chuẩn của T

let _sys = load("__sys")

fn clock():
    return _sys.clock()

fn time():
    return _sys.time()

fn sleep(sec):
    return _sys.sleep(sec)

fn measure(f):
    t0 = _sys.clock()
    res = f()
    return [res, _sys.clock() - t0]
