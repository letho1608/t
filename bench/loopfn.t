# bench/loopfn.t — vòng lặp trong hàm (biến cục bộ)
fn work(n):
    mut s = 0
    mut i = 0
    while i < n:
        s = s + i % 7
        i = i + 1
    return s

print(work(10000000))
