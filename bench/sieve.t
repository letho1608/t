# bench/sieve.t — list index/gán (sàng nguyên tố)
fn sieve(n):
    mut flags = [true] * (n + 1)
    mut count = 0
    mut i = 2
    while i <= n:
        if flags[i]:
            count = count + 1
            mut j = i * i
            while j <= n:
                flags[j] = false
                j = j + i
        i = i + 1
    return count

print(sieve(2000000))
