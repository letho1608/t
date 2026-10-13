# bench/dict.t — dict + chuỗi (đếm từ)
fn count_words(n):
    mut d = {}
    mut i = 0
    while i < n:
        k = "w" + str(i % 1000)
        d[k] = d.get(k, 0) + 1
        i = i + 1
    return d.len

print(count_words(300000))
