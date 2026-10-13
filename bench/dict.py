def count_words(n):
    d = {}
    i = 0
    while i < n:
        k = "w" + str(i % 1000)
        d[k] = d.get(k, 0) + 1
        i = i + 1
    return len(d)

print(count_words(300000))
