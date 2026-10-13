def work(n):
    s = 0
    i = 0
    while i < n:
        s = s + i % 7
        i = i + 1
    return s

print(work(10000000))
