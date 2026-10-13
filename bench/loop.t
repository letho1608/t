# bench/loop.t — vòng lặp toàn cục (global)
mut s = 0
for i in range(5000000):
    s = s + i % 7
print(s)
