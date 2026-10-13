def sieve(n):
    flags = [True] * (n + 1)
    count = 0
    i = 2
    while i <= n:
        if flags[i]:
            count = count + 1
            j = i * i
            while j <= n:
                flags[j] = False
                j = j + i
        i = i + 1
    return count

print(sieve(2000000))
