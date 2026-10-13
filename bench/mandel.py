def mandel(w, h, maxit):
    count = 0
    y = 0
    while y < h:
        x = 0
        while x < w:
            cr = 2.0 * x / w - 1.5
            ci = 2.0 * y / h - 1.0
            zr = 0.0
            zi = 0.0
            k = 0
            while k < maxit and zr * zr + zi * zi < 4.0:
                t = zr * zr - zi * zi + cr
                zi = 2.0 * zr * zi + ci
                zr = t
                k = k + 1
            if k == maxit:
                count = count + 1
            x = x + 1
        y = y + 1
    return count

print(mandel(200, 200, 100))
