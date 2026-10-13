print("== core ==")
let a = 1 + 2 * 3
print(a)
print(7 / 2)
print(7 % 3)
print(1 == 1)
print(1 != 2)
print("ab" + "cd")
print([1, 2] + [3])
print(true and false)
print(true or false)
print(not false)
if 1 > 2:
    print("bad")
elif 2 > 3:
    print("bad2")
else:
    print("elif-ok")
mut i = 0
mut s = 0
while true:
    i = i + 1
    if i == 3:
        continue
    if i > 5:
        break
    s = s + i
print(s)
fn fact(n):
    if n < 2:
        return 1
    else:
        return n * fact(n - 1)
print(fact(10))
print("== tensor ==")
t = tensor([1, 2, 3])
print(t * 2)
print(t - 1)
print(t / 2)
print(len(t))
print(t[0])
t[0] = 9
print(t[0])
fn sq(x):
    return x * x
print(sq(9))
