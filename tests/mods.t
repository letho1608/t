import ai

a = ai.zeros(3)
print(a)
print(a.sum())
print(ai.dot(tensor([1, 2]), tensor([3, 4])))

from ai import relu, mse
print(relu(tensor([-1, 2, -3])))
print(mse(tensor([1, 2]), tensor([1, 4])))

try:
    raise "boom-xyz"
except boom:
    print("caught:" + err)

try:
    print(1 / 0)
except zero:
    print("caught-div0")

try:
    raise "boom-xyz-123"
except "boom-*-1?3":
    print("caught-glob")

mut hit = "no"
try:
    try:
        raise "boom-xyz"
    except "zzz-*":
        hit = "bad"
except boom:
    hit = "good"
print(hit)

print("PI=" + str(3.14))
print(type([1]))
print("hi".upper())
mut xs = [1, 2]
xs.append(3)
print(xs.len)
print(xs.pop())
