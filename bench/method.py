class Vec:
    __slots__ = ("x", "y")
    def __init__(self, x, y):
        self.x = x
        self.y = y
    def add(self, o):
        return Vec(self.x + o.x, self.y + o.y)
    def dot(self, o):
        return self.x * o.x + self.y * o.y

def run(n):
    acc = Vec(0, 0)
    d = Vec(1, 2)
    s = 0
    i = 0
    while i < n:
        acc = acc.add(d)
        s = s + acc.dot(d)
        i = i + 1
    return s

print(run(200000))
