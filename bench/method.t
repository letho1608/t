# bench/method.t — struct + gọi method
struct Vec:
    x, y
    fn add(self, o):
        return Vec(self.x + o.x, self.y + o.y)
    fn dot(self, o):
        return self.x * o.x + self.y * o.y

fn run(n):
    mut acc = Vec(0, 0)
    d = Vec(1, 2)
    mut s = 0
    mut i = 0
    while i < n:
        acc = acc.add(d)
        s = s + acc.dot(d)
        i = i + 1
    return s

print(run(200000))
