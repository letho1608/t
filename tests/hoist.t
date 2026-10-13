# tests/hoist.t — dùng GLOBAL khai báo SAU khi định nghĩa hàm (hoist slot)
# Giới hạn còn lại: biến có block scope (vòng for/if) định nghĩa sau
# thì closure chưa thấy — cả 4 runtime đều giống nhau (xem SPEC).
fn peek():
    return later

fn bump():
    later = later + 1
    return later

fn add(k):
    return base + k

mut later = 10
print(peek())
print(bump())
print(later)
let base = 5
print(add(7))

# đệ quy chéo: hai hàm gọi nhau
fn even_(n):
    if n == 0:
        return true
    return odd_(n - 1)

fn odd_(n):
    if n == 0:
        return false
    return even_(n - 1)
print(even_(4), odd_(4))

# gọi hàm qua biến hàm
fn maker(n):
    return n * 3
h = maker
print(h(5))

# biến lặp + closure tạo trong vòng lặp (biến lặp đã bind trước)
fns = []
for i in range(3):
    fn g():
        return i * 10
    fns.append(g)
mut k = 0
while k < fns.len:
    print(fns[k]())
    k = k + 1
print("HOIST OK")
