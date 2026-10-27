# examples/hello_tour.t — Tổng quan cú pháp ngôn ngữ T
import math
import random

print("=== 1. Biến & Kiểu Dữ Liệu ===")
let name = "Ngôn ngữ T"
mut version = 3
print(f"Chào mừng bạn đến với {name} v{version}!")

print("\n=== 2. Struct & Phương thức ===")
struct Point:
    x
    y
    fn norm(self):
        return math.hypot(float(self.x), float(self.y))
    fn add(self, other):
        return Point(self.x + other.x, self.y + other.y)

let p1 = Point(3, 4)
let p2 = Point(1, 2)
let p3 = p1.add(p2)
print(f"p1: ({p1.x}, {p1.y}) -> độ dài: {p1.norm()}")
print(f"p1 + p2 = ({p3.x}, {p3.y})")

print("\n=== 3. Toán tử Bitwise 64-bit ===")
mut mask = 0b1010
mask |= 0b0101
print(f"mask: {mask} (nhị phân 0b1111 = 15)")
print(f"mask << 2 = {mask << 2}, mask >> 1 = {mask >> 1}")

print("\n=== 4. Tham số biến thiên (*args) & Tham số mặc định ===")
fn log_event(level, *messages):
    content = " ".join([str(m) for m in messages])
    print(f"[{level}] {content}")

log_event("INFO", "Hệ thống", "sẵn sàng", 100, "%")

print("\n=== 5. Xử lý ngoại lệ với try / except / finally ===")
fn safe_div(a, b):
    try:
        if b == 0:
            raise "div0"
        return a / b
    except div0:
        print("Bắt lỗi chia cho 0!")
        return 0
    finally:
        print("  -> Khối finally luôn được thực thi.")

print(f"10 / 2 = {safe_div(10, 2)}")
print(f"10 / 0 = {safe_div(10, 0)}")

print("\n=== 6. List Comprehension ===")
let evens = [x * x for x in range(10) if x % 2 == 0]
print(f"Bình phương số chẵn: {evens}")
