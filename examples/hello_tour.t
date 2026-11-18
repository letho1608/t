# examples/hello_tour.t — A tour of T language syntax
import math
import random

print("=== 1. Variables & Types ===")
let name = "T Language"
mut version = 3
print(f"Welcome to {name} v{version}!")

print("\n=== 2. Structs & Methods ===")
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
print(f"p1: ({p1.x}, {p1.y}) -> length: {p1.norm()}")
print(f"p1 + p2 = ({p3.x}, {p3.y})")

print("\n=== 3. 64-bit Bitwise Operators ===")
mut mask = 0b1010
mask |= 0b0101
print(f"mask: {mask} (binary 0b1111 = 15)")
print(f"mask << 2 = {mask << 2}, mask >> 1 = {mask >> 1}")

print("\n=== 4. Variadic *args & Default Parameters ===")
fn log_event(level, *messages):
    content = " ".join([str(m) for m in messages])
    print(f"[{level}] {content}")

log_event("INFO", "System", "ready", 100, "%")

print("\n=== 5. Exception Handling: try / except / finally ===")
fn safe_div(a, b):
    try:
        if b == 0:
            raise "div0"
        return a / b
    except div0:
        print("Caught division by zero!")
        return 0
    finally:
        print("  -> finally block always runs.")

print(f"10 / 2 = {safe_div(10, 2)}")
print(f"10 / 0 = {safe_div(10, 0)}")

print("\n=== 6. List Comprehension ===")
let evens = [x * x for x in range(10) if x % 2 == 0]
print(f"Squares of even numbers: {evens}")
