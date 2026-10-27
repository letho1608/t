# examples/data_pipeline.t — Pipeline xử lý dữ liệu: CSV -> Regex -> Thống kê -> JSON
import csv
import json
import re
import math
import random

print("=== 1. Sinh dữ liệu bán hàng ngẫu nhiên ===")
random.seed(123)
let categories = ["Laptop", "Điện thoại", "Phụ kiện", "Màn hình"]
mut raw_records = []
for i in range(1, 21):
    cat = random.choice(categories)
    price = random.randint(15, 300) * 100000
    sku = f"SKU-{cat[0:3].upper()}-{1000 + i}"
    email = f"khach_{i}@shop{random.randint(1, 3)}.vn"
    raw_records.append([str(i), sku, cat, str(price), email])

csv_content = csv.dumps([["id", "sku", "category", "price", "email"]] + raw_records)
print(f"Đã tạo bảng CSV {raw_records.len} dòng.")

print("\n=== 2. Phân tích CSV & Lọc hợp lệ bằng Regex ===")
rows = csv.loads(csv_content)
let header = rows[0]
mut valid_items = []
email_rx = re.compile("^[a-z0-9_]+@[a-z0-9]+\\.[a-z]+$")

for i in range(1, rows.len):
    r = rows[i]
    sku = r[1]
    cat = r[2]
    price = int(r[3])
    email = r[4]
    if re.fullmatch(email_rx, email):
        valid_items.append({"sku": sku, "category": cat, "price": price, "email": email})

print(f"Số bản ghi hợp lệ: {valid_items.len} / {raw_records.len}")

print("\n=== 3. Tính toán Thống kê ===")
mut total_rev = 0
mut prices = []
for it in valid_items:
    p = it["price"]
    total_rev = total_rev + p
    prices.append(p)

mean_price = float(total_rev) / float(valid_items.len)
mut var_sum = 0.0
for p in prices:
    diff = float(p) - mean_price
    var_sum = var_sum + diff * diff
std_dev = math.sqrt(var_sum / float(valid_items.len))

print(f"Tổng doanh thu: {total_rev} VND")
print(f"Giá trung bình: {int(mean_price)} VND | Độ lệch chuẩn: {int(std_dev)} VND")

print("\n=== 4. Xuất Báo Cáo JSON ===")
let report = {
    "title": "Báo cáo tổng hợp doanh thu",
    "total_records": valid_items.len,
    "total_revenue": total_rev,
    "average_price": int(mean_price),
    "sample_items": valid_items[0:3]
}
let json_str = json.dumps(report, 2)
print(json_str)
