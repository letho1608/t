# examples/data_pipeline.t — Data Processing Pipeline: CSV -> Regex -> Statistics -> JSON
import csv
import json
import re
import math
import random

print("=== 1. Generating Mock Sales Data ===")
random.seed(123)
let categories = ["Laptop", "Phone", "Accessories", "Monitor"]
mut raw_records = []
for i in range(1, 21):
    cat = random.choice(categories)
    price = random.randint(15, 300) * 10
    sku = f"SKU-{cat[0:3].upper()}-{1000 + i}"
    email = f"customer_{i}@store{random.randint(1, 3)}.com"
    raw_records.append([str(i), sku, cat, str(price), email])

csv_content = csv.dumps([["id", "sku", "category", "price", "email"]] + raw_records)
print(f"Generated CSV table with {raw_records.len} rows.")

print("\n=== 2. Parsing CSV & Validating with Regex ===")
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

print(f"Valid records: {valid_items.len} / {raw_records.len}")

print("\n=== 3. Computing Statistics ===")
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

print(f"Total Revenue: ${total_rev}")
print(f"Average Price: ${int(mean_price)} | Standard Deviation: ${int(std_dev)}")

print("\n=== 4. Exporting JSON Report ===")
let report = {
    "title": "Sales Revenue Summary Report",
    "total_records": valid_items.len,
    "total_revenue": total_rev,
    "average_price": int(mean_price),
    "sample_items": valid_items[0:3]
}
let json_str = json.dumps(report, 2)
print(json_str)
