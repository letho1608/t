# tests/sqlite_test.t — Kiểm thử module std/sqlite

let sqlite = load("std/sqlite")

fn test_sqlite_basic():
    print("=== Test 1: SQLite Basic Operations ===")
    assert sqlite.is_available()
    let ver = sqlite.version()
    print("SQLite Version:", ver)
    assert len(ver) > 0

    let db = sqlite.connect(":memory:")
    assert db != nil

    # Tạo bảng
    db.execute("CREATE TABLE products (id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT, price REAL, stock INTEGER, description TEXT);")

    # Insert với parameters
    db.execute("INSERT INTO products (name, price, stock, description) VALUES (?, ?, ?, ?);", ["Laptop", 1250.50, 15, "High performance laptop"])
    let id1 = db.last_insert_rowid()
    assert id1 == 1

    db.execute("INSERT INTO products (name, price, stock, description) VALUES (?, ?, ?, ?);", ["Mouse", 25.0, 100, nil])
    let id2 = db.last_insert_rowid()
    assert id2 == 2

    # Query all
    let rows = db.query("SELECT id, name, price, stock, description FROM products ORDER BY id;")
    assert len(rows) == 2

    let r0 = rows[0]
    assert r0["id"] == 1
    assert r0["name"] == "Laptop"
    assert r0["price"] == 1250.50
    assert r0["stock"] == 15
    assert r0["description"] == "High performance laptop"

    let r1 = rows[1]
    assert r1["id"] == 2
    assert r1["name"] == "Mouse"
    assert r1["price"] == 25.0
    assert r1["stock"] == 100
    assert r1["description"] == nil

    # Query one with parameter filter
    let item = db.query_one("SELECT * FROM products WHERE name = ?;", ["Laptop"])
    assert item != nil
    assert item["id"] == 1

    # Transaction test: Commit
    db.begin()
    db.execute("INSERT INTO products (name, price, stock, description) VALUES (?, ?, ?, ?);", ["Keyboard", 45.0, 50, "Mechanical"])
    db.commit()

    let count_after_commit = len(db.query("SELECT * FROM products;"))
    assert count_after_commit == 3

    # Transaction test: Rollback
    db.begin()
    db.execute("INSERT INTO products (name, price, stock, description) VALUES (?, ?, ?, ?);", ["Monitor", 300.0, 10, "4K"])
    db.rollback()

    let count_after_rollback = len(db.query("SELECT * FROM products;"))
    assert count_after_rollback == 3

    db.close()
    print("SQLite Basic Test: PASSED")

test_sqlite_basic()
print("All sqlite tests passed successfully!")
