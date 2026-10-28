# std/sqlite.t — SQLite Database Client cho T (Pure FFI, Zero external C dependencies)

let ffi = load("std/ffi")
let _sys = load("__sys")

# Tìm và nạp thư viện SQLite native của hệ điều hành
fn _load_sqlite_lib():
    let candidates = []
    let plat = _sys.platform()
    if plat == "windows" or plat == "win32":
        candidates.append("winsqlite3.dll")
        candidates.append("sqlite3.dll")
    else:
        candidates.append("libsqlite3.so.0")
        candidates.append("libsqlite3.so")
        candidates.append("libsqlite3.dylib")

    for c in candidates:
        let lib = ffi.open(c)
        if lib != nil:
            return lib
    return nil

let _lib = _load_sqlite_lib()

# Hằng số SQLite
let SQLITE_OK = 0
let SQLITE_ROW = 100
let SQLITE_DONE = 101

let SQLITE_INTEGER = 1
let SQLITE_FLOAT = 2
let SQLITE_TEXT = 3
let SQLITE_BLOB = 4
let SQLITE_NULL = 5

struct SQLiteDriver:
    lib
    fn_open
    fn_close
    fn_errmsg
    fn_prepare
    fn_step
    fn_finalize
    fn_reset
    fn_col_count
    fn_col_name
    fn_col_type
    fn_col_int64
    fn_col_double
    fn_col_text
    fn_bind_null
    fn_bind_int64
    fn_bind_double
    fn_bind_text
    fn_changes
    fn_last_id
    fn_libversion

fn _init_driver():
    if _lib == nil:
        return nil
    return SQLiteDriver(
        _lib,
        _lib.get_fn("sqlite3_open", "i", "sp"),
        _lib.get_fn("sqlite3_close", "i", "p"),
        _lib.get_fn("sqlite3_errmsg", "s", "p"),
        _lib.get_fn("sqlite3_prepare_v2", "i", "psipi"),
        _lib.get_fn("sqlite3_step", "i", "p"),
        _lib.get_fn("sqlite3_finalize", "i", "p"),
        _lib.get_fn("sqlite3_reset", "i", "p"),
        _lib.get_fn("sqlite3_column_count", "i", "p"),
        _lib.get_fn("sqlite3_column_name", "s", "pi"),
        _lib.get_fn("sqlite3_column_type", "i", "pi"),
        _lib.get_fn("sqlite3_column_int64", "l", "pi"),
        _lib.get_fn("sqlite3_column_double", "d", "pi"),
        _lib.get_fn("sqlite3_column_text", "s", "pi"),
        _lib.get_fn("sqlite3_bind_null", "i", "pi"),
        _lib.get_fn("sqlite3_bind_int64", "i", "pil"),
        _lib.get_fn("sqlite3_bind_double", "i", "pid"),
        _lib.get_fn("sqlite3_bind_text", "i", "pisip"),
        _lib.get_fn("sqlite3_changes", "i", "p"),
        _lib.get_fn("sqlite3_last_insert_rowid", "l", "p"),
        _lib.get_fn("sqlite3_libversion", "s", "")
    )

let _drv = _init_driver()

fn version():
    if _drv == nil:
        return "not available"
    return _drv.fn_libversion.call()

fn is_available():
    return _drv != nil

struct Connection:
    db
    drv

    fn errmsg(self):
        return self.drv.fn_errmsg.call(self.db)

    fn _bind_params(self, stmt, params):
        if params == nil:
            return nil
        mut idx = 1
        for val in params:
            let t = type(val)
            if val == nil:
                self.drv.fn_bind_null.call(stmt, idx)
            elif t == "int" or t == "bool":
                self.drv.fn_bind_int64.call(stmt, idx, int(val))
            elif t == "float":
                self.drv.fn_bind_double.call(stmt, idx, val)
            elif t == "str":
                self.drv.fn_bind_text.call(stmt, idx, val, -1, 0)
            else:
                self.drv.fn_bind_text.call(stmt, idx, str(val), -1, 0)
            idx += 1
        return nil

    fn execute(self, sql, params = []):
        let stmt_buf = ffi.buffer(8)
        let rc = self.drv.fn_prepare.call(self.db, sql, -1, stmt_buf.ptr, 0)
        if rc != SQLITE_OK:
            let err = self.errmsg()
            stmt_buf.free()
            raise "sqlite error in prepare: " + err
        let stmt = stmt_buf.read_ptr()
        stmt_buf.free()

        self._bind_params(stmt, params)
        mut step_rc = self.drv.fn_step.call(stmt)
        while step_rc == SQLITE_ROW:
            step_rc = self.drv.fn_step.call(stmt)

        self.drv.fn_finalize.call(stmt)
        if step_rc != SQLITE_DONE and step_rc != SQLITE_ROW:
            raise "sqlite error in execute: " + self.errmsg()
        return self.changes()

    fn query(self, sql, params = []):
        let stmt_buf = ffi.buffer(8)
        let rc = self.drv.fn_prepare.call(self.db, sql, -1, stmt_buf.ptr, 0)
        if rc != SQLITE_OK:
            let err = self.errmsg()
            stmt_buf.free()
            raise "sqlite error in prepare: " + err
        let stmt = stmt_buf.read_ptr()
        stmt_buf.free()

        self._bind_params(stmt, params)

        let ncols = self.drv.fn_col_count.call(stmt)
        let col_names = []
        for i in range(ncols):
            col_names.append(self.drv.fn_col_name.call(stmt, i))

        let rows = []
        while self.drv.fn_step.call(stmt) == SQLITE_ROW:
            mut row = {}
            for i in range(ncols):
                let name = col_names[i]
                let ctype = self.drv.fn_col_type.call(stmt, i)
                if ctype == SQLITE_INTEGER:
                    row[name] = self.drv.fn_col_int64.call(stmt, i)
                elif ctype == SQLITE_FLOAT:
                    row[name] = self.drv.fn_col_double.call(stmt, i)
                elif ctype == SQLITE_TEXT:
                    row[name] = self.drv.fn_col_text.call(stmt, i)
                elif ctype == SQLITE_NULL:
                    row[name] = nil
                else:
                    row[name] = self.drv.fn_col_text.call(stmt, i)
            rows.append(row)

        self.drv.fn_finalize.call(stmt)
        return rows

    fn query_one(self, sql, params = []):
        let rows = self.query(sql, params)
        if len(rows) == 0:
            return nil
        return rows[0]

    fn changes(self):
        return self.drv.fn_changes.call(self.db)

    fn last_insert_rowid(self):
        return self.drv.fn_last_id.call(self.db)

    fn begin(self):
        return self.execute("BEGIN TRANSACTION;")

    fn commit(self):
        return self.execute("COMMIT;")

    fn rollback(self):
        return self.execute("ROLLBACK;")

    fn close(self):
        if self.db != 0:
            self.drv.fn_close.call(self.db)
            self.db = 0
        return nil

fn connect(path = ":memory:"):
    if _drv == nil:
        raise "sqlite library not found on system"
    let buf = ffi.buffer(8)
    let rc = _drv.fn_open.call(path, buf.ptr)
    if rc != SQLITE_OK:
        buf.free()
        raise "cannot open sqlite database: " + path
    let db = buf.read_ptr()
    buf.free()
    return Connection(db, _drv)
