# std/result.t — Kiểu Đại số Result<T, E> (Ok/Err) & Option<T> (Some/None) chuẩn Rust (v7.0 L12)

struct Ok:
    val

    fn is_ok(self) -> bool:
        return true

    fn is_err(self) -> bool:
        return false

    fn unwrap(self):
        return self.val

    fn unwrap_or(self, dflt):
        return self.val

    fn expect(self, msg: str):
        return self.val

    fn __str__(self) -> str:
        return "Ok(" + str(self.val) + ")"

    fn __eq__(self, o) -> bool:
        return type(o) == "Ok" and self.val == o.val

struct Err:
    val

    fn is_ok(self) -> bool:
        return false

    fn is_err(self) -> bool:
        return true

    fn err(self):
        return self.val

    fn unwrap(self):
        raise "unwrap() tren Err: " + str(self.val)

    fn unwrap_or(self, dflt):
        return dflt

    fn expect(self, msg: str):
        raise msg + ": " + str(self.val)

    fn __str__(self) -> str:
        return "Err(" + str(self.val) + ")"

    fn __eq__(self, o) -> bool:
        return type(o) == "Err" and self.val == o.val

struct Some:
    val

    fn is_some(self) -> bool:
        return true

    fn is_none(self) -> bool:
        return false

    fn unwrap(self):
        return self.val

    fn unwrap_or(self, dflt):
        return self.val

    fn expect(self, msg: str):
        return self.val

    fn __str__(self) -> str:
        return "Some(" + str(self.val) + ")"

    fn __eq__(self, o) -> bool:
        return type(o) == "Some" and self.val == o.val

struct None:
    fn is_some(self) -> bool:
        return false

    fn is_none(self) -> bool:
        return true

    fn unwrap(self):
        raise "unwrap() tren None"

    fn unwrap_or(self, dflt):
        return dflt

    fn expect(self, msg: str):
        raise msg

    fn __str__(self) -> str:
        return "None"

    fn __eq__(self, o) -> bool:
        return type(o) == "None"

fn move(x):
    return x
