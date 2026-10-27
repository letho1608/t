# Tiện ích mở rộng VS Code cho Ngôn ngữ T

Cung cấp tô màu cú pháp (Syntax Highlighting), quy tắc thụt lề (Indentation Rules) và tích hợp Language Server cho ngôn ngữ lập trình T (`.t`).

## Tính năng

- **Tô màu cú pháp đầy đủ**: Từ khóa (`fn`, `struct`, `let`, `mut`, `try/except/finally`), toán tử bitwise (`&`, `|`, `^`, `~`, `<<`, `>>`), f-strings, số nhị phân/hex/float, tensor và builtins.
- **Tự động đóng ngoặc & bình luận**: Đóng mở `()`, `[]`, `{}`, nháy đơn/kép, và comment dòng với `#`.
- **Tự động thụt lề**: Tự động tăng 4 spaces sau các khối có kết thúc bằng `:`.
- **Tương thích LSP**: Sử dụng Language Server chuẩn của T ([tools/tlsp.t](../../tools/tlsp.t)) qua lệnh `trun tools/tlsp.t`.

## Cài đặt cục bộ

Chép thư mục `editors/vscode` vào thư mục extensions của VS Code:
- **Windows**: `%USERPROFILE%\.vscode\extensions\t-lang`
- **Linux / macOS**: `~/.vscode/extensions/t-lang`

Hoặc mở thư mục này trong VS Code và nhấn `F5` để chạy thử nghiệm.
