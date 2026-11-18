# T Language — VS Code Extension

Provides rich language support for the T programming language (`.t` files) in Visual Studio Code.

## Features

- **Full syntax highlighting** — keywords (`fn`, `struct`, `let`, `mut`, `try/except/finally`), 64-bit bitwise operators (`&`, `|`, `^`, `~`, `<<`, `>>`), f-strings, binary/hex/float literals, tensor builtins, and more.
- **Auto-closing pairs** — brackets `()`, `[]`, `{}`, single/double quotes, and line comments with `#`.
- **Smart indentation** — automatically increases indent by 4 spaces after blocks ending with `:`.
- **LSP integration** — connects to the T Language Server ([tools/tlsp.t](../../tools/tlsp.t)) via `trun tools/tlsp.t`, providing hover docs, go-to-definition, find references, rename, completions, and diagnostics.

## Local Installation

Copy the `editors/vscode` directory into your VS Code extensions folder:

| Platform | Path |
|---|---|
| Windows | `%USERPROFILE%\.vscode\extensions\t-lang` |
| Linux / macOS | `~/.vscode/extensions/t-lang` |

Alternatively, open this folder in VS Code and press `F5` to launch an Extension Development Host for testing.
