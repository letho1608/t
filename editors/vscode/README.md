# T Language Extension for Visual Studio Code

This extension provides language support for the T programming language (`.t` files) in Visual Studio Code.

## Features

- **Syntax Highlighting**: The extension includes a TextMate grammar that highlights keywords (`fn`, `struct`, `let`, `mut`, `try`, `except`, `finally`), 64-bit bitwise operators (`&`, `|`, `^`, `~`, `<<`, `>>`), formatted string interpolations, numeric literals, and tensor builtins.
- **Bracket Matching**: The editor configuration defines automatic balancing for parentheses, square brackets, curly braces, string quotes, and hash-prefixed line comments.
- **Smart Indentation**: The language configuration automatically indents newly created lines by 4 spaces when following a colon-terminated block statement.
- **Language Server Protocol Integration**: The extension connects to the native T Language Server executable (`tools/tlsp.t`) via `trun tools/tlsp.t`. This connection provides real-time hover documentation, jump-to-definition, reference lookups, symbol renaming, code completions, and syntax diagnostics directly in the editor.

## Local Installation

To install the extension locally, copy the `editors/vscode` directory into your Visual Studio Code extensions folder:

| Operating System | Destination Path |
|---|---|
| Windows | `%USERPROFILE%\.vscode\extensions\t-lang` |
| Linux and macOS | `~/.vscode/extensions/t-lang` |

You can also open this directory in Visual Studio Code and press `F5` to start an Extension Development Host session for live testing.
