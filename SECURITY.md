# Security Policy

## Supported Versions

Security updates are applied to the active development branch (`main`) and released versions:

| Version | Supported          |
| ------- | ------------------ |
| 7.x     | :white_check_mark: |
| 6.x     | :white_check_mark: |
| < 6.0   | :x:                |

---

## Reporting a Vulnerability

If you discover a security vulnerability in the T programming language runtime, compiler, or standard library, please report it privately:

1. **Do not open a public GitHub issue.**
2. Send details to: `letho16082003@gmail.com`
3. Include:
   - Description of the vulnerability and attack vector.
   - Minimal reproduction code (`.t` script or `.tb` bytecode).
   - Expected vs observed behavior.
   - Potential impact.

We will acknowledge receipt within 48 hours and work with you on a coordinated disclosure timeline.

---

## Built-In Security Architecture

The T language runtime incorporates several defensive measures:

1. **Bytecode Verification (`verify_bytecode`)**:
   - Validates opcode ranges, operand offsets, magic identifiers, and jump boundaries before execution.
   - Verifies symbol table and exception table indices to prevent out-of-bounds reads.

2. **Command Injection Mitigation (CWE-78)**:
   - Subprocesses created via `os.run` invoke OS APIs directly (`CreateProcessA` on Windows, `fork`/`execvp` on POSIX) with typed argument arrays rather than passing shell-interpreted strings.

3. **Zip-Slip Mitigation (CWE-22)**:
   - The package extractor rejects archive entries containing path traversal sequences (`..`), absolute root paths, or drive letter specifiers.

4. **Continuous Security Auditing**:
   - Every push and pull request runs automated secret scanning (`Gitleaks`) and vulnerability/misconfiguration auditing (`Trivy`) on GitHub Actions.
