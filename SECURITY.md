# Security Policy

## Supported Versions

The development team actively maintains and releases security patches for the versions listed below:

| Version | Supported Status |
| ------- | ---------------- |
| 7.x     | Supported        |
| 6.x     | Supported        |
| < 6.0   | Unsupported      |

---

## Reporting a Security Vulnerability

If you identify a security vulnerability in the runtime virtual machine, the self-hosted compiler, or the standard library, please report it through private communication channels:

1. You should not disclose the vulnerability in public GitHub issues or forum discussions.
2. Please send a detailed report via email to: `letho16082003@gmail.com`
3. We ask that you include the following information in your advisory:
   - You should describe the vulnerability mechanism and the potential attack vector.
   - You should provide a minimal reproduction script written in `.t` source or compiled `.tb` bytecode.
   - You should contrast the observed behavior against the expected behavior.
   - You should include an assessment of the security impact and affected configurations.

The maintainers will acknowledge receipt of your vulnerability report within 48 hours and coordinate with you regarding the disclosure timeline and subsequent security release.

---

## Defensive Runtime Safeguards

The T language architecture implements several defensive mechanisms to isolate untrusted input and prevent exploitation:

1. **Bytecode Structure Verification**:
   The `verify_bytecode` function inspects all compiled `.tb` files before the virtual machine begins execution. The verifier validates binary header magic numbers, verifies that opcode indices stay within bounds, checks operand alignment, and ensures that jump destinations and exception handler entries point to valid instructions.

2. **Subprocess Isolation and Command Injection Defense (CWE-78)**:
   The `os.run` primitive spawns external processes by invoking operating system APIs directly (`CreateProcessA` on Windows and `fork`/`execvp` on POSIX systems). The runtime passes arguments as discrete arrays of strings, which eliminates shell interpretation and prevents command injection attacks.

3. **Archive Extraction Path Traversal Defense (CWE-22)**:
   The package extraction subsystem checks every archive entry before extracting files to disk. The extractor rejects paths containing directory traversal elements (`..`), absolute path roots, or Windows drive prefixes to prevent unauthorized file overwrites.

4. **Continuous Integration Security Audits**:
   The GitHub Actions workflow runs automated secret scans with Gitleaks and filesystem vulnerability audits with Trivy on every commit pushed to the repository.
