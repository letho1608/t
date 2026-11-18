# Contributing to the T Programming Language

Thank you for your interest in contributing to **T**! We welcome contributions from developers worldwide, ranging from bug reports and documentation enhancements to compiler optimizations, standard library modules, and tooling improvements.

---

## Getting Started

### Prerequisites

- **Windows**: MSVC (`cl.exe` from Visual Studio 2019+ or Build Tools).
- **Linux / macOS**: GCC (version 11+) or Clang (version 13+).
- **Git** with standard command-line tools.

### Building the Project

Clone the repository and build from source:

```bash
git clone https://github.com/letho1608/t.git
cd t

# On Windows
vm\build.bat

# On Linux or macOS
sh vm/build.sh
```

This compiles the C VM (`trun` / `trun.exe`), uses it to bootstrap the compiler (`self/tcomp.t` -> `tcomp.tb`), and builds the REPL (`tools/repl.t` -> `trepl.tb`).

---

## Running the Test Suite

All contributions must pass the entire test suite before submitting a Pull Request:

```bash
# Windows
.\trun.exe tools\run_all.t

# Linux / macOS
./trun tools/run_all.t
```

All 52 test suites must report `[PASS]` and conclude with `ALL-GREEN`.

If you add a new feature or fix a bug, please include a corresponding regression test in `tests/`.

---

## Development Workflow

1. **Fork and Branch**: Create a feature branch with a descriptive name:
   ```bash
   git checkout -b feat/your-feature-name
   # or
   git checkout -b fix/your-bugfix-name
   ```

2. **Commit Convention**: We follow [Conventional Commits](https://www.conventionalcommits.org/):
   - `feat(...)`: New language features, stdlib modules, or optimizations.
   - `fix(...)`: Bug fixes in compiler, VM, or libraries.
   - `docs(...)`: Documentation and README improvements.
   - `test(...)`: Adding or refactoring tests.
   - `ci(...)`: GitHub Actions and build system changes.
   - `perf(...)`: Performance enhancements and benchmark additions.

3. **Code Style**:
   - **T code**: 4-space indentation, no tabs, idiomatic syntax. You can format files using:
     ```bash
     trun tools/ffmt.t <file.t>
     ```
   - **C code** (VM & JIT): Clean C99, 4-space indentation, no compiler warnings at `-Wall` (GCC/Clang) and `/W3` (MSVC).

4. **Self-Hosting Verification**: If modifying the compiler (`self/*.t`), verify that it can self-compile to a fixed point:
   ```bash
   trun tcomp.tb self/tcli.t tcomp.new.tb
   trun tcomp.new.tb self/tcli.t tcomp.check.tb
   # tcomp.new.tb and tcomp.check.tb should be byte-identical
   ```

---

## Pull Request Guidelines

1. Ensure the CI pipeline passes cleanly on both Windows and Linux.
2. Provide a clear summary of what the PR accomplishes and reference any related issues.
3. Keep PRs focused on a single logical change.

---

## Code of Conduct

We are committed to providing a friendly, safe, and welcoming environment for all contributors, regardless of experience level, background, or identity. Please be respectful and constructive in all interactions.
