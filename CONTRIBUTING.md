# Contributing to the T Programming Language

We welcome contributions to the T programming language project. This document outlines the development workflow, coding standards, and procedures for submitting issues and pull requests.

---

## Getting Started

### Prerequisites

To build and test the project, your environment requires the following tools:
- On Windows, you need the Microsoft Visual C++ compiler (`cl.exe`), provided by Visual Studio 2019 or later or the standalone Visual Studio Build Tools.
- On Linux and macOS, you need GCC (version 11 or later) or Clang (version 13 or later).
- You need the Git version control system installed and accessible in your shell `PATH`.

### Building the Project

You can clone the repository and compile the runtime binary with the following commands:

```bash
git clone https://github.com/letho1608/t.git
cd t

# On Windows
vm\build.bat

# On Linux or macOS
sh vm/build.sh
```

These scripts execute the following tasks:
1. The host C compiler builds the virtual machine executable (`trun` or `trun.exe`).
2. The newly compiled runtime executes the self-hosted compiler source (`self/tcli.t`) to produce the bootstrapped compiler bytecode (`tcomp.tb`).
3. The bootstrapped compiler compiles the interactive REPL tool (`tools/repl.t`) to generate `trepl.tb`.

---

## Running the Test Suite

All contributions must pass the entire regression test harness before they can be merged:

```bash
# On Windows
.\trun.exe tools\run_all.t

# On Linux or macOS
./trun tools/run_all.t
```

The test runner evaluates all 52 subsystem suites. Every suite must report `[PASS]`, and the execution must conclude with an `ALL-GREEN` status message.

When you implement a new feature or resolve a defect, you should add a corresponding regression test in the `tests/` directory to protect against future regressions.

---

## Development Workflow

### Branching and Commits

1. You should create a dedicated feature branch from the latest `main` branch before starting work:
   ```bash
   git checkout -b feat/your-feature-name
   ```

2. We follow the Conventional Commits specification for all commit messages:
   - `feat(...)`: Use this prefix when you introduce a new language feature, standard library module, or compiler optimization.
   - `fix(...)`: Use this prefix when you correct a defect in the compiler, virtual machine, or standard library.
   - `docs(...)`: Use this prefix when you improve documentation, comments, or sample programs.
   - `test(...)`: Use this prefix when you add or refactor regression test cases.
   - `ci(...)`: Use this prefix when you modify build scripts, container definitions, or GitHub Actions workflows.
   - `perf(...)`: Use this prefix when you improve execution throughput or optimize memory footprint.

### Code Style Guidelines

- **T Source Code**:
  All T code must use 4-space indentation without tab characters. You can format source files automatically using the built-in code formatter:
  ```bash
  trun tools/ffmt.t <file.t>
  ```

- **C Source Code**:
  The virtual machine and JIT compiler implementations follow the C99 standard. C code must compile cleanly without warnings under `-Wall` on GCC and Clang and under `/W3` on MSVC.

### Compiler Self-Hosting Verification

If you modify the compiler implementation under `self/`, you must verify that the compiler can compile itself to a fixed point:

```bash
trun tcomp.tb self/tcli.t tcomp.new.tb
trun tcomp.new.tb self/tcli.t tcomp.check.tb
```

The two generated files, `tcomp.new.tb` and `tcomp.check.tb`, must be byte-identical.

---

## Pull Request Guidelines

Before submitting your pull request, please verify the following items:
1. Continuous integration workflows must pass on both Linux and Windows runners.
2. The pull request description must clearly summarize the rationale for the change and reference relevant issue numbers.
3. Each pull request should represent a single logical change to facilitate code review.

---

## Code of Conduct

We are dedicated to providing a respectful, productive, and welcoming environment for all contributors. Please communicate constructively and respectfully across all issue discussions and pull request reviews.
