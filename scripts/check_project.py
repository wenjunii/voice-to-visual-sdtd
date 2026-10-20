"""Validate the checkout locally without installing packages or contacting GitHub."""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import subprocess
import sys
from typing import Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TEST_DEPENDENCIES = {
    "numpy": "numpy",
    "pythonosc": "python-osc",
    "requests": "requests",
}
EXCLUDED_DIRECTORIES = {
    "__pycache__", ".git", ".venv", "venv", "env", "build", "dist",
}


def source_files(root: Path) -> list[Path]:
    """Limit validation to project modules, scripts, and tests."""
    paths = list(root.glob("*.py"))
    for directory in ("scripts", "tests"):
        for path in (root / directory).rglob("*.py"):
            if not any(
                part.lower() in EXCLUDED_DIRECTORIES
                for part in path.relative_to(root).parts[:-1]
            ):
                paths.append(path)
    return sorted(path for path in paths if path.is_file())


def check_syntax(paths: Sequence[Path], root: Path) -> bool:
    """Compile in memory: do not import modules or write bytecode."""
    valid = True
    for path in paths:
        try:
            compile(path.read_bytes(), str(path), "exec")
        except (OSError, SyntaxError, ValueError) as exc:
            print(f"Syntax check failed: {path.relative_to(root)}: {exc}", file=sys.stderr)
            valid = False
    return valid


def missing_dependencies() -> list[str]:
    missing = []
    for module, package in TEST_DEPENDENCIES.items():
        try:
            available = importlib.util.find_spec(module) is not None
        except (ImportError, ValueError):
            available = False
        if not available:
            missing.append(package)
    return missing


def validate(root: Path, *, verbose: bool = False) -> int:
    print(f"Local validation with Python {sys.version.split()[0]}", flush=True)
    paths = source_files(root)
    if not paths:
        print("No project Python files found.", file=sys.stderr)
        return 2
    if not check_syntax(paths, root):
        return 1
    print(f"Syntax OK ({len(paths)} files).", flush=True)

    if not any(path.is_file() for path in (root / "tests").glob("test*.py")):
        print("No discoverable test files found in tests/.", file=sys.stderr)
        return 2

    missing = missing_dependencies()
    if missing:
        print(f"Missing test dependencies: {', '.join(missing)}", file=sys.stderr)
        invocation_prefix = "& " if sys.platform == "win32" else ""
        print(
            "Install them once with this interpreter, then rerun the check:\n"
            f'  {invocation_prefix}"{sys.executable}" -m pip install '
            f'-r "{root / "requirements-test.txt"}"',
            file=sys.stderr,
        )
        return 2

    command = [sys.executable, "-m", "unittest", "discover", "-s", "tests"]
    if verbose:
        command.append("-v")
    print("Running unit tests...", flush=True)
    try:
        # Explicit pipes also work under Windows launchers without inheritable
        # console handles. Keep test diagnostics visible on both success and failure.
        result = subprocess.run(
            command, cwd=root, check=False, capture_output=True,
            text=True, errors="replace",
        )
    except OSError as exc:
        print(f"Could not start the test runner: {exc}", file=sys.stderr)
        return 2
    if result.stdout:
        print(result.stdout, end="", flush=True)
    if result.stderr:
        print(result.stderr, end="", file=sys.stderr, flush=True)
    if result.returncode:
        print(
            f"Unit tests failed (runner exit code {result.returncode}).",
            file=sys.stderr,
        )
        return 1
    print("Local validation passed. No GitHub sync was performed.", flush=True)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--verbose", action="store_true", help="show each unit test as it runs"
    )
    args = parser.parse_args(argv)
    try:
        return validate(PROJECT_ROOT, verbose=args.verbose)
    except KeyboardInterrupt:
        print("\nLocal validation interrupted.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
