"""Validate the checkout locally without installing packages or contacting GitHub."""

from __future__ import annotations

import argparse
import codecs
import importlib.util
import io
import math
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
from typing import Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TEST_RUNNER = Path(__file__).resolve().with_name("run_unit_tests.py")
DEFAULT_TIMEOUT_SECONDS = 300.0
PROCESS_CLEANUP_SECONDS = 5.0
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


def positive_timeout(value: str) -> float:
    try:
        seconds = float(value)
    except (TypeError, ValueError):
        raise argparse.ArgumentTypeError(
            "timeout must be a finite number greater than zero"
        ) from None
    if not math.isfinite(seconds) or seconds <= 0:
        raise argparse.ArgumentTypeError(
            "timeout must be a finite number greater than zero"
        )
    return seconds


def _stream_test_output(pipe, output, errors) -> None:
    """Forward available chunks, including dots and lines without a newline."""
    # Normalize CRLF before writing to a Windows text stream, which performs
    # its own newline translation. Both UTF-8 and CRLF may cross chunk boundaries.
    decoder = io.IncrementalNewlineDecoder(
        codecs.getincrementaldecoder("utf-8")(errors="replace"), translate=True,
    )
    try:
        with pipe:
            while chunk := pipe.read1(4096):
                output.write(decoder.decode(chunk))
                output.flush()
            output.write(decoder.decode(b"", final=True))
            output.flush()
    except Exception as exc:
        errors.append(exc)


def _stop_test_process(process) -> None:
    if process.poll() is None:
        try:
            process.kill()
        except ProcessLookupError:
            pass
    process.wait(timeout=PROCESS_CLEANUP_SECONDS)


def run_tests(root: Path, *, verbose: bool, timeout_seconds: float) -> int:
    command = [sys.executable, "-u", str(TEST_RUNNER)]
    if verbose:
        command.append("--verbose")
    environment = os.environ.copy()
    environment["PYTHONIOENCODING"] = "utf-8"
    deadline = time.monotonic() + timeout_seconds
    # Explicit pipes work under Windows launchers without inheritable console
    # handles. Unbuffered child output and a reader keep progress visible while
    # the main thread independently enforces the deadline.
    process = subprocess.Popen(
        command, cwd=root, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, env=environment,
    )
    errors = []
    reader = None
    try:
        try:
            reader = threading.Thread(
                target=_stream_test_output, args=(process.stdout, sys.stdout, errors),
                name="local-check-output", daemon=True,
            )
            reader.start()
        except RuntimeError as exc:
            raise OSError(f"Could not start test output reader: {exc}") from exc
        while True:
            if errors:
                raise OSError(f"Could not forward test output: {errors[0]}")
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise subprocess.TimeoutExpired(command, timeout_seconds)
            try:
                code = process.wait(timeout=min(remaining, 0.05))
            except subprocess.TimeoutExpired:
                continue
            reader.join(timeout=min(remaining, 0.05))
            if not reader.is_alive():
                if errors:
                    raise OSError(f"Could not forward test output: {errors[0]}")
                return code
    finally:
        try:
            _stop_test_process(process)
        finally:
            if reader is not None and reader.ident is not None:
                # An inherited output handle in a descendant must not make
                # cancellation wait indefinitely. The daemon owns its pipe.
                reader.join(timeout=1.0)
            else:
                process.stdout.close()


def validate(
    root: Path, *, verbose: bool = False,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
) -> int:
    try:
        timeout_seconds = positive_timeout(str(timeout_seconds))
    except argparse.ArgumentTypeError as exc:
        print(f"Invalid timeout: {exc}", file=sys.stderr)
        return 2
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

    print(f"Running unit tests (timeout: {timeout_seconds:g}s)...", flush=True)
    try:
        code = run_tests(
            root, verbose=verbose, timeout_seconds=timeout_seconds,
        )
    except subprocess.TimeoutExpired:
        print(
            f"\nUnit tests exceeded the {timeout_seconds:g}s timeout. "
            "Validation failed; increase --timeout only if the suite needs more time.",
            file=sys.stderr,
        )
        return 124
    except OSError as exc:
        print(f"Could not run the test runner: {exc}", file=sys.stderr)
        return 2
    if code:
        print(
            f"Unit tests failed (runner exit code {code}).",
            file=sys.stderr,
        )
        return 130 if code == 130 else 1
    print("Local validation passed. No GitHub sync was performed.", flush=True)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--verbose", action="store_true", help="show each unit test as it runs"
    )
    parser.add_argument(
        "--timeout", type=positive_timeout, default=DEFAULT_TIMEOUT_SECONDS,
        metavar="SECONDS", help="test-run deadline in seconds (default: 300)",
    )
    args = parser.parse_args(argv)
    try:
        return validate(
            PROJECT_ROOT, verbose=args.verbose, timeout_seconds=args.timeout,
        )
    except KeyboardInterrupt:
        print("\nLocal validation interrupted.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
