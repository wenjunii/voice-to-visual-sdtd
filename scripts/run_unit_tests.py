"""Internal unittest entry point; normally invoked by check_project.py."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
from typing import Sequence
import unittest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    # Keep the direct entry point as tidy as check_project.py. Set the process
    # flag before importing test modules and pass the preference to child Python.
    sys.dont_write_bytecode = True
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

    # Match `python -m unittest` imports even when this script lives outside the
    # checkout under test (for example, a temporary regression-test fixture).
    sys.path.insert(0, str(Path.cwd()))
    suite = unittest.TestLoader().discover("tests")
    if suite.countTestCases() == 0:
        print("No tests discovered in tests/; validation cannot pass.", file=sys.stderr)
        return 1

    result = unittest.TextTestRunner(verbosity=2 if args.verbose else 1).run(suite)
    if result.testsRun == 0:
        print("No tests ran; validation cannot pass.", file=sys.stderr)
        return 1
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
