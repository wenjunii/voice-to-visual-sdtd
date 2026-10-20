import contextlib
import io
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

from scripts import check_project


class LocalValidationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.output = io.StringIO()
        stack = contextlib.ExitStack()
        self.addCleanup(stack.close)
        stack.enter_context(contextlib.redirect_stdout(self.output))
        stack.enter_context(contextlib.redirect_stderr(self.output))

    def write_source(self, relative_path, source="value = 1\n"):
        path = self.root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source, encoding="utf-8")
        return path

    def make_checkout(self):
        self.write_source("module.py")
        self.write_source("tests/test_example.py")

    def test_source_selection_includes_only_project_code(self):
        included = {
            self.write_source("module.py"),
            self.write_source("scripts/check.py"),
            self.write_source("scripts/helpers/helper.py"),
            self.write_source("tests/test_example.py"),
            self.write_source("tests/helpers/helper.py"),
        }
        for directory in (
            ".git", ".venv", "venv", "env", "ENV", "build", "dist", "unrelated",
            "tests/__pycache__", "tests/.venv", "scripts/ENV",
        ):
            self.write_source(f"{directory}/invalid.py", "def broken(\n")
        (self.root / "not_a_file.py").mkdir()

        self.assertEqual(set(check_project.source_files(self.root)), included)

    def test_syntax_check_does_not_execute_code_or_write_bytecode(self):
        path = self.write_source("module.py", "raise RuntimeError('not imported')\n")

        self.assertTrue(check_project.check_syntax([path], self.root))
        self.assertEqual(list(self.root.rglob("*.pyc")), [])
        self.assertFalse((self.root / "__pycache__").exists())

    def test_syntax_check_honors_source_encoding(self):
        path = self.root / "module.py"
        path.write_bytes(b"# coding: latin-1\nvalue = 'caf\xe9'\n")

        self.assertTrue(check_project.check_syntax([path], self.root))

    def test_syntax_check_reports_all_bad_files(self):
        first = self.write_source("first.py", "def broken(\n")
        second = self.write_source("second.py", "if:\n")

        self.assertFalse(check_project.check_syntax([first, second], self.root))
        self.assertIn("first.py", self.output.getvalue())
        self.assertIn("second.py", self.output.getvalue())

    def test_syntax_check_reports_unreadable_files(self):
        path = self.root / "missing.py"

        self.assertFalse(check_project.check_syntax([path], self.root))
        self.assertIn("missing.py", self.output.getvalue())

    def test_missing_dependencies_use_installable_package_names(self):
        with mock.patch.object(
            check_project.importlib.util,
            "find_spec",
            side_effect=[object(), None, ImportError("missing")],
        ) as find_spec:
            self.assertEqual(
                check_project.missing_dependencies(), ["python-osc", "requests"]
            )
        self.assertEqual(
            find_spec.call_args_list,
            [mock.call("numpy"), mock.call("pythonosc"), mock.call("requests")],
        )

    def test_broken_module_spec_is_reported_as_missing(self):
        with mock.patch.object(
            check_project.importlib.util, "find_spec", side_effect=ValueError
        ):
            self.assertEqual(
                check_project.missing_dependencies(),
                ["numpy", "python-osc", "requests"],
            )

    def test_empty_checkout_fails_without_starting_tests(self):
        with mock.patch.object(check_project.subprocess, "run") as run:
            self.assertEqual(check_project.validate(self.root), 2)
        run.assert_not_called()
        self.assertIn("No project Python files", self.output.getvalue())

    def test_missing_tests_fail_without_starting_runner(self):
        self.write_source("module.py")
        with mock.patch.object(check_project.subprocess, "run") as run:
            self.assertEqual(check_project.validate(self.root), 2)
        run.assert_not_called()
        self.assertIn("No discoverable test files", self.output.getvalue())

    def test_syntax_failure_does_not_check_dependencies_or_run_tests(self):
        self.make_checkout()
        self.write_source("module.py", "def broken(\n")
        with (
            mock.patch.object(check_project, "missing_dependencies") as dependencies,
            mock.patch.object(check_project.subprocess, "run") as run,
        ):
            self.assertEqual(check_project.validate(self.root), 1)
        dependencies.assert_not_called()
        run.assert_not_called()

    def test_missing_dependencies_fail_with_manual_install_hint(self):
        self.make_checkout()
        with (
            mock.patch.object(
                check_project, "missing_dependencies", return_value=["python-osc"]
            ),
            mock.patch.object(check_project.subprocess, "run") as run,
        ):
            self.assertEqual(check_project.validate(self.root), 2)
        run.assert_not_called()
        self.assertIn("Missing test dependencies: python-osc", self.output.getvalue())
        self.assertIn(sys.executable, self.output.getvalue())
        self.assertIn(str(self.root / "requirements-test.txt"), self.output.getvalue())

    def test_success_uses_current_interpreter_and_explicit_working_directory(self):
        self.make_checkout()
        with (
            mock.patch.object(check_project, "missing_dependencies", return_value=[]),
            mock.patch.object(
                check_project.subprocess,
                "run",
                return_value=subprocess.CompletedProcess([], 0),
            ) as run,
        ):
            self.assertEqual(check_project.validate(self.root), 0)
        run.assert_called_once_with(
            [sys.executable, "-m", "unittest", "discover", "-s", "tests"],
            cwd=self.root,
            check=False,
            capture_output=True,
            text=True,
            errors="replace",
        )
        self.assertIn("Local validation passed", self.output.getvalue())

    def test_verbose_mode_and_failed_test_exit_status(self):
        self.make_checkout()
        with (
            mock.patch.object(check_project, "missing_dependencies", return_value=[]),
            mock.patch.object(
                check_project.subprocess,
                "run",
                return_value=subprocess.CompletedProcess([], 1),
            ) as run,
        ):
            self.assertEqual(check_project.validate(self.root, verbose=True), 1)
        self.assertEqual(run.call_args.args[0][-1], "-v")
        self.assertIn("Unit tests failed", self.output.getvalue())
        self.assertNotIn("Local validation passed", self.output.getvalue())

    def test_real_runner_reports_success(self):
        self.make_checkout()
        self.write_source(
            "tests/test_example.py",
            "import unittest\n"
            "class ExampleTests(unittest.TestCase):\n"
            "    def test_example(self):\n"
            "        self.assertTrue(True)\n",
        )
        with mock.patch.object(check_project, "missing_dependencies", return_value=[]):
            self.assertEqual(check_project.validate(self.root), 0)
        self.assertIn("Ran 1 test", self.output.getvalue())
        self.assertIn("Local validation passed", self.output.getvalue())

    def test_real_runner_preserves_assertion_failure_details(self):
        self.make_checkout()
        self.write_source(
            "tests/test_example.py",
            "import unittest\n"
            "class ExampleTests(unittest.TestCase):\n"
            "    def test_example(self):\n"
            "        self.fail('deliberate fixture failure')\n",
        )
        with mock.patch.object(check_project, "missing_dependencies", return_value=[]):
            self.assertEqual(check_project.validate(self.root), 1)
        self.assertIn("Ran 1 test", self.output.getvalue())
        self.assertIn("AssertionError: deliberate fixture failure", self.output.getvalue())
        self.assertNotIn("Local validation passed", self.output.getvalue())

    def test_runner_start_failure_is_actionable(self):
        self.make_checkout()
        with (
            mock.patch.object(check_project, "missing_dependencies", return_value=[]),
            mock.patch.object(
                check_project.subprocess, "run", side_effect=OSError("cannot start")
            ),
        ):
            self.assertEqual(check_project.validate(self.root), 2)
        self.assertIn("Could not start the test runner", self.output.getvalue())

    def test_main_uses_script_location_and_forwards_verbose_flag(self):
        with mock.patch.object(check_project, "validate", return_value=0) as validate:
            self.assertEqual(check_project.main(["--verbose"]), 0)
        validate.assert_called_once_with(check_project.PROJECT_ROOT, verbose=True)
        self.assertEqual(
            check_project.PROJECT_ROOT,
            Path(check_project.__file__).resolve().parents[1],
        )

    def test_main_reports_interruption(self):
        with mock.patch.object(check_project, "validate", side_effect=KeyboardInterrupt):
            self.assertEqual(check_project.main([]), 130)
        self.assertIn("Local validation interrupted", self.output.getvalue())

    def test_help_does_not_start_validation(self):
        with mock.patch.object(check_project, "validate") as validate:
            with self.assertRaises(SystemExit) as raised:
                check_project.main(["--help"])
        self.assertEqual(raised.exception.code, 0)
        validate.assert_not_called()


if __name__ == "__main__":
    unittest.main()
