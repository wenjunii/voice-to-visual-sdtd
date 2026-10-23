import contextlib
import io
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

from scripts import check_project
from scripts import run_unit_tests


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
        with mock.patch.object(check_project, "run_tests") as run:
            self.assertEqual(check_project.validate(self.root), 2)
        run.assert_not_called()
        self.assertIn("No project Python files", self.output.getvalue())

    def test_missing_tests_fail_without_starting_runner(self):
        self.write_source("module.py")
        with mock.patch.object(check_project, "run_tests") as run:
            self.assertEqual(check_project.validate(self.root), 2)
        run.assert_not_called()
        self.assertIn("No discoverable test files", self.output.getvalue())

    def test_syntax_failure_does_not_check_dependencies_or_run_tests(self):
        self.make_checkout()
        self.write_source("module.py", "def broken(\n")
        with (
            mock.patch.object(check_project, "missing_dependencies") as dependencies,
            mock.patch.object(check_project, "run_tests") as run,
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
            mock.patch.object(check_project, "run_tests") as run,
        ):
            self.assertEqual(check_project.validate(self.root), 2)
        run.assert_not_called()
        self.assertIn("Missing test dependencies: python-osc", self.output.getvalue())
        self.assertIn(sys.executable, self.output.getvalue())
        self.assertIn(str(self.root / "requirements-test.txt"), self.output.getvalue())

    def test_success_uses_local_runner_and_default_deadline(self):
        self.make_checkout()
        with (
            mock.patch.object(check_project, "missing_dependencies", return_value=[]),
            mock.patch.object(
                check_project, "run_tests", return_value=0,
            ) as run,
        ):
            self.assertEqual(check_project.validate(self.root), 0)
        run.assert_called_once_with(
            self.root, verbose=False,
            timeout_seconds=check_project.DEFAULT_TIMEOUT_SECONDS,
        )
        self.assertIn("Local validation passed", self.output.getvalue())

    def test_verbose_mode_and_failed_test_exit_status(self):
        self.make_checkout()
        with (
            mock.patch.object(check_project, "missing_dependencies", return_value=[]),
            mock.patch.object(
                check_project, "run_tests", return_value=1,
            ) as run,
        ):
            self.assertEqual(
                check_project.validate(self.root, verbose=True, timeout_seconds=2.5), 1
            )
        run.assert_called_once_with(self.root, verbose=True, timeout_seconds=2.5)
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
                check_project.subprocess, "Popen", side_effect=OSError("cannot start")
            ),
        ):
            self.assertEqual(check_project.validate(self.root), 2)
        self.assertIn("Could not run the test runner", self.output.getvalue())

    def test_main_uses_script_location_and_forwards_verbose_flag(self):
        with mock.patch.object(check_project, "validate", return_value=0) as validate:
            self.assertEqual(check_project.main(["--verbose", "--timeout", "12.5"]), 0)
        validate.assert_called_once_with(
            check_project.PROJECT_ROOT, verbose=True, timeout_seconds=12.5,
        )
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

    def test_main_uses_five_minute_default_timeout(self):
        with mock.patch.object(check_project, "validate", return_value=0) as validate:
            self.assertEqual(check_project.main([]), 0)
        validate.assert_called_once_with(
            check_project.PROJECT_ROOT, verbose=False, timeout_seconds=300.0,
        )

    def test_invalid_cli_timeouts_do_not_start_validation(self):
        for value in ("0", "-1", "nan", "inf", "-inf", "1e309", "invalid"):
            with self.subTest(value=value):
                with mock.patch.object(check_project, "validate") as validate:
                    with self.assertRaises(SystemExit) as raised:
                        check_project.main([f"--timeout={value}"])
                self.assertEqual(raised.exception.code, 2)
                validate.assert_not_called()

    def test_invalid_programmatic_timeouts_do_not_start_tests(self):
        self.make_checkout()
        for value in (0, -1, float("inf"), float("nan")):
            with self.subTest(value=value):
                with mock.patch.object(check_project, "run_tests") as run:
                    self.assertEqual(
                        check_project.validate(self.root, timeout_seconds=value), 2
                    )
                run.assert_not_called()

    def test_timeout_reports_a_distinct_failure_exit_code(self):
        self.make_checkout()
        with (
            mock.patch.object(check_project, "missing_dependencies", return_value=[]),
            mock.patch.object(
                check_project, "run_tests",
                side_effect=subprocess.TimeoutExpired("runner", 1.5),
            ),
        ):
            self.assertEqual(
                check_project.validate(self.root, timeout_seconds=1.5), 124
            )
        self.assertIn("exceeded the 1.5s timeout", self.output.getvalue())
        self.assertNotIn("Local validation passed", self.output.getvalue())

    def test_child_interruption_is_not_reported_as_success(self):
        self.make_checkout()
        with (
            mock.patch.object(check_project, "missing_dependencies", return_value=[]),
            mock.patch.object(check_project, "run_tests", return_value=130),
        ):
            self.assertEqual(check_project.validate(self.root), 130)
        self.assertNotIn("Local validation passed", self.output.getvalue())

    def test_real_runner_rejects_test_files_with_no_test_cases(self):
        self.make_checkout()
        with mock.patch.object(check_project, "missing_dependencies", return_value=[]):
            self.assertEqual(check_project.validate(self.root), 1)
        self.assertIn("No tests discovered", self.output.getvalue())
        self.assertNotIn("Local validation passed", self.output.getvalue())

    def test_real_runner_rejects_a_load_tests_hook_that_removes_every_test(self):
        self.make_checkout()
        self.write_source(
            "tests/test_example.py",
            "import unittest\n"
            "class ExampleTests(unittest.TestCase):\n"
            "    def test_example(self):\n"
            "        pass\n"
            "def load_tests(loader, tests, pattern):\n"
            "    return unittest.TestSuite()\n",
        )
        with mock.patch.object(check_project, "missing_dependencies", return_value=[]):
            self.assertEqual(check_project.validate(self.root), 1)
        self.assertIn("No tests discovered", self.output.getvalue())

    def test_runner_rejects_custom_suites_that_claim_cases_but_run_none(self):
        class CountOnlySuite(unittest.TestSuite):
            def countTestCases(self):
                return 1

            def run(self, result, debug=False):
                return result

        with (
            mock.patch.object(sys, "path", sys.path.copy()),
            mock.patch.object(
                unittest.TestLoader, "discover", return_value=CountOnlySuite()
            ),
        ):
            self.assertEqual(run_unit_tests.main([]), 1)
        self.assertIn("No tests ran", self.output.getvalue())

    def test_real_runner_reports_discovery_errors(self):
        self.make_checkout()
        self.write_source(
            "tests/test_example.py", "raise RuntimeError('discovery fixture failure')\n"
        )
        with mock.patch.object(check_project, "missing_dependencies", return_value=[]):
            self.assertEqual(check_project.validate(self.root), 1)
        self.assertIn("discovery fixture failure", self.output.getvalue())
        self.assertNotIn("Local validation passed", self.output.getvalue())

    def test_real_runner_streams_utf8_stderr_without_waiting_for_a_newline(self):
        self.make_checkout()
        acknowledgement = self.root / "progress-seen"

        class ProgressOutput(io.StringIO):
            def write(self, text):
                count = super().write(text)
                if "LIVE caf\u00e9 \u4f60\u597d" in self.getvalue():
                    acknowledgement.write_text("seen", encoding="utf-8")
                return count

        self.write_source(
            "tests/test_example.py",
            "import sys, time, unittest\n"
            "from pathlib import Path\n"
            "class ExampleTests(unittest.TestCase):\n"
            "    def test_progress(self):\n"
            "        sys.stderr.write('LIVE caf\u00e9 \u4f60\u597d')\n"
            "        sys.stderr.flush()\n"
            "        deadline = time.monotonic() + 5\n"
            "        while not Path('progress-seen').exists() and time.monotonic() < deadline:\n"
            "            time.sleep(0.01)\n"
            "        self.assertTrue(Path('progress-seen').exists(), 'output was buffered')\n",
        )
        progress = ProgressOutput()
        with (
            contextlib.redirect_stdout(progress),
            mock.patch.object(check_project, "missing_dependencies", return_value=[]),
        ):
            self.assertEqual(check_project.validate(self.root, timeout_seconds=10), 0)
        self.assertTrue(acknowledgement.exists())
        self.assertIn("LIVE caf\u00e9 \u4f60\u597d", progress.getvalue())
        self.assertIn("Ran 1 test", progress.getvalue())

    def test_real_timeout_stops_and_reaps_discovery_or_test_process(self):
        self.make_checkout()
        sources = (
            "import time\ntime.sleep(30)\n",
            "import time, unittest\n"
            "class ExampleTests(unittest.TestCase):\n"
            "    def test_hang(self):\n"
            "        time.sleep(30)\n",
        )
        real_popen = subprocess.Popen
        for source in sources:
            with self.subTest(source=source):
                self.write_source("tests/test_example.py", source)
                processes = []

                def start(*args, **kwargs):
                    process = real_popen(*args, **kwargs)
                    processes.append(process)
                    return process

                with (
                    mock.patch.object(check_project, "missing_dependencies", return_value=[]),
                    mock.patch.object(check_project.subprocess, "Popen", side_effect=start),
                ):
                    self.assertEqual(
                        check_project.validate(self.root, timeout_seconds=0.5), 124
                    )
                self.assertEqual(len(processes), 1)
                self.assertIsNotNone(processes[0].poll())
                self.assertTrue(processes[0].stdout.closed)

    def test_runner_uses_current_interpreter_and_explicit_working_directory(self):
        process = mock.Mock(stdout=io.BytesIO(b"test progress"))
        process.wait.return_value = 0
        process.poll.return_value = 0
        with mock.patch.object(check_project.subprocess, "Popen", return_value=process) as start:
            self.assertEqual(
                check_project.run_tests(self.root, verbose=True, timeout_seconds=3), 0
            )
        self.assertEqual(
            start.call_args.args[0],
            [sys.executable, "-u", str(check_project.TEST_RUNNER), "--verbose"],
        )
        options = start.call_args.kwargs
        self.assertEqual(options["cwd"], self.root)
        self.assertEqual(options["stdin"], subprocess.DEVNULL)
        self.assertEqual(options["stdout"], subprocess.PIPE)
        self.assertEqual(options["stderr"], subprocess.STDOUT)
        self.assertEqual(options["env"]["PYTHONIOENCODING"], "utf-8")
        self.assertIn("test progress", self.output.getvalue())
        self.assertTrue(process.stdout.closed)
        process.kill.assert_not_called()

    def test_interruption_stops_and_reaps_the_runner(self):
        process = mock.Mock(stdout=io.BytesIO())
        process.wait.side_effect = [KeyboardInterrupt, 0]
        process.poll.return_value = None
        with mock.patch.object(check_project.subprocess, "Popen", return_value=process):
            with self.assertRaises(KeyboardInterrupt):
                check_project.run_tests(self.root, verbose=False, timeout_seconds=3)
        process.kill.assert_called_once_with()
        self.assertEqual(
            process.wait.call_args, mock.call(timeout=check_project.PROCESS_CLEANUP_SECONDS)
        )
        self.assertTrue(process.stdout.closed)

    def test_output_reader_creation_or_start_failure_stops_the_runner(self):
        for phase in ("create", "start"):
            with self.subTest(phase=phase):
                process = mock.Mock(stdout=io.BytesIO())
                process.wait.return_value = 0
                process.poll.return_value = None
                target = "threading.Thread" if phase == "create" else "threading.Thread.start"
                with (
                    mock.patch.object(check_project.subprocess, "Popen", return_value=process),
                    mock.patch(target, side_effect=RuntimeError("cannot start")),
                ):
                    with self.assertRaisesRegex(OSError, "Could not start test output reader"):
                        check_project.run_tests(self.root, verbose=False, timeout_seconds=3)
                process.kill.assert_called_once_with()
                self.assertTrue(process.stdout.closed)

    def test_output_read_failure_stops_the_runner(self):
        class BrokenPipe(io.BytesIO):
            def read1(self, size=-1):
                raise OSError("broken output pipe")

        process = mock.Mock(stdout=BrokenPipe())
        process.wait.return_value = 0
        process.poll.return_value = None
        with mock.patch.object(check_project.subprocess, "Popen", return_value=process):
            with self.assertRaisesRegex(OSError, "broken output pipe"):
                check_project.run_tests(self.root, verbose=False, timeout_seconds=3)
        process.kill.assert_called_once_with()
        self.assertTrue(process.stdout.closed)

    def test_stream_normalizes_split_utf8_and_windows_line_endings(self):
        class ChunkedPipe(io.BytesIO):
            def __init__(self):
                super().__init__()
                self.chunks = iter((b"caf\xc3", b"\xa9\r", b"\nnext\rline\n", b""))

            def read1(self, size=-1):
                return next(self.chunks)

        pipe = ChunkedPipe()
        output = io.StringIO()
        errors = []
        check_project._stream_test_output(pipe, output, errors)
        self.assertEqual(output.getvalue(), "caf\u00e9\nnext\nline\n")
        self.assertEqual(errors, [])
        self.assertTrue(pipe.closed)

    def test_stream_flushes_incomplete_utf8_without_losing_prior_output(self):
        pipe = io.BytesIO(b"progress\n\xc3")
        output = io.StringIO()
        errors = []
        check_project._stream_test_output(pipe, output, errors)
        self.assertEqual(output.getvalue(), "progress\n\ufffd")
        self.assertEqual(errors, [])
        self.assertTrue(pipe.closed)


if __name__ == "__main__":
    unittest.main()
