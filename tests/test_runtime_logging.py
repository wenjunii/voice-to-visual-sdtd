import json
import tempfile
import unittest
from contextlib import redirect_stderr
from dataclasses import replace
from io import StringIO
from pathlib import Path
from unittest.mock import Mock, patch

from runtime_config import RuntimeConfig
from runtime_logging import RuntimeLogSession
from transcriber import RealTimePipeline


class RuntimeLogSessionTests(unittest.TestCase):
    def test_disabling_every_log_destination_is_silent(self):
        stream = StringIO()
        config = replace(
            RuntimeConfig(),
            runtime_log_console_enabled=False,
            runtime_log_file="",
        )
        session = RuntimeLogSession(config, session_id="silent-test")

        with redirect_stderr(stream):
            session.logger("orchestrator").warning("must stay silent")
        session.close()

        self.assertEqual(stream.getvalue(), "")

    def test_console_logs_include_context_and_redact_credentials(self):
        stream = StringIO()
        config = replace(
            RuntimeConfig(),
            groq_api_key="console-secret",
            runtime_log_level="warning",
            runtime_log_console_enabled=True,
            runtime_log_file="",
        )
        session = RuntimeLogSession(
            config,
            stream=stream,
            session_id="console-test",
        )

        logger = session.logger("audio")
        logger.info("hidden at warning level")
        logger.warning(
            "request used Bearer console-secret",
            extra={"event": "credential_test"},
        )
        session.close()

        output = stream.getvalue()
        self.assertNotIn("hidden at warning level", output)
        self.assertNotIn("console-secret", output)
        self.assertIn("WARNING", output)
        self.assertIn("audio", output)
        self.assertIn("Bearer <redacted>", output)

    def test_file_logs_are_structured_and_redact_extra_fields(self):
        with tempfile.TemporaryDirectory() as directory:
            log_path = Path(directory) / "runtime.jsonl"
            config = replace(
                RuntimeConfig(),
                groq_api_key="file-secret",
                runtime_log_level="debug",
                runtime_log_console_enabled=False,
                runtime_log_file=str(log_path),
            )
            session = RuntimeLogSession(
                config,
                session_id="file-test",
            )

            session.logger("backend").info(
                "Backend connected with file-secret",
                extra={
                    "event": "backend_connected",
                    "backend": "groq",
                    "authorization": "Bearer file-secret",
                    "latency_seconds": 0.25,
                },
            )
            session.close()

            raw_log = log_path.read_text(encoding="utf-8")
            payload = json.loads(raw_log)

        self.assertNotIn("file-secret", raw_log)
        self.assertEqual(payload["session_id"], "file-test")
        self.assertEqual(payload["subsystem"], "backend")
        self.assertEqual(payload["event"], "backend_connected")
        self.assertEqual(payload["backend"], "groq")
        self.assertEqual(payload["authorization"], "Bearer <redacted>")
        self.assertEqual(payload["latency_seconds"], 0.25)

    def test_rotates_file_logs_at_the_configured_size(self):
        with tempfile.TemporaryDirectory() as directory:
            log_path = Path(directory) / "runtime.jsonl"
            config = replace(
                RuntimeConfig(),
                runtime_log_console_enabled=False,
                runtime_log_file=str(log_path),
                runtime_log_max_bytes=300,
                runtime_log_backup_count=2,
            )
            session = RuntimeLogSession(
                config,
                session_id="rotation-test",
            )
            logger = session.logger("scheduler")

            for index in range(20):
                logger.warning(
                    "Queue pressure detected during runtime scheduling",
                    extra={
                        "event": "queue_pressure",
                        "iteration": index,
                    },
                )
            session.close()

            rotated_files = list(Path(directory).glob("runtime.jsonl.*"))

        self.assertTrue(rotated_files)
        self.assertLessEqual(len(rotated_files), 2)

    def test_file_initialization_errors_are_actionable(self):
        with tempfile.TemporaryDirectory() as directory:
            config = replace(
                RuntimeConfig(),
                runtime_log_console_enabled=False,
                runtime_log_file=str(Path(directory) / "runtime.jsonl"),
            )

            with patch(
                "runtime_logging.RotatingFileHandler",
                side_effect=OSError("access denied"),
            ):
                with self.assertRaisesRegex(
                    RuntimeError,
                    "Could not initialize RUNTIME_LOG_FILE",
                ):
                    RuntimeLogSession(
                        config,
                        session_id="file-error-test",
                    )

    def test_existing_directory_cannot_be_used_as_log_file(self):
        with tempfile.TemporaryDirectory() as directory:
            config = replace(
                RuntimeConfig(),
                runtime_log_console_enabled=False,
                runtime_log_file=directory,
            )

            with self.assertRaisesRegex(
                RuntimeError,
                "Could not initialize RUNTIME_LOG_FILE",
            ):
                RuntimeLogSession(
                    config,
                    session_id="directory-error-test",
                )


class PipelineLoggingIntegrationTests(unittest.TestCase):
    def test_pipeline_records_session_start_and_stop(self):
        with tempfile.TemporaryDirectory() as directory:
            log_path = Path(directory) / "session.jsonl"
            config = replace(
                RuntimeConfig(),
                runtime_log_console_enabled=False,
                runtime_log_file=str(log_path),
            )
            adapter = Mock()
            adapter.name = "whisper"
            adapter.online = False
            adapter.request_interval = 0.8
            adapter.minimum_audio_seconds = 0.8
            adapter.maximum_audio_seconds = 6.0
            pipeline = RealTimePipeline(
                enable_vad=False,
                enable_osc=False,
                enable_prompt_budget=False,
                enable_osc_controls=False,
                config=config,
                backend_adapter=adapter,
            )

            pipeline.close()

            records = [
                json.loads(line)
                for line in log_path.read_text(encoding="utf-8").splitlines()
            ]

        self.assertEqual(
            [record["event"] for record in records],
            ["session_start", "session_stop"],
        )
        self.assertEqual(records[0]["backend"], "whisper")
        self.assertEqual(records[1]["processed_jobs"], 0)
        self.assertEqual(records[1]["exit_code"], 0)
        adapter.close.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
