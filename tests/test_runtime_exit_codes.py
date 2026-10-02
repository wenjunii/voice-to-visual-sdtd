import time
import unittest
from contextlib import redirect_stdout
from dataclasses import replace
from io import StringIO
from unittest.mock import Mock, patch

import numpy as np

from audio_sources import AudioSourceFinished
from backend_errors import RetryableTranscriptionError
from runtime_config import RuntimeConfig
from streaming_core import AudioSegment
from tests.test_transcriber_controls import make_pipeline
from transcriber import main


class RuntimeExitCodeTests(unittest.TestCase):
    def run_pipeline(self, pipeline, *, keyboard_error=None):
        self.addCleanup(pipeline.close)
        deadline = time.perf_counter() + 3.0

        def check_keyboard():
            if keyboard_error is not None:
                raise keyboard_error
            if time.perf_counter() >= deadline:
                raise AssertionError("Pipeline did not finish shutting down")
            return False

        with (
            patch("transcriber.load_runtime_config", return_value=pipeline.config),
            patch("transcriber.RealTimePipeline", return_value=pipeline),
            patch("transcriber.msvcrt.kbhit", side_effect=check_keyboard),
            redirect_stdout(StringIO()),
        ):
            return main([])

    @staticmethod
    def audio_source(*, reconnectable):
        source = Mock()
        source.kind = "microphone" if reconnectable else "wav_replay"
        source.name = "test audio"
        source.device_index = -1
        source.finite = not reconnectable
        source.reconnectable = reconnectable
        return source

    def assert_closed_with_status(self, pipeline, exit_code):
        self.assertEqual(pipeline.exit_code, exit_code)
        self.assertTrue(pipeline.stop_event.is_set())
        pipeline.backend_adapter.close.assert_called_once_with()
        stop_event = pipeline.runtime_logger.info.call_args.kwargs["extra"]
        self.assertEqual(stop_event["event"], "session_stop")
        self.assertEqual(stop_event["exit_code"], exit_code)

    def test_worker_crashes_fail_the_cli_after_cleanup(self):
        for worker_name in ("audio", "transcription"):
            with self.subTest(worker=worker_name):
                pipeline = make_pipeline()
                pipeline.audio_callback = lambda: pipeline.stop_event.wait(1.0)
                pipeline.transcription_loop = lambda: pipeline.stop_event.wait(1.0)
                crashing_worker = Mock(side_effect=RuntimeError("simulated worker crash"))
                if worker_name == "audio":
                    pipeline.audio_callback = crashing_worker
                else:
                    pipeline.transcription_loop = crashing_worker

                self.assertEqual(self.run_pipeline(pipeline), 1)

                self.assert_closed_with_status(pipeline, 1)
                crash = pipeline.runtime_logger.exception.call_args.kwargs["extra"]
                self.assertEqual(crash["event"], "worker_crashed")
                self.assertEqual(crash["worker"], worker_name)

    def test_terminal_audio_errors_fail_the_cli(self):
        for reconnectable in (False, True):
            for operation in ("open", "read"):
                with self.subTest(reconnectable=reconnectable, operation=operation):
                    source = self.audio_source(reconnectable=reconnectable)
                    getattr(source, operation).side_effect = OSError("audio unavailable")
                    config = replace(
                        RuntimeConfig(), audio_reconnect_enabled=not reconnectable,
                        audio_max_consecutive_read_errors=1,
                    )
                    pipeline = make_pipeline(config, audio_source=source)

                    self.assertEqual(self.run_pipeline(pipeline), 1)

                    self.assert_closed_with_status(pipeline, 1)
                    self.assertEqual(pipeline.audio_status, "error")
                    source.close.assert_called_once_with()
                    pipeline.runtime_logger.exception.assert_not_called()

    def test_normal_keyboard_interrupt_exits_successfully(self):
        pipeline = make_pipeline()
        pipeline.audio_callback = lambda: pipeline.stop_event.wait(1.0)
        pipeline.transcription_loop = lambda: pipeline.stop_event.wait(1.0)

        self.assertEqual(self.run_pipeline(pipeline, keyboard_error=KeyboardInterrupt()), 0)

        self.assert_closed_with_status(pipeline, 0)

    def test_worker_failure_during_shutdown_preserves_failure_status(self):
        pipeline = make_pipeline()

        def fail_during_shutdown():
            if not pipeline.stop_event.wait(1.0):
                raise AssertionError("Shutdown was never requested")
            raise RuntimeError("worker failed while stopping")

        pipeline.audio_callback = fail_during_shutdown
        pipeline.transcription_loop = lambda: pipeline.stop_event.wait(1.0)

        self.assertEqual(self.run_pipeline(pipeline, keyboard_error=KeyboardInterrupt()), 1)

        pipeline.request_shutdown("later shutdown")
        pipeline.close()
        self.assert_closed_with_status(pipeline, 1)

    def test_completed_replay_exits_successfully(self):
        source = self.audio_source(reconnectable=False)
        source.read.side_effect = AudioSourceFinished()
        pipeline = make_pipeline(audio_source=source)

        self.assertEqual(self.run_pipeline(pipeline), 0)

        self.assertTrue(pipeline.audio_source_finished.is_set())
        self.assert_closed_with_status(pipeline, 0)
        source.close.assert_called_once_with()

    def test_recovered_audio_error_does_not_fail_the_session(self):
        source = self.audio_source(reconnectable=True)
        source.open.side_effect = [OSError("temporary device failure"), None]
        pipeline = make_pipeline(audio_source=source)
        pipeline.wait_for_audio_retry = Mock(return_value=True)

        def stop_after_recovery():
            pipeline.request_shutdown("test completed")
            return b"\x00\x00" * 16

        source.read.side_effect = stop_after_recovery
        pipeline.process_audio_data = Mock()

        self.assertEqual(self.run_pipeline(pipeline), 0)

        self.assertEqual(pipeline.audio_reconnects, 1)
        self.assertEqual(source.open.call_count, 2)
        self.assertEqual(source.close.call_count, 2)
        self.assert_closed_with_status(pipeline, 0)

    def test_failed_transcription_jobs_do_not_fail_the_session(self):
        for error in (RuntimeError("bad segment"), RetryableTranscriptionError("busy")):
            with self.subTest(error=type(error).__name__):
                config = replace(RuntimeConfig(), transcription_final_max_retries=0)
                pipeline = make_pipeline(config)
                pipeline.audio_callback = pipeline.audio_source_finished.set
                pipeline.backend_adapter.transcribe.side_effect = error
                segment = AudioSegment(1, 1, np.ones(1600, dtype=np.int16), True)
                with patch("transcriber.time.monotonic", return_value=100.0):
                    pipeline.scheduler.submit_final(segment, now=100.0)

                    self.assertEqual(self.run_pipeline(pipeline), 0)

                self.assertEqual(pipeline.scheduler.metrics().failed, 1)
                self.assert_closed_with_status(pipeline, 0)


if __name__ == "__main__":
    unittest.main()
