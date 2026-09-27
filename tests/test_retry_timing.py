import math
import unittest
from contextlib import redirect_stdout
from dataclasses import replace
from io import StringIO
from unittest.mock import Mock, patch

import numpy as np

from backend_errors import RetryableTranscriptionError
from osc_output import RecordingOutputPublisher
from runtime_config import RuntimeConfig
from streaming_core import AudioSegment
from tests.test_transcriber_controls import make_pipeline
from tests.test_transcription_backends import response
from transcription_backends import GroqTranslationBackend


def segment(segment_id=1, *, is_final=True):
    return AudioSegment(segment_id, 1, np.ones(1600, dtype=np.int16), is_final)


class RetryTimingTests(unittest.TestCase):
    def make_pipeline(self, **settings):
        config = replace(
            RuntimeConfig(), transcription_retry_base_seconds=1.0,
            transcription_retry_max_seconds=4.0,
            transcription_final_max_retries=5, **settings,
        )
        publisher = RecordingOutputPublisher()
        pipeline = make_pipeline(config, online=True, output_publisher=publisher)
        self.addCleanup(pipeline.close)
        return pipeline, publisher

    def take_job(self, pipeline, *, is_final=True):
        submit = pipeline.scheduler.submit_final if is_final else pipeline.scheduler.submit_partial
        submit(segment(is_final=is_final), now=100.0)
        return pipeline.scheduler.next_job(now=100.0)

    def test_invalid_adapter_hints_use_configured_capped_backoff(self):
        for hint in (float("inf"), float("-inf"), float("nan"), "1e309", "bad", -1, True, [], 10 ** 400):
            for attempt, delay in ((0, 1.0), (1, 2.0), (2, 4.0), (4, 4.0)):
                with self.subTest(hint=hint, attempt=attempt):
                    pipeline, publisher = self.make_pipeline()
                    job = replace(self.take_job(pipeline), attempts=attempt)
                    with patch("transcriber.time.monotonic", return_value=100.0):
                        pipeline.handle_retryable_failure(
                            job, RetryableTranscriptionError("temporary failure", retry_after=hint)
                        )
                    self.assertEqual(pipeline.backend_retry_not_before, 100.0 + delay)
                    self.assertFalse(pipeline.request_interval_ready(100.0 + delay - 0.01))
                    self.assertTrue(pipeline.request_interval_ready(100.0 + delay))
                    self.assertIsNone(pipeline.scheduler.next_job(100.0 + delay - 0.01))
                    retry = pipeline.scheduler.next_job(100.0 + delay)
                    self.assertEqual(retry.ready_at, 100.0 + delay)
                    self.assertEqual(retry.attempts, attempt + 1)
                    status = {m.address: m.value for m in publisher.messages}
                    self.assertEqual(status["/retry_in"], delay)
                    event = pipeline.transcription_logger.warning.call_args.kwargs["extra"]
                    self.assertEqual(event["retry_in_seconds"], delay)
                    self.assertTrue(math.isfinite(event["retry_in_seconds"]))

    def test_valid_adapter_hints_are_preserved_even_above_fallback_cap(self):
        for hint, delay in ((0, 0.0), ("0.5", 0.5), (60.0, 60.0)):
            with self.subTest(hint=hint):
                pipeline, publisher = self.make_pipeline()
                job = self.take_job(pipeline)
                with patch("transcriber.time.monotonic", return_value=100.0):
                    pipeline.handle_retryable_failure(
                        job, RetryableTranscriptionError("busy", retry_after=hint)
                    )
                self.assertEqual(pipeline.backend_retry_not_before, 100.0 + delay)
                status = {m.address: m.value for m in publisher.messages}
                self.assertEqual(status["/retry_in"], delay)

    def test_fallback_does_not_shorten_an_existing_endpoint_cooldown(self):
        pipeline, _ = self.make_pipeline()
        job = self.take_job(pipeline)
        pipeline.backend_retry_not_before = 110.0
        with patch("transcriber.time.monotonic", return_value=100.0):
            pipeline.handle_retryable_failure(
                job, RetryableTranscriptionError("busy", retry_after=float("nan"))
            )
        self.assertEqual(pipeline.backend_retry_not_before, 110.0)
        self.assertFalse(pipeline.request_interval_ready(109.0))
        self.assertTrue(pipeline.request_interval_ready(110.0))

    def test_groq_bad_header_recovers_through_the_real_transcription_loop(self):
        for status in (429, 503):
            for hint in ("1e309", "nan", "inf", "-2.5", "not-a-date"):
                with self.subTest(status=status, hint=hint):
                    config = replace(
                        RuntimeConfig(), transcription_backend="groq",
                        groq_api_key="test-key", groq_log_latency=False,
                        groq_transcription_interval=0.1,
                        transcription_retry_base_seconds=1.0,
                        transcription_retry_max_seconds=4.0,
                    )
                    clock = Mock(return_value=100.0)
                    session = Mock()
                    attempts = []
                    replies = [response(status=status, headers={"Retry-After": hint}),
                               response(text="a fresh lake")]

                    def post(*args, **kwargs):
                        attempts.append(clock.return_value)
                        return replies.pop(0)

                    def advance(_timeout):
                        clock.return_value += 0.25
                        if clock.return_value > 105.0:
                            raise AssertionError("Backend did not recover from its retry hint")
                        return False

                    session.post.side_effect = post
                    backend = GroqTranslationBackend(config, sample_rate=16000, session=session)
                    publisher = RecordingOutputPublisher()
                    pipeline = make_pipeline(config, backend_adapter=backend, output_publisher=publisher)
                    self.addCleanup(pipeline.close)
                    pipeline.scheduler.submit_final(segment(), now=100.0)
                    pipeline.audio_source_finished.set()
                    with (
                        patch("transcriber.time.monotonic", clock),
                        patch.object(pipeline.stop_event, "wait", side_effect=advance),
                        redirect_stdout(StringIO()),
                    ):
                        pipeline.transcription_loop()

                    self.assertEqual(attempts, [100.0, 101.0])
                    self.assertEqual(pipeline.last_text, "a fresh lake")
                    self.assertEqual(pipeline.scheduler.metrics().retries, 1)
                    self.assertEqual(pipeline.scheduler.metrics().processed, 1)
                    self.assertEqual(pipeline.scheduler.metrics().failed, 0)
                    self.assertEqual(pipeline.scheduler.metrics().dropped_stale, 0)
                    self.assertTrue(all(
                        math.isfinite(m.value) for m in publisher.messages if m.address == "/retry_in"
                    ))
                    self.assertEqual(
                        [m.value for m in publisher.messages if m.address == "/transcript_final"],
                        ["a fresh lake"],
                    )

    def test_expired_queue_and_scene_reset_do_not_leave_permanent_cooldown(self):
        pipeline, _ = self.make_pipeline()
        job = self.take_job(pipeline)
        clock = Mock(return_value=100.0)
        with patch("transcriber.time.monotonic", clock), redirect_stdout(StringIO()):
            pipeline.handle_retryable_failure(
                job, RetryableTranscriptionError("busy", retry_after=float("inf"))
            )
            clock.return_value = 131.0
            pipeline.send_runtime_status(force=True)
            self.assertEqual(pipeline.scheduler.metrics().queue_depth, 0)
            self.assertEqual(pipeline.scheduler.metrics().dropped_stale, 1)
            self.assertTrue(pipeline.request_interval_ready(131.0))
            pipeline.reset_scene()
            self.assertTrue(pipeline.request_interval_ready(131.0))
            pipeline.scheduler.submit_final(segment(2), now=131.0)
            pipeline.audio_source_finished.set()
            pipeline.transcription_loop()
        pipeline.backend_adapter.transcribe.assert_called_once()
        self.assertEqual(pipeline.scheduler.metrics().processed, 1)

    def test_late_failure_after_reset_uses_finite_endpoint_delay_without_requeue(self):
        pipeline, _ = self.make_pipeline()
        job = self.take_job(pipeline)
        generation = pipeline._scene_generation
        with patch("transcriber.time.monotonic", return_value=100.0):
            pipeline.reset_scene()
            pipeline.handle_retryable_failure(
                job, RetryableTranscriptionError("old failure", retry_after=float("inf")),
                scene_generation=generation,
            )
        self.assertEqual(pipeline.backend_retry_not_before, 101.0)
        self.assertTrue(pipeline.request_interval_ready(101.0))
        self.assertEqual(pipeline.scheduler.metrics().queue_depth, 0)
        self.assertEqual(pipeline.scheduler.metrics().retries, 0)
        self.assertEqual(pipeline.scheduler.metrics().failed, 0)

    def test_partial_and_exhausted_final_failures_cannot_poison_cooldown(self):
        for is_final in (False, True):
            with self.subTest(is_final=is_final):
                pipeline, _ = self.make_pipeline()
                job = replace(self.take_job(pipeline, is_final=is_final), attempts=5)
                with patch("transcriber.time.monotonic", return_value=100.0):
                    pipeline.handle_retryable_failure(
                        job, RetryableTranscriptionError("busy", retry_after=float("inf"))
                    )
                self.assertEqual(pipeline.backend_retry_not_before, 104.0)
                self.assertTrue(pipeline.request_interval_ready(104.0))
                self.assertEqual(pipeline.scheduler.metrics().queue_depth, 0)
                self.assertEqual(pipeline.scheduler.metrics().failed, 1)


if __name__ == "__main__":
    unittest.main()
