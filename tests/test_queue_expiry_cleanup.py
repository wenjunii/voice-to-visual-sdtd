import threading
import unittest
from contextlib import redirect_stdout
from dataclasses import replace
from io import StringIO
from unittest.mock import Mock, patch

import numpy as np

from backend_errors import RetryableTranscriptionError
from osc_output import RecordingOutputPublisher
from runtime_config import RuntimeConfig
from streaming_core import AudioSegment, TranscriptStabilizer
from tests.test_transcriber_controls import make_pipeline


def segment(segment_id, *, is_final=True):
    return AudioSegment(segment_id, 1, np.ones(1600, dtype=np.int16), is_final)


class QueueExpiryCleanupTests(unittest.TestCase):
    def make_pipeline(self, **settings):
        publisher = RecordingOutputPublisher()
        pipeline = make_pipeline(
            replace(RuntimeConfig(), transcript_confirm_updates=1, **settings),
            output_publisher=publisher,
        )
        self.addCleanup(pipeline.close)
        return pipeline, publisher

    def test_repeated_expiry_releases_real_partial_state_during_status_checks(self):
        pipeline, publisher = self.make_pipeline()
        clock = Mock(return_value=100.0)
        with patch("transcriber.time.monotonic", clock), redirect_stdout(StringIO()):
            for segment_id in range(1, 26):
                now = clock.return_value
                pipeline.scheduler.submit_partial(segment(segment_id, is_final=False), now)
                job = pipeline.scheduler.next_job(now)
                with pipeline.scene_lock:
                    pipeline._complete_transcription_locked(job, "a quiet forest", now, now)
                self.assertIn(segment_id, pipeline.stabilizers)
                pipeline.submit_final_segment(segment(segment_id), now, source="completed")
                last_text = pipeline.last_text
                prompts_before = sum(m.address == "/prompt" for m in publisher.messages)

                clock.return_value += 31.0
                pipeline.send_runtime_status(force=True)

                self.assertEqual(pipeline.stabilizers, {})
                metrics = pipeline.scheduler.metrics()
                self.assertEqual(metrics.queue_depth, 0)
                self.assertEqual(metrics.dropped_stale, segment_id)
                self.assertEqual(metrics.dropped_expired_results, 0)
                self.assertEqual(metrics.failed, 0)
                self.assertEqual(pipeline.last_text, last_text)
                self.assertEqual(
                    sum(m.address == "/prompt" for m in publisher.messages), prompts_before
                )

        status = {m.address: m.value for m in publisher.messages}
        self.assertEqual(status["/dropped_jobs"], 25)
        self.assertEqual(status["/dropped_expired_results"], 0)
        events = [call.kwargs["extra"] for call in pipeline.scheduler_logger.info.call_args_list]
        self.assertEqual(len(events), 25)
        self.assertEqual(events[-1], {"event": "scheduler_final_expired", "segment_id": 25})
        self.assertNotIn("a quiet forest", str(events))

    def test_partial_expiry_preserves_state_until_its_final_expires(self):
        pipeline, _ = self.make_pipeline()
        state = TranscriptStabilizer()
        pipeline.stabilizers[1] = state
        pipeline.scheduler.submit_partial(segment(1, is_final=False), now=100.0)

        with patch("transcriber.time.monotonic", return_value=105.0):
            pipeline.send_runtime_status(force=True)
        self.assertIs(pipeline.stabilizers[1], state)

        pipeline.submit_final_segment(segment(1), now=105.0, source="completed")
        with patch("transcriber.time.monotonic", return_value=136.0):
            pipeline.send_runtime_status(force=True)
        self.assertEqual(pipeline.stabilizers, {})
        self.assertEqual(pipeline.scheduler.metrics().dropped_stale, 2)

    def test_expiry_leaves_other_segments_state_intact(self):
        pipeline, _ = self.make_pipeline()
        pipeline.stabilizers = {i: TranscriptStabilizer() for i in (1, 2, 3)}
        fresh_final_state = pipeline.stabilizers[2]
        active_partial_state = pipeline.stabilizers[3]
        pipeline.scheduler.submit_final(segment(1), now=100.0)
        pipeline.scheduler.submit_final(segment(2), now=101.0)
        pipeline.scheduler.submit_partial(segment(3, is_final=False), now=130.0)

        with patch("transcriber.time.monotonic", return_value=131.0):
            pipeline.send_runtime_status(force=True)

        self.assertNotIn(1, pipeline.stabilizers)
        self.assertIs(pipeline.stabilizers[2], fresh_final_state)
        self.assertIs(pipeline.stabilizers[3], active_partial_state)
        self.assertEqual(pipeline.scheduler.metrics().queue_depth, 2)

    def test_replay_completion_cleans_expired_queue_before_shutdown(self):
        pipeline, _ = self.make_pipeline()
        pipeline.stabilizers[1] = TranscriptStabilizer()
        pipeline.scheduler.submit_final(segment(1), now=100.0)
        pipeline.audio_source_finished.set()

        with patch("transcriber.time.monotonic", return_value=131.0):
            pipeline.transcription_loop()

        self.assertFalse(pipeline.is_running)
        pipeline.backend_adapter.transcribe.assert_not_called()
        self.assertEqual(pipeline.stabilizers, {})
        self.assertEqual(pipeline.scheduler.metrics().dropped_stale, 1)

    def test_waiting_retry_releases_state_when_it_expires_during_cooldown(self):
        pipeline, _ = self.make_pipeline()
        state = TranscriptStabilizer()
        pipeline.stabilizers[1] = state
        pipeline.scheduler.submit_final(segment(1), now=100.0)
        job = pipeline.scheduler.next_job(now=100.0)

        with patch("transcriber.time.monotonic", return_value=101.0):
            pipeline.handle_retryable_failure(
                job, RetryableTranscriptionError("busy", retry_after=60.0)
            )
        self.assertIs(pipeline.stabilizers[1], state)
        self.assertEqual(pipeline.scheduler.metrics().retries, 1)

        with patch("transcriber.time.monotonic", return_value=131.0):
            pipeline.send_runtime_status(force=True)

        self.assertEqual(pipeline.stabilizers, {})
        self.assertEqual(pipeline.scheduler.metrics().queue_depth, 0)
        self.assertEqual(pipeline.scheduler.metrics().dropped_stale, 1)
        self.assertEqual(pipeline.scheduler.metrics().failed, 0)
        self.assertEqual(pipeline.scheduler.metrics().dropped_expired_results, 0)
        self.assertFalse(pipeline.request_interval_ready(131.0))

    def test_running_partial_cannot_leave_state_after_its_final_is_discarded(self):
        for discard in ("expiry", "capacity"):
            with self.subTest(discard=discard):
                pipeline, publisher = self.make_pipeline(
                    transcription_partial_max_age_seconds=0.0,
                    transcription_max_final_jobs=1,
                )
                pipeline.stabilizers[1] = TranscriptStabilizer(1)
                pipeline.scheduler.submit_partial(segment(1, is_final=False), now=100.0)
                clock = Mock(return_value=100.0)
                started = threading.Event()
                release = threading.Event()

                def transcribe(*args, **kwargs):
                    if not started.is_set():
                        started.set()
                        if not release.wait(3.0):
                            raise AssertionError("Transcription was not released")
                        return "a quiet forest"
                    return "a fresh lake"

                pipeline.backend_adapter.transcribe.side_effect = transcribe
                worker = threading.Thread(target=pipeline.transcription_loop, daemon=True)
                with patch("transcriber.time.monotonic", clock), redirect_stdout(StringIO()):
                    worker.start()
                    try:
                        self.assertTrue(started.wait(3.0))
                        pipeline.submit_final_segment(segment(1), now=100.0, source="completed")
                        clock.return_value = 131.0 if discard == "expiry" else 103.0
                        pipeline.submit_final_segment(
                            segment(2), clock.return_value, source="completed"
                        )
                        self.assertNotIn(1, pipeline.stabilizers)
                        pipeline.audio_source_finished.set()
                        release.set()
                        worker.join(3.0)
                        self.assertFalse(worker.is_alive())
                    finally:
                        release.set()
                        pipeline.request_shutdown()
                        worker.join(3.0)

                self.assertEqual(pipeline.backend_adapter.transcribe.call_count, 2)
                self.assertEqual(pipeline.stabilizers, {})
                self.assertIsNone(pipeline._inflight_segment_id)
                self.assertFalse(pipeline._inflight_segment_discarded)
                self.assertEqual(pipeline.scheduler.metrics().processed, 2)
                self.assertEqual(pipeline.scheduler.metrics().dropped_expired_results, 0)
                self.assertIn("a quiet forest", pipeline.last_text)
                self.assertEqual(
                    [m.value for m in publisher.messages if m.address == "/transcript_final"],
                    ["a fresh lake"],
                )

    def test_running_partial_failure_also_clears_inflight_cleanup_tracking(self):
        for error in (RuntimeError("backend unavailable"), RetryableTranscriptionError("busy")):
            with self.subTest(error=type(error).__name__):
                pipeline, _ = self.make_pipeline(transcription_partial_max_age_seconds=0.0)
                pipeline.stabilizers[1] = TranscriptStabilizer()
                pipeline.scheduler.submit_partial(segment(1, is_final=False), now=100.0)
                pipeline.audio_source_finished.set()
                clock = Mock(return_value=100.0)

                def transcribe(*args, **kwargs):
                    pipeline.submit_final_segment(segment(1), 100.0, source="completed")
                    clock.return_value = 131.0
                    pipeline.send_runtime_status(force=True)
                    raise error

                pipeline.backend_adapter.transcribe.side_effect = transcribe
                with patch("transcriber.time.monotonic", clock):
                    pipeline.transcription_loop()

                self.assertEqual(pipeline.stabilizers, {})
                self.assertIsNone(pipeline._inflight_segment_id)
                self.assertFalse(pipeline._inflight_segment_discarded)
                self.assertEqual(pipeline.scheduler.metrics().failed, 1)
                self.assertEqual(pipeline.scheduler.metrics().dropped_stale, 1)


if __name__ == "__main__":
    unittest.main()
