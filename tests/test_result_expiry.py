import unittest
from contextlib import redirect_stdout
from dataclasses import replace
from io import StringIO
from unittest.mock import Mock, patch

import numpy as np

from osc_output import RecordingOutputPublisher
from runtime_config import RuntimeConfig
from streaming_core import AudioSegment, TranscriptStabilizer
from tests.test_transcriber_controls import make_pipeline


def segment(segment_id, *, is_final=True):
    return AudioSegment(
        segment_id, 1, np.full(1600, segment_id, dtype=np.int16), is_final
    )


class ResultExpiryTests(unittest.TestCase):
    def run_job(
        self, *, is_final=True, duration, limit=None, queue_wait=0.0,
        apply_delay=0.0, reset=False,
    ):
        settings = {"transcript_confirm_updates": 1}
        if limit is not None:
            setting = (
                "transcription_final_max_age_seconds" if is_final
                else "transcription_partial_max_age_seconds"
            )
            settings[setting] = limit
        publisher = RecordingOutputPublisher()
        pipeline = make_pipeline(
            replace(RuntimeConfig(), **settings), output_publisher=publisher
        )
        self.addCleanup(pipeline.close)
        clock = Mock(return_value=100.0)
        pipeline.scene_memory.update(0, "existing lake", is_final=True, now=100.0)
        pipeline.scene_memory.update = Mock(wraps=pipeline.scene_memory.update)
        pipeline.last_text = "existing lake"
        stabilizer = Mock(wraps=TranscriptStabilizer(1))
        pipeline.stabilizers[1] = stabilizer
        submit = pipeline.scheduler.submit_final if is_final else pipeline.scheduler.submit_partial
        submit(segment(1, is_final=is_final), now=100.0)
        pipeline.audio_source_finished.set()
        clock.return_value += queue_wait

        def transcribe(*args, **kwargs):
            clock.return_value += duration
            if reset:
                pipeline.apply_control("reset_scene", None)
            return "private late transcript"

        pipeline.backend_adapter.transcribe.side_effect = transcribe
        complete = pipeline._complete_transcription_locked

        def apply_result(*args):
            clock.return_value += apply_delay
            return complete(*args)

        with (
            patch("transcriber.time.monotonic", clock),
            patch.object(pipeline, "_complete_transcription_locked", side_effect=apply_result),
            redirect_stdout(StringIO()),
        ):
            pipeline.transcription_loop()
        return pipeline, publisher, stabilizer

    def test_expired_final_and_partial_results_never_reach_the_scene(self):
        for is_final, duration, limit in ((True, 31.0, 30.0), (False, 5.0, 4.0)):
            with self.subTest(is_final=is_final):
                pipeline, publisher, stabilizer = self.run_job(
                    is_final=is_final, duration=duration
                )

                pipeline.backend_adapter.transcribe.assert_called_once()
                pipeline.scene_memory.update.assert_not_called()
                stabilizer.update.assert_not_called()
                self.assertEqual(pipeline.last_text, "existing lake")
                if is_final:
                    self.assertNotIn(1, pipeline.stabilizers)
                else:
                    self.assertIs(pipeline.stabilizers[1], stabilizer)
                self.assertFalse(any(
                    m.address in {"/prompt", "/partial_text", "/scene_context", "/transcript_final"}
                    for m in publisher.messages
                ))
                metrics = pipeline.scheduler.metrics()
                self.assertEqual(metrics.processed, 0)
                self.assertEqual(metrics.failed, 0)
                self.assertEqual(metrics.dropped_stale, 1)
                self.assertEqual(metrics.dropped_expired_results, 1)
                status = {m.address: m.value for m in publisher.messages}
                self.assertEqual(status["/dropped_jobs"], 1)
                self.assertEqual(status["/dropped_expired_results"], 1)
                self.assertEqual(status["/latency_total"], duration)
                self.assertEqual(status["/latency_asr"], duration)
                self.assertEqual(status["/backend_status"], "ready")
                event = pipeline.transcription_logger.warning.call_args.kwargs["extra"]
                self.assertEqual(event["event"], "transcription_result_expired")
                self.assertEqual(event["result_age_seconds"], duration)
                self.assertEqual(event["max_age_seconds"], limit)
                self.assertNotIn("private late transcript", str(event))
                pipeline.close()
                stop = pipeline.runtime_logger.info.call_args.kwargs["extra"]
                self.assertEqual(stop["event"], "session_stop")
                self.assertEqual(stop["dropped_expired_results"], 1)
                self.assertEqual(stop["dropped_jobs"], 1)

    def test_results_within_or_at_the_limit_and_unlimited_results_are_accepted(self):
        for is_final, limit in ((False, 4.0), (True, 30.0)):
            cases = ((limit - 0.1, limit), (limit, limit), (1000.0, 0.0))
            for duration, configured_limit in cases:
                with self.subTest(is_final=is_final, duration=duration, limit=configured_limit):
                    pipeline, publisher, stabilizer = self.run_job(
                        is_final=is_final, duration=duration, limit=configured_limit
                    )
                    stabilizer.update.assert_called_once()
                    pipeline.scene_memory.update.assert_called_once()
                    self.assertIn("private late transcript", pipeline.last_text)
                    self.assertTrue(any(m.address == "/prompt" for m in publisher.messages))
                    self.assertEqual(pipeline.scheduler.metrics().processed, 1)
                    self.assertEqual(pipeline.scheduler.metrics().dropped_expired_results, 0)

    def test_age_includes_queue_time_and_wait_before_applying_the_result(self):
        for queue_wait, apply_delay in ((2.0, 0.0), (0.0, 2.0)):
            with self.subTest(queue_wait=queue_wait, apply_delay=apply_delay):
                pipeline, _, stabilizer = self.run_job(
                    duration=2.0, limit=3.0,
                    queue_wait=queue_wait, apply_delay=apply_delay,
                )
                stabilizer.update.assert_not_called()
                self.assertEqual(pipeline.last_total_latency, 4.0)
                self.assertEqual(pipeline.last_inference_latency, 2.0)
                self.assertEqual(pipeline.scheduler.metrics().dropped_expired_results, 1)

    def test_scene_reset_takes_precedence_over_result_expiry(self):
        pipeline, publisher, stabilizer = self.run_job(duration=31.0, reset=True)

        stabilizer.update.assert_not_called()
        self.assertEqual(pipeline.last_text, "")
        self.assertEqual(pipeline.stabilizers, {})
        self.assertEqual(pipeline.scheduler.metrics().dropped_stale, 0)
        self.assertEqual(pipeline.scheduler.metrics().dropped_expired_results, 0)
        self.assertFalse(any(m.address == "/prompt" for m in publisher.messages))
        events = [
            call.kwargs["extra"]["event"]
            for call in pipeline.transcription_logger.info.call_args_list
        ]
        self.assertIn("transcription_scene_discarded", events)

    def test_pipeline_continues_with_fresh_speech_after_expired_result(self):
        publisher = RecordingOutputPublisher()
        pipeline = make_pipeline(output_publisher=publisher)
        self.addCleanup(pipeline.close)
        clock = Mock(return_value=100.0)
        pipeline.scheduler.submit_final(segment(1), now=100.0)
        pipeline.audio_source_finished.set()

        def transcribe(samples, **kwargs):
            if samples[0] == 1:
                clock.return_value += 31.0
                pipeline.submit_final_segment(
                    segment(2), clock.return_value, source="completed"
                )
                return "outdated forest"
            clock.return_value += 1.0
            return "fresh lake"

        pipeline.backend_adapter.transcribe.side_effect = transcribe
        with patch("transcriber.time.monotonic", clock), redirect_stdout(StringIO()):
            pipeline.transcription_loop()

        self.assertEqual(pipeline.backend_adapter.transcribe.call_count, 2)
        self.assertEqual(pipeline.last_text, "fresh lake")
        self.assertEqual(pipeline.stabilizers, {})
        self.assertEqual(pipeline.scheduler.metrics().processed, 1)
        self.assertEqual(pipeline.scheduler.metrics().dropped_expired_results, 1)
        self.assertEqual(
            [m.value for m in publisher.messages if m.address == "/transcript_final"],
            ["fresh lake"],
        )


if __name__ == "__main__":
    unittest.main()
