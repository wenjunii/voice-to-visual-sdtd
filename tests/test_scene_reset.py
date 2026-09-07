import threading
import time
import unittest
from contextlib import redirect_stdout
from io import StringIO
from unittest.mock import Mock

import numpy as np

from audio_sources import AudioSourceFinished
from backend_errors import RetryableTranscriptionError
from osc_output import RecordingOutputPublisher
from streaming_core import AudioSegment, TranscriptStabilizer
from tests.test_transcriber_controls import make_pipeline


def segment(segment_id, *, is_final=True):
    return AudioSegment(
        segment_id, 1, np.full(1600, segment_id, dtype=np.int16), is_final
    )


class SceneResetTests(unittest.TestCase):
    def make_pipeline(self, **kwargs):
        publisher = RecordingOutputPublisher()
        pipeline = make_pipeline(output_publisher=publisher, **kwargs)
        self.addCleanup(pipeline.close)
        return pipeline, publisher

    def test_reset_clears_buffered_queued_and_stabilized_speech(self):
        pipeline, publisher = self.make_pipeline()
        now = time.monotonic()
        pipeline.segmenter.add_chunk(np.ones(1024, dtype=np.int16), True)
        pipeline.scheduler.submit_final(segment(1), now)
        retry = pipeline.scheduler.next_job(now)
        pipeline.scheduler.retry_final(retry, now, 20.0)
        pipeline.scheduler.submit_final(segment(2), now)
        pipeline.scheduler.submit_partial(segment(3, is_final=False), now)
        pipeline.stabilizers[1] = TranscriptStabilizer()
        pipeline.scene_memory.update(1, "old forest", is_final=True)
        pipeline.last_text = "old forest"
        pipeline.last_prompt_token_count = 42
        pipeline.is_speaking = True

        pipeline.apply_control("reset_scene", None)

        self.assertEqual(pipeline.last_text, "")
        self.assertEqual(pipeline.stabilizers, {})
        self.assertFalse(pipeline.segmenter.active)
        self.assertIsNone(pipeline.segmenter.snapshot())
        self.assertIsNone(pipeline.scheduler.next_job(now))
        self.assertEqual(pipeline.scheduler.metrics().dropped_finals, 0)
        self.assertEqual(pipeline.scene_memory.update(4, "new lake"), "new lake")
        messages = [
            (message.address, message.value) for message in publisher.messages
        ]
        for expected in (
            ("/partial_text", ""),
            ("/scene_context", ""),
            ("/prompt_tokens", 0),
            ("/scene_reset", 1),
            ("/queue_depth", 0),
            ("/is_speaking", 0),
        ):
            self.assertIn(expected, messages)
        self.assertFalse(any(address == "/prompt" for address, _ in messages))
        event = pipeline.control_logger.info.call_args.kwargs["extra"]
        self.assertEqual(event["discarded_queued_jobs"], 3)
        pipeline.apply_control("reset_scene", None)
        event = pipeline.control_logger.info.call_args.kwargs["extra"]
        self.assertEqual(event["discarded_queued_jobs"], 0)

    def test_reset_during_transcription_discards_old_result_and_accepts_new_speech(self):
        for is_final in (True, False):
            with self.subTest(is_final=is_final):
                pipeline, publisher = self.make_pipeline()
                pipeline.request_interval_ready = Mock(return_value=True)
                if is_final:
                    pipeline.scheduler.submit_final(segment(1), time.monotonic())
                else:
                    pipeline.scheduler.submit_partial(
                        segment(1, is_final=False), time.monotonic()
                    )
                started = threading.Event()
                release = threading.Event()

                def transcribe(samples, **kwargs):
                    if samples[0] == 1:
                        started.set()
                        if not release.wait(3.0):
                            raise AssertionError("Transcription was not released")
                        return "old forest"
                    return "new lake"

                pipeline.backend_adapter.transcribe.side_effect = transcribe
                worker = threading.Thread(
                    target=pipeline.transcription_loop, daemon=True
                )
                with redirect_stdout(StringIO()):
                    worker.start()
                    try:
                        self.assertTrue(started.wait(3.0))
                        pipeline.scheduler.submit_final(segment(2), time.monotonic())
                        pipeline.apply_control("reset_scene", None)
                        self.assertTrue(worker.is_alive())
                        self.assertEqual(pipeline.last_text, "")
                        pipeline.submit_final_segment(
                            segment(3), time.monotonic(), source="completed"
                        )
                        pipeline.audio_source_finished.set()
                        release.set()
                        worker.join(3.0)
                        self.assertFalse(worker.is_alive())
                    finally:
                        release.set()
                        pipeline.request_shutdown()
                        worker.join(3.0)

                self.assertEqual(pipeline.backend_adapter.transcribe.call_count, 2)
                self.assertEqual(pipeline.last_text, "new lake")
                self.assertEqual(pipeline.stabilizers, {})
                self.assertEqual(pipeline.scheduler.metrics().processed, 1)
                scene_messages = [
                    message.value for message in publisher.messages
                    if message.address == "/scene_context"
                ]
                self.assertEqual(scene_messages, ["", "new lake"])
                self.assertFalse(any(
                    "old forest" in str(m.value) for m in publisher.messages
                ))
                events = [
                    call.kwargs["extra"]["event"]
                    for call in pipeline.transcription_logger.info.call_args_list
                ]
                self.assertIn("transcription_scene_discarded", events)

    def test_late_errors_cannot_requeue_work_or_fail_the_new_scene(self):
        for error in (
            RuntimeError("old failure"),
            RetryableTranscriptionError("busy", retry_after=10.0),
        ):
            with self.subTest(error=type(error).__name__):
                pipeline, publisher = self.make_pipeline()
                pipeline.scheduler.submit_final(segment(1), time.monotonic())
                pipeline.audio_source_finished.set()

                def fail_after_reset(*args, **kwargs):
                    pipeline.apply_control("reset_scene", None)
                    raise error

                pipeline.backend_adapter.transcribe.side_effect = fail_after_reset
                pipeline.transcription_loop()

                metrics = pipeline.scheduler.metrics()
                self.assertEqual(metrics.retries, 0)
                self.assertEqual(metrics.failed, 0)
                self.assertEqual(metrics.queue_depth, 0)
                self.assertEqual(pipeline.stabilizers, {})
                self.assertEqual(pipeline.last_text, "")
                pipeline.transcription_logger.error.assert_not_called()
                self.assertFalse(any(
                    m.address == "/prompt" for m in publisher.messages
                ))
                if isinstance(error, RetryableTranscriptionError):
                    self.assertFalse(pipeline.request_interval_ready(time.monotonic()))
                    self.assertEqual(pipeline.backend_status, "retrying")
                else:
                    self.assertEqual(pipeline.backend_status, "ready")

    def test_reset_during_audio_read_drops_that_chunk_and_keeps_later_audio(self):
        source = Mock()
        source.kind = "wav_replay"
        source.name = "test recording"
        source.device_index = -1
        source.finite = True
        source.reconnectable = False
        pipeline, _ = self.make_pipeline(audio_source=source)
        pipeline.vad = Mock()
        pipeline.vad.is_speech.return_value = True
        old_audio = np.full(1024, 111, dtype=np.int16).tobytes()
        new_audio = np.full(1024, 222, dtype=np.int16).tobytes()

        def read():
            if source.read.call_count == 1:
                pipeline.apply_control("reset_scene", None)
                return old_audio
            if source.read.call_count == 2:
                return new_audio
            raise AudioSourceFinished()

        source.read.side_effect = read
        pipeline.audio_callback()

        job = pipeline.scheduler.next_job(time.monotonic())
        self.assertIsNotNone(job)
        np.testing.assert_array_equal(job.segment.samples, np.full(1024, 222))
        self.assertIsNone(pipeline.scheduler.next_job(time.monotonic()))
        self.assertTrue(pipeline.audio_source_finished.is_set())

    def test_reset_during_vad_discards_the_chunk_before_segmentation(self):
        pipeline, _ = self.make_pipeline()
        pipeline.vad = Mock()

        def detect_speech(samples):
            pipeline.apply_control("reset_scene", None)
            return True

        pipeline.vad.is_speech.side_effect = detect_speech
        pipeline.process_audio_data(np.ones(1024, dtype=np.int16).tobytes())

        self.assertFalse(pipeline.segmenter.active)
        self.assertEqual(pipeline.scheduler.metrics().queue_depth, 0)

    def test_reset_acknowledgement_cannot_interleave_with_prompt_publication(self):
        for refresh in (False, True):
            with self.subTest(refresh=refresh):
                pipeline, publisher = self.make_pipeline()
                pipeline.last_text = "old forest" if refresh else ""
                building = threading.Event()
                release = threading.Event()
                reset_started = threading.Event()
                reset_finished = threading.Event()

                def build(text):
                    building.set()
                    if not release.wait(3.0):
                        raise AssertionError("Prompt build was not released")
                    return "prompt for " + text

                def reset():
                    reset_started.set()
                    pipeline.apply_control("reset_scene", None)
                    reset_finished.set()

                pipeline.build_visual_prompt = Mock(side_effect=build)
                emit = (
                    pipeline.refresh_visual_prompt if refresh
                    else lambda: pipeline.emit_transcript("old forest", is_final=True)
                )
                emitter = threading.Thread(target=emit, daemon=True)
                resetter = threading.Thread(target=reset, daemon=True)
                with redirect_stdout(StringIO()):
                    emitter.start()
                    try:
                        self.assertTrue(building.wait(3.0))
                        resetter.start()
                        self.assertTrue(reset_started.wait(3.0))
                        self.assertFalse(reset_finished.wait(0.1))
                    finally:
                        release.set()
                        emitter.join(3.0)
                        if resetter.ident is not None:
                            resetter.join(3.0)
                self.assertFalse(emitter.is_alive())
                self.assertFalse(resetter.is_alive())
                self.assertTrue(reset_finished.is_set())
                messages = publisher.messages
                reset_index = next(
                    i for i, m in enumerate(messages) if m.address == "/scene_reset"
                )
                self.assertTrue(any(
                    m.address == "/prompt" for m in messages[:reset_index]
                ))
                self.assertFalse(any(
                    m.address in {"/prompt", "/transcript_final"}
                    for m in messages[reset_index:]
                ))
                self.assertEqual(pipeline.last_text, "")
                self.assertFalse(pipeline.refresh_visual_prompt())


if __name__ == "__main__":
    unittest.main()
