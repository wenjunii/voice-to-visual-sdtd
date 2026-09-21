import threading
import unittest
from contextlib import redirect_stdout
from dataclasses import replace
from io import StringIO
from unittest.mock import Mock, patch

from osc_output import OscOutputPublisher, RecordingOutputPublisher
from runtime_config import RuntimeConfig
from tests.test_osc_output import make_snapshot
from tests.test_transcriber_controls import (
    TestLogSession, make_backend_adapter, make_pipeline,
)
from transcriber import RealTimePipeline


class SimulatedClient:
    def __init__(self):
        self.available = False
        self.attempted = []
        self.sent = []
        self.close_count = 0

    def send_message(self, address, value):
        self.attempted.append((address, value))
        if not self.available:
            raise OSError("simulated temporary send failure")
        self.sent.append((address, value))

    def close(self):
        self.close_count += 1


def prompts(messages):
    return [value for address, value in messages if address == "/prompt"]


class OscPromptRetryTests(unittest.TestCase):
    def make_output(self, **settings):
        client = SimulatedClient()
        clock = Mock(return_value=100.0)
        logger = Mock()
        publisher = OscOutputPublisher(
            "127.0.0.1", 7000, client_factory=lambda *_: client,
            clock=clock, logger=logger, **settings,
        )
        self.addCleanup(publisher.close)
        return publisher, client, clock, logger

    def test_recovery_retries_only_latest_prompt_once_due(self):
        publisher, client, clock, logger = self.make_output()
        self.assertFalse(publisher.send("/prompt", "old private forest"))
        clock.return_value = 100.1
        self.assertFalse(publisher.send("/prompt", "latest private lake"))
        self.assertEqual(prompts(client.attempted), ["old private forest"])

        client.available = True
        clock.return_value = 100.49
        publisher.publish_status(make_snapshot(), force=True)
        self.assertEqual(prompts(client.sent), [])
        self.assertEqual(logger.info.call_args.kwargs["extra"]["event"], "osc_output_recovered")

        clock.return_value = 100.5
        publisher.publish_status(make_snapshot(), force=True)
        self.assertEqual(prompts(client.sent), ["latest private lake"])
        self.assertIsNone(publisher._pending_prompt)
        self.assertEqual(logger.info.call_args.kwargs["extra"]["event"], "osc_prompt_recovered")
        self.assertNotIn("private", str(logger.mock_calls))

        clock.return_value = 200.0
        publisher.publish_status(make_snapshot(), force=True)
        self.assertEqual(prompts(client.attempted), ["old private forest", "latest private lake"])

    def test_backoff_doubles_to_cap_and_forced_status_never_bypasses_it(self):
        publisher, client, clock, _ = self.make_output(
            prompt_retry_base_seconds=0.25, prompt_retry_max_seconds=1.0
        )
        publisher.send("/prompt", "forest")
        for now, attempts in (
            (100.24, 1), (100.25, 2), (100.74, 2), (100.75, 3),
            (101.74, 3), (101.75, 4), (102.74, 4), (102.75, 5),
        ):
            with self.subTest(now=now):
                clock.return_value = now
                publisher.publish_status(make_snapshot(), force=True)
                self.assertEqual(len(prompts(client.attempted)), attempts)

    def test_throttled_status_still_services_due_prompt(self):
        publisher, client, clock, _ = self.make_output(status_interval=10.0)
        publisher.publish_status(make_snapshot(), force=True)
        publisher.send("/prompt", "forest")
        client.available = True
        clock.return_value = 100.5

        self.assertFalse(publisher.publish_status(make_snapshot()))

        self.assertEqual(client.sent, [("/prompt", "forest")])

    def test_new_prompt_at_deadline_replaces_old_and_success_resets_backoff(self):
        publisher, client, clock, _ = self.make_output()
        publisher.send("/prompt", "old forest")
        clock.return_value = 100.5
        publisher.publish_status(make_snapshot())
        client.available = True
        clock.return_value = 101.5
        self.assertTrue(publisher.send("/prompt", "fresh lake"))
        self.assertEqual(prompts(client.sent), ["fresh lake"])

        client.available = False
        clock.return_value = 102.0
        publisher.send("/prompt", "new city")
        client.available = True
        clock.return_value = 102.5
        publisher.publish_status(make_snapshot(), force=True)
        self.assertEqual(prompts(client.sent), ["fresh lake", "new city"])

    def test_reset_clears_pending_prompt_even_when_reset_send_fails(self):
        publisher, client, clock, _ = self.make_output()
        publisher.send("/prompt", "old forest")
        self.assertFalse(publisher.send("/scene_reset", 1))
        self.assertIsNone(publisher._pending_prompt)
        client.available = True
        clock.return_value = 200.0
        publisher.publish_status(make_snapshot(), force=True)
        self.assertEqual(prompts(client.sent), [])
        self.assertTrue(publisher.send("/prompt", "new lake"))
        self.assertEqual(prompts(client.sent), ["new lake"])

    def test_close_clears_pending_without_flushing_or_accepting_new_work(self):
        publisher, client, clock, _ = self.make_output()
        publisher.send("/prompt", "forest")
        publisher.close()
        publisher.close()
        client.available = True
        clock.return_value = 200.0
        self.assertFalse(publisher.publish_status(make_snapshot(), force=True))
        self.assertFalse(publisher.send("/prompt", "new lake"))
        self.assertIsNone(publisher._pending_prompt)
        self.assertEqual(prompts(client.attempted), ["forest"])
        self.assertEqual(client.close_count, 1)

    def test_event_and_transcript_messages_are_not_replayed(self):
        publisher, client, clock, _ = self.make_output()
        for address in ("/transcript_final", "/partial_text", "/control_ack", "/scene_reset"):
            self.assertFalse(publisher.send(address, "original event"))
        client.available = True
        clock.return_value = 200.0
        publisher.publish_status(make_snapshot(), force=True)
        self.assertFalse(any(value == "original event" for _, value in client.sent))

    def test_retry_intervals_are_validated_before_creating_socket(self):
        factory = Mock()
        for name in ("prompt_retry_base_seconds", "prompt_retry_max_seconds"):
            for value in (0, -1, float("nan"), float("inf")):
                with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                    OscOutputPublisher("127.0.0.1", 7000, client_factory=factory, **{name: value})
        with self.assertRaises(ValueError):
            OscOutputPublisher(
                "127.0.0.1", 7000, client_factory=factory,
                prompt_retry_base_seconds=6.0, prompt_retry_max_seconds=5.0,
            )
        factory.assert_not_called()

    def test_reset_is_serialized_with_an_in_progress_retry(self):
        publisher, client, clock, _ = self.make_output()
        publisher.send("/prompt", "old forest")
        client.available = True
        clock.return_value = 100.5
        started = threading.Event()
        release = threading.Event()
        resetting = threading.Event()
        reset_done = threading.Event()
        send = client.send_message

        def blocked_send(address, value):
            if address == "/prompt":
                started.set()
                if not release.wait(3.0):
                    raise AssertionError("Retry was not released")
            send(address, value)

        def reset():
            resetting.set()
            publisher.send("/scene_reset", 1)
            reset_done.set()

        client.send_message = blocked_send
        retry = threading.Thread(target=lambda: publisher.publish_status(make_snapshot()), daemon=True)
        resetter = threading.Thread(target=reset, daemon=True)
        retry.start()
        try:
            self.assertTrue(started.wait(3.0))
            resetter.start()
            self.assertTrue(resetting.wait(3.0))
            self.assertFalse(reset_done.wait(0.05))
        finally:
            release.set()
            retry.join(3.0)
            if resetter.ident is not None:
                resetter.join(3.0)
        self.assertFalse(retry.is_alive())
        self.assertFalse(resetter.is_alive())
        self.assertTrue(reset_done.is_set())
        clock.return_value = 200.0
        publisher.publish_status(make_snapshot(), force=True)
        reset_index = client.sent.index(("/scene_reset", 1))
        self.assertEqual(prompts(client.sent[:reset_index]), ["old forest"])
        self.assertEqual(prompts(client.sent[reset_index:]), [])

    def test_pipeline_recovers_without_new_speech_or_duplicate_final_events(self):
        publisher, client, clock, _ = self.make_output()
        pipeline = make_pipeline(output_publisher=publisher)
        self.addCleanup(pipeline.close)
        with redirect_stdout(StringIO()):
            pipeline.emit_transcript("a quiet forest")
            self.assertFalse(pipeline.prompt_logger.info.call_args.kwargs["extra"]["osc_sent"])
            client.available = True
            pipeline.send_runtime_status(force=True)
            pipeline.emit_transcript("a quiet forest", is_final=True)
            self.assertEqual(prompts(client.sent), [])
            clock.return_value = 100.5
            pipeline.send_runtime_status(force=True)
            clock.return_value = 200.0
            pipeline.send_runtime_status(force=True)
        self.assertEqual(len(prompts(client.sent)), 1)
        self.assertIn("a quiet forest", prompts(client.sent)[0])
        self.assertEqual(
            [value for address, value in client.sent if address == "/transcript_final"],
            ["a quiet forest"],
        )

    def test_new_speech_and_controls_replace_pending_formatted_prompt(self):
        publisher, client, clock, _ = self.make_output()
        pipeline = make_pipeline(output_publisher=publisher)
        self.addCleanup(pipeline.close)
        with redirect_stdout(StringIO()):
            pipeline.emit_transcript("old forest")
            clock.return_value = 100.1
            pipeline.emit_transcript("fresh lake")
            pipeline.apply_control("gender", "woman")
            self.assertEqual(len(prompts(client.attempted)), 1)
            self.assertFalse(pipeline.prompt_logger.info.call_args.kwargs["extra"]["osc_sent"])
            client.available = True
            clock.return_value = 100.5
            pipeline.send_runtime_status(force=True)
        self.assertEqual(len(prompts(client.sent)), 1)
        self.assertIn("fresh lake", prompts(client.sent)[0])
        self.assertIn("woman", prompts(client.sent)[0])
        self.assertNotIn("old forest", prompts(client.sent)[0])

    def test_pipeline_scene_reset_does_not_replay_old_pending_prompt(self):
        publisher, client, clock, _ = self.make_output()
        pipeline = make_pipeline(output_publisher=publisher)
        self.addCleanup(pipeline.close)
        with redirect_stdout(StringIO()):
            pipeline.emit_transcript("old forest")
            pipeline.reset_scene()
            client.available = True
            clock.return_value = 200.0
            pipeline.send_runtime_status(force=True)
            self.assertEqual(prompts(client.sent), [])
            pipeline.emit_transcript("fresh lake")
        self.assertEqual(len(prompts(client.sent)), 1)
        self.assertIn("fresh lake", prompts(client.sent)[0])

    def test_pipeline_passes_instance_retry_configuration_to_transport(self):
        config = replace(
            RuntimeConfig(), osc_prompt_retry_base_seconds=0.25,
            osc_prompt_retry_max_seconds=2.0,
        )
        with patch("transcriber.OscOutputPublisher", return_value=RecordingOutputPublisher()) as factory:
            pipeline = RealTimePipeline(
                enable_vad=False, enable_osc=True, enable_prompt_budget=False,
                enable_osc_controls=False, config=config,
                backend_adapter=make_backend_adapter(), log_session=TestLogSession(),
            )
            self.addCleanup(pipeline.close)
        self.assertEqual(factory.call_args.kwargs["prompt_retry_base_seconds"], 0.25)
        self.assertEqual(factory.call_args.kwargs["prompt_retry_max_seconds"], 2.0)


if __name__ == "__main__":
    unittest.main()
