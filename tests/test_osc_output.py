import unittest
from unittest.mock import Mock

from osc_output import (
    NullOutputPublisher,
    OscMessage,
    OscOutputPublisher,
    RecordingOutputPublisher,
    RuntimeStatusSnapshot,
)


def make_snapshot(**changes):
    values = {
        "backend_status": "ready",
        "backend": "whisper",
        "is_speaking": True,
        "queue_depth": 2,
        "latency_total": 0.5,
        "latency_asr": 0.25,
        "retry_in": 1.5,
        "dropped_jobs": 7,
        "audio_status": "ready",
        "audio_source": "wav_replay",
        "audio_reconnects": 1,
        "audio_error": "",
        "audio_device_index": -1,
        "audio_device_name": "WAV replay: fixture.wav",
        "gender": "neutral",
        "age": "adult",
        "visual_mode": "asian_american",
        "prompt_style": "human_focus",
        "language": "auto",
        "prompt_budget_mode": "exact",
        "dropped_final_oldest": 2,
        "dropped_final_newest": 1,
        "dropped_expired_results": 4,
    }
    values.update(changes)
    return RuntimeStatusSnapshot(**values)


class RuntimeStatusSnapshotTests(unittest.TestCase):
    def test_maps_every_status_field_to_the_protocol_in_stable_order(self):
        messages = make_snapshot().messages()

        self.assertEqual(
            [message.address for message in messages],
            [
                "/backend_status",
                "/backend",
                "/is_speaking",
                "/queue_depth",
                "/latency_total",
                "/latency_asr",
                "/retry_in",
                "/dropped_jobs",
                "/audio_status",
                "/audio_source",
                "/audio_reconnects",
                "/audio_error",
                "/audio_device_index",
                "/audio_device_name",
                "/gender",
                "/age",
                "/visual_mode",
                "/prompt_style",
                "/language",
                "/prompt_budget_mode",
                "/dropped_final_oldest",
                "/dropped_final_newest",
                "/dropped_expired_results",
            ],
        )
        self.assertEqual(messages[2], OscMessage("/is_speaking", 1))
        self.assertEqual(messages[4], OscMessage("/latency_total", 0.5))
        self.assertEqual(messages[-1], OscMessage("/dropped_expired_results", 4))


class OscOutputPublisherTests(unittest.TestCase):
    def test_publishes_status_as_one_stable_protocol_batch(self):
        client = Mock()
        publisher = OscOutputPublisher(
            "127.0.0.1",
            7000,
            client_factory=Mock(return_value=client),
        )

        delivered = publisher.publish_status(make_snapshot(), force=True)

        self.assertTrue(delivered)
        self.assertEqual(
            [call.args for call in client.send_message.call_args_list],
            [
                (message.address, message.value)
                for message in make_snapshot().messages()
            ],
        )

    def test_throttles_periodic_status_but_force_bypasses_the_interval(self):
        client = Mock()
        clock = Mock(side_effect=[0.0, 0.1, 0.2])
        publisher = OscOutputPublisher(
            "127.0.0.1",
            7000,
            status_interval=0.5,
            client_factory=Mock(return_value=client),
            clock=clock,
        )

        self.assertTrue(publisher.publish_status(make_snapshot()))
        self.assertFalse(publisher.publish_status(make_snapshot()))
        self.assertTrue(
            publisher.publish_status(make_snapshot(), force=True)
        )
        self.assertEqual(client.send_message.call_count, 46)

    def test_delivery_errors_are_isolated_rate_limited_and_recoverable(self):
        client = Mock()
        client.send_message.side_effect = [
            OSError("network down"),
            OSError("still down"),
            None,
        ]
        logger = Mock()
        clock = Mock(side_effect=[0.0, 1.0])
        publisher = OscOutputPublisher(
            "127.0.0.1",
            7000,
            error_log_interval=5.0,
            logger=logger,
            client_factory=Mock(return_value=client),
            clock=clock,
        )

        self.assertFalse(publisher.send("/prompt", "private prompt"))
        self.assertFalse(publisher.send("/prompt", "private prompt"))
        self.assertTrue(publisher.send("/backend", "whisper"))

        logger.warning.assert_called_once()
        warning = logger.warning.call_args.kwargs["extra"]
        self.assertEqual(warning["event"], "osc_output_error")
        self.assertNotIn("private prompt", str(warning))
        logger.info.assert_called_once()
        self.assertEqual(
            logger.info.call_args.kwargs["extra"]["event"],
            "osc_output_recovered",
        )

    def test_close_releases_the_udp_socket_once(self):
        client = Mock(spec=["send_message", "_sock"])
        client._sock = Mock()
        publisher = OscOutputPublisher(
            "127.0.0.1",
            7000,
            client_factory=Mock(return_value=client),
        )

        publisher.close()
        publisher.close()

        client._sock.close.assert_called_once_with()
        self.assertFalse(publisher.send("/prompt", "ignored"))


class RecordingOutputPublisherTests(unittest.TestCase):
    def test_records_messages_and_status_without_a_network_socket(self):
        publisher = RecordingOutputPublisher()

        publisher.send("/prompt", "a forest")
        publisher.publish_status(make_snapshot(), force=True)

        self.assertEqual(publisher.messages[0], OscMessage("/prompt", "a forest"))
        self.assertEqual(
            publisher.messages[1:],
            make_snapshot().messages(),
        )

    def test_null_publisher_accepts_the_protocol_without_recording(self):
        publisher = NullOutputPublisher()

        self.assertTrue(publisher.send("/prompt", "unused"))
        self.assertTrue(publisher.publish_status(make_snapshot()))
        self.assertIsNone(publisher.close())


if __name__ == "__main__":
    unittest.main()
