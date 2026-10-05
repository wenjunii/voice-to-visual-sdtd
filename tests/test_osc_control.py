import threading
import unittest
from unittest.mock import Mock, patch

from pythonosc import udp_client
from pythonosc.osc_server import ThreadingOSCUDPServer

from osc_control import OscControlServer, normalize_control_value


class OscControlTests(unittest.TestCase):
    def test_normalizes_keyboard_aliases_and_auto_language(self):
        self.assertEqual(normalize_control_value("visual_mode", "x"), "asian_black_brown")
        self.assertEqual(normalize_control_value("prompt_style", "general scene"), "general_scene")
        self.assertIsNone(normalize_control_value("language", "auto"))

    def test_failed_thread_start_releases_the_port_and_allows_restart(self):
        thread_type = threading.Thread
        real_start = thread_type.start
        for phase in ("create", "start", "interrupt_after_start"):
            with self.subTest(phase=phase):
                server = OscControlServer("127.0.0.1", 0, Mock())
                bound_servers = []
                workers = []
                failure = KeyboardInterrupt() if phase == "interrupt_after_start" else RuntimeError("cannot start control thread")

                def bind(*args, **kwargs):
                    bound = ThreadingOSCUDPServer(*args, **kwargs)
                    bound_servers.append(bound)
                    return bound

                def create_thread(**kwargs):
                    if phase == "create":
                        raise failure
                    worker = thread_type(**kwargs)
                    workers.append(worker)
                    return worker

                def start(worker):
                    if phase == "interrupt_after_start":
                        real_start(worker)
                    raise failure

                try:
                    with (
                        patch("osc_control.ThreadingOSCUDPServer", side_effect=bind),
                        patch("osc_control.threading.Thread", side_effect=create_thread),
                        patch.object(thread_type, "start", start),
                    ):
                        with self.assertRaises(type(failure)) as raised:
                            server.start()

                    self.assertIs(raised.exception, failure)
                    self.assertIsNone(server.server)
                    self.assertIsNone(server.thread)
                    self.assertTrue(all(not worker.is_alive() for worker in workers))
                    self.assertEqual(bound_servers[0].socket.fileno(), -1)
                    server.stop()
                finally:
                    for worker in workers:
                        if worker.is_alive():
                            bound_servers[0].shutdown()
                            worker.join(2.0)
                    for bound in bound_servers:
                        bound.server_close()

                server.port = bound_servers[0].server_address[1]
                try:
                    self.assertEqual(server.start()[1], server.port)
                finally:
                    server.stop()

    def test_receives_a_loopback_control_message(self):
        received = []
        ready = threading.Event()

        def on_control(name, value):
            received.append((name, value))
            ready.set()

        server = OscControlServer("127.0.0.1", 0, on_control)
        address = server.start()
        self.assertEqual(server.thread.name, "voice-to-visual-osc-control")
        self.assertFalse(server.thread.daemon)
        client = None
        try:
            with self.assertRaisesRegex(RuntimeError, "already started"):
                server.start()
            client = udp_client.SimpleUDPClient(*address)
            client.send_message("/control/language", "chinese")
            self.assertTrue(ready.wait(timeout=2.0))
        finally:
            if client is not None:
                client._sock.close()
            server.stop()

        self.assertEqual(received, [("language", "zh")])


if __name__ == "__main__":
    unittest.main()
