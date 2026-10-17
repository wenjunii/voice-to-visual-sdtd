import threading
import unittest
from contextlib import redirect_stdout
from io import StringIO
from unittest.mock import Mock, patch

from osc_control import OscControlServer
from tests.test_transcriber_controls import make_pipeline


class OscStartupLifecycleTests(unittest.TestCase):
    def make_pipeline(self):
        pipeline = make_pipeline(output_publisher=Mock())
        pipeline.osc_control_enabled = True
        pipeline._owns_log_session = True
        pipeline.log_session.close = Mock()
        self.addCleanup(pipeline.close)
        return pipeline

    @staticmethod
    def worker(target, errors):
        def run():
            try:
                target()
            except BaseException as exc:
                errors.append(exc)

        return threading.Thread(target=run)

    def test_close_waits_for_control_construction_and_start_registration(self):
        for phase in ("construct", "start"):
            with self.subTest(phase=phase):
                pipeline = self.make_pipeline()
                entered = threading.Event()
                release = threading.Event()
                errors = []
                close_order = []
                server = Mock()
                server.start.return_value = ("127.0.0.1", 7001)

                def pause_startup():
                    entered.set()
                    if not release.wait(3.0):
                        raise AssertionError("OSC startup was not released")

                def construct(*_args):
                    if phase == "construct":
                        pause_startup()
                    return server

                def start():
                    pause_startup()
                    return ("127.0.0.1", 7001)

                if phase == "start":
                    server.start.side_effect = start
                server.stop.side_effect = lambda: close_order.append("controls")
                pipeline.output_publisher.close.side_effect = lambda: close_order.append("output")
                pipeline.backend_adapter.close.side_effect = lambda: close_order.append("backend")
                pipeline.log_session.close.side_effect = lambda: close_order.append("logs")
                starter = self.worker(pipeline.start_osc_control_server, errors)
                closer = self.worker(pipeline.close, errors)

                with patch("transcriber.OscControlServer", side_effect=construct):
                    starter.start()
                    try:
                        self.assertTrue(entered.wait(3.0))
                        closer.start()
                        self.assertTrue(pipeline.stop_event.wait(3.0))
                        pipeline.output_publisher.close.assert_not_called()
                        pipeline.backend_adapter.close.assert_not_called()
                        pipeline.log_session.close.assert_not_called()
                    finally:
                        release.set()
                        starter.join(3.0)
                        if closer.ident is not None:
                            closer.join(3.0)

                self.assertFalse(starter.is_alive())
                self.assertFalse(closer.is_alive())
                self.assertEqual(errors, [])
                self.assertEqual(close_order, ["controls", "output", "backend", "logs"])
                self.assertIsNone(pipeline.osc_control_server)
                self.assertEqual(pipeline.exit_code, 0)
                server.start.assert_called_once_with()
                server.stop.assert_called_once_with()

    def test_cancelled_or_closed_pipeline_cannot_open_a_control_listener(self):
        for close_first in (False, True):
            with self.subTest(close_first=close_first):
                pipeline = self.make_pipeline()
                if close_first:
                    pipeline.close()
                else:
                    pipeline.request_shutdown()
                with patch("transcriber.OscControlServer") as server_type:
                    with self.assertRaisesRegex(RuntimeError, "after shutdown"):
                        pipeline.start_osc_control_server()
                server_type.assert_not_called()
                self.assertIsNone(pipeline.osc_control_server)

    def test_disabled_controls_remain_a_no_op_after_close(self):
        pipeline = self.make_pipeline()
        pipeline.osc_control_enabled = False
        pipeline.close()
        with patch("transcriber.OscControlServer") as server_type:
            pipeline.start_osc_control_server()
        server_type.assert_not_called()

    def test_duplicate_start_preserves_the_owned_listener(self):
        pipeline = self.make_pipeline()
        server = Mock()
        server.start.return_value = ("127.0.0.1", 7001)
        with patch("transcriber.OscControlServer", return_value=server) as server_type:
            pipeline.start_osc_control_server()
            with self.assertRaisesRegex(RuntimeError, "already started"):
                pipeline.start_osc_control_server()

        self.assertIs(pipeline.osc_control_server, server)
        server_type.assert_called_once()
        server.start.assert_called_once_with()
        server.stop.assert_not_called()
        pipeline.close()
        pipeline.close()
        server.stop.assert_called_once_with()

    def test_bind_failure_can_retry_without_losing_listener_ownership(self):
        pipeline = self.make_pipeline()
        failed = Mock()
        failed.start.side_effect = OSError("address already in use")
        working = Mock()
        working.start.return_value = ("127.0.0.1", 7001)
        with patch("transcriber.OscControlServer", side_effect=[failed, working]):
            pipeline.start_osc_control_server()
            self.assertIsNone(pipeline.osc_control_server)
            self.assertTrue(pipeline.is_running)
            self.assertEqual(
                pipeline.osc_logger.error.call_args.kwargs["extra"]["event"],
                "osc_control_bind_error",
            )
            pipeline.start_osc_control_server()

        self.assertIs(pipeline.osc_control_server, working)
        pipeline.close()
        failed.stop.assert_not_called()
        working.stop.assert_called_once_with()

    def test_start_failure_releases_registration_lock_for_concurrent_close(self):
        for failure in (OSError("bind failed"), RuntimeError("start failed"), KeyboardInterrupt()):
            with self.subTest(failure=type(failure).__name__):
                pipeline = self.make_pipeline()
                entered = threading.Event()
                release = threading.Event()
                errors = []
                server = Mock()

                def start():
                    entered.set()
                    if not release.wait(3.0):
                        raise AssertionError("Failing OSC startup was not released")
                    raise failure

                server.start.side_effect = start
                starter = self.worker(pipeline.start_osc_control_server, errors)
                closer = self.worker(pipeline.close, errors)
                with patch("transcriber.OscControlServer", return_value=server):
                    starter.start()
                    try:
                        self.assertTrue(entered.wait(3.0))
                        closer.start()
                        self.assertTrue(pipeline.stop_event.wait(3.0))
                        pipeline.backend_adapter.close.assert_not_called()
                    finally:
                        release.set()
                        starter.join(3.0)
                        if closer.ident is not None:
                            closer.join(3.0)

                self.assertFalse(starter.is_alive())
                self.assertFalse(closer.is_alive())
                self.assertEqual(errors, [] if isinstance(failure, OSError) else [failure])
                self.assertIsNone(pipeline.osc_control_server)
                pipeline.output_publisher.close.assert_called_once_with()
                pipeline.backend_adapter.close.assert_called_once_with()
                pipeline.log_session.close.assert_called_once_with()

    def test_shutdown_during_full_start_releases_the_real_listener_and_port(self):
        pipeline = self.make_pipeline()
        pipeline.start_worker_threads = Mock()
        control = OscControlServer("127.0.0.1", 0, pipeline.apply_control)
        self.addCleanup(control.stop)
        real_start = control.start
        entered = threading.Event()
        release = threading.Event()
        errors = []
        bound = []

        def start():
            address = real_start()
            bound.append((address, control.server, control.thread))
            entered.set()
            if not release.wait(3.0):
                raise AssertionError("Bound OSC listener was not released")
            return address

        starter = self.worker(pipeline.start, errors)
        closer = self.worker(pipeline.close, errors)
        with (
            patch("transcriber.OscControlServer", return_value=control),
            patch.object(control, "start", side_effect=start),
            redirect_stdout(StringIO()),
        ):
            starter.start()
            try:
                self.assertTrue(entered.wait(3.0))
                closer.start()
                self.assertTrue(pipeline.stop_event.wait(3.0))
                pipeline.backend_adapter.close.assert_not_called()
            finally:
                release.set()
                starter.join(3.0)
                if closer.ident is not None:
                    closer.join(3.0)

        self.assertFalse(starter.is_alive())
        self.assertFalse(closer.is_alive())
        self.assertEqual(errors, [])
        self.assertEqual(pipeline.exit_code, 0)
        pipeline.start_worker_threads.assert_not_called()
        self.assertIsNone(pipeline.osc_control_server)
        self.assertIsNone(control.server)
        self.assertIsNone(control.thread)
        address, receiver, listener = bound[0]
        self.assertFalse(listener.is_alive())
        self.assertEqual(receiver.socket.fileno(), -1)
        pipeline.output_publisher.close.assert_called_once_with()
        pipeline.backend_adapter.close.assert_called_once_with()
        pipeline.log_session.close.assert_called_once_with()

        replacement = OscControlServer(*address, Mock())
        self.addCleanup(replacement.stop)
        self.assertEqual(replacement.start(), address)


if __name__ == "__main__":
    unittest.main()
