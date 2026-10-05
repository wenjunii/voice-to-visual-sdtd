import threading
import unittest
from contextlib import redirect_stdout
from io import StringIO
from unittest.mock import Mock, patch

from tests.test_transcriber_controls import make_pipeline
from transcriber import main


class StartupCleanupTests(unittest.TestCase):
    def make_pipeline(self):
        publisher = Mock()
        pipeline = make_pipeline(output_publisher=publisher)
        pipeline._owns_log_session = True
        pipeline.log_session.close = Mock()
        self.addCleanup(pipeline.close)
        return pipeline

    def test_worker_creation_and_start_failures_preserve_the_error_and_close_resources(self):
        thread_type = threading.Thread
        real_start = thread_type.start
        for phase in ("create", "start"):
            for failed_worker in ("audio", "transcription"):
                for entrypoint in ("start", "main"):
                    with self.subTest(phase=phase, worker=failed_worker, entrypoint=entrypoint):
                        pipeline = self.make_pipeline()
                        failure = RuntimeError("simulated worker startup failure")
                        workers = []
                        close_order = []

                        def work():
                            if not pipeline.stop_event.wait(2.0):
                                raise AssertionError("Startup failure did not cancel workers")
                            close_order.append("worker")

                        pipeline.audio_callback = work
                        pipeline.transcription_loop = work
                        control_server = Mock()
                        control_server.stop.side_effect = lambda: close_order.append("controls")

                        def start_controls():
                            pipeline.osc_control_server = control_server

                        pipeline.start_osc_control_server = start_controls
                        pipeline.output_publisher.close.side_effect = lambda: close_order.append("output")
                        pipeline.backend_adapter.close.side_effect = lambda: close_order.append("backend")
                        pipeline.log_session.close.side_effect = lambda: close_order.append("logs")

                        def create_thread(**kwargs):
                            if phase == "create" and kwargs["name"] == f"voice-to-visual-{failed_worker}":
                                raise failure
                            worker = thread_type(**kwargs)
                            workers.append(worker)
                            return worker

                        def start_thread(worker):
                            if phase == "start" and worker.name == f"voice-to-visual-{failed_worker}":
                                raise failure
                            return real_start(worker)

                        with (
                            patch("transcriber.threading.Thread", side_effect=create_thread),
                            patch.object(thread_type, "start", start_thread),
                            patch("transcriber.load_runtime_config", return_value=pipeline.config),
                            patch("transcriber.RealTimePipeline", return_value=pipeline),
                            redirect_stdout(StringIO()),
                        ):
                            with self.assertRaises(RuntimeError) as raised:
                                pipeline.start() if entrypoint == "start" else main([])

                        self.assertIs(raised.exception, failure)
                        self.assertEqual(pipeline.exit_code, 1)
                        self.assertTrue(pipeline.stop_event.is_set())
                        self.assertEqual(pipeline._worker_threads, {})
                        self.assertTrue(all(not worker.is_alive() for worker in workers))
                        self.assertEqual(close_order[-3:], ["output", "backend", "logs"])
                        self.assertEqual(close_order.count("worker"), int(failed_worker == "transcription"))
                        control_server.stop.assert_called_once_with()
                        pipeline.output_publisher.close.assert_called_once_with()
                        pipeline.backend_adapter.close.assert_called_once_with()
                        pipeline.log_session.close.assert_called_once_with()
                        event = pipeline.runtime_logger.exception.call_args.kwargs["extra"]
                        self.assertEqual(event["event"], "worker_start_error")
                        self.assertEqual(event["worker"], failed_worker)

    def test_keyboard_interrupt_during_startup_joins_started_workers(self):
        real_start = threading.Thread.start
        for after_start in (False, True):
            with self.subTest(after_start=after_start):
                pipeline = self.make_pipeline()
                pipeline.audio_callback = lambda: pipeline.stop_event.wait(2.0)
                pipeline.transcription_loop = lambda: pipeline.stop_event.wait(2.0)
                started = []

                def start(worker):
                    if worker.name == "voice-to-visual-transcription" and not after_start:
                        raise KeyboardInterrupt()
                    real_start(worker)
                    started.append(worker)
                    if worker.name == "voice-to-visual-transcription":
                        raise KeyboardInterrupt()

                with patch("transcriber.threading.Thread.start", start), redirect_stdout(StringIO()):
                    pipeline.start()

                self.assertEqual(pipeline.exit_code, 0)
                self.assertEqual(len(started), 2 if after_start else 1)
                self.assertTrue(all(not worker.is_alive() for worker in started))
                pipeline.backend_adapter.close.assert_called_once_with()
                pipeline.log_session.close.assert_called_once_with()

    def test_control_start_failure_closes_pipeline_resources(self):
        pipeline = self.make_pipeline()
        failure = RuntimeError("control thread could not start")
        pipeline.start_osc_control_server = Mock(side_effect=failure)
        pipeline.start_worker_threads = Mock()

        with self.assertRaises(RuntimeError) as raised:
            pipeline.start()

        self.assertIs(raised.exception, failure)
        self.assertEqual(pipeline.exit_code, 1)
        pipeline.start_worker_threads.assert_not_called()
        pipeline.output_publisher.close.assert_called_once_with()
        pipeline.backend_adapter.close.assert_called_once_with()
        pipeline.log_session.close.assert_called_once_with()

    def test_concurrent_close_waits_for_worker_start_registration(self):
        pipeline = self.make_pipeline()
        real_start = threading.Thread.start
        starting = threading.Event()
        release_start = threading.Event()
        worker_stopped = threading.Event()
        errors = []
        started_workers = []
        closed_after_worker = []

        def work():
            pipeline.stop_event.wait(2.0)
            worker_stopped.set()

        def start(worker):
            real_start(worker)
            if worker.name == "voice-to-visual-audio":
                started_workers.append(worker)
                starting.set()
                if not release_start.wait(2.0):
                    raise AssertionError("Worker start was never released")

        def run(target):
            try:
                target()
            except Exception as exc:
                errors.append(exc)

        pipeline.audio_callback = work
        pipeline.transcription_loop = Mock()
        pipeline.backend_adapter.close.side_effect = lambda: closed_after_worker.append(worker_stopped.is_set())
        starter = threading.Thread(target=run, args=(pipeline.start_worker_threads,))
        closer = threading.Thread(target=run, args=(pipeline.close,))
        with patch("transcriber.threading.Thread.start", start):
            starter.start()
            try:
                self.assertTrue(starting.wait(2.0))
                closer.start()
                self.assertTrue(pipeline.stop_event.wait(2.0))
                pipeline.backend_adapter.close.assert_not_called()
            finally:
                release_start.set()
                pipeline.request_shutdown()
                starter.join(2.0)
                if closer.ident is not None:
                    closer.join(2.0)

        self.assertFalse(starter.is_alive())
        self.assertFalse(closer.is_alive())
        self.assertEqual(errors, [])
        self.assertTrue(all(not worker.is_alive() for worker in started_workers))
        self.assertEqual(closed_after_worker, [True])
        pipeline.transcription_loop.assert_not_called()

    def test_closed_pipeline_cannot_open_new_control_resources(self):
        pipeline = self.make_pipeline()
        pipeline.close()
        pipeline.start_osc_control_server = Mock()

        with self.assertRaisesRegex(RuntimeError, "already started or stopped"):
            pipeline.start()

        pipeline.start_osc_control_server.assert_not_called()
        pipeline.backend_adapter.close.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
