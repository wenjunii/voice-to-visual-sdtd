import sys
import tempfile
import threading
import types
import unittest
import wave
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import replace
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from audio_sources import WavReplaySource
from backend_errors import RetryableTranscriptionError
from osc_output import RecordingOutputPublisher
from runtime_config import RuntimeConfig
from transcriber import RealTimePipeline, load_runtime_config, main, parse_args


def make_backend_adapter(*, online=False):
    adapter = Mock()
    adapter.name = "google" if online else "whisper"
    adapter.online = online
    adapter.request_interval = 2.5
    adapter.minimum_audio_seconds = 1.25
    adapter.maximum_audio_seconds = 5.0
    adapter.transcribe.return_value = "adapter transcript"
    return adapter


class TestLogSession:
    path = None

    def __init__(self):
        self.loggers = {}

    def logger(self, subsystem):
        return self.loggers.setdefault(subsystem, Mock())

    def close(self):
        return None


def make_pipeline(
    config=None,
    *,
    online=False,
    backend_adapter=None,
    audio_source=None,
    output_publisher=None,
):
    return RealTimePipeline(
        enable_vad=False,
        enable_osc=False,
        enable_prompt_budget=False,
        enable_osc_controls=False,
        config=config if config is not None else RuntimeConfig(),
        backend_adapter=(
            backend_adapter
            if backend_adapter is not None
            else make_backend_adapter(online=online)
        ),
        log_session=TestLogSession(),
        audio_source=audio_source,
        output_publisher=output_publisher,
    )


class CommandLineTests(unittest.TestCase):
    def test_accepts_the_config_check_flag(self):
        args = parse_args(["--check-config"])

        self.assertTrue(args.check_config)

    def test_accepts_an_audio_input_device_override(self):
        args = parse_args(["--input-device", "4"])

        self.assertEqual(args.input_device, 4)

    def test_accepts_startup_control_overrides(self):
        args = parse_args(
            [
                "--gender",
                "woman",
                "--age",
                "elder",
                "--visual-mode",
                "black_brown",
                "--prompt-style",
                "general_scene",
                "--language",
                "zh",
            ]
        )

        self.assertEqual(args.gender, "woman")
        self.assertEqual(args.age, "elder")
        self.assertEqual(args.visual_mode, "black_brown")
        self.assertEqual(args.prompt_style, "general_scene")
        self.assertEqual(args.language, "zh")

    def test_passes_startup_control_overrides_to_runtime_configuration(self):
        args = parse_args(
            [
                "--gender",
                "woman",
                "--age",
                "elder",
                "--visual-mode",
                "black_brown",
                "--prompt-style",
                "general_scene",
                "--language",
                "es",
            ]
        )
        with (
            patch("transcriber.load_env_file"),
            patch(
                "transcriber.RuntimeConfig.from_environment",
                return_value=RuntimeConfig(),
            ) as config_loader,
        ):
            load_runtime_config(args)

        config_loader.assert_called_once_with(
            backend_override=None,
            input_device_override=None,
            gender_override="woman",
            age_override="elder",
            visual_mode_override="black_brown",
            prompt_style_override="general_scene",
            language_override="es",
        )

    def test_rejects_a_negative_audio_input_device(self):
        with redirect_stderr(StringIO()):
            with self.assertRaises(SystemExit):
                parse_args(["--input-device", "-1"])

    def test_rejects_an_invalid_startup_control_override(self):
        with redirect_stderr(StringIO()):
            with self.assertRaises(SystemExit):
                parse_args(["--visual-mode", "unknown"])

    def test_accepts_a_wav_replay_path(self):
        args = parse_args(["--replay", "session.wav"])

        self.assertEqual(args.replay, "session.wav")

    def test_replay_cannot_be_combined_with_benchmark_or_input_device(self):
        for arguments in (
            ["--replay", "session.wav", "--benchmark", "session.wav"],
            ["--replay", "session.wav", "--input-device", "2"],
        ):
            with self.subTest(arguments=arguments):
                with redirect_stderr(StringIO()):
                    with self.assertRaises(SystemExit):
                        parse_args(arguments)

    def test_replay_builds_a_finite_source_for_the_live_pipeline(self):
        source = Mock()
        pipeline = Mock()
        with (
            patch("transcriber.load_runtime_config", return_value=RuntimeConfig()),
            patch("transcriber.WavReplaySource", return_value=source) as source_type,
            patch("transcriber.RealTimePipeline", return_value=pipeline) as pipeline_type,
        ):
            result = main(["--replay", "session.wav"])

        self.assertEqual(result, 0)
        source_type.assert_called_once_with(
            "session.wav",
            sample_rate=16000,
            chunk_samples=1024,
            realtime=True,
        )
        self.assertIs(pipeline_type.call_args.kwargs["audio_source"], source)
        self.assertTrue(pipeline_type.call_args.kwargs["enable_vad"])
        self.assertTrue(pipeline_type.call_args.kwargs["enable_osc"])
        pipeline.start.assert_called_once_with()
        pipeline.close.assert_called_once_with()


class RealTimePipelineControlTests(unittest.TestCase):
    def make_pipeline(self):
        publisher = RecordingOutputPublisher()
        pipeline = make_pipeline(output_publisher=publisher)
        pipeline.recording_publisher = publisher
        pipeline.last_text = "a city glowing after rain"
        pipeline.last_prompt_token_count = 42
        pipeline.audio_status = "ready"
        pipeline.audio_device_index = 2
        pipeline.audio_device_name = "USB Microphone"
        pipeline.last_total_latency = 0.25
        pipeline.last_inference_latency = 0.2
        return pipeline

    def test_visual_control_refreshes_the_current_prompt_immediately(self):
        pipeline = self.make_pipeline()
        pipeline.prompt_budgeter = None
        pipeline.build_visual_prompt = Mock(wraps=pipeline.build_visual_prompt)
        pipeline.send_runtime_status = Mock()

        with redirect_stdout(StringIO()):
            pipeline.apply_control("visual_mode", "black_brown")

        pipeline.build_visual_prompt.assert_called_once_with(
            "a city glowing after rain"
        )
        messages = [
            (message.address, message.value)
            for message in pipeline.recording_publisher.messages
        ]
        self.assertIn(
            ("/control_ack", "visual_mode:black_brown"),
            messages,
        )
        prompt_messages = [
            value
            for address, value in messages
            if address == "/prompt"
        ]
        self.assertEqual(len(prompt_messages), 1)
        self.assertIn("adult Black or Brown person", prompt_messages[0])
        self.assertIn(("/prompt_tokens", 0), messages)
        self.assertIn(("/prompt_budget_mode", "disabled"), messages)
        pipeline.send_runtime_status.assert_called_once_with(force=True)

    def test_language_control_does_not_rebuild_the_visual_prompt(self):
        pipeline = self.make_pipeline()
        pipeline.build_visual_prompt = Mock(return_value="unused")
        pipeline.send_runtime_status = Mock()

        with redirect_stdout(StringIO()):
            pipeline.apply_control("language", "es")

        pipeline.build_visual_prompt.assert_not_called()
        self.assertEqual(pipeline.current_language, "es")

    def test_runtime_status_includes_the_active_control_snapshot(self):
        pipeline = self.make_pipeline()
        pipeline.current_gender = "woman"
        pipeline.current_age = "elder"
        pipeline.current_visual_mode = "asian_black_brown"
        pipeline.current_prompt_style = "general_scene"
        pipeline.current_language = "zh"

        pipeline.send_runtime_status(force=True)

        expected = {
            ("/audio_status", "ready"),
            ("/audio_source", "microphone"),
            ("/audio_reconnects", 0),
            ("/audio_error", ""),
            ("/audio_device_index", 2),
            ("/audio_device_name", "USB Microphone"),
            ("/gender", "woman"),
            ("/age", "elder"),
            ("/visual_mode", "asian_black_brown"),
            ("/prompt_style", "general_scene"),
            ("/language", "zh"),
            ("/prompt_budget_mode", "disabled"),
            ("/dropped_final_oldest", 0),
            ("/dropped_final_newest", 0),
            ("/dropped_expired_results", 0),
        }
        calls = {
            (message.address, message.value)
            for message in pipeline.recording_publisher.messages
        }
        self.assertTrue(expected.issubset(calls))


class SchedulerOverflowLoggingTests(unittest.TestCase):
    @staticmethod
    def segment(segment_id):
        return SimpleNamespace(
            segment_id=segment_id,
            version=1,
            is_final=True,
        )

    def test_logs_oldest_drop_and_cleans_replaced_segment_state(self):
        publisher = RecordingOutputPublisher()
        pipeline = make_pipeline(
            replace(
                RuntimeConfig(),
                transcription_max_final_jobs=1,
                transcription_final_overflow_policy="drop_oldest",
            ),
            output_publisher=publisher,
        )
        pipeline.stabilizers[1] = object()
        pipeline.submit_final_segment(
            self.segment(1),
            1.0,
            source="completed",
        )

        submission = pipeline.submit_final_segment(
            self.segment(2),
            1.1,
            source="completed",
        )

        self.assertTrue(submission.accepted)
        self.assertNotIn(1, pipeline.stabilizers)
        event = pipeline.scheduler_logger.warning.call_args.kwargs["extra"]
        self.assertEqual(event["event"], "scheduler_final_oldest_dropped")
        self.assertEqual(event["segment_id"], 1)
        self.assertEqual(event["incoming_segment_id"], 2)
        with patch("transcriber.time.monotonic", return_value=1.2):
            pipeline.send_runtime_status(force=True)
        status = {
            (message.address, message.value)
            for message in publisher.messages
        }
        self.assertIn(("/dropped_jobs", 1), status)
        self.assertIn(("/dropped_final_oldest", 1), status)
        self.assertIn(("/dropped_final_newest", 0), status)
        pipeline.close()

    def test_logs_newest_drop_when_fifo_completeness_is_selected(self):
        pipeline = make_pipeline(
            replace(
                RuntimeConfig(),
                transcription_max_final_jobs=1,
                transcription_final_overflow_policy="drop_newest",
            )
        )
        pipeline.submit_final_segment(
            self.segment(1),
            1.0,
            source="completed",
        )

        submission = pipeline.submit_final_segment(
            self.segment(2),
            1.1,
            source="interrupted",
        )

        self.assertFalse(submission.accepted)
        event = pipeline.scheduler_logger.warning.call_args.kwargs["extra"]
        self.assertEqual(event["event"], "scheduler_final_newest_dropped")
        self.assertEqual(event["segment_id"], 2)
        self.assertEqual(event["source"], "interrupted")
        pipeline.close()


class BackendAdapterIntegrationTests(unittest.TestCase):
    def make_pipeline(self, *, online=True):
        adapter = make_backend_adapter(online=online)
        pipeline = make_pipeline(
            online=online,
            backend_adapter=adapter,
        )
        return pipeline, adapter

    def test_delegates_transcription_and_language_to_the_adapter(self):
        pipeline, adapter = self.make_pipeline()
        pipeline.current_language = "es"
        samples = object()

        text = pipeline.transcribe_audio(samples)

        self.assertEqual(text, "adapter transcript")
        adapter.transcribe.assert_called_once_with(
            samples,
            language="es",
        )

    def test_uses_adapter_audio_and_timing_limits(self):
        pipeline, _adapter = self.make_pipeline()

        self.assertEqual(pipeline.minimum_audio_seconds(), 1.25)
        self.assertEqual(pipeline.maximum_audio_seconds(), 5.0)
        self.assertEqual(pipeline.online_request_interval(), 2.5)
        self.assertEqual(pipeline.local_request_interval(), 0.0)

    def test_closes_the_backend_adapter(self):
        publisher = Mock()
        pipeline = make_pipeline(
            backend_adapter=make_backend_adapter(online=True),
            output_publisher=publisher,
        )
        adapter = pipeline.backend_adapter

        pipeline.close()
        pipeline.close()

        publisher.close.assert_called_once_with()
        adapter.close.assert_called_once_with()


class PipelineLifecycleTests(unittest.TestCase):
    def test_shutdown_interrupts_an_audio_retry_wait(self):
        pipeline = make_pipeline()
        result = []
        waiter = threading.Thread(
            target=lambda: result.append(
                pipeline.wait_for_audio_retry(10.0)
            )
        )
        waiter.start()

        pipeline.request_shutdown("test")
        waiter.join(timeout=0.5)

        self.assertFalse(waiter.is_alive())
        self.assertEqual(result, [False])
        pipeline.close()

    def test_close_waits_for_workers_before_closing_resources(self):
        pipeline = make_pipeline(
            replace(
                RuntimeConfig(),
                runtime_shutdown_grace_seconds=0.02,
            )
        )
        worker_started = threading.Event()
        release_worker = threading.Event()
        worker_stopped = threading.Event()
        backend_closed_after_worker = []

        def blocked_audio_worker():
            worker_started.set()
            release_worker.wait()
            worker_stopped.set()

        pipeline.audio_callback = blocked_audio_worker
        pipeline.transcription_loop = Mock()
        pipeline.backend_adapter.close.side_effect = lambda: (
            backend_closed_after_worker.append(worker_stopped.is_set())
        )
        workers = pipeline.start_worker_threads()
        self.assertTrue(worker_started.wait(timeout=0.5))
        self.assertTrue(all(not worker.daemon for worker in workers))

        release_timer = threading.Timer(0.08, release_worker.set)
        release_timer.start()
        try:
            pipeline.close()
        finally:
            release_worker.set()
            release_timer.join(timeout=0.5)

        self.assertTrue(worker_stopped.is_set())
        self.assertEqual(backend_closed_after_worker, [True])
        warning = pipeline.runtime_logger.warning.call_args
        self.assertEqual(
            warning.kwargs["extra"]["event"],
            "worker_shutdown_overdue",
        )
        self.assertEqual(
            warning.kwargs["extra"]["workers"],
            ["voice-to-visual-audio"],
        )

    def test_worker_crash_requests_pipeline_shutdown(self):
        pipeline = make_pipeline()

        def crashing_audio_worker():
            raise RuntimeError("audio worker failed")

        pipeline.audio_callback = crashing_audio_worker
        pipeline.transcription_loop = lambda: pipeline.stop_event.wait(0.5)
        pipeline.start_worker_threads()
        pipeline.join_worker_threads()

        self.assertTrue(pipeline.stop_event.is_set())
        crash = pipeline.runtime_logger.exception.call_args
        self.assertEqual(crash.kwargs["extra"]["event"], "worker_crashed")
        self.assertEqual(crash.kwargs["extra"]["worker"], "audio")
        pipeline.close()


class WavReplayPipelineTests(unittest.TestCase):
    def test_finite_source_drains_the_final_segment_before_shutdown(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fixture.wav"
            with wave.open(str(path), "wb") as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(16000)
                wav_file.writeframes(b"\x10\x27" * 3200)
            source = WavReplaySource(path, realtime=False)
            adapter = make_backend_adapter()
            publisher = RecordingOutputPublisher()
            pipeline = make_pipeline(
                backend_adapter=adapter,
                audio_source=source,
                output_publisher=publisher,
            )
            pipeline.vad = Mock()
            pipeline.vad.is_speech.return_value = True

            with redirect_stdout(StringIO()):
                pipeline.start_worker_threads()
                pipeline.join_worker_threads()

        self.assertTrue(pipeline.audio_source_finished.is_set())
        self.assertTrue(pipeline.stop_event.is_set())
        adapter.transcribe.assert_called_once()
        self.assertEqual(pipeline.last_text, "adapter transcript")
        self.assertFalse(source._open)
        prompt_start = next(
            index
            for index, message in enumerate(publisher.messages)
            if message.address == "/prompt"
        )
        transcript_end = next(
            index
            for index, message in enumerate(
                publisher.messages[prompt_start:],
                start=prompt_start,
            )
            if message.address == "/transcript_final"
        )
        transcript_messages = [
            (message.address, message.value)
            for message in publisher.messages[prompt_start:transcript_end + 1]
            if message.address
            in {
                "/prompt",
                "/partial_text",
                "/scene_context",
                "/prompt_tokens",
                "/prompt_budget_mode",
                "/transcript_final",
            }
        ]
        self.assertEqual(
            [address for address, _value in transcript_messages],
            [
                "/prompt",
                "/partial_text",
                "/scene_context",
                "/prompt_tokens",
                "/prompt_budget_mode",
                "/transcript_final",
            ],
        )
        self.assertEqual(transcript_messages[1][1], "adapter transcript")
        self.assertEqual(transcript_messages[2][1], "adapter transcript")
        self.assertEqual(transcript_messages[5][1], "adapter transcript")
        pipeline.close()


class PipelineConfigurationIsolationTests(unittest.TestCase):
    def test_each_pipeline_configures_its_own_runtime_components(self):
        first_config = replace(
            RuntimeConfig(),
            audio_input_device_index=2,
            scene_memory_max_words=12,
            scene_memory_max_age_seconds=8.0,
            transcription_max_final_jobs=3,
            transcription_final_overflow_policy="drop_newest",
            transcription_partial_max_age_seconds=1.5,
            transcription_final_max_age_seconds=9.0,
            vad_pre_roll_seconds=0.1,
            vad_silence_seconds=0.25,
            stream_overlap_seconds=0.2,
            default_gender="woman",
            default_age="elder",
            default_visual_mode="black_brown",
            default_prompt_style="general_scene",
            default_language="zh",
        )
        second_config = replace(
            RuntimeConfig(),
            audio_input_device_index=7,
            scene_memory_max_words=48,
            scene_memory_max_age_seconds=30.0,
            transcription_max_final_jobs=11,
            transcription_final_overflow_policy="drop_oldest",
            transcription_partial_max_age_seconds=5.0,
            transcription_final_max_age_seconds=45.0,
            vad_pre_roll_seconds=0.6,
            vad_silence_seconds=1.1,
            stream_overlap_seconds=0.8,
            default_gender="man",
            default_age="young",
            default_visual_mode="asian_black_brown",
            default_prompt_style="human_focus",
            default_language="es",
        )

        first = make_pipeline(first_config)
        second = make_pipeline(second_config)

        self.assertIs(first.config, first_config)
        self.assertIs(second.config, second_config)
        self.assertEqual(first.audio_device_index, 2)
        self.assertEqual(second.audio_device_index, 7)
        self.assertEqual(first.scheduler.max_final_jobs, 3)
        self.assertEqual(second.scheduler.max_final_jobs, 11)
        self.assertEqual(first.scheduler.final_overflow_policy, "drop_newest")
        self.assertEqual(second.scheduler.final_overflow_policy, "drop_oldest")
        self.assertEqual(first.scheduler.partial_max_age_seconds, 1.5)
        self.assertEqual(second.scheduler.final_max_age_seconds, 45.0)
        self.assertEqual(first.scene_memory.max_words, 12)
        self.assertEqual(second.scene_memory.max_words, 48)
        self.assertEqual(first.segmenter.end_silence_samples, 4000)
        self.assertEqual(second.segmenter.end_silence_samples, 17600)
        self.assertEqual(first.segmenter.overlap_chunks, 4)
        self.assertEqual(second.segmenter.overlap_chunks, 13)
        self.assertEqual(first.current_gender, "woman")
        self.assertEqual(second.current_gender, "man")
        self.assertEqual(first.current_age, "elder")
        self.assertEqual(second.current_age, "young")
        self.assertEqual(first.current_visual_mode, "black_brown")
        self.assertEqual(
            second.current_visual_mode,
            "asian_black_brown",
        )
        self.assertEqual(first.current_prompt_style, "general_scene")
        self.assertEqual(second.current_prompt_style, "human_focus")
        self.assertEqual(first.current_language, "zh")
        self.assertEqual(second.current_language, "es")
        self.assertIn(
            "no central human figure",
            first.build_visual_prompt("a city after rain"),
        )
        self.assertIn(
            "young Asian, Black, or Brown man",
            second.build_visual_prompt("a city after rain"),
        )

    def test_each_pipeline_uses_its_own_osc_configuration(self):
        first_config = replace(
            RuntimeConfig(),
            osc_ip="127.0.0.2",
            osc_port=7100,
            osc_control_enabled=True,
            osc_control_ip="127.0.0.4",
            osc_control_port=7101,
        )
        second_config = replace(
            RuntimeConfig(),
            osc_ip="127.0.0.3",
            osc_port=7200,
            osc_control_enabled=False,
        )

        first_publisher = Mock()
        second_publisher = Mock()
        with patch(
            "transcriber.OscOutputPublisher",
            side_effect=[first_publisher, second_publisher],
        ) as publisher_type:
            first = RealTimePipeline(
                enable_vad=False,
                enable_osc=True,
                enable_prompt_budget=False,
                enable_osc_controls=True,
                config=first_config,
                backend_adapter=make_backend_adapter(),
                log_session=TestLogSession(),
            )
            second = RealTimePipeline(
                enable_vad=False,
                enable_osc=True,
                enable_prompt_budget=False,
                enable_osc_controls=True,
                config=second_config,
                backend_adapter=make_backend_adapter(),
                log_session=TestLogSession(),
            )

        self.assertEqual(publisher_type.call_count, 2)
        first_call, second_call = publisher_type.call_args_list
        self.assertEqual(first_call.args, ("127.0.0.2", 7100))
        self.assertEqual(second_call.args, ("127.0.0.3", 7200))
        self.assertEqual(first_call.kwargs["status_interval"], 0.5)
        self.assertEqual(first_call.kwargs["error_log_interval"], 5.0)
        self.assertIs(first_call.kwargs["logger"], first.osc_logger)
        self.assertIs(second_call.kwargs["logger"], second.osc_logger)
        self.assertIs(first.output_publisher, first_publisher)
        self.assertIs(second.output_publisher, second_publisher)
        self.assertTrue(first.osc_control_enabled)
        self.assertFalse(second.osc_control_enabled)

        server = Mock()
        server.start.return_value = ("127.0.0.4", 7101)
        with (
            patch(
                "transcriber.OscControlServer",
                return_value=server,
            ) as server_type,
            redirect_stdout(StringIO()),
        ):
            first.start_osc_control_server()
            second.start_osc_control_server()

        server_type.assert_called_once_with(
            "127.0.0.4",
            7101,
            first.apply_control,
            first.osc_logger.warning,
        )

    def test_vad_and_prompt_budgeting_use_the_pipeline_configuration(self):
        config = replace(
            RuntimeConfig(),
            vad_engine="energy",
            vad_energy_threshold=275.0,
            prompt_tokenizer_models=("tokenizer-a", "tokenizer-b"),
            prompt_max_tokens=61,
            prompt_min_transcript_tokens=17,
        )
        auto_tokenizer = Mock()
        auto_tokenizer.from_pretrained.side_effect = [Mock(), Mock()]
        transformers = types.ModuleType("transformers")
        transformers.AutoTokenizer = auto_tokenizer

        with (
            patch.dict(sys.modules, {"transformers": transformers}),
            redirect_stdout(StringIO()),
        ):
            pipeline = RealTimePipeline(
                enable_vad=True,
                enable_osc=False,
                enable_prompt_budget=True,
                enable_osc_controls=False,
                config=config,
                backend_adapter=make_backend_adapter(),
                log_session=TestLogSession(),
            )

        self.assertEqual(pipeline.vad.threshold, 275.0)
        self.assertEqual(
            [call.args[0] for call in auto_tokenizer.from_pretrained.call_args_list],
            ["tokenizer-a", "tokenizer-b"],
        )
        self.assertEqual(pipeline.prompt_budgeter.max_tokens, 61)
        self.assertEqual(pipeline.prompt_budgeter.mode, "exact")
        self.assertEqual(pipeline.prompt_budget_mode, "exact")
        self.assertEqual(
            pipeline.prompt_budgeter.min_transcript_tokens,
            17,
        )

    def test_uses_conservative_budgeting_when_tokenizers_are_unavailable(self):
        config = replace(
            RuntimeConfig(),
            prompt_max_tokens=77,
            prompt_token_budget_fallback="conservative",
        )
        transformers = types.ModuleType("transformers")

        with patch.dict(sys.modules, {"transformers": transformers}):
            pipeline = RealTimePipeline(
                enable_vad=False,
                enable_osc=False,
                enable_prompt_budget=True,
                enable_osc_controls=False,
                config=config,
                backend_adapter=make_backend_adapter(),
                log_session=TestLogSession(),
            )

        prompt = pipeline.build_visual_prompt(
            "旧场景变成一个霓虹森林 with a very long cinematic description"
        )

        self.assertEqual(pipeline.prompt_budget_mode, "fallback")
        self.assertEqual(pipeline.prompt_budgeter.mode, "fallback")
        self.assertLessEqual(pipeline.last_prompt_token_count, 77)
        self.assertEqual(
            pipeline.last_prompt_token_count,
            len(prompt.encode("utf-8")) + 2,
        )
        self.assertEqual(pipeline.last_prompt_variant, "minimal")
        self.assertIn("a very long cinematic description", prompt)
        fallback_log = pipeline.prompt_logger.warning.call_args
        self.assertEqual(
            fallback_log.kwargs["extra"]["event"],
            "prompt_budget_fallback",
        )
        pipeline.close()

    def test_can_explicitly_disable_the_offline_budget_fallback(self):
        config = replace(
            RuntimeConfig(),
            prompt_token_budget_fallback="off",
        )
        transformers = types.ModuleType("transformers")

        with patch.dict(sys.modules, {"transformers": transformers}):
            pipeline = RealTimePipeline(
                enable_vad=False,
                enable_osc=False,
                enable_prompt_budget=True,
                enable_osc_controls=False,
                config=config,
                backend_adapter=make_backend_adapter(),
                log_session=TestLogSession(),
            )

        self.assertIsNone(pipeline.prompt_budgeter)
        self.assertEqual(pipeline.prompt_budget_mode, "unavailable")
        pipeline.close()

    def test_transcription_retries_use_the_pipeline_configuration(self):
        config = replace(
            RuntimeConfig(),
            transcription_final_max_retries=4,
            transcription_retry_base_seconds=2.25,
            transcription_retry_max_seconds=7.5,
        )
        pipeline = make_pipeline(config)
        pipeline.scheduler = Mock()
        pipeline.scheduler.retry_final.return_value = True
        pipeline.send_runtime_status = Mock()
        job = SimpleNamespace(
            is_final=True,
            attempts=1,
            segment=SimpleNamespace(segment_id=9),
        )

        with (
            patch("transcriber.time.monotonic", return_value=20.0),
            patch(
                "transcriber.exponential_backoff",
                return_value=4.5,
            ) as backoff,
            redirect_stdout(StringIO()),
        ):
            pipeline.handle_retryable_failure(
                job,
                RetryableTranscriptionError("temporary failure"),
            )

        backoff.assert_called_once_with(
            1,
            base_seconds=2.25,
            max_seconds=7.5,
        )
        pipeline.scheduler.retry_final.assert_called_once_with(
            job,
            20.0,
            4.5,
        )
        self.assertEqual(pipeline.backend_retry_not_before, 24.5)


class MicrophoneRecoveryTests(unittest.TestCase):
    @staticmethod
    def make_source():
        source = Mock()
        source.kind = "microphone"
        source.finite = False
        source.reconnectable = True
        source.device_index = 4
        source.name = "Stage USB Microphone"
        return source

    def make_pipeline(self, config=None, source=None):
        pipeline = make_pipeline(
            config,
            audio_source=source or self.make_source(),
        )
        pipeline.send_runtime_status = Mock()
        pipeline.process_audio_data = Mock()
        pipeline.finalize_interrupted_audio = Mock()
        pipeline.wait_for_audio_retry = Mock(return_value=True)
        return pipeline

    @staticmethod
    def make_stopping_read(pipeline):
        def read_once():
            pipeline.request_shutdown()
            return b"\x00\x00" * 16
        return read_once

    def test_recovers_when_the_microphone_is_unavailable_at_startup(self):
        source = self.make_source()
        pipeline = self.make_pipeline(source=source)
        source.open.side_effect = [OSError("device unavailable"), source]
        source.read.side_effect = self.make_stopping_read(pipeline)

        with redirect_stdout(StringIO()):
            pipeline.audio_callback()

        self.assertEqual(pipeline.audio_reconnects, 1)
        self.assertEqual(pipeline.audio_status, "stopped")
        self.assertEqual(pipeline.last_audio_error, "")
        pipeline.process_audio_data.assert_called_once()
        self.assertEqual(source.open.call_count, 2)
        self.assertEqual(source.close.call_count, 2)

    def test_reopens_the_stream_after_repeated_read_failures(self):
        source = self.make_source()
        pipeline = self.make_pipeline(
            replace(RuntimeConfig(), audio_max_consecutive_read_errors=2),
            source=source,
        )
        source.open.return_value = source
        read_count = 0

        def read_with_recovery():
            nonlocal read_count
            read_count += 1
            if read_count == 1:
                raise OSError("input overflow")
            if read_count == 2:
                raise OSError("device disconnected")
            pipeline.request_shutdown()
            return b"\x00\x00" * 16

        source.read.side_effect = read_with_recovery

        with redirect_stdout(StringIO()):
            pipeline.audio_callback()

        self.assertEqual(pipeline.audio_reconnects, 1)
        self.assertEqual(pipeline.audio_status, "stopped")
        pipeline.process_audio_data.assert_called_once()
        pipeline.finalize_interrupted_audio.assert_called_once()
        self.assertEqual(source.open.call_count, 2)
        self.assertEqual(source.close.call_count, 2)


if __name__ == "__main__":
    unittest.main()
