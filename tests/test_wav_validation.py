import struct
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import patch

import numpy as np

from audio_sources import load_wav_samples
from runtime_config import RuntimeConfig
from tests.test_transcriber_controls import make_pipeline
from transcriber import main


def riff_chunk(name, payload):
    padding = b"\x00" if len(payload) % 2 else b""
    return name + struct.pack("<I", len(payload)) + payload + padding


def pcm_wav(payload, *, channels=1, sample_rate=16000, width=2,
            before_data=b"", after_data=b""):
    frame_bytes = channels * width
    format_chunk = riff_chunk(
        b"fmt ",
        struct.pack(
            "<HHIIHH", 1, channels, sample_rate,
            sample_rate * frame_bytes, frame_bytes, width * 8,
        ),
    )
    chunks = format_chunk + before_data + riff_chunk(b"data", payload) + after_data
    return b"RIFF" + struct.pack("<I", 4 + len(chunks)) + b"WAVE" + chunks


class WavValidationTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.path = Path(directory.name) / "recording.wav"

    def test_rejects_every_truncated_prefix_of_a_pcm_recording(self):
        recording = pcm_wav(b"\x01\x00" * 8)
        for length in range(len(recording)):
            with self.subTest(length=length):
                self.path.write_bytes(recording[:length])
                with self.assertRaisesRegex(ValueError, "WAV"):
                    load_wav_samples(self.path)

    def test_reports_missing_whole_frames_instead_of_accepting_short_audio(self):
        for channels in (1, 2, 3):
            with self.subTest(channels=channels):
                frame_bytes = channels * 2
                recording = pcm_wav(b"\x01\x00" * channels * 8, channels=channels)
                self.path.write_bytes(recording[:-frame_bytes * 2])

                with self.assertRaisesRegex(
                    ValueError,
                    f"truncated: expected {frame_bytes * 8} PCM bytes, "
                    f"read {frame_bytes * 6}",
                ):
                    load_wav_samples(self.path)

    def test_rejects_data_chunks_with_partial_samples_or_channel_frames(self):
        for channels, payload in (
            (1, b"\x01"),
            (1, b"\x01\x00" * 8 + b"\x01"),
            (2, b"\x01\x00"),
            (2, b"\x01\x00" * 3),
            (3, b"\x01\x00" * 5),
        ):
            with self.subTest(channels=channels, payload_bytes=len(payload)):
                self.path.write_bytes(pcm_wav(payload, channels=channels))
                with self.assertRaisesRegex(ValueError, "incomplete PCM frame"):
                    load_wav_samples(self.path)

    def test_reports_malformed_chunk_boundaries_as_validation_errors(self):
        oversized_chunk = b"JUNK" + struct.pack("<I", 1000) + b"unused"
        self.path.write_bytes(pcm_wav(b"\x01\x00", before_data=oversized_chunk))

        with self.assertRaisesRegex(ValueError, "malformed or incomplete WAV") as error:
            load_wav_samples(self.path)

        self.assertIn(str(self.path.resolve()), str(error.exception))

    def test_empty_or_short_headers_have_actionable_errors(self):
        for recording in (b"", b"RIFF", pcm_wav(b"\x01\x00")[:30]):
            with self.subTest(recording=recording):
                self.path.write_bytes(recording)
                with self.assertRaisesRegex(
                    ValueError, "malformed or incomplete WAV"
                ) as error:
                    load_wav_samples(self.path)
                self.assertIn(str(self.path.resolve()), str(error.exception))

    def test_rejects_empty_pcm_data(self):
        self.path.write_bytes(pcm_wav(b""))
        with self.assertRaisesRegex(ValueError, "no audio samples"):
            load_wav_samples(self.path)

    def test_checks_format_before_reading_audio_payload(self):
        for options, message in (
            ({"width": 1}, "16-bit PCM"),
            ({"width": 3}, "16-bit PCM"),
            ({"width": 4}, "16-bit PCM"),
            ({"sample_rate": 0}, "invalid sample rate"),
            ({"channels": 0}, "channels"),
        ):
            with self.subTest(options=options):
                self.path.write_bytes(pcm_wav(b"\x00" * 12, **options))
                with patch("audio_sources.wave.Wave_read.readframes") as read_frames:
                    with self.assertRaisesRegex(ValueError, message):
                        load_wav_samples(self.path)
                read_frames.assert_not_called()

    def test_valid_metadata_and_padding_are_not_treated_as_audio(self):
        original = np.array([-30000, -1, 0, 1, 30000], dtype="<i2")
        self.path.write_bytes(
            pcm_wav(
                original.tobytes(),
                before_data=riff_chunk(b"JUNK", b"odd"),
                after_data=riff_chunk(b"LIST", b"INFO"),
            )
        )

        samples = load_wav_samples(self.path)

        np.testing.assert_array_equal(samples, original)
        self.assertEqual(samples.dtype, np.int16)
        self.assertTrue(samples.flags.writeable)

    def test_missing_recording_reports_its_path(self):
        with self.assertRaisesRegex(ValueError, "Could not read WAV file") as error:
            load_wav_samples(self.path)
        self.assertIn(str(self.path.resolve()), str(error.exception))

    def test_read_errors_report_the_resolved_path_for_noncanonical_input(self):
        nested_directory = self.path.parent / "nested"
        nested_directory.mkdir()
        input_path = nested_directory / ".." / self.path.name
        # Diagnostics canonicalize paths, including Windows 8.3 temp aliases.
        # A '..' component exercises a spelling difference on every platform.
        expected_prefix = f"Could not read WAV file '{self.path.resolve()}':"
        for recording in (None, b""):
            with self.subTest(recording=recording):
                if recording is not None:
                    self.path.write_bytes(recording)
                with self.assertRaises(ValueError) as error:
                    load_wav_samples(input_path)
                self.assertTrue(str(error.exception).startswith(expected_prefix))


class WavCommandLineTests(unittest.TestCase):
    def test_invalid_replay_and_benchmark_files_fail_before_model_initialization(self):
        cases = {
            "missing": None,
            "empty": b"",
            "header": b"RIFF",
            "truncated": pcm_wav(b"\x01\x00" * 8)[:-4],
            "partial_frame": pcm_wav(b"\x01\x00" * 3, channels=2),
            "no_samples": pcm_wav(b""),
            "unsupported_format": pcm_wav(b"\x80" * 8, width=1),
        }
        with tempfile.TemporaryDirectory() as directory:
            for mode in ("replay", "benchmark"):
                for name, recording in cases.items():
                    with self.subTest(mode=mode, case=name):
                        path = Path(directory) / f"{name}.wav"
                        if recording is not None:
                            path.write_bytes(recording)
                        output = StringIO()
                        with (
                            patch("transcriber.load_runtime_config", return_value=RuntimeConfig()),
                            patch("transcriber.RealTimePipeline") as pipeline_type,
                            redirect_stdout(output),
                        ):
                            result = main([f"--{mode}", str(path)])

                        self.assertEqual(result, 2)
                        self.assertIn(f"Invalid {mode} audio:", output.getvalue())
                        pipeline_type.assert_not_called()

    def test_benchmark_loads_once_before_startup_and_reuses_validated_samples(self):
        pipeline = make_pipeline()
        self.addCleanup(pipeline.close)
        original = np.arange(48000, dtype="<i2")
        output = StringIO()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "benchmark.wav"
            path.write_bytes(pcm_wav(original.tobytes()))

            def initialize_pipeline(**_kwargs):
                # Model initialization may take a long time. The benchmark
                # must use the already validated audio, even if the file changes.
                path.unlink()
                return pipeline

            with (
                patch("transcriber.load_runtime_config", return_value=RuntimeConfig()),
                patch("transcriber.load_wav_samples", wraps=load_wav_samples) as decoder,
                patch("transcriber.RealTimePipeline", side_effect=initialize_pipeline) as pipeline_type,
                patch("transcriber.time.perf_counter", side_effect=[10.0, 10.25, 20.0, 20.5]),
                redirect_stdout(output),
            ):
                result = main(["--benchmark", str(path), "--benchmark-runs", "2"])

            decoder.assert_called_once_with(str(path), target_sample_rate=16000)

        self.assertEqual(result, 0)
        for flag in ("enable_vad", "enable_osc", "enable_prompt_budget", "enable_osc_controls"):
            self.assertFalse(pipeline_type.call_args.kwargs[flag])
        calls = pipeline.backend_adapter.transcribe.call_args_list
        self.assertEqual(len(calls), 3)
        np.testing.assert_array_equal(calls[0].args[0], original[:32000])
        for call in calls[1:]:
            np.testing.assert_array_equal(call.args[0], original)
        self.assertIs(calls[1].args[0], calls[2].args[0])
        self.assertIn("Average: 0.375s | real-time factor 0.125", output.getvalue())
        pipeline.backend_adapter.close.assert_called_once_with()

    def test_direct_benchmark_call_still_loads_a_wav_and_runs_a_warmup(self):
        pipeline = make_pipeline()
        self.addCleanup(pipeline.close)
        original = np.array([1000, -1000, 0], dtype="<i2")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "direct.wav"
            path.write_bytes(pcm_wav(original.tobytes()))
            with redirect_stdout(StringIO()):
                pipeline.benchmark(path, runs=0)

        calls = pipeline.backend_adapter.transcribe.call_args_list
        self.assertEqual(len(calls), 2)
        for call in calls:
            np.testing.assert_array_equal(call.args[0], original)

    def test_benchmark_transcription_failure_still_closes_pipeline(self):
        pipeline = make_pipeline()
        self.addCleanup(pipeline.close)
        pipeline.backend_adapter.transcribe.side_effect = RuntimeError("model failure")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "valid.wav"
            path.write_bytes(pcm_wav(b"\x01\x00" * 8))
            with (
                patch("transcriber.load_runtime_config", return_value=RuntimeConfig()),
                patch("transcriber.RealTimePipeline", return_value=pipeline),
                redirect_stdout(StringIO()),
            ):
                with self.assertRaisesRegex(RuntimeError, "model failure"):
                    main(["--benchmark", str(path)])

        pipeline.backend_adapter.close.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
