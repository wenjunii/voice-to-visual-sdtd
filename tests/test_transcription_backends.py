import sys
import types
import unittest
import wave
from dataclasses import replace
from io import BytesIO
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import requests

from backend_errors import RetryableTranscriptionError
from runtime_config import RuntimeConfig
from transcription_backends import (
    FasterWhisperBackend,
    GoogleBackend,
    GroqHybridBackend,
    GroqTranslationBackend,
    LocalTextTranslator,
    WhisperBackend,
    contains_cjk,
    create_transcription_backend,
    encode_pcm_wav,
    normalize_language_code,
    required_modules_for_backend,
)


def config_for(backend, **changes):
    values = {
        "transcription_backend": backend,
        "groq_api_key": "test-key",
        "groq_log_latency": False,
        "whisper_log_latency": False,
        "local_translator_auto_install": False,
    }
    values.update(changes)
    return replace(RuntimeConfig(), **values)


def response(status=200, text="", json_data=None, headers=None):
    result = Mock()
    result.status_code = status
    result.text = text
    result.headers = headers or {}
    result.json.return_value = json_data or {}
    return result


class AudioEncodingTests(unittest.TestCase):
    def test_encodes_mono_16_bit_pcm_wav(self):
        samples = np.array([-32768, 0, 32767], dtype=np.int16)

        encoded = encode_pcm_wav(samples, 16000)

        with wave.open(BytesIO(encoded), "rb") as wav_file:
            self.assertEqual(wav_file.getnchannels(), 1)
            self.assertEqual(wav_file.getsampwidth(), 2)
            self.assertEqual(wav_file.getframerate(), 16000)
            self.assertEqual(wav_file.getnframes(), 3)


class LocalBackendTests(unittest.TestCase):
    def test_openai_whisper_builds_the_translation_request(self):
        model = Mock()
        model.transcribe.return_value = {
            "text": "fallback",
            "segments": [{"text": " hello "}, {"text": "world"}],
        }
        backend = WhisperBackend(
            config_for("whisper"),
            model,
            device="cuda",
            sample_rate=16000,
        )
        samples = np.array([0, 16384, -16384], dtype=np.int16)

        text = backend.transcribe(samples, language="es")

        self.assertEqual(text, "hello world")
        kwargs = model.transcribe.call_args.kwargs
        self.assertEqual(kwargs["task"], "translate")
        self.assertEqual(kwargs["language"], "es")
        self.assertTrue(kwargs["fp16"])
        np.testing.assert_allclose(
            model.transcribe.call_args.args[0],
            samples.astype(np.float32) / 32768.0,
        )

    def test_faster_whisper_consumes_the_segment_generator(self):
        model = Mock()
        model.transcribe.return_value = (
            iter(
                [
                    SimpleNamespace(text=" first "),
                    SimpleNamespace(text=""),
                    SimpleNamespace(text="second"),
                ]
            ),
            object(),
        )
        backend = FasterWhisperBackend(
            config_for("faster_whisper"),
            model,
            device="cuda",
            sample_rate=16000,
        )

        text = backend.transcribe(
            np.array([0, 1], dtype=np.int16),
            language=None,
        )

        self.assertEqual(text, "first second")
        kwargs = model.transcribe.call_args.kwargs
        self.assertEqual(kwargs["task"], "translate")
        self.assertFalse(kwargs["vad_filter"])
        self.assertFalse(kwargs["word_timestamps"])


class GroqTranslationBackendTests(unittest.TestCase):
    def make_backend(self, session, **config_changes):
        return GroqTranslationBackend(
            config_for("groq", **config_changes),
            sample_rate=16000,
            session=session,
        )

    def test_posts_translation_audio_with_the_configured_contract(self):
        session = Mock()
        session.post.return_value = response(text=" translated prompt ")
        backend = self.make_backend(session)

        text = backend.transcribe(np.array([0, 1], dtype=np.int16))

        self.assertEqual(text, "translated prompt")
        call = session.post.call_args
        self.assertEqual(
            call.args[0],
            backend.config.groq_translations_endpoint,
        )
        self.assertEqual(
            call.kwargs["headers"]["Authorization"],
            "Bearer test-key",
        )
        self.assertEqual(
            call.kwargs["data"]["model"],
            "whisper-large-v3",
        )
        self.assertEqual(call.kwargs["data"]["response_format"], "text")
        self.assertEqual(
            call.kwargs["timeout"],
            backend.config.groq_request_timeout,
        )
        self.assertEqual(call.kwargs["files"]["file"][0], "speech.wav")
        self.assertTrue(
            call.kwargs["files"]["file"][1].startswith(b"RIFF")
        )

    def test_reads_json_response_formats(self):
        session = Mock()
        session.post.return_value = response(
            json_data={"text": "json transcript"}
        )
        backend = self.make_backend(
            session,
            groq_response_format="json",
        )

        text = backend.transcribe(np.array([0], dtype=np.int16))

        self.assertEqual(text, "json transcript")

    def test_respects_retry_after_for_rate_limits(self):
        session = Mock()
        session.post.return_value = response(
            status=429,
            text="slow down",
            headers={"Retry-After": "3.5"},
        )
        backend = self.make_backend(session)

        with self.assertRaises(RetryableTranscriptionError) as context:
            backend.transcribe(np.array([0], dtype=np.int16))

        self.assertEqual(context.exception.retry_after, 3.5)
        self.assertIn("rate-limited", str(context.exception))

    def test_treats_transient_server_errors_as_retryable(self):
        session = Mock()
        session.post.return_value = response(
            status=503,
            text="temporarily unavailable",
        )
        backend = self.make_backend(session)

        with self.assertRaisesRegex(
            RetryableTranscriptionError,
            r"temporarily unavailable \(HTTP 503\)",
        ):
            backend.transcribe(np.array([0], dtype=np.int16))

    def test_explains_when_the_model_cannot_translate_audio(self):
        session = Mock()
        session.post.return_value = response(
            status=400,
            text="model does not support `translate`",
        )
        backend = self.make_backend(session)

        with self.assertRaisesRegex(
            RuntimeError,
            "GROQ_TRANSCRIPTION_MODEL=whisper-large-v3",
        ):
            backend.transcribe(np.array([0], dtype=np.int16))

    def test_converts_network_errors_to_retryable_failures(self):
        session = Mock()
        session.post.side_effect = requests.ConnectionError("offline")
        backend = self.make_backend(session)

        with self.assertRaisesRegex(
            RetryableTranscriptionError,
            "request failed",
        ):
            backend.transcribe(np.array([0], dtype=np.int16))

    def test_uses_text_translation_for_cjk_fallback(self):
        session = Mock()
        session.post.side_effect = [
            response(text="你好"),
            response(
                json_data={
                    "choices": [
                        {"message": {"content": "hello"}}
                    ]
                }
            ),
        ]
        backend = self.make_backend(session)

        text = backend.transcribe(np.array([0], dtype=np.int16))

        self.assertEqual(text, "hello")
        self.assertEqual(session.post.call_count, 2)
        second_call = session.post.call_args_list[1]
        self.assertEqual(
            second_call.args[0],
            backend.config.groq_chat_endpoint,
        )
        self.assertEqual(
            second_call.kwargs["json"]["messages"][1]["content"],
            "你好",
        )

    def test_closes_the_http_session_once(self):
        session = Mock()
        backend = self.make_backend(session)

        backend.close()
        backend.close()

        session.close.assert_called_once_with()


class GroqHybridBackendTests(unittest.TestCase):
    def test_passes_language_to_groq_and_local_translation(self):
        session = Mock()
        session.post.return_value = response(text="hola mundo")
        translator = Mock()
        translator.translate.return_value = "hello world"
        backend = GroqHybridBackend(
            config_for("groq_hybrid"),
            sample_rate=16000,
            session=session,
            local_translator=translator,
        )

        text = backend.transcribe(
            np.array([0], dtype=np.int16),
            language="es",
        )

        self.assertEqual(text, "hello world")
        self.assertEqual(session.post.call_args.kwargs["data"]["language"], "es")
        translator.translate.assert_called_once_with(
            "hola mundo",
            selected_language="es",
        )

    def test_falls_back_to_groq_text_when_local_output_remains_cjk(self):
        session = Mock()
        session.post.side_effect = [
            response(text="你好"),
            response(
                json_data={
                    "choices": [
                        {"message": {"content": "hello"}}
                    ]
                }
            ),
        ]
        translator = Mock()
        translator.translate.return_value = "你好"
        backend = GroqHybridBackend(
            config_for("groq_hybrid"),
            sample_rate=16000,
            session=session,
            local_translator=translator,
        )

        text = backend.transcribe(np.array([0], dtype=np.int16))

        self.assertEqual(text, "hello")


class LocalTextTranslatorTests(unittest.TestCase):
    def test_selected_language_is_normalized_before_translation(self):
        translation = Mock()
        translation.translate.return_value = "hello"
        source = Mock(code="es")
        target = Mock(code="en")
        source.get_translation.return_value = translation
        argos_translate = Mock()
        argos_translate.get_installed_languages.return_value = [
            source,
            target,
        ]
        translator = LocalTextTranslator(
            config_for("groq_hybrid"),
            argos_package=Mock(),
            argos_translate=argos_translate,
            logger=Mock(),
        )

        text = translator.translate(
            "hola",
            selected_language="es-MX",
        )

        self.assertEqual(text, "hello")
        source.get_translation.assert_called_once_with(target)

    def test_english_text_bypasses_argos(self):
        argos_translate = Mock()
        translator = LocalTextTranslator(
            config_for("groq_hybrid"),
            detect_language=Mock(return_value="en"),
            argos_package=Mock(),
            argos_translate=argos_translate,
        )

        text = translator.translate("a glowing city after rain")

        self.assertEqual(text, "a glowing city after rain")
        argos_translate.get_installed_languages.assert_not_called()


class GoogleBackendTests(unittest.TestCase):
    @staticmethod
    def speech_module():
        class UnknownValueError(Exception):
            pass

        class RequestError(Exception):
            pass

        module = SimpleNamespace(
            AudioData=Mock(side_effect=lambda data, rate, width: (data, rate, width)),
            UnknownValueError=UnknownValueError,
            RequestError=RequestError,
        )
        return module

    def test_maps_short_language_codes(self):
        recognizer = Mock()
        recognizer.recognize_google.return_value = " hola "
        speech_module = self.speech_module()
        backend = GoogleBackend(
            config_for("google"),
            recognizer,
            speech_module,
            sample_rate=16000,
        )

        text = backend.transcribe(
            np.array([0, 1], dtype=np.int16),
            language="es",
        )

        self.assertEqual(text, "hola")
        self.assertEqual(
            recognizer.recognize_google.call_args.kwargs["language"],
            "es-ES",
        )

    def test_configures_timeout_on_the_injected_recognizer(self):
        recognizer = Mock()
        recognizer.operation_timeout = None
        recognizer.recognize_google.return_value = "hello"
        backend = GoogleBackend(
            config_for("google", google_request_timeout=3.5), recognizer,
            self.speech_module(), sample_rate=16000,
        )
        backend.transcribe(np.array([0], dtype=np.int16))
        self.assertEqual(recognizer.operation_timeout, 3.5)

    def test_socket_timeouts_are_retryable(self):
        recognizer = Mock()
        timeout = TimeoutError("socket stalled")
        recognizer.recognize_google.side_effect = timeout
        backend = GoogleBackend(
            config_for("google"), recognizer, self.speech_module(), sample_rate=16000,
        )
        with self.assertRaisesRegex(RetryableTranscriptionError, "timed out") as context:
            backend.transcribe(np.array([0], dtype=np.int16))
        self.assertIs(context.exception.__cause__, timeout)
        self.assertIsNone(context.exception.retry_after)

    def test_unknown_speech_returns_an_empty_transcript(self):
        recognizer = Mock()
        speech_module = self.speech_module()
        recognizer.recognize_google.side_effect = (
            speech_module.UnknownValueError()
        )
        backend = GoogleBackend(
            config_for("google"),
            recognizer,
            speech_module,
            sample_rate=16000,
        )

        text = backend.transcribe(np.array([0], dtype=np.int16))

        self.assertEqual(text, "")

    def test_request_errors_are_retryable(self):
        recognizer = Mock()
        speech_module = self.speech_module()
        recognizer.recognize_google.side_effect = (
            speech_module.RequestError("unavailable")
        )
        backend = GoogleBackend(
            config_for("google"),
            recognizer,
            speech_module,
            sample_rate=16000,
        )

        with self.assertRaisesRegex(
            RetryableTranscriptionError,
            "Google transcription request failed",
        ):
            backend.transcribe(np.array([0], dtype=np.int16))


class BackendFactoryTests(unittest.TestCase):
    @staticmethod
    def torch_module(cuda_available=True):
        module = types.ModuleType("torch")
        module.cuda = SimpleNamespace(
            is_available=Mock(return_value=cuda_available)
        )
        return module

    def test_builds_a_groq_backend_with_an_injected_session(self):
        session = Mock()
        logs = []

        backend = create_transcription_backend(
            config_for("groq"),
            sample_rate=16000,
            session_factory=Mock(return_value=session),
            logger=logs.append,
        )

        self.assertIsInstance(backend, GroqTranslationBackend)
        self.assertIs(backend.session, session)
        self.assertTrue(any("Groq online" in message for message in logs))

    def test_rejects_a_missing_groq_key(self):
        config = config_for("groq", groq_api_key="")

        with self.assertRaisesRegex(RuntimeError, "requires GROQ_API_KEY"):
            create_transcription_backend(config, sample_rate=16000)

    def test_loads_the_openai_whisper_model_through_the_factory(self):
        torch_module = self.torch_module()
        whisper_module = types.ModuleType("whisper")
        whisper_module.load_model = Mock(return_value=Mock())

        with patch.dict(
            sys.modules,
            {
                "torch": torch_module,
                "whisper": whisper_module,
            },
        ):
            backend = create_transcription_backend(
                config_for("whisper"),
                sample_rate=16000,
                logger=Mock(),
            )

        self.assertIsInstance(backend, WhisperBackend)
        whisper_module.load_model.assert_called_once_with(
            "small",
            device="cuda",
        )

    def test_missing_backend_packages_name_the_matching_profile(self):
        torch_module = self.torch_module()
        cases = (
            ("whisper", {"torch": torch_module, "whisper": None}),
            (
                "faster_whisper",
                {"torch": torch_module, "faster_whisper": None},
            ),
            ("google", {"speech_recognition": None}),
        )

        for backend, modules in cases:
            with self.subTest(backend=backend):
                profile = backend.replace("_", "-")
                with patch.dict(sys.modules, modules):
                    with self.assertRaisesRegex(
                        RuntimeError,
                        rf"requirements/{profile}\.txt",
                    ):
                        create_transcription_backend(
                            config_for(backend),
                            sample_rate=16000,
                            logger=Mock(),
                        )

    def test_wraps_faster_whisper_cuda_initialization_failures(self):
        torch_module = self.torch_module()
        faster_module = types.ModuleType("faster_whisper")
        faster_module.WhisperModel = Mock(
            side_effect=OSError("CUDA DLL unavailable")
        )

        with patch.dict(
            sys.modules,
            {
                "torch": torch_module,
                "faster_whisper": faster_module,
            },
        ):
            with self.assertRaisesRegex(
                RuntimeError,
                "could not initialize CUDA",
            ):
                create_transcription_backend(
                    config_for("faster_whisper"),
                    sample_rate=16000,
                    logger=Mock(),
                )

    def test_rejects_unknown_backend_names(self):
        with self.assertRaisesRegex(
            RuntimeError,
            "Unsupported transcription backend",
        ):
            create_transcription_backend(
                config_for("unknown"),
                sample_rate=16000,
            )

    def test_exposes_backend_requirements_for_diagnostics(self):
        self.assertEqual(
            required_modules_for_backend("faster_whisper"),
            {"torch", "faster_whisper", "ctranslate2"},
        )
        self.assertEqual(required_modules_for_backend("groq"), set())
        self.assertEqual(
            required_modules_for_backend("groq_hybrid"),
            {"argostranslate", "langdetect"},
        )


class LanguageHelperTests(unittest.TestCase):
    def test_detects_cjk_characters(self):
        self.assertTrue(contains_cjk("scene with 上海 lights"))
        self.assertFalse(contains_cjk("scene with city lights"))

    def test_normalizes_supported_language_variants(self):
        self.assertEqual(normalize_language_code("zh-CN"), "zh")
        self.assertEqual(normalize_language_code("es-MX"), "es")
        self.assertEqual(normalize_language_code("en-US"), "en")


if __name__ == "__main__":
    unittest.main()
