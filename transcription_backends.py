import io
import time
import wave

import numpy as np
import requests

from backend_errors import RetryableTranscriptionError, retry_after_seconds
from dependency_profiles import install_command_for_backend


BACKEND_REQUIRED_MODULES = {
    "whisper": {"torch", "whisper"},
    "faster_whisper": {"torch", "faster_whisper", "ctranslate2"},
    "groq": set(),
    "groq_hybrid": {"argostranslate", "langdetect"},
    "google": {"speech_recognition"},
}


def required_modules_for_backend(name):
    return set(BACKEND_REQUIRED_MODULES.get(name, set()))


class TranscriptionBackend:
    def __init__(
        self,
        name,
        *,
        online,
        request_interval,
        minimum_audio_seconds,
        maximum_audio_seconds,
    ):
        self.name = name
        self.online = online
        self.request_interval = request_interval
        self.minimum_audio_seconds = minimum_audio_seconds
        self.maximum_audio_seconds = max(1.0, maximum_audio_seconds)

    def transcribe(self, audio_samples, language=None):
        raise NotImplementedError

    def close(self):
        return None


class WhisperBackend(TranscriptionBackend):
    def __init__(
        self,
        config,
        model,
        *,
        device,
        sample_rate,
        timer=time.perf_counter,
        logger=print,
    ):
        super().__init__(
            "whisper",
            online=False,
            request_interval=config.whisper_transcription_interval,
            minimum_audio_seconds=config.whisper_min_audio_seconds,
            maximum_audio_seconds=config.whisper_max_audio_seconds,
        )
        self.config = config
        self.model = model
        self.device = device
        self.sample_rate = sample_rate
        self.timer = timer
        self.logger = logger

    def transcribe(self, audio_samples, language=None):
        full_audio = audio_samples.astype(np.float32) / 32768.0
        started = self.timer()
        result = self.model.transcribe(
            full_audio,
            fp16=self.device.startswith("cuda"),
            task="translate",
            language=language,
            beam_size=self.config.whisper_beam_size,
            best_of=self.config.whisper_best_of,
            temperature=self.config.whisper_temperature,
            condition_on_previous_text=(
                self.config.whisper_condition_on_previous_text
            ),
        )
        elapsed = self.timer() - started
        if self.config.whisper_log_latency:
            self.logger(
                f"\n[WHISPER LATENCY]: {elapsed:.2f}s for "
                f"{len(audio_samples) / self.sample_rate:.1f}s audio"
            )

        segments = result.get("segments", [])
        if segments:
            return " ".join(
                segment["text"].strip()
                for segment in segments
                if segment["text"].strip()
            )
        return result["text"].strip()


class FasterWhisperBackend(TranscriptionBackend):
    def __init__(
        self,
        config,
        model,
        *,
        device,
        sample_rate,
        timer=time.perf_counter,
        logger=print,
    ):
        super().__init__(
            "faster_whisper",
            online=False,
            request_interval=config.whisper_transcription_interval,
            minimum_audio_seconds=config.whisper_min_audio_seconds,
            maximum_audio_seconds=config.whisper_max_audio_seconds,
        )
        self.config = config
        self.model = model
        self.device = device
        self.sample_rate = sample_rate
        self.timer = timer
        self.logger = logger
        self.compute_type = config.faster_whisper_compute_type

    def transcribe(self, audio_samples, language=None):
        full_audio = audio_samples.astype(np.float32) / 32768.0
        started = self.timer()
        segments, _info = self.model.transcribe(
            full_audio,
            task="translate",
            language=language,
            beam_size=self.config.whisper_beam_size,
            best_of=self.config.whisper_best_of,
            temperature=self.config.whisper_temperature,
            condition_on_previous_text=(
                self.config.whisper_condition_on_previous_text
            ),
            vad_filter=False,
            word_timestamps=False,
        )
        segments = list(segments)
        elapsed = self.timer() - started
        if self.config.whisper_log_latency:
            self.logger(
                f"\n[FASTER-WHISPER LATENCY]: {elapsed:.2f}s for "
                f"{len(audio_samples) / self.sample_rate:.1f}s audio"
            )
        return " ".join(
            segment.text.strip()
            for segment in segments
            if segment.text.strip()
        )


class _GroqBackend(TranscriptionBackend):
    def __init__(
        self,
        name,
        config,
        *,
        sample_rate,
        session=None,
        timer=time.perf_counter,
        logger=print,
    ):
        super().__init__(
            name,
            online=True,
            request_interval=config.groq_transcription_interval,
            minimum_audio_seconds=config.groq_min_audio_seconds,
            maximum_audio_seconds=config.groq_max_audio_seconds,
        )
        self.config = config
        self.sample_rate = sample_rate
        self.session = (
            session if session is not None else requests.Session()
        )
        self.timer = timer
        self.logger = logger
        self._closed = False

    def close(self):
        if self._closed:
            return
        self._closed = True
        self.session.close()

    def _authorization_headers(self):
        return {"Authorization": f"Bearer {self.config.groq_api_key}"}

    def _audio_files(self, audio_samples):
        return {
            "file": (
                "speech.wav",
                encode_pcm_wav(audio_samples, self.sample_rate),
                "audio/wav",
            )
        }

    def _response_text(self, response):
        if self.config.groq_response_format == "text":
            return response.text.strip()
        return response.json().get("text", "").strip()

    def translate_text_to_english(self, text):
        payload = {
            "model": self.config.groq_text_translation_model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a translation engine. Translate the user's "
                        "text completely into natural concise English for a "
                        "visual generation prompt. Output only English text. "
                        "Do not include any Chinese characters."
                    ),
                },
                {"role": "user", "content": text},
            ],
            "temperature": 0,
            "max_tokens": 120,
        }

        try:
            started = self.timer()
            response = self.session.post(
                self.config.groq_chat_endpoint,
                headers={
                    **self._authorization_headers(),
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=self.config.groq_request_timeout,
            )
            elapsed = self.timer() - started
            raise_for_retryable_http_response(
                response,
                "Groq text translation",
            )
            if response.status_code >= 400:
                self.logger(
                    "\n[GROQ TEXT TRANSLATION ERROR]: "
                    f"HTTP {response.status_code} {response.text[:160]}"
                )
                return ""
            translation = (
                response.json()
                .get("choices", [{}])[0]
                .get("message", {})
                .get("content", "")
                .strip()
            )
            if self.config.groq_log_latency:
                self.logger(f"\n[GROQ TEXT TRANSLATION]: {elapsed:.2f}s")
            return translation
        except requests.RequestException as exc:
            raise RetryableTranscriptionError(
                f"Groq text translation request failed: {exc}"
            ) from exc


class GroqTranslationBackend(_GroqBackend):
    def __init__(self, config, **kwargs):
        super().__init__("groq", config, **kwargs)

    def transcribe(self, audio_samples, language=None):
        data = {
            "model": self.config.groq_transcription_model,
            "response_format": self.config.groq_response_format,
            "temperature": "0",
        }
        if self.config.groq_translation_prompt:
            data["prompt"] = self.config.groq_translation_prompt

        try:
            started = self.timer()
            response = self.session.post(
                self.config.groq_translations_endpoint,
                headers=self._authorization_headers(),
                files=self._audio_files(audio_samples),
                data=data,
                timeout=self.config.groq_request_timeout,
            )
            elapsed = self.timer() - started
            if self.config.groq_log_latency:
                self.logger(
                    f"\n[GROQ LATENCY]: {elapsed:.2f}s for "
                    f"{len(audio_samples) / self.sample_rate:.1f}s audio"
                )
            raise_for_retryable_http_response(response, "Groq transcription")
            if response.status_code >= 400:
                if (
                    response.status_code == 400
                    and "does not support `translate`" in response.text
                ):
                    raise RuntimeError(
                        "Groq audio translation requires "
                        "GROQ_TRANSCRIPTION_MODEL=whisper-large-v3"
                    )
                raise RuntimeError(
                    "Groq transcription rejected the request "
                    f"(HTTP {response.status_code}): {response.text[:160]}"
                )
            text = self._response_text(response)
            return self._ensure_english_prompt_text(text)
        except requests.RequestException as exc:
            raise RetryableTranscriptionError(
                f"Groq transcription request failed: {exc}"
            ) from exc

    def _ensure_english_prompt_text(self, text):
        if not text:
            return ""
        if self.config.groq_english_fallback == "off":
            return text
        if (
            self.config.groq_english_fallback == "always"
            or contains_cjk(text)
        ):
            translated = self.translate_text_to_english(text)
            return translated or text
        return text


class GroqHybridBackend(_GroqBackend):
    def __init__(self, config, *, local_translator=None, **kwargs):
        super().__init__("groq_hybrid", config, **kwargs)
        self.local_translator = (
            local_translator
            if local_translator is not None
            else LocalTextTranslator(
                config,
                timer=self.timer,
                logger=self.logger,
            )
        )

    def transcribe(self, audio_samples, language=None):
        data = {
            "model": self.config.groq_hybrid_model,
            "response_format": self.config.groq_response_format,
            "temperature": "0",
        }
        if language:
            data["language"] = language

        try:
            started = self.timer()
            response = self.session.post(
                self.config.groq_transcriptions_endpoint,
                headers=self._authorization_headers(),
                files=self._audio_files(audio_samples),
                data=data,
                timeout=self.config.groq_request_timeout,
            )
            elapsed = self.timer() - started
            if self.config.groq_log_latency:
                self.logger(
                    f"\n[GROQ HYBRID LATENCY]: {elapsed:.2f}s for "
                    f"{len(audio_samples) / self.sample_rate:.1f}s audio"
                )
            raise_for_retryable_http_response(
                response,
                "Groq hybrid transcription",
            )
            if response.status_code >= 400:
                raise RuntimeError(
                    "Groq hybrid transcription rejected the request "
                    f"(HTTP {response.status_code}): {response.text[:160]}"
                )
            text = self._response_text(response)
            return self._ensure_local_english_prompt_text(text, language)
        except requests.RequestException as exc:
            raise RetryableTranscriptionError(
                f"Groq hybrid transcription request failed: {exc}"
            ) from exc

    def _ensure_local_english_prompt_text(self, text, selected_language):
        if not text:
            return ""
        translated = self.local_translator.translate(
            text,
            selected_language=selected_language,
        )
        if translated and not contains_cjk(translated):
            return translated
        if self.config.hybrid_translation_fallback == "groq_text":
            fallback = self.translate_text_to_english(text)
            if fallback:
                return fallback
        return translated or text


class LocalTextTranslator:
    def __init__(
        self,
        config,
        *,
        detect_language=None,
        language_error=Exception,
        argos_package=None,
        argos_translate=None,
        timer=time.perf_counter,
        logger=print,
    ):
        self.config = config
        self.detect_language = detect_language
        self.language_error = language_error
        self.argos_package = argos_package
        self.argos_translate = argos_translate
        self.timer = timer
        self.logger = logger
        self.translation_cache = {}

    def initialize(self):
        if self.config.local_translator in {"none", "off"}:
            self.logger(
                "[LOCAL TRANSLATOR]: Disabled. "
                "Non-English text will pass through."
            )
            return
        if self.config.local_translator != "argos":
            self.logger(
                f"[LOCAL TRANSLATOR]: Unknown translator "
                f"'{self.config.local_translator}'. "
                "Non-English text will pass through."
            )
            return
        if self.argos_package is None or self.argos_translate is None:
            try:
                import argostranslate.package as argos_package
                import argostranslate.translate as argos_translate

                self.argos_package = argos_package
                self.argos_translate = argos_translate
            except ImportError:
                self.logger(
                    "[LOCAL TRANSLATOR]: Argos Translate is not installed. "
                    f"Run: {install_command_for_backend('groq_hybrid')}"
                )
                return
            except Exception as exc:
                self.logger(
                    f"[LOCAL TRANSLATOR]: Argos Translate could not load: {exc}"
                )
                self.logger(
                    "[LOCAL TRANSLATOR]: Non-English hybrid transcripts "
                    "will pass through untranslated."
                )
                return

        if self.detect_language is None:
            try:
                from langdetect import detect
                from langdetect.lang_detect_exception import LangDetectException

                self.detect_language = detect
                self.language_error = LangDetectException
            except ImportError:
                self.detect_language = None

        for source_code in self.config.local_translator_preload_languages:
            self._get_translation(source_code)

    def translate(self, text, selected_language=None):
        source_language = self._detect_source_language(
            text,
            selected_language,
        )
        if (
            not source_language
            or source_language == self.config.local_translator_target_language
        ):
            return text
        if self.config.local_translator != "argos":
            return ""
        translation = self._get_translation(source_language)
        if translation is None:
            return ""
        try:
            started = self.timer()
            translated = translation.translate(text).strip()
            elapsed = self.timer() - started
            if self.config.local_translator_log_latency:
                self.logger(
                    f"\n[LOCAL TRANSLATION]: {source_language}->en "
                    f"{elapsed:.2f}s"
                )
            return translated
        except Exception as exc:
            self.logger(f"\n[LOCAL TRANSLATION ERROR]: {exc}")
            return ""

    def _detect_source_language(self, text, selected_language):
        if selected_language:
            return normalize_language_code(selected_language)
        if contains_cjk(text):
            return "zh"
        if self.detect_language is not None and len(text.strip()) >= 8:
            try:
                return normalize_language_code(self.detect_language(text))
            except self.language_error:
                pass
        if looks_like_english(text):
            return "en"
        return self.config.local_translator_default_source_language

    def _get_translation(self, source_language):
        source_language = normalize_language_code(source_language)
        target = self.config.local_translator_target_language
        cache_key = (source_language, target)
        if cache_key in self.translation_cache:
            return self.translation_cache[cache_key]
        if self.argos_translate is None:
            return None

        translation = self._find_translation(source_language)
        if translation is None and self.config.local_translator_auto_install:
            self._install_package(source_language)
            translation = self._find_translation(source_language)

        if translation is None:
            self.logger(
                f"\n[LOCAL TRANSLATION]: No Argos package for "
                f"{source_language}->en. Text will pass through."
            )
        self.translation_cache[cache_key] = translation
        return translation

    def _find_translation(self, source_language):
        installed_languages = self.argos_translate.get_installed_languages()
        target = self.config.local_translator_target_language
        from_language = next(
            (
                language
                for language in installed_languages
                if language.code == source_language
            ),
            None,
        )
        to_language = next(
            (
                language
                for language in installed_languages
                if language.code == target
            ),
            None,
        )
        if not from_language or not to_language:
            return None
        try:
            return from_language.get_translation(to_language)
        except Exception:
            return None

    def _install_package(self, source_language):
        if self.argos_package is None:
            return
        target = self.config.local_translator_target_language
        try:
            self.logger(
                f"\n[LOCAL TRANSLATION]: Installing Argos package "
                f"{source_language}->{target}..."
            )
            self.argos_package.update_package_index()
            available_packages = self.argos_package.get_available_packages()
            package = next(
                (
                    candidate
                    for candidate in available_packages
                    if candidate.from_code == source_language
                    and candidate.to_code == target
                ),
                None,
            )
            if package is None:
                self.logger(
                    f"\n[LOCAL TRANSLATION]: No downloadable Argos package "
                    f"for {source_language}->{target}."
                )
                return
            download_path = package.download()
            self.argos_package.install_from_path(download_path)
        except Exception as exc:
            self.logger(
                f"\n[LOCAL TRANSLATION ERROR]: Could not install "
                f"{source_language}->{target} package: {exc}"
            )


class GoogleBackend(TranscriptionBackend):
    def __init__(
        self,
        config,
        recognizer,
        speech_recognition,
        *,
        sample_rate,
    ):
        super().__init__(
            "google",
            online=True,
            request_interval=config.google_transcription_interval,
            minimum_audio_seconds=config.google_min_audio_seconds,
            maximum_audio_seconds=config.google_max_audio_seconds,
        )
        self.config = config
        self.recognizer = recognizer
        self.recognizer.operation_timeout = config.google_request_timeout
        self.speech_recognition = speech_recognition
        self.sample_rate = sample_rate

    def transcribe(self, audio_samples, language=None):
        audio_data = self.speech_recognition.AudioData(
            audio_samples.astype(np.int16).tobytes(),
            self.sample_rate,
            2,
        )
        selected_language = (
            self._language_map().get(language, language)
            if language
            else self.config.google_speech_language
        )
        try:
            return self.recognizer.recognize_google(
                audio_data,
                language=selected_language,
            ).strip()
        except self.speech_recognition.UnknownValueError:
            return ""
        except TimeoutError as exc:
            raise RetryableTranscriptionError(
                "Google transcription request timed out"
            ) from exc
        except self.speech_recognition.RequestError as exc:
            raise RetryableTranscriptionError(
                f"Google transcription request failed: {exc}"
            ) from exc

    def _language_map(self):
        return {
            "en": self.config.google_speech_english_language,
            "zh": self.config.google_speech_chinese_language,
            "es": self.config.google_speech_spanish_language,
        }


def create_transcription_backend(
    config,
    *,
    sample_rate,
    session_factory=requests.Session,
    logger=print,
):
    name = config.transcription_backend
    if name in {"whisper", "faster_whisper"}:
        try:
            import torch
        except Exception as exc:
            raise RuntimeError(
                f"TRANSCRIPTION_BACKEND={name} requires torch. "
                f"Run: {install_command_for_backend(name)}"
            ) from exc

        device = config.whisper_device
        if not device:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if not device.startswith("cuda") or not torch.cuda.is_available():
            raise RuntimeError(
                f"TRANSCRIPTION_BACKEND={name} is configured to require a "
                "CUDA GPU. Use TRANSCRIPTION_BACKEND=groq for online "
                "translation without local GPU usage."
            )

        if name == "whisper":
            try:
                import whisper
            except ImportError as exc:
                raise RuntimeError(
                    "TRANSCRIPTION_BACKEND=whisper requires openai-whisper. "
                    f"Run: {install_command_for_backend(name)}"
                ) from exc
            logger(
                f"Loading Whisper model '{config.whisper_model_size}' "
                f"on {device}..."
            )
            model = whisper.load_model(
                config.whisper_model_size,
                device=device,
            )
            return WhisperBackend(
                config,
                model,
                device=device,
                sample_rate=sample_rate,
                logger=logger,
            )

        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise RuntimeError(
                "TRANSCRIPTION_BACKEND=faster_whisper requires "
                "faster-whisper. "
                f"Run: {install_command_for_backend(name)}"
            ) from exc
        logger(
            f"Loading faster-whisper model '{config.whisper_model_size}' "
            f"on {device} ({config.faster_whisper_compute_type})..."
        )
        try:
            model = WhisperModel(
                config.whisper_model_size,
                device=device,
                compute_type=config.faster_whisper_compute_type,
                cpu_threads=config.faster_whisper_cpu_threads,
                num_workers=config.faster_whisper_num_workers,
            )
        except Exception as exc:
            raise RuntimeError(
                "faster-whisper could not initialize CUDA. This project pins "
                "CTranslate2 for CUDA 12 with cuDNN 8; verify that the NVIDIA "
                "libraries are available on PATH."
            ) from exc
        return FasterWhisperBackend(
            config,
            model,
            device=device,
            sample_rate=sample_rate,
            logger=logger,
        )

    if name in {"groq", "groq_hybrid"}:
        if not config.is_secret_configured(config.groq_api_key):
            raise RuntimeError(
                f"TRANSCRIPTION_BACKEND={name} requires GROQ_API_KEY "
                "in your environment or .env file."
            )
        session = session_factory()
        if name == "groq":
            logger(
                "Using Groq online Whisper translation backend "
                f"({config.groq_transcription_model}). "
                "No local Whisper model loaded."
            )
            logger(
                "Note: this sends microphone audio to Groq and returns "
                "English text for StreamDiffusion prompts."
            )
            return GroqTranslationBackend(
                config,
                sample_rate=sample_rate,
                session=session,
                logger=logger,
            )

        logger(
            f"Using Groq hybrid backend ({config.groq_hybrid_model}) "
            "with local CPU text translation."
        )
        logger(
            "Note: Groq transcribes audio online; non-English text is "
            "translated locally when possible."
        )
        translator = LocalTextTranslator(config, logger=logger)
        translator.initialize()
        return GroqHybridBackend(
            config,
            sample_rate=sample_rate,
            session=session,
            local_translator=translator,
            logger=logger,
        )

    if name != "google":
        raise RuntimeError(f"Unsupported transcription backend: {name}")

    try:
        import speech_recognition
    except ImportError as exc:
        raise RuntimeError(
            "TRANSCRIPTION_BACKEND=google requires the SpeechRecognition "
            "package. "
            f"Run: {install_command_for_backend(name)}"
        ) from exc
    logger(
        "Using online Google Speech Recognition backend. "
        "No local Whisper model loaded."
    )
    logger(
        "Note: this sends microphone audio to Google's speech service "
        "and does not translate to English."
    )
    return GoogleBackend(
        config,
        speech_recognition.Recognizer(),
        speech_recognition,
        sample_rate=sample_rate,
    )


def raise_for_retryable_http_response(response, operation):
    if response.status_code == 429:
        raise RetryableTranscriptionError(
            f"{operation} was rate-limited by Groq (HTTP 429)",
            retry_after=retry_after_seconds(response.headers),
        )
    if response.status_code in {408, 409, 425} or response.status_code >= 500:
        raise RetryableTranscriptionError(
            f"{operation} is temporarily unavailable "
            f"(HTTP {response.status_code})",
            retry_after=retry_after_seconds(response.headers),
        )


def encode_pcm_wav(audio_samples, sample_rate, channels=1):
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio_samples.astype(np.int16).tobytes())
    return buffer.getvalue()


def contains_cjk(text):
    return any(
        "\u3400" <= character <= "\u9fff"
        or "\uf900" <= character <= "\ufaff"
        for character in text
    )


def normalize_language_code(language_code):
    code = (language_code or "").lower()
    if code.startswith("zh"):
        return "zh"
    if code.startswith("es"):
        return "es"
    if code.startswith("en"):
        return "en"
    return code


def looks_like_english(text):
    letters = [character for character in text if character.isalpha()]
    if not letters:
        return True
    ascii_letters = [
        character
        for character in letters
        if "a" <= character.lower() <= "z"
    ]
    return len(ascii_letters) / len(letters) > 0.85
