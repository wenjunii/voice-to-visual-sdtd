import math
import os
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Mapping, Optional
from urllib.parse import urlparse


SUPPORTED_BACKENDS = {"whisper", "faster_whisper", "groq", "groq_hybrid", "google"}
SUPPORTED_GENDERS = {"man", "neutral", "woman"}
SUPPORTED_AGES = {"adult", "elder", "young"}
SUPPORTED_VISUAL_MODES = {
    "asian_american",
    "asian_black_brown",
    "black_brown",
}
SUPPORTED_PROMPT_STYLES = {"general_scene", "human_focus"}
SUPPORTED_LANGUAGES = {"auto", "en", "es", "zh"}
SUPPORTED_PROMPT_REFINEMENT_PROVIDERS = {"capriole", "ollama"}
SUPPORTED_PROMPT_BUDGET_FALLBACKS = {"conservative", "off"}
SUPPORTED_FINAL_OVERFLOW_POLICIES = {"drop_oldest", "drop_newest"}
TRUE_VALUES = {"1", "true", "yes", "on"}
FALSE_VALUES = {"0", "false", "no", "off"}
NORMALIZED_ENV_NAMES = {
    "TRANSCRIPTION_BACKEND",
    "WHISPER_DEVICE",
    "GROQ_RESPONSE_FORMAT",
    "GROQ_ENGLISH_FALLBACK",
    "LOCAL_TRANSLATOR",
    "LOCAL_TRANSLATOR_TARGET_LANGUAGE",
    "LOCAL_TRANSLATOR_DEFAULT_SOURCE_LANGUAGE",
    "LOCAL_TRANSLATOR_PRELOAD_LANGUAGES",
    "HYBRID_TRANSLATION_FALLBACK",
    "VAD_ENGINE",
    "RUNTIME_LOG_LEVEL",
    "PROMPT_TOKEN_BUDGET_FALLBACK",
    "TRANSCRIPTION_FINAL_OVERFLOW_POLICY",
    "DEFAULT_GENDER",
    "DEFAULT_AGE",
    "DEFAULT_VISUAL_MODE",
    "DEFAULT_PROMPT_STYLE",
    "DEFAULT_LANGUAGE",
}


class ConfigError(ValueError):
    def __init__(self, errors):
        self.errors = tuple(errors)
        super().__init__("; ".join(self.errors))


def _config_field(env_name, default, *, secret=False):
    return field(
        default=default,
        metadata={"env": env_name, "secret": secret},
    )


class _EnvironmentReader:
    def __init__(self, values):
        self.values = values
        self.errors = []

    def value(self, name, default):
        raw = self.values.get(name)
        if raw is None:
            return default
        text = str(raw).strip()

        if default is None:
            if not text:
                return None
            try:
                parsed = int(text)
            except (TypeError, ValueError):
                self.errors.append(f"{name} must be a non-negative integer")
                return default
            if parsed < 0:
                self.errors.append(f"{name} must be a non-negative integer")
                return default
            return parsed

        if isinstance(default, bool):
            if not text:
                return default
            normalized = text.lower()
            if normalized in TRUE_VALUES:
                return True
            if normalized in FALSE_VALUES:
                return False
            self.errors.append(
                f"{name} must be one of: true, false, 1, 0, yes, no, on, off"
            )
            return default

        if isinstance(default, int):
            if not text:
                return default
            try:
                return int(text)
            except (TypeError, ValueError):
                self.errors.append(f"{name} must be an integer")
                return default

        if isinstance(default, float):
            if not text:
                return default
            try:
                parsed = float(text)
            except (TypeError, ValueError):
                self.errors.append(f"{name} must be a number")
                return default
            if not math.isfinite(parsed):
                self.errors.append(f"{name} must be finite")
                return default
            return parsed

        if isinstance(default, tuple):
            values = tuple(item.strip() for item in text.split(",") if item.strip())
            if name in NORMALIZED_ENV_NAMES:
                return tuple(item.lower() for item in values)
            return values

        return text.lower() if name in NORMALIZED_ENV_NAMES else text


@dataclass(frozen=True)
class RuntimeConfig:
    transcription_backend: str = _config_field("TRANSCRIPTION_BACKEND", "whisper")
    whisper_model_size: str = _config_field("WHISPER_MODEL_SIZE", "small")
    whisper_device: str = _config_field("WHISPER_DEVICE", "cuda")
    whisper_transcription_interval: float = _config_field(
        "WHISPER_TRANSCRIPTION_INTERVAL", 0.8
    )
    whisper_min_audio_seconds: float = _config_field(
        "WHISPER_MIN_AUDIO_SECONDS", 0.8
    )
    whisper_max_audio_seconds: float = _config_field(
        "WHISPER_MAX_AUDIO_SECONDS", 6.0
    )
    whisper_beam_size: int = _config_field("WHISPER_BEAM_SIZE", 1)
    whisper_best_of: int = _config_field("WHISPER_BEST_OF", 1)
    whisper_temperature: float = _config_field("WHISPER_TEMPERATURE", 0.0)
    whisper_condition_on_previous_text: bool = _config_field(
        "WHISPER_CONDITION_ON_PREVIOUS_TEXT", False
    )
    whisper_log_latency: bool = _config_field("WHISPER_LOG_LATENCY", True)
    faster_whisper_compute_type: str = _config_field(
        "FASTER_WHISPER_COMPUTE_TYPE", "int8_float16"
    )
    faster_whisper_cpu_threads: int = _config_field(
        "FASTER_WHISPER_CPU_THREADS", 4
    )
    faster_whisper_num_workers: int = _config_field(
        "FASTER_WHISPER_NUM_WORKERS", 1
    )

    scene_memory_max_words: int = _config_field("SCENE_MEMORY_MAX_WORDS", 36)
    scene_memory_max_age_seconds: float = _config_field(
        "SCENE_MEMORY_MAX_AGE_SECONDS", 20.0
    )
    prompt_token_budget_enabled: bool = _config_field(
        "PROMPT_TOKEN_BUDGET_ENABLED", True
    )
    prompt_token_budget_fallback: str = _config_field(
        "PROMPT_TOKEN_BUDGET_FALLBACK", "conservative"
    )
    prompt_max_tokens: int = _config_field("PROMPT_MAX_TOKENS", 77)
    prompt_min_transcript_tokens: int = _config_field(
        "PROMPT_MIN_TRANSCRIPT_TOKENS", 20
    )
    prompt_log_tokens: bool = _config_field("PROMPT_LOG_TOKENS", True)
    prompt_tokenizer_models: tuple = _config_field(
        "PROMPT_TOKENIZER_MODELS",
        (
            "openai/clip-vit-large-patch14",
            "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k",
        ),
    )
    prompt_refinement_chain: tuple = _config_field(
        "PROMPT_REFINEMENT_CHAIN",
        (
            "ollama:gemini-3-flash-preview",
            "ollama:kimi-k2.6:cloud",
            "ollama:gemma4:31b-cloud",
        ),
    )
    prompt_refinement_ollama_endpoint: str = _config_field(
        "PROMPT_REFINEMENT_OLLAMA_ENDPOINT",
        "http://localhost:11434/api/generate",
    )
    prompt_refinement_capriole_endpoint: str = _config_field(
        "PROMPT_REFINEMENT_CAPRIOLE_ENDPOINT",
        "https://api.caprioletech.com/v1/chat",
    )
    prompt_refinement_request_timeout: float = _config_field(
        "PROMPT_REFINEMENT_REQUEST_TIMEOUT", 15.0
    )
    prompt_refinement_target_tokens: int = _config_field(
        "PROMPT_REFINEMENT_TARGET_TOKENS", 70
    )
    prompt_refinement_max_output_characters: int = _config_field(
        "PROMPT_REFINEMENT_MAX_OUTPUT_CHARACTERS", 1000
    )

    default_gender: str = _config_field("DEFAULT_GENDER", "neutral")
    default_age: str = _config_field("DEFAULT_AGE", "adult")
    default_visual_mode: str = _config_field(
        "DEFAULT_VISUAL_MODE", "asian_american"
    )
    default_prompt_style: str = _config_field(
        "DEFAULT_PROMPT_STYLE", "human_focus"
    )
    default_language: str = _config_field("DEFAULT_LANGUAGE", "auto")

    osc_ip: str = _config_field("OSC_IP", "127.0.0.1")
    osc_port: int = _config_field("OSC_PORT", 7000)
    osc_control_enabled: bool = _config_field("OSC_CONTROL_ENABLED", True)
    osc_control_ip: str = _config_field("OSC_CONTROL_IP", "127.0.0.1")
    osc_control_port: int = _config_field("OSC_CONTROL_PORT", 7001)
    osc_status_interval: float = _config_field("OSC_STATUS_INTERVAL", 0.5)
    osc_output_error_log_interval: float = _config_field(
        "OSC_OUTPUT_ERROR_LOG_INTERVAL", 5.0
    )
    osc_prompt_retry_base_seconds: float = _config_field(
        "OSC_PROMPT_RETRY_BASE_SECONDS", 0.5
    )
    osc_prompt_retry_max_seconds: float = _config_field(
        "OSC_PROMPT_RETRY_MAX_SECONDS", 5.0
    )

    runtime_log_level: str = _config_field("RUNTIME_LOG_LEVEL", "info")
    runtime_log_console_enabled: bool = _config_field(
        "RUNTIME_LOG_CONSOLE_ENABLED", True
    )
    runtime_log_file: str = _config_field("RUNTIME_LOG_FILE", "")
    runtime_log_max_bytes: int = _config_field(
        "RUNTIME_LOG_MAX_BYTES", 5_000_000
    )
    runtime_log_backup_count: int = _config_field(
        "RUNTIME_LOG_BACKUP_COUNT", 3
    )
    runtime_shutdown_grace_seconds: float = _config_field(
        "RUNTIME_SHUTDOWN_GRACE_SECONDS", 25.0
    )

    transcription_max_final_jobs: int = _config_field(
        "TRANSCRIPTION_MAX_FINAL_JOBS", 8
    )
    transcription_final_overflow_policy: str = _config_field(
        "TRANSCRIPTION_FINAL_OVERFLOW_POLICY", "drop_oldest"
    )
    transcription_partial_max_age_seconds: float = _config_field(
        "TRANSCRIPTION_PARTIAL_MAX_AGE_SECONDS", 4.0
    )
    transcription_final_max_age_seconds: float = _config_field(
        "TRANSCRIPTION_FINAL_MAX_AGE_SECONDS", 30.0
    )
    transcription_final_max_retries: int = _config_field(
        "TRANSCRIPTION_FINAL_MAX_RETRIES", 2
    )
    transcription_retry_base_seconds: float = _config_field(
        "TRANSCRIPTION_RETRY_BASE_SECONDS", 1.0
    )
    transcription_retry_max_seconds: float = _config_field(
        "TRANSCRIPTION_RETRY_MAX_SECONDS", 10.0
    )

    groq_api_key: str = _config_field("GROQ_API_KEY", "", secret=True)
    groq_transcription_model: str = _config_field(
        "GROQ_TRANSCRIPTION_MODEL", "whisper-large-v3"
    )
    groq_hybrid_model: str = _config_field(
        "GROQ_HYBRID_MODEL", "whisper-large-v3-turbo"
    )
    groq_text_translation_model: str = _config_field(
        "GROQ_TEXT_TRANSLATION_MODEL", "llama-3.1-8b-instant"
    )
    groq_transcriptions_endpoint: str = _config_field(
        "GROQ_TRANSCRIPTIONS_ENDPOINT",
        "https://api.groq.com/openai/v1/audio/transcriptions",
    )
    groq_translations_endpoint: str = _config_field(
        "GROQ_TRANSLATIONS_ENDPOINT",
        "https://api.groq.com/openai/v1/audio/translations",
    )
    groq_chat_endpoint: str = _config_field(
        "GROQ_CHAT_ENDPOINT",
        "https://api.groq.com/openai/v1/chat/completions",
    )
    groq_translation_prompt: str = _config_field(
        "GROQ_TRANSLATION_PROMPT",
        "Translate all speech to natural concise English for a visual generation prompt.",
    )
    groq_response_format: str = _config_field("GROQ_RESPONSE_FORMAT", "text")
    groq_transcription_interval: float = _config_field(
        "GROQ_TRANSCRIPTION_INTERVAL", 3.2
    )
    groq_min_audio_seconds: float = _config_field("GROQ_MIN_AUDIO_SECONDS", 1.0)
    groq_max_audio_seconds: float = _config_field("GROQ_MAX_AUDIO_SECONDS", 6.0)
    groq_request_timeout: float = _config_field("GROQ_REQUEST_TIMEOUT", 20.0)
    groq_log_latency: bool = _config_field("GROQ_LOG_LATENCY", True)
    groq_english_fallback: str = _config_field("GROQ_ENGLISH_FALLBACK", "auto")

    local_translator: str = _config_field("LOCAL_TRANSLATOR", "argos")
    local_translator_target_language: str = _config_field(
        "LOCAL_TRANSLATOR_TARGET_LANGUAGE", "en"
    )
    local_translator_default_source_language: str = _config_field(
        "LOCAL_TRANSLATOR_DEFAULT_SOURCE_LANGUAGE", "zh"
    )
    local_translator_auto_install: bool = _config_field(
        "LOCAL_TRANSLATOR_AUTO_INSTALL", True
    )
    local_translator_preload_languages: tuple = _config_field(
        "LOCAL_TRANSLATOR_PRELOAD_LANGUAGES", ("zh", "es")
    )
    local_translator_log_latency: bool = _config_field(
        "LOCAL_TRANSLATOR_LOG_LATENCY", True
    )
    hybrid_translation_fallback: str = _config_field(
        "HYBRID_TRANSLATION_FALLBACK", "groq_text"
    )

    google_transcription_interval: float = _config_field(
        "GOOGLE_TRANSCRIPTION_INTERVAL", 2.0
    )
    google_min_audio_seconds: float = _config_field(
        "GOOGLE_MIN_AUDIO_SECONDS", 1.5
    )
    google_max_audio_seconds: float = _config_field(
        "GOOGLE_MAX_AUDIO_SECONDS", 5.0
    )
    google_speech_language: str = _config_field(
        "GOOGLE_SPEECH_LANGUAGE", "en-US"
    )
    google_speech_english_language: str = _config_field(
        "GOOGLE_SPEECH_ENGLISH_LANGUAGE", "en-US"
    )
    google_speech_chinese_language: str = _config_field(
        "GOOGLE_SPEECH_CHINESE_LANGUAGE", "zh-CN"
    )
    google_speech_spanish_language: str = _config_field(
        "GOOGLE_SPEECH_SPANISH_LANGUAGE", "es-ES"
    )

    audio_input_device_index: Optional[int] = _config_field(
        "AUDIO_INPUT_DEVICE_INDEX", None
    )
    vad_engine: str = _config_field("VAD_ENGINE", "silero")
    vad_threshold: float = _config_field("VAD_THRESHOLD", 0.5)
    vad_energy_threshold: float = _config_field("VAD_ENERGY_THRESHOLD", 400.0)
    vad_pre_roll_seconds: float = _config_field("VAD_PRE_ROLL_SECONDS", 0.32)
    vad_silence_seconds: float = _config_field("VAD_SILENCE_SECONDS", 0.7)
    stream_overlap_seconds: float = _config_field("STREAM_OVERLAP_SECONDS", 0.5)
    transcript_confirm_updates: int = _config_field(
        "TRANSCRIPT_CONFIRM_UPDATES", 2
    )
    audio_reconnect_enabled: bool = _config_field(
        "AUDIO_RECONNECT_ENABLED", True
    )
    audio_reconnect_base_seconds: float = _config_field(
        "AUDIO_RECONNECT_BASE_SECONDS", 0.5
    )
    audio_reconnect_max_seconds: float = _config_field(
        "AUDIO_RECONNECT_MAX_SECONDS", 8.0
    )
    audio_max_consecutive_read_errors: int = _config_field(
        "AUDIO_MAX_CONSECUTIVE_READ_ERRORS", 3
    )
    audio_read_retry_seconds: float = _config_field(
        "AUDIO_READ_RETRY_SECONDS", 0.1
    )

    capriole_api_key: str = _config_field("CAPRIOLE_API_KEY", "", secret=True)

    @classmethod
    def from_environment(
        cls,
        values: Optional[Mapping[str, str]] = None,
        *,
        backend_override=None,
        input_device_override=None,
        gender_override=None,
        age_override=None,
        visual_mode_override=None,
        prompt_style_override=None,
        language_override=None,
    ):
        source = dict(os.environ if values is None else values)
        legacy_interval = source.get("ONLINE_TRANSCRIPTION_INTERVAL")
        if legacy_interval is not None and str(legacy_interval).strip():
            source.setdefault("GROQ_TRANSCRIPTION_INTERVAL", legacy_interval)
            source.setdefault("GOOGLE_TRANSCRIPTION_INTERVAL", legacy_interval)
        legacy_min_audio = source.get("ONLINE_MIN_AUDIO_SECONDS")
        if legacy_min_audio is not None and str(legacy_min_audio).strip():
            source.setdefault("GROQ_MIN_AUDIO_SECONDS", legacy_min_audio)
            source.setdefault("GOOGLE_MIN_AUDIO_SECONDS", legacy_min_audio)
        if backend_override is not None:
            source["TRANSCRIPTION_BACKEND"] = backend_override
        if input_device_override is not None:
            source["AUDIO_INPUT_DEVICE_INDEX"] = input_device_override
        for name, value in {
            "DEFAULT_GENDER": gender_override,
            "DEFAULT_AGE": age_override,
            "DEFAULT_VISUAL_MODE": visual_mode_override,
            "DEFAULT_PROMPT_STYLE": prompt_style_override,
            "DEFAULT_LANGUAGE": language_override,
        }.items():
            if value is not None:
                source[name] = value

        reader = _EnvironmentReader(source)
        defaults = cls()
        parsed = {
            config_field.name: reader.value(
                config_field.metadata["env"],
                getattr(defaults, config_field.name),
            )
            for config_field in fields(defaults)
        }
        config = cls(**parsed)

        errors = list(reader.errors)
        errors.extend(config.validation_errors())
        if errors:
            raise ConfigError(errors)
        return config

    def validation_errors(self):
        errors = []

        if self.transcription_backend not in SUPPORTED_BACKENDS:
            errors.append(
                "TRANSCRIPTION_BACKEND must be one of: "
                + ", ".join(sorted(SUPPORTED_BACKENDS))
            )
        if not self.whisper_model_size:
            errors.append("WHISPER_MODEL_SIZE must not be empty")
        if (
            self.transcription_backend in {"whisper", "faster_whisper"}
            and self.whisper_device
            and not self.whisper_device.startswith("cuda")
        ):
            errors.append(
                f"TRANSCRIPTION_BACKEND={self.transcription_backend} requires "
                "WHISPER_DEVICE to select CUDA"
            )
        if not self.faster_whisper_compute_type:
            errors.append("FASTER_WHISPER_COMPUTE_TYPE must not be empty")

        self._validate_choice(
            errors,
            "DEFAULT_GENDER",
            self.default_gender,
            SUPPORTED_GENDERS,
        )
        self._validate_choice(
            errors,
            "DEFAULT_AGE",
            self.default_age,
            SUPPORTED_AGES,
        )
        self._validate_choice(
            errors,
            "DEFAULT_VISUAL_MODE",
            self.default_visual_mode,
            SUPPORTED_VISUAL_MODES,
        )
        self._validate_choice(
            errors,
            "DEFAULT_PROMPT_STYLE",
            self.default_prompt_style,
            SUPPORTED_PROMPT_STYLES,
        )
        self._validate_choice(
            errors,
            "DEFAULT_LANGUAGE",
            self.default_language,
            SUPPORTED_LANGUAGES,
        )
        self._validate_choice(
            errors,
            "TRANSCRIPTION_FINAL_OVERFLOW_POLICY",
            self.transcription_final_overflow_policy,
            SUPPORTED_FINAL_OVERFLOW_POLICIES,
        )

        self._require_positive(
            errors,
            {
                "WHISPER_TRANSCRIPTION_INTERVAL": self.whisper_transcription_interval,
                "WHISPER_MIN_AUDIO_SECONDS": self.whisper_min_audio_seconds,
                "WHISPER_MAX_AUDIO_SECONDS": self.whisper_max_audio_seconds,
                "FASTER_WHISPER_CPU_THREADS": self.faster_whisper_cpu_threads,
                "FASTER_WHISPER_NUM_WORKERS": self.faster_whisper_num_workers,
                "PROMPT_MAX_TOKENS": self.prompt_max_tokens,
                "PROMPT_REFINEMENT_REQUEST_TIMEOUT": (
                    self.prompt_refinement_request_timeout
                ),
                "PROMPT_REFINEMENT_TARGET_TOKENS": (
                    self.prompt_refinement_target_tokens
                ),
                "PROMPT_REFINEMENT_MAX_OUTPUT_CHARACTERS": (
                    self.prompt_refinement_max_output_characters
                ),
                "OSC_STATUS_INTERVAL": self.osc_status_interval,
                "OSC_OUTPUT_ERROR_LOG_INTERVAL": (
                    self.osc_output_error_log_interval
                ),
                "OSC_PROMPT_RETRY_BASE_SECONDS": self.osc_prompt_retry_base_seconds,
                "OSC_PROMPT_RETRY_MAX_SECONDS": self.osc_prompt_retry_max_seconds,
                "RUNTIME_LOG_MAX_BYTES": self.runtime_log_max_bytes,
                "RUNTIME_LOG_BACKUP_COUNT": self.runtime_log_backup_count,
                "RUNTIME_SHUTDOWN_GRACE_SECONDS": (
                    self.runtime_shutdown_grace_seconds
                ),
                "TRANSCRIPTION_MAX_FINAL_JOBS": self.transcription_max_final_jobs,
                "TRANSCRIPTION_RETRY_BASE_SECONDS": self.transcription_retry_base_seconds,
                "TRANSCRIPTION_RETRY_MAX_SECONDS": self.transcription_retry_max_seconds,
                "GROQ_TRANSCRIPTION_INTERVAL": self.groq_transcription_interval,
                "GROQ_MIN_AUDIO_SECONDS": self.groq_min_audio_seconds,
                "GROQ_MAX_AUDIO_SECONDS": self.groq_max_audio_seconds,
                "GROQ_REQUEST_TIMEOUT": self.groq_request_timeout,
                "GOOGLE_TRANSCRIPTION_INTERVAL": self.google_transcription_interval,
                "GOOGLE_MIN_AUDIO_SECONDS": self.google_min_audio_seconds,
                "GOOGLE_MAX_AUDIO_SECONDS": self.google_max_audio_seconds,
                "VAD_ENERGY_THRESHOLD": self.vad_energy_threshold,
                "VAD_SILENCE_SECONDS": self.vad_silence_seconds,
                "TRANSCRIPT_CONFIRM_UPDATES": self.transcript_confirm_updates,
                "AUDIO_RECONNECT_BASE_SECONDS": self.audio_reconnect_base_seconds,
                "AUDIO_RECONNECT_MAX_SECONDS": self.audio_reconnect_max_seconds,
                "AUDIO_MAX_CONSECUTIVE_READ_ERRORS": self.audio_max_consecutive_read_errors,
            },
        )
        self._require_non_negative(
            errors,
            {
                "WHISPER_TEMPERATURE": self.whisper_temperature,
                "SCENE_MEMORY_MAX_AGE_SECONDS": self.scene_memory_max_age_seconds,
                "PROMPT_MIN_TRANSCRIPT_TOKENS": self.prompt_min_transcript_tokens,
                "TRANSCRIPTION_PARTIAL_MAX_AGE_SECONDS": self.transcription_partial_max_age_seconds,
                "TRANSCRIPTION_FINAL_MAX_AGE_SECONDS": self.transcription_final_max_age_seconds,
                "TRANSCRIPTION_FINAL_MAX_RETRIES": self.transcription_final_max_retries,
                "VAD_PRE_ROLL_SECONDS": self.vad_pre_roll_seconds,
                "STREAM_OVERLAP_SECONDS": self.stream_overlap_seconds,
                "AUDIO_READ_RETRY_SECONDS": self.audio_read_retry_seconds,
            },
        )
        if self.whisper_beam_size < 1:
            errors.append("WHISPER_BEAM_SIZE must be at least 1")
        if self.whisper_best_of < 1:
            errors.append("WHISPER_BEST_OF must be at least 1")
        if self.scene_memory_max_words < 1:
            errors.append("SCENE_MEMORY_MAX_WORDS must be at least 1")
        if not 0.0 <= self.vad_threshold <= 1.0:
            errors.append("VAD_THRESHOLD must be between 0 and 1")
        if self.vad_engine not in {"silero", "energy"}:
            errors.append("VAD_ENGINE must be one of: energy, silero")
        if self.runtime_log_level not in {
            "debug",
            "info",
            "warning",
            "error",
            "critical",
        }:
            errors.append(
                "RUNTIME_LOG_LEVEL must be one of: "
                "critical, debug, error, info, warning"
            )
        if self.groq_response_format not in {"json", "text", "verbose_json"}:
            errors.append(
                "GROQ_RESPONSE_FORMAT must be one of: json, text, verbose_json"
            )
        if self.local_translator not in {"argos", "none", "off"}:
            errors.append("LOCAL_TRANSLATOR must be one of: argos, none, off")
        if self.prompt_min_transcript_tokens > self.prompt_max_tokens:
            errors.append(
                "PROMPT_MIN_TRANSCRIPT_TOKENS must not exceed PROMPT_MAX_TOKENS"
            )
        if self.prompt_max_tokens < 2:
            errors.append("PROMPT_MAX_TOKENS must be at least 2")
        if self.prompt_token_budget_enabled and not self.prompt_tokenizer_models:
            errors.append(
                "PROMPT_TOKENIZER_MODELS must include at least one model when "
                "PROMPT_TOKEN_BUDGET_ENABLED is true"
            )
        self._validate_choice(
            errors,
            "PROMPT_TOKEN_BUDGET_FALLBACK",
            self.prompt_token_budget_fallback,
            SUPPORTED_PROMPT_BUDGET_FALLBACKS,
        )

        refinement_providers = set()
        if not self.prompt_refinement_chain:
            errors.append(
                "PROMPT_REFINEMENT_CHAIN must include at least one provider:model entry"
            )
        for route in self.prompt_refinement_chain:
            provider, separator, model = route.partition(":")
            provider = provider.strip().lower()
            model = model.strip()
            if not separator or not provider or not model:
                errors.append(
                    "PROMPT_REFINEMENT_CHAIN entries must use provider:model"
                )
                continue
            if provider not in SUPPORTED_PROMPT_REFINEMENT_PROVIDERS:
                errors.append(
                    "PROMPT_REFINEMENT_CHAIN provider must be one of: "
                    + ", ".join(sorted(SUPPORTED_PROMPT_REFINEMENT_PROVIDERS))
                )
                continue
            refinement_providers.add(provider)

        self._validate_port(errors, "OSC_PORT", self.osc_port)
        self._validate_port(errors, "OSC_CONTROL_PORT", self.osc_control_port)
        self._validate_order(
            errors,
            "OSC_PROMPT_RETRY_BASE_SECONDS",
            self.osc_prompt_retry_base_seconds,
            "OSC_PROMPT_RETRY_MAX_SECONDS",
            self.osc_prompt_retry_max_seconds,
        )
        self._validate_order(
            errors,
            "WHISPER_MIN_AUDIO_SECONDS",
            self.whisper_min_audio_seconds,
            "WHISPER_MAX_AUDIO_SECONDS",
            self.whisper_max_audio_seconds,
        )
        self._validate_order(
            errors,
            "GROQ_MIN_AUDIO_SECONDS",
            self.groq_min_audio_seconds,
            "GROQ_MAX_AUDIO_SECONDS",
            self.groq_max_audio_seconds,
        )
        self._validate_order(
            errors,
            "GOOGLE_MIN_AUDIO_SECONDS",
            self.google_min_audio_seconds,
            "GOOGLE_MAX_AUDIO_SECONDS",
            self.google_max_audio_seconds,
        )
        self._validate_order(
            errors,
            "TRANSCRIPTION_RETRY_BASE_SECONDS",
            self.transcription_retry_base_seconds,
            "TRANSCRIPTION_RETRY_MAX_SECONDS",
            self.transcription_retry_max_seconds,
        )
        self._validate_order(
            errors,
            "AUDIO_RECONNECT_BASE_SECONDS",
            self.audio_reconnect_base_seconds,
            "AUDIO_RECONNECT_MAX_SECONDS",
            self.audio_reconnect_max_seconds,
        )
        active_max_audio = {
            "whisper": self.whisper_max_audio_seconds,
            "faster_whisper": self.whisper_max_audio_seconds,
            "groq": self.groq_max_audio_seconds,
            "groq_hybrid": self.groq_max_audio_seconds,
            "google": self.google_max_audio_seconds,
        }.get(self.transcription_backend)
        if (
            active_max_audio is not None
            and self.stream_overlap_seconds >= active_max_audio
        ):
            errors.append(
                "STREAM_OVERLAP_SECONDS must be less than the active backend's "
                "maximum audio duration"
            )

        for name, value in {
            "GROQ_TRANSCRIPTIONS_ENDPOINT": self.groq_transcriptions_endpoint,
            "GROQ_TRANSLATIONS_ENDPOINT": self.groq_translations_endpoint,
            "GROQ_CHAT_ENDPOINT": self.groq_chat_endpoint,
            "PROMPT_REFINEMENT_OLLAMA_ENDPOINT": (
                self.prompt_refinement_ollama_endpoint
            ),
            "PROMPT_REFINEMENT_CAPRIOLE_ENDPOINT": (
                self.prompt_refinement_capriole_endpoint
            ),
        }.items():
            parsed = urlparse(value)
            if parsed.scheme not in {"http", "https"} or not parsed.netloc:
                errors.append(f"{name} must be an absolute HTTP(S) URL")

        if self.groq_english_fallback not in {"auto", "always", "off"}:
            errors.append("GROQ_ENGLISH_FALLBACK must be one of: always, auto, off")
        if self.hybrid_translation_fallback not in {"groq_text", "off"}:
            errors.append(
                "HYBRID_TRANSLATION_FALLBACK must be one of: groq_text, off"
            )
        if (
            self.transcription_backend in {"groq", "groq_hybrid"}
            and not self.is_secret_configured(self.groq_api_key)
        ):
            errors.append(
                f"TRANSCRIPTION_BACKEND={self.transcription_backend} requires GROQ_API_KEY"
            )
        if (
            "capriole" in refinement_providers
            and not self.is_secret_configured(self.capriole_api_key)
        ):
            errors.append(
                "PROMPT_REFINEMENT_CHAIN with capriole requires CAPRIOLE_API_KEY"
            )
        return errors

    def redacted_items(self):
        items = []
        for config_field in fields(self):
            env_name = config_field.metadata.get("env")
            if not env_name:
                continue
            value = getattr(self, config_field.name)
            if config_field.metadata.get("secret"):
                display = (
                    "<redacted>"
                    if self.is_secret_configured(value)
                    else "<not configured>"
                )
            elif value is None:
                display = "<system default>"
            elif isinstance(value, bool):
                display = str(value).lower()
            elif isinstance(value, tuple):
                display = ",".join(str(item) for item in value)
            elif env_name == "RUNTIME_LOG_FILE" and value == "":
                display = "<disabled>"
            elif value == "":
                display = "<auto>"
            else:
                display = str(value)
            items.append((env_name, display))
        return sorted(items)

    @staticmethod
    def is_secret_configured(value):
        normalized = str(value or "").strip().lower()
        return bool(normalized) and normalized not in {
            "your_key_here",
            "your_groq_key_here",
            "your_capriole_key_here",
        }

    @staticmethod
    def _require_positive(errors, values):
        for name, value in values.items():
            if value <= 0:
                errors.append(f"{name} must be greater than 0")

    @staticmethod
    def _require_non_negative(errors, values):
        for name, value in values.items():
            if value < 0:
                errors.append(f"{name} must be 0 or greater")

    @staticmethod
    def _validate_port(errors, name, value):
        if not 1 <= value <= 65535:
            errors.append(f"{name} must be between 1 and 65535")

    @staticmethod
    def _validate_choice(errors, name, value, choices):
        if value not in choices:
            errors.append(
                f"{name} must be one of: " + ", ".join(sorted(choices))
            )

    @staticmethod
    def _validate_order(errors, min_name, minimum, max_name, maximum):
        if minimum > maximum:
            errors.append(f"{min_name} must not exceed {max_name}")


def load_env_file(path=".env", environ=None):
    target = os.environ if environ is None else environ
    env_path = Path(path)
    if not env_path.exists():
        return

    with env_path.open("r", encoding="utf-8") as env_file:
        for line in env_file:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            target.setdefault(
                key.strip(),
                value.strip().strip('"').strip("'"),
            )


def format_config_report(config):
    width = max(len(name) for name, _value in config.redacted_items())
    lines = [
        "VOICE-TO-VISUAL EFFECTIVE CONFIGURATION",
        "=" * 72,
    ]
    lines.extend(
        f"{name:<{width}} = {value}"
        for name, value in config.redacted_items()
    )
    lines.extend(
        [
            "=" * 72,
            "Configuration is valid. Secret values are redacted.",
        ]
    )
    return "\n".join(lines)


def format_config_error(error):
    lines = [
        "VOICE-TO-VISUAL CONFIGURATION ERRORS",
        "=" * 72,
    ]
    lines.extend(f"- {message}" for message in error.errors)
    lines.extend(
        [
            "=" * 72,
            "Fix the values in .env or the current environment and try again.",
        ]
    )
    return "\n".join(lines)
