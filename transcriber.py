import os
import argparse
import numpy as np
import threading
import time
import msvcrt

from audio_runtime import (
    EnergyVoiceActivityDetector,
    SileroVoiceActivityDetector,
)
from audio_sources import (
    AudioSourceFinished,
    AudioSourceStopped,
    PyAudioSource,
    WavReplaySource,
    list_system_audio_input_devices,
    load_wav_samples,
)
from backend_errors import RetryableTranscriptionError, exponential_backoff
from diagnostics import run_diagnostics
from osc_control import OscControlServer
from osc_output import (
    NullOutputPublisher,
    OscOutputPublisher,
    RuntimeStatusSnapshot,
)
from prompt_engine import (
    ConservativeUtf8Tokenizer,
    PromptBudgeter,
    RollingSceneMemory,
)
from runtime_config import (
    ConfigError,
    RuntimeConfig,
    SUPPORTED_AGES,
    SUPPORTED_GENDERS,
    SUPPORTED_LANGUAGES,
    SUPPORTED_PROMPT_STYLES,
    SUPPORTED_VISUAL_MODES,
    format_config_error,
    format_config_report,
    load_env_file,
)
from runtime_logging import RuntimeLogSession
from runtime_scheduler import RealtimeJobScheduler
from streaming_core import AudioSegmenter, TranscriptStabilizer
from transcript_filter import is_probable_whisper_hallucination
from transcription_backends import create_transcription_backend

def optional_non_negative_int(value):
    if value is None or str(value).strip() == "":
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("must be a non-negative integer") from exc
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be a non-negative integer")
    return parsed


def parse_args(args=None):
    parser = argparse.ArgumentParser(description="Live speech-to-visual prompt bridge for StreamDiffusionTD.")
    parser.add_argument(
        "-b",
        "--backend",
        choices=["whisper", "faster_whisper", "groq", "groq_hybrid", "google"],
        help="Transcription backend. Overrides TRANSCRIPTION_BACKEND from .env for this run."
    )
    parser.add_argument(
        "--input-device",
        type=optional_non_negative_int,
        metavar="INDEX",
        help="PyAudio input-device index. Overrides AUDIO_INPUT_DEVICE_INDEX from .env."
    )
    startup_controls = parser.add_argument_group("startup controls")
    startup_controls.add_argument(
        "--gender",
        choices=sorted(SUPPORTED_GENDERS),
        help="Initial gender mode. Overrides DEFAULT_GENDER for this run.",
    )
    startup_controls.add_argument(
        "--age",
        choices=sorted(SUPPORTED_AGES),
        help="Initial age mode. Overrides DEFAULT_AGE for this run.",
    )
    startup_controls.add_argument(
        "--visual-mode",
        choices=sorted(SUPPORTED_VISUAL_MODES),
        help=(
            "Initial visual identity mode. Overrides DEFAULT_VISUAL_MODE "
            "for this run."
        ),
    )
    startup_controls.add_argument(
        "--prompt-style",
        choices=sorted(SUPPORTED_PROMPT_STYLES),
        help=(
            "Initial prompt style. Overrides DEFAULT_PROMPT_STYLE for this run."
        ),
    )
    startup_controls.add_argument(
        "--language",
        choices=sorted(SUPPORTED_LANGUAGES),
        help=(
            "Initial transcription language. Overrides DEFAULT_LANGUAGE "
            "for this run."
        ),
    )
    parser.add_argument(
        "--list-audio-devices",
        action="store_true",
        help="List input-capable audio devices and exit."
    )
    runtime_mode = parser.add_mutually_exclusive_group()
    runtime_mode.add_argument(
        "--benchmark",
        metavar="WAV_PATH",
        help="Benchmark a local backend with a PCM WAV file instead of opening the microphone."
    )
    runtime_mode.add_argument(
        "--replay",
        metavar="WAV_PATH",
        help="Replay a PCM WAV file through the live segmentation, transcription, prompt, logging, and OSC pipeline."
    )
    parser.add_argument(
        "--benchmark-runs",
        type=int,
        default=3,
        help="Number of measured benchmark runs after warm-up (default: 3)."
    )
    parser.add_argument(
        "--diagnose",
        action="store_true",
        help="Check the GPU, microphone, packages, OSC port, model cache, and credential hygiene."
    )
    parser.add_argument(
        "--check-config",
        action="store_true",
        help="Validate and print the effective configuration with secrets redacted."
    )
    parsed = parser.parse_args(args)
    if parsed.replay and parsed.input_device is not None:
        parser.error("--input-device cannot be used with --replay")
    return parsed


def load_runtime_config(args=None):
    load_env_file()
    return RuntimeConfig.from_environment(
        backend_override=getattr(args, "backend", None),
        input_device_override=getattr(args, "input_device", None),
        gender_override=getattr(args, "gender", None),
        age_override=getattr(args, "age", None),
        visual_mode_override=getattr(args, "visual_mode", None),
        prompt_style_override=getattr(args, "prompt_style", None),
        language_override=getattr(args, "language", None),
    )


def configure_dependency_environment():
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

# --- FIXED PROMPT STRATEGY ---
# GENDER MODES: Press 'm' for Man, 'w' for Woman, 'n' for Neutral (General)
GENDER_MODES = {
    "man": "man",
    "woman": "woman",
    "neutral": "person"
}

# AGE MODES: Press '1' for Young, '2' for Adult, '3' for Elder
AGE_MODES = {
    "young": "young",
    "adult": "adult",
    "elder": "elderly"
}

VISUAL_MODES = {
    "asian_american": {
        "label": "ASIAN AMERICAN",
        "subject_prefix": "Asian-American",
        "context": "capturing a diverse Asian-American identity, blending modern US urban settings with subtle traditional Asian cultural motifs and textures",
        "scene_context": "modern Asian-American neighborhoods and interiors, subtle traditional Asian cultural motifs, layered urban textures, natural cinematic atmosphere",
        "compact_context": "Asian-American US setting with subtle Asian cultural motifs",
        "compact_scene_context": "Asian-American US setting with subtle Asian cultural motifs and cinematic atmosphere",
        "minimal_scene_context": "Asian-American setting",
    },
    "black_brown": {
        "label": "BLACK AND BROWN PEOPLE",
        "subject_prefix": "Black or Brown",
        "context": "centering Black and Brown people, contemporary US urban life, rich diasporic cultural textures, warm natural skin tones, dignified and vibrant representation",
        "scene_context": "contemporary US neighborhoods shaped by Black and Brown diasporic culture, warm natural color palettes, rich textures, vibrant lived-in atmosphere",
        "compact_context": "Black and Brown diasporic US setting with warm tones and rich textures",
        "compact_scene_context": "Black and Brown diasporic US setting with warm colors and rich textures",
        "minimal_scene_context": "Black and Brown setting",
    },
    "asian_black_brown": {
        "label": "ASIAN + BLACK AND BROWN PEOPLE",
        "subject_prefix": "Asian, Black, or Brown",
        "context": "centering Asian, Black, and Brown people together, diverse contemporary US community life, layered diasporic cultural textures, warm natural skin tones, dignified and vibrant representation",
        "scene_context": "diverse contemporary US community spaces shaped by Asian, Black, and Brown diasporic culture, layered cultural textures, vibrant lived-in atmosphere",
        "compact_context": "diverse Asian, Black, and Brown diasporic US community",
        "compact_scene_context": "diverse Asian, Black, and Brown diasporic US community spaces",
        "minimal_scene_context": "Asian, Black, and Brown community",
    },
}

FIXED_PROMPT_TEMPLATE = "A hyper-realistic photorealistic cinematic shot of {text} featuring a prominent {age_desc} {subject_focus}, {visual_context}, 8k UHD, highly detailed, masterfully lit, RAW photo, shot on 35mm lens, f/1.8, natural colors, masterpiece"
SCENE_PROMPT_TEMPLATE = "A hyper-realistic photorealistic cinematic scene of {text}, {visual_context}, environment-focused composition, no central human figure, no portrait framing, 8k UHD, highly detailed, masterfully lit, RAW photo, shot on 35mm lens, f/1.8, natural colors, masterpiece"
COMPACT_FIXED_PROMPT_TEMPLATE = "Photorealistic cinematic scene: {text}, prominent {age_desc} {subject_focus}, {visual_context}, highly detailed, natural colors, masterfully lit, RAW 35mm photo, f/1.8, 8k"
COMPACT_SCENE_PROMPT_TEMPLATE = "Photorealistic cinematic scene: {text}, {visual_context}, environment-focused, no central human figure, highly detailed, natural colors, masterfully lit, RAW 35mm photo, f/1.8, 8k"
MINIMAL_FIXED_PROMPT_TEMPLATE = "Cinematic {text}, {age_desc} {subject_focus}"
MINIMAL_SCENE_PROMPT_TEMPLATE = "Cinematic {text}, {visual_context}"

PROMPT_STYLES = {
    "human_focus": {
        "label": "HUMAN FIGURE",
        "template": FIXED_PROMPT_TEMPLATE,
    },
    "general_scene": {
        "label": "GENERAL SCENE",
        "template": SCENE_PROMPT_TEMPLATE,
    },
}

KEYBOARD_CONTROLS = {
    "m": ("gender", "man"),
    "w": ("gender", "woman"),
    "n": ("gender", "neutral"),
    "1": ("age", "young"),
    "2": ("age", "adult"),
    "3": ("age", "elder"),
    "d": ("visual_mode", "asian_american"),
    "b": ("visual_mode", "black_brown"),
    "x": ("visual_mode", "asian_black_brown"),
    "f": ("prompt_style", "human_focus"),
    "g": ("prompt_style", "general_scene"),
    "e": ("language", "en"),
    "c": ("language", "zh"),
    "s": ("language", "es"),
    "a": ("language", None),
}

# Audio recording constants
CHUNK = 1024
CHANNELS = 1
RATE = 16000


def print_audio_input_devices():
    try:
        devices = list_system_audio_input_devices()
        if not devices:
            print("No input-capable audio devices were found.")
            return 1

        print("\nINPUT AUDIO DEVICES")
        print("=" * 72)
        for device in devices:
            marker = "*" if device.is_default else " "
            rate = f"{device.default_sample_rate:.0f} Hz" if device.default_sample_rate else "unknown rate"
            print(
                f"{marker} [{device.index}] {device.name} "
                f"({device.max_input_channels} input channel(s), {rate})"
            )
        print("=" * 72)
        print("* system default")
        return 0
    except Exception as exc:
        print(f"Could not enumerate audio devices: {exc}")
        return 1


class RealTimePipeline:
    def __init__(
        self,
        enable_vad=True,
        enable_osc=True,
        enable_prompt_budget=True,
        enable_osc_controls=True,
        config=None,
        backend_adapter=None,
        log_session=None,
        audio_source=None,
        output_publisher=None,
    ):
        configure_dependency_environment()
        self.config = (
            config if config is not None else load_runtime_config()
        )
        self.log_session = (
            log_session
            if log_session is not None
            else RuntimeLogSession(self.config)
        )
        self._owns_log_session = log_session is None
        self.runtime_logger = self.log_session.logger("runtime")
        self.audio_logger = self.log_session.logger("audio")
        self.scheduler_logger = self.log_session.logger("scheduler")
        self.transcription_logger = self.log_session.logger("transcription")
        self.prompt_logger = self.log_session.logger("prompt")
        self.osc_logger = self.log_session.logger("osc")
        self.control_logger = self.log_session.logger("control")
        self.backend_logger = self.log_session.logger("backend")
        try:
            self.backend_adapter = (
                backend_adapter
                if backend_adapter is not None
                else create_transcription_backend(
                    self.config,
                    sample_rate=RATE,
                    logger=self.backend_logger.info,
                )
            )
        except Exception as exc:
            self.runtime_logger.error(
                "Transcription backend initialization failed",
                extra={
                    "event": "backend_initialization_error",
                    "backend": self.config.transcription_backend,
                    "error": self.clean_audio_error(exc),
                },
            )
            if self._owns_log_session:
                self.log_session.close()
            raise
        self._backend_closed = False
        self.stop_event = threading.Event()
        self._worker_lock = threading.Lock()
        self._close_lock = threading.Lock()
        self._worker_threads = {}
        self._workers_started = False
        self.audio_source = (
            audio_source
            if audio_source is not None
            else PyAudioSource(
                device_index=self.config.audio_input_device_index,
                sample_rate=RATE,
                chunk_samples=CHUNK,
                channels=CHANNELS,
            )
        )
        if hasattr(self.audio_source, "stop_event"):
            self.audio_source.stop_event = self.stop_event
        self.audio_source_finished = threading.Event()
        self.backend = self.backend_adapter.name
        self.last_online_request_time = 0
        self.last_whisper_request_time = 0
        self.backend_retry_not_before = 0.0

        self.output_publisher = (
            output_publisher
            if output_publisher is not None
            else (
                OscOutputPublisher(
                    self.config.osc_ip,
                    self.config.osc_port,
                    status_interval=self.config.osc_status_interval,
                    error_log_interval=(
                        self.config.osc_output_error_log_interval
                    ),
                    logger=self.osc_logger,
                )
                if enable_osc
                else NullOutputPublisher()
            )
        )
        self.osc_control_enabled = bool(
            enable_osc_controls and self.config.osc_control_enabled
        )
        self.osc_control_server = None
        self.vad = self.create_voice_activity_detector() if enable_vad else None
        self.segmenter = AudioSegmenter(
            sample_rate=RATE,
            chunk_samples=CHUNK,
            pre_roll_seconds=self.config.vad_pre_roll_seconds,
            end_silence_seconds=self.config.vad_silence_seconds,
            max_segment_seconds=self.maximum_audio_seconds(),
            overlap_seconds=max(
                0.0,
                min(
                    self.config.stream_overlap_seconds,
                    self.maximum_audio_seconds() / 2,
                ),
            ),
        )
        self.scheduler = RealtimeJobScheduler(
            max_final_jobs=self.config.transcription_max_final_jobs,
            final_overflow_policy=(
                self.config.transcription_final_overflow_policy
            ),
            partial_max_age_seconds=(
                self.config.transcription_partial_max_age_seconds
            ),
            final_max_age_seconds=(
                self.config.transcription_final_max_age_seconds
            ),
        )
        self.stabilizers = {}
        self.scene_memory = RollingSceneMemory(
            max_words=self.config.scene_memory_max_words,
            max_age_seconds=self.config.scene_memory_max_age_seconds,
        )
        self.prompt_budget_mode = "disabled"
        self.prompt_budgeter = (
            self.create_prompt_budgeter()
            if enable_prompt_budget
            else None
        )
        self.last_prompt_token_count = 0
        self.last_prompt_variant = "unbudgeted"
        self.last_prompt_trimmed = False
        self.last_text = ""
        self.is_speaking = False
        self.audio_status = "starting"
        self.audio_reconnects = 0
        self.last_audio_error = ""
        self.audio_device_index = self.audio_source.device_index
        self.audio_device_name = self.audio_source.name
        self.backend_status = "ready"
        self.last_inference_latency = 0.0
        self.last_total_latency = 0.0
        self.current_gender = self.config.default_gender
        self.current_age = self.config.default_age
        self.current_visual_mode = self.config.default_visual_mode
        self.current_prompt_style = self.config.default_prompt_style
        self.current_language = (
            None
            if self.config.default_language == "auto"
            else self.config.default_language
        )
        self.lock = threading.Lock()
        self.state_lock = threading.Lock()
        # Acquire scene_lock before the segmenter lock or scheduler lock.
        # Reentrancy lets scene updates publish and clean up under one boundary.
        self.scene_lock = threading.RLock()
        self._scene_generation = 0
        self.runtime_logger.info(
            "Runtime session initialized",
            extra={
                "event": "session_start",
                "backend": self.backend,
                "audio_source": self.audio_source.kind,
                "gender": self.current_gender,
                "age": self.current_age,
                "visual_mode": self.current_visual_mode,
                "prompt_style": self.current_prompt_style,
                "prompt_budget_mode": self.prompt_budget_mode,
                "language": self.current_language or "auto",
                "log_file": (
                    str(self.log_session.path)
                    if self.log_session.path is not None
                    else ""
                ),
            },
        )

    @property
    def is_running(self):
        return not self.stop_event.is_set()

    def request_shutdown(self, reason=None):
        was_requested = self.stop_event.is_set()
        self.stop_event.set()
        if not was_requested and reason:
            self.runtime_logger.info(
                "Runtime shutdown requested",
                extra={
                    "event": "shutdown_requested",
                    "reason": reason,
                },
            )
        return not was_requested

    def start_worker_threads(self):
        with self._worker_lock:
            if self._workers_started:
                raise RuntimeError("Runtime workers have already been started")
            if self._backend_closed:
                raise RuntimeError("Runtime workers cannot start after close")

            self._workers_started = True
            workers = {
                "audio": threading.Thread(
                    name="voice-to-visual-audio",
                    target=self._run_worker,
                    args=("audio", self.audio_callback),
                ),
                "transcription": threading.Thread(
                    name="voice-to-visual-transcription",
                    target=self._run_worker,
                    args=("transcription", self.transcription_loop),
                ),
            }
            self._worker_threads = workers

        for worker in workers.values():
            worker.start()

        self.runtime_logger.info(
            "Runtime workers started",
            extra={
                "event": "workers_started",
                "workers": sorted(workers),
            },
        )
        return tuple(workers.values())

    def _run_worker(self, worker_name, target):
        try:
            target()
        except Exception as exc:
            self.runtime_logger.exception(
                "Runtime worker crashed",
                extra={
                    "event": "worker_crashed",
                    "worker": worker_name,
                    "error": self.clean_audio_error(exc),
                },
            )
            self.request_shutdown()
        finally:
            self.runtime_logger.debug(
                "Runtime worker stopped",
                extra={
                    "event": "worker_stopped",
                    "worker": worker_name,
                },
            )

    def join_worker_threads(self):
        with self._worker_lock:
            workers = tuple(self._worker_threads.values())

        if not workers:
            return ()

        current_worker = threading.current_thread()
        workers = tuple(
            worker for worker in workers if worker is not current_worker
        )
        started = time.monotonic()
        deadline = started + self.config.runtime_shutdown_grace_seconds
        for worker in workers:
            remaining = max(0.0, deadline - time.monotonic())
            worker.join(timeout=remaining)

        overdue = tuple(worker for worker in workers if worker.is_alive())
        if overdue:
            self.runtime_logger.warning(
                "Runtime workers exceeded the shutdown grace period",
                extra={
                    "event": "worker_shutdown_overdue",
                    "grace_seconds": (
                        self.config.runtime_shutdown_grace_seconds
                    ),
                    "workers": [worker.name for worker in overdue],
                },
            )
            for worker in overdue:
                worker.join()

        with self._worker_lock:
            self._worker_threads.clear()

        self.runtime_logger.info(
            "Runtime workers stopped",
            extra={
                "event": "workers_stopped",
                "elapsed_seconds": time.monotonic() - started,
                "overdue_workers": [worker.name for worker in overdue],
            },
        )
        return overdue

    def create_prompt_budgeter(self):
        if not self.config.prompt_token_budget_enabled:
            self.prompt_budget_mode = "disabled"
            self.prompt_logger.info(
                "Prompt token budgeting is disabled",
                extra={"event": "prompt_budget_disabled"},
            )
            return None
        try:
            from transformers import AutoTokenizer

            tokenizers = [
                AutoTokenizer.from_pretrained(model)
                for model in self.config.prompt_tokenizer_models
            ]
            self.prompt_logger.info(
                "Prompt token budgeting initialized",
                extra={
                    "event": "prompt_budget_initialized",
                    "tokenizer_count": len(tokenizers),
                    "max_tokens": self.config.prompt_max_tokens,
                    "budget_mode": "exact",
                },
            )
            self.prompt_budget_mode = "exact"
            return PromptBudgeter(
                tokenizers,
                max_tokens=self.config.prompt_max_tokens,
                min_transcript_tokens=(
                    self.config.prompt_min_transcript_tokens
                ),
                mode="exact",
            )
        except Exception as exc:
            if self.config.prompt_token_budget_fallback == "conservative":
                self.prompt_budget_mode = "fallback"
                self.prompt_logger.warning(
                    "Prompt tokenizers unavailable; using conservative "
                    "offline budgeting",
                    extra={
                        "event": "prompt_budget_fallback",
                        "budget_mode": "fallback",
                        "max_tokens": self.config.prompt_max_tokens,
                        "error": self.clean_audio_error(exc),
                    },
                )
                return PromptBudgeter(
                    [ConservativeUtf8Tokenizer()],
                    max_tokens=self.config.prompt_max_tokens,
                    min_transcript_tokens=(
                        self.config.prompt_min_transcript_tokens
                    ),
                    mode="fallback",
                )

            self.prompt_budget_mode = "unavailable"
            self.prompt_logger.warning(
                "Prompt tokenizers unavailable; prompts will not be limited",
                extra={
                    "event": "prompt_budget_unavailable",
                    "budget_mode": "unavailable",
                    "error": self.clean_audio_error(exc),
                },
            )
            return None

    def create_voice_activity_detector(self):
        if self.config.vad_engine == "silero":
            try:
                detector = SileroVoiceActivityDetector(
                    self.config.vad_threshold,
                    sample_rate=RATE,
                )
                self.audio_logger.info(
                    "Silero voice activity detection initialized",
                    extra={
                        "event": "vad_initialized",
                        "engine": "silero",
                        "threshold": self.config.vad_threshold,
                    },
                )
                return detector
            except Exception as exc:
                self.audio_logger.warning(
                    "Silero unavailable; falling back to energy detection",
                    extra={
                        "event": "vad_fallback",
                        "from_engine": "silero",
                        "to_engine": "energy",
                        "error": self.clean_audio_error(exc),
                    },
                )
        elif self.config.vad_engine != "energy":
            self.audio_logger.warning(
                "Unknown VAD engine; falling back to energy detection",
                extra={
                    "event": "vad_fallback",
                    "from_engine": self.config.vad_engine,
                    "to_engine": "energy",
                },
            )

        self.audio_logger.info(
            "Energy voice activity detection initialized",
            extra={
                "event": "vad_initialized",
                "engine": "energy",
                "threshold": self.config.vad_energy_threshold,
            },
        )
        return EnergyVoiceActivityDetector(
            self.config.vad_energy_threshold
        )

    def process_audio_data(self, data, *, scene_generation=None):
        with self.scene_lock:
            if scene_generation is None:
                scene_generation = self._scene_generation
            if scene_generation != self._scene_generation:
                return
        audio_data = np.frombuffer(data, dtype=np.int16)
        try:
            is_speech = self.vad.is_speech(audio_data)
        except Exception as exc:
            self.audio_logger.error(
                "Voice activity detection failed; switching to energy",
                extra={
                    "event": "vad_runtime_fallback",
                    "error": self.clean_audio_error(exc),
                },
            )
            self.vad = EnergyVoiceActivityDetector(
                self.config.vad_energy_threshold
            )
            is_speech = self.vad.is_speech(audio_data)

        with self.scene_lock:
            if scene_generation != self._scene_generation:
                return
            with self.lock:
                completed = self.segmenter.add_chunk(audio_data, is_speech)
                speaking = self.segmenter.active

            if speaking != self.is_speaking:
                self.is_speaking = speaking
                self.send_runtime_status(force=True)

            now = time.monotonic()
            for segment in completed:
                self.submit_final_segment(segment, now, source="completed")
            self.send_runtime_status()

    def submit_final_segment(self, segment, now, *, source):
        with self.scene_lock:
            return self._submit_final_segment_locked(segment, now, source=source)

    def _submit_final_segment_locked(self, segment, now, *, source):
        submission = self.scheduler.submit_final(segment, now)
        dropped_job = submission.dropped_job
        if dropped_job is None:
            return submission

        self.cleanup_final_job(dropped_job)
        extra = {
            "segment_id": dropped_job.segment.segment_id,
            "incoming_segment_id": segment.segment_id,
            "source": source,
            "overflow_policy": self.scheduler.final_overflow_policy,
        }
        if submission.drop_reason == "oldest_capacity":
            extra["event"] = "scheduler_final_oldest_dropped"
            self.scheduler_logger.warning(
                "Final queue is full; oldest pending segment was dropped "
                "for fresher speech",
                extra=extra,
            )
        else:
            extra["event"] = "scheduler_final_newest_dropped"
            self.scheduler_logger.warning(
                "Final queue is full; newest segment was dropped",
                extra=extra,
            )
        return submission

    def wait_for_audio_retry(self, delay_seconds):
        return not self.stop_event.wait(max(0.0, delay_seconds))

    def finalize_interrupted_audio(self):
        with self.scene_lock:
            with self.lock:
                segment = self.segmenter.interrupt()
            if segment is None or segment.samples.size == 0:
                return
            self.submit_final_segment(
                segment,
                time.monotonic(),
                source="interrupted",
            )

    def audio_callback(self):
        reconnect_attempt = 0
        source = self.audio_source

        while self.is_running:
            opened = False
            finished = False
            failure = None
            try:
                source.open()
                opened = True
                self.audio_device_index = source.device_index
                self.audio_device_name = source.name

                if self.audio_reconnects:
                    self.audio_logger.info(
                        "Audio source connection recovered",
                        extra={
                            "event": "audio_recovered",
                            "source_kind": source.kind,
                            "device_index": self.audio_device_index,
                            "device_name": self.audio_device_name,
                            "reconnects": self.audio_reconnects,
                        },
                    )
                else:
                    self.audio_logger.info(
                        "Audio source opened",
                        extra={
                            "event": "audio_stream_opened",
                            "source_kind": source.kind,
                            "device_index": self.audio_device_index,
                            "device_name": self.audio_device_name,
                        },
                    )
                self.audio_status = "ready"
                self.last_audio_error = ""
                self.send_runtime_status(force=True)

                consecutive_read_errors = 0
                while self.is_running:
                    with self.scene_lock:
                        scene_generation = self._scene_generation
                    try:
                        data = source.read()
                    except AudioSourceFinished:
                        self.finalize_interrupted_audio()
                        self.audio_source_finished.set()
                        finished = True
                        self.audio_logger.info(
                            "Finite audio source completed",
                            extra={
                                "event": "audio_source_completed",
                                "source_kind": source.kind,
                                "source_name": source.name,
                            },
                        )
                        break
                    except AudioSourceStopped:
                        break
                    except Exception as exc:
                        consecutive_read_errors += 1
                        self.audio_status = "degraded"
                        self.last_audio_error = self.clean_audio_error(exc)
                        self.is_speaking = False
                        self.send_runtime_status(force=True)

                        max_read_errors = (
                            self.config.audio_max_consecutive_read_errors
                        )
                        if (
                            not source.reconnectable
                            or consecutive_read_errors >= max_read_errors
                        ):
                            raise RuntimeError(
                                "audio source read failed "
                                f"{consecutive_read_errors} consecutive times: {exc}"
                            ) from exc

                        self.audio_logger.warning(
                            "Audio source read failed; retrying current stream",
                            extra={
                                "event": "audio_read_error",
                                "source_kind": source.kind,
                                "error": self.clean_audio_error(exc),
                                "consecutive_errors": consecutive_read_errors,
                                "max_consecutive_errors": max_read_errors,
                            },
                        )
                        if not self.wait_for_audio_retry(
                            self.config.audio_read_retry_seconds
                        ):
                            break
                        continue

                    if consecutive_read_errors:
                        self.audio_logger.info(
                            "Audio source reads resumed",
                            extra={
                                "event": "audio_reads_resumed",
                                "source_kind": source.kind,
                            },
                        )
                        self.audio_status = "ready"
                        self.last_audio_error = ""
                        self.send_runtime_status(force=True)
                    consecutive_read_errors = 0
                    reconnect_attempt = 0
                    try:
                        self.process_audio_data(
                            data, scene_generation=scene_generation
                        )
                    except Exception as exc:
                        self.audio_logger.error(
                            "Audio processing failed",
                            extra={
                                "event": "audio_processing_error",
                                "error": self.clean_audio_error(exc),
                            },
                        )
            except Exception as exc:
                failure = exc
            finally:
                source.close()

            if finished or failure is None or not self.is_running:
                break

            if opened:
                self.finalize_interrupted_audio()
            self.is_speaking = False
            self.last_audio_error = self.clean_audio_error(failure)
            if (
                not source.reconnectable
                or not self.config.audio_reconnect_enabled
            ):
                self.audio_status = "error"
                self.request_shutdown()
                self.audio_logger.error(
                    "Audio source failed and cannot reconnect",
                    extra={
                        "event": "audio_terminal_error",
                        "source_kind": source.kind,
                        "error": self.clean_audio_error(failure),
                    },
                )
                self.send_runtime_status(force=True)
                break

            retry_delay = exponential_backoff(
                reconnect_attempt,
                base_seconds=self.config.audio_reconnect_base_seconds,
                max_seconds=self.config.audio_reconnect_max_seconds,
            )
            reconnect_attempt += 1
            self.audio_reconnects += 1
            self.audio_status = "reconnecting"
            self.audio_logger.warning(
                "Audio source unavailable; scheduling reconnect",
                extra={
                    "event": "audio_reconnect_scheduled",
                    "source_kind": source.kind,
                    "error": self.clean_audio_error(failure),
                    "retry_in_seconds": retry_delay,
                    "attempt": self.audio_reconnects,
                },
            )
            self.send_runtime_status(force=True)
            if not self.wait_for_audio_retry(retry_delay):
                break

        if self.audio_status != "error":
            self.audio_status = "stopped"
            self.is_speaking = False
            self.audio_logger.info(
                "Audio capture stopped",
                extra={
                    "event": "audio_stopped",
                    "source_kind": source.kind,
                    "reconnects": self.audio_reconnects,
                },
            )
            self.send_runtime_status(force=True)

    @staticmethod
    def clean_audio_error(error):
        return " ".join(str(error).split())[:240]

    def transcription_loop(self):
        while self.is_running:
            now = time.monotonic()
            if (
                self.audio_source_finished.is_set()
                and self.scheduler.metrics(now).queue_depth == 0
            ):
                self.request_shutdown("audio_source_completed")
                break
            if not self.request_interval_ready(now):
                self.send_runtime_status()
                if self.stop_event.wait(0.05):
                    break
                continue

            with self.scene_lock:
                with self.lock:
                    partial = self.segmenter.snapshot(self.minimum_audio_seconds())
                if partial is not None:
                    self.scheduler.submit_partial(partial, now)
                job = self.scheduler.next_job(now)
                scene_generation = self._scene_generation
                if job is not None:
                    self.mark_request_started(now)
                    self.backend_status = "transcribing"
                    self.send_runtime_status(force=True)
            if job is None:
                self.send_runtime_status()
                if self.stop_event.wait(0.05):
                    break
                continue

            started = time.monotonic()
            try:
                text = self.transcribe_audio(job.segment.samples)
            except RetryableTranscriptionError as exc:
                self.handle_retryable_failure(
                    job, exc, scene_generation=scene_generation
                )
                continue
            except Exception as exc:
                with self.scene_lock:
                    if self._discard_obsolete_job(job, scene_generation, "error"):
                        continue
                    self.transcription_logger.error(
                        "Transcription failed",
                        extra={
                            "event": "transcription_error",
                            "error": self.clean_audio_error(exc),
                            "segment_id": job.segment.segment_id,
                            "is_final": job.is_final,
                        },
                    )
                    self.scheduler.mark_failed()
                    self.backend_status = "error"
                    self.cleanup_final_job(job)
                    self.send_runtime_status(force=True)
                continue

            finished = time.monotonic()
            with self.scene_lock:
                if self._discard_obsolete_job(job, scene_generation, "result"):
                    continue
                self._complete_transcription_locked(job, text, started, finished)

    def _discard_obsolete_job(self, job, scene_generation, outcome):
        """Called under scene_lock before applying an asynchronous result."""
        if scene_generation is None or scene_generation == self._scene_generation:
            return False
        self.backend_status = (
            "retrying" if time.monotonic() < self.backend_retry_not_before else "ready"
        )
        self.transcription_logger.info(
            "Transcription from before the scene reset discarded",
            extra={
                "event": "transcription_scene_discarded",
                "segment_id": job.segment.segment_id,
                "scene_generation": scene_generation,
                "current_scene_generation": self._scene_generation,
                "outcome": outcome,
            },
        )
        self.send_runtime_status(force=True)
        return True

    def _complete_transcription_locked(self, job, text, started, finished):
        # Include time spent waiting to reacquire scene_lock before applying text.
        result_time = time.monotonic()
        self.last_inference_latency = finished - started
        self.last_total_latency = result_time - job.created_at
        self.backend_status = "ready"
        self.backend_retry_not_before = 0.0
        if not self.scheduler.complete_job(job, result_time):
            self.cleanup_final_job(job)
            self.transcription_logger.warning(
                "Transcription result exceeded its age limit and was discarded",
                extra={
                    "event": "transcription_result_expired",
                    "backend": self.backend,
                    "job_id": job.job_id,
                    "segment_id": job.segment.segment_id,
                    "is_final": job.is_final,
                    "attempt": job.attempts + 1,
                    "result_age_seconds": self.last_total_latency,
                    "max_age_seconds": self.scheduler.max_age_seconds(job),
                    "latency_asr_seconds": self.last_inference_latency,
                },
            )
            self.send_runtime_status(force=True)
            return
        self.transcription_logger.info(
            "Transcription completed",
            extra={
                "event": "transcription_completed",
                "backend": self.backend,
                "segment_id": job.segment.segment_id,
                "is_final": job.is_final,
                "audio_seconds": len(job.segment.samples) / RATE,
                "latency_asr_seconds": self.last_inference_latency,
                "latency_total_seconds": self.last_total_latency,
                "character_count": len(text or ""),
            },
        )

        if is_probable_whisper_hallucination(text):
            text = ""
        stabilizer = self.stabilizers.setdefault(
            job.segment.segment_id,
            TranscriptStabilizer(self.config.transcript_confirm_updates),
        )
        update = stabilizer.update(text, is_final=job.is_final)
        scene_text = ""
        if update.text:
            scene_text = self.scene_memory.update(
                job.segment.segment_id,
                update.text,
                is_final=update.is_final,
            )
        if scene_text and (update.changed or update.is_final):
            self.emit_transcript(
                scene_text, raw_text=update.text, is_final=update.is_final
            )
        self.cleanup_final_job(job)
        self.send_runtime_status(force=True)

    def request_interval_ready(self, now):
        if now < self.backend_retry_not_before:
            return False
        if self.backend_adapter.online:
            return now - self.last_online_request_time >= self.online_request_interval()
        return now - self.last_whisper_request_time >= self.local_request_interval()

    def mark_request_started(self, now):
        if self.backend_adapter.online:
            self.last_online_request_time = now
        else:
            self.last_whisper_request_time = now

    def handle_retryable_failure(self, job, exc, *, scene_generation=None):
        with self.scene_lock:
            self._handle_retryable_failure_locked(job, exc, scene_generation)

    def _handle_retryable_failure_locked(self, job, exc, scene_generation):
        now = time.monotonic()
        retry_delay = exc.retry_after
        if retry_delay is None:
            retry_delay = exponential_backoff(
                job.attempts,
                base_seconds=(
                    self.config.transcription_retry_base_seconds
                ),
                max_seconds=(
                    self.config.transcription_retry_max_seconds
                ),
            )
        self.backend_retry_not_before = max(
            self.backend_retry_not_before,
            now + retry_delay,
        )
        # Provider cooldown applies to the endpoint even if its old job was reset.
        if self._discard_obsolete_job(job, scene_generation, "retry"):
            return

        max_retries = self.config.transcription_final_max_retries
        should_retry = job.is_final and job.attempts < max_retries
        retry_submission = (
            self.scheduler.retry_final(job, now, retry_delay)
            if should_retry
            else None
        )
        if retry_submission:
            self.backend_status = "retrying"
            self.transcription_logger.warning(
                "Final transcription segment scheduled for retry",
                extra={
                    "event": "transcription_retry_scheduled",
                    "segment_id": job.segment.segment_id,
                    "retry_in_seconds": retry_delay,
                    "attempt": job.attempts + 1,
                    "max_retries": max_retries,
                    "error": self.clean_audio_error(exc),
                },
            )
        else:
            drop_reason = getattr(retry_submission, "drop_reason", "")
            if drop_reason == "oldest_capacity":
                self.scheduler_logger.warning(
                    "Older retry was dropped to preserve fresher queued speech",
                    extra={
                        "event": "scheduler_final_oldest_dropped",
                        "segment_id": job.segment.segment_id,
                        "source": "retry",
                        "overflow_policy": (
                            self.scheduler.final_overflow_policy
                        ),
                    },
                )
            self.scheduler.mark_failed()
            self.backend_status = "error"
            self.cleanup_final_job(job)
            capacity_drop = drop_reason == "oldest_capacity"
            self.transcription_logger.error(
                (
                    "Final transcription retry dropped because the queue is full"
                    if capacity_drop
                    else "Transcription retries exhausted"
                ),
                extra={
                    "event": (
                        "transcription_retry_capacity_dropped"
                        if capacity_drop
                        else "transcription_retry_exhausted"
                    ),
                    "segment_id": job.segment.segment_id,
                    "attempts": job.attempts,
                    "drop_reason": drop_reason,
                    "error": self.clean_audio_error(exc),
                },
            )
        self.send_runtime_status(force=True)

    def cleanup_final_job(self, job):
        with self.scene_lock:
            if job.is_final:
                self.stabilizers.pop(job.segment.segment_id, None)

    def send_osc_message(self, address, value):
        return self.output_publisher.send(address, value)

    def send_runtime_status(self, force=False):
        now = time.monotonic()
        with self.state_lock:
            current_gender = self.current_gender
            current_age = self.current_age
            current_visual_mode = self.current_visual_mode
            current_prompt_style = self.current_prompt_style
            current_language = self.current_language or "auto"
        metrics = self.scheduler.metrics(now)
        snapshot = RuntimeStatusSnapshot(
            backend_status=self.backend_status,
            backend=self.backend,
            is_speaking=self.is_speaking,
            queue_depth=metrics.queue_depth,
            latency_total=self.last_total_latency,
            latency_asr=self.last_inference_latency,
            retry_in=max(0.0, self.backend_retry_not_before - now),
            dropped_jobs=(
                metrics.dropped_stale + metrics.dropped_finals
            ),
            audio_status=self.audio_status,
            audio_source=self.audio_source.kind,
            audio_reconnects=self.audio_reconnects,
            audio_error=self.last_audio_error,
            audio_device_index=self.audio_device_index,
            audio_device_name=self.audio_device_name,
            gender=current_gender,
            age=current_age,
            visual_mode=current_visual_mode,
            prompt_style=current_prompt_style,
            language=current_language,
            prompt_budget_mode=self.prompt_budget_mode,
            dropped_final_oldest=metrics.dropped_final_oldest,
            dropped_final_newest=metrics.dropped_final_newest,
            dropped_expired_results=metrics.dropped_expired_results,
        )
        return self.output_publisher.publish_status(
            snapshot,
            force=force,
        )

    def selected_language(self):
        with self.state_lock:
            return self.current_language

    def emit_transcript(self, text, raw_text=None, is_final=False):
        with self.scene_lock:
            self._emit_transcript_locked(text, raw_text, is_final)

    def _emit_transcript_locked(self, text, raw_text, is_final):
        if not text:
            return
        raw_text = raw_text or text
        unchanged = text == self.last_text
        if not unchanged:
            self.last_text = text
        if unchanged:
            if is_final:
                self.send_osc_message("/transcript_final", raw_text)
            return
        final_prompt = self.build_visual_prompt(text)

        self.send_osc_message("/prompt", final_prompt)
        self.send_osc_message("/partial_text", raw_text)
        self.send_osc_message("/scene_context", text)
        self.send_osc_message("/prompt_tokens", self.last_prompt_token_count)
        self.send_osc_message("/prompt_budget_mode", self.prompt_budget_mode)
        if is_final:
            self.send_osc_message("/transcript_final", raw_text)

        state = "FINAL" if is_final else "STABLE"
        self.prompt_logger.info(
            "Visual prompt emitted",
            extra={
                "event": "prompt_emitted",
                "state": state.lower(),
                "scene_character_count": len(text),
                "transcript_character_count": len(raw_text),
                "prompt_tokens": self.last_prompt_token_count,
                "prompt_budget_mode": self.prompt_budget_mode,
                "prompt_variant": self.last_prompt_variant,
                "transcript_trimmed": self.last_prompt_trimmed,
            },
        )
        print(f"\n[PROMPT {state}]: {text}")
        if raw_text != text:
            print(f"[CURRENT TRANSCRIPT]: {raw_text}")

    def refresh_visual_prompt(self):
        with self.scene_lock:
            return self._refresh_visual_prompt_locked()

    def _refresh_visual_prompt_locked(self):
        text = self.last_text
        if not text:
            return False

        final_prompt = self.build_visual_prompt(text)
        self.send_osc_message("/prompt", final_prompt)
        self.send_osc_message("/prompt_tokens", self.last_prompt_token_count)
        self.send_osc_message("/prompt_budget_mode", self.prompt_budget_mode)
        self.prompt_logger.info(
            "Visual prompt refreshed after control change",
            extra={
                "event": "prompt_refreshed",
                "prompt_tokens": self.last_prompt_token_count,
                "prompt_budget_mode": self.prompt_budget_mode,
                "prompt_variant": self.last_prompt_variant,
            },
        )
        print(f"\n[PROMPT REFRESH]: Applied the current visual controls to: {text}")
        return True

    def build_visual_prompt(self, text):
        with self.state_lock:
            current_visual_mode = self.current_visual_mode
            current_prompt_style = self.current_prompt_style
            current_gender = self.current_gender
            current_age = self.current_age

        visual_mode = VISUAL_MODES[current_visual_mode]

        if current_prompt_style == "general_scene":
            variants = [
                (
                    "full",
                    lambda value: SCENE_PROMPT_TEMPLATE.format(
                        text=value,
                        visual_context=visual_mode["scene_context"],
                    ),
                ),
                (
                    "compact_context",
                    lambda value: SCENE_PROMPT_TEMPLATE.format(
                        text=value,
                        visual_context=visual_mode["compact_scene_context"],
                    ),
                ),
                (
                    "compact",
                    lambda value: COMPACT_SCENE_PROMPT_TEMPLATE.format(
                        text=value,
                        visual_context=visual_mode["compact_scene_context"],
                    ),
                ),
                (
                    "minimal",
                    lambda value: MINIMAL_SCENE_PROMPT_TEMPLATE.format(
                        text=value,
                        visual_context=visual_mode["minimal_scene_context"],
                    ),
                ),
            ]
        else:
            gender_focus = GENDER_MODES.get(current_gender, "person")
            age_desc = AGE_MODES.get(current_age, "")
            subject_focus = f"{visual_mode['subject_prefix']} {gender_focus}"
            variants = [
                (
                    "full",
                    lambda value: FIXED_PROMPT_TEMPLATE.format(
                        age_desc=age_desc,
                        subject_focus=subject_focus,
                        visual_context=visual_mode["context"],
                        text=value,
                    ),
                ),
                (
                    "compact_context",
                    lambda value: FIXED_PROMPT_TEMPLATE.format(
                        age_desc=age_desc,
                        subject_focus=subject_focus,
                        visual_context=visual_mode["compact_context"],
                        text=value,
                    ),
                ),
                (
                    "compact",
                    lambda value: COMPACT_FIXED_PROMPT_TEMPLATE.format(
                        age_desc=age_desc,
                        subject_focus=subject_focus,
                        visual_context=visual_mode["compact_context"],
                        text=value,
                    ),
                ),
                (
                    "minimal",
                    lambda value: MINIMAL_FIXED_PROMPT_TEMPLATE.format(
                        age_desc=age_desc,
                        subject_focus=subject_focus,
                        text=value,
                    ),
                ),
            ]

        if self.prompt_budgeter is None:
            self.last_prompt_token_count = 0
            self.last_prompt_variant = "unbudgeted"
            self.last_prompt_trimmed = False
            return variants[0][1](text)

        result = self.prompt_budgeter.fit(variants, text)
        self.last_prompt_token_count = result.token_count
        self.last_prompt_variant = result.variant
        self.last_prompt_trimmed = result.transcript_trimmed
        self.prompt_budget_mode = result.budget_mode
        if self.config.prompt_log_tokens:
            trimmed = ", newest transcript retained" if result.transcript_trimmed else ""
            self.prompt_logger.info(
                "Prompt token budget measured",
                extra={
                    "event": "prompt_budget_measured",
                    "token_count": result.token_count,
                    "max_tokens": self.config.prompt_max_tokens,
                    "budget_mode": result.budget_mode,
                    "variant": result.variant,
                    "transcript_trimmed": result.transcript_trimmed,
                    "detail": trimmed.lstrip(", "),
                },
            )
        return result.text

    def transcribe_audio(self, audio_samples):
        return self.backend_adapter.transcribe(
            audio_samples,
            language=self.selected_language(),
        )

    def minimum_audio_seconds(self):
        return self.backend_adapter.minimum_audio_seconds

    def online_request_interval(self):
        if self.backend_adapter.online:
            return self.backend_adapter.request_interval
        return 0.0

    def local_request_interval(self):
        if not self.backend_adapter.online:
            return self.backend_adapter.request_interval
        return 0.0

    def maximum_audio_seconds(self):
        return self.backend_adapter.maximum_audio_seconds

    def benchmark(self, wav_path, runs=3):
        if self.backend not in {"whisper", "faster_whisper"}:
            raise RuntimeError("Benchmark mode supports only whisper and faster_whisper backends.")

        samples = load_wav_samples(
            wav_path,
            target_sample_rate=RATE,
        )
        audio_seconds = len(samples) / RATE
        runs = max(1, runs)
        warmup = samples[:min(len(samples), int(2 * RATE))]

        print("\n" + "=" * 50)
        print(f"LOCAL ASR BENCHMARK: {self.backend}")
        device = getattr(
            self.backend_adapter,
            "device",
            self.config.whisper_device or "auto",
        )
        print(
            f"Model: {self.config.whisper_model_size} | "
            f"Device: {device} | Audio: {audio_seconds:.2f}s"
        )
        if self.backend == "faster_whisper":
            print(
                "Compute type: "
                f"{self.config.faster_whisper_compute_type}"
            )
        print("Warming up...")
        self.transcribe_audio(warmup)

        latencies = []
        for run_number in range(1, runs + 1):
            started = time.perf_counter()
            text = self.transcribe_audio(samples)
            elapsed = time.perf_counter() - started
            latencies.append(elapsed)
            rtf = elapsed / audio_seconds
            print(f"Run {run_number}: {elapsed:.3f}s | real-time factor {rtf:.3f} | {len(text)} chars")

        average = sum(latencies) / len(latencies)
        print(f"Average: {average:.3f}s | real-time factor {average / audio_seconds:.3f}")
        print("=" * 50)

    def start_osc_control_server(self):
        if not self.osc_control_enabled:
            return
        server = OscControlServer(
            self.config.osc_control_ip,
            self.config.osc_control_port,
            self.apply_control,
            self.osc_logger.warning,
        )
        try:
            address = server.start()
        except OSError as exc:
            self.osc_logger.error(
                "OSC control server could not bind",
                extra={
                    "event": "osc_control_bind_error",
                    "ip": self.config.osc_control_ip,
                    "port": self.config.osc_control_port,
                    "error": self.clean_audio_error(exc),
                },
            )
            return
        self.osc_control_server = server
        self.osc_logger.info(
            "OSC control server listening",
            extra={
                "event": "osc_control_listening",
                "ip": address[0],
                "port": address[1],
            },
        )

    def apply_control(self, control_name, value):
        if control_name == "request_status":
            self.send_runtime_status(force=True)
            return
        if control_name == "reset_scene":
            self.reset_scene()
            return

        valid_values = {
            "gender": SUPPORTED_GENDERS,
            "age": SUPPORTED_AGES,
            "visual_mode": SUPPORTED_VISUAL_MODES,
            "prompt_style": SUPPORTED_PROMPT_STYLES,
            "language": {None} | (SUPPORTED_LANGUAGES - {"auto"}),
        }
        if control_name not in valid_values or value not in valid_values[control_name]:
            raise ValueError(f"Invalid {control_name} control value: {value}")

        with self.state_lock:
            if control_name == "gender":
                self.current_gender = value
                label = value.upper()
            elif control_name == "age":
                self.current_age = value
                label = value.upper()
            elif control_name == "visual_mode":
                self.current_visual_mode = value
                label = VISUAL_MODES[value]["label"]
            elif control_name == "prompt_style":
                self.current_prompt_style = value
                label = PROMPT_STYLES[value]["label"]
            else:
                self.current_language = value
                label = "AUTO-DETECT" if value is None else value.upper()

        self.control_logger.info(
            "Runtime control changed",
            extra={
                "event": "control_changed",
                "control": control_name,
                "value": "auto" if value is None else value,
                "label": label,
            },
        )
        acknowledgement = "auto" if value is None else str(value)
        self.send_osc_message("/control_ack", f"{control_name}:{acknowledgement}")
        if control_name in {"gender", "age", "visual_mode", "prompt_style"}:
            self.refresh_visual_prompt()
        self.send_runtime_status(force=True)

    def reset_scene(self):
        with self.scene_lock:
            self._scene_generation += 1
            with self.lock:
                self.segmenter.reset()
            discarded_jobs = self.scheduler.clear()
            self.stabilizers.clear()
            self.scene_memory.reset()
            self.last_text = ""
            self.is_speaking = False
            self.last_prompt_token_count = 0
            self.last_prompt_variant = "reset"
            self.last_prompt_trimmed = False
            self.send_osc_message("/partial_text", "")
            self.send_osc_message("/scene_context", "")
            self.send_osc_message("/prompt_tokens", 0)
            self.send_osc_message("/scene_reset", 1)
            self.control_logger.info(
                "Scene memory and pending speech reset",
                extra={
                    "event": "scene_memory_reset",
                    "scene_generation": self._scene_generation,
                    "discarded_queued_jobs": discarded_jobs,
                },
            )
            self.send_runtime_status(force=True)

    def start(self):
        self.start_osc_control_server()
        self.start_worker_threads()
        
        try:
            print("\n" + "="*50)
            print(
                f"BACKEND: {self.backend} | "
                f"VAD: {self.config.vad_engine} | "
                f"CONFIRMATIONS: {self.config.transcript_confirm_updates}"
            )
            print(
                f"AUDIO SOURCE: {self.audio_source.name} "
                f"({self.audio_source.kind})"
            )
            print(
                "STARTUP CONTROLS: "
                f"{self.current_gender}, {self.current_age}, "
                f"{self.current_visual_mode}, {self.current_prompt_style}, "
                f"{self.current_language or 'auto'}"
            )
            print("CONTROL KEYS:")
            print("  [GENDER] 'm' -> Man | 'w' -> Woman | 'n' -> Neutral")
            print("  [AGE]    '1' -> Young | '2' -> Adult | '3' -> Elder")
            print("  [VISUAL] 'd' -> Asian American | 'b' -> Black and Brown people | 'x' -> Asian + Black and Brown")
            print("  [PROMPT] 'f' -> Human figure focus | 'g' -> General scene")
            print("  [LANG]   'e' -> English | 'c' -> Chinese | 's' -> Spanish | 'a' -> Auto")
            if self.osc_control_server is not None:
                print(
                    f"  [OSC IN] {self.config.osc_control_ip}:"
                    f"{self.config.osc_control_port} "
                    "for TouchDesigner controls"
                )
            if self.backend == "groq":
                print("  [ONLINE] Groq translates detected speech to English automatically")
            if self.backend == "google":
                print(
                    "  [ONLINE] Auto/default language -> "
                    f"{self.config.google_speech_language}"
                )
            print("  Ctrl+C   -> Exit")
            print("="*50 + "\n")
            self.send_runtime_status(force=True)
            
            while self.is_running:
                if msvcrt.kbhit():
                    try:
                        key = msvcrt.getch().decode("utf-8").lower()
                    except UnicodeDecodeError:
                        continue
                    control = KEYBOARD_CONTROLS.get(key)
                    if control is not None:
                        self.apply_control(*control)
                if self.stop_event.wait(0.1):
                    break
        except KeyboardInterrupt:
            self.request_shutdown("keyboard_interrupt")
        finally:
            self.close()

    def close(self):
        with self._close_lock:
            if self._backend_closed:
                return

            self.request_shutdown()
            if self._workers_started:
                self.backend_status = "stopped"
                self.is_speaking = False
                self.send_runtime_status(force=True)
            if self.osc_control_server is not None:
                self.osc_control_server.stop()
                self.osc_control_server = None

            self.join_worker_threads()
            self.backend_status = "stopped"
            self.is_speaking = False
            if self._workers_started:
                self.send_runtime_status(force=True)

            self.output_publisher.close()
            self._backend_closed = True
            try:
                self.backend_adapter.close()
            finally:
                metrics = self.scheduler.metrics()
                self.runtime_logger.info(
                    "Runtime session stopped",
                    extra={
                        "event": "session_stop",
                        "backend": self.backend,
                        "audio_source": self.audio_source.kind,
                        "audio_reconnects": self.audio_reconnects,
                        "processed_jobs": metrics.processed,
                        "failed_jobs": metrics.failed,
                        "dropped_jobs": (
                            metrics.dropped_stale + metrics.dropped_finals
                        ),
                        "dropped_final_oldest": (
                            metrics.dropped_final_oldest
                        ),
                        "dropped_final_newest": (
                            metrics.dropped_final_newest
                        ),
                        "dropped_expired_results": (
                            metrics.dropped_expired_results
                        ),
                    },
                )
                if self._owns_log_session:
                    self.log_session.close()


def main(argv=None):
    args = parse_args(argv)
    try:
        config = load_runtime_config(args)
    except ConfigError as exc:
        print(format_config_error(exc))
        return 2

    if args.check_config:
        print(format_config_report(config))
        return 0

    if args.list_audio_devices:
        return print_audio_input_devices()

    if args.diagnose:
        return run_diagnostics(
            config=config,
            sample_rate=RATE,
            chunk_size=CHUNK,
            device=config.whisper_device or "cuda",
        )

    audio_source = None
    if args.replay:
        try:
            audio_source = WavReplaySource(
                args.replay,
                sample_rate=RATE,
                chunk_samples=CHUNK,
                realtime=True,
            )
        except ValueError as exc:
            print(f"Invalid replay audio: {exc}")
            return 2

    pipeline = RealTimePipeline(
        enable_vad=not bool(args.benchmark),
        enable_osc=not bool(args.benchmark),
        enable_prompt_budget=not bool(args.benchmark),
        enable_osc_controls=not bool(args.benchmark),
        config=config,
        audio_source=audio_source,
    )
    try:
        if args.benchmark:
            pipeline.benchmark(args.benchmark, args.benchmark_runs)
        else:
            pipeline.start()
    finally:
        pipeline.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
