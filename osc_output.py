import threading
import time
from dataclasses import dataclass

from pythonosc import udp_client


@dataclass(frozen=True)
class OscMessage:
    address: str
    value: object


@dataclass(frozen=True)
class RuntimeStatusSnapshot:
    backend_status: str
    backend: str
    is_speaking: bool
    queue_depth: int
    latency_total: float
    latency_asr: float
    retry_in: float
    dropped_jobs: int
    audio_status: str
    audio_source: str
    audio_reconnects: int
    audio_error: str
    audio_device_index: int
    audio_device_name: str
    gender: str
    age: str
    visual_mode: str
    prompt_style: str
    language: str
    prompt_budget_mode: str
    dropped_final_oldest: int
    dropped_final_newest: int
    dropped_expired_results: int = 0

    def messages(self):
        return (
            OscMessage("/backend_status", self.backend_status),
            OscMessage("/backend", self.backend),
            OscMessage("/is_speaking", int(self.is_speaking)),
            OscMessage("/queue_depth", self.queue_depth),
            OscMessage("/latency_total", float(self.latency_total)),
            OscMessage("/latency_asr", float(self.latency_asr)),
            OscMessage("/retry_in", float(self.retry_in)),
            OscMessage("/dropped_jobs", self.dropped_jobs),
            OscMessage("/audio_status", self.audio_status),
            OscMessage("/audio_source", self.audio_source),
            OscMessage("/audio_reconnects", self.audio_reconnects),
            OscMessage("/audio_error", self.audio_error),
            OscMessage("/audio_device_index", self.audio_device_index),
            OscMessage("/audio_device_name", self.audio_device_name),
            OscMessage("/gender", self.gender),
            OscMessage("/age", self.age),
            OscMessage("/visual_mode", self.visual_mode),
            OscMessage("/prompt_style", self.prompt_style),
            OscMessage("/language", self.language),
            OscMessage("/prompt_budget_mode", self.prompt_budget_mode),
            OscMessage("/dropped_final_oldest", self.dropped_final_oldest),
            OscMessage("/dropped_final_newest", self.dropped_final_newest),
            OscMessage("/dropped_expired_results", self.dropped_expired_results),
        )


class OscOutputPublisher:
    """Thread-safe, failure-isolated OSC output transport."""

    def __init__(
        self,
        ip,
        port,
        *,
        status_interval=0.5,
        error_log_interval=5.0,
        logger=None,
        client_factory=None,
        clock=None,
    ):
        if status_interval <= 0 or error_log_interval <= 0:
            raise ValueError("OSC output intervals must be positive")
        self.ip = ip
        self.port = port
        self.status_interval = status_interval
        self.error_log_interval = error_log_interval
        self.logger = logger
        self.clock = clock or time.monotonic
        factory = client_factory or udp_client.SimpleUDPClient
        self.client = factory(ip, port)
        self._lock = threading.Lock()
        self._last_status_time = None
        self._last_error_log_time = None
        self._delivery_degraded = False
        self._failure_count = 0
        self._closed = False

    def send(self, address, value):
        with self._lock:
            return self._send_unlocked(address, value)

    def publish_status(self, snapshot, *, force=False):
        with self._lock:
            if self._closed:
                return False
            now = self.clock()
            if (
                not force
                and self._last_status_time is not None
                and now - self._last_status_time < self.status_interval
            ):
                return False
            self._last_status_time = now
            delivered = True
            for message in snapshot.messages():
                delivered = (
                    self._send_unlocked(message.address, message.value)
                    and delivered
                )
            return delivered

    def close(self):
        with self._lock:
            if self._closed:
                return
            self._closed = True
            close = getattr(self.client, "close", None)
            if close is None:
                close = getattr(getattr(self.client, "_sock", None), "close", None)
            if close is not None:
                try:
                    close()
                except Exception:
                    pass

    def _send_unlocked(self, address, value):
        if self._closed:
            return False
        try:
            self.client.send_message(address, value)
        except Exception as exc:
            self._failure_count += 1
            self._delivery_degraded = True
            self._log_delivery_error(address, exc)
            return False

        if self._delivery_degraded:
            self._delivery_degraded = False
            if self.logger is not None:
                self.logger.info(
                    "OSC output delivery recovered",
                    extra={
                        "event": "osc_output_recovered",
                        "ip": self.ip,
                        "port": self.port,
                        "failures": self._failure_count,
                    },
                )
        return True

    def _log_delivery_error(self, address, error):
        now = self.clock()
        if (
            self._last_error_log_time is not None
            and now - self._last_error_log_time < self.error_log_interval
        ):
            return
        self._last_error_log_time = now
        if self.logger is not None:
            cleaned_error = " ".join(str(error).split())[:240]
            self.logger.warning(
                "OSC output delivery failed; runtime processing will continue",
                extra={
                    "event": "osc_output_error",
                    "ip": self.ip,
                    "port": self.port,
                    "address": address,
                    "error": cleaned_error,
                    "failures": self._failure_count,
                },
            )


class RecordingOutputPublisher:
    """In-memory publisher for deterministic protocol and replay tests."""

    def __init__(self, *, status_interval=0.5, clock=None):
        if status_interval <= 0:
            raise ValueError("OSC status interval must be positive")
        self.status_interval = status_interval
        self.clock = clock or time.monotonic
        self._lock = threading.Lock()
        self._messages = []
        self._last_status_time = None
        self._closed = False

    @property
    def messages(self):
        with self._lock:
            return tuple(self._messages)

    def send(self, address, value):
        with self._lock:
            if self._closed:
                return False
            self._messages.append(OscMessage(address, value))
            return True

    def publish_status(self, snapshot, *, force=False):
        with self._lock:
            if self._closed:
                return False
            now = self.clock()
            if (
                not force
                and self._last_status_time is not None
                and now - self._last_status_time < self.status_interval
            ):
                return False
            self._last_status_time = now
            self._messages.extend(snapshot.messages())
            return True

    def close(self):
        with self._lock:
            self._closed = True


class NullOutputPublisher:
    def send(self, _address, _value):
        return True

    def publish_status(self, _snapshot, *, force=False):
        return True

    def close(self):
        return None
