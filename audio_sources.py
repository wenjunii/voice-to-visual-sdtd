import time
import wave
from pathlib import Path

import numpy as np

from audio_runtime import get_audio_input_device, list_audio_input_devices


class AudioSourceFinished(EOFError):
    """Raised when a finite audio source has no more samples."""


class AudioSourceStopped(Exception):
    """Raised when cooperative cancellation interrupts a source read."""


def _load_pyaudio():
    try:
        import pyaudio
    except ImportError as exc:
        raise RuntimeError(
            "PyAudio is not installed. Run 'python transcriber.py --diagnose' "
            "for setup details, then install the project requirements."
        ) from exc
    return pyaudio


def list_system_audio_input_devices(pyaudio_module=None):
    """List input devices while keeping PyAudio lifetime inside this module."""

    module = pyaudio_module or _load_pyaudio()
    audio_interface = module.PyAudio()
    try:
        return list_audio_input_devices(audio_interface)
    finally:
        try:
            audio_interface.terminate()
        except Exception:
            pass


class PyAudioSource:
    kind = "microphone"
    finite = False
    reconnectable = True

    def __init__(
        self,
        *,
        device_index=None,
        sample_rate=16000,
        chunk_samples=1024,
        channels=1,
        pyaudio_module=None,
    ):
        if sample_rate <= 0 or chunk_samples <= 0 or channels <= 0:
            raise ValueError(
                "sample rate, chunk size, and channels must be positive"
            )
        self.configured_device_index = device_index
        self.sample_rate = sample_rate
        self.chunk_samples = chunk_samples
        self.channels = channels
        self._pyaudio_module = pyaudio_module
        self._audio_interface = None
        self._stream = None
        self.device_index = device_index if device_index is not None else -1
        self.name = "system default" if device_index is None else ""

    def open(self):
        if self._stream is not None:
            raise RuntimeError("The microphone source is already open")

        module = self._pyaudio_module or _load_pyaudio()
        audio_interface = module.PyAudio()
        try:
            device = get_audio_input_device(
                audio_interface,
                self.configured_device_index,
            )
            stream = audio_interface.open(
                format=module.paInt16,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                input_device_index=device.index,
                frames_per_buffer=self.chunk_samples,
            )
        except Exception:
            try:
                audio_interface.terminate()
            except Exception:
                pass
            raise

        self._audio_interface = audio_interface
        self._stream = stream
        self.device_index = device.index
        self.name = device.name
        return self

    def read(self):
        if self._stream is None:
            raise RuntimeError("The microphone source is not open")
        data = self._stream.read(
            self.chunk_samples,
            exception_on_overflow=False,
        )
        if not data:
            raise OSError("microphone returned no audio data")
        return data

    def close(self):
        stream = self._stream
        audio_interface = self._audio_interface
        self._stream = None
        self._audio_interface = None

        if stream is not None:
            try:
                stream.stop_stream()
            except Exception:
                pass
            try:
                stream.close()
            except Exception:
                pass
        if audio_interface is not None:
            try:
                audio_interface.terminate()
            except Exception:
                pass


class WavReplaySource:
    kind = "wav_replay"
    finite = True
    reconnectable = False
    device_index = -1

    def __init__(
        self,
        wav_path,
        *,
        sample_rate=16000,
        chunk_samples=1024,
        realtime=True,
        stop_event=None,
    ):
        if sample_rate <= 0 or chunk_samples <= 0:
            raise ValueError("sample rate and chunk size must be positive")
        self.path = Path(wav_path).expanduser().resolve()
        self.sample_rate = sample_rate
        self.chunk_samples = chunk_samples
        self.realtime = realtime
        self.stop_event = stop_event
        self.samples = load_wav_samples(
            self.path,
            target_sample_rate=sample_rate,
        )
        self.name = f"WAV replay: {self.path.name}"
        self._cursor = 0
        self._started_at = None
        self._open = False

    @property
    def duration_seconds(self):
        return len(self.samples) / self.sample_rate

    def open(self):
        if self._open:
            raise RuntimeError("The WAV replay source is already open")
        self._cursor = 0
        self._started_at = time.monotonic()
        self._open = True
        return self

    def read(self):
        if not self._open:
            raise RuntimeError("The WAV replay source is not open")
        if self._cursor >= len(self.samples):
            raise AudioSourceFinished(str(self.path))

        end = min(self._cursor + self.chunk_samples, len(self.samples))
        chunk = self.samples[self._cursor:end]
        self._cursor = end

        if self.realtime:
            target_time = self._started_at + self._cursor / self.sample_rate
            delay = max(0.0, target_time - time.monotonic())
            if delay and self.stop_event is not None:
                if self.stop_event.wait(delay):
                    raise AudioSourceStopped()
            elif delay:
                time.sleep(delay)

        return chunk.tobytes()

    def close(self):
        self._open = False


def load_wav_samples(wav_path, *, target_sample_rate=16000):
    if target_sample_rate <= 0:
        raise ValueError("target sample rate must be positive")
    path = Path(wav_path).expanduser().resolve()
    try:
        with wave.open(str(path), "rb") as wav_file:
            channels = wav_file.getnchannels()
            sample_width = wav_file.getsampwidth()
            source_rate = wav_file.getframerate()
            frame_count = wav_file.getnframes()
            compression = wav_file.getcomptype()
            if compression != "NONE":
                raise ValueError("WAV audio requires uncompressed PCM audio.")
            if sample_width != 2:
                raise ValueError("WAV audio requires 16-bit PCM audio.")
            if channels < 1:
                raise ValueError("WAV audio requires at least one audio channel.")
            if source_rate < 1:
                raise ValueError("WAV audio has an invalid sample rate.")

            # getnframes() rounds down a partial final frame. Read one extra
            # frame to expose leftover bytes, without crossing the data chunk.
            raw_audio = wav_file.readframes(frame_count + 1)
    except (OSError, wave.Error) as exc:
        raise ValueError(f"Could not read WAV file '{path}': {exc}") from exc
    except (EOFError, RuntimeError) as exc:
        # wave also raises EOFError for short headers and RuntimeError when
        # a malformed chunk would seek outside its enclosing RIFF container.
        raise ValueError(
            f"Could not read WAV file '{path}': "
            "malformed or incomplete WAV structure."
        ) from exc

    frame_bytes = channels * sample_width
    if len(raw_audio) % frame_bytes:
        raise ValueError("WAV audio ends with an incomplete PCM frame.")
    expected_bytes = frame_count * frame_bytes
    if len(raw_audio) != expected_bytes:
        raise ValueError(
            "WAV audio is truncated: "
            f"expected {expected_bytes} PCM bytes, read {len(raw_audio)}."
        )
    if not raw_audio:
        raise ValueError("WAV audio contains no audio samples.")

    samples = np.frombuffer(raw_audio, dtype="<i2")
    if channels > 1:
        samples = (
            samples.reshape(-1, channels)
            .astype(np.int32)
            .mean(axis=1)
            .astype(np.int16)
        )
    if source_rate != target_sample_rate:
        target_length = max(
            1,
            round(len(samples) * target_sample_rate / source_rate),
        )
        source_positions = np.arange(len(samples), dtype=np.float64)
        target_positions = np.linspace(
            0,
            len(samples) - 1,
            num=target_length,
        )
        samples = np.interp(
            target_positions,
            source_positions,
            samples,
        ).astype(np.int16)
    return samples.copy()
