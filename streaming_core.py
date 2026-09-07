import math
import re
from collections import deque
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class AudioSegment:
    segment_id: int
    version: int
    samples: np.ndarray
    is_final: bool


@dataclass(frozen=True)
class TranscriptUpdate:
    text: str
    changed: bool
    is_final: bool


class AudioSegmenter:
    """Build contiguous speech segments with pre-roll, silence hangover, and overlap."""

    def __init__(
        self,
        sample_rate,
        chunk_samples,
        pre_roll_seconds=0.32,
        end_silence_seconds=0.7,
        max_segment_seconds=6.0,
        overlap_seconds=0.5,
    ):
        if sample_rate <= 0 or chunk_samples <= 0:
            raise ValueError("sample_rate and chunk_samples must be positive")
        if end_silence_seconds <= 0 or max_segment_seconds <= 0:
            raise ValueError("silence and segment durations must be positive")
        if overlap_seconds < 0 or overlap_seconds >= max_segment_seconds:
            raise ValueError("overlap must be non-negative and shorter than a segment")

        self.sample_rate = sample_rate
        self.chunk_samples = chunk_samples
        self.pre_roll_chunks = max(1, math.ceil(pre_roll_seconds * sample_rate / chunk_samples))
        self.end_silence_samples = max(1, int(end_silence_seconds * sample_rate))
        self.max_segment_samples = max(chunk_samples, int(max_segment_seconds * sample_rate))
        self.overlap_chunks = max(0, math.ceil(overlap_seconds * sample_rate / chunk_samples))

        self._pre_roll = deque(maxlen=self.pre_roll_chunks)
        self._active_chunks = []
        self._active = False
        self._silent_samples = 0
        self._segment_id = 0
        self._version = 0

    @property
    def active(self):
        return self._active

    def add_chunk(self, samples, is_speech):
        chunk = np.asarray(samples, dtype=np.int16).reshape(-1).copy()
        if chunk.size == 0:
            return []

        completed = []
        self._version += 1

        if not self._active:
            if not is_speech:
                self._pre_roll.append(chunk)
                return completed

            self._active = True
            self._segment_id += 1
            self._active_chunks = list(self._pre_roll) + [chunk]
            self._pre_roll.clear()
            self._silent_samples = 0
        else:
            self._active_chunks.append(chunk)

        if is_speech:
            self._silent_samples = 0
        else:
            self._silent_samples += chunk.size

        if self._silent_samples >= self.end_silence_samples:
            completed.append(self._snapshot(is_final=True))
            trailing = self._active_chunks[-self.pre_roll_chunks:]
            self._reset_active()
            self._pre_roll.extend(trailing)
        elif self._active_sample_count() >= self.max_segment_samples:
            completed.append(self._snapshot(is_final=True))
            overlap = self._active_chunks[-self.overlap_chunks:] if self.overlap_chunks else []
            self._segment_id += 1
            self._active_chunks = list(overlap)
            self._silent_samples = min(
                self._silent_samples,
                sum(chunk.size for chunk in self._active_chunks),
            )

        return completed

    def snapshot(self, min_audio_seconds=0.0):
        if not self._active:
            return None
        if self._active_sample_count() < int(min_audio_seconds * self.sample_rate):
            return None
        return self._snapshot(is_final=False)

    def interrupt(self):
        """Finalize active speech and clear device-specific pre-roll after an input outage."""

        completed = self._snapshot(is_final=True) if self._active else None
        self.reset()
        return completed

    def reset(self):
        """Discard buffered speech and pre-roll without reusing segment IDs."""
        self._reset_active()
        self._pre_roll.clear()

    def _snapshot(self, is_final):
        if not self._active_chunks:
            return AudioSegment(
                segment_id=self._segment_id,
                version=self._version,
                samples=np.empty(0, dtype=np.int16),
                is_final=is_final,
            )
        return AudioSegment(
            segment_id=self._segment_id,
            version=self._version,
            samples=np.concatenate(self._active_chunks).copy(),
            is_final=is_final,
        )

    def _active_sample_count(self):
        return sum(chunk.size for chunk in self._active_chunks)

    def _reset_active(self):
        self._active = False
        self._active_chunks = []
        self._silent_samples = 0


class TranscriptStabilizer:
    """Confirm only the word prefix shared by consecutive ASR hypotheses."""

    def __init__(self, agreement_updates=2):
        if agreement_updates < 1:
            raise ValueError("agreement_updates must be at least 1")
        self.agreement_updates = agreement_updates
        self._hypotheses = deque(maxlen=agreement_updates)
        self._confirmed_words = []
        self._last_emitted = ""

    def update(self, text, is_final=False):
        words = _split_words(text)

        if is_final:
            final_words = words or self._confirmed_words
            final_text = " ".join(final_words).strip()
            changed = bool(final_text) and final_text != self._last_emitted
            self._last_emitted = final_text
            self._confirmed_words = list(final_words)
            self._hypotheses.clear()
            return TranscriptUpdate(final_text, changed, True)

        if not words:
            return TranscriptUpdate(self._last_emitted, False, False)

        self._hypotheses.append(words)
        if len(self._hypotheses) >= self.agreement_updates:
            prefix_length = _common_prefix_length(self._hypotheses)
            candidate = words[:prefix_length]
            if len(candidate) > len(self._confirmed_words):
                confirmed_prefix = candidate[:len(self._confirmed_words)]
                if _words_equal(confirmed_prefix, self._confirmed_words):
                    self._confirmed_words = list(candidate)

        stable_text = " ".join(self._confirmed_words).strip()
        changed = bool(stable_text) and stable_text != self._last_emitted
        if changed:
            self._last_emitted = stable_text
        return TranscriptUpdate(stable_text, changed, False)


def _split_words(text):
    return [word for word in (text or "").strip().split() if word]


def _normalize_word(word):
    normalized = re.sub(r"^\W+|\W+$", "", word, flags=re.UNICODE).casefold()
    return normalized or word.casefold()


def _words_equal(left, right):
    if len(left) != len(right):
        return False
    return all(_normalize_word(a) == _normalize_word(b) for a, b in zip(left, right))


def _common_prefix_length(sequences):
    if not sequences:
        return 0
    sequences = list(sequences)
    shortest = min(len(sequence) for sequence in sequences)
    for index in range(shortest):
        word = sequences[0][index]
        if any(_normalize_word(sequence[index]) != _normalize_word(word) for sequence in sequences[1:]):
            return index
    return shortest
