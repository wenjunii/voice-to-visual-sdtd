import math
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime


class RetryableTranscriptionError(RuntimeError):
    def __init__(self, message, retry_after=None):
        super().__init__(message)
        self.retry_after = retry_after


def normalize_retry_delay(value):
    """Return finite, non-negative seconds, or None for an unusable hint."""
    if isinstance(value, bool):
        return None
    try:
        seconds = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return seconds if math.isfinite(seconds) and seconds >= 0 else None


def retry_after_seconds(headers, default=None, now=None):
    """Parse numeric/date retry hints without returning an invalid delay."""
    fallback = normalize_retry_delay(default)
    value = headers.get("Retry-After") if headers is not None else None
    seconds = normalize_retry_delay(value)
    if seconds is not None:
        return seconds
    if not isinstance(value, str) or not value.strip():
        return fallback

    try:
        retry_at = parsedate_to_datetime(value)
        if retry_at.tzinfo is None:
            retry_at = retry_at.replace(tzinfo=timezone.utc)
        current = now or datetime.now(timezone.utc)
        seconds = normalize_retry_delay(max(0.0, (retry_at - current).total_seconds()))
        return seconds if seconds is not None else fallback
    except (TypeError, ValueError, OverflowError):
        return fallback


def exponential_backoff(attempt, base_seconds=1.0, max_seconds=10.0):
    if base_seconds < 0 or max_seconds < 0:
        raise ValueError("backoff durations must be non-negative")

    delay = min(max_seconds, base_seconds)
    remaining = max(0, int(attempt))
    while remaining and 0 < delay < max_seconds:
        delay = min(max_seconds, delay * 2)
        remaining -= 1
    return delay
