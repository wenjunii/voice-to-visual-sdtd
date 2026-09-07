import threading
from collections import deque
from dataclasses import dataclass, replace
from typing import Optional


FINAL_OVERFLOW_POLICIES = {"drop_oldest", "drop_newest"}


@dataclass(frozen=True)
class ScheduledJob:
    job_id: int
    segment: object
    created_at: float
    ready_at: float
    attempts: int = 0

    @property
    def is_final(self):
        return bool(self.segment.is_final)


@dataclass(frozen=True)
class FinalJobSubmission:
    job: Optional[ScheduledJob]
    dropped_job: Optional[ScheduledJob] = None
    drop_reason: str = ""

    @property
    def accepted(self):
        return self.job is not None

    def __bool__(self):
        return self.accepted


@dataclass(frozen=True)
class SchedulerMetrics:
    queue_depth: int
    final_queue_depth: int
    has_partial: bool
    submitted_finals: int
    submitted_partials: int
    replaced_partials: int
    dropped_stale: int
    dropped_finals: int
    dropped_final_oldest: int
    dropped_final_newest: int
    retries: int
    processed: int
    failed: int


class RealtimeJobScheduler:
    """Prioritize finalized audio while retaining only the newest partial snapshot."""

    def __init__(
        self,
        max_final_jobs=8,
        final_overflow_policy="drop_oldest",
        partial_max_age_seconds=4.0,
        final_max_age_seconds=30.0,
    ):
        if max_final_jobs < 1:
            raise ValueError("max_final_jobs must be positive")
        if final_overflow_policy not in FINAL_OVERFLOW_POLICIES:
            raise ValueError(
                "final_overflow_policy must be one of: "
                + ", ".join(sorted(FINAL_OVERFLOW_POLICIES))
            )
        self.max_final_jobs = max_final_jobs
        self.final_overflow_policy = final_overflow_policy
        self.partial_max_age_seconds = partial_max_age_seconds
        self.final_max_age_seconds = final_max_age_seconds
        self._finals = deque()
        self._partial = None
        self._last_partial_key = None
        self._next_job_id = 1
        self._lock = threading.Lock()
        self._submitted_finals = 0
        self._submitted_partials = 0
        self._replaced_partials = 0
        self._dropped_stale = 0
        self._dropped_final_oldest = 0
        self._dropped_final_newest = 0
        self._retries = 0
        self._processed = 0
        self._failed = 0

    def submit_final(self, segment, now):
        with self._lock:
            self._purge_stale_locked(now)
            if (
                self._partial is not None
                and self._partial.segment.segment_id == segment.segment_id
            ):
                self._partial = None
            job = self._new_job(segment, now)
            if len(self._finals) >= self.max_final_jobs:
                if self.final_overflow_policy == "drop_newest":
                    self._dropped_final_newest += 1
                    return FinalJobSubmission(
                        job=None,
                        dropped_job=job,
                        drop_reason="newest_capacity",
                    )
                dropped_job = self._drop_oldest_final_locked()
                self._dropped_final_oldest += 1
            else:
                dropped_job = None
            self._finals.append(job)
            self._submitted_finals += 1
            return FinalJobSubmission(
                job=job,
                dropped_job=dropped_job,
                drop_reason=("oldest_capacity" if dropped_job else ""),
            )

    def submit_partial(self, segment, now):
        with self._lock:
            key = _segment_key(segment)
            if key == self._last_partial_key:
                return self._partial
            if self._partial is not None:
                self._replaced_partials += 1
            job = self._new_job(segment, now)
            self._partial = job
            self._last_partial_key = key
            self._submitted_partials += 1
            return job

    def next_job(self, now):
        with self._lock:
            self._purge_stale_locked(now)
            for index, job in enumerate(self._finals):
                if job.ready_at <= now:
                    del self._finals[index]
                    return job
            if self._partial is not None and self._partial.ready_at <= now:
                job = self._partial
                self._partial = None
                return job
            return None

    def retry_final(self, job, now, delay_seconds):
        if not job.is_final:
            return FinalJobSubmission(job=None)
        with self._lock:
            self._purge_stale_locked(now)
            if self._is_stale(job, now):
                self._dropped_stale += 1
                return FinalJobSubmission(
                    job=None,
                    dropped_job=job,
                    drop_reason="stale",
                )
            if len(self._finals) >= self.max_final_jobs:
                self._dropped_final_oldest += 1
                return FinalJobSubmission(
                    job=None,
                    dropped_job=job,
                    drop_reason="oldest_capacity",
                )
            retried = replace(
                job,
                ready_at=now + max(0.0, delay_seconds),
                attempts=job.attempts + 1,
            )
            self._finals.appendleft(retried)
            self._retries += 1
            return FinalJobSubmission(job=retried)

    def mark_processed(self):
        with self._lock:
            self._processed += 1

    def clear(self):
        """Discard pending work for an explicit reset, preserving run metrics."""
        with self._lock:
            discarded = len(self._finals) + int(self._partial is not None)
            self._finals.clear()
            self._partial = None
            self._last_partial_key = None
            return discarded

    def mark_failed(self):
        with self._lock:
            self._failed += 1

    def metrics(self, now=None):
        with self._lock:
            if now is not None:
                self._purge_stale_locked(now)
            return SchedulerMetrics(
                queue_depth=len(self._finals) + int(self._partial is not None),
                final_queue_depth=len(self._finals),
                has_partial=self._partial is not None,
                submitted_finals=self._submitted_finals,
                submitted_partials=self._submitted_partials,
                replaced_partials=self._replaced_partials,
                dropped_stale=self._dropped_stale,
                dropped_finals=(
                    self._dropped_final_oldest
                    + self._dropped_final_newest
                ),
                dropped_final_oldest=self._dropped_final_oldest,
                dropped_final_newest=self._dropped_final_newest,
                retries=self._retries,
                processed=self._processed,
                failed=self._failed,
            )

    def _new_job(self, segment, now):
        job = ScheduledJob(
            job_id=self._next_job_id,
            segment=segment,
            created_at=now,
            ready_at=now,
        )
        self._next_job_id += 1
        return job

    def _drop_oldest_final_locked(self):
        oldest_index = min(
            range(len(self._finals)),
            key=lambda index: (
                self._finals[index].created_at,
                self._finals[index].job_id,
            ),
        )
        dropped_job = self._finals[oldest_index]
        del self._finals[oldest_index]
        return dropped_job

    def _purge_stale_locked(self, now):
        retained = deque()
        for job in self._finals:
            if self._is_stale(job, now):
                self._dropped_stale += 1
            else:
                retained.append(job)
        self._finals = retained

        if self._partial is not None and self._is_stale(self._partial, now):
            self._partial = None
            self._dropped_stale += 1

    def _is_stale(self, job, now):
        max_age = self.final_max_age_seconds if job.is_final else self.partial_max_age_seconds
        return max_age > 0 and now - job.created_at > max_age


def _segment_key(segment):
    return segment.segment_id, segment.version, segment.is_final
