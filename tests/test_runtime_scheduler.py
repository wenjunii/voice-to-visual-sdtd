import unittest

import numpy as np

from runtime_scheduler import RealtimeJobScheduler
from streaming_core import AudioSegment


def make_segment(segment_id, version, is_final=False):
    return AudioSegment(
        segment_id=segment_id,
        version=version,
        samples=np.array([version], dtype=np.int16),
        is_final=is_final,
    )


class RealtimeJobSchedulerTests(unittest.TestCase):
    def test_keeps_only_the_newest_partial(self):
        scheduler = RealtimeJobScheduler()
        scheduler.submit_partial(make_segment(1, 1), now=1.0)
        scheduler.submit_partial(make_segment(1, 2), now=1.1)

        job = scheduler.next_job(now=1.2)
        metrics = scheduler.metrics()

        self.assertEqual(job.segment.version, 2)
        self.assertEqual(metrics.replaced_partials, 1)
        self.assertEqual(metrics.queue_depth, 0)

    def test_does_not_requeue_an_unchanged_partial(self):
        scheduler = RealtimeJobScheduler()
        segment = make_segment(1, 1)
        scheduler.submit_partial(segment, now=1.0)
        scheduler.next_job(now=1.0)

        duplicate = scheduler.submit_partial(segment, now=1.1)

        self.assertIsNone(duplicate)
        self.assertIsNone(scheduler.next_job(now=1.1))

    def test_prioritizes_final_audio_and_removes_its_partial(self):
        scheduler = RealtimeJobScheduler()
        scheduler.submit_partial(make_segment(1, 2), now=1.0)
        scheduler.submit_final(make_segment(1, 3, is_final=True), now=1.1)
        scheduler.submit_partial(make_segment(2, 1), now=1.2)

        first = scheduler.next_job(now=1.3)
        second = scheduler.next_job(now=1.3)

        self.assertTrue(first.is_final)
        self.assertEqual(first.segment.segment_id, 1)
        self.assertFalse(second.is_final)
        self.assertEqual(second.segment.segment_id, 2)

    def test_retried_final_waits_until_ready(self):
        scheduler = RealtimeJobScheduler()
        scheduler.submit_final(make_segment(3, 4, is_final=True), now=1.0)
        job = scheduler.next_job(now=1.0)

        self.assertTrue(scheduler.retry_final(job, now=1.1, delay_seconds=2.0))
        self.assertIsNone(scheduler.next_job(now=3.0))
        retried = scheduler.next_job(now=3.1)

        self.assertEqual(retried.attempts, 1)
        self.assertEqual(scheduler.metrics().retries, 1)

    def test_drops_oldest_final_by_default_to_admit_fresher_speech(self):
        scheduler = RealtimeJobScheduler(max_final_jobs=1)
        first = scheduler.submit_final(
            make_segment(1, 1, is_final=True),
            now=1.0,
        )
        second = scheduler.submit_final(
            make_segment(2, 1, is_final=True),
            now=1.1,
        )

        queued = scheduler.next_job(now=1.1)
        metrics = scheduler.metrics()

        self.assertTrue(first.accepted)
        self.assertTrue(second.accepted)
        self.assertEqual(second.drop_reason, "oldest_capacity")
        self.assertEqual(second.dropped_job.segment.segment_id, 1)
        self.assertEqual(queued.segment.segment_id, 2)
        self.assertEqual(metrics.submitted_finals, 2)
        self.assertEqual(metrics.dropped_finals, 1)
        self.assertEqual(metrics.dropped_final_oldest, 1)
        self.assertEqual(metrics.dropped_final_newest, 0)

    def test_can_drop_newest_final_to_preserve_complete_fifo_history(self):
        scheduler = RealtimeJobScheduler(
            max_final_jobs=1,
            final_overflow_policy="drop_newest",
        )
        scheduler.submit_final(make_segment(1, 1, is_final=True), now=1.0)
        second = scheduler.submit_final(
            make_segment(2, 1, is_final=True),
            now=1.1,
        )

        queued = scheduler.next_job(now=1.1)
        metrics = scheduler.metrics()

        self.assertFalse(second.accepted)
        self.assertEqual(second.drop_reason, "newest_capacity")
        self.assertEqual(second.dropped_job.segment.segment_id, 2)
        self.assertEqual(queued.segment.segment_id, 1)
        self.assertEqual(metrics.submitted_finals, 1)
        self.assertEqual(metrics.dropped_final_oldest, 0)
        self.assertEqual(metrics.dropped_final_newest, 1)

    def test_older_retry_never_evicts_fresher_queued_speech(self):
        scheduler = RealtimeJobScheduler(
            max_final_jobs=1,
            final_overflow_policy="drop_oldest",
        )
        scheduler.submit_final(make_segment(1, 1, is_final=True), now=1.0)
        older_job = scheduler.next_job(now=1.0)
        scheduler.submit_final(make_segment(2, 1, is_final=True), now=1.1)

        retry = scheduler.retry_final(
            older_job,
            now=1.2,
            delay_seconds=0.0,
        )
        queued = scheduler.next_job(now=1.2)
        metrics = scheduler.metrics()

        self.assertFalse(retry.accepted)
        self.assertEqual(retry.drop_reason, "oldest_capacity")
        self.assertEqual(queued.segment.segment_id, 2)
        self.assertEqual(metrics.retries, 0)
        self.assertEqual(metrics.dropped_final_oldest, 1)

    def test_drops_stale_work_separately_from_capacity_drops(self):
        scheduler = RealtimeJobScheduler(
            max_final_jobs=2,
            partial_max_age_seconds=1.0,
            final_max_age_seconds=1.0,
        )
        scheduler.submit_final(make_segment(1, 1, is_final=True), now=1.0)
        scheduler.submit_partial(make_segment(3, 1), now=1.1)

        self.assertIsNone(scheduler.next_job(now=2.2))
        metrics = scheduler.metrics()
        self.assertEqual(metrics.dropped_finals, 0)
        self.assertEqual(metrics.dropped_stale, 2)

    def test_rejects_unknown_overflow_policy(self):
        with self.assertRaisesRegex(
            ValueError,
            "final_overflow_policy must be one of",
        ):
            RealtimeJobScheduler(final_overflow_policy="discard_random")

    def test_clear_removes_finals_retries_and_partials_without_capacity_drops(self):
        scheduler = RealtimeJobScheduler()
        scheduler.submit_final(make_segment(1, 1, is_final=True), now=1.0)
        job = scheduler.next_job(now=1.0)
        scheduler.retry_final(job, now=1.1, delay_seconds=10.0)
        scheduler.submit_final(make_segment(2, 1, is_final=True), now=1.2)
        partial = make_segment(3, 1)
        scheduler.submit_partial(partial, now=1.3)

        self.assertEqual(scheduler.clear(), 3)
        self.assertIsNone(scheduler.next_job(now=1.4))
        self.assertIsNone(scheduler.next_job(now=12.0))
        metrics = scheduler.metrics()
        self.assertEqual(metrics.submitted_finals, 2)
        self.assertEqual(metrics.retries, 1)
        self.assertEqual(metrics.dropped_finals, 0)
        self.assertEqual(metrics.dropped_stale, 0)
        self.assertEqual(scheduler.clear(), 0)
        self.assertIsNotNone(scheduler.submit_partial(partial, now=12.1))

    def test_result_age_limits_accept_boundary_and_reject_older_results(self):
        for is_final, limit in ((False, 4.0), (True, 30.0)):
            for age, accepted in ((limit - 0.1, True), (limit, True), (limit + 0.1, False)):
                with self.subTest(is_final=is_final, age=age):
                    scheduler = RealtimeJobScheduler()
                    submit = scheduler.submit_final if is_final else scheduler.submit_partial
                    submit(make_segment(1, 1, is_final=is_final), now=100.0)
                    job = scheduler.next_job(now=100.0)

                    self.assertEqual(scheduler.complete_job(job, 100.0 + age), accepted)
                    metrics = scheduler.metrics(now=200.0)
                    self.assertEqual(metrics.processed, int(accepted))
                    self.assertEqual(metrics.dropped_stale, int(not accepted))
                    self.assertEqual(metrics.dropped_expired_results, int(not accepted))
                    self.assertEqual(metrics.failed, 0)

    def test_zero_disables_only_the_selected_job_types_age_limit(self):
        for unlimited_final in (False, True):
            with self.subTest(unlimited_final=unlimited_final):
                scheduler = RealtimeJobScheduler(
                    final_max_age_seconds=0 if unlimited_final else 1,
                    partial_max_age_seconds=1 if unlimited_final else 0,
                )
                scheduler.submit_final(make_segment(1, 1, is_final=True), now=1.0)
                scheduler.submit_partial(make_segment(2, 1), now=1.0)
                final = scheduler.next_job(now=1.0)
                partial = scheduler.next_job(now=1.0)

                self.assertEqual(scheduler.complete_job(final, 1000.0), unlimited_final)
                self.assertEqual(scheduler.complete_job(partial, 1000.0), not unlimited_final)
                self.assertEqual(scheduler.metrics().dropped_expired_results, 1)

    def test_successful_retry_keeps_the_original_submission_deadline(self):
        scheduler = RealtimeJobScheduler()
        scheduler.submit_final(make_segment(1, 1, is_final=True), now=100.0)
        original = scheduler.next_job(now=100.0)
        self.assertTrue(scheduler.retry_final(original, now=110.0, delay_seconds=2.0))
        retry = scheduler.next_job(now=112.0)

        self.assertEqual(retry.created_at, 100.0)
        self.assertFalse(scheduler.complete_job(retry, now=131.0))
        metrics = scheduler.metrics()
        self.assertEqual(metrics.retries, 1)
        self.assertEqual(metrics.processed, 0)
        self.assertEqual(metrics.dropped_expired_results, 1)

    def test_expired_result_counter_is_a_subset_of_total_stale_drops(self):
        scheduler = RealtimeJobScheduler(final_max_age_seconds=2.0)
        scheduler.submit_final(make_segment(1, 1, is_final=True), now=100.0)
        running_job = scheduler.next_job(now=100.0)
        scheduler.submit_final(make_segment(2, 1, is_final=True), now=100.0)

        self.assertFalse(scheduler.complete_job(running_job, now=103.0))
        metrics = scheduler.metrics(now=103.0)
        self.assertEqual(metrics.dropped_stale, 2)
        self.assertEqual(metrics.dropped_expired_results, 1)
        self.assertEqual(metrics.dropped_finals, 0)
        scheduler.clear()
        self.assertEqual(scheduler.metrics(), metrics)


if __name__ == "__main__":
    unittest.main()
