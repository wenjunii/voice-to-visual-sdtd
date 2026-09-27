import unittest
from datetime import datetime, timezone

from backend_errors import exponential_backoff, normalize_retry_delay, retry_after_seconds


class BackendErrorTests(unittest.TestCase):
    def test_reads_numeric_retry_after(self):
        for value, expected in (("2.5", 2.5), ("0", 0.0), (" 60 ", 60.0), (0, 0.0)):
            with self.subTest(value=value):
                self.assertEqual(retry_after_seconds({"Retry-After": value}), expected)

    def test_reads_http_date_retry_after(self):
        now = datetime(2026, 1, 1, tzinfo=timezone.utc)
        headers = {"Retry-After": "Thu, 01 Jan 2026 00:00:03 GMT"}

        self.assertEqual(retry_after_seconds(headers, now=now), 3.0)

    def test_uses_default_for_invalid_retry_after(self):
        for value in (
            None, "", " ", "later", "nan", "NaN", "inf", "Infinity", "-inf",
            "1e309", "-1e309", "-2.5", [], {}, True, False, 10 ** 400,
        ):
            with self.subTest(value=value):
                self.assertEqual(retry_after_seconds({"Retry-After": value}, default=4), 4)
                self.assertIsNone(retry_after_seconds({"Retry-After": value}))

    def test_missing_header_and_invalid_default_cannot_return_non_finite_delay(self):
        for headers in (None, {}, {"Retry-After": "invalid"}):
            with self.subTest(headers=headers):
                self.assertEqual(retry_after_seconds(headers, default=2.5), 2.5)
                for fallback in (float("inf"), float("nan"), -1, "invalid"):
                    self.assertIsNone(retry_after_seconds(headers, default=fallback))
        self.assertEqual(retry_after_seconds({"Retry-After": "3"}, default=float("inf")), 3.0)

    def test_past_http_date_means_zero_delay_and_malformed_dates_use_fallback(self):
        now = datetime(2026, 1, 1, tzinfo=timezone.utc)
        self.assertEqual(
            retry_after_seconds({"Retry-After": "Wed, 31 Dec 2025 23:59:59 GMT"}, now=now),
            0.0,
        )
        self.assertEqual(
            retry_after_seconds({"Retry-After": "Thu, 01 Jan 2026 00:00:03"}, now=now),
            3.0,
        )
        for value in ("Thu, 32 Jan 2026 00:00:03 GMT", "Thu, 01 Jan 99999 00:00:03 GMT"):
            with self.subTest(value=value):
                self.assertEqual(retry_after_seconds({"Retry-After": value}, default=4, now=now), 4)

    def test_normalizes_direct_backend_hints_without_clamping_valid_server_delays(self):
        for value, expected in ((0, 0.0), (2.5, 2.5), ("60", 60.0), (3600, 3600.0)):
            with self.subTest(value=value):
                self.assertEqual(normalize_retry_delay(value), expected)
        for value in (None, True, False, -1, float("inf"), float("-inf"), float("nan"), "bad", [], 10 ** 400):
            with self.subTest(value=value):
                self.assertIsNone(normalize_retry_delay(value))

    def test_exponential_backoff_is_capped(self):
        self.assertEqual(exponential_backoff(0, base_seconds=1, max_seconds=5), 1)
        self.assertEqual(exponential_backoff(3, base_seconds=1, max_seconds=5), 5)
        self.assertEqual(exponential_backoff(100000, base_seconds=1, max_seconds=5), 5)


if __name__ == "__main__":
    unittest.main()
