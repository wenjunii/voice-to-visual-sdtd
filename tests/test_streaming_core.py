import unittest

import numpy as np

from streaming_core import AudioSegmenter, TranscriptStabilizer


class AudioSegmenterTests(unittest.TestCase):
    def make_segmenter(self, **overrides):
        settings = {
            "sample_rate": 10,
            "chunk_samples": 2,
            "pre_roll_seconds": 0.4,
            "end_silence_seconds": 0.4,
            "max_segment_seconds": 1.0,
            "overlap_seconds": 0.2,
        }
        settings.update(overrides)
        return AudioSegmenter(**settings)

    def test_preserves_quiet_pre_roll_and_internal_silence(self):
        segmenter = self.make_segmenter()
        segmenter.add_chunk(np.array([1, 1], dtype=np.int16), False)
        segmenter.add_chunk(np.array([2, 2], dtype=np.int16), False)
        segmenter.add_chunk(np.array([3, 3], dtype=np.int16), True)
        segmenter.add_chunk(np.array([0, 0], dtype=np.int16), False)

        snapshot = segmenter.snapshot()

        np.testing.assert_array_equal(
            snapshot.samples,
            np.array([1, 1, 2, 2, 3, 3, 0, 0], dtype=np.int16),
        )
        self.assertFalse(snapshot.is_final)

    def test_silence_finalizes_the_active_segment(self):
        segmenter = self.make_segmenter(pre_roll_seconds=0.2)
        segmenter.add_chunk(np.array([5, 5], dtype=np.int16), True)
        self.assertEqual(segmenter.add_chunk(np.array([0, 0], dtype=np.int16), False), [])

        completed = segmenter.add_chunk(np.array([0, 0], dtype=np.int16), False)

        self.assertEqual(len(completed), 1)
        self.assertTrue(completed[0].is_final)
        self.assertFalse(segmenter.active)

    def test_long_speech_rolls_over_with_overlap(self):
        segmenter = self.make_segmenter(
            pre_roll_seconds=0.2,
            max_segment_seconds=0.6,
            overlap_seconds=0.2,
        )
        completed = []
        for value in (1, 2, 3):
            completed.extend(segmenter.add_chunk(np.array([value, value], dtype=np.int16), True))

        self.assertEqual(len(completed), 1)
        np.testing.assert_array_equal(
            completed[0].samples,
            np.array([1, 1, 2, 2, 3, 3], dtype=np.int16),
        )
        overlap_snapshot = segmenter.snapshot()
        np.testing.assert_array_equal(overlap_snapshot.samples, np.array([3, 3], dtype=np.int16))
        self.assertNotEqual(completed[0].segment_id, overlap_snapshot.segment_id)

    def test_input_interruption_finalizes_active_audio_and_clears_pre_roll(self):
        segmenter = self.make_segmenter(pre_roll_seconds=0.4)
        segmenter.add_chunk(np.array([1, 1], dtype=np.int16), False)
        segmenter.add_chunk(np.array([2, 2], dtype=np.int16), True)

        interrupted = segmenter.interrupt()

        self.assertTrue(interrupted.is_final)
        np.testing.assert_array_equal(
            interrupted.samples,
            np.array([1, 1, 2, 2], dtype=np.int16),
        )
        self.assertFalse(segmenter.active)
        self.assertIsNone(segmenter.snapshot())

        segmenter.add_chunk(np.array([3, 3], dtype=np.int16), True)
        np.testing.assert_array_equal(
            segmenter.snapshot().samples,
            np.array([3, 3], dtype=np.int16),
        )


    def test_reset_discards_active_audio_and_pre_roll_without_reusing_ids(self):
        segmenter = self.make_segmenter()
        segmenter.add_chunk(np.array([1, 1], dtype=np.int16), False)
        segmenter.add_chunk(np.array([2, 2], dtype=np.int16), True)
        previous_id = segmenter.snapshot().segment_id

        self.assertIsNone(segmenter.reset())
        self.assertFalse(segmenter.active)
        self.assertIsNone(segmenter.snapshot())

        segmenter.add_chunk(np.array([3, 3], dtype=np.int16), False)
        segmenter.reset()
        segmenter.add_chunk(np.array([4, 4], dtype=np.int16), True)
        snapshot = segmenter.snapshot()
        self.assertGreater(snapshot.segment_id, previous_id)
        np.testing.assert_array_equal(snapshot.samples, [4, 4])


class TranscriptStabilizerTests(unittest.TestCase):
    def test_confirms_shared_prefix_across_two_updates(self):
        stabilizer = TranscriptStabilizer(agreement_updates=2)

        first = stabilizer.update("Hello there")
        second = stabilizer.update("Hello there, friend")
        third = stabilizer.update("Hello there friend today")

        self.assertFalse(first.changed)
        self.assertEqual(second.text, "Hello there,")
        self.assertTrue(second.changed)
        self.assertEqual(third.text, "Hello there friend")

    def test_final_update_flushes_unconfirmed_words(self):
        stabilizer = TranscriptStabilizer(agreement_updates=2)
        stabilizer.update("A bright city")

        final = stabilizer.update("A bright city at night", is_final=True)

        self.assertEqual(final.text, "A bright city at night")
        self.assertTrue(final.changed)
        self.assertTrue(final.is_final)


if __name__ == "__main__":
    unittest.main()
