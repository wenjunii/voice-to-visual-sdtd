import unittest
from unittest.mock import Mock, patch

from transcriber import read_control_key


class ConsoleControlTests(unittest.TestCase):
    def test_headless_host_does_not_need_windows_console_module(self):
        with patch("transcriber.msvcrt", None):
            self.assertIsNone(read_control_key())

    def test_windows_shortcut_is_decoded_and_normalized(self):
        console = Mock()
        console.kbhit.return_value = True
        console.getch.return_value = b"M"
        with patch("transcriber.msvcrt", console):
            self.assertEqual(read_control_key(), "m")

    def test_idle_console_does_not_block_reading_a_key(self):
        console = Mock()
        console.kbhit.return_value = False
        with patch("transcriber.msvcrt", console):
            self.assertIsNone(read_control_key())
        console.getch.assert_not_called()

    def test_non_text_console_key_is_ignored(self):
        console = Mock()
        console.kbhit.return_value = True
        console.getch.return_value = b"\xe0"
        with patch("transcriber.msvcrt", console):
            self.assertIsNone(read_control_key())


if __name__ == "__main__":
    unittest.main()
