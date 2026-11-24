import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from kaggle_status import parse_status


class StatusParsingTests(unittest.TestCase):
    def test_complete(self):
        self.assertEqual(
            parse_status('yeatasimbillah/test has status "KernelWorkerStatus.COMPLETE"'),
            "complete",
        )
        self.assertEqual(parse_status('user/test has status "running"'), "running")

    def test_unknown_fails_closed(self):
        with self.assertRaises(ValueError):
            parse_status("API temporarily unavailable")
        with self.assertRaises(ValueError):
            parse_status('user/test has status "MAGIC_UNKNOWN"')


if __name__ == "__main__":
    unittest.main()
