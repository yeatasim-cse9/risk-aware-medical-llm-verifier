import sys
import unittest
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from kaggle_status import check, parse_list, parse_status


class KaggleStatusTests(unittest.TestCase):
    def test_status_parser(self):
        self.assertEqual(
            parse_status('user/kernel has status "KernelWorkerStatus.COMPLETE"'),
            "complete",
        )
        with self.assertRaises(ValueError):
            parse_status("Unexpected response")

    def test_missing_csv_list(self):
        self.assertFalse(parse_list("ref,title\n", "user/target"))
        self.assertFalse(parse_list("No kernels found", "user/target"))
        self.assertFalse(parse_list("ref,title\nuser/other,Other\n", "user/target"))

    def test_present_csv_list(self):
        self.assertTrue(parse_list("ref,title\nuser/target,Target\n", "user/target"))

    def test_unknown_csv_fails_closed(self):
        with self.assertRaises(ValueError):
            parse_list("Some error message", "user/target")

    @patch("kaggle_status.subprocess.run")
    def test_missing_kernel_status_not_called(self, run):
        run.return_value = SimpleNamespace(returncode=0, stdout="ref,title\n", stderr="")
        self.assertEqual(check("user/target", "not-active"), "missing")
        self.assertEqual(run.call_count, 1)
        self.assertIn("--mine", run.call_args.args[0])

    @patch("kaggle_status.subprocess.run")
    def test_listing_errors_fail_closed(self, run):
        run.return_value = SimpleNamespace(returncode=1, stdout="", stderr="Unauthorized")
        with self.assertRaises(RuntimeError):
            check("user/target", "not-active")

    @patch("kaggle_status.subprocess.run")
    def test_running_kernel_blocks_submit(self, run):
        run.side_effect = [
            SimpleNamespace(returncode=0, stdout="ref,title\nuser/target,Target\n", stderr=""),
            SimpleNamespace(returncode=0, stdout='user/target has status "KernelWorkerStatus.RUNNING"', stderr=""),
        ]
        with self.assertRaises(RuntimeError):
            check("user/target", "not-active")
        self.assertEqual(run.call_count, 2)


if __name__ == "__main__":
    unittest.main()
