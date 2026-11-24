import sys
import unittest
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from kaggle_status import check, parse_kernel_list, parse_status


class KaggleStatusTests(unittest.TestCase):
    def test_status_parser(self):
        self.assertEqual(
            parse_status('user/kernel has status "KernelWorkerStatus.COMPLETE"'),
            "complete",
        )
        with self.assertRaises(ValueError):
            parse_status("Unexpected response")

    def test_empty_csv_header_is_missing(self):
        self.assertFalse(parse_kernel_list("ref,title\n", "user/target"))
        self.assertFalse(parse_kernel_list("ref,title", "user/target"))
        self.assertFalse(parse_kernel_list("ref,title,author,lastRunTime,totalVotes\n", "user/target"))

    def test_csv_list_exact_reference(self):
        self.assertTrue(parse_kernel_list("ref,title\nuser/target,Pilot\n", "user/target"))
        self.assertFalse(parse_kernel_list("ref,title\nuser/target-else,Other\n", "user/target"))

    def test_invalid_text_not_treated_as_absent(self):
        for value in ("Access denied", "ref", "Not found: permission denied", "No kernels found today", "ref,wrong"):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    parse_kernel_list(value, "user/target")

    def test_empty_json_list_is_missing(self):
        self.assertFalse(parse_kernel_list("[]", "user/target"))
        self.assertFalse(parse_kernel_list("", "user/target"))
        self.assertFalse(parse_kernel_list("No kernels found.", "user/target"))
        self.assertFalse(parse_kernel_list("Not found\n", "user/target"))
        self.assertFalse(parse_kernel_list("Not found", "user/target"))
        self.assertFalse(parse_kernel_list("No kernels", "user/target"))
        self.assertFalse(parse_kernel_list("no kernels\n", "user/target"))

    def test_exact_ref_required(self):
        self.assertTrue(
            parse_kernel_list('[{"ref":"user/target","title":"Pilot"}]', "user/target")
        )
        self.assertFalse(
            parse_kernel_list('[{"ref":"user/target2","title":"Other"}]', "user/target")
        )

    def test_unexpected_json_rejected(self):
        for text in ('{"status":"OK"}', "An error", "No kernels found today", '[{"title":"No ref"}]'):
            with self.subTest(text=text):
                with self.assertRaises(ValueError):
                    parse_kernel_list(text, "user/target")

    @patch("kaggle_status.subprocess.run")
    def test_missing_kernel_skips_status_call(self, run):
        run.return_value = SimpleNamespace(returncode=0, stdout="[]", stderr="")
        self.assertEqual(check("user/target", "not-active"), "missing")
        self.assertEqual(run.call_count, 1)
        self.assertIn("--format", run.call_args.args[0])

    @patch("kaggle_status.subprocess.run")
    def test_listing_error_fails_closed(self, run):
        run.return_value = SimpleNamespace(returncode=1, stdout="", stderr="Unauthorized")
        with self.assertRaises(RuntimeError):
            check("user/target", "not-active")

    @patch("kaggle_status.subprocess.run")
    def test_running_kernel_blocks_submission(self, run):
        run.side_effect = [
            SimpleNamespace(
                returncode=0, stdout='[{"ref":"user/target"}]', stderr=""
            ),
            SimpleNamespace(
                returncode=0,
                stdout='user/target has status "KernelWorkerStatus.RUNNING"',
                stderr=""
            ),
        ]
        with self.assertRaises(RuntimeError):
            check("user/target", "not-active")
        self.assertEqual(run.call_count, 2)


if __name__ == "__main__":
    unittest.main()
