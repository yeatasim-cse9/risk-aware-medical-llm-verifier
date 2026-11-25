"""No network, no Kaggle calls and no raw error logs during synthetic tests."""
from __future__ import annotations
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

from scripts.diagnose_medhallu_metadata_failure import categories, diagnose


class SafeDiagnosticTests(unittest.TestCase):
    def test_known_classification(self):
        self.assertEqual(categories("Traceback: ModuleNotFoundError: No module named 'manager'"), ["PYTHON_IMPORT_ERROR"])
        self.assertEqual(categories("HTTP Error 503: outage"), ["REMOTE_HTTP_503"])
        self.assertEqual(categories("unrelated diagnostic"), [])

    def test_redacts_raw_remote_diagnostics(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp) / "audit"
            raw = "ModuleNotFoundError: No module named manager PRIVATE-EXAMPLE-CONTENT"
            with patch("scripts.diagnose_medhallu_metadata_failure.subprocess.run") as run:
                def mock_run(*args, **kwargs):
                    folder.mkdir(parents=True, exist_ok=True)
                    (folder / "trace.log").write_text(raw, encoding="utf-8")
                    return SimpleNamespace(stdout=raw, stderr="", returncode=0)
                run.side_effect = mock_run
                output = StringIO()
                with redirect_stdout(output):
                    state = diagnose(folder)
                self.assertEqual(state["safe_error_categories"], ["PYTHON_IMPORT_ERROR"])
                self.assertNotIn("PRIVATE-EXAMPLE-CONTENT", output.getvalue())
                self.assertNotIn("No module named", output.getvalue())

    def test_no_existing_directory_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp) / "audit"
            folder.mkdir()
            (folder / "old.txt").write_text("old", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                diagnose(folder)


if __name__ == "__main__":
    unittest.main()
