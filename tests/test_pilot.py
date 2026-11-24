import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from research_automation import pilot


class PilotTests(unittest.TestCase):
    def test_resume_skips_finished_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            first = pilot.run(base / "first", base / "empty", max_new=40)
            self.assertEqual(len(first["completed_task_ids"]), 40)
            source = base / "next_input" / "previous"
            source.mkdir(parents=True)
            checkpoint = base / "first" / "checkpoints" / pilot.CHECKPOINT_NAME
            (source / pilot.CHECKPOINT_NAME).write_bytes(checkpoint.read_bytes())
            last = pilot.run(base / "second", base / "next_input", max_new=60)
            self.assertEqual(len(last["completed_task_ids"]), 100)
            self.assertEqual(last["status"], "completed")
            again = pilot.run(base / "second", base / "next_input", max_new=60)
            self.assertEqual(last["results"], again["results"])

    def test_corruption_detected(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            pilot.run(base / "out", base / "empty", max_new=2)
            path = base / "out" / "checkpoints" / pilot.CHECKPOINT_NAME
            data = json.loads(path.read_text())
            data["completed_task_ids"].append(0)
            path.write_text(json.dumps(data))
            with self.assertRaises(ValueError):
                pilot.run(base / "out", base / "empty")

    def test_ambiguous_source_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            for name in ("a", "b"):
                dest = base / "input" / name
                dest.mkdir(parents=True)
                (dest / pilot.CHECKPOINT_NAME).write_text("{}")
            with self.assertRaises(ValueError):
                pilot.run(base / "output", base / "input")


if __name__ == "__main__":
    unittest.main()
