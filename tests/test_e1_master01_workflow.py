"""Synthetic-only controller tests: no Kaggle/Telegram/data/model downloads."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
import e1_master01_workflow as c

class TestMaster01Workflow(unittest.TestCase):
    def test_package_private_gpu_pinned_and_no_secrets(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)/"package"
            c.build(out)
            m=json.loads((out/"kernel-metadata.json").read_text())
            self.assertEqual(m["id"],c.KERNEL)
            self.assertIs(m["is_private"],True)
            self.assertIs(m["enable_gpu"],True)
            self.assertIs(m["enable_internet"],True)
            self.assertIs(m["enable_tpu"],False)
            for key in ("dataset_sources","competition_sources","kernel_sources","model_sources"):
                self.assertEqual(m[key],[])
            runner=(out/"runner.py").read_text()
            compile(runner,"runner.py","exec")
            self.assertIn(c.SHA,runner)
            self.assertIn(c.PIN,runner)
            self.assertEqual({p.name for p in out.iterdir()},{"runner.py","kernel-metadata.json"})
            with self.assertRaises(FileExistsError):
                c.build(out)
    def test_bad_integrity_and_unknown_schema_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/c.REPORT
            p.write_text(json.dumps({"integrity_sha256":"wrong"}))
            with self.assertRaisesRegex(ValueError,"REPORT_DIGEST_MISMATCH"):
                c.validate_report(p)
            data={"unexpected":"synthetic"}
            data["integrity_sha256"]=hashlib.sha256(c.master.canonical(data)).hexdigest()
            p.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError,"REPORT_SCHEMA"):
                c.validate_report(p)
    @patch.object(c,"check",return_value="running")
    @patch.object(c.subprocess,"run")
    def test_duplicate_rejected(self,run,check):
        run.return_value.returncode=0
        with self.assertRaisesRegex(ValueError,"DUPLICATE_MASTER01_KERNEL_BLOCKED"):
            c.preflight()
    @patch.object(c,"check",return_value="missing")
    @patch.object(c.subprocess,"run")
    def test_preflight_only_read_calls(self,run,check):
        run.return_value.returncode=0
        c.preflight()
        self.assertEqual(run.call_args.args[0],["kaggle","quota"])
    @patch.object(c,"check",return_value="error")
    @patch.object(c,"telegram")
    def test_failed_job_not_retried(self,notify,check):
        with self.assertRaisesRegex(ValueError,"REMOTE_TERMINAL_FAILURE"):
            c.monitor(Path("unused"))
        notify.assert_called_once()
    @patch.object(c,"check",return_value="running")
    def test_verify_requires_completion(self,check):
        with self.assertRaisesRegex(ValueError,"REMOTE_NOT_COMPLETE"):
            c.verify(Path("unused"))
