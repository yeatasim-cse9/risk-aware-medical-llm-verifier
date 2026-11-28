import sys
import unittest
from unittest.mock import patch
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
from kaggle_status import parse_status
import r13h_v3_recovery_controller as v3

class CancelRegression(unittest.TestCase):
    def test_observed_enum(self):
        for s in ("CANCEL_ACKNOWLEDGED","KernelWorkerStatus.CANCEL_ACKNOWLEDGED","cancel_acknowledged","cancelacknowledged"):
            self.assertEqual(v3.parse_recovery_status("Kernel x/y has status "+s),"cancelacknowledged")
    def test_historical_parser_unchanged(self):
        with self.assertRaises(ValueError):
            parse_status('Kernel x/y has status CANCEL_ACKNOWLEDGED')
    def test_nonterminal_not_accepted(self):
        for s in ("CANCEL_REQUESTED","CANCELING","CANCEL_ACKNOWLEDGED_EXTRA","UNKNOWN"):
            with self.assertRaises(ValueError):
                v3.parse_recovery_status("Kernel x/y has status "+s)
        self.assertEqual(v3.parse_recovery_status("Kernel x/y has status RUNNING"),"running")
    def test_preflight_never_push(self):
        cfg={"dataset_source":"synthetic/private"}
        with patch.object(v3,"history"),patch.object(v3.v2,"config",return_value=cfg),patch.object(v3.v2,"get_manifest",return_value={}),patch.object(v3.v2,"cli",return_value=b"--timeout --accelerator") as cli,patch.object(v3,"state",side_effect=["cancelacknowledged","missing","missing"]),patch.dict("os.environ",{"GITHUB_RUN_ID":"1","GITHUB_SHA":"a"}):
            c,k=v3.preflight()
            self.assertEqual(k,"synthetic/master-01f-r13h-private-gpu-v3")
            cli.assert_called_once_with("kernels","push","--help")
    def test_v2_present_blocks(self):
        with patch.object(v3,"history"),patch.object(v3.v2,"config",return_value={"dataset_source":"synthetic/private"}),patch.object(v3.v2,"get_manifest",return_value={}),patch.object(v3.v2,"cli",return_value=b"--timeout --accelerator"),patch.object(v3,"state",side_effect=["cancelacknowledged","queued"]),patch.dict("os.environ",{"GITHUB_RUN_ID":"1","GITHUB_SHA":"a"}):
            with self.assertRaises(ValueError): v3.preflight()
            self.assertEqual(v3.STAGE,"V2_ABSENCE")
