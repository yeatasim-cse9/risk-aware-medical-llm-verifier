import copy
import hashlib
import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import e1_master01f_r_protocol_v1 as p


class Protocol(unittest.TestCase):
    def setUp(self):
        self.rev = "a"*40
        self.m = {"repository": p.CANDIDATE, "revision": self.rev,
                  "shards": [{"path": "data/"+s+"-00000-of-00001.parquet",
                              "sha256": str(i)*64, "size": 10}
                             for i,s in enumerate(p.SPLITS,1)]}

    def plan(self):
        return p.train_plan(self.m, self.rev, p.digest(self.m))

    def test_train_only(self):
        a = self.plan()
        self.assertEqual([x["path"] for x in a["train_files"]],
                         ["data/train-00000-of-00001.parquet"])
        self.assertFalse(a["execution_authorized"])
        self.assertFalse(a["source_disjointness_verified"])
        self.assertFalse(a["rights_approved"])

    def test_bad_pin(self):
        with self.assertRaisesRegex(ValueError, "METADATA_PIN_MISMATCH"):
            p.train_plan(self.m, self.rev, "0"*64)

    def test_mixed_and_traversal(self):
        for name in ("data/mixed.parquet", "../data/train-00000-of-00001.parquet"):
            self.m["shards"][0]["path"] = name
            with self.assertRaisesRegex(ValueError, "SHARD_PATH"):
                self.plan()

    def test_alias(self):
        self.m["shards"][2]["sha256"] = self.m["shards"][0]["sha256"]
        with self.assertRaisesRegex(ValueError, "SHARD_IDENTITY"):
            self.plan()

    def test_missing_shard(self):
        self.m["shards"][0]["path"] = "data/train-00000-of-00002.parquet"
        with self.assertRaisesRegex(ValueError, "INCOMPLETE_SHARDS"):
            self.plan()

    def test_revision_drift(self):
        self.m["revision"] = "b"*40
        with self.assertRaisesRegex(ValueError, "METADATA_SCOPE"):
            self.plan()

    def test_group_split_order_invariant(self):
        groups = [hashlib.sha256(str(i).encode()).hexdigest() for i in range(10)]
        x = p.group_split(groups, 4, 3)
        self.assertEqual(x, p.group_split(list(reversed(groups)), 4, 3))
        self.assertFalse(set(x["fit"]) & set(x["selection"]))
        with self.assertRaisesRegex(ValueError, "DUPLICATE_GROUP"):
            p.group_split(groups + groups[:1], 4, 3)

    def test_abstention_not_repair(self):
        a = p.transitions(["wrong","wrong","correct","wrong"],
                          ["correct","abstain","wrong","invalid"])
        self.assertEqual(a["wrong_to_correct"], 1)
        self.assertEqual(a["correct_to_wrong"], 1)
        self.assertEqual(a["correction_rate"], 1/3)
        self.assertEqual(a["net_valid_correct_change"], 0)
        self.assertEqual(a["revised_abstain"], 1)
        self.assertEqual(a["revised_invalid"], 1)

    def test_zero_initial_error(self):
        self.assertIsNone(p.transitions(["correct"],["abstain"])["correction_rate"])

    def test_outcome_scope(self):
        with self.assertRaisesRegex(ValueError, "PAIRED_OUTCOMES"):
            p.transitions(["wrong"], ["clinical_safe"])
        with self.assertRaisesRegex(ValueError, "PAIRED_OUTCOMES"):
            p.transitions(["wrong"], [])


if __name__ == "__main__":
    unittest.main()
