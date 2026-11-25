"""Synthetic-only tests. No medical text, model downloads, Kaggle or Telegram."""
import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
import e1_master01b_kaggle as m
import e1_master01b_workflow as c

def zero():
    return {"pairs":0,"left_rows":0,"right_rows":0}

def fixture():
    # Synthetic structural fixture using protocol dimensions, not empirical observations.
    pairs={a+"__"+b:(a,b) for a,b in m.PAIRS}
    coverage={s:{"parents":n,"windows":n,"total_tokens":n,"covered_tokens":n,
             "multiwindow_parents":0,"max_windows_per_parent":1,"payload_tokens":254}
             for s,n in m.SIZES.items()}
    views={v:{"coverage":copy.deepcopy(coverage),
              "comparisons":{k:{"parent_pairs":m.SIZES[a]*m.SIZES[b],
                "window_pairs":m.SIZES[a]*m.SIZES[b],"maximum_cosine":0.5,
                "thresholds":{str(t):zero() for t in m.base.THRESHOLDS}}
                for k,(a,b) in pairs.items()}} for v in ("question","context")}
    enc={k:{"model":model,"revision":rev,"weights_sha256":weight,
       "file_sha256":{"model.safetensors":weight,"config.json":"1"*64,"tokenizer.json":"2"*64},
       "views":copy.deepcopy(views)} for k,(model,rev,weight) in m.MODELS.items()}
    diagnostics={k:{s:{"parents":n,"benchmark_role_0":n,"benchmark_role_1":n,
       "category_frequency_sorted":c.EXPECTED_CATEGORIES[k][s],"difficulty_frequency_sorted":[n],
       "min_observed_category_count":min(c.EXPECTED_CATEGORIES[k][s])}
       for s,n in v.items()} for k,v in c.EXPECTED_CONFIGS.items()}
    lexical=lambda pop:{"population_pairs":pop,"sample_pairs":min(pop,m.REVIEW_PER_STRATUM),
                         "question_jaccard_ge_0_5":0,"context_jaccard_ge_0_5":0}
    return {"schema_version":1,"status":"SCREEN_COMPLETE_REVIEW_REQUIRED","source_commit":m.base.BASE,
            "runner_sha256":c.RUNNER_SHA,"mapping_sha256":m.base.EXPECTED_MAPPING,
            "pinned_module_sha256":m.base.PINS,"seed":m.base.SEED,"parents":m.SIZES,
            "window_length":m.MAX_LENGTH,"window_overlap":m.OVERLAP,
            "device":"Tesla T4","versions":{k:"1.2.3" for k in c.VERSIONS},
            "encoders":enc,"exact_screen":{v:{k:{"shared_groups":0,"left_rows":0,"right_rows":0}
                  for k in pairs} for v in ("question","context")},"label_diagnostics":diagnostics,
            "review_candidates":{k:{"union_pairs":0,"left_rows":0,"right_rows":0,
             "lexical_sample":{"flagged":lexical(0),"unflagged":lexical(m.SIZES[a]*m.SIZES[b])}}
             for k,(a,b) in pairs.items()},"affected_parents":{s:0 for s in m.SIZES},
            **{k:False for k in c.FLAGS}}

def checked(data):
    with tempfile.TemporaryDirectory() as tmp:
        p=Path(tmp)/m.REPORT
        data=copy.deepcopy(data)
        data["integrity_sha256"]=hashlib.sha256(m.base.canonical(data)).hexdigest()
        p.write_text(json.dumps(data),encoding="utf-8")
        return c.validate_report(p)

class TestMaster01B(unittest.TestCase):
    def test_windows_no_truncation_and_boundaries(self):
        for n in (1,254,255,444,445,1000):
            spans=m.window_ranges(n,254,64)
            self.assertEqual(set(range(n)),{i for a,b in spans for i in range(a,b)})
            self.assertTrue(all(b-a<=254 for a,b in spans))
        self.assertEqual(m.window_ranges(255,254,64),[(0,254),(190,255)])
        with self.assertRaisesRegex(ValueError,"WINDOW_LIMIT"):
            m.window_ranges(100000,254,64)
        with self.assertRaises(ValueError):
            m.window_ranges(0,254,64)

    def test_window_pairwise_exhaustive_against_brute_force(self):
        try:
            import numpy as np
            import torch
        except ImportError:
            self.skipTest("CPU torch/numpy not installed in lightweight CI; dedicated workflow requires them")
        left=[torch.tensor([[1.,0.],[0.,1.]]),torch.tensor([[-1.,0.]]),torch.tensor([[0.,-1.]])]
        right=[torch.tensor([[0.,1.]]),torch.tensor([[1.,0.],[-1.,0.]])]
        union=np.zeros((3,2),dtype=bool)
        with patch.object(m,"WINDOW_BLOCK",2):
            out=m.compare_windows(left,right,union)
        scores=np.array([[float((a@b.T).max()) for b in right] for a in left])
        self.assertEqual(out["parent_pairs"],6)
        self.assertEqual(out["window_pairs"],12)
        self.assertTrue(np.array_equal(union,scores>=0.8))
        for t in m.base.THRESHOLDS:
            mask=scores>=t
            self.assertEqual(out["thresholds"][str(t)],{"pairs":int(mask.sum()),
                             "left_rows":int(mask.any(axis=1).sum()),"right_rows":int(mask.any(axis=0).sum())})
        # Union remains set even when a second encoder flags disjoint pairs.
        before=union.copy()
        m.compare_windows([torch.tensor([[0.,-1.]])]*3,right,union)
        self.assertTrue(np.array_equal(union,before))

    def test_lexical_sample_no_row_export(self):
        try:
            import numpy as np
        except ImportError:
            self.skipTest("numpy unavailable in lightweight CI")
        a=[{"question":"invented alpha","context":"invented beta"}]
        b=[{"question":"invented alpha","context":"other words"},
           {"question":"other words","context":"invented beta"}]
        result=m.lexical_sample(a,b,np.array([[True,False]]),str.lower,1)
        self.assertEqual(result["flagged"]["question_jaccard_ge_0_5"],1)
        self.assertEqual(result["unflagged"]["context_jaccard_ge_0_5"],1)
        self.assertNotIn("invented",str(result))

    def test_positive_structure_and_fail_closed(self):
        result=checked(fixture())
        self.assertFalse(result["final_split_approved"])
        self.assertEqual(result["runner_commit"],c.RUNNER_COMMIT)
        cases=[("coverage",lambda d:d["encoders"]["minilm"]["views"]["context"]["coverage"]["train"].update(covered_tokens=0)),
               ("weights",lambda d:d["encoders"]["pubmedbert"].update(weights_sha256="0"*64)),
               ("union",lambda d:d["review_candidates"]["train__test"].update(union_pairs=1)),
               ("approval",lambda d:d.update(final_split_approved=True)),
               ("schema",lambda d:d.update(raw_question="synthetic prohibited field"))]
        for name,alter in cases:
            with self.subTest(name=name):
                d=fixture()
                alter(d)
                with self.assertRaises(ValueError):
                    checked(d)

    def test_package_pins_privacy_and_no_credential(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/"package"
            c.build(p)
            meta=json.loads((p/"kernel-metadata.json").read_text())
            self.assertIs(meta["is_private"],True)
            self.assertIs(meta["enable_gpu"],True)
            self.assertIs(meta["enable_internet"],True)
            self.assertIs(meta["enable_tpu"],False)
            self.assertEqual(meta["id"],c.KERNEL)
            self.assertEqual({x.name for x in p.iterdir()},{"runner.py","kernel-metadata.json"})
            compile((p/"runner.py").read_bytes(),"runner.py","exec")
            self.assertIn(c.RUNNER_SHA,(p/"runner.py").read_text())
            with self.assertRaises(FileExistsError):
                c.build(p)

    @patch.object(c,"check",return_value="running")
    @patch.object(c.subprocess,"run")
    def test_duplicate_refused(self,run,check):
        run.return_value.returncode=0
        with self.assertRaisesRegex(ValueError,"DUPLICATE"):
            c.preflight()
        self.assertEqual(run.call_args.args[0],["kaggle","quota"])

    @patch.object(c,"check",return_value="error")
    @patch.object(c,"telegram")
    def test_monitor_does_not_retry(self,telegram,check):
        with self.assertRaisesRegex(ValueError,"REMOTE_TERMINAL_FAILURE"):
            c.monitor(Path("unused"))
        telegram.assert_called_once()
