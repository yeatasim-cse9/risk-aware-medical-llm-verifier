import hashlib,json,sys,tempfile,unittest,importlib.util
from pathlib import Path
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"scripts"))
import e1_master01e_v2_neural as n
import e1_master01e_v2_pilot as p

def pair(i,s):
    return {"parent":str(i),"source_group":str(i),"partition":s,
        "question":"invented question "+str(i),"context":"alpha beta "+str(i),
        "answers":{"ground_truth":"alpha beta longer answer","hallucinated":"wrong"}}
class FakeEngine:
    def windows(self,text):return [[1,2]],2
    def tokens(self,text):return [1]
    def encode(self,a,b):return {}
    def infer(self,parts,hypothesis,*,removed=False):
        z=.5 if removed else .8 if hypothesis=="wrong" else .2
        return [1-z,z,1-z,z,1-z,float(z<=.5)],dict(forward_examples=n.SLOTS,
            padded_positions=n.SLOTS*n.MAX_LEN,active_tokens=32,real_windows=0 if removed else 1)

class TestMaster01EV2(unittest.TestCase):
    def test_assets_fail_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);fs={}
            for name in ("config.json","model.safetensors","tokenizer_config.json","vocab.txt"):
                (root/name).write_bytes(b"synthetic fixture")
                fs[name]=hashlib.sha256(b"synthetic fixture").hexdigest()
            manifest={"model":n.CANDIDATES[0],"revision":"a"*40,"files":fs}
            h=hashlib.sha256(json.dumps(manifest,sort_keys=True,separators=(",",":")).encode()).hexdigest()
            n.check_assets(root,manifest,h)
            with self.assertRaisesRegex(ValueError,"PRIVATE_AUTHORIZATION"):
                n.load_offline(root,manifest,h,authorized=False)
            with self.assertRaisesRegex(ValueError,"MANIFEST_PIN"):
                n.check_assets(root,manifest,"0"*64)
            (root/"config.json").write_text("changed")
            with self.assertRaisesRegex(ValueError,"ASSET_HASH"):n.check_assets(root,manifest,h)
    def test_aggregation(self):
        rows=[{"entailment":.7,"contradiction":.2,"neutral":.1},
              {"entailment":.1,"contradiction":.6,"neutral":.3}]
        a=n.aggregate(rows)
        self.assertEqual(a[0:2],[.7,.6]);self.assertAlmostEqual(a[2],.4)
        self.assertEqual(a[5],.5)
        with self.assertRaisesRegex(ValueError,"NLI_PROBABILITY"):
            n.aggregate([{"entailment":.9,"contradiction":.9,"neutral":.2}])
    def test_pilot_and_scope(self):
        tr=[pair(i,"train") for i in range(4)]
        dv=[pair(i,"dev") for i in range(4,8)]
        with self.assertRaisesRegex(ValueError,"PRIVATE_AUTHORIZATION"):
            p.run(FakeEngine(),tr,dv,authorized=False)
        with self.assertRaisesRegex(ValueError,"RETAINED_PILOT_COUNTS"):
            p.run(FakeEngine(),tr,dv,authorized=True)
        with patch.object(p,"LIMITS",{"train":4,"dev":4}),patch.object(p,"validate_private_inputs",return_value=None):
            out=p.run(FakeEngine(),tr,dv,authorized=True)
        self.assertFalse(out["test_used"]);self.assertFalse(out["public_export_approved"])
        self.assertEqual(out["metrics"]["question_only"]["paired_ranking"],.5)
        self.assertEqual(out["metrics"]["neural_full"]["paired_ranking"],1.)
        self.assertEqual(out["metrics"]["neural_removed_frozen"]["paired_ranking"],.5)
        self.assertEqual(len(out["metrics"]),13)
        for costs in out["inference_budget"].values():
            self.assertEqual(len({x["padded_positions"] for x in costs.values()}),1)
    def test_mapping_gate(self):
        with self.assertRaisesRegex(ValueError,"PRIVATE_AUTHORIZATION"):
            p.prepare_verified_inputs({}, {}, {}, None, {},authorized=False)
    @unittest.skipUnless(importlib.util.find_spec("torch") and importlib.util.find_spec("transformers"),
                         "dedicated offline transformer CI installs CPU dependencies")
    def test_real_transformer_forward(self):
        import torch
        from transformers import BertTokenizer,BertConfig,BertForSequenceClassification
        torch.manual_seed(20261126);torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as tmp:
            vocab=Path(tmp)/"vocab.txt"
            vocab.write_text("[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\nalpha\nbeta\nwrong\n")
            tok=BertTokenizer(vocab_file=str(vocab))
            cfg=BertConfig(vocab_size=len(tok),hidden_size=8,num_hidden_layers=1,
                num_attention_heads=2,intermediate_size=16,max_position_embeddings=512,
                num_labels=3,id2label={0:"contradiction",1:"entailment",2:"neutral"})
            model=BertForSequenceClassification(cfg)
            engine=n.Engine(tok,model,"cpu")
            windows,total=engine.windows("alpha "*400)
            self.assertEqual(total,400);self.assertEqual(len(windows),3)
            self.assertEqual(len(engine.encode(windows[0],[5])["input_ids"]),512)
            full,c1=engine.infer(windows,"alpha beta")
            removed,c2=engine.infer(windows,"alpha beta",removed=True)
            self.assertEqual(len(full),6);self.assertEqual(len(removed),6)
            self.assertEqual(c1["padded_positions"],c2["padded_positions"])
            self.assertLess(c2["active_tokens"],c1["active_tokens"])
            with self.assertRaisesRegex(ValueError,"CONTEXT_BUDGET_EXCEEDED"):
                engine.windows("alpha "*2500)
            with self.assertRaisesRegex(ValueError,"HYPOTHESIS_BUDGET_EXCEEDED"):
                engine.infer(windows,"alpha "*400)
if __name__=="__main__":unittest.main()
