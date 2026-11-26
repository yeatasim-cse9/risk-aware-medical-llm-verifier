"""Regression for removed tokenizer methods and unchanged inference budgets."""
import ast
from pathlib import Path
import sys
import unittest
from types import SimpleNamespace
S=Path(__file__).resolve().parents[1]/"scripts"
sys.path.insert(0,str(S))
import e1_master01e_v4_neural as n
import e1_master01e_v4_public_gpu_diagnostic as gpu
from unittest.mock import patch
import os

class ModernTokenizer:
    cls_token_id=1;sep_token_id=2;pad_token_id=0
    padding_side="right"
    model_input_names=["input_ids","attention_mask","token_type_ids"]
    def num_special_tokens_to_add(self,pair):return 3 if pair else 2
    def __call__(self,a,text_pair=None,add_special_tokens=True,**kwargs):
        raw=lambda x:[3+len(t)%8 for t in x.split()]
        x=raw(a)
        if not add_special_tokens:return {"input_ids":x}
        y=raw(text_pair)
        ids=[1,*x,2,*y,2]
        return {"input_ids":ids,"attention_mask":[1]*len(ids),
                "token_type_ids":[0]*(len(x)+2)+[1]*(len(y)+1)}

class Model:
    config=SimpleNamespace(id2label={0:"contradiction",1:"entailment",2:"neutral"},
                           max_position_embeddings=512)
    def to(self,device):return self
    def eval(self):return self

class V4(unittest.TestCase):
    def test_modern_api_no_legacy_methods(self):
        t=ModernTokenizer()
        for name in ("encode","build_inputs_with_special_tokens","create_token_type_ids_from_sequences"):
            self.assertFalse(hasattr(t,name))
        e=n.Engine(t,Model(),"cpu")
        z=e.encode([4,5],[6])
        self.assertEqual(z["input_ids"][:6],[1,4,5,2,6,2])
        self.assertEqual(z["token_type_ids"][:6],[0,0,0,0,1,1])
        self.assertEqual(z["attention_mask"][:7],[1,1,1,1,1,1,0])
        z=e.encode([],[6])
        self.assertEqual(z["input_ids"][:4],[1,2,6,2])
        self.assertEqual(z["token_type_ids"][:4],[0,0,1,1])
        self.assertEqual(len(z["input_ids"]),512)

    def test_mismatch_fails_closed(self):
        class Wrong(ModernTokenizer):
            def __call__(self,*args,**kw):
                x=super().__call__(*args,**kw)
                if "token_type_ids" in x:x["token_type_ids"]=[0]*len(x["input_ids"])
                return x
        with self.assertRaisesRegex(ValueError,"PUBLIC_PAIR_CONTRACT"):
            n.Engine(Wrong(),Model(),"cpu")

    def test_boundary(self):
        e=n.Engine(ModernTokenizer(),Model(),"cpu")
        parts,count=e.windows("alpha "*400)
        self.assertEqual((count,len(parts)),(400,3))
        self.assertEqual(sum(e.encode([4]*192,[5]*317)["attention_mask"]),512)
        with self.assertRaisesRegex(ValueError,"HYPOTHESIS_BUDGET_EXCEEDED"):e.encode([],[5]*318)
        with self.assertRaisesRegex(ValueError,"CONTEXT_BUDGET_EXCEEDED"):e.windows("alpha "*2500)

    def test_original_inference_unchanged(self):
        def node(path,name,cls=None):
            a=ast.parse(path.read_text())
            if cls:a=next(x for x in a.body if isinstance(x,ast.ClassDef) and x.name==cls)
            return ast.dump(next(x for x in a.body if isinstance(x,ast.FunctionDef) and x.name==name))
        a=S/"e1_master01e_v2_neural.py";b=S/"e1_master01e_v4_neural.py"
        for name,cls in (("infer","Engine"),("windows","Engine"),("aggregate",None),("check_assets",None)):
            self.assertEqual(node(a,name,cls),node(b,name,cls))

class GPUQualification(unittest.TestCase):
    def test_fresh_authorization_required(self):
        with patch.dict(os.environ,{},clear=True):
            with self.assertRaisesRegex(ValueError,"GPU_AUTHORIZATION_REQUIRED"):
                gpu.main()

    def test_qualification_checks_all_conditions_and_repeat(self):
        class Fake:
            sep=2;special=3
            def windows(self,text):return [[4]*192,[5]*192],300
            def encode(self,a,b):raise ValueError("HYPOTHESIS_BUDGET_EXCEEDED")
            def infer(self,parts,hypothesis,*,removed,batch_size):
                return [0.5]*6,{"forward_examples":16,"padded_positions":8192,
                    "real_windows":0 if removed else len(parts),"active_tokens":32 if removed else 400}
        with patch.object(gpu,"stage") as stage:
            gpu.qualify_engine(Fake())
        self.assertEqual([x.args[0] for x in stage.call_args_list],
            ["WINDOW_BOUNDARY","FULL_FORWARD","REMOVED_FORWARD","PERMUTED_FORWARD","REPEAT_FULL"])
        class Bad(Fake):
            calls=0
            def infer(self,*args,**kwargs):
                f,c=super().infer(*args,**kwargs)
                self.calls+=1
                if self.calls==4:f[0]=0.6
                return f,c
        with patch.object(gpu,"stage"):
            with self.assertRaisesRegex(ValueError,"DETERMINISTIC_REPEAT_MISMATCH"):
                gpu.qualify_engine(Bad())

if __name__=="__main__":unittest.main()
