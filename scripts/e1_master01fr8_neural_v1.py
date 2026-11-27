"""R8: frozen NLI features + strong lexical comparator. Private development only.
No automatic acquisition of model weights, no dispatch, calibration or test access.
"""
import base64
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import tempfile

import e1_master01fr7_lexical_v1 as r7
import e1_master01f_r3_audit_v1 as r3
import e1_master01f_r2_native_v1 as native
import e1_master01fr6r_checkpoint_v1 as cp
import e1_master01e_v4_neural as n
from e1_master01e_v4_public_cpu_diagnostic import PIN, DIGEST

require=r7.require
R7_ARCHIVE="5cd14b623bf4f0bd2593903c86d555889fdc8745"
R7_PATH=cp.PREFIX+"r6r-r7-lexical-001/complete.json"
# Fixed historical comparator, no selection-time model choice in R8.
CONTRAST=lambda prefix:[prefix+x for x in ("_delta","_lo","_hi")]
STAGES=[
    {"name":"lexical_reproduction","metrics":["ap","auroc","brier"]},
    {"name":"token_budget","metrics":["fit_groups","selection_groups","over_budget",
                                      "max_context_windows","max_statement_tokens"]},
    {"name":"neural_hybrid_full","metrics":[prefix+"_"+m for prefix in ("neural","hybrid")
            for m in ("ap","auroc","brier")]+["raw_contradiction_ap","raw_nonentailment_ap"]
            +CONTRAST("neural_vs_lexical")+CONTRAST("hybrid_vs_lexical")},
    {"name":"evidence_ablation","metrics":["neural_removed_ap","neural_permuted_ap",
            "hybrid_removed_ap","hybrid_permuted_ap"]+
            sum((CONTRAST(k) for k in ("neural_vs_removed","neural_vs_permuted",
                                       "hybrid_vs_removed","hybrid_vs_permuted")),[])}
]
BATCH=4

def runtime():
    return {**r7.runtime(),**{k:importlib.metadata.version(k) for k in
        ("torch","transformers","huggingface_hub","sentencepiece","safetensors","tokenizers")}}

def read_r7(store,expected):
    store.verify()
    require(cp.match(cp.HEX,expected),"R7_REPORT_PIN")
    obj=cp.request_json(cp.API+"/contents/"+R7_PATH+"?ref="+R7_ARCHIVE,store.headers)
    require(type(obj) is dict and obj.get("type")=="file" and obj.get("path")==R7_PATH
            and obj.get("encoding")=="base64" and type(obj.get("content")) is str,"R7_ARCHIVE_SCHEMA")
    raw=base64.b64decode("".join(obj["content"].splitlines()),validate=True)
    require(len(raw)<=cp.MAX and len(raw)==obj.get("size") and cp.blob(raw)==obj.get("sha")
            and cp.digest(raw)==expected,"R7_REPORT_INTEGRITY")
    report=json.loads(raw)
    require(report.get("public_export_approved") is False and report.get("scientific_approval") is False
            and set(report.get("aggregates",{}))=={s["name"] for s in r7.STAGES},"R7_REPORT_SCOPE")
    return report

def make_plan(commit,run_id,r6_hash,r7_hash,approval_pin,blobs,versions,device):
    require(device in ("cpu","cuda:0") and all(cp.match(cp.HEX,x) for x in
            (r6_hash,r7_hash,approval_pin)),"R8_INPUT_PINS")
    require(type(blobs) is dict and bool(blobs) and all(cp.match(cp.COMMIT,x) for x in blobs.values()),
            "R8_CODE_PINS")
    plan={"schema":1,"run_id":run_id,"scope":"private_development_aggregate","source_commit":commit,
          "input_sha256":cp.digest(cp.canonical({"revision":r3.REV,"metadata":r3.PIN,
            "r6":r6_hash,"r7":r7_hash,"approval":approval_pin,"model":DIGEST})),
          "recipe_sha256":cp.digest(cp.canonical({"blobs":blobs,"versions":versions,
            "device":device,"batch":BATCH,"bootstrap":r7.NBOOT,"seed":r7.SEED,
            "slots":n.SLOTS,"window":n.WINDOW,"overlap":n.OVERLAP,"max_len":n.MAX_LEN})),
          "stages":json.loads(cp.canonical(STAGES))}
    return cp.validate_plan(plan,cp.digest(cp.canonical(plan)))

def verify_lexical(metrics,report,tol=1e-8):
    old=report["aggregates"]["evidence_ablation"]
    require(all(abs(metrics[k]-old[k])<=tol for k in ("ap","auroc","brier")),
            "R7_LEXICAL_REPRODUCTION_MISMATCH")

class Study:
    """RAM-only feature arrays. Frozen full-trained heads for perturbations."""
    def __init__(self,fit,selection,engine_factory,report):
        import numpy as np
        from scipy.sparse import hstack,csr_matrix
        from sklearn.preprocessing import StandardScaler
        self.sf,self.cf,self.yf=r7.split(fit)
        self.ss,self.cs,self.ys=r7.split(selection)
        self.engine_factory=engine_factory;self.engine=None
        self.report=report;self.parts=None;self.full=None;self.lex=None
        self.np=np;self.hstack=hstack;self.csr=csr_matrix;self.scaler=StandardScaler

    def lexical(self):
        if self.lex is not None:return self.lex
        a,b,_=r7.block(self.sf,self.ss,["word","char_wb"])
        d,e,cv=r7.block(self.cf,self.cs,["word","char_wb"])
        scaler=self.scaler().fit(r7.numeric(self.sf,self.cf))
        def pack(s,c,sv,ctx):
            return self.hstack([sv,ctx,self.csr(scaler.transform(r7.numeric(s,c)))],format="csr")
        x=pack(self.sf,self.cf,a,d); z=pack(self.ss,self.cs,b,e)
        model=r7.classifier(x,self.yf); scores=r7.prob(model,z)
        verify_lexical(r7.measures(self.ys,scores),self.report)
        self.lex=(x,z,scores,lambda contexts:pack(self.ss,contexts,b,r7.transformed(cv,contexts)))
        return self.lex

    def budget(self):
        if self.parts is not None:return self.budget_result
        if self.engine is None:self.engine=self.engine_factory()
        eng=self.engine;parts=[];bad=0;maxw=0;maxs=0
        for s,c in zip(self.sf+self.ss,self.cf+self.cs):
            ids=eng.tokens(s);maxs=max(maxs,len(ids))
            if len(ids)>n.MAX_LEN-n.WINDOW-eng.special:
                bad+=1;parts.append(None);continue
            try: w,_=eng.windows(c)
            except ValueError as exc:
                if exc.args!=("CONTEXT_BUDGET_EXCEEDED",):raise
                bad+=1;parts.append(None);continue
            maxw=max(maxw,len(w));parts.append(w)
        self.parts=parts
        self.budget_result={"fit_groups":len(self.sf),"selection_groups":len(self.ss),
                            "over_budget":bad,"max_context_windows":maxw,"max_statement_tokens":maxs}
        return self.budget_result

    def features(self,statements,parts,removed=False):
        out=[]
        for s,w in zip(statements,parts):
            f,c=self.engine.infer(w,s,removed=removed,batch_size=BATCH)
            require(len(f)==6 and all(0<=v<=1 for v in f)
                    and c["forward_examples"]==n.SLOTS and c["padded_positions"]==n.SLOTS*n.MAX_LEN
                    and c["real_windows"]==(0 if removed else len(w)),"NLI_FORWARD_CONTRACT")
            out.append(f)
        return self.np.asarray(out,dtype=float)

    def full_models(self):
        if self.full is not None:return self.full
        x,z,lex,_=self.lexical()
        require(self.budget()["over_budget"]==0,"R8_COHORT_TOKEN_BUDGET_BLOCKED")
        k=len(self.sf)
        f=self.features(self.sf,self.parts[:k]);s=self.features(self.ss,self.parts[k:])
        scaler=self.scaler().fit(f)
        a,b=scaler.transform(f),scaler.transform(s)
        neural=r7.classifier(a,self.yf)
        hybrid=r7.classifier(self.hstack([x,self.csr(a)],format="csr"),self.yf)
        ns=r7.prob(neural,b);hs=r7.prob(hybrid,self.hstack([z,self.csr(b)],format="csr"))
        self.full=(scaler,neural,hybrid,ns,hs,s)
        return self.full

    def compute(self,stage,prior,nboot=r7.NBOOT):
        from threadpoolctl import threadpool_limits
        with threadpool_limits(limits=1):
            if stage=="lexical_reproduction":
                return r7.measures(self.ys,self.lexical()[2])
            if stage=="token_budget":return self.budget()
            scaler,neural,hybrid,ns,hs,s=self.full_models()
            lex=self.lexical()[2]
            if stage=="neural_hybrid_full":
                out={prefix+"_"+k:v for prefix,scores in (("neural",ns),("hybrid",hs))
                     for k,v in r7.measures(self.ys,scores).items()}
                out.update(raw_contradiction_ap=r7.measures(self.ys,s[:,1])["ap"],
                           raw_nonentailment_ap=r7.measures(self.ys,1-s[:,0])["ap"])
                out.update(r7.paired(self.ys,ns,{"neural_vs_lexical":lex},nboot))
                out.update(r7.paired(self.ys,hs,{"hybrid_vs_lexical":lex},nboot))
                return out
            require(stage=="evidence_ablation","R8_STAGE")
            k=len(self.sf);parts=self.parts[k:]
            rem=self.features(self.ss,parts,removed=True)
            per=self.features(self.ss,parts[1:]+parts[:1])
            out={};ncomp={};hcomp={}
            for name,features,contexts in (("removed",rem,[""]*len(self.cs)),
                                           ("permuted",per,r7.permute(self.cs))):
                b=scaler.transform(features)
                a=self.lexical()[3](contexts)
                npred=r7.prob(neural,b)
                hpred=r7.prob(hybrid,self.hstack([a,self.csr(b)],format="csr"))
                out["neural_"+name+"_ap"]=r7.measures(self.ys,npred)["ap"]
                out["hybrid_"+name+"_ap"]=r7.measures(self.ys,hpred)["ap"]
                ncomp["neural_vs_"+name]=npred;hcomp["hybrid_vs_"+name]=hpred
            out.update(r7.paired(self.ys,ns,ncomp,nboot))
            out.update(r7.paired(self.ys,hs,hcomp,nboot))
            return out

def execute(plan,pin,store,metadata,receipt,approval_pin,r6_hash,r7_hash,*,commit,blobs,
            device,engine_factory,authorized=False,model_rights=False,label_reviewed=False,
            aggregate_reviewed=False,gpu_authorized=False,max_new=None):
    require(all(x is True for x in (authorized,model_rights,label_reviewed,aggregate_reviewed)),
            "R8_APPROVAL_REQUIRED")
    require(device in ("cpu","cuda:0") and (device=="cpu" or gpu_authorized is True),
            "R8_GPU_AUTHORIZATION_REQUIRED")
    require(device!="cpu" or os.environ.get("CUDA_VISIBLE_DEVICES")=="","R8_CPU_DEVICE_REQUIRED")
    expected=make_plan(commit,plan["run_id"],r6_hash,r7_hash,approval_pin,blobs,runtime(),device)
    require(plan==expected and cp.digest(cp.canonical(expected))==pin,"R8_PLAN_BINDING")
    native.approved_plan(metadata,r3.REV,r3.PIN,receipt,approval_pin)
    old=r7.read_r6(store,r6_hash);report=read_r7(store,r7_hash)
    study=None
    def compute(stage,prior):
        nonlocal study
        if stage in ("neural_hybrid_full","evidence_ablation"):
            require(prior.get("token_budget",{}).get("over_budget")==0,"R8_COHORT_TOKEN_BUDGET_BLOCKED")
        if study is None:
            require(os.environ.get("HF_HUB_OFFLINE")=="1","OFFLINE_DATA_READER_REQUIRED")
            temp=Path("/kaggle/temp");temp.mkdir(exist_ok=True)
            root=Path(tempfile.mkdtemp(prefix="master01fr8-",dir=temp));root.chmod(0o700)
            try:
                # Both native transfer and row reading require CPU visibility
                # flag. Restore the original environment before engine creation.
                prev=os.environ.get("CUDA_VISIBLE_DEVICES")
                os.environ["CUDA_VISIBLE_DEVICES"]=""
                try:
                    native.acquire(metadata,r3.REV,r3.PIN,receipt,approval_pin,root/"train",allow=True)
                    factory=lambda:r3.verified_rows(root/"train",metadata,receipt,approval_pin,
                                                   allow=True,label_semantics_confirmed=True)
                    fit,sel,summary=r7.reconstruct(factory)
                finally:
                    if prev is None:os.environ.pop("CUDA_VISIBLE_DEVICES",None)
                    else:os.environ["CUDA_VISIBLE_DEVICES"]=prev
                r7.check_r6(summary,old)
                study=Study(fit,sel,engine_factory,report)
            finally:
                shutil.rmtree(root)
        return study.compute(stage,prior)
    return cp.run(plan,pin,compute,store,authorized=True,max_new=max_new)
