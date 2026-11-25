"""MASTER 01B: exhaustive window-based two-encoder screening, not adjudication."""
from __future__ import annotations
import collections
import hashlib
import importlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import random
import re
import sys
import tempfile

import e1_master01_validation_kaggle as base

MODELS = {
    "minilm": ("sentence-transformers/all-MiniLM-L6-v2", "c9745ed1d9f207416be6d2e6f8de32d1f16199bf",
               "53aa51172d142c89d9012cce15ae4d6cc0ca6895895114379cacb4fab128d9db"),
    "pubmedbert": ("NeuML/pubmedbert-base-embeddings", "27cf1ec4e51deabc22b419742453d5bf4923e38c",
                   "929ddc16369bb4ff6f8e92a6ee9f5e0748f2bef33989b5617533dd50beebe4ef"),
}
SIZES = {"train":7579,"dev":947,"test":948,"official_holdout":500}
PAIRS = [(a,b) for i,a in enumerate(base.SPLITS) for b in base.SPLITS[i+1:]]
MAX_LENGTH = 256
OVERLAP = 64
MAX_WINDOWS_PER_PARENT = 64
WINDOW_BLOCK = 512
REVIEW_PER_STRATUM = 100
REPORT = "master01b_full_context_v2.json"
STAGE = "BOOT"

def require(ok, code):
    if not ok:
        raise ValueError(code)

def stage(name):
    global STAGE
    STAGE = name
    print("MASTER01B_STAGE="+name,flush=True)

def window_ranges(n, payload, overlap):
    require(type(n) is int and n>0 and 0<=overlap<payload, "INVALID_WINDOW_INPUT")
    starts=[0]
    while starts[-1]+payload<n:
        starts.append(starts[-1]+payload-overlap)
        require(len(starts)<=MAX_WINDOWS_PER_PARENT,"WINDOW_LIMIT_EXCEEDED_NO_TRUNCATION_ALLOWED")
    spans=[(i,min(i+payload,n)) for i in starts]
    require(spans[0][0]==0 and spans[-1][1]==n and
            all(a[1]>=b[0] for a,b in zip(spans,spans[1:])), "WINDOW_COVERAGE_FAILURE")
    return spans

def bert_special_ids(tok):
    """Both pinned encoders are BERT single-sequence models.
    Validate the explicit framing against the public tokenizer call on synthetic text.
    Do not depend on legacy build_inputs_with_special_tokens or pad methods.
    """
    cls,sep,pad=tok.cls_token_id,tok.sep_token_id,tok.pad_token_id
    require(all(type(i) is int and i>=0 for i in (cls,sep,pad)) and len({cls,sep,pad})==3,
            "BERT_SPECIAL_ID_POLICY")
    for text in ("synthetic alpha beta", "A second invented test."):
        raw=tok(text,add_special_tokens=False,truncation=False)["input_ids"]
        framed=tok(text,add_special_tokens=True,truncation=False)["input_ids"]
        require(framed==[cls,*raw,sep],"BERT_FRAMING_MISMATCH")
    return cls,sep,pad

def padded_batch(rows,pad,device):
    import torch
    require(bool(rows) and all(0<len(r)<=MAX_LENGTH for r in rows),"BATCH_WINDOW_POLICY")
    width=max(map(len,rows))
    ids=[r+[pad]*(width-len(r)) for r in rows]
    masks=[[1]*len(r)+[0]*(width-len(r)) for r in rows]
    return {"input_ids":torch.tensor(ids,dtype=torch.long,device=device),
            "attention_mask":torch.tensor(masks,dtype=torch.long,device=device)}

def encode_windows(texts, tok, model, device, *, label="SYNTHETIC"):

    import torch
    stage(label+"_TOKENIZE")
    cls,sep,pad=bert_special_ids(tok)
    payload=MAX_LENGTH-2
    require(payload>OVERLAP,"TOKENIZER_SPECIAL_TOKEN_POLICY")
    pieces=[]
    counts=[]
    total=0
    covered=0
    for offset in range(0,len(texts),64):
        batch=tok(texts[offset:offset+64],add_special_tokens=False,
                  truncation=False)["input_ids"]
        for ids in batch:
            spans=window_ranges(len(ids),payload,OVERLAP)
            total+=len(ids)
            covered+=spans[0][1]+sum(max(0,b[1]-a[1]) for a,b in zip(spans,spans[1:]))
            counts.append(len(spans))
            for a,b in spans:
                row=[cls,*ids[a:b],sep]
                require(len(row)<=MAX_LENGTH,"WINDOW_OVERFLOW")
                pieces.append(row)
    require(len(counts)==len(texts) and total==covered and len(pieces)<=len(texts)*MAX_WINDOWS_PER_PARENT,
            "INCOMPLETE_WINDOW_COVERAGE")
    stage(label+"_FORWARD")
    vectors=[]
    with torch.inference_mode():
        for i in range(0,len(pieces),32):
            batch=padded_batch(pieces[i:i+32],pad,device)
            hidden=model(**batch).last_hidden_state.float()
            mask=batch["attention_mask"].unsqueeze(-1)
            pooled=(hidden*mask).sum(1)/mask.sum(1).clamp(min=1)
            norm=pooled.norm(dim=1)
            require(bool(torch.isfinite(pooled).all().item()) and bool((norm>0).all().item()),
                    "INVALID_EMBEDDING")
            vectors.append(torch.nn.functional.normalize(pooled,p=2,dim=1))
    flat=torch.cat(vectors)
    groups=list(flat.split(counts))
    return groups,{"parents":len(texts),"windows":len(pieces),"total_tokens":total,
        "covered_tokens":covered,"multiwindow_parents":sum(n>1 for n in counts),
        "max_windows_per_parent":max(counts),"payload_tokens":payload}

def blocks(groups):
    """Bucket by exact window count; no padding or skipped pairs."""
    import torch
    buckets=collections.defaultdict(list)
    for i,v in enumerate(groups):
        n=len(v)
        require(0<n<=MAX_WINDOWS_PER_PARENT,"INVALID_WINDOW_COUNT")
        buckets[n].append(i)
    for n,ids in sorted(buckets.items()):
        batch=max(1,min(128,WINDOW_BLOCK//n))
        for off in range(0,len(ids),batch):
            ix=ids[off:off+batch]
            yield ix,n,torch.cat([groups[i] for i in ix])

def compare_windows(left,right,union):
    """Exhaustive cross-parent maxima; all window comparisons, one count per parent pair."""
    import numpy as np
    import torch
    require(union.shape==(len(left),len(right)) and len(left)>0 and len(right)>0,"PAIR_SHAPE")
    thresholds=base.THRESHOLDS
    counts=[0]*len(thresholds)
    lflags=np.zeros((len(thresholds),len(left)),dtype=bool)
    rflags=np.zeros((len(thresholds),len(right)),dtype=bool)
    parent_pairs=window_pairs=0
    maximum=-1.0
    rblocks=list(blocks(right))
    for li,lw,lv in blocks(left):
        for ri,rw,rv in rblocks:
            sim=(lv@rv.T).reshape(len(li),lw,len(ri),rw).amax(dim=(1,3))
            require(bool(torch.isfinite(sim).all().item()),"NONFINITE_SIMILARITY")
            a=sim.clamp(-1,1).cpu().numpy()
            parent_pairs+=len(li)*len(ri)
            window_pairs+=len(li)*lw*len(ri)*rw
            maximum=max(maximum,float(a.max()))
            for k,t in enumerate(thresholds):
                mask=a>=t
                counts[k]+=int(mask.sum())
                lflags[k,li]|=mask.any(axis=1)
                rflags[k,ri]|=mask.any(axis=0)
                if k==0:
                    union[np.ix_(li,ri)]|=mask
    require(parent_pairs==len(left)*len(right) and
            window_pairs==sum(map(len,left))*sum(map(len,right)),"PAIR_COVERAGE_FAILURE")
    return {"parent_pairs":parent_pairs,"window_pairs":window_pairs,"maximum_cosine":round(maximum,7),
            "thresholds":{str(t):{"pairs":counts[k],"left_rows":int(lflags[k].sum()),
              "right_rows":int(rflags[k].sum())} for k,t in enumerate(thresholds)}}

def lexical_sample(left,right,mask,normalize,seed):
    """Reproducible sample of flagged and unflagged pairs; no pair IDs leave memory."""
    import numpy as np
    rng=np.random.default_rng(seed)
    result={}
    for label,flag in (("flagged",True),("unflagged",False)):
        flat=np.flatnonzero(mask.ravel()==flag)
        n=min(REVIEW_PER_STRATUM,len(flat))
        sample=rng.choice(flat,size=n,replace=False) if n else []
        counts={"question_jaccard_ge_0_5":0,"context_jaccard_ge_0_5":0}
        for k in sample:
            i,j=divmod(int(k),len(right))
            for view in ("question","context"):
                a=set(normalize(left[i][view]).split())
                b=set(normalize(right[j][view]).split())
                if a|b and len(a&b)/len(a|b)>=0.5:
                    counts[view+"_jaccard_ge_0_5"]+=1
        result[label]={"population_pairs":len(flat),"sample_pairs":n,**counts}
    return result

def file_hash(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        while chunk:=f.read(1024*1024):
            h.update(chunk)
    return h.hexdigest()

def model_load(key,root,device):
    stage("MODEL_"+key.upper()+"_DOWNLOAD")
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer,AutoModel
    model_id,revision,weight=MODELS[key]
    # Immutable public model assets only; no external inference service.
    path=Path(snapshot_download(repo_id=model_id,revision=revision,token=False,
         cache_dir=str(root/"models"),allow_patterns=["config.json","tokenizer.json","tokenizer_config.json",
          "special_tokens_map.json","added_tokens.json","vocab.txt","model.safetensors"]))
    require(path.name==revision and file_hash(path/"model.safetensors")==weight,"MODEL_WEIGHT_PIN_MISMATCH")
    stage("MODEL_"+key.upper()+"_TOKENIZER")
    tok=AutoTokenizer.from_pretrained(str(path),local_files_only=True,trust_remote_code=False)
    stage("MODEL_"+key.upper()+"_WEIGHTS")
    model=AutoModel.from_pretrained(str(path),local_files_only=True,trust_remote_code=False,
                                   use_safetensors=True).to(device).eval()
    require(model.config.model_type=="bert" and model.config.max_position_embeddings>=MAX_LENGTH,"MODEL_CONTEXT_LIMIT")
    stage("MODEL_"+key.upper()+"_FILE_HASHES")
    files={p.name:file_hash(p) for p in path.iterdir() if p.is_file()}
    return tok,model,files

def reconstruct(root):
    pkg=root/"research_data"
    pkg.mkdir()
    (pkg/"__init__.py").write_text("")
    for path in base.PINS:
        if path.startswith("src/"):
            (pkg/Path(path).name).write_bytes(base.download_module(path))
    sys.path.insert(0,str(root))
    p=root/"helper.py"
    p.write_bytes(base.download_module("scripts/e1_pubmedqa_lineage_kaggle.py"))
    spec=importlib.util.spec_from_file_location("master01b_helper",p)
    helper=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    candidate=importlib.import_module("research_data.e1_candidate_splits")
    advanced=importlib.import_module("research_data.e1_advanced_lineage")
    hard=importlib.import_module("research_data.e1_pubmedqa_test_exclusion")
    snap,pq=helper.libraries()
    med=helper.checked_parquet(snap,pq,helper.MEDHALLU,helper.MEDHALLU_REVISION,
                                helper.SOURCE_SHA256,root/"med",is_source=False)
    pub=helper.checked_parquet(snap,pq,helper.PUBMEDQA,helper.PUBMEDQA_REVISION,
                                helper.PUBMEDQA_SHA256,root/"pub",is_source=True)
    ids,digest=helper.official_test_ids()
    require(digest=="939fe566f09017d13b1ca64d2ddfee0bc2374b366048152997669cccedc44d51","HOLDOUT_PIN")
    prior,old=candidate.candidate_plan(med)
    require(prior["candidate_assignment_sha256"]==helper.CANDIDATE_ASSIGNMENT_SHA256,"OLD_MAPPING_PIN")
    summary,assignment=hard.candidate_remediation(med,old,advanced.pubmed_index(pub),ids,seed=base.SEED)
    hard.verify_summary(summary)
    require(summary["private_mapping_sha256"]==base.EXPECTED_MAPPING,"MAPPING_PIN")
    diagnostics=base.grouped_diagnostics(med,assignment)
    sets={s:[] for s in base.SPLITS}
    for (c,i),s in sorted(assignment.items()):
        row=med[c][i]
        sets[s].append({"question":row["Question"],"context":" ".join(row["Knowledge"])})
    held={}
    for c in sorted(pub):
        for row in pub[c]:
            pid=str(row["pubid"])
            if pid in ids:
                ctx=row["context"]
                parts=ctx.get("contexts",[]) if isinstance(ctx,dict) else ctx
                item={"question":row["question"],"context":" ".join(parts)}
                require(pid not in held or held[pid]==item,"AMBIGUOUS_HOLDOUT")
                held[pid]=item
    require(set(held)==ids,"HOLDOUT_COVERAGE")
    sets["official_holdout"]=[held[p] for p in sorted(held)]
    require({s:len(v) for s,v in sets.items()}==SIZES,"PARENT_COUNTS")
    exact=base.exact_screen(sets,advanced._norm)
    require(all(v["shared_groups"]==0 for view in exact.values() for v in view.values()),"EXACT_CONFLICT_REVIEW")
    return sets,advanced._norm,diagnostics,exact

def run():
    require(os.environ.get("MASTER01B_PRIVATE_RIGHTS_CONFIRMED")=="YES","RIGHTS_REQUIRED")
    os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:8"
    os.environ["HF_HUB_DISABLE_TELEMETRY"]="1"
    import numpy as np
    import torch
    require(torch.cuda.is_available(),"KAGGLE_GPU_REQUIRED")
    random.seed(base.SEED)
    torch.manual_seed(base.SEED)
    torch.cuda.manual_seed_all(base.SEED)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.use_deterministic_algorithms(True)
    device=torch.device("cuda:0")
    work=Path("/kaggle/working")
    scratch=Path("/kaggle/temp")
    require(work.is_dir() and not work.is_symlink() and not scratch.is_symlink(),"PRIVATE_STORAGE")
    scratch.mkdir(mode=0o700,exist_ok=True)
    out=work/REPORT
    require(not out.exists() and not out.is_symlink(),"DO_NOT_OVERWRITE")
    with tempfile.TemporaryDirectory(dir=scratch,prefix="master01b-") as tmp:
        root=Path(tmp)
        os.environ["HF_HOME"]=str(root/"hf")
        # Test both exact pinned encoders with only invented strings before reading medical data.
        for key in MODELS:
            tok,model,_=model_load(key,root,device)
            groups,cov=encode_windows(["synthetic alpha beta", "invented example "*400],
                                     tok,model,device,label="SMOKE_"+key.upper())
            require(cov["total_tokens"]==cov["covered_tokens"] and cov["multiwindow_parents"]==1,
                    "MODEL_SMOKE_COVERAGE")
            test=compare_windows(groups[:1],groups[1:],np.zeros((1,1),dtype=bool))
            require(test["parent_pairs"]==1,"MODEL_SMOKE_PAIR")
            del groups,tok,model
            torch.cuda.empty_cache()
        print("MASTER01B_BOTH_ENCODER_GPU_SMOKE_PASSED",flush=True)
        stage("PINNED_SOURCE_RECONSTRUCTION")
        sets,norm,diagnostics,exact=reconstruct(root)
        union={a+"__"+b:np.zeros((SIZES[a],SIZES[b]),dtype=bool) for a,b in PAIRS}
        results={}
        for key in MODELS:
            stage("ENCODER_"+key.upper())
            tok,model,files=model_load(key,root,device)
            result={"model":MODELS[key][0],"revision":MODELS[key][1],"weights_sha256":MODELS[key][2],
                    "file_sha256":files,"views":{}}
            for view in ("question","context"):
                groups={}
                coverage={}
                for s in base.SPLITS:
                    groups[s],coverage[s]=encode_windows([r[view] for r in sets[s]],tok,model,device,
                        label=key.upper()+"_"+view.upper()+"_"+s.upper())
                comparisons={}
                for a,b in PAIRS:
                    name=a+"__"+b
                    stage(key.upper()+"_"+view.upper()+"_COMPARE_"+name.upper())
                    comparisons[name]=compare_windows(groups[a],groups[b],union[name])
                result["views"][view]={"coverage":coverage,"comparisons":comparisons}
                del groups
                torch.cuda.empty_cache()
            results[key]=result
            del tok,model
            torch.cuda.empty_cache()
        stage("PRIVATE_REVIEW_CANDIDATE_DIAGNOSTICS")
        affected={s:np.zeros(SIZES[s],dtype=bool) for s in base.SPLITS}
        review={}
        for idx,(a,b) in enumerate(PAIRS):
            name=a+"__"+b
            mask=union[name]
            affected[a]|=mask.any(axis=1)
            affected[b]|=mask.any(axis=0)
            review[name]={"union_pairs":int(mask.sum()),"left_rows":int(mask.any(axis=1).sum()),
                          "right_rows":int(mask.any(axis=0).sum()),
                          "lexical_sample":lexical_sample(sets[a],sets[b],mask,norm,base.SEED+idx)}
        del sets,union
        report={"schema_version":1,"status":"SCREEN_COMPLETE_REVIEW_REQUIRED","source_commit":base.BASE,
                "runner_sha256":os.environ["MASTER01B_RUNNER_SHA256"],"mapping_sha256":base.EXPECTED_MAPPING,
                "pinned_module_sha256":base.PINS,"seed":base.SEED,"parents":SIZES,
                "window_length":MAX_LENGTH,"window_overlap":OVERLAP,
                "device":torch.cuda.get_device_name(0),
                "versions":{k:importlib.metadata.version(k) for k in
                            ("torch","transformers","huggingface_hub","numpy","pyarrow")},
                "encoders":results,"exact_screen":exact,"label_diagnostics":diagnostics,
                "review_candidates":review,"affected_parents":{s:int(v.sum()) for s,v in affected.items()},
                "clinical_adjudication_completed":False,"article_lineage_fully_certified":False,
                "complete_semantic_leakage_excluded":False,"split_changed":False,
                "final_split_approved":False,"scientific_e1_experiment_executed":False,
                "private_rows_exported":False}
        report["integrity_sha256"]=hashlib.sha256(base.canonical(report)).hexdigest()
        payload=json.dumps(report,sort_keys=True,indent=2)
        require(len(payload.encode())<100000,"REPORT_SIZE")
        stage("AGGREGATE_ONLY")
        with out.open("x",encoding="utf-8") as f:
            f.write(payload+"\n")
    print("MASTER01B_REPORT_SAVED="+REPORT,flush=True)
    print("MASTER01B_FINAL_STATUS=BLOCKED_PENDING_SCIENTIFIC_APPROVAL",flush=True)
