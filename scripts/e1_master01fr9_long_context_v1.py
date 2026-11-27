"""R9 candidate A: full-coverage, development-only segmented NLI.
No acquisition, dispatch, checkpoint writes or clinical truth claims here.
The operator must explicitly approve the new recipe before private execution.
"""
import math
import json
import e1_master01fr8_neural_v1 as r8
import e1_master01fr7_lexical_v1 as r7
import e1_master01fr6r_checkpoint_v1 as cp
import e1_master01fr9_token_audit_v1 as audit
import e1_master01e_v4_neural as n

CWIDTH, COVERLAP, SWIDTH, SOVERLAP = 192, 48, 317, 64
BATCH=4
POLICY="r9_cartesian_segment_max_pool_v1"
LEXICAL_KEYS=["ap","auroc","brier","r7_ap","r7_auroc","r7_brier",
              "delta_ap","delta_auroc","delta_brier","repeat_max_abs_score_delta","historical_match"]
BUDGET_KEYS=["fit_groups","selection_groups","over_budget","fit_full_pairs",
             "selection_full_pairs","selection_removed_pairs","selection_permuted_pairs",
             "total_pairs","max_context_windows","max_statement_windows"]
STAGES=[{"name":"lexical_same_run","metrics":LEXICAL_KEYS},
        {"name":"token_budget","metrics":BUDGET_KEYS}]+json.loads(cp.canonical(r8.STAGES[2:]))
def require(ok,code):
    if not ok: raise ValueError(code)

def tiles(ids,width,overlap):
    require(type(ids) is list and bool(ids) and
            all(type(i) is int and i>=0 for i in ids),"R9_TOKEN_IDS")
    count=audit.windows(len(ids),width,overlap)
    parts=[ids[i*(width-overlap):i*(width-overlap)+width] for i in range(count)]
    require(all(0<len(v)<=width for v in parts) and
            (count-1)*(width-overlap)+len(parts[-1])==len(ids),"R9_COVERAGE")
    return parts

def pool(rows):
    # Each hypothesis segment has separate best entailment and contradiction
    # across all context windows; these maxima need not sum to one.
    require(type(rows) is list and bool(rows),"R9_POOL_EMPTY")
    e=[];c=[]
    for group in rows:
        require(type(group) is list and bool(group),"R9_POOL_GROUP")
        for x in group:
            require(type(x) is dict and set(x)==n.LABELS and
                    all(type(v) in (int,float) and math.isfinite(v) and 0<=v<=1
                        for v in x.values()) and abs(sum(x.values())-1)<1e-5,"R9_POSTERIOR")
        e.append(max(x["entailment"] for x in group))
        c.append(max(x["contradiction"] for x in group))
    return [max(e),max(c),sum(e)/len(e),sum(c)/len(c),min(e),
            sum(x>=.5 for x in e)/len(e)]

def gpu_forward(engine,batch):
    import torch
    require(engine.device=="cuda:0" and torch.cuda.is_available(),"R9_GPU_REQUIRED")
    with torch.inference_mode():
        tensors={k:torch.tensor([row[k] for row in batch],device=engine.device,dtype=torch.long)
                 for k in batch[0]}
        logits=engine.model(**tensors).logits
        require(tuple(logits.shape)==(len(batch),3) and bool(torch.isfinite(logits).all()),"R9_NLI_OUTPUT")
        return [{engine.labels[j]:v for j,v in enumerate(row)}
                for row in logits.float().softmax(dim=-1).cpu().tolist()]

def infer_case(engine,statement,contexts,*,removed=False,forward=gpu_forward):
    require(type(removed) is bool and type(contexts) is list and bool(contexts),"R9_CONTEXT")
    require(all(type(w) is list and 0<len(w)<=CWIDTH for w in contexts),"R9_CONTEXT")
    hyps=tiles(engine.tokens(statement),SWIDTH,SOVERLAP)
    premises=[[]] if removed else contexts
    require(engine.special==3 and n.MAX_LEN==512 and n.WINDOW==CWIDTH,"R9_ENCODER_CONTRACT")
    # No dummy windows or dropped pairs. Batch at most four; partial batch valid.
    groups=[]
    for h in hyps:
        posts=[]
        for start in range(0,len(premises),BATCH):
            batch=[engine.encode(c,h) for c in premises[start:start+BATCH]]
            result=forward(engine,batch)
            require(type(result) is list and len(result)==len(batch),"R9_FORWARD_COUNT")
            posts.extend(result)
        require(len(posts)==len(premises),"R9_FORWARD_COUNT")
        groups.append(posts)
    return pool(groups),len(hyps)*len(premises)

def make_plan(commit,run_id,*,blobs,runtime,receipt_pin,r6_pin,r7_pin,
              audit_pin,v5_plan_pin,v5_lexical_pin,v5_budget_pin):
    pins={"receipt":receipt_pin,"r6":r6_pin,"r7":r7_pin,"audit":audit_pin,
          "v5_plan":v5_plan_pin,"v5_lexical":v5_lexical_pin,"v5_budget":v5_budget_pin,
          "model":r8.DIGEST,"dataset":r8.r3.PIN}
    require(all(cp.match(cp.HEX,v) for v in pins.values()),"R9_INPUT_PINS")
    require(type(blobs) is dict and bool(blobs) and
            all(cp.match(cp.COMMIT,v) for v in blobs.values()),"R9_BLOB_PINS")
    recipe={"policy":POLICY,"blobs":blobs,"runtime":runtime,"device":"cuda:0",
            "context_width":CWIDTH,"context_overlap":COVERLAP,
            "statement_width":SWIDTH,"statement_overlap":SOVERLAP,
            "batch":BATCH,"seed":r7.SEED,"bootstrap":r7.NBOOT,
            "gpu_seed":20261125,"deterministic":True,"raw_nonentailment":"one_minus_min_entailment",
            "lexical_repeats":2,"lexical_repeat_tolerance":1e-8}
    plan={"schema":1,"run_id":run_id,"scope":"private_development_aggregate",
          "source_commit":commit,"input_sha256":cp.digest(cp.canonical(pins)),
          "recipe_sha256":cp.digest(cp.canonical(recipe)),
          "stages":json.loads(cp.canonical(STAGES))}
    return cp.validate_plan(plan,cp.digest(cp.canonical(plan)))

class Study(r8.Study):
    def __init__(self,fit,selection,engine_factory,report,expected_audit,*,authorized=False,
                 forward=gpu_forward,emit=lambda _:None):
        require(authorized is True,"R9_PROTOCOL_APPROVAL_REQUIRED")
        super().__init__(fit,selection,engine_factory,report)
        self.fit_rows=fit;self.selection_rows=selection
        self.expected_audit=expected_audit;self.forward=forward;self.emit=emit
        self.actual_pairs=[]
    def lexical(self):
        if self.lex is not None:return self.lex
        from threadpoolctl import threadpool_limits
        with threadpool_limits(limits=1):
            def fit_once():
                a,b,_=r7.block(self.sf,self.ss,["word","char_wb"])
                d,e,cv=r7.block(self.cf,self.cs,["word","char_wb"])
                scaler=self.scaler().fit(r7.numeric(self.sf,self.cf))
                def pack(s,c,sv,ctx):
                    return self.hstack([sv,ctx,self.csr(scaler.transform(r7.numeric(s,c)))],format="csr")
                x=pack(self.sf,self.cf,a,d);z=pack(self.ss,self.cs,b,e)
                scores=r7.prob(r7.classifier(x,self.yf),z)
                return x,z,scores,lambda contexts:pack(self.ss,contexts,b,r7.transformed(cv,contexts))
            first,second=fit_once(),fit_once()
            for v in (first[2],second[2]):
                require(v.shape==(len(self.ys),) and self.np.isfinite(v).all()
                        and ((v>=0)&(v<=1)).all(),"R9_LEXICAL_SCORE")
            require(first[0].shape==second[0].shape and first[1].shape==second[1].shape
                    and (first[0]!=second[0]).nnz==0 and (first[1]!=second[1]).nnz==0,"R9_LEXICAL_FEATURE_DRIFT")
            delta=float(self.np.max(self.np.abs(first[2]-second[2])))
            a=r7.measures(self.ys,first[2]);b=r7.measures(self.ys,second[2])
            require(delta<=1e-8 and all(abs(a[k]-b[k])<=1e-8 for k in a),"R9_LEXICAL_REPEAT")
            old=self.report["aggregates"]["evidence_ablation"]
            self.lexical_record={**a,**{"r7_"+k:float(old[k]) for k in a},
                  **{"delta_"+k:float(a[k]-old[k]) for k in a},
                  "repeat_max_abs_score_delta":delta,
                  "historical_match":int(all(abs(a[k]-old[k])<=1e-8 for k in a))}
            self.lex=first
            return self.lex
    def budget(self):
        if self.parts is not None:return self.budget_result
        if self.engine is None:self.engine=self.engine_factory()
        eng=self.engine
        observed=audit.audit(self.fit_rows,self.selection_rows,eng.tokens,authorized=True)
        require(observed==self.expected_audit,"R9_AUDIT_REPRODUCTION")
        cost=observed["candidate_costs"][0]
        require((cost["context_width"],cost["context_overlap"],cost["statement_width"],
                 cost["statement_overlap"])==(CWIDTH,COVERLAP,SWIDTH,SOVERLAP),"R9_CANDIDATE_BINDING")
        parts=[tiles(eng.tokens(c),CWIDTH,COVERLAP) for c in self.cf+self.cs]
        sw=[len(tiles(eng.tokens(s),SWIDTH,SOVERLAP)) for s in self.sf+self.ss]
        k=len(self.sf); cw=[len(p) for p in parts]
        full=[a*b for a,b in zip(sw,cw)]
        perm=cw[k+1:]+cw[k:k+1]
        vals={"fit_groups":k,"selection_groups":len(self.ss),"over_budget":0,
              "fit_full_pairs":sum(full[:k]),"selection_full_pairs":sum(full[k:]),
              "selection_removed_pairs":sum(sw[k:]),
              "selection_permuted_pairs":sum(a*b for a,b in zip(sw[k:],perm)),
              "max_context_windows":max(cw),"max_statement_windows":max(sw)}
        keys=("fit_full_pairs","selection_full_pairs","selection_removed_pairs","selection_permuted_pairs")
        vals["total_pairs"]=sum(vals[q] for q in keys)
        require(all(vals[q]==cost[q]["sum"] for q in keys) and
                vals["total_pairs"]==cost["total_pairs"],"R9_PAIR_BUDGET_MISMATCH")
        self.parts=parts;self.budget_result=vals
        return vals
    def features(self,statements,parts,removed=False):
        require(len(statements)==len(parts) and bool(parts),"R9_FEATURE_INPUT")
        out=[];pairs=0
        for i,(s,c) in enumerate(zip(statements,parts),1):
            f,count=infer_case(self.engine,s,c,removed=removed,forward=self.forward)
            out.append(f);pairs+=count
            if i%16==0 or i==len(parts):self.emit("R9_CASES_COMPLETED="+str(i)+"/"+str(len(parts)))
        self.actual_pairs.append(pairs)
        return self.np.asarray(out,dtype=float)
    def compute(self,stage,prior,nboot=r7.NBOOT):
        if stage=="lexical_same_run":
            self.lexical();return dict(self.lexical_record)
        if stage=="token_budget":
            return self.budget()
        require(stage in ("neural_hybrid_full","evidence_ablation"),"R9_STAGE")
        self.lexical();budget=self.budget()
        require(prior.get("lexical_same_run")==self.lexical_record and
                prior.get("token_budget")==budget,"R9_PRIOR_STAGE_DRIFT")
        result=super().compute(stage,prior,nboot=nboot)
        k=("fit_full_pairs","selection_full_pairs")
        if stage=="evidence_ablation":k+=("selection_removed_pairs","selection_permuted_pairs")
        require(self.actual_pairs==[budget[x] for x in k],"R9_ACTUAL_PAIR_COUNT")
        if stage=="neural_hybrid_full":
            # Segment-level weakest best entailment, not original R8 max.
            result["raw_nonentailment_ap"]=r7.measures(self.ys,1-self.full[5][:,4])["ap"]
        return result
