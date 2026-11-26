"""Private pre-inference audit of exactly the existing train/dev pilot sample.
No I/O, model forward, row export, truncation, resampling, or fallback.
"""
import math
import e1_master01e_v4_neural as n
import e1_master01e_v2_pilot as p
import e1_master01e_baselines as b
require=n.require

def checked_windows(engine,text,vocab):
    raw=engine.tokens(text)
    require(all(type(i) is int and 0<=i<vocab for i in raw),"AUDIT_TOKEN_RANGE")
    parts,total=engine.windows(text)
    expected=[]
    for start in range(0,len(raw),n.WINDOW-n.OVERLAP):
        expected.append(raw[start:start+n.WINDOW])
        if start+n.WINDOW>=len(raw):break
    require(total==len(raw) and parts==expected and 1<=len(parts)<=n.SLOTS,
            "AUDIT_WINDOW_COVERAGE")
    return parts

def checked_pair(engine,premise,hyp,vocab):
    enc=engine.encode(premise,hyp)
    length=len(premise)+len(hyp)+engine.special
    keys={"input_ids","attention_mask"}|({"token_type_ids"} if engine.has_types else set())
    require(type(enc) is dict and set(enc)==keys and all(type(v) is list and len(v)==n.MAX_LEN
            and all(type(x) is int for x in v) for v in enc.values()),"AUDIT_TENSOR_SHAPE")
    require(enc["input_ids"]==[engine.cls,*premise,engine.sep,*hyp,engine.sep]+
            [engine.pad]*(n.MAX_LEN-length)
            and all(0<=x<vocab for x in enc["input_ids"]),"AUDIT_TOKEN_RANGE")
    require(enc["attention_mask"]==[1]*length+[0]*(n.MAX_LEN-length),"AUDIT_ATTENTION_MASK")
    if engine.has_types:
        require(enc["token_type_ids"]==[0]*(len(premise)+2)+[1]*(len(hyp)+1)+
                [0]*(n.MAX_LEN-length),"AUDIT_SEGMENTS")
    return length

def audit_partition(engine,pairs,partition):
    selected=p.sample(pairs,partition)
    perm=b.context_permutation(selected,partition,p.SEED)
    donor={row["parent"]:row["context"] for row in perm}
    require(set(donor)=={row["parent"] for row in selected},"AUDIT_PERMUTATION")
    vocab=engine.model.config.vocab_size
    require(type(vocab) is int and vocab>0 and engine.special==3,"AUDIT_MODEL_CONTRACT")
    cost={k:dict(forward_examples=0,padded_positions=0,active_tokens=0,real_windows=0)
          for k in p.CONDITIONS}
    for row in selected:
        full=checked_windows(engine,row["context"],vocab)
        other=checked_windows(engine,donor[row["parent"]],vocab)
        for role in ("ground_truth","hallucinated"):
            answer=row["answers"][role]
            hyp=engine.tokens(answer)
            require(all(type(i) is int and 0<=i<vocab for i in hyp),"AUDIT_TOKEN_RANGE")
            x={"question":row["question"],"answer":answer,"context":row["context"]}
            require(bool(b.words(answer)),"AUDIT_BASELINE_EMPTY_ANSWER")
            for mode in b.MODES:
                f=b.features(x,mode)
                require(bool(f) and all(math.isfinite(v) for v in f),"AUDIT_BASELINE_FEATURES")
            for k,parts in (("full",full),("removed",[]),("permuted",other)):
                slots=parts+[[] for _ in range(n.SLOTS-len(parts))]
                require(len(slots)==n.SLOTS,"AUDIT_SLOT_COUNT")
                active=sum(checked_pair(engine,part,hyp,vocab) for part in slots)
                cost[k]["forward_examples"]+=n.SLOTS
                cost[k]["padded_positions"]+=n.SLOTS*n.MAX_LEN
                cost[k]["active_tokens"]+=active
                cost[k]["real_windows"]+=len(parts)
    require(all(c["forward_examples"]==len(selected)*2*n.SLOTS for c in cost.values()),
            "AUDIT_BUDGET")
    return cost

def audit_both(engine,train,dev,mark):
    p.validate_private_inputs(train,dev)
    mark("AUDIT_TRAIN")
    tr=audit_partition(engine,train,"train")
    mark("AUDIT_DEV")
    dv=audit_partition(engine,dev,"dev")
    return {"train":tr,"dev":dv}
