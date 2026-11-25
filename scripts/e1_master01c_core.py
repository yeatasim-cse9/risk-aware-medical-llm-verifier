"""Private in-memory MASTER 01C review and conservative quarantine core.
No data downloads, external writes, automatic adjudication or final freeze.
"""
from collections import Counter, defaultdict
from bisect import bisect_right
from itertools import combinations
import hashlib
import hmac
import json
import random

SPLITS=("train","dev","test","official_holdout")
PAIRS=tuple(combinations(SPLITS,2))
SEED=20261125
RUBRIC={"same_source","near_duplicate","topic_only","clinically_distinct","unresolved"}
CLINICAL={"supported","contradicted","insufficient_evidence","ambiguous","not_applicable"}
GATES=("dataset_rights","source_provenance_review","semantic_disposition",
       "clinical_label_review","independent_scientific_approval",
       "locked_analysis_protocol","private_manifest_review")

def require(ok,code):
    if not ok:
        raise ValueError(code)

def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":"),
                          ensure_ascii=True).encode()).hexdigest()

def validate(records,flags):
    require(type(records) is dict and set(records)==set(SPLITS),"SPLIT_SET")
    seen=set()
    for s in SPLITS:
        require(type(records[s]) is list and bool(records[s]),"EMPTY_SPLIT")
        for row in records[s]:
            require(type(row) is dict and set(row)=={"key","source","question","context",
                    "config","category","difficulty"},"ROW_SCHEMA")
            k=row["key"]
            require(type(k) is tuple and len(k)==2 and type(k[0]) is str and
                    type(k[1]) is int and k[1]>=0 and k not in seen,"PRIVATE_KEY")
            require(all(type(row[f]) is str and row[f].strip() for f in
                    ("source","question","context","config","category","difficulty")),
                    "MISSING_SOURCE_OR_METADATA")
            require(k[0]==row["config"] and
                    (s=="official_holdout")==(k[0]=="official_holdout"),"HOLDOUT_IDENTITY")
            seen.add(k)
    require(type(flags) is dict and set(flags)==set(PAIRS),"PAIR_SET")
    for a,b in PAIRS:
        require(type(flags[a,b]) is set,"PAIR_TYPE")
        for p in flags[a,b]:
            require(type(p) is tuple and len(p)==2 and all(type(i) is int for i in p)
                    and 0<=p[0]<len(records[a]) and 0<=p[1]<len(records[b]),"PAIR_INDEX")

def review_plan(records,flags,normalize,sample_per_stratum=100):
    """Private queue: all flags + uniform unflagged SRSWOR, no public pair IDs."""
    validate(records,flags)
    require(type(sample_per_stratum) is int and 1<=sample_per_stratum<=1000,"SAMPLE_SIZE")
    queue=[]; strata={}
    def jaccard(x,y):
        x,y=set(normalize(x).split()),set(normalize(y).split())
        return len(x&y)/len(x|y) if x|y else 0.0
    for idx,(a,b) in enumerate(PAIRS):
        width=len(records[b]); total=len(records[a])*width
        blocked=sorted(i*width+j for i,j in flags[a,b])
        available=total-len(blocked)
        rng=random.Random(SEED+1000+idx)
        ranks=rng.sample(range(available),min(sample_per_stratum,available))
        def unrank(rank):
            lo,hi=rank,total-1
            while lo<hi:
                mid=(lo+hi)//2
                if mid+1-bisect_right(blocked,mid)>=rank+1:
                    hi=mid
                else:
                    lo=mid+1
            return lo
        sample=[unrank(rank) for rank in ranks]
        require(len(set(sample))==len(sample) and not set(sample)&set(blocked),"COMPLEMENT")
        for flag,values in ((True,blocked),(False,sample)):
            for flat in values:
                i,j=divmod(flat,width)
                x,y=records[a][i],records[b][j]
                queue.append({"pair":(x["key"],y["key"]),"stratum":(a,b),"flagged":flag,
                    "same_source":x["source"]==y["source"],
                    "question_jaccard":jaccard(x["question"],y["question"]),
                    "context_jaccard":jaccard(x["context"],y["context"])})
        strata[a+"__"+b]={"flagged_population":len(blocked),
            "unflagged_population":available,"unflagged_review_sample":len(sample)}
    return queue,strata

def quarantine(records,flags,normalize):
    """Exclude flagged candidate endpoints and connected source/text proxies.
    Holdout rows are kept unchanged; source-group conflicts fail closed.
    Returns PRIVATE retained records and count-only aggregate.
    """
    validate(records,flags)
    rows={r["key"]:(s,r) for s in SPLITS for r in records[s]}
    parent={k:k for k in rows}
    def find(k):
        while parent[k]!=k:
            parent[k]=parent[parent[k]]
            k=parent[k]
        return k
    def merge(k,l):
        a,b=find(k),find(l)
        if a!=b: parent[max(a,b)]=min(a,b)
    cache={}
    for k,(_,r) in rows.items():
        for field in ("source","question","context"):
            v=r[field] if field=="source" else normalize(r[field])
            require(type(v) is str and bool(v.strip()),"EMPTY_PROXY")
            proxy=(field,v)
            if proxy in cache: merge(k,cache[proxy])
            else: cache[proxy]=k
    affected={s:set() for s in SPLITS}
    for a,b in PAIRS:
        for i,j in flags[a,b]:
            affected[a].add(records[a][i]["key"])
            affected[b].add(records[b][j]["key"])
    roots={find(k) for s in SPLITS for k in affected[s]}
    roots.update(find(r["key"]) for r in records["official_holdout"])
    retained={s:[r for r in records[s] if s=="official_holdout" or
                 find(r["key"]) not in roots] for s in SPLITS}
    keep={r["key"] for s in SPLITS for r in retained[s]}
    groups=defaultdict(set)
    for s in SPLITS:
        require(bool(retained[s]),"EMPTY_RETAINED_SPLIT")
        for r in retained[s]: groups[find(r["key"])].add(s)
    require(all(len(v)==1 for v in groups.values()),"RESIDUAL_SOURCE_PROXY_CONFLICT")
    residual=sum(records[a][i]["key"] in keep and records[b][j]["key"] in keep
                 for a,b in PAIRS for i,j in flags[a,b])
    require(residual==0,"RESIDUAL_FLAGGED_PAIR")
    require(retained["official_holdout"]==records["official_holdout"],"HOLDOUT_CHANGED")
    mapping=sorted([*r["key"],s] for s in SPLITS[:3] for r in retained[s])
    counts={}
    for s in SPLITS:
        n,m=len(records[s]),len(retained[s])
        counts[s]={"original":n,"flagged_endpoints":len(affected[s]),"retained":m,
            "quarantined":n-m,"paired_examples":2*m if s!="official_holdout" else 0}
    return retained,{"counts":counts,"mapping_sha256":digest({"seed":SEED,"assignment":mapping}),
                     "residual_flagged_pairs":residual,"source_proxy_conflicts":0}

def balance(records,retained):
    """Compare true private category identities; export only sorted counts and TV."""
    out={}
    for s in SPLITS[:3]:
        out[s]={}
        def compare(a,b,field):
            x=Counter(r[field] for r in a);y=Counter(r[field] for r in b)
            tv=round(sum(abs(x[k]/sum(x.values())-y[k]/sum(y.values()))
                         for k in x.keys()|y.keys())/2,8) if x and y else None
            return {"before_sorted":sorted(x.values(),reverse=True),
                    "after_sorted":sorted(y.values(),reverse=True),"total_variation":tv}
        for field in ("config","category","difficulty"):
            out[s][field]=compare(records[s],retained[s],field)
        out[s]["by_config"]={}
        for cfg in sorted({r["config"] for r in records[s]}):
            a=[r for r in records[s] if r["config"]==cfg]
            b=[r for r in retained[s] if r["config"]==cfg]
            out[s]["by_config"][cfg]={f:compare(a,b,f) for f in ("category","difficulty")}
    return out

def reviewer_summary(expected,first,second,adjudicated,*,clinical=False):
    """Structural coverage and agreement only; cannot authenticate human judgments."""
    require(type(clinical) is bool,"REVIEW_MODE")
    vocabulary=CLINICAL if clinical else RUBRIC
    unresolved={"insufficient_evidence","ambiguous","not_applicable"} if clinical else {"unresolved"}
    require(type(expected) is set and bool(expected),"REVIEW_EXPECTED")
    for v in (first,second):
        require(type(v) is dict and set(v)==expected and all(type(x) is str and x in vocabulary
                for x in v.values()),"REVIEW_COVERAGE")
    needed={k for k in expected if first[k]!=second[k] or first[k] in unresolved}
    require(type(adjudicated) is dict and set(adjudicated)==needed and
            all(type(x) is str and x in vocabulary for x in adjudicated.values()),"ADJUDICATION_COVERAGE")
    n=len(expected);agree=sum(first[k]==second[k] for k in expected)
    a,b=Counter(first.values()),Counter(second.values())
    pe=sum(a[v]*b[v] for v in vocabulary)/n**2
    return {"n":n,"agreement_count":agree,"agreement":agree/n,
            "cohen_kappa":(agree/n-pe)/(1-pe) if pe<1 else None,
            "adjudication_required":len(needed),
            "unresolved":sum(adjudicated.get(k,first[k]) in unresolved for k in expected),
            "human_authenticity_verified":False,"scientific_approval":False}

def freeze_preparation(mapping,expected_digest,evidence):
    """No final freeze; even structurally complete gates need authentication."""
    require(type(mapping) is list and bool(mapping),"MANIFEST_SCHEMA")
    seen=set()
    for row in mapping:
        require(type(row) is list and len(row)==3 and type(row[0]) is str and
                row[0]!="official_holdout" and type(row[1]) is int and row[1]>=0
                and row[2] in SPLITS[:3],"MANIFEST_ROW")
        k=tuple(row[:2]);require(k not in seen,"MANIFEST_DUPLICATE");seen.add(k)
    require(mapping==sorted(mapping) and {r[2] for r in mapping}==set(SPLITS[:3]),
            "MANIFEST_ORDER_OR_SPLIT")
    require(digest({"seed":SEED,"assignment":mapping})==expected_digest,"MANIFEST_DIGEST")
    require(type(evidence) is dict and set(evidence)==set(GATES) and
            all(type(v) is bool for v in evidence.values()),"GATE_SCHEMA")
    return {"missing_gates":[k for k in GATES if not evidence[k]],
            "external_authentication_required":True,"final_split_approved":False,
            "final_split_frozen":False,"scientific_e1_experiment_executed":False}

def private_token(nonce,kind,value):
    require(type(nonce) is bytes and len(nonce)>=32,"PRIVATE_REVIEW_NONCE")
    payload=json.dumps([kind,value],sort_keys=True,separators=(",",":"),
                       ensure_ascii=True).encode()
    return hmac.new(nonce,payload,hashlib.sha256).hexdigest()

def semantic_packets(records,queue,nonce):
    """Authorized private environment only. Do not display or export packets.

    Blind proposed split, semantic flag, similarity and answer roles.
    Source identity is necessary evidence for source-equivalence adjudication.
    Return reviewer packets and separate custodian-only key.
    """
    validate(records,{p:set() for p in PAIRS})
    rows={r["key"]:r for s in SPLITS for r in records[s]}
    packets=[]; key={}
    require(type(queue) is list and bool(queue),"QUEUE_SCHEMA")
    for q in queue:
        require(type(q) is dict and "pair" in q and type(q["pair"]) is tuple and
                len(q["pair"])==2 and all(k in rows for k in q["pair"]),"QUEUE_PAIR")
        pair=q["pair"]
        token=private_token(nonce,"semantic",pair)
        require(token not in key,"DUPLICATE_REVIEW_PAIR")
        sides=list(pair)
        if int(token[-1],16)%2: sides.reverse()
        packets.append({"review_id":token,"cases":[
            {f:rows[k][f] for f in ("question","context","source")} for k in sides]})
        key[token]=pair
    packets.sort(key=lambda x:x["review_id"])
    return packets,key

def clinical_review_plan(records,answers,allocation,nonce):
    """Private preregistered stratified paired-role clinical review.

    Allocation is explicit for every observed candidate stratum. Neither
    official holdout answers nor reviewer outcomes may enter the sample plan.
    Returns private blinded packets + separate custodian role key and strata.
    """
    validate(records,{p:set() for p in PAIRS})
    groups=defaultdict(list)
    rows={}
    for s in SPLITS[:3]:
        for row in records[s]:
            k=row["key"];rows[k]=row
            groups[(s,row["config"],row["category"],row["difficulty"])].append(k)
    require(type(answers) is dict and set(answers)==set(rows),"CLINICAL_ANSWER_COVERAGE")
    require(all(type(a) is tuple and len(a)==2 and
                all(type(x) is str and x.strip() for x in a) for a in answers.values()),
            "CLINICAL_ANSWER_PAIR")
    require(type(allocation) is dict and set(allocation)==set(groups),"CLINICAL_ALLOCATION")
    packets=[];key={};strata={}
    for i,g in enumerate(sorted(groups)):
        population=sorted(groups[g])
        n=allocation[g]
        require(type(n) is int and 0<=n<=len(population),"CLINICAL_SAMPLE_SIZE")
        selected=random.Random(SEED+2000+i).sample(population,n)
        strata[g]={"population":len(population),"sample":n}
        for k in selected:
            parent_token=private_token(nonce,"clinical-parent",k)
            order=(1,0) if int(parent_token[-1],16)%2 else (0,1)
            for role in order:
                token=private_token(nonce,"clinical-answer",(k,role))
                require(token not in key,"DUPLICATE_REVIEW_ANSWER")
                packets.append({"review_id":token,"question":rows[k]["question"],
                    "context":rows[k]["context"],"source":rows[k]["source"],
                    "answer":answers[k][role]})
                key[token]=(k,role)
    require(bool(packets),"EMPTY_CLINICAL_SAMPLE")
    packets.sort(key=lambda x:x["review_id"])
    return packets,key,strata
