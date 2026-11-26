"""Label-blind, private calibration cohort planning; no datasets or inference.

Only a single release and task per candidate cohort. Protected prior rows
are used solely as covariates to quarantine connected candidate components.
No cross-task pooling, test reassignment, or claim of iid source groups.
"""
from collections import Counter,defaultdict
import re
from e1_master01d_v2_integrity import normalize,require,ORIGINAL,verify_checkpoint
from e1_master01c_core import digest,validate,PAIRS,SPLITS
from pirc_v2_power import feasibility

PARTITIONS=("train","tune","calibration","test")
TASKS={"bioasq14b":"yes_no","pubmedqa":"yes_no_maybe",
       "medqa":"mcq","medmcqa":"mcq"}
SEED=20261126

def _key(row):
    return (row["dataset"],row["release"],row["id"])

def plan(rows,*,dataset,release,task,near_links=(),seed=SEED):
    """Private inputs with namespaced stable source aliases (e.g. pmid:123).
    Protected rows include all prior 01C originals and official holdout,
    not just retained parents. Candidate rows carry published split
    designation: development or official_test. No labels accepted.
    Unknown provenance, unavailable evidence or unconfirmed reference
    availability => excluded; not fabricated. Semantic near_links are
    externally generated proxy edges, not certified exhaustive.
    """
    require(dataset in TASKS and TASKS[dataset]==task,"TASK_NOT_SUPPORTED")
    require(type(release) is str and bool(release.strip()) and
            type(seed) is int and type(rows) is list and bool(rows),"PLAN_INPUT")
    require(type(near_links) in (tuple,list),"LINK_SCHEMA")
    by={};parent={}
    fields={"dataset","release","id","task","native_split","sources","question",
            "context","evidence_available","reference_available","provenance_known"}
    for x in rows:
        require(type(x) is dict and set(x)==fields,"ROW_SCHEMA")
        require(all(type(x[k]) is str and bool(x[k].strip()) for k in
                    ("dataset","release","id","task","native_split","question")),
                    "ROW_IDENTITY")
        require(x["dataset"].casefold()!="medval" and
                "medval" not in x["dataset"].casefold(),"RESTRICTED_DATASET")
        require(x["native_split"] in ("development","official_test","protected") and
                all(type(x[k]) is bool for k in ("evidence_available","reference_available",
                                                "provenance_known")),"ROW_FLAGS")
        require(type(x["sources"]) is tuple and len(set(x["sources"]))==len(x["sources"])
                and all(type(v) is str and re.fullmatch(r"[a-z][a-z0-9_-]*:[^\s:]+",v)
                        for v in x["sources"]),"SOURCE_ALIAS_SCHEMA")
        require(all(not v.startswith("pmid:") or re.fullmatch(r"pmid:[1-9][0-9]*",v)
                    for v in x["sources"]),"PMID_CANONICAL")
        require(type(x["context"]) is str and
                (not x["evidence_available"] or bool(x["context"].strip())) and
                (not x["provenance_known"] or bool(x["sources"])),"MISSING_PROVENANCE")
        if x["native_split"]!="protected":
            require((x["dataset"],x["release"],x["task"])==(dataset,release,task),
                    "MIXED_TARGET_POPULATION")
        k=_key(x);require(k not in by,"DUPLICATE_ROW");by[k]=x;parent[k]=k
    def find(k):
        while parent[k]!=k:
            parent[k]=parent[parent[k]];k=parent[k]
        return k
    def merge(a,b):
        a,b=find(a),find(b)
        if a!=b: parent[max(a,b)]=min(a,b)
    cache={}
    for k,x in by.items():
        proxies=[("source",s) for s in x["sources"]]+[("question",normalize(x["question"]))]
        if x["context"].strip(): proxies.append(("context",normalize(x["context"])))
        for p in proxies:
            if p in cache: merge(k,cache[p])
            else: cache[p]=k
    for e in near_links:
        require(type(e) in (tuple,list) and len(e)==2 and
                all(type(k) is tuple and k in by for k in e) and e[0]!=e[1],"LINK_SCHEMA")
        merge(*e)
    components=defaultdict(list)
    for k in sorted(by): components[find(k)].append(k)
    # Exclude an entire development component if any member overlaps a
    # protected or official-test member; never move official test to dev.
    # Keep test inventory only if not connected to protected data; full native
    # test remains untouched, excluded count explicitly reported.
    assignments={};excluded=Counter();group_counts=Counter();candidates=Counter()
    def eligible(x):
        return x["provenance_known"] and x["evidence_available"] and x["reference_available"]
    for root,ks in sorted(components.items()):
        protected=any(by[k]["native_split"]=="protected" for k in ks)
        has_test=any(by[k]["native_split"]=="official_test" for k in ks)
        eligible_keys=[]
        for k in ks:
            x=by[k];native=x["native_split"]
            if native=="protected": continue
            candidates[native]+=1
            if protected: excluded["prior_protected_overlap"]+=1
            elif native=="development" and has_test: excluded["official_test_overlap"]+=1
            elif not eligible(x): excluded["missing_eligibility"]+=1
            else: eligible_keys.append(k)
        if not eligible_keys: continue
        # One deterministic representative, chosen without labels/outcomes.
        rank=lambda k:digest([seed,list(k)])
        k=min(eligible_keys,key=lambda k:(rank(k),k))
        excluded["nonrepresentative_same_group"]+=len(eligible_keys)-1
        if has_test: part="test"
        else:
            # Prespecified approximate 50/20/30 group allocation.
            u=int(digest([seed,"split",list(root)])[:16],16)/2**64
            part="train" if u<.5 else "tune" if u<.7 else "calibration"
        assignments[k]=part;group_counts[part]+=1
    require(sum(candidates.values())==len(assignments)+sum(excluded.values()),
            "INVENTORY_RECONCILIATION")
    mapping=sorted([*k,v] for k,v in assignments.items())
    summary={"candidate_rows":{k:candidates[k] for k in ("development","official_test")},
             "excluded_rows":{k:excluded[k] for k in ("prior_protected_overlap",
               "official_test_overlap","missing_eligibility","nonrepresentative_same_group")},
             "eligible_group_representatives":{k:group_counts[k] for k in PARTITIONS},
             "candidate_manifest_sha256":digest({"seed":seed,"mapping":mapping}),
             "official_test_reassigned":False,"independence_verified":False,
             "complete_semantic_screen_verified":False,"rights_verified":False,
             "final_split_frozen":False,"execution_authorized":False}
    return mapping,summary

def protected_from_01c(report,records,official_test_bytes):
    """Private protected-covariate adapter. Input reconstruction must be
    authenticated externally; checks do not prove original text membership.
    All 01C originals (including quarantined) plus holdout, not the full
    universe of previously excluded datasets.
    """
    from e1_master01d_v2_integrity import verify_holdout_identity
    verify_checkpoint(report)
    validate(records,{p:set() for p in PAIRS})
    require({s:len(records[s]) for s in SPLITS}==ORIGINAL,"PROTECTED_COUNTS")
    verify_holdout_identity(records,official_test_bytes)
    result=[]
    for s in SPLITS:
        for r in records[s]:
            require(re.fullmatch(r"[1-9][0-9]*",r["source"]) is not None,"PROTECTED_PMID")
            result.append({"dataset":"prior01c","release":"master01c","id":str(r["key"][0])+":"+str(r["key"][1]),
              "task":"protected_covariates","native_split":"protected","sources":("pmid:"+r["source"],),
              "question":r["question"],"context":r["context"],"evidence_available":True,
              "reference_available":False,"provenance_known":True})
    return result

def prospective_power(n):
    """Only hypothetical rates, not pilot measurements. 2000 is a computational
    cap for this planning function, not an imposed scientific sample cap."""
    require(type(n) is int and n>=0,"GROUP_COUNT")
    if n==0:
        return {"status":"NO_ELIGIBLE_CALIBRATION_GROUPS","groups":0,"scenarios":[]}
    m=min(n,2000)
    scenarios=[]
    for k in (1,3,12):
        for r,i in ((.03,.01),(.07,.03)):
            scenarios.append({"hypothetical_release_error":r,
                "hypothetical_introduced_error":i,
                **feasibility(m,policies=k,true_release_error=r,true_introduced_error=i)})
    return {"status":"HYPOTHETICAL_PLANNING_ONLY","groups":n,"evaluated_groups":m,
            "scenario_count":len(scenarios),"scenarios":scenarios,
            "actual_power_verified":False,"policy_family_selected":False}
