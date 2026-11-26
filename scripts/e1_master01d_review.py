"""MASTER 01D private review orchestration and design-based uncertainty.

Pure in-memory functions. No data access on import, no file export, no automatic
scientific approval. Private return values must never enter logs/public artifacts.
"""
from collections import Counter, defaultdict
from fractions import Fraction
from math import comb
import re

import e1_master01c_core as c
import e1_master01c_private as prior

require=c.require
CHECKPOINT="83de2eb19a6ab64331a88a4ac1a7fe6b26f440fb8430acd0ff8d277e20871a5d"
MAPPING="025d4323b4db15435b21c7479bfc3f63d9dbbef087b2976a26c542959e66c9cb"
COUNTS={"train":7427,"dev":890,"test":887,"official_holdout":500}
ORIGINAL={"train":7579,"dev":947,"test":948,"official_holdout":500}
CLINICAL_CAP=12
SEMANTIC_POSITIVE={"same_source","near_duplicate"}
CLINICAL_UNCERTAIN={"insufficient_evidence","ambiguous","not_applicable"}
IMPACT={"none","minor","potentially_major","uncertain"}

def check_checkpoint(report,private):
    """Pin reconstructed checkpoint, NOT identity of original unpersisted pairs."""
    require(report["status"]=="PROVISIONAL_REMEDIATION_REVIEW_REQUIRED" and
            report["previous_evidence_sha256"]==prior.PRIOR_EVIDENCE and
            report["previous_mapping_sha256"]==prior.base.EXPECTED_MAPPING and
            report["encoder_aggregate_sha256"]==prior.EXPECTED_ENCODERS and
            report["dataset_pins"]==prior.DATA_PINS and
            report["model_asset_set_sha256"]==prior.EXPECTED_ASSETS,"CHECKPOINT_PINS")
    prior.compare_maxima(report["maxima"])
    require(report["quarantine"]["mapping_sha256"]==MAPPING and
            report["quarantine"]["residual_flagged_pairs"]==0 and
            report["quarantine"]["source_proxy_conflicts"]==0 and
            report["review_queue_count"]==771,"CHECKPOINT_REMEDIATION")
    require({s:report["quarantine"]["counts"][s]["retained"] for s in c.SPLITS}==COUNTS,
            "CHECKPOINT_RETENTION")
    mapping=private["retained_mapping"]
    c.freeze_preparation(mapping,MAPPING,dict.fromkeys(c.GATES,False))
    require(Counter(x[2] for x in mapping)=={s:COUNTS[s] for s in c.SPLITS[:3]},
            "CHECKPOINT_MAPPING_COUNTS")
    records=private["records"]
    c.validate(records,{p:set() for p in c.PAIRS})
    require({s:len(records[s]) for s in c.SPLITS}==ORIGINAL,"CHECKPOINT_RECORD_COUNTS")
    queue=private["review_queue"]
    require(type(queue) is list and len(queue)==771,"QUEUE_COUNT")
    rows={r["key"]:(s,r) for s in c.SPLITS for r in records[s]}
    seen=set(); observed=Counter()
    for item in queue:
        a,b=item["stratum"];pair=item["pair"]
        require((a,b) in c.PAIRS and len(pair)==2 and pair not in seen and
                all(k in rows for k in pair) and (rows[pair[0]][0],rows[pair[1]][0])==(a,b)
                and type(item["flagged"]) is bool,"QUEUE_MEMBERSHIP")
        seen.add(pair);observed[a,b,item["flagged"]]+=1
    for a,b in c.PAIRS:
        require(observed[a,b,True]==prior.EXPECTED_UNION[a+"__"+b] and
                observed[a,b,False]==100,"QUEUE_STRATA")
    require(all(tuple(x[:2]) in rows and rows[tuple(x[:2])][0]==x[2] for x in mapping),
            "MAPPING_MEMBERSHIP")
    answers=private["candidate_answers"]
    require(set(answers)=={k for k,(s,r) in rows.items() if s!="official_holdout"},
            "ANSWER_COVERAGE")
    return records,queue,mapping,answers

def prepare_review(report,private,nonce,*,approved_plan_digest):
    """Use only after independent authorization and prospective plan sign-off.

    Plan is bound to a digest; caller must authenticate the signed approval
    separately. This method does not establish institutional permission.
    """
    require(approved_plan_digest==plan_digest(),"PLAN_MISMATCH")
    records,queue,mapping,answers=check_checkpoint(report,private)
    retained={s:[] for s in c.SPLITS}
    selected={tuple(x[:2]) for x in mapping}
    for s in c.SPLITS:
        retained[s]=[r for r in records[s] if s=="official_holdout" or r["key"] in selected]
    strata=Counter((s,r["config"],r["category"],r["difficulty"])
                   for s in c.SPLITS[:3] for r in retained[s])
    allocation={g:min(n,CLINICAL_CAP) for g,n in strata.items()}
    semantic,sem_key=c.semantic_packets(records,queue,nonce)
    clinical,clin_key,clin_strata=c.clinical_review_plan(
        retained,{k:answers[k] for k in selected},allocation,nonce)
    return {"semantic_packets":semantic,"clinical_packets":clinical,
            "custodian":{"semantic_key":sem_key,"clinical_key":clin_key,
                         "clinical_strata":clin_strata,"records":records,"queue":queue,
                         "mapping":mapping}}

def plan_digest():
    return c.digest({"version":1,"checkpoint":CHECKPOINT,"mapping":MAPPING,
        "clinical_sample":"retained_parent_SRSWOR_within_split_config_category_difficulty",
        "cap_per_stratum":CLINICAL_CAP,"seed":c.SEED+2000,
        "semantic_sample":"all_171_flags_plus_100_uniform_unflagged_per_six_strata",
        "endpoints":["strict_role_concordance","confirmed_role_discordance","uncertain_parent",
                     "potentially_major_impact","both_roles_label_agree"],
        "interval":"exact_hypergeometric_Bonferroni_within_endpoint_95_percent",
        "reviewers":2,"adjudicator":1,"no_test_tuning":True})

def _tail(N,K,n,x,upper):
    """Exact rational hypergeometric tail, n small; no iid assumption."""
    lo=max(0,n-(N-K));hi=min(n,K)
    lo=max(lo,x) if upper else lo
    hi=hi if upper else min(hi,x)
    return Fraction(sum(comb(K,i)*comb(N-K,n-i) for i in range(lo,hi+1)),comb(N,n))

def interval(N,n,x,alpha=Fraction(1,20)):
    """Invert both exact hypergeometric tails, finite fixed population.

    Returns integer bounds on total positive population, not iid binomial CI.
    """
    require(all(type(v) is int for v in (N,n,x)) and 0<n<=N and 0<=x<=n and
            isinstance(alpha,Fraction) and 0<alpha<1,"INTERVAL_INPUT")
    if n==N: return x,x
    threshold=alpha/2
    lo,hi=x,N-n+x
    while lo<hi:
        mid=(lo+hi)//2
        if _tail(N,mid,n,x,True)>=threshold: hi=mid
        else: lo=mid+1
    lower=lo
    lo,hi=x,N-n+x
    while lo<hi:
        mid=(lo+hi+1)//2
        if _tail(N,mid,n,x,False)>=threshold: lo=mid
        else: hi=mid-1
    return lower,lo

def stratified_binary(cells):
    """Population-weighted rate and conservative simultaneous-strata 95% CI.

    Each cell: N,n,x for simple random sampling without replacement.
    Dependence of fixed outcomes across cells is not an iid assumption.
    Coverage is per endpoint, not simultaneous across multiple endpoints.
    """
    require(type(cells) is list and bool(cells),"EMPTY_CELLS")
    total=0;estimate=Fraction(0);lower=upper=0
    for N,n,x in cells:
        l,u=interval(N,n,x,Fraction(1,20*len(cells)))
        total+=N;estimate+=Fraction(N*x,n);lower+=l;upper+=u
    return {"population":total,"sample":sum(z[1] for z in cells),
            "estimate":float(estimate/total),"ci95":[lower/total,upper/total]}

def _judgments(expected,values,clinical):
    require(type(values) is dict and set(values)==expected,"REVIEW_COVERAGE")
    vocab=c.CLINICAL if clinical else c.RUBRIC
    for v in values.values():
        fields={"label","rationale","evidence"}|({"impact"} if clinical else set())
        require(type(v) is dict and set(v)==fields and v["label"] in vocab and
                all(type(v[k]) is str and 0<len(v[k].strip())<=10000
                    for k in ("rationale","evidence")),"REVIEW_SCHEMA")
        if clinical: require(v["impact"] in IMPACT,"IMPACT_SCHEMA")

def resolve(expected,first,second,adjudicated,*,clinical=False):
    """Review IDs and rationale stay private. Authentication is external.
    Clinical disagreement in label OR impact requires third adjudication.
    """
    require(type(expected) is set and bool(expected),"REVIEW_EXPECTED")
    for data in (first,second): _judgments(expected,data,clinical)
    unresolved=CLINICAL_UNCERTAIN if clinical else {"unresolved"}
    needed={k for k in expected if first[k]["label"]!=second[k]["label"] or
            first[k]["label"] in unresolved or (clinical and
            (first[k]["impact"]!=second[k]["impact"] or first[k]["impact"]=="uncertain"))}
    _judgments(needed,adjudicated,clinical)
    final={k:adjudicated.get(k,first[k]) for k in expected}
    a={k:v["label"] for k,v in first.items()};b={k:v["label"] for k,v in second.items()}
    label_needed={k for k in expected if a[k]!=b[k] or a[k] in unresolved}
    summary=c.reviewer_summary(expected,a,b,{k:final[k]["label"] for k in label_needed},
                               clinical=clinical)
    summary["adjudication_required"]=len(needed)
    summary["unresolved"]=sum(v["label"] in unresolved for v in final.values())
    summary["uncertain_impact"]=sum(v.get("impact")=="uncertain" for v in final.values())
    return final,summary

def semantic_analysis(queue,key,final,*,first,second):
    """All flagged pairs are a census of reconstructed flags, not original IDs.
    For unflagged sample, unresolved is included in conservative positive bound.
    """
    require(set(key)==set(final)==set(first)==set(second),"REVIEW_COVERAGE")
    by_pair={x["pair"]:x for x in queue}
    require(len(by_pair)==len(queue)==len(key) and set(key.values())==set(by_pair),
            "SEMANTIC_KEY_MISMATCH")
    cells=defaultdict(Counter); flagged=Counter(); agreements=Counter(); flagged_agree=0
    for token,pair in key.items():
        item=by_pair[pair];label=final[token]["label"]
        require(label in c.RUBRIC,"REVIEW_LABEL")
        agree=int(first[token]["label"]==second[token]["label"])
        if item["flagged"]:
            flagged[label]+=1;flagged_agree+=agree
        else:
            g=tuple(item["stratum"]);cells[g][label]+=1;agreements[g]+=agree
    require(sum(flagged.values())==171 and set(cells)==set(c.PAIRS),"SEMANTIC_COUNTS")
    summaries={}
    for a,b in c.PAIRS:
        counts=cells[a,b];n=sum(counts.values())
        N=ORIGINAL[a]*ORIGINAL[b]-prior.EXPECTED_UNION[a+"__"+b]
        require(n==100,"SEMANTIC_SAMPLE")
        x=sum(counts[k] for k in SEMANTIC_POSITIVE)
        u=counts["unresolved"]
        # Simultaneous across six strata; upper bound for possible positives.
        _,upper=interval(N,n,x+u,Fraction(1,120))
        al,au=interval(N,n,agreements[a,b],Fraction(1,120))
        summaries[a+"__"+b]={"population":N,"sample":n,"confirmed_positive":x,
                            "unresolved":u,"possible_positive_ci95_upper":upper/N,
                            "label_agreement":agreements[a,b]/n,"label_agreement_ci95":[al/N,au/N]}
    return {"flagged_dispositions":{k:flagged[k] for k in sorted(c.RUBRIC)},
            "flagged_label_agreement":flagged_agree/171,
            "unflagged_strata":summaries,"complete_semantic_leakage_excluded":False,
            "requires_missed_pair_remediation":any(z["confirmed_positive"] or z["unresolved"]
                                                 for z in summaries.values())}

def clinical_analysis(custodian,final,*,first,second):
    """Private decode only after both reviews are locked. Pair is analysis unit.
    Test-level summaries are custodian-only; no tuning or public small-cell exports.
    """
    key=custodian["clinical_key"];strata=custodian["clinical_strata"]
    require(set(key)==set(final),"REVIEW_COVERAGE")
    rows={r["key"]:(s,r) for s in c.SPLITS[:3] for r in custodian["records"][s]}
    retained={tuple(x[:2]) for x in custodian["mapping"]}
    require(retained<=set(rows),"RETAINED_KEY")
    population=Counter((rows[k][0],rows[k][1]["config"],rows[k][1]["category"],
                        rows[k][1]["difficulty"]) for k in retained)
    require(set(population)==set(strata) and all(strata[g]["population"]==n
            for g,n in population.items()),"CLINICAL_POPULATION")
    paired=defaultdict(dict)
    for token,(k,role) in key.items():
        require(k in retained and type(role) is int and role in (0,1) and
                role not in paired[k],"CLINICAL_KEY")
        label=final[token]["label"]
        require(label in c.CLINICAL,"REVIEW_LABEL")
        paired[k][role]=token
    cells=defaultdict(Counter)
    for k,roles in paired.items():
        require(set(roles)=={0,1},"PAIRED_COVERAGE")
        s,r=rows[k];g=(s,r["config"],r["category"],r["difficulty"])
        a,b=(final[roles[i]]["label"] for i in (0,1))
        cells[g]["n"]+=1
        cells[g]["strict_role_concordance"]+=int(a=="supported" and b=="contradicted")
        cells[g]["confirmed_role_discordance"]+=int(a=="contradicted" or b=="supported")
        cells[g]["uncertain_parent"]+=int(a in CLINICAL_UNCERTAIN or b in CLINICAL_UNCERTAIN)
        cells[g]["potentially_major_impact"]+=int(any(final[t]["impact"]=="potentially_major"
                                                    for t in roles.values()))
        cells[g]["both_roles_label_agree"]+=int(all(first[t]["label"]==second[t]["label"]
                                                   for t in roles.values()))
    require(set(cells)==set(strata),"CLINICAL_STRATA")
    for g,z in strata.items():
        require(cells[g]["n"]==z["sample"]==min(z["population"],CLINICAL_CAP),"CLINICAL_COUNTS")
    endpoints=("strict_role_concordance","confirmed_role_discordance","uncertain_parent",
               "potentially_major_impact","both_roles_label_agree")
    return {s:{e:stratified_binary([(z["population"],z["sample"],cells[g][e])
                   for g,z in strata.items() if g[0]==s]) for e in endpoints}
            for s in c.SPLITS[:3]}
