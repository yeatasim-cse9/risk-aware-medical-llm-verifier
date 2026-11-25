"""MASTER 01C strict public aggregate validator; no private records accepted."""
import math
import re
import e1_master01c_core as c
import e1_master01c_private as p
import e1_master01_validation_kaggle as b
import e1_master01b_kaggle as prior

require=c.require
CONFIGS=("pqa_artificial","pqa_labeled")
KNOWN_GPU={"Tesla T4","Tesla P100-PCIE-16GB","Tesla P100-SXM2-16GB","NVIDIA T4",
           "NVIDIA L4","NVIDIA A100-SXM4-40GB","NVIDIA A100 80GB PCIe",
           "NVIDIA H100 80GB HBM3","NVIDIA GeForce RTX 4090"}
FALSE_FLAGS=("exact_original_pair_membership_verified","human_review_completed",
             "clinical_labels_approved","full_article_provenance_certified",
             "complete_semantic_leakage_excluded","private_rows_exported",
             "private_review_queue_persisted")
MODULES=("e1_master01_validation_kaggle","e1_master01b_kaggle",
         "e1_master01c_core","e1_master01c_private")

def keys(d,expected):
    require(type(d) is dict and set(d)==set(expected),"AGGREGATE_SCHEMA")

def integer(v,lo,hi):
    return type(v) is int and lo<=v<=hi

def hexhash(x):
    return type(x) is str and re.fullmatch(r"[0-9a-f]{64}",x) is not None

def dist(d,n,m,limit):
    keys(d,("before_sorted","after_sorted","total_variation"))
    for k,total in (("before_sorted",n),("after_sorted",m)):
        a=d[k]
        require(type(a) is list and len(a)<=limit and all(integer(v,1,total) for v in a)
                and sum(a)==total and a==sorted(a,reverse=True),"BALANCE_COUNTS")
    tv=d["total_variation"]
    require((n*m==0 and tv is None) or (n*m>0 and type(tv) in (int,float)
        and math.isfinite(tv) and 0<=tv<=1),"BALANCE_TV")

def validate(d,commit,hashes):
    """Validate exact schema and known bounds, not independently authenticate data."""
    keys(d,("schema_version","status","previous_evidence_sha256","previous_mapping_sha256",
            "seed","encoder_aggregate_sha256","maxima","strata","review_queue_count",
            "quarantine","balance","versions","device","gates","package_source_commit",
            "package_hashes","integrity_sha256",*FALSE_FLAGS))
    raw=d.copy();h=raw.pop("integrity_sha256")
    require(hexhash(h) and c.digest(raw)==h,"REPORT_INTEGRITY")
    require(type(commit) is str and re.fullmatch("[0-9a-f]{40}",commit) and
            d["package_source_commit"]==commit,"SOURCE_COMMIT")
    keys(hashes,MODULES)
    require(all(hexhash(v) for v in hashes.values()) and d["package_hashes"]==hashes and
            hashes[MODULES[0]]==p.BASE_SHA and hashes[MODULES[1]]==p.PRIOR_SHA,"PACKAGE_PIN")
    require(type(d["schema_version"]) is int and d["schema_version"]==1 and
            d["status"]=="PROVISIONAL_REMEDIATION_REVIEW_REQUIRED" and
            d["previous_evidence_sha256"]==p.PRIOR_EVIDENCE and
            d["previous_mapping_sha256"]==b.EXPECTED_MAPPING and d["seed"]==c.SEED and
            d["encoder_aggregate_sha256"]==p.EXPECTED_ENCODERS,"EVIDENCE_PIN")
    require(all(d[k] is False for k in FALSE_FLAGS),"FALSE_APPROVAL_OR_PRIVACY")
    p.compare_maxima(d["maxima"])
    keys(d["versions"],p.OLD_VERSIONS)
    require(all(type(v) is str and re.fullmatch(r"[0-9][0-9A-Za-z.+_-]{0,39}",v)
                for v in d["versions"].values()),"VERSION_FORMAT")
    require(type(d["device"]) is str and d["device"] in KNOWN_GPU,"UNKNOWN_GPU")
    keys(d["strata"],p.EXPECTED_UNION)
    total=0
    for a,z in c.PAIRS:
        name=a+"__"+z; x=d["strata"][name]
        keys(x,("flagged_population","unflagged_population","unflagged_review_sample"))
        n=prior.SIZES[a]*prior.SIZES[z];f=p.EXPECTED_UNION[name]
        require(all(type(v) is int for v in x.values()) and
                x=={"flagged_population":f,"unflagged_population":n-f,
                    "unflagged_review_sample":min(100,n-f)},"STRATUM_COUNTS")
        total+=f+x["unflagged_review_sample"]
    require(type(d["review_queue_count"]) is int and d["review_queue_count"]==total,"QUEUE_COUNT")
    q=d["quarantine"]
    keys(q,("counts","mapping_sha256","residual_flagged_pairs","source_proxy_conflicts"))
    require(hexhash(q["mapping_sha256"]) and q["mapping_sha256"]!=b.EXPECTED_MAPPING and
            all(type(q[k]) is int and q[k]==0 for k in ("residual_flagged_pairs","source_proxy_conflicts")),
            "QUARANTINE_INTEGRITY")
    keys(q["counts"],c.SPLITS)
    for s,n in prior.SIZES.items():
        x=q["counts"][s];f=p.EXPECTED_ENDPOINTS[s]
        keys(x,("original","flagged_endpoints","retained","quarantined","paired_examples"))
        require(all(type(v) is int for v in x.values()) and x["original"]==n and
                x["flagged_endpoints"]==f and x["retained"]+x["quarantined"]==n,"RETENTION_COUNTS")
        if s=="official_holdout":
            require(x["retained"]==n and x["quarantined"]==0 and x["paired_examples"]==0,
                    "HOLDOUT_CHANGED")
        else:
            require(integer(x["retained"],1,n-f) and x["paired_examples"]==2*x["retained"],
                    "PAIRED_RETENTION")
    keys(d["balance"],c.SPLITS[:3])
    expected={"pqa_artificial":{"train":7185,"dev":890,"test":899},
              "pqa_labeled":{"train":394,"dev":57,"test":49}}
    for s,x in d["balance"].items():
        keys(x,("config","category","difficulty","by_config"))
        n,m=q["counts"][s]["original"],q["counts"][s]["retained"]
        for field,limit in (("config",2),("category",4),("difficulty",3)):
            dist(x[field],n,m,limit)
        keys(x["by_config"],CONFIGS)
        retained=[];before=[]
        for cfg in CONFIGS:
            z=x["by_config"][cfg]
            keys(z,("category","difficulty"))
            nc=expected[cfg][s]
            a=z["category"]
            keys(a,("before_sorted","after_sorted","total_variation"))
            require(type(a["after_sorted"]) is list and len(a["after_sorted"])<=4 and
                    all(integer(v,1,nc) for v in a["after_sorted"]),"CONFIG_COUNTS")
            mc=sum(a["after_sorted"])
            require(0<=mc<=nc,"CONFIG_COUNTS")
            dist(a,nc,mc,4);dist(z["difficulty"],nc,mc,3)
            before.append(nc);retained.append(mc)
        require(sum(retained)==m and x["config"]["before_sorted"]==sorted(before,reverse=True)
                and x["config"]["after_sorted"]==sorted((v for v in retained if v),reverse=True),
                "CONFIG_RECONCILIATION")
        expected_tv=round(sum(abs(a/n-z/m) for a,z in zip(before,retained))/2,8)
        require(abs(x["config"]["total_variation"]-expected_tv)<1e-7,"CONFIG_SHIFT")
    g=d["gates"]
    keys(g,("missing_gates","external_authentication_required","final_split_approved",
            "final_split_frozen","scientific_e1_experiment_executed"))
    require(g["missing_gates"]==list(c.GATES) and g["external_authentication_required"] is True
            and all(g[k] is False for k in ("final_split_approved","final_split_frozen",
                                            "scientific_e1_experiment_executed")),"FALSE_FREEZE")
    return d
