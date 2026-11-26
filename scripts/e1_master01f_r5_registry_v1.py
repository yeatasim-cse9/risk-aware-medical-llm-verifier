"""R5 private CPU registry development-only preparation. No model or GPU.
No public CI data acquisition. Use a private no-accelerator Kaggle session.
"""
import argparse
from datetime import date
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
from urllib.parse import urlencode
from urllib.request import Request, build_opener, ProxyHandler, HTTPRedirectHandler

BASE = "https://clinicaltrials.gov/api/v2"
START, END = "2026-10-01", "2026-10-07"
FRAME = f"AREA[StudyFirstPostDate]RANGE[{START},{END}]"
ID_FIELDS = "NCTId,NCTIdAlias,OrgStudyId,SecondaryId,StudyFirstPostDate"
FACT_FIELDS = "NCTId,StudyFirstPostDate,LastUpdatePostDate,StudyType,OverallStatus,EnrollmentCount,EnrollmentType"
NCT = re.compile(r"NCT[0-9]{8}")
TYPES = {"INTERVENTIONAL","OBSERVATIONAL","EXPANDED_ACCESS"}
STATUSES = {"ACTIVE_NOT_RECRUITING","COMPLETED","ENROLLING_BY_INVITATION",
            "NOT_YET_RECRUITING","RECRUITING","SUSPENDED","TERMINATED",
            "WITHDRAWN","AVAILABLE","NO_LONGER_AVAILABLE","TEMPORARILY_NOT_AVAILABLE",
            "APPROVED_FOR_MARKETING","WITHHELD","UNKNOWN"}
TARGET = 128
MAX_BYTES = 4_000_000

def require(ok, code):
    if not ok:
        raise ValueError(code)

def strict_object(pairs):
    out = {}
    for k,v in pairs:
        require(k not in out,"DUPLICATE_JSON_KEY")
        out[k] = v
    return out

def canonical(x):
    return json.dumps(x,sort_keys=True,separators=(",",":"),allow_nan=False).encode()

def sha(x):
    return hashlib.sha256(canonical(x)).hexdigest()

class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self,*args,**kwargs):
        raise ValueError("REDIRECT_BLOCKED")

def get(endpoint, params=None):
    require(endpoint in ("/version","/studies"),"ENDPOINT")
    if endpoint == "/version":
        require(params is None,"VERSION_SCOPE")
    else:
        require(type(params) is dict and params.get("format")=="json","QUERY_SCOPE")
        if params.get("fields")==ID_FIELDS:
            require(set(params) <= {"format","fields","filter.advanced","pageSize","pageToken"}
                    and params.get("filter.advanced")==FRAME
                    and params.get("pageSize")==500,"FRAME_SCOPE")
        else:
            require(set(params)=={"format","fields","filter.ids","pageSize"}
                    and params["fields"]==FACT_FIELDS
                    and type(params["filter.ids"]) is str
                    and 1<=len(params["filter.ids"].split(","))<=16
                    and all(NCT.fullmatch(x) for x in params["filter.ids"].split(","))
                    and params["pageSize"]==16,"FACT_QUERY_SCOPE")
    url=BASE+endpoint+("?" + urlencode(params) if params else "")
    with build_opener(ProxyHandler({}),NoRedirect()).open(
            Request(url,headers={"Accept":"application/json","Accept-Encoding":"identity"}),timeout=40) as r:
        require(r.status==200 and r.headers.get("Content-Encoding") in (None,"identity"),"HTTP")
        raw=r.read(MAX_BYTES+1)
    require(0<len(raw)<=MAX_BYTES,"RESPONSE_SIZE")
    return json.loads(raw,object_pairs_hook=strict_object,
                      parse_constant=lambda _: (_ for _ in ()).throw(ValueError("JSON_CONSTANT")))

def version():
    x=get("/version")
    require(type(x) is dict and set(x)=={"apiVersion","dataTimestamp"}
            and all(type(v) is str and 0<len(v)<100 for v in x.values()),"VERSION_SCHEMA")
    return x

def obj(x, allowed, required=()):
    require(type(x) is dict and set(required)<=set(x)<=set(allowed),"PROJECTION_SCHEMA")
    return x

def posted(x):
    d=obj(x,{"date","type"},{"date"})["date"]
    require(type(d) is str and len(d)==10 and date.fromisoformat(d).isoformat()==d,
            "POSTED_DATE")
    return d

def identifier(x):
    require(type(x) is str and NCT.fullmatch(x),"NCT_ID")
    return x

def link(x):
    require(type(x) is str and 0<len(x.strip())<=300,"LINK_ID")
    return x.strip().casefold()

def parse_id(study):
    p=obj(obj(study,{"protocolSection"},{"protocolSection"})["protocolSection"],
          {"identificationModule","statusModule"},{"identificationModule","statusModule"})
    i=obj(p["identificationModule"],{"nctId","nctIdAliases","orgStudyIdInfo","secondaryIdInfos"},{"nctId"})
    s=obj(p["statusModule"],{"studyFirstPostDateStruct"},{"studyFirstPostDateStruct"})
    rid=identifier(i["nctId"])
    day=posted(s["studyFirstPostDateStruct"])
    require(START<=day<=END,"OUT_OF_FRAME")
    aliases=i.get("nctIdAliases",[])
    secondary=i.get("secondaryIdInfos",[])
    require(type(aliases) is list and type(secondary) is list,"LINK_SCHEMA")
    links={"nct:"+rid,*("nct:"+identifier(a) for a in aliases)}
    if "orgStudyIdInfo" in i:
        links.add("other:"+link(obj(i["orgStudyIdInfo"],{"id"},{"id"})["id"]))
    for a in secondary:
        links.add("other:"+link(obj(a,{"id"},{"id"})["id"]))
    return {"id":rid,"first_post":day,"links":sorted(links)}

def page(x):
    obj(x,{"studies","nextPageToken","totalCount"},{"studies"})
    require(type(x["studies"]) is list,"PAGE_SCHEMA")
    token=x.get("nextPageToken")
    require(token is None or type(token) is str and 0<len(token)<=4000,"PAGE_TOKEN")
    return x["studies"],token

def frame_ids():
    rows=[];token=None;seen=set()
    for _ in range(20):
        q={"format":"json","fields":ID_FIELDS,"filter.advanced":FRAME,"pageSize":500}
        if token is not None:
            q["pageToken"]=token
        batch,token=page(get("/studies",q))
        rows.extend(parse_id(x) for x in batch)
        if token is None:
            return rows
        require(token not in seen,"PAGINATION_LOOP")
        seen.add(token)
    raise ValueError("FRAME_PAGE_LIMIT")

def choose(rows,target=TARGET):
    # All identifiers in the frame are development-exposed, including unselected.
    by={r["id"]:r for r in rows}
    require(len(by)==len(rows) and len(by)<=10000,"DUPLICATE_FRAME_ID")
    parent={k:k for k in by}; tokens={}
    def root(k):
        while parent[k]!=k:
            parent[k]=parent[parent[k]]
            k=parent[k]
        return k
    for rid,r in sorted(by.items()):
        for t in r["links"]:
            if t in tokens:
                a,b=root(rid),root(tokens[t])
                parent[max(a,b)]=min(a,b)
            else:
                tokens[t]=rid
    groups={}
    for k in sorted(by):
        groups.setdefault(root(k),[]).append(k)
    rank=lambda x: hashlib.sha256(("r5-20261126:"+x).encode()).hexdigest()
    reps=[min(v,key=lambda x:(rank(x),x)) for v in groups.values()]
    require(len(reps)>=target,"INSUFFICIENT_PROXY_GROUPS")
    return sorted(reps,key=lambda x:(rank(x),x))[:target],len(groups)

def parse_fact(study, expected):
    p=obj(obj(study,{"protocolSection"},{"protocolSection"})["protocolSection"],
          {"identificationModule","statusModule","designModule"},{"identificationModule","statusModule"})
    i=obj(p["identificationModule"],{"nctId"},{"nctId"})
    rid=identifier(i["nctId"])
    require(rid in expected,"UNSELECTED_ID")
    s=obj(p["statusModule"],{"overallStatus","studyFirstPostDateStruct","lastUpdatePostDateStruct"},{"studyFirstPostDateStruct"})
    require(posted(s["studyFirstPostDateStruct"])==expected[rid],"FRAME_DATE_DRIFT")
    d=obj(p.get("designModule",{}),{"studyType","enrollmentInfo"})
    out={"id":rid,"first_post":expected[rid],"last_update":posted(s["lastUpdatePostDateStruct"]) if "lastUpdatePostDateStruct" in s else None,
         "fields":{}}
    if d.get("studyType") in TYPES:
        out["fields"]["study_type"]=d["studyType"]
    if s.get("overallStatus") in STATUSES:
        out["fields"]["recruitment_status"]=s["overallStatus"]
    e=obj(d.get("enrollmentInfo",{}),{"count","type"})
    if type(e.get("count")) is int and e["count"]>=0 and e.get("type") in ("ACTUAL","ESTIMATED"):
        out["fields"]["enrollment"]={"count":e["count"],"type":e["type"]}
    return out

def fetch_selected(ids, frame):
    expected={r["id"]:r["first_post"] for r in frame}
    require(len(ids)==len(set(ids)) and set(ids)<=set(expected),"SELECTION_SCOPE")
    results=[]
    for j in range(0,len(ids),16):
        batch=ids[j:j+16]
        scoped={k:expected[k] for k in batch}
        studies,token=page(get("/studies",{"format":"json","fields":FACT_FIELDS,
                               "filter.ids":",".join(batch),"pageSize":16}))
        require(token is None,"FACT_PAGINATION")
        records=[parse_fact(x,scoped) for x in studies]
        require(sorted(r["id"] for r in records)==sorted(batch),"FACT_MEMBERSHIP")
        results.extend(records)
    return results

def score(response, reference, rid):
    # Strict typed scoring independent of any LLM/verifier. No fuzzy matching.
    if type(response) is not dict:
        return "invalid"
    if response=={"abstain":True} and type(response["abstain"]) is bool:
        return "abstain"
    if set(response)!={"nct_id","value"} or type(response["nct_id"]) is not str:
        return "invalid"
    value=response["value"]
    if type(reference) is str:
        if type(value) is not str:
            return "invalid"
    elif type(reference) is dict:
        if type(value) is not dict or set(value)!={"count","type"} or type(value["count"]) is not int or type(value["type"]) is not str:
            return "invalid"
    else:
        raise ValueError("REFERENCE_SCHEMA")
    return "correct" if response["nct_id"]==rid and value==reference else "wrong"

def prepare(out,*,authorized):
    require(authorized is True,"EXPLICIT_RIGHTS_AND_CPU_APPROVAL_REQUIRED")
    require(os.environ.get("CUDA_VISIBLE_DEVICES")=="","CPU_ONLY")
    temp=Path("/kaggle/temp").resolve()
    require(temp in out.parent.resolve().parents and out.parent.is_dir() and not out.parent.is_symlink()
            and out.parent.stat().st_uid==os.getuid() and out.parent.stat().st_mode&0o077==0
            and not out.exists() and not out.is_symlink(),"PRIVATE_EPHEMERAL_DESTINATION")
    v=version()
    rows=frame_ids()
    require(version()==v,"SOURCE_REFRESHED")
    ids,groups=choose(rows)
    records=fetch_selected(ids,rows)
    require(version()==v,"SOURCE_REFRESHED")
    q=[];refs=[];check=0
    for record in records:
        for field,value in sorted(record["fields"].items()):
            key=record["id"]+":"+field
            q.append({"key":key,"nct_id":record["id"],"field":field,
                      "as_of":v["dataTimestamp"]})
            refs.append({"key":key,"value":value})
            check+=score({"nct_id":record["id"],"value":value},value,record["id"])=="correct"
    require(len(q)>0 and check==len(q),"LOOKUP_SCORER_CHECK")
    report={"source":"ClinicalTrials.gov","source_processed":v["dataTimestamp"],
            "api_version":v["apiVersion"],"frame":[START,END],"frame_records":len(rows),
            "proxy_groups":groups,"selected_development_groups":len(ids),"eligible_questions":len(q),
            "lookup_scorer_self_check_passed":True,"llm_experiment_executed":False,
            "source_independence_certified":False,"calibration_or_test_accessed":False,
            "modifications":"field projection, deterministic proxy grouping, question/reference construction",
            "public_export_approved":False}
    tmp=Path(tempfile.mkdtemp(prefix=".r5-",dir=out.parent))
    try:
        for name,x in (("frame.json",rows),("selected.json",ids),("source.json",records),
                       ("questions.json",q),("references.json",refs),("report.json",report)):
            with (tmp/name).open("xb") as f:
                f.write(canonical(x))
            (tmp/name).chmod(0o600)
        manifest={name:sha(x) for name,x in (("frame.json",rows),("selected.json",ids),("source.json",records),
                  ("questions.json",q),("references.json",refs),("report.json",report))}
        (tmp/"manifest.json").write_bytes(canonical(manifest))
        (tmp/"manifest.json").chmod(0o600)
        require(not out.exists(),"DESTINATION_EXISTS")
        os.rename(tmp,out)
    finally:
        if tmp.exists():
            shutil.rmtree(tmp)
    return report

def main():
    a=argparse.ArgumentParser()
    sub=a.add_subparsers(dest="op",required=True)
    sub.add_parser("metadata")
    r=sub.add_parser("prepare")
    r.add_argument("--out",type=Path,required=True)
    r.add_argument("--confirmed-rights-and-private-cpu",action="store_true")
    x=a.parse_args()
    try:
        if x.op=="metadata":
            print("MASTER01FR5_PUBLIC_VERSION="+canonical(version()).decode())
        else:
            prepare(x.out,authorized=x.confirmed_rights_and_private_cpu)
            print("MASTER01FR5_PRIVATE_DEVELOPMENT_PREPARED_LOOKUP_CHECK_ONLY")
    except Exception:
        print("MASTER01FR5_BLOCKED")
        raise SystemExit(1) from None

if __name__=="__main__":
    main()
