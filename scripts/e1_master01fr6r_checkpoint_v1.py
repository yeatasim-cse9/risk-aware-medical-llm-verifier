"""Private, append-only aggregate stage checkpoints. No Kaggle submit or GPU code.
Only a reviewed deterministic stage recipe may call run(); no raw rows,
individual predictions, weights, optimizer state or credentials in records.
"""
import base64
import hashlib
import json
import math
import os
import re
import urllib.error
import urllib.parse
import urllib.request

REPO = "yeatasim-cse9/medical-llm-thesis-private-evidence"
REPO_ID = 1473079957
PREFIX = "checkpoints/master01f-r6r/"
API = "https://api.github.com/repos/" + REPO
MAX = 256_000
NAME = re.compile(r"[a-z][a-z0-9_]{0,39}")
RUN = re.compile(r"r6r-[a-z0-9][a-z0-9-]{0,63}")
HEX = re.compile(r"[0-9a-f]{64}")
COMMIT = re.compile(r"[0-9a-f]{40}")

def require(ok, code):
    if not ok:
        raise ValueError(code)

def canonical(x):
    return json.dumps(x, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode("ascii")

def digest(raw):
    return hashlib.sha256(raw).hexdigest()

def blob(raw):
    return hashlib.sha1(("blob %d" % len(raw)).encode()+b"\0"+raw).hexdigest()

def match(pattern, x):
    return type(x) is str and pattern.fullmatch(x) is not None

def validate_plan(plan, pin):
    require(type(plan) is dict and set(plan)=={
        "schema","run_id","scope","source_commit","input_sha256",
        "recipe_sha256","stages"},"PLAN_SCHEMA")
    require(type(plan["schema"]) is int and plan["schema"]==1
            and match(RUN,plan["run_id"])
            and plan["scope"] in ("synthetic","private_development_aggregate")
            and match(COMMIT,plan["source_commit"])
            and match(HEX,plan["input_sha256"]) and match(HEX,plan["recipe_sha256"])
            and match(HEX,pin) and digest(canonical(plan))==pin,"PLAN_BINDING")
    stages=plan["stages"]
    require(type(stages) is list and 1<=len(stages)<=64,"STAGES")
    names=[]
    for s in stages:
        require(type(s) is dict and set(s)=={"name","metrics"} and match(NAME,s["name"]),"STAGE")
        keys=s["metrics"]
        require(type(keys) is list and 1<=len(keys)<=16
                and all(match(NAME,k) for k in keys) and len(set(keys))==len(keys),"METRIC_KEYS")
        names.append(s["name"])
    require(len(set(names))==len(names),"DUPLICATE_STAGE")
    return plan

def metrics(x, keys):
    require(type(x) is dict and set(x)==set(keys),"AGGREGATE_SCHEMA")
    require(all(type(v) in (int,float) and abs(v)<=10**12 and math.isfinite(v)
                for v in x.values()),"AGGREGATE_VALUES")
    return x

class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self,*a,**kw):
        raise ValueError("REDIRECT_BLOCKED")

def request_json(url, headers, body=None, method="GET", missing=False):
    req=urllib.request.Request(url,headers=headers,data=body,method=method)
    opener=urllib.request.build_opener(urllib.request.ProxyHandler({}),NoRedirect())
    try:
        with opener.open(req,timeout=30) as res:
            require(res.status==(201 if method=="PUT" else 200),"HTTP_STATUS")
            raw=res.read(MAX*2+1)
        require(len(raw)<=MAX*2,"HTTP_SIZE")
        return json.loads(raw)
    except urllib.error.HTTPError as exc:
        code=exc.code
        exc.close()
        if missing and code==404:
            return None
        raise ValueError("HTTP_"+str(code)) from None
    except (OSError,ValueError,TypeError):
        raise ValueError("REMOTE_REQUEST_FAILED_RECONCILE") from None

class PrivateStore:
    def __init__(self, token):
        require(type(token) is str and 10<=len(token)<=512 and token.isascii()
                and not any(c.isspace() for c in token),"ARCHIVE_SECRET_REQUIRED")
        self.headers={"Authorization":"Bearer "+token,"Accept":"application/vnd.github+json",
                      "X-GitHub-Api-Version":"2022-11-28","Content-Type":"application/json",
                      "Cache-Control":"no-cache","User-Agent":"medical-llm-r6r"}
    def verify(self):
        info=request_json(API,self.headers)
        require(type(info) is dict and info.get("id")==REPO_ID
                and info.get("full_name")==REPO and info.get("private") is True
                and info.get("visibility")=="private" and info.get("default_branch")=="main"
                and info.get("archived") is False,"PRIVATE_REPOSITORY_REQUIRED")
    def url(self,path):
        require(type(path) is str and re.fullmatch(re.escape(PREFIX)+r"r6r-[a-z0-9][a-z0-9-]{0,63}/(?:plan|complete|stage-[0-9]{3})[.]json",path),"ARCHIVE_PATH")
        return API+"/contents/"+path
    def read(self,path):
        self.verify()
        obj=request_json(self.url(path)+"?ref=main",self.headers,missing=True)
        if obj is None:
            return None
        require(type(obj) is dict and obj.get("type")=="file"
                and obj.get("path")==path and obj.get("encoding")=="base64"
                and type(obj.get("size")) is int and 0<obj["size"]<=MAX
                and type(obj.get("content")) is str,"REMOTE_FILE_SCHEMA")
        raw=base64.b64decode("".join(obj["content"].splitlines()),validate=True)
        require(len(raw)==obj["size"] and blob(raw)==obj.get("sha"),"REMOTE_FILE_INTEGRITY")
        return raw
    def create(self,path,raw):
        self.verify()
        require(type(raw) is bytes and 0<len(raw)<=MAX,"RECORD_SIZE")
        # Omit sha: create only. A conflict or ambiguous response stops the run.
        payload=canonical({"message":"Archive reviewed aggregate checkpoint",
                           "content":base64.b64encode(raw).decode("ascii"),"branch":"main"})
        obj=request_json(self.url(path),self.headers,payload,"PUT")
        require(type(obj) is dict and type(obj.get("content")) is dict
                and obj["content"].get("sha")==blob(raw),"WRITE_RESPONSE")
        require(self.read(path)==raw,"WRITE_READBACK")

STATUSES={"started","resumed","checkpoint_saved","paused","complete","already_complete","failed"}

def notify(status,count,total):
    require(status in STATUSES and type(count) is int and type(total) is int
            and 0<=count<=total<=64,"NOTIFICATION_SCOPE")
    print("MASTER01FR6R "+status+" "+str(count)+"/"+str(total),flush=True)
    token=os.environ.get("TELEGRAM_BOT_TOKEN","")
    chat=os.environ.get("TELEGRAM_CHAT_ID","")
    if not token or not chat:
        print("TELEGRAM_NOT_CONFIGURED",flush=True)
        return
    try:
        require(re.fullmatch(r"[0-9]+:[A-Za-z0-9_-]+",token) and
                re.fullmatch(r"-?[0-9]+",chat),"TELEGRAM_SECRET_FORMAT")
        data=urllib.parse.urlencode({"chat_id":chat,"text":
            "MASTER01FR6R "+status+" "+str(count)+"/"+str(total)}).encode()
        req=urllib.request.Request("https://api.telegram.org/bot"+token+"/sendMessage",data=data)
        opener=urllib.request.build_opener(urllib.request.ProxyHandler({}),NoRedirect())
        with opener.open(req,timeout=10) as res:
            raw=res.read(32769)
            require(res.status==200 and len(raw)<=32768 and json.loads(raw).get("ok") is True,"TELEGRAM_RESPONSE")
        print("TELEGRAM_SENT",flush=True)
    except Exception:
        print("TELEGRAM_DELIVERY_UNCONFIRMED",flush=True)

def run(plan,pin,compute,store,*,authorized=False,max_new=None,emit=notify):
    """Resume complete independent stages, not arbitrary process/model memory.
    compute(stage_name, prior_aggregates) must be reviewed and deterministic.
    One operator at a time. No automatic retry/submit; no secrets in output.
    """
    require(authorized is True,"OPERATOR_AUTHORIZATION_REQUIRED")
    validate_plan(plan,pin)
    require(max_new is None or type(max_new) is int and 1<=max_new<=64,"STAGE_BUDGET")
    stages=plan["stages"]; total=len(stages); count=0
    prefix=PREFIX+plan["run_id"]+"/"
    plan_raw=canonical(plan)
    previous=pin
    records=[]
    values={}
    try:
        store.verify()
        old=store.read(prefix+"plan.json")
        existing=[store.read(prefix+("stage-%03d.json"%i)) for i in range(total)]
        final=store.read(prefix+"complete.json")
        if old is None:
            require(all(x is None for x in existing) and final is None,"ORPHAN_RECORDS")
            store.create(prefix+"plan.json",plan_raw)
        else:
            require(old==plan_raw,"PLAN_CONFLICT")
        gap=False
        for i,raw in enumerate(existing):
            if raw is None:
                gap=True
                continue
            require(not gap,"CHECKPOINT_GAP")
            obj=json.loads(raw)
            s=stages[i]
            require(type(obj) is dict and set(obj)=={"schema","plan_sha256","index","stage",
                    "previous_sha256","metrics"},"CHECKPOINT_SCHEMA")
            expected={"schema":1,"plan_sha256":pin,"index":i,"stage":s["name"],
                      "previous_sha256":previous,"metrics":metrics(obj["metrics"],s["metrics"])}
            require(raw==canonical(expected),"CHECKPOINT_BINDING")
            records.append(digest(raw)); previous=records[-1]
            values[s["name"]]=obj["metrics"]; count+=1
        def complete_bytes():
            return canonical({"schema":1,"plan_sha256":pin,"stage_sha256":records,
                              "aggregates":values,"public_export_approved":False,
                              "scientific_approval":False})
        if final is not None:
            require(count==total and final==complete_bytes(),"FINAL_REPORT_CONFLICT")
            emit("already_complete",count,total)
            return json.loads(final)
        emit("resumed" if count else "started",count,total)
        new=0
        while count<total:
            if max_new is not None and new>=max_new:
                emit("paused",count,total)
                return None
            s=stages[count]
            # Copy so the callback cannot mutate previously verified metrics.
            result=metrics(compute(s["name"],json.loads(canonical(values))),s["metrics"])
            raw=canonical({"schema":1,"plan_sha256":pin,"index":count,"stage":s["name"],
                           "previous_sha256":previous,"metrics":result})
            store.create(prefix+("stage-%03d.json"%count),raw)
            records.append(digest(raw)); previous=records[-1]
            values[s["name"]]=result; count+=1; new+=1
            emit("checkpoint_saved",count,total)
        raw=complete_bytes()
        store.create(prefix+"complete.json",raw)
        emit("complete",count,total)
        return json.loads(raw)
    except Exception:
        emit("failed",count,total)
        raise ValueError("R6R_STOPPED_RECONCILE_PRIVATE_CHECKPOINTS") from None

def synthetic_plan(run_id,source_commit):
    return {"schema":1,"run_id":run_id,"scope":"synthetic","source_commit":source_commit,
            "input_sha256":digest(b"synthetic integers 1 through 10"),
            "recipe_sha256":digest(b"r6r synthetic sum squares v1"),
            "stages":[{"name":"first_half","metrics":["sum_squares"]},
                      {"name":"second_half","metrics":["sum_squares"]},
                      {"name":"total","metrics":["sum_squares"]}]}

def synthetic_compute(stage,prior):
    if stage=="first_half":
        return {"sum_squares":sum(i*i for i in range(1,6))}
    if stage=="second_half":
        return {"sum_squares":sum(i*i for i in range(6,11))}
    require(stage=="total","SYNTHETIC_STAGE")
    return {"sum_squares":sum(v["sum_squares"] for v in prior.values())}
