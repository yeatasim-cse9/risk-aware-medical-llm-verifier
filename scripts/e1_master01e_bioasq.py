"""Offline BioASQ release inventory. No network, account login, or data loading.
Call only inside a separately authorized private runtime with pinned official
JSON bytes. Returns private metadata for the existing cohort planner; never
export it. Release attestations are necessary, not proof of legal permission.
"""
import hashlib,json,re
from urllib.parse import urlsplit
from e1_master01d_v2_cohort import plan,prospective_power
from e1_master01d_v2_integrity import require

def pmid(url):
    require(type(url) is str,"ARTICLE_URL")
    u=urlsplit(url)
    require(u.scheme in ("http","https") and not u.query and not u.fragment
        and u.username is None and u.password is None and u.port is None,"ARTICLE_URL")
    pattern=r"/([1-9][0-9]*)/?" if u.hostname=="pubmed.ncbi.nlm.nih.gov" else r"/pubmed/([1-9][0-9]*)/?"
    require(u.hostname in ("pubmed.ncbi.nlm.nih.gov","www.ncbi.nlm.nih.gov","ncbi.nlm.nih.gov"),"ARTICLE_HOST")
    m=re.fullmatch(pattern,u.path);require(m is not None,"ARTICLE_PATH")
    return "pmid:"+m.group(1)

def decode(raw,*,release_sha256,expected_count,split,authorized):
    require(authorized is True,"PRIVATE_AUTHORIZATION_REQUIRED")
    require(type(raw) is bytes and 0<len(raw)<=100_000_000 and
        type(release_sha256) is str and re.fullmatch("[0-9a-f]{64}",release_sha256)
        and hashlib.sha256(raw).hexdigest()==release_sha256,"RELEASE_PIN")
    require(type(expected_count) is int and 0<expected_count<=10000 and
        split in ("development","official_test"),"RELEASE_SCHEMA")
    def strict_object(pairs):
        out={}
        for k,v in pairs:
            require(k not in out,"DUPLICATE_JSON_KEY");out[k]=v
        return out
    obj=json.loads(raw,object_pairs_hook=strict_object)
    require(type(obj) is dict and type(obj.get("questions")) is list and
        len(obj["questions"])==expected_count,"QUESTION_COUNT")
    ids=set();counts=dict.fromkeys(("yesno","factoid","list","summary"),0);rows=[]
    for q in obj["questions"]:
        require(type(q) is dict and type(q.get("id")) is str and bool(q["id"].strip())
            and q["id"] not in ids and q.get("type") in counts,"QUESTION_SCHEMA")
        ids.add(q["id"]);counts[q["type"]]+=1
        if q["type"]!="yesno": continue
        require(type(q.get("body")) is str and bool(q["body"].strip()),"QUESTION_BODY")
        docs=q.get("documents",[]);snips=q.get("snippets",[])
        require(type(docs) is list and type(snips) is list,"EVIDENCE_SCHEMA")
        sources={pmid(u) for u in docs};parts=[]
        for x in snips:
            require(type(x) is dict and type(x.get("text")) is str and bool(x["text"].strip()),"SNIPPET_SCHEMA")
            sources.add(pmid(x.get("document")));parts.append(x["text"])
        # Do not retain exact or ideal answers. Reference availability only.
        exact=q.get("exact_answer")
        rows.append(dict(dataset="bioasq14b",release="14b",id=q["id"],task="yes_no",
            native_split=split,sources=tuple(sorted(sources)),question=q["body"],
            context=" ".join(parts),evidence_available=bool(parts),
            reference_available=type(exact) is str and exact in ("yes","no"),
            provenance_known=bool(sources)))
    return rows,counts

def inventory(dev_raw,*,dev_sha,dev_count,protected,authorized,
              test_raw=None,test_sha=None,test_count=None,near_links=()):
    require(authorized is True,"PRIVATE_AUTHORIZATION_REQUIRED")
    require(type(protected) is list and bool(protected) and
        all(x.get("native_split")=="protected" for x in protected),"PROTECTED_SCOPE")
    dev,counts=decode(dev_raw,release_sha256=dev_sha,expected_count=dev_count,
                      split="development",authorized=authorized)
    test=[];test_counts=None
    if test_raw is not None:
        test,test_counts=decode(test_raw,release_sha256=test_sha,expected_count=test_count,
                                split="official_test",authorized=authorized)
    else:
        require(test_sha is None and test_count is None,"TEST_PIN_WITHOUT_BYTES")
    require(bool(dev),"NO_YESNO_DEVELOPMENT")
    _,summary=plan(protected+dev+test,dataset="bioasq14b",release="14b",
                   task="yes_no",near_links=near_links)
    # Aggregate only. No candidate manifest digest or IDs.
    safe={k:summary[k] for k in ("candidate_rows","excluded_rows","eligible_group_representatives")}
    return {"development_types":counts,"test_types":test_counts,"inventory":safe,
            "power":prospective_power(safe["eligible_group_representatives"]["calibration"]),
            "official_test_inventory_present":test_raw is not None,
            "rights_verified":False,"independence_verified":False,
            "complete_semantic_screen_verified":False,"final_split_frozen":False,
            "execution_authorized":False}
