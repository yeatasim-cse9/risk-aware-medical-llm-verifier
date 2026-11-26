"""Public README only, no viewer, data files, or row endpoints."""
import hashlib
from urllib.request import Request, build_opener, ProxyHandler
import e1_master01f_r2_native_v1 as n
from e1_master01f_r3_audit_v1 import REV

URL=n.HF+"/datasets/"+n.p.CANDIDATE+"/raw/"+REV+"/README.md"

def main():
    try:
        opener=build_opener(ProxyHandler({}),n.NoRedirect())
        with opener.open(Request(URL,headers={"Accept":"text/plain"}),timeout=25) as r:
            n.require(r.status==200,"README_HTTP")
            raw=r.read(32769)
        n.require(0<len(raw)<=32768,"README_SIZE")
        text=raw.decode("utf-8")
        # Output fixed documentation indicators only; no arbitrary card contents.
        import re
        label=bool(re.search(r"label.{0,100}whether the statement is factual or not.{0,40}true or false",text,re.I))
        print("MASTER01FR3_README_SHA256="+hashlib.sha256(raw).hexdigest())
        print("MASTER01FR3_FACTUAL_BOOLEAN_DOCUMENTED="+str(label).upper())
        for s in ("sumpubmed","medmcqa","medqa","acm","mednli"):
            print("MASTER01FR3_CARD_MENTIONS_"+s.upper()+"="+str(s in text.casefold()).upper())
        print("MASTER01FR3_PINNED_README_OBSERVED_REVIEW_REQUIRED")
    except Exception:
        print("MASTER01FR3_PINNED_README_BLOCKED")

if __name__=="__main__":
    main()
