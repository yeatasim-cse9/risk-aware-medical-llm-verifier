"""R9: private development token-length audit, no neural inference or output I/O.
Inputs must be the previously reconstructed fit/selection groups. No labels
are read by the audit. Only bounded aggregate counts leave this function.
Candidate tilings are resource estimates, not approved inference recipes.
"""
from math import ceil

CANDIDATES = ((192,48,317,64),(192,48,256,64),(256,64,253,64))

def require(ok, code):
    if not ok:
        raise ValueError(code)

def windows(length, width, overlap):
    require(type(length) is int and length > 0 and type(width) is int
            and type(overlap) is int and 0 <= overlap < width, "R9_WINDOW_INPUT")
    return 1 + max(0, (length-width+width-overlap-1)//(width-overlap))

def summary(values):
    require(bool(values), "R9_EMPTY")
    x=sorted(values)
    return {"min":x[0],"p25":x[ceil(.25*len(x))-1],
            "p50":x[ceil(.50*len(x))-1],"p75":x[ceil(.75*len(x))-1],
            "p90":x[ceil(.90*len(x))-1],"p95":x[ceil(.95*len(x))-1],
            "p99":x[ceil(.99*len(x))-1],"max":x[-1],"sum":sum(x)}

def audit(fit,selection,tokenize,*,authorized=False,expected_sizes=(256,128)):
    require(authorized is True,"R9_AUDIT_AUTHORIZATION")
    require((len(fit),len(selection))==expected_sizes and all(n>1 for n in expected_sizes),
            "R9_COHORT_SIZE")
    def lengths(rows):
        out=[]
        for row in rows:
            require(type(row) in (tuple,list) and len(row)==3
                    and all(type(v) is str and bool(v.strip()) for v in row[:2]),
                    "R9_ROW_CONTRACT")
            pair=[]
            for text in row[:2]:
                ids=tokenize(text)
                require(type(ids) is list and bool(ids) and
                        all(type(i) is int and i>=0 for i in ids),"R9_TOKEN_CONTRACT")
                pair.append(len(ids))
            out.append(tuple(pair))
        return out
    f=lengths(fit);s=lengths(selection)
    def partition(rows):
        st=[a for a,b in rows];ct=[b for a,b in rows]
        sw=[a>317 for a in st];cw=[windows(b,192,48)>16 for b in ct]
        return {"groups":len(rows),"statement_tokens":summary(st),"context_tokens":summary(ct),
                "context_windows_192_48":summary([windows(b,192,48) for b in ct]),
                "statement_over":sum(sw),"context_over":sum(cw),
                "both_over":sum(a and b for a,b in zip(sw,cw)),
                "either_over":sum(a or b for a,b in zip(sw,cw)),
                "within":sum(not(a or b) for a,b in zip(sw,cw)),
                "legacy_max_context_windows":max([windows(b,192,48) for a,b in rows
                    if a<=317 and windows(b,192,48)<=16],default=0)}
    estimates=[]
    for c,co,h,ho in CANDIDATES:
        require(c+h+3<=512,"R9_PAIR_LIMIT")
        def costs(rows):
            return [windows(a,h,ho)*windows(b,c,co) for a,b in rows]
        fc,sc=costs(f),costs(s)
        rotated=[(a,s[(i+1)%len(s)][1]) for i,(a,b) in enumerate(s)]
        pc=costs(rotated)
        # Removed uses one empty context per statement window, not fixed dummy slots.
        rc=[windows(a,h,ho) for a,b in s]
        estimates.append({"context_width":c,"context_overlap":co,"statement_width":h,
            "statement_overlap":ho,"fit_full_pairs":summary(fc),
            "selection_full_pairs":summary(sc),"selection_permuted_pairs":summary(pc),
            "selection_removed_pairs":summary(rc),
            "total_pairs":sum(fc)+sum(sc)+sum(pc)+sum(rc)})
    return {"schema":1,"scope":"private_development_token_audit",
            "fit":partition(f),"selection":partition(s),"all":partition(f+s),
            "candidate_costs":estimates,"neural_inference":False,
            "scientific_approval":False,"public_export_approved":False}
