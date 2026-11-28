"""Experimental sentence-level evidence localization, without I/O or fitting.
Use the pinned R9 tiler/encoder/forward through explicit dependency injection.
Scores are alignment proxies; evidence windows are not clinical truth.
"""
import re
from math import isfinite

ABBREVIATIONS=frozenset(("dr.","mr.","mrs.","ms.","prof.","vs.","etc.","e.g.","i.e.","fig.","ref.","al.","no.","approx."))
BOUNDARY=re.compile(r"""[.!?]+["')\]]*(?=\s|$)|\n{2,}""")

def require(ok,code):
    if not ok: raise ValueError(code)

def sentences(text):
    """Deterministic conservative segmentation with lossless source offsets."""
    require(type(text) is str and bool(text.strip()),"TEXT")
    start=0; out=[]
    def add(a,b):
        while a<b and text[a].isspace(): a+=1
        while b>a and text[b-1].isspace(): b-=1
        if a<b: out.append((a,b))
    for m in BOUNDARY.finditer(text):
        end=m.end()
        if m.group().startswith(".") and m.group()==".":
            token=re.search(r"[A-Za-z.]+$",text[:end])
            if token and (token.group().lower() in ABBREVIATIONS or
                          re.fullmatch(r"(?:[A-Za-z]\.)+",token.group())):
                continue
        add(start,end);start=end
    add(start,len(text))
    require(bool(out) and "".join(c for c in text if not c.isspace())==
            "".join(c for a,b in out for c in text[a:b] if not c.isspace()),"COVERAGE")
    return tuple(out)

def localize(engine, text, context, *, r9, forward, max_pairs):
    """Full context coverage; long sentences tiled by the unchanged R9 recipe.
    Abort before forwarding if pair budget exceeded; no truncation or fallback
    score. Returns private evidence indices, not a release decision.
    """
    require(type(max_pairs) is int and max_pairs>0,"BUDGET")
    require((r9.CWIDTH,r9.COVERLAP,r9.SWIDTH,r9.SOVERLAP,r9.BATCH,engine.special)==(192,48,317,64,4,3),"RECIPE")
    require(type(context) is str and bool(context.strip()),"CONTEXT")
    windows=r9.tiles(engine.tokens(context),192,48)
    spans=sentences(text)
    units=[(i,r9.tiles(engine.tokens(text[a:b]),317,64)) for i,(a,b) in enumerate(spans)]
    pairs=len(windows)*sum(len(v) for _,v in units)
    require(0<pairs<=max_pairs,"PAIR_BUDGET")
    details=[]
    for i,segments in units:
        for j,h in enumerate(segments):
            posts=[]
            for k in range(0,len(windows),4):
                batch=[engine.encode(c,h) for c in windows[k:k+4]]
                values=forward(engine,batch)
                require(type(values) is list and len(values)==len(batch),"FORWARD_COUNT")
                for v in values:
                    require(type(v) is dict and set(v)=={"entailment","contradiction","neutral"} and
                            all(type(x) in (int,float) and isfinite(x) and 0<=x<=1 for x in v.values()) and
                            abs(sum(v.values())-1)<1e-5,"POSTERIOR")
                posts.extend(values)
            e=max(range(len(posts)),key=lambda k:posts[k]["entailment"])
            c=max(range(len(posts)),key=lambda k:posts[k]["contradiction"])
            details.append(dict(sentence=i,segment=j,entailment=posts[e]["entailment"],
                contradiction=posts[c]["contradiction"],support_window=e,conflict_window=c))
    raw=max(x["contradiction"] for x in details)
    unsupported=max(1-x["entailment"] for x in details)
    return dict(sentence_spans=spans,units=details,pairs=pairs,context_windows=len(windows),
                localized_contradiction=raw,localized_support_deficit=unsupported,
                localized_joint=max(raw,unsupported))

def ranking_diagnostics(rows, *, coverages=(.25,.5,.75,1.)):
    """Low score accepted first. Expected tie-randomized AURC; coverage points
    use whole score ties, at most floor(n*target), no fitted thresholds.
    Descriptive only; no release decision or risk guarantee.
    """
    require(type(rows) in (tuple,list) and bool(rows),"ROWS")
    require(type(coverages) is tuple and bool(coverages) and
            all(type(c) in (int,float) and isfinite(c) and 0<c<=1 for c in coverages) and
            tuple(sorted(set(coverages)))==coverages,"COVERAGES")
    bins={}; seen=set()
    for row in rows:
        require(type(row) in (tuple,list) and len(row)==3,"ROW")
        g,y,s=row
        require(type(g) is str and bool(g) and g not in seen and type(y) is bool and
                type(s) in (int,float) and isfinite(s) and 0<=s<=1,"INPUT")
        seen.add(g);a,e=bins.get(s,(0,0));bins[s]=(a+1,e+int(y))
    n=len(rows);a=e=0;area=0.;points=[(0,0)]
    for s,(m,f) in sorted(bins.items()):
        area+=sum((e+j*f/m)/(a+j) for j in range(1,m+1))
        a+=m;e+=f;points.append((a,e))
    result=[]
    for c in coverages:
        a,e=max((a,e) for a,e in points if a<=n*c)
        result.append(dict(target_coverage=c,accepted=a,accepted_nonfactual=e,
                           coverage=a/n,risk=e/a if a else None))
    return dict(n=n,aurc=area/n,coverage_points=result,release_decision=None)
