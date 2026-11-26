"""V3 feature extraction from fixed, precomputed NLI observations.
No data/model access, no labels, no free-text logging. Upstream inference and
source lineage enforcement are not implemented here. Synthetic fixtures only.
"""
from pirc_v2_method import probability, require, features

def aligned_extract(original,revised,*,unit_ids,allowed):
    """Require identical ordered evidence IDs; IDs remain private.
    Cannot establish authenticity of upstream scores or ordering.
    INVALID outputs must be forced to abstain by the evaluation wrapper.
    """
    require(type(unit_ids) is tuple and len(unit_ids)<=64 and
            all(type(k) is str and bool(k) for k in unit_ids) and
            len(set(unit_ids))==len(unit_ids),"UNIT_IDS")
    candidates=[]
    for row in (original,revised):
        require(type(row) is dict and set(row)=={"text","confidence","units"}
                and type(row["units"]) is list,"ALIGNED_SCHEMA")
        require(all(type(u) is dict and set(u)=={"id","nli"} for u in row["units"]),
                "UNIT_SCHEMA")
        require(tuple(u["id"] for u in row["units"])==unit_ids,"UNIT_ALIGNMENT")
        parsed=parse_answer(row["text"],allowed)
        candidates.append({"answer":parsed,"confidence":row["confidence"],
                           "full":[u["nli"] for u in row["units"]]})
    return {"features":extract(*candidates,evidence_count=len(unit_ids)),
            "original_valid":candidates[0]["answer"]!="INVALID",
            "revised_valid":candidates[1]["answer"]!="INVALID"}


def aggregate(scores):
    # Every evidence unit evaluated for the same single closed-answer claim.
    require(type(scores) is list and bool(scores),"EVIDENCE_SCORES")
    for s in scores:
        require(type(s) is dict and set(s)=={"entailment","contradiction","neutral"}
            and all(probability(v) for v in s.values()) and
            abs(sum(s.values())-1)<1e-6,"NLI_PROBABILITIES")
    return (max(s["entailment"] for s in scores),
            max(s["contradiction"] for s in scores),
            sum(s["entailment"]>=.5 for s in scores)/len(scores))

def extract(original,revised,*,evidence_count):
    """Low-level inputs: parsed answer symbol, confidence and ordered full NLI scores.
    Use aligned_extract to check identical evidence IDs. This low-level helper
    assumes aligned ordering. No gold labels accepted. Evidence
    coverage is supportive-unit fraction, not recall of clinical evidence.
    Perturbation = removal of first evidence unit (fixed order, answer-blind).
    When one unit is removed all evidence is absent; support becomes zero.
    """
    require(type(evidence_count) is int and 0<=evidence_count<=64,"EVIDENCE_COUNT")
    out={}
    for prefix,row in (("original",original),("revised",revised)):
        require(type(row) is dict and set(row)=={"answer","confidence","full"}
            and type(row["answer"]) is str and row["answer"] in
            ("yes","no","maybe","A","B","C","D","E","INVALID")
            and probability(row["confidence"]) and type(row["full"]) is list
            and len(row["full"])==evidence_count,"CANDIDATE_SCHEMA")
        if evidence_count:
            support,contradiction,coverage=aggregate(row["full"])
            perturbed=aggregate(row["full"][1:])[0] if evidence_count>1 else 0.0
        else: support=contradiction=coverage=perturbed=0.0
        out.update({prefix+"_support":support,prefix+"_contradiction":contradiction,
            prefix+"_evidence_coverage":coverage,prefix+"_support_drop":max(0.,support-perturbed),
            prefix+"_confidence":row["confidence"]})
    out.update(answer_changed=int(original["answer"]!=revised["answer"]),
               evidence_available=int(evidence_count>0))
    features(out)
    return out

def parse_answer(text,allowed):
    """Strict exact-symbol parser; no answer extraction by gold match."""
    require(type(text) is str and type(allowed) is tuple and
        allowed in (("yes","no","maybe"),("yes","no"),("A","B","C","D"),("A","B","C","D","E")),
        "PARSER_INPUT")
    normalized=text.strip()
    normalized=normalized.lower() if allowed[0]=="yes" else normalized.upper()
    return normalized if normalized in allowed else "INVALID"
