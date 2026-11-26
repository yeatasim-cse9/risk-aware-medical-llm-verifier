"""Authorized private session orchestration, never a public workflow entrypoint.

A qualified custodian invokes these functions inside an independently approved
private runtime. Returned review packets, review submissions and custodian keys
are sensitive. Do not display, log, upload, or persist them on unauthorized
surfaces. No actual access control, storage service or reviewer UI is provisioned
by this module.
"""
from pathlib import Path
import hashlib
import os
import e1_master01c_core as c
import e1_master01c_private as p
import e1_master01d_review as d
import e1_master01d_gate as g

require=c.require
PINS={"e1_master01_validation_kaggle.py":
      "11d803b06167689df6a3caa290be69a8b9b519ee9e83ec0d5de619e95a43761e",
      "e1_master01b_kaggle.py":
      "89bdecce4109aa0cc4dc90f1f8a537ce085f13a707c9c0df9258d36625f62e2f",
      "e1_master01c_core.py":
      "a30e9116430a568e8fdb3d2aaa06dd791225246fe8b73b7f580b9c010d7a9602",
      "e1_master01c_private.py":
      "6b3f764fa79c6f839cc6705c4cc05c3178d3c4432d52c762ecffa25a8c100cbc"}

def verify_prior_source():
    root=Path(__file__).resolve().parent
    require(all(hashlib.sha256((root/name).read_bytes()).hexdigest()==h
                for name,h in PINS.items()),"PRIOR_SOURCE_PIN")

def prepare_authorized_session(root,nonce,receipts,subject,registry,trust_sha):
    """Does not submit Kaggle. Requires a separately authorized GPU session.

    The institution must independently enforce access/retention controls,
    externally authenticate trust roots, and consume the one-run authorization
    before calling this function. This check cannot enforce single-use across
    separate processes or independently establish rights.
    """
    g.pre_access(receipts,subject,registry,trust_sha)
    require(type(nonce) is bytes and len(nonce)>=32 and
            hashlib.sha256(nonce).hexdigest()==subject["session"],"SESSION_NONCE_BINDING")
    require(os.environ.get("MASTER01D_AUTHORIZED_PRIVATE_ENVIRONMENT")=="YES" and
            os.environ.get("MASTER01C_PRIVATE_RIGHTS_CONFIRMED")=="YES" and
            os.environ.get("MASTER01C_PRIVATE_REVIEW_CONFIRMED")=="YES","PRIVATE_ENVIRONMENT_REQUIRED")
    verify_prior_source()
    require(isinstance(root,Path) and root.is_dir() and not root.is_symlink(),"PRIVATE_SCRATCH")
    report,private=p.prepare_private(root,return_private=True)
    return d.prepare_review(report,private,nonce,approved_plan_digest=subject["review_plan"])

def analyze_authenticated_reviews(session,semantic_submissions,clinical_submissions,
                                  subject,registry,trust_sha):
    """Run only after independent review and adjudication submissions are locked.

    Both output and source data remain private. Do not give final-test summaries
    or review labels to model developers; public release requires separate
    small-cell disclosure review and institutional approval.
    """
    sem=session["semantic_packets"];clin=session["clinical_packets"];cust=session["custodian"]
    sf,ss=g.review_signed({x["review_id"] for x in sem},semantic_submissions,
        registry,trust_sha,subject,packet_digest=c.digest(sem))
    cf,cs=g.review_signed({x["review_id"] for x in clin},clinical_submissions,
        registry,trust_sha,subject,clinical=True,packet_digest=c.digest(clin))
    s=d.semantic_analysis(cust["queue"],cust["semantic_key"],sf,
        first=semantic_submissions["semantic_reviewer_1"]["judgments"],
        second=semantic_submissions["semantic_reviewer_2"]["judgments"])
    v=d.clinical_analysis(cust,cf,
        first=clinical_submissions["clinical_reviewer_1"]["judgments"],
        second=clinical_submissions["clinical_reviewer_2"]["judgments"])
    return {"semantic_result":s,"clinical_result":v,
            "semantic_review_summary":ss,"clinical_review_summary":cs}

def validate_manual_freeze(session,semantic_submissions,clinical_submissions,
                           receipts,subject,registry,trust_sha):
    """Recompute signed review analysis rather than accept hand-entered summaries."""
    analyses=analyze_authenticated_reviews(session,semantic_submissions,clinical_submissions,
                                          subject,registry,trust_sha)
    return g.freeze_candidate(session["custodian"]["mapping"],receipts,subject,
                             registry,trust_sha,**analyses)
