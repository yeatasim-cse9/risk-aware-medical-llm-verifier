"""Private signed evidence gates for MASTER 01D.

Trust registry and its digest MUST be provisioned independently by institution.
Self-created trust roots are not proof of authority. No signing keys, signatures,
review data or private registry should enter public GitHub or CI.
No network, file write, final freeze, or E1 launch is implemented here.
"""
import re
import json
import hashlib
from pathlib import Path
from collections import Counter
import e1_master01c_core as c
import e1_master01d_review as d

require=c.require
PRE_ACCESS={"dataset_rights":"rights_officer","private_environment":"data_custodian",
            "review_plan":"scientific_lead","one_gpu_run":"compute_owner"}
FREEZE={"dataset_rights":"rights_officer","source_provenance_review":"provenance_reviewer",
        "semantic_disposition":"semantic_adjudicator","clinical_label_review":"clinical_adjudicator",
        "independent_scientific_approval":"independent_scientist",
        "locked_analysis_protocol":"scientific_lead","private_manifest_review":"data_custodian"}
REVIEW_ROLES={"semantic_reviewer_1","semantic_reviewer_2","semantic_adjudicator",
              "clinical_reviewer_1","clinical_reviewer_2","clinical_adjudicator"}
ROLES=set(PRE_ACCESS.values())|set(FREEZE.values())|REVIEW_ROLES
SCOPE="benchmark_answer_role_only"

def hex_digest(x):
    return type(x) is str and re.fullmatch("[0-9a-f]{64}",x) is not None

def canonical(value):
    return json.dumps(value,sort_keys=True,separators=(",",":"),
                      ensure_ascii=True,allow_nan=False).encode()

def verify_registry(registry,expected_registry_sha256):
    require(hex_digest(expected_registry_sha256) and c.digest(registry)==expected_registry_sha256,
            "TRUST_ROOT_MISMATCH")
    require(type(registry) is dict and set(registry)==ROLES,"TRUST_ROLES")
    for role,row in registry.items():
        require(type(row) is dict and set(row)=={"identity","public_key_hex"} and
                type(row["identity"]) is str and bool(row["identity"].strip()) and
                hex_digest(row["public_key_hex"]),"TRUST_ENTRY")
    # One person may hold compatible roles, but keys and identities must map
    # one-to-one. The same qualified panel may review both tasks if approved.
    identity_to_key={};key_to_identity={}
    for row in registry.values():
        i,k=row["identity"],row["public_key_hex"]
        require(identity_to_key.get(i,k)==k and key_to_identity.get(k,i)==i,"TRUST_IDENTITY_KEY")
        identity_to_key[i]=k;key_to_identity[k]=i
    for kind in ("semantic","clinical"):
        panel=[registry[kind+s]["identity"] for s in ("_reviewer_1","_reviewer_2","_adjudicator")]
        require(len(set(panel))==3,"ROLE_SEPARATION")
    independent=registry["independent_scientist"]["identity"]
    custodian=registry["data_custodian"]["identity"]
    panels={registry[role]["identity"] for role in REVIEW_ROLES}
    require(independent not in panels|{custodian,registry["scientific_lead"]["identity"]}
            and custodian not in panels,"ROLE_SEPARATION")

def verify_signature(registry,role,payload,signature):
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
    from cryptography.exceptions import InvalidSignature
    require(role in registry and type(signature) is str and
            re.fullmatch("[0-9a-f]{128}",signature) is not None,"SIGNATURE_FORMAT")
    try:
        Ed25519PublicKey.from_public_bytes(bytes.fromhex(registry[role]["public_key_hex"])).verify(
            bytes.fromhex(signature),canonical(payload))
    except (InvalidSignature,ValueError):
        raise ValueError("INVALID_APPROVAL_SIGNATURE") from None

def implementation_digest():
    root=Path(__file__).resolve().parent
    names=("e1_master01d_review.py","e1_master01d_gate.py","e1_master01d_session.py",
           "e1_master01c_core.py","e1_master01c_private.py",
           "e1_master01b_kaggle.py","e1_master01_validation_kaggle.py")
    return c.digest({name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in names})

def subject(*,environment_digest,session_digest,analysis_digest):
    require(all(hex_digest(x) for x in (environment_digest,session_digest,analysis_digest)),
            "SUBJECT_DIGEST")
    return {"checkpoint":d.CHECKPOINT,"mapping":d.MAPPING,"review_plan":d.plan_digest(),
            "environment":environment_digest,"session":session_digest,"implementation":implementation_digest(),
            "analysis_protocol":analysis_digest,"scope":SCOPE}

def validate_subject(s):
    require(type(s) is dict and set(s)=={"checkpoint","mapping","review_plan","environment",
            "session","analysis_protocol","scope","implementation"} and s==subject(
            environment_digest=s["environment"],session_digest=s["session"],
            analysis_digest=s["analysis_protocol"]),"APPROVAL_SUBJECT")

def verify_receipts(receipts,roles,s,registry,trust_sha,*,phase):
    verify_registry(registry,trust_sha);validate_subject(s)
    require(type(receipts) is dict and set(receipts)==set(roles),"MISSING_APPROVAL_RECEIPTS")
    for gate,role in roles.items():
        record=receipts[gate]
        require(type(record) is dict and set(record)=={"payload","signature"},"RECEIPT_SCHEMA")
        p=record["payload"]
        require(type(p) is dict and set(p)=={"phase","gate","role","subject","decision",
                "evidence_sha256"} and p["phase"]==phase and p["gate"]==gate and
                p["role"]==role and p["subject"]==s and p["decision"]=="approve" and
                hex_digest(p["evidence_sha256"]),"RECEIPT_BINDING")
        verify_signature(registry,role,p,record["signature"])

def pre_access(receipts,s,registry,trust_sha):
    """Signatures establish possession of keys, not legal rights or actual quota.

    Custodian must independently authenticate registry, attest environment,
    enforce single-use session and obtain fresh authorization before data access.
    """
    verify_receipts(receipts,PRE_ACCESS,s,registry,trust_sha,phase="pre_access")
    require(receipts["private_environment"]["payload"]["evidence_sha256"]==s["environment"] and
            receipts["review_plan"]["payload"]["evidence_sha256"]==s["review_plan"] and
            receipts["one_gpu_run"]["payload"]["evidence_sha256"]==s["session"],
            "PRE_ACCESS_EVIDENCE_BINDING")
    return {"signed_pre_access_receipts_valid":True,
            "external_environment_and_single_use_enforcement_required":True,
            "medical_data_accessed":False,"gpu_submitted":False}

def review_signed(expected,submissions,registry,trust_sha,s,*,clinical=False,packet_digest):
    """Verify separate signed reviewer submissions, then validate full coverage.

    Packet digest MUST be independently computed from exact blinded packet
    collection issued to reviewers by authorized custodian.
    """
    verify_registry(registry,trust_sha);validate_subject(s)
    require(hex_digest(packet_digest),"PACKET_DIGEST")
    kind="clinical" if clinical else "semantic"
    roles=(kind+"_reviewer_1",kind+"_reviewer_2",kind+"_adjudicator")
    require(type(submissions) is dict and set(submissions)==set(roles),"REVIEW_SIGNATURE_COVERAGE")
    data=[]
    for role in roles:
        row=submissions[role]
        require(type(row) is dict and set(row)=={"judgments","signature"},"REVIEW_SIGNATURE_SCHEMA")
        p={"phase":"review","role":role,"subject":s,"packet_sha256":packet_digest,
           "judgments_sha256":c.digest(row["judgments"])}
        verify_signature(registry,role,p,row["signature"])
        data.append(row["judgments"])
    return d.resolve(expected,*data,clinical=clinical)

def freeze_candidate(mapping,receipts,s,registry,trust_sha,*,semantic_result,
                     clinical_result,semantic_review_summary,clinical_review_summary):
    """Return manual freeze eligibility only; no write or scientific approval flag.

    Custodian must compute results from authenticated review_signed returns and
    keep all detailed evidence privately. Review signatures and evidence digests
    do not prove reviewer qualification, article identity, or correct judgments.
    """
    c.freeze_preparation(mapping,d.MAPPING,dict.fromkeys(c.GATES,False))
    require(Counter(x[2] for x in mapping)=={k:d.COUNTS[k] for k in c.SPLITS[:3]},
            "MAPPING_COUNTS")
    verify_receipts(receipts,FREEZE,s,registry,trust_sha,phase="freeze")
    require(semantic_result["requires_missed_pair_remediation"] is False and
            semantic_result["complete_semantic_leakage_excluded"] is False,
            "MISSED_PAIR_REMEDIATION_REQUIRED")
    require(semantic_review_summary["n"]==771 and clinical_review_summary["n"]>0,
            "REVIEW_DENOMINATORS")
    require(receipts["semantic_disposition"]["payload"]["evidence_sha256"]==
            c.digest([semantic_result,semantic_review_summary]) and
            receipts["clinical_label_review"]["payload"]["evidence_sha256"]==
            c.digest([clinical_result,clinical_review_summary]) and
            receipts["private_manifest_review"]["payload"]["evidence_sha256"]==d.MAPPING and
            receipts["locked_analysis_protocol"]["payload"]["evidence_sha256"]==s["analysis_protocol"],
            "FREEZE_EVIDENCE_BINDING")
    return {"status":"SIGNED_GATE_CHECKS_PASSED_MANUAL_FREEZE_REQUIRED",
            "scope":SCOPE,"mapping_sha256":d.MAPPING,
            "all_labels_clinically_certified":False,"final_split_frozen":False,
            "final_split_approved":False,"scientific_e1_experiment_executed":False}
