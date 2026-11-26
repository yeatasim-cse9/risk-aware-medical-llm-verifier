"""Fixed-vocabulary recovery telemetry. No data, secrets, network, or submission.
Caller must retain a private-safe output descriptor before suppressing raw stdout/stderr.
Never write exception text, traceback, dynamic names or row-derived values.
"""
import os

STAGES=("BOOTSTRAP","AUTHORIZATION","CHECKPOINT","DEPENDENCIES","RUNTIME_PIN","GPU_SETUP","MODEL_ASSET_ACQUIRE","MODEL_LOAD_SMOKE","GPU_SMOKE_WINDOWS","GPU_SMOKE_FULL","GPU_SMOKE_REMOVED","GPU_SMOKE_PERMUTED","GPU_SMOKE_REPEAT","HISTORICAL_RECONSTRUCTION","HISTORICAL_SCREEN","MAPPING_HOLDOUT","TRAIN_DEV_ANSWERS","PILOT_INPUTS","MODEL_RELOAD","AUDIT_TRAIN","AUDIT_DEV","PILOT_EXECUTION","AGGREGATE_PREPARE",
        "AGGREGATE_VALIDATE","AGGREGATE_WRITE","COMPLETE")
CODES=frozenset((
    "V4_AUTHORIZATION_REQUIRED","T4_X2_REQUIRED","PUBLIC_PAIR_CONTRACT","BOUNDARY_CONTRACT","PAIR_IDS","MULTIWINDOW_SMOKE","BUDGET_SMOKE","DETERMINISTIC_REPEAT_MISMATCH","REMOVAL_SMOKE","AUDIT_TOKEN_RANGE","AUDIT_WINDOW_COVERAGE","AUDIT_TENSOR_SHAPE","AUDIT_ATTENTION_MASK","AUDIT_SEGMENTS","AUDIT_PERMUTATION","AUDIT_MODEL_CONTRACT","AUDIT_BASELINE_EMPTY_ANSWER","AUDIT_BASELINE_FEATURES","AUDIT_SLOT_COUNT","AUDIT_BUDGET","AUDIT_ACTUAL_BUDGET_MISMATCH","PAIR_SCHEMA","ANSWER_ROLES","NO_CROSS_SOURCE_PERMUTATION","BUDGET_CONTRACT","PARENT_GROUP_ALIGNMENT","AUTHORIZATION_REQUIRED","PRIVATE_SCRATCH",
    "HISTORICAL_CHECKPOINT","PRIOR_SOURCE_PIN","MANIFEST_PIN","RUNTIME_VERSION_DRIFT",
    "GPU_REQUIRED","ASSET_HASH","MODEL_LOAD_MISMATCH","CANDIDATE_ARCHITECTURE",
    "NLI_LABEL_MAP","MODEL_SEQUENCE_CONTRACT","SPECIAL_TOKEN_CONTRACT","PAIR_TOKENIZATION",
    "TOKEN_TYPES","NLI_OUTPUT","NLI_SMOKE","CONTEXT_BUDGET_EXCEEDED",
    "HYPOTHESIS_BUDGET_EXCEEDED","CHECKPOINT_DRIFT","HOLDOUT_PIN","HOLDOUT_IDENTITY",
    "PRIVATE_ANSWER_SCOPE","LOCAL_PARQUET_PIN","LOCAL_PARQUET_SCHEMA",
    "REPORT_INTEGRITY","REPORT_SIZE","AGGREGATE_SCHEMA","NUMERICAL_REPRODUCTION",
    "ENCODER_AGGREGATE_REPRODUCTION","AGGREGATE_REPRODUCTION"))
KINDS={ModuleNotFoundError:"MODULE_MISSING",ImportError:"IMPORT_ERROR",
       FileNotFoundError:"FILE_MISSING",PermissionError:"PERMISSION_ERROR",
       TimeoutError:"TIMEOUT",MemoryError:"MEMORY_ERROR",OSError:"OS_ERROR",
       RuntimeError:"RUNTIME_ERROR",TypeError:"TYPE_ERROR",
       AttributeError:"ATTRIBUTE_ERROR",ValueError:"VALUE_ERROR"}

def classify(exc):
    kind=KINDS.get(type(exc),"OTHER_EXCEPTION")
    # Only an exact, fixed ValueError token may be exported; never str(exc).
    code="UNCLASSIFIED"
    if type(exc) is ValueError and len(exc.args)==1 and type(exc.args[0]) is str:
        if exc.args[0] in CODES:
            code=exc.args[0]
    return kind,code

class Recorder:
    def __init__(self,emit):
        self._emit=emit
        self._index=0
        self._failed=False
        self._emit("MASTER01EV4_STAGE=BOOTSTRAP")

    def mark(self,stage):
        if (self._failed or self._index+1>=len(STAGES)
                or type(stage) is not str or STAGES[self._index+1]!=stage):
            raise ValueError("RECOVERY_STAGE_ORDER")
        self._index+=1
        self._emit("MASTER01EV4_STAGE="+STAGES[self._index])

    def failure(self,exc):
        if self._failed or self._index==len(STAGES)-1:
            raise ValueError("RECOVERY_STAGE_TERMINAL")
        self._failed=True
        kind,code=classify(exc)
        self._emit("MASTER01EV4_FAILED_STAGE="+STAGES[self._index])
        self._emit("MASTER01EV4_FAILURE_KIND="+kind)
        self._emit("MASTER01EV4_FAILURE_CODE="+code)

def recorder_for_fd(fd):
    # fd must be the wrapper's saved output descriptor, never a raw log relay.
    if type(fd) is not int or fd<0:
        raise ValueError("RECOVERY_DESCRIPTOR")
    def emit(line):
        data=(line+"\n").encode("ascii")
        if os.write(fd,data)!=len(data):
            raise OSError("RECOVERY_TELEMETRY_WRITE")
    return Recorder(emit)
