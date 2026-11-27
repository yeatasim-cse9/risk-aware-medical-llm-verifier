# R9 development-only token audit

The prior fixed window contract is a feasibility gate, not a reason to discard
long examples. Before any new neural recipe, measure both statement and context
lengths with the pinned tokenizer on the reconstructed development cohort. Audit
both sides even when one exceeds the legacy limit. Export only approved aggregate
length summaries and cost estimates in the private environment. No individual
lengths, texts, labels, predictions, or private results enter the public repository.

The candidate cost estimates enumerate Cartesian context-window / statement-window
pairs with full token coverage, using the same tokenizer and 512-token limit.
They include fit and selection full evidence, selection cyclic context permutation,
and selection evidence removal. They are not measured GPU runtimes or an approved
semantic pooling rule. Splitting hypotheses can change their meaning; the pooling
rule and its limitations require explicit scientific review before a new immutable
GPU plan. No test answers, calibration, automatic GPU dispatch or altered prior
checkpoints. Synthetic tests use invented lengths only.
