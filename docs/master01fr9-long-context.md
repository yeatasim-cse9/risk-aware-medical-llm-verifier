# Candidate A: full-coverage segmented NLI (proposed)

Choose the existing pair-encoder limit (192 context, 317 statement, three
special tokens) with context overlap 48 and statement overlap 64. Enumerate
every context window against every statement window, no silent truncation,
no exclusions, no dummy pairs. Use the same pinned NLI model and batch size
four. Validate the private CPU length audit by exact aggregate reproduction
before inference. Keep the same fit/selection cohorts, same-run two-fit lexical
comparator and fit-only logistic heads; preserve historical comparator drift.

For each statement window independently take the maximum entailment and
maximum contradiction across all context windows. Across statement windows
report max entailment, max contradiction, mean entailment, mean contradiction,
min entailment and proportion of best entailment >= 0.5. Raw nonentailment
is one minus the minimum best entailment. These are uncalibrated proxy
features, not probabilities of clinical truth. No claim that every fragment
is a complete assertion or that full token coverage preserves cross-window
reasoning. Overlap may duplicate information; longer contexts get more
opportunities for extreme scores. Report this limitation and keep all
inferences exploratory.

For ablation, freeze full-trained heads. Removed evidence uses one empty
premise per statement window; permuted evidence uses cyclic whole-context
rotation. No claim of equal compute across conditions; count and verify
the actual forward examples. Bootstrap intervals are exploratory paired
group estimates (not final test or multiple-comparison-adjusted claims).
The recipe is a scientific change from previous R8; supervisor/operator
review, immutable private audit binding, actual GPU qualification and fresh
one-run authorization remain required. No actual private measurements or
private artifacts in this repository.
