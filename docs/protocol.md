# Experimental Protocol

## Hypothesis

The same LLM agent can produce materially different success/failure outcomes on deterministic DARE-Bench instruction-following tasks, and a larger interaction budget may improve mean score without eliminating run-to-run unreliability.

## Unit of analysis

A unit is one `(task_id, model, condition, repeat)` run.

The task set is fixed before the first model run. Every selected task receives the same number of planned repeats under each condition.

## Phase-1 sample

- 12 classification-IF tasks
- 12 regression-IF tasks
- 5 repeats
- 2 interaction-budget conditions
- 240 planned runs for one model

The task subset is selected once using a fixed seed and then committed.

## Independent variable

Agent interaction budget:

- 3 turns
- 5 turns

The code execution timeout is fixed at 200 seconds in both conditions to avoid confounding the number of turns with per-execution time.

## Primary dependent variable

DARE-Bench v1 exact-match score:

- pass = `1.0`
- fail = `0.0`

Scoring must be produced by the official DARE-Bench evaluator against locally generated IF reference predictions.

## Reliability measures

For each task/condition:

- success rate across repeats
- **always-pass**: passed all planned repeats
- **never-pass**: failed all planned repeats
- **flaky**: at least one pass and at least one fail
- pairwise disagreement rate between repeated outcomes

For each condition:

- mean run-level success
- fraction of always-pass tasks
- fraction of flaky tasks
- mean pairwise disagreement
- bootstrap 95% confidence intervals resampling tasks

## Secondary measures

When available:

- wall-clock runtime
- tool-call count
- input/output token usage
- prediction-file SHA-256
- failure category

## Stopping rule

Do not stop early because results look good or bad. Complete the predefined repeat count unless a provider outage or benchmark-breaking technical issue invalidates the batch. Invalidated batches must remain documented.

## Environment capture

Before final analysis record:

- operating system
- CPU/RAM
- Python version
- `pip freeze`
- Docker/Sandbox Fusion image
- DARE-Bench commit
- model/provider exact identifier
- experiment config hash or committed revision
