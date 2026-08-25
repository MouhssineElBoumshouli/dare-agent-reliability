# Phase-1 execution safety and API accounting

This note documents the provenance/safety revision made after the separated
pilot and before any final run. It does not change the frozen task subset,
model, decoding parameters, turn conditions, executor timeout, or repeat count.

## Per-run accounting

The study runner captures the `usage` object returned by every successful
OpenAI Chat Completions request. Every immutable final run directory will contain:

- `token_usage.json`: request IDs, returned model IDs, SDK retry setting, and
  provider-reported prompt, cached-prompt, completion, and total tokens.
- `api_cost.json`: the mechanical cost calculation using the price table frozen
  in `configs/execution.yaml`.
- `metadata.json`: embeds both records and their relative artifact paths.

The derived run collector exposes `api_cost_usd` for every final identity.

## Fail-stop behavior

- Before allocating each new run identity, the runner executes a small sandbox
  contract preflight. A failed preflight stops the batch without creating the
  next identity.
- A terminal provider error (authentication, quota/billing, rate limit,
  connection/timeout, or provider internal error), evaluator exception, or
  detected sandbox connection failure stops the batch with a nonzero exit code.
- If an error occurs after an identity has been allocated, its logs, usage seen
  so far, evaluator output, and terminal metadata are preserved before stopping.
- The orchestration layer never retries a completed or failed run identity.
  `--resume` skips every existing identity directory, including incomplete or
  failed identities, and never overwrites one.

The frozen OpenAI SDK setting is `max_retries=2`. These are bounded HTTP-request
retries inside the same immutable run, not replacement benchmark runs. After
the SDK exhausts them, the terminal error is recorded and the batch stops. This
setting was already part of the original execution freeze and was not changed
by the safety revision.

## Cost projection

`results/validation/api_cost_projection.json` is generated from two separate
pilot identities with provider-reported usage, one for each turn condition. The
projection is a budget estimate, not a benchmark result and not a hard spending
cap.
