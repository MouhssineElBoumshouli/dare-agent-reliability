# Research Summary

## Background

Aggregate benchmark scores can conceal whether an agent reliably solves the same deterministic task. DARE-Bench supplies process-aware data-science tasks with verifiable outputs, making its instruction-following tasks suitable for separating average capability from repeatability. This project is an independent reliability study built on DARE-Bench, not a new benchmark and not an effort affiliated with Snowflake or the DARE-Bench authors.

## Research Question

When the same deterministic DARE-Bench instruction-following task is given repeatedly to the same LLM agent, how consistently does it succeed, and how does increasing the turn budget from three to five affect both average exact-match performance and run-to-run reliability?

## Method

The preregistered Phase-1 sample comprised 12 Classification-IF and 12 Regression-IF tasks selected with seed `20260825` from DARE-Bench revision `01447145304c67b861a004ada6d86f29640de61a`. The frozen subset SHA-256 was `2FA2F54C22265E04F3A39F2B8CFFA3DB12B58B508F2DF0C12940780ACA03C77E`. The agent used `gpt-4.1-mini-2025-04-14` under two conditions: 3 or 5 maximum turns, with a 200-second executor timeout in both. Five independent repetitions produced 240 immutable run identities.

References were generated locally through the pinned DARE-Bench reference solution. Every produced prediction was scored, and then independently rescored, with the pinned official evaluator. Exact match was binary. Reliability outcomes were always pass, never pass, flaky, and pairwise disagreement across the 10 repeat pairs per task-condition. Percentile 95% confidence intervals resampled tasks using 10,000 bootstrap samples and seed `20260825`.

## Results

All 240 expected identities were present with no missing or duplicate identities. Official exact-match success increased from 34.2% (41/120) under three turns to 50.0% (60/120) under five turns. The task-bootstrap 95% intervals were 18.3%–50.8% and 34.2%–65.8%, respectively. The paired mean task-level difference was +15.8 percentage points (95% CI: +5.0 percentage points to +27.5 percentage points); 9 tasks improved, 2 declined, and 13 tied.

Classification success rose from 40.0% to 61.7%, while regression rose from 28.3% to 38.3%. Classification contributed 13 of 19 additional passes (68.4%); regression contributed 6. This concentration is descriptive because each family contains only 12 tasks.

Reliability did not move uniformly with average performance. Always-pass tasks increased from 20.8% to 29.2%, and never-pass tasks decreased from 50.0% to 29.2%. However, flaky tasks increased from 29.2% to 41.7%, while mean pairwise disagreement increased from 15.0% to 21.7%.

Among 139 official failures, mechanically supported labels were 94 code errors, 42 wrong predictions without stronger causal evidence, 2 malformed predictions, and 1 turn/token-limit failure. Recorded API usage totaled 2,260,610 tokens at $1.160817.

## Interpretation

Increasing the turn budget improved mean capability and converted some never-pass tasks into successful or intermittently successful tasks. It did not make outcomes uniformly more stable. The simultaneous increase in flaky-task prevalence and pairwise disagreement indicates that a higher success rate can coexist with lower repeatability among tasks near the agent's capability boundary. Average score and reliability therefore answer different evaluation questions and should be reported together.

## Limitations

The study covers one model (`gpt-4.1-mini-2025-04-14`), 24 fixed IF tasks rather than the entire benchmark, and 5 repetitions per condition. It includes Classification-IF and Regression-IF only. Results should not automatically generalize to other models, agent architectures, decoding configurations, execution environments, or DARE-Bench task families. Bootstrap intervals are descriptive, and no formal significance claim is made. Failure categories are limited to what preserved logs and evaluator diagnostics mechanically support.

## Reproducibility

The execution protocol was frozen at commit `94d1fe40d4a6569dcdb664c3e3f64f5738d49271` and tag `phase1-execution-freeze-v2` before the final runs. The committed integrity report verifies 240 unique identities, 145 prediction hashes, exact returned model IDs for 873 provider responses, per-run token and cost artifacts, and 240 independent official rescores with no exceptions. Statistical outputs are regenerated from `results/derived/runs.csv`; publication figures read only `results/derived/`. Documentation rendering and tests fail if committed metrics drift from recomputed values.
