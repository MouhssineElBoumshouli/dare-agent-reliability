# DARE-Bench Agent Reliability Study

**Working research title:** *Beyond Average Score: Repeatability of LLM Data-Science Agents on DARE-Bench*

This repository studies a question that is related to, but narrower than, the original DARE-Bench paper:

> When the same deterministic data-science instruction-following task is given to the same agent repeatedly, how often does the agent succeed consistently rather than only on average?

The project deliberately does **not** contain fabricated or hand-entered benchmark scores. Results are generated from DARE-Bench prediction files with the official evaluator and stored with hashes and run metadata.

## Research questions

- **RQ1 — Repeatability:** How often does an agent pass the same DARE-Bench IF task across repeated runs?
- **RQ2 — Interaction budget:** Does increasing the agent turn budget from 3 to 5 improve both average success and repeatability when the code-execution timeout is held constant?
- **RQ3 — Failure modes:** Which failures drive unreliability: tool/API errors, Python/runtime errors, timeouts, malformed outputs, instruction deviations, or incorrect predictions?

## Phase-1 experiment

We intentionally start small enough to finish and reproduce.

- Benchmark: official DARE-Bench evaluation release
- Task variants: `question_v1` only
- Families: classification-IF + regression-IF
- Tasks: 12 classification + 12 regression = **24 fixed tasks**
- Repeats: **5** independent runs per task/condition
- Conditions:
  - `turns_3`: 3 agent turns, 200 s code-execution timeout
  - `turns_5`: 5 agent turns, 200 s code-execution timeout
- Runs for one model: `24 × 5 × 2 = 240`
- Primary outcome: official DARE-Bench IF exact-match success (0/1)
- Secondary outcomes: all-pass rate, flaky-task rate, pairwise disagreement, runtime, tool calls, and failure category

Why IF first? Classification-IF and regression-IF have deterministic reference workflows and strict pass/fail evaluation, which makes them unusually clean for a repeatability study. Time-series and open-ended MM tasks can be a later extension.

## Reproducibility rules

1. Pin the exact DARE-Bench commit.
2. Generate IF reference predictions in the same local environment used for evaluation.
3. Keep the model name/snapshot, provider, agent settings, timeout, prompt version, Python version, package lock, and run IDs.
4. Never type a score into the final results table manually.
5. Save a SHA-256 hash of every evaluated `prediction.csv`.
6. Preserve raw run logs. Derived tables/figures must be regenerable from them.
7. Do not silently rerun failed tasks and replace failures. A retry is a new recorded run.
8. Predefine the 24-task subset before model runs begin.

## Pinned upstream

At project creation, the inspected upstream DARE-Bench revision was:

`01447145304c67b861a004ada6d86f29640de61a`

The bootstrap script checks out this commit.

## Quick start on Windows PowerShell

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\scripts\bootstrap.ps1
.\.venv\Scripts\Activate.ps1

python src\select_tasks.py `
  --dare-root vendor\dare-bench `
  --per-type 12 `
  --seed 20260825 `
  --output configs\task_subset.json

python src\prepare_subset.py `
  --dare-root vendor\dare-bench `
  --subset configs\task_subset.json `
  --output workspace\eval_subset

python vendor\dare-bench\scripts\reference_solution.py `
  --databases_dir workspace\eval_subset\databases `
  --question_list workspace\eval_subset\question_list.json `
  --output_dir workspace\reference_solutions `
  --execute
```

Then verify that every selected classification/regression task contains:

`verify/simulated_pred_local.csv`

Do **not** begin paid/API agent runs until this reference-generation step is clean.

## Sandbox

DARE-Bench's released agent path expects an HTTP code sandbox. Sandbox Fusion can be started with Docker:

```powershell
docker run --rm -p 8080:8080 volcengine/sandbox-fusion:server-20250609
```

In another terminal, test it:

```powershell
Invoke-RestMethod `
  -Uri "http://localhost:8080/run_code" `
  -Method Post `
  -ContentType "application/json" `
  -Body '{"code":"print(\"hello\")","language":"python"}'
```

The actual model/provider launch command is intentionally not hard-coded yet. We will choose the first model after the benchmark + sandbox setup is verified, then pin its exact configuration before collecting data.

## Repository layout

```text
dare-agent-reliability/
├── configs/
│   ├── experiment.yaml
│   └── task_subset.json          # generated, then frozen
├── docs/
│   ├── protocol.md
│   ├── failure_taxonomy.md
│   └── results_policy.md
├── results/
│   ├── raw/                      # immutable run-level artifacts/logs
│   ├── derived/                  # regenerated tables
│   └── figures/                  # regenerated plots
├── scripts/
│   ├── bootstrap.ps1
│   └── bootstrap.sh
├── src/
│   ├── select_tasks.py
│   ├── prepare_subset.py
│   ├── record_run.py
│   └── reliability_metrics.py
├── vendor/
│   └── dare-bench/               # created by bootstrap; not committed
├── workspace/                    # generated local benchmark working data
├── .gitignore
├── CITATION.cff
├── requirements-analysis.txt
└── README.md
```

## Planned outputs

Once real runs exist, the repo should expose:

- `results/derived/task_reliability.csv`
- `results/derived/condition_summary.csv`
- `results/derived/summary.json`
- figures comparing average success vs strict repeatability
- a small failure-mode analysis with traceable run IDs
- an environment manifest and exact upstream/model revisions

Only after these exist should we write the GitHub headline, LinkedIn post, CV bullet, or research follow-up email.
