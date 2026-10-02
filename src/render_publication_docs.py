"""Render public research documentation from validated committed artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

from publication_data import load_publication_data, robustness_report, sha256_file


REPO_ROOT = Path(__file__).resolve().parents[1]
DERIVED_DIR = REPO_ROOT / "results" / "derived"
README_PATH = REPO_ROOT / "README.md"
SUMMARY_PATH = REPO_ROOT / "docs" / "research_summary.md"
RELEASE_NOTES_PATH = REPO_ROOT / "docs" / "releases" / "v1.0.0-phase1.md"
ROBUSTNESS_PATH = DERIVED_DIR / "robustness_checks.json"


def pct(value: float, digits: int = 1) -> str:
    return f"{value * 100:.{digits}f}%"


def pp(value: float, digits: int = 1, sign: bool = True) -> str:
    prefix = "+" if sign and value > 0 else ""
    return f"{prefix}{value * 100:.{digits}f} percentage points"


def whole_count(rate: float, total: float) -> int:
    return int(round(float(rate) * float(total)))


def ci(row: dict[str, Any], prefix: str) -> str:
    return (
        f"{pct(float(row[f'{prefix}_ci95_low']))}–"
        f"{pct(float(row[f'{prefix}_ci95_high']))}"
    )


def task_list(rows: list[dict[str, Any]]) -> str:
    return "\n".join(
        f"- `{row['task_id']}` ({row['task_type']}): "
        f"{pct(row['turns_3_success_rate'], 0)} → {pct(row['turns_5_success_rate'], 0)} "
        f"({pp(row['difference'], 0)})"
        for row in rows
    )


def load_context() -> dict[str, Any]:
    data = load_publication_data(DERIVED_DIR)
    execution = yaml.safe_load((REPO_ROOT / "configs" / "execution.yaml").read_text(encoding="utf-8"))
    experiment = yaml.safe_load((REPO_ROOT / "configs" / "experiment.yaml").read_text(encoding="utf-8"))
    environment = json.loads(
        (REPO_ROOT / "results" / "environment" / "environment.json").read_text(encoding="utf-8")
    )
    citation = yaml.safe_load((REPO_ROOT / "CITATION.cff").read_text(encoding="utf-8"))
    subset_path = REPO_ROOT / "configs" / "task_subset.json"
    subset = json.loads(subset_path.read_text(encoding="utf-8"))
    subset_hash = sha256_file(subset_path)

    if subset_hash != execution["task_subset_sha256"]:
        raise ValueError("Frozen task subset hash differs from execution configuration")
    if execution["dare_bench_commit"] != experiment["study"]["upstream_commit"]:
        raise ValueError("Pinned upstream revisions disagree")
    if data["integrity"]["frozen_study_commit"] != "94d1fe40d4a6569dcdb664c3e3f64f5738d49271":
        raise ValueError("Unexpected frozen execution commit in integrity report")
    if len(subset) != execution["planned_run_count"] // execution["repeats"] // len(
        execution["conditions"]
    ):
        raise ValueError("Subset size does not match frozen run design")

    classification_count = sum(item["task"] == "classification" for item in subset)
    regression_count = sum(item["task"] == "regression" for item in subset)
    if classification_count != regression_count or classification_count * 2 != len(subset):
        raise ValueError("Frozen subset is not the expected balanced design")

    report = robustness_report(data)
    return {
        **data,
        "execution": execution,
        "experiment": experiment,
        "environment": environment,
        "citation": citation,
        "subset": subset,
        "subset_hash": subset_hash,
        "classification_count": classification_count,
        "regression_count": regression_count,
        "robustness": report,
    }


FAILURE_DESCRIPTIONS = {
    "code_error": "no answer file, and a Python error shows up in the log",
    "wrong_prediction_unclassified": "a valid answer file with wrong predictions",
    "malformed_prediction": "an answer file the grader could not line up with the reference",
    "max_turn_or_token_limit": "the agent ran out of turns or tokens",
    "execution_timeout": "the code ran past the time limit",
    "tool_call_error": "a broken tool call",
    "instruction_deviation": "valid output, but a required step was skipped or changed",
    "infrastructure_error": "a problem outside the agent, like the sandbox going down",
}


def render_readme(ctx: dict[str, Any]) -> str:
    condition = ctx["condition_records"]
    paired = ctx["paired_records"]
    integrity = ctx["integrity"]
    execution = ctx["execution"]
    experiment = ctx["experiment"]
    environment = ctx["environment"]
    failures = ctx["failure_summary"]
    movements = ctx["robustness"]["task_movements"]
    bootstrap = ctx["summary"]["bootstrap"]

    t3 = condition["turns_3"]["overall"]
    t5 = condition["turns_5"]["overall"]
    class3 = condition["turns_3"]["classification"]
    class5 = condition["turns_5"]["classification"]
    reg3 = condition["turns_3"]["regression"]
    reg5 = condition["turns_5"]["regression"]
    overall_pair = paired["overall"]
    class_pair = paired["classification"]
    reg_pair = paired["regression"]
    costs = integrity["actual_api_cost_usd"]
    tokens = integrity["token_totals"]
    env_packages = environment["sandbox"]["packages"]
    turns3 = execution["conditions"]["turns_3"]
    turns5 = execution["conditions"]["turns_5"]
    repeats = int(execution["repeats"])
    n_tasks = len(ctx["subset"])

    runs = ctx["runs"]
    no_file = runs[runs["prediction_path"].isna()].groupby("condition").size()
    modes = ctx["failure_modes"]
    wrong = (
        modes[modes["failure_category"] == "wrong_prediction_unclassified"]
        .groupby("condition")["count"]
        .sum()
    )
    movement = ctx["movement"]
    rescued = movement[(movement["turns_3"] == 0) & (movement["turns_5"] > 0)]
    rescued_flaky = int((rescued["turns_5"] < 1).sum())
    rescued_flaky_text = (
        "all of them" if rescued_flaky == len(rescued) else f"{rescued_flaky} of them"
    )
    score_steps = ", ".join(f"{round(100 * i / repeats)}%" for i in range(repeats + 1))

    failure_lines = "\n".join(
        f"- `{name}` (**{count}**): {FAILURE_DESCRIPTIONS.get(name, 'see docs/failure_taxonomy.md')}"
        for name, count in sorted(
            failures["failure_category_counts"].items(), key=lambda item: (-item[1], item[0])
        )
    )

    def counts(row: dict[str, Any]) -> str:
        return (
            f"{whole_count(row['run_success_rate'], row['n_runs'])}/{int(row['n_runs'])}; "
            f"CI {ci(row, 'mean_task_success_rate')}"
        )

    def group(row: dict[str, Any], rate: str) -> str:
        return f"{whole_count(row[rate], row['n_tasks'])}/{int(row['n_tasks'])} ({pct(row[rate])})"

    return f"""# Beyond Average Score: Repeatability of LLM Data-Science Agents on DARE-Bench

[![analysis](https://github.com/MouhssineElBoumshouli/dare-agent-reliability/actions/workflows/analysis.yml/badge.svg)](https://github.com/MouhssineElBoumshouli/dare-agent-reliability/actions/workflows/analysis.yml)

If you give an AI agent the same task {repeats} times, does it get the same result every time? This project tests that. It runs one AI agent on {n_tasks} data-science tasks from DARE-Bench, {repeats} times each, under two different turn limits. That's {execution['planned_run_count']} runs in total.

This is an independent reliability study built on [DARE-Bench](https://github.com/Snowflake-Labs/dare-bench). It is not a new benchmark, and it is not made or endorsed by Snowflake or the DARE-Bench authors.

## The short version

Most AI benchmarks run each task once and report an average score. That score can hide a lot. A 50% score could mean the agent always solves half the tasks and never solves the rest. Or it could mean the agent solves every task about half the time. If you rely on the agent, those are very different.

So I gave the same agent the same tasks over and over. Think of a student who gets the same problem {repeats} times and starts fresh each time. First the student gets a short time limit ({turns3['max_turn']} turns). Then a longer one ({turns5['max_turn']} turns).

With more turns, the agent passed more often: **{pct(t3['run_success_rate'])}** of runs passed with {turns3['max_turn']} turns, and **{pct(t5['run_success_rate'])}** with {turns5['max_turn']}. But it also got less predictable. Tasks with mixed results, where some tries pass and some fail, went from **{whole_count(t3['flaky_task_rate'], t3['n_tasks'])} to {whole_count(t5['flaky_task_rate'], t5['n_tasks'])}** out of {n_tasks}. More turns made the agent better on average, but not more consistent.

![Success and variability on a shared percentage scale](results/figures/figure_07_main_result.png)

## How the study worked

- **The tasks.** {ctx['classification_count']} classification tasks (predict a category) and {ctx['regression_count']} regression tasks (predict a number), picked at random from DARE-Bench with a fixed seed. These are instruction-following tasks. The instructions spell out every step, so there is exactly one correct answer.
- **The agent.** DARE-Bench's own agent code, unchanged, running OpenAI's `{execution['model_snapshot']}`. It writes Python, runs it in a locked sandbox, reads the output, and tries again.
- **Turns.** One turn is one round of writing code, running it, and reading the result. The agent got either {turns3['max_turn']} or {turns5['max_turn']} turns per try. The code timeout was {turns3['executor_timeout_seconds']} seconds in both cases, so more turns didn't also mean more time per step.
- **Tries.** Each task got {repeats} separate tries per turn limit. Every try started from scratch, with no memory of the others.
- **Grading.** DARE-Bench's official grader. A try passes only if every prediction exactly matches the correct answer.

That's {n_tasks} tasks × {len(execution['conditions'])} turn limits × {repeats} tries = **{execution['planned_run_count']} runs**.

```mermaid
flowchart LR
    A["{n_tasks} fixed tasks<br/>{ctx['classification_count']} classification + {ctx['regression_count']} regression"] --> B{{"Turn limit"}}
    B --> C["{turns3['max_turn']} turns"]
    B --> D["{turns5['max_turn']} turns"]
    C --> E["{repeats} fresh tries per task"]
    D --> E
    E --> F["{execution['planned_run_count']} runs"]
    F --> G["Official grader<br/>exact match"]
    G --> H["Consistency per task"]
```

I wrote the plan and picked the tasks before running anything, and saved both to Git first. The code for the runs was locked at one commit, and the runner refused to start from any other version. Each run was saved to its own folder that can't be overwritten. Failed runs were never re-run. Afterwards, every run was graded a second time from the saved files.

<details>
<summary>Exact settings</summary>

| Setting | Value |
|---|---|
| DARE-Bench version | `{execution['dare_bench_commit']}` |
| Frozen study code | `{integrity['frozen_study_commit']}` / tag `{integrity['frozen_tag']}` |
| Task variant | `question_{execution['question_version']}` (instruction-following) |
| Task selection seed | `{experiment['benchmark']['task_selection_seed']}` |
| Task list SHA-256 | `{ctx['subset_hash']}` |
| Model | `{execution['model_snapshot']}` via `{execution['provider']}` `{execution['api_mode']}` |
| Decoding | temperature `{execution['temperature']}`, top-p `{execution['top_p']}`, max output tokens `{execution['max_tokens']}` |
| Tools | function calls, provider-default tool choice |
| SDK retries | `{execution['provider_sdk_request_max_retries']}` per request, inside the same run |
| Sandbox | `{execution['sandbox_image']}` at `{execution['sandbox_image_digest']}` |
| Sandbox Python | Python {env_packages['python']}, pandas {env_packages['pandas']}, NumPy {env_packages['numpy']}, scikit-learn {env_packages['scikit_learn']}, pyarrow {env_packages['pyarrow']} |

The temperature is {execution['temperature']} and not 0 because DARE-Bench's code treats 0 as "not set" and switches to 0.7. Using {execution['temperature']} keeps it as close to 0 as possible without changing their code. More detail is in [`docs/model_selection.md`](docs/model_selection.md) and [`docs/protocol.md`](docs/protocol.md).

The machine and software record in [`results/environment/environment.json`](results/environment/environment.json) was saved during setup, before the final freeze. That's why its Git section shows an earlier commit with uncommitted changes. For the final runs, the record is the freeze commit above and [`results/derived/raw_run_integrity.json`](results/derived/raw_run_integrity.json), which checks that all {execution['planned_run_count']} runs used it.

</details>

## What I found

### How often the agent passed

| Turn limit | All tasks | Classification | Regression |
|---|---:|---:|---:|
| {turns3['max_turn']} turns | **{pct(t3['run_success_rate'])}** ({counts(t3)}) | {pct(class3['run_success_rate'])} ({counts(class3)}) | {pct(reg3['run_success_rate'])} ({counts(reg3)}) |
| {turns5['max_turn']} turns | **{pct(t5['run_success_rate'])}** ({counts(t5)}) | {pct(class5['run_success_rate'])} ({counts(class5)}) | {pct(reg5['run_success_rate'])} ({counts(reg5)}) |

"CI" is a 95% confidence interval. It shows the range the true value probably falls in. The ranges are wide because there are only {n_tasks} tasks. They come from resampling tasks {bootstrap['samples']:,} times (seed `{bootstrap['seed']}`).

![Overall exact-match success with task-bootstrap intervals](results/figures/figure_01_success_rate_ci.png)

### How consistent the agent was

For each task, I looked at its {repeats} tries and put it in one of three groups:

- **Always pass:** all {repeats} tries passed.
- **Never pass:** no tries passed.
- **Flaky:** some tries passed and some failed. You can't predict what you'll get.

I also measured **pairwise disagreement**. With {repeats} tries there are {repeats * (repeats - 1) // 2} ways to pick two of them. Disagreement is the share of those pairs where one try passed and the other failed.

| Turn limit | Always pass | Flaky | Never pass | Pairwise disagreement |
|---|---:|---:|---:|---:|
| {turns3['max_turn']} turns | {group(t3, 'always_pass_task_rate')} | {group(t3, 'flaky_task_rate')} | {group(t3, 'never_pass_task_rate')} | {pct(t3['mean_pairwise_disagreement'])} |
| {turns5['max_turn']} turns | {group(t5, 'always_pass_task_rate')} | {group(t5, 'flaky_task_rate')} | {group(t5, 'never_pass_task_rate')} | {pct(t5['mean_pairwise_disagreement'])} |

![Always-pass, flaky, and never-pass composition](results/figures/figure_03_reliability_composition.png)

![Pairwise disagreement by condition](results/figures/figure_04_pairwise_disagreement.png)

### Where the extra turns went

With {turns3['max_turn']} turns, {int(no_file.get('turns_3', 0))} of {int(t3['n_runs'])} runs ended without the agent ever saving an answer file. With {turns5['max_turn']} turns, that dropped to {int(no_file.get('turns_5', 0))}. But wrong answers went up from {int(wrong.get('turns_3', 0))} to {int(wrong.get('turns_5', 0))}. The extra turns mostly helped the agent finish, and many of the runs that finished were still wrong.

### Task by task

Out of {int(overall_pair['n_paired_tasks'])} tasks, {int(overall_pair['tasks_improved'])} did better with {turns5['max_turn']} turns, {int(overall_pair['tasks_declined'])} did worse, and {int(overall_pair['tasks_tied'])} stayed the same. The average change per task was {pp(overall_pair['mean_task_success_rate_difference'])} (95% CI: {pp(overall_pair['ci95_low'])} to {pp(overall_pair['ci95_high'])}). Classification changed by {pp(class_pair['mean_task_success_rate_difference'])} and regression by {pp(reg_pair['mean_task_success_rate_difference'])}. With only {ctx['classification_count']} tasks of each type, that gap is a hint, not proof.

![Paired movement of all fixed tasks](results/figures/figure_05_task_movements.png)

![Classification and regression success by condition](results/figures/figure_02_task_type_success.png)

<details>
<summary>Every task, grouped by how it changed</summary>

#### Improved ({len(movements['improved'])})

{task_list(movements['improved'])}

#### Declined ({len(movements['declined'])})

{task_list(movements['declined'])}

#### No change ({len(movements['tied'])})

{task_list(movements['tied'])}

</details>

## What this does and doesn't show

- **Part of the rise in flakiness is built into the measure.** A task that never passes can't be flaky. {len(rescued)} tasks went from never passing to passing sometimes, and {rescued_flaky_text} ended up flaky. So some of the rise comes from tasks getting better, not from the agent getting less stable. For someone using the agent, the result is the same: with {turns5['max_turn']} turns, {whole_count(t5['flaky_task_rate'], t5['n_tasks'])} of {n_tasks} tasks give unpredictable results.
- **It doesn't explain why tries differ.** The temperature was almost zero, and results still changed from try to try. Small differences probably add up over several turns, but this study didn't test the cause.
- **The gain is not claimed as statistically significant.** The confidence intervals describe the data. No formal significance claim is made.

## Why runs failed

Of {failures['total_runs']} runs, {failures['official_passes']} passed and {failures['official_failures']} failed. Each failure got a label from simple rules based on the saved logs and the grader's output:

{failure_lines}

These labels are rough. A run counts as a `code_error` if a Python error shows up anywhere in its log. Many of those runs probably hit an error, fixed it, and then ran out of turns. A better rule would separate those cases. The rules are in [`docs/failure_taxonomy.md`](docs/failure_taxonomy.md).

![Failure categories by turn condition](results/figures/figure_06_failure_categories.png)

## Cost

The whole study made {integrity['provider_request_count']} API calls and used {tokens['total_tokens']:,} tokens. It cost **${costs['total']:.2f}**: ${costs['by_condition']['turns_3']:.2f} for {turns3['max_turn']} turns and ${costs['by_condition']['turns_5']:.2f} for {turns5['max_turn']} turns, at the prices saved in [`configs/execution.yaml`](configs/execution.yaml).

## Limitations

- One model: `{execution['model_snapshot']}`.
- {n_tasks} tasks, not the whole DARE-Bench set, and only instruction-following classification and regression tasks.
- {repeats} tries per task, so a task's success rate can only be {score_steps}.
- The results might not hold for other models, agents, settings, or task types.
- The failure labels are rough, as explained above.
- The raw agent logs are too large for this repo. The run table and integrity report in `results/derived/` are the public record.

## Check the numbers yourself

You don't need an API key or Docker to check the results. Every table, figure, and this README are built from [`results/derived/runs.csv`](results/derived/runs.csv), which has one row per run. The tests fail if any published number stops matching the data. GitHub runs these checks on Linux and Windows after every change.

```
python -m pip install -r requirements-analysis.txt
python -X utf8 src/render_publication_docs.py --check
python -X utf8 -m unittest discover -s tests -v
```

To rebuild everything from the run table instead:

```
python src/reliability_metrics.py --runs results/derived/runs.csv --subset configs/task_subset.json --out-dir results/derived --expected-repeats {repeats} --bootstrap-samples {bootstrap['samples']} --seed {bootstrap['seed']}
python src/render_publication_docs.py
python src/generate_figures.py
```

The published SHA-256 hashes were taken on Windows. [`.gitattributes`](.gitattributes) makes Git check out those files with the same line endings on every system, so the hashes match anywhere.

Key files:

- [`configs/task_subset.json`](configs/task_subset.json): the {n_tasks} tasks and their full instructions.
- [`configs/execution.yaml`](configs/execution.yaml): the frozen model, settings, and prices.
- [`results/derived/runs.csv`](results/derived/runs.csv): one re-graded row per run.
- [`results/derived/raw_run_integrity.json`](results/derived/raw_run_integrity.json): proof that all {integrity['expected_identity_count']} runs are there, with no duplicates.
- [`results/derived/robustness_checks.json`](results/derived/robustness_checks.json): cross-checks of the numbers in this README.
- [`results/figures/manifest.json`](results/figures/manifest.json): which data each figure was drawn from.

To rerun the agent itself, you need Docker, an OpenAI API key, and the pinned DARE-Bench code. [`scripts/bootstrap.sh`](scripts/bootstrap.sh) or [`scripts/bootstrap.ps1`](scripts/bootstrap.ps1) sets that up.

## About DARE-Bench

[DARE-Bench](https://openreview.net/forum?id=eJV3JhJvZF) tests how well AI agents do data-science work, using tasks with answers that can be checked exactly. This project uses its tasks, reference answers, agent code, and official grader at version `{execution['dare_bench_commit']}`. It asks a narrower question: how consistent is one agent on the same tasks?

This is an independent analysis. It does not change or replace DARE-Bench, is not a new benchmark, and does not imply affiliation with Snowflake or the original authors.

## License

The code and docs in this repo are under the [Apache License 2.0](LICENSE). DARE-Bench has its own [license and dataset licenses](https://github.com/Snowflake-Labs/dare-bench#license). No DARE-Bench databases or source datasets are copied here.

## Citation

If you use this work, please cite the DARE-Bench paper too:

```bibtex
@inproceedings{{shu{ctx['citation']['references'][0]['year']}darebench,
  title     = {{DARE-Bench: Evaluating Modeling and Instruction Fidelity of LLMs in Data Science}},
  author    = {{Shu, Fan and Wang, Yite and Wu, Ruofan and Liu, Boyi and Yao, Zhewei and He, Yuxiong and Yan, Feng}},
  booktitle = {{International Conference on Learning Representations}},
  year      = {{{ctx['citation']['references'][0]['year']}}}
}}
```

Citation details for this repo are in [`CITATION.cff`](CITATION.cff).
"""


def render_release_notes(ctx: dict[str, Any]) -> str:
    execution = ctx["execution"]
    integrity = ctx["integrity"]
    t3 = ctx["condition_records"]["turns_3"]["overall"]
    t5 = ctx["condition_records"]["turns_5"]["overall"]
    return f"""# Phase 1 — DARE-Bench Agent Reliability Study

First public release of the independent study *Beyond Average Score: Repeatability of LLM Data-Science Agents on DARE-Bench*.

## Included

- Frozen {len(ctx['subset'])}-task Classification-IF and Regression-IF subset.
- {execution['repeats']} repetitions under {execution['conditions']['turns_3']['max_turn']}-turn and {execution['conditions']['turns_5']['max_turn']}-turn budgets, totaling {execution['planned_run_count']} immutable run identities.
- Independently rescored run table, raw-run integrity audit, reliability statistics, task-bootstrap confidence intervals, failure analysis, and publication figures.
- Deterministic documentation and figure generators with drift-detection tests.

## Headline result

Official exact-match success increased from {pct(t3['run_success_rate'])} to {pct(t5['run_success_rate'])}; flaky-task rate and pairwise disagreement also increased. The README reports the complete result with descriptive task-bootstrap intervals and limitations.

## Provenance

- Frozen execution commit: `{integrity['frozen_study_commit']}`
- Frozen execution tag: `{integrity['frozen_tag']}`
- DARE-Bench commit: `{execution['dare_bench_commit']}`
- Task subset SHA-256: `{ctx['subset_hash']}`
- Exact model snapshot: `{execution['model_snapshot']}`

This release does not redistribute upstream benchmark databases or source datasets. It is independent of Snowflake and the DARE-Bench authors.
"""


def render_summary(ctx: dict[str, Any]) -> str:
    condition = ctx["condition_records"]
    paired = ctx["paired_records"]
    integrity = ctx["integrity"]
    execution = ctx["execution"]
    failures = ctx["failure_summary"]
    bootstrap = ctx["summary"]["bootstrap"]
    t3 = condition["turns_3"]["overall"]
    t5 = condition["turns_5"]["overall"]
    class3 = condition["turns_3"]["classification"]
    class5 = condition["turns_5"]["classification"]
    reg3 = condition["turns_3"]["regression"]
    reg5 = condition["turns_5"]["regression"]
    overall_pair = paired["overall"]
    added = ctx["added_passes"]
    return f"""# Research Summary

## Background

Aggregate benchmark scores can conceal whether an agent reliably solves the same deterministic task. DARE-Bench supplies process-aware data-science tasks with verifiable outputs, making its instruction-following tasks suitable for separating average capability from repeatability. This project is an independent reliability study built on DARE-Bench, not a new benchmark and not an effort affiliated with Snowflake or the DARE-Bench authors.

## Research Question

When the same deterministic DARE-Bench instruction-following task is given repeatedly to the same LLM agent, how consistently does it succeed, and how does increasing the turn budget from three to five affect both average exact-match performance and run-to-run reliability?

## Method

The preregistered Phase-1 sample comprised {ctx['classification_count']} Classification-IF and {ctx['regression_count']} Regression-IF tasks selected with seed `{ctx['experiment']['benchmark']['task_selection_seed']}` from DARE-Bench revision `{execution['dare_bench_commit']}`. The frozen subset SHA-256 was `{ctx['subset_hash']}`. The agent used `{execution['model_snapshot']}` under two conditions: {execution['conditions']['turns_3']['max_turn']} or {execution['conditions']['turns_5']['max_turn']} maximum turns, with a {execution['conditions']['turns_3']['executor_timeout_seconds']}-second executor timeout in both. Five independent repetitions produced {execution['planned_run_count']} immutable run identities.

References were generated locally through the pinned DARE-Bench reference solution. Every produced prediction was scored, and then independently rescored, with the pinned official evaluator. Exact match was binary. Reliability outcomes were always pass, never pass, flaky, and pairwise disagreement across the {execution['repeats'] * (execution['repeats'] - 1) // 2} repeat pairs per task-condition. Percentile 95% confidence intervals resampled tasks using {bootstrap['samples']:,} bootstrap samples and seed `{bootstrap['seed']}`.

## Results

All {integrity['expected_identity_count']} expected identities were present with no missing or duplicate identities. Official exact-match success increased from {pct(t3['run_success_rate'])} ({whole_count(t3['run_success_rate'], t3['n_runs'])}/{int(t3['n_runs'])}) under three turns to {pct(t5['run_success_rate'])} ({whole_count(t5['run_success_rate'], t5['n_runs'])}/{int(t5['n_runs'])}) under five turns. The task-bootstrap 95% intervals were {ci(t3, 'mean_task_success_rate')} and {ci(t5, 'mean_task_success_rate')}, respectively. The paired mean task-level difference was {pp(overall_pair['mean_task_success_rate_difference'])} (95% CI: {pp(overall_pair['ci95_low'])} to {pp(overall_pair['ci95_high'])}); {int(overall_pair['tasks_improved'])} tasks improved, {int(overall_pair['tasks_declined'])} declined, and {int(overall_pair['tasks_tied'])} tied.

Classification success rose from {pct(class3['run_success_rate'])} to {pct(class5['run_success_rate'])}, while regression rose from {pct(reg3['run_success_rate'])} to {pct(reg5['run_success_rate'])}. Classification contributed {added['classification']} of {added['total']} additional passes ({pct(added['classification_share'])}); regression contributed {added['regression']}. This concentration is descriptive because each family contains only {ctx['classification_count']} tasks.

The repeatability results were mixed. Always-pass tasks increased from {pct(t3['always_pass_task_rate'])} to {pct(t5['always_pass_task_rate'])}, and never-pass tasks decreased from {pct(t3['never_pass_task_rate'])} to {pct(t5['never_pass_task_rate'])}. At the same time, flaky tasks increased from {pct(t3['flaky_task_rate'])} to {pct(t5['flaky_task_rate'])}, and mean pairwise disagreement increased from {pct(t3['mean_pairwise_disagreement'])} to {pct(t5['mean_pairwise_disagreement'])}.

Among {failures['official_failures']} official failures, mechanically supported labels were {failures['failure_category_counts']['code_error']} code errors, {failures['failure_category_counts']['wrong_prediction_unclassified']} wrong predictions without stronger causal evidence, {failures['failure_category_counts']['malformed_prediction']} malformed predictions, and {failures['failure_category_counts']['max_turn_or_token_limit']} turn/token-limit failure. Recorded API usage totaled {integrity['token_totals']['total_tokens']:,} tokens at ${integrity['actual_api_cost_usd']['total']:.6f}.

## Interpretation

Five turns produced more passes, including on tasks that never passed with three turns. The runs were not simply more stable: flakiness and pairwise disagreement both rose. On this fixed sample, average success and repeatability moved in different directions, so the mean score alone would miss part of the result.

## Limitations

The study covers one model (`{execution['model_snapshot']}`), {len(ctx['subset'])} fixed IF tasks rather than the entire benchmark, and {execution['repeats']} repetitions per condition. It includes Classification-IF and Regression-IF only. Results should not automatically generalize to other models, agent architectures, decoding configurations, execution environments, or DARE-Bench task families. Bootstrap intervals are descriptive, and no formal significance claim is made. Failure categories are limited to what preserved logs and evaluator diagnostics mechanically support.

## Reproducibility

The execution protocol was frozen at commit `{integrity['frozen_study_commit']}` and tag `{integrity['frozen_tag']}` before the final runs. The committed integrity report verifies {integrity['unique_recorded_identity_count']} unique identities, {integrity['prediction_hash_verified_count']} prediction hashes, exact returned model IDs for {integrity['provider_request_count']} provider responses, per-run token and cost artifacts, and {integrity['independent_official_rescore']['row_count']} independent official rescores with no exceptions. Statistical outputs are regenerated from `results/derived/runs.csv`; publication figures read only `results/derived/`. Documentation rendering and tests fail if committed metrics drift from recomputed values.
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Fail if rendered files differ")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    ctx = load_context()
    report_text = json.dumps(ctx["robustness"], indent=2, allow_nan=False) + "\n"
    rendered = {
        ROBUSTNESS_PATH: report_text,
        README_PATH: render_readme(ctx),
        SUMMARY_PATH: render_summary(ctx),
        RELEASE_NOTES_PATH: render_release_notes(ctx),
    }
    if args.check:
        drift = [path for path, text in rendered.items() if not path.is_file() or path.read_text(encoding="utf-8") != text]
        if drift:
            raise SystemExit("Publication artifacts are stale: " + ", ".join(str(path) for path in drift))
        print("Publication documentation and robustness report match validated artifacts")
        return 0

    for path, text in rendered.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8", newline="\n")
    print("Rendered README, research summary, and robustness checks from validated artifacts")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
