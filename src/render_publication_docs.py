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
    added = ctx["added_passes"]
    costs = integrity["actual_api_cost_usd"]
    tokens = integrity["token_totals"]
    env_packages = environment["sandbox"]["packages"]

    failure_lines = "\n".join(
        f"- `{name}`: **{count}**"
        for name, count in sorted(
            failures["failure_category_counts"].items(), key=lambda item: (-item[1], item[0])
        )
    )

    return f"""# Beyond Average Score: Repeatability of LLM Data-Science Agents on DARE-Bench

An independent reliability study built on [DARE-Bench](https://github.com/Snowflake-Labs/dare-bench). This repository is not a new benchmark, is not affiliated with Snowflake, and is not produced or endorsed by the DARE-Bench authors.

> **Research question:** When the same deterministic DARE-Bench instruction-following task is given to the same LLM agent repeatedly, how consistently does it succeed?

## Key result

Increasing the agent turn budget from three to five raised mean official exact-match success from **{pct(t3['run_success_rate'])}** to **{pct(t5['run_success_rate'])}** on this fixed {len(ctx['subset'])}-task subset. At the same time, flaky tasks increased from **{pct(t3['flaky_task_rate'])}** to **{pct(t5['flaky_task_rate'])}**, and mean pairwise disagreement rose from **{pct(t3['mean_pairwise_disagreement'])}** to **{pct(t5['mean_pairwise_disagreement'])}**. Additional turns improved average capability without uniformly improving repeatability.

![Success and variability on a shared percentage scale](results/figures/figure_07_main_result.png)

The paired task-level mean difference was **{pp(overall_pair['mean_task_success_rate_difference'])}** (task-bootstrap 95% CI: {pp(overall_pair['ci95_low'])} to {pp(overall_pair['ci95_high'])}). This interval is descriptive; no formal significance claim is made.

## Study design

```mermaid
flowchart LR
    A["Frozen DARE-Bench IF subset<br/>{ctx['classification_count']} classification + {ctx['regression_count']} regression"] --> B{{"Turn budget"}}
    B --> C["3 turns<br/>{execution['conditions']['turns_3']['executor_timeout_seconds']} s executor timeout"]
    B --> D["5 turns<br/>{execution['conditions']['turns_5']['executor_timeout_seconds']} s executor timeout"]
    C --> E["{execution['repeats']} independent repeats per task"]
    D --> E
    E --> F["{execution['planned_run_count']} immutable run identities"]
    F --> G["Pinned official evaluator<br/>exact match"]
    G --> H["Task-level reliability<br/>task-bootstrap intervals"]
```

| Design element | Frozen value |
|---|---|
| DARE-Bench revision | `{execution['dare_bench_commit']}` |
| Execution freeze | `{integrity['frozen_study_commit']}` / `{integrity['frozen_tag']}` |
| Task variant | `question_{execution['question_version']}` (IF) |
| Task sample | {ctx['classification_count']} Classification-IF + {ctx['regression_count']} Regression-IF |
| Selection seed | `{experiment['benchmark']['task_selection_seed']}` |
| Subset SHA-256 | `{ctx['subset_hash']}` |
| Model snapshot | `{execution['model_snapshot']}` |
| Provider/API | `{execution['provider']}` / `{execution['api_mode']}` |
| Agent conditions | {execution['conditions']['turns_3']['max_turn']} turns and {execution['conditions']['turns_5']['max_turn']} turns; {execution['conditions']['turns_3']['executor_timeout_seconds']} s executor timeout in both |
| Repetitions | {execution['repeats']} per task-condition |
| Planned and recorded runs | {len(ctx['subset'])} × {execution['repeats']} × {len(execution['conditions'])} = {execution['planned_run_count']} |

The decoding and tool configuration was frozen before final execution: temperature `{execution['temperature']}`, top-p `{execution['top_p']}`, maximum output tokens `{execution['max_tokens']}`, provider-default automatic tool choice, and function-call tool mode. Provider SDK retries remained at `{execution['provider_sdk_request_max_retries']}`; failed or incomplete study identities were never silently replaced.

## Method

### Tasks and agent

The sample was selected once from the pinned DARE-Bench evaluation release using seed `{experiment['benchmark']['task_selection_seed']}`. Only `question_{execution['question_version']}` Classification-IF and Regression-IF tasks were included. Each task was executed five times under each turn budget using `{execution['model_snapshot']}` with the same prompt variant, sandbox, timeout, and decoding configuration.

The code sandbox was `{execution['sandbox_image']}` at digest `{execution['sandbox_image_digest']}`. Its relevant runtime was Python {env_packages['python']}, pandas {env_packages['pandas']}, NumPy {env_packages['numpy']}, scikit-learn {env_packages['scikit_learn']}, and pyarrow {env_packages['pyarrow']}. Captured host and container provenance is in [`results/environment/environment.json`](results/environment/environment.json). That manifest was recorded during preflight, so its embedded study-Git subsection predates the final execution freeze; the freeze commit and raw-run integrity report are authoritative for the code revision used by the final runs.

### Official evaluation

Reference outputs were generated locally with the pinned DARE-Bench `reference_solution.py` in the agent-compatible sandbox environment. Every produced `prediction.csv` was independently rescored with the pinned official `evaluation.py`. IF exact match is binary: a run passes only when the official final score is `1.0`. The integrity audit found {integrity['expected_identity_count']} expected identities, {integrity['unique_recorded_identity_count']} unique recorded identities, no missing or duplicate identities, {integrity['prediction_file_count']} produced prediction files, and {integrity['prediction_hash_verified_count']} verified prediction hashes. Runs that completed without producing a valid passing prediction remain failures.

### Reliability measures

For each task-condition, `always pass` means {execution['repeats']} passes, `never pass` means zero passes, and `flaky` means a mixture of pass and fail outcomes. Pairwise disagreement is the fraction of the {execution['repeats'] * (execution['repeats'] - 1) // 2} unordered repeat pairs with different binary outcomes. Confidence intervals resample tasks, not runs: {bootstrap['samples']:,} bootstrap resamples, seed `{bootstrap['seed']}`, and percentile {bootstrap['interval'].split('_')[1]} intervals.

## Results

### Official exact-match success

| Condition | Overall | Classification-IF | Regression-IF |
|---|---:|---:|---:|
| 3 turns | **{pct(t3['run_success_rate'])}** ({whole_count(t3['run_success_rate'], t3['n_runs'])}/{int(t3['n_runs'])}); 95% CI {ci(t3, 'mean_task_success_rate')} | {pct(class3['run_success_rate'])} ({whole_count(class3['run_success_rate'], class3['n_runs'])}/{int(class3['n_runs'])}); CI {ci(class3, 'mean_task_success_rate')} | {pct(reg3['run_success_rate'])} ({whole_count(reg3['run_success_rate'], reg3['n_runs'])}/{int(reg3['n_runs'])}); CI {ci(reg3, 'mean_task_success_rate')} |
| 5 turns | **{pct(t5['run_success_rate'])}** ({whole_count(t5['run_success_rate'], t5['n_runs'])}/{int(t5['n_runs'])}); 95% CI {ci(t5, 'mean_task_success_rate')} | {pct(class5['run_success_rate'])} ({whole_count(class5['run_success_rate'], class5['n_runs'])}/{int(class5['n_runs'])}); CI {ci(class5, 'mean_task_success_rate')} | {pct(reg5['run_success_rate'])} ({whole_count(reg5['run_success_rate'], reg5['n_runs'])}/{int(reg5['n_runs'])}); CI {ci(reg5, 'mean_task_success_rate')} |

![Overall exact-match success with task-bootstrap intervals](results/figures/figure_01_success_rate_ci.png)

Classification improved by **{pp(class_pair['mean_task_success_rate_difference'])}** (descriptive 95% CI: {pp(class_pair['ci95_low'])} to {pp(class_pair['ci95_high'])}); regression improved by **{pp(reg_pair['mean_task_success_rate_difference'])}** (CI: {pp(reg_pair['ci95_low'], sign=False)} to {pp(reg_pair['ci95_high'])}). Classification contributed {added['classification']} of the {added['total']} additional exact-match passes ({pct(added['classification_share'])}), compared with {added['regression']} additional regression passes. With only {ctx['classification_count']} tasks per family, this is a descriptive concentration, not evidence of a general task-family interaction.

![Classification and regression success by condition](results/figures/figure_02_task_type_success.png)

### Repeatability

| Condition | Always-pass tasks | Flaky tasks | Never-pass tasks | Mean pairwise disagreement |
|---|---:|---:|---:|---:|
| 3 turns | {whole_count(t3['always_pass_task_rate'], t3['n_tasks'])}/{int(t3['n_tasks'])} ({pct(t3['always_pass_task_rate'])}) | {whole_count(t3['flaky_task_rate'], t3['n_tasks'])}/{int(t3['n_tasks'])} ({pct(t3['flaky_task_rate'])}) | {whole_count(t3['never_pass_task_rate'], t3['n_tasks'])}/{int(t3['n_tasks'])} ({pct(t3['never_pass_task_rate'])}) | {pct(t3['mean_pairwise_disagreement'])} |
| 5 turns | {whole_count(t5['always_pass_task_rate'], t5['n_tasks'])}/{int(t5['n_tasks'])} ({pct(t5['always_pass_task_rate'])}) | {whole_count(t5['flaky_task_rate'], t5['n_tasks'])}/{int(t5['n_tasks'])} ({pct(t5['flaky_task_rate'])}) | {whole_count(t5['never_pass_task_rate'], t5['n_tasks'])}/{int(t5['n_tasks'])} ({pct(t5['never_pass_task_rate'])}) | {pct(t5['mean_pairwise_disagreement'])} |

![Always-pass, flaky, and never-pass composition](results/figures/figure_03_reliability_composition.png)

![Pairwise disagreement by condition](results/figures/figure_04_pairwise_disagreement.png)

### Paired task movements

Across the {int(overall_pair['n_paired_tasks'])} tasks, {int(overall_pair['tasks_improved'])} improved, {int(overall_pair['tasks_declined'])} declined, and {int(overall_pair['tasks_tied'])} tied. The median task-level change was {pp(overall_pair['median_task_success_rate_difference'], sign=False)} because more than half the tasks tied. Individual movements ranged from {pp(float(ctx['movement']['difference'].min()), 0, sign=False)} to {pp(float(ctx['movement']['difference'].max()), 0)}. These extremes illustrate heterogeneity and are not treated as representative cases.

![Paired movement of all fixed tasks](results/figures/figure_05_task_movements.png)

<details>
<summary>All paired task movements, grouped by outcome</summary>

#### Improved ({len(movements['improved'])})

{task_list(movements['improved'])}

#### Declined ({len(movements['declined'])})

{task_list(movements['declined'])}

#### Tied ({len(movements['tied'])})

{task_list(movements['tied'])}

</details>

### Failure analysis

Of {failures['total_runs']} runs, {failures['official_passes']} passed and {failures['official_failures']} failed official exact match. Failure labels were assigned only when mechanically supported by preserved evaluator or log evidence:

{failure_lines}

`wrong_prediction_unclassified` is the conservative fallback when a structurally valid prediction scored zero and no stronger cause was demonstrated. The two malformed predictions were official evaluator row-count mismatch diagnostics. Category counts should be read as an evidence-based diagnostic description, not a causal decomposition of model behavior.

![Failure categories by turn condition](results/figures/figure_06_failure_categories.png)

## API usage and cost

The {integrity['provider_request_count']} recorded provider requests used {tokens['total_tokens']:,} tokens: {tokens['prompt_tokens']:,} prompt tokens, including {tokens['cached_prompt_tokens']:,} cached prompt tokens, and {tokens['completion_tokens']:,} completion tokens. Recorded API cost was **${costs['total']:.6f}**: ${costs['by_condition']['turns_3']:.6f} for three turns and ${costs['by_condition']['turns_5']:.6f} for five turns. Cost is a study-execution measurement under the frozen pricing metadata, not a general cost estimate.

## Reproducibility

The public, lightweight provenance chain is:

- [`configs/task_subset.json`](configs/task_subset.json): frozen task identities and questions;
- [`configs/execution.yaml`](configs/execution.yaml): frozen provider, model, decoding, tool, timeout, sandbox, and pricing configuration;
- [`results/derived/runs.csv`](results/derived/runs.csv): one independently rescored row per immutable identity;
- [`results/derived/raw_run_integrity.json`](results/derived/raw_run_integrity.json): identity, hash, token, cost, model-ID, and rescore checks;
- [`results/derived/robustness_checks.json`](results/derived/robustness_checks.json): documentation claim cross-checks and all paired task movements;
- [`results/figures/manifest.json`](results/figures/manifest.json): figure inputs, source hashes, and generator version.

Raw provider traces and run directories remain preserved in the study archive under `results/raw/` and are intentionally excluded from Git by the repository policy. They were not altered during publication preparation.

To regenerate the statistical tables, robustness checks, figures, and documentation from the committed run table:

```powershell
python -m pip install -r requirements-analysis.txt

python src/reliability_metrics.py `
  --runs results/derived/runs.csv `
  --subset configs/task_subset.json `
  --out-dir results/derived `
  --expected-repeats {execution['repeats']} `
  --bootstrap-samples {bootstrap['samples']} `
  --seed {bootstrap['seed']}

python src/render_publication_docs.py
python src/generate_figures.py
python src/render_publication_docs.py --check
python -X utf8 -m unittest discover -s tests -v
```

The explicit UTF-8 mode is needed on Windows because the pinned upstream loader otherwise inherits the system text encoding for a UTF-8 task file. The benchmark workspace and pinned upstream code can be restored with `scripts/bootstrap.ps1` or `scripts/bootstrap.sh`. Re-running the paid benchmark is neither required nor performed by the publication pipeline.

## Limitations

- The study evaluates one model snapshot: `{execution['model_snapshot']}`.
- The sample contains {len(ctx['subset'])} fixed IF tasks, not the complete DARE-Bench evaluation set.
- Each task-condition has {execution['repeats']} repetitions, limiting precision for task-specific reliability.
- Only Classification-IF and Regression-IF are included; modeling and time-series families are outside scope.
- Results should not automatically generalize to other models, agent architectures, decoding configurations, execution environments, or DARE-Bench task families.
- Bootstrap intervals are descriptive task-resampling intervals. No formal significance claim is made.
- Failure taxonomy assignment is mechanical and evidence-limited. `wrong_prediction_unclassified` deliberately avoids unsupported causal attribution.
- Raw traces are locally preserved but are not part of the lightweight Git history, so the committed integrity and run tables are the public audit layer.

## Relationship to DARE-Bench

[DARE-Bench](https://openreview.net/forum?id=eJV3JhJvZF) evaluates modeling and instruction fidelity for LLM data-science agents using verifiable ground truth. This project uses its released tasks, reference-generation path, agent implementation, and official evaluator at pinned revision `{execution['dare_bench_commit']}`. The contribution here is narrower: repeated execution of a preregistered subset to characterize run-to-run reliability under two turn budgets.

This is an independent analysis. It does not modify or supersede DARE-Bench, does not constitute a new benchmark, and does not imply affiliation with Snowflake or the original authors.

### Citation

Please cite the original DARE-Bench paper alongside this repository:

```bibtex
@inproceedings{{shu{ctx['citation']['references'][0]['year']}darebench,
  title     = {{DARE-Bench: Evaluating Modeling and Instruction Fidelity of LLMs in Data Science}},
  author    = {{Shu, Fan and Wang, Yite and Wu, Ruofan and Liu, Boyi and Yao, Zhewei and He, Yuxiong and Yan, Feng}},
  booktitle = {{International Conference on Learning Representations}},
  year      = {{{ctx['citation']['references'][0]['year']}}}
}}
```

Repository citation metadata is provided in [`CITATION.cff`](CITATION.cff).
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

Reliability did not move uniformly with average performance. Always-pass tasks increased from {pct(t3['always_pass_task_rate'])} to {pct(t5['always_pass_task_rate'])}, and never-pass tasks decreased from {pct(t3['never_pass_task_rate'])} to {pct(t5['never_pass_task_rate'])}. However, flaky tasks increased from {pct(t3['flaky_task_rate'])} to {pct(t5['flaky_task_rate'])}, while mean pairwise disagreement increased from {pct(t3['mean_pairwise_disagreement'])} to {pct(t5['mean_pairwise_disagreement'])}.

Among {failures['official_failures']} official failures, mechanically supported labels were {failures['failure_category_counts']['code_error']} code errors, {failures['failure_category_counts']['wrong_prediction_unclassified']} wrong predictions without stronger causal evidence, {failures['failure_category_counts']['malformed_prediction']} malformed predictions, and {failures['failure_category_counts']['max_turn_or_token_limit']} turn/token-limit failure. Recorded API usage totaled {integrity['token_totals']['total_tokens']:,} tokens at ${integrity['actual_api_cost_usd']['total']:.6f}.

## Interpretation

Increasing the turn budget improved mean capability and converted some never-pass tasks into successful or intermittently successful tasks. It did not make outcomes uniformly more stable. The simultaneous increase in flaky-task prevalence and pairwise disagreement indicates that a higher success rate can coexist with lower repeatability among tasks near the agent's capability boundary. Average score and reliability therefore answer different evaluation questions and should be reported together.

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
