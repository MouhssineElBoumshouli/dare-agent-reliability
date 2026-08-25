"""Generate deterministic publication figures exclusively from results/derived."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from publication_data import load_publication_data, sha256_file  # noqa: E402


BLUE = "#0077BB"
ORANGE = "#EE7733"
TEAL = "#009988"
RED = "#CC3311"
GREY = "#BBBBBB"
DARK_GREY = "#555555"
LIGHT_GREY = "#E6E6E6"
CONDITION_COLORS = {"turns_3": BLUE, "turns_5": ORANGE}
CONDITION_LABELS = {"turns_3": "3 turns", "turns_5": "5 turns"}


plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 9,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


def percent_axis(ax: plt.Axes, maximum: float = 1.0) -> None:
    ax.set_ylim(0, maximum)
    ticks = np.arange(0, maximum + 0.001, 0.2 if maximum > 0.7 else 0.1)
    ax.set_yticks(ticks)
    ax.set_yticklabels([f"{value:.0%}" for value in ticks])
    ax.set_ylabel("Rate")
    ax.grid(axis="y", color=LIGHT_GREY, linewidth=0.7, zorder=0)


def add_value_labels(ax: plt.Axes, bars, values: list[float], pad: float = 0.018) -> None:
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + pad,
            f"{value:.1%}",
            ha="center",
            va="bottom",
            fontsize=8,
            fontweight="bold",
        )


def save_figure(fig: plt.Figure, out_dir: Path, stem: str) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    outputs = [out_dir / f"{stem}.png", out_dir / f"{stem}.pdf"]
    fig.savefig(outputs[0], metadata={"Software": f"matplotlib {matplotlib.__version__}"})
    fig.savefig(
        outputs[1],
        metadata={
            "Creator": f"matplotlib {matplotlib.__version__}",
            "CreationDate": None,
            "ModDate": None,
        },
    )
    plt.close(fig)
    return outputs


def figure_success_ci(data: dict, out_dir: Path) -> list[Path]:
    overall = data["conditions"][data["conditions"]["stratum"] == "overall"].set_index(
        "condition"
    )
    order = ["turns_3", "turns_5"]
    values = overall.loc[order, "run_success_rate"].to_numpy(float)
    low = overall.loc[order, "mean_task_success_rate_ci95_low"].to_numpy(float)
    high = overall.loc[order, "mean_task_success_rate_ci95_high"].to_numpy(float)
    errors = np.vstack([values - low, high - values])
    fig, ax = plt.subplots(figsize=(6.9, 4.3))
    bars = ax.bar(
        range(2), values, yerr=errors, capsize=5,
        color=[CONDITION_COLORS[c] for c in order], edgecolor="black", linewidth=0.6, zorder=3,
    )
    ax.set_xticks(range(2), [CONDITION_LABELS[c] for c in order])
    percent_axis(ax, 0.8)
    ax.set_title("Official IF exact-match success by turn budget", loc="left", fontweight="bold")
    ax.text(0.99, 0.94, "24 tasks × 5 repeats per condition", transform=ax.transAxes,
            ha="right", va="top", color=DARK_GREY, fontsize=8)
    add_value_labels(ax, bars, values, 0.025)
    fig.tight_layout()
    return save_figure(fig, out_dir, "figure_01_success_rate_ci")


def figure_task_type(data: dict, out_dir: Path) -> list[Path]:
    frame = data["conditions"][data["conditions"]["stratum"].isin(["classification", "regression"])]
    strata = ["classification", "regression"]
    conditions = ["turns_3", "turns_5"]
    x = np.arange(2)
    width = 0.34
    fig, ax = plt.subplots(figsize=(6.9, 4.5))
    for offset, condition in zip((-width / 2, width / 2), conditions):
        subset = frame[frame["condition"] == condition].set_index("stratum").loc[strata]
        values = subset["run_success_rate"].to_numpy(float)
        low = subset["mean_task_success_rate_ci95_low"].to_numpy(float)
        high = subset["mean_task_success_rate_ci95_high"].to_numpy(float)
        bars = ax.bar(
            x + offset, values, width,
            yerr=np.vstack([values - low, high - values]), capsize=4,
            label=CONDITION_LABELS[condition], color=CONDITION_COLORS[condition],
            edgecolor="black", linewidth=0.5, zorder=3,
        )
        add_value_labels(ax, bars, values, 0.025)
    ax.set_xticks(x, ["Classification-IF", "Regression-IF"])
    percent_axis(ax, 1.0)
    ax.set_title("Exact-match success by task family", loc="left", fontweight="bold")
    ax.legend(frameon=False, ncol=2, loc="upper right")
    fig.tight_layout()
    return save_figure(fig, out_dir, "figure_02_task_type_success")


def figure_composition(data: dict, out_dir: Path) -> list[Path]:
    overall = data["conditions"][data["conditions"]["stratum"] == "overall"].set_index(
        "condition"
    )
    order = ["turns_3", "turns_5"]
    components = [
        ("always_pass_task_rate", "Always pass", TEAL, ""),
        ("flaky_task_rate", "Flaky", ORANGE, "///"),
        ("never_pass_task_rate", "Never pass", GREY, ".."),
    ]
    fig, ax = plt.subplots(figsize=(6.9, 4.2))
    left = np.zeros(2)
    for column, label, color, hatch in components:
        values = overall.loc[order, column].to_numpy(float)
        bars = ax.barh(
            range(2), values, left=left, label=label, color=color,
            hatch=hatch, edgecolor="white", linewidth=0.8,
        )
        for bar, value, start in zip(bars, values, left):
            if value >= 0.11:
                ax.text(start + value / 2, bar.get_y() + bar.get_height() / 2,
                        f"{value:.1%}", ha="center", va="center", fontsize=8,
                        color="black", fontweight="bold")
        left += values
    ax.set_yticks(range(2), [CONDITION_LABELS[c] for c in order])
    ax.set_xlim(0, 1)
    ax.set_xticks(np.arange(0, 1.01, 0.2), [f"{v:.0%}" for v in np.arange(0, 1.01, 0.2)])
    ax.set_xlabel("Share of 24 tasks")
    ax.set_title("Task reliability composition", loc="left", fontweight="bold")
    ax.legend(frameon=False, ncol=3, loc="lower center", bbox_to_anchor=(0.5, -0.28))
    ax.invert_yaxis()
    fig.tight_layout()
    return save_figure(fig, out_dir, "figure_03_reliability_composition")


def figure_disagreement(data: dict, out_dir: Path) -> list[Path]:
    overall = data["conditions"][data["conditions"]["stratum"] == "overall"].set_index(
        "condition"
    )
    order = ["turns_3", "turns_5"]
    values = overall.loc[order, "mean_pairwise_disagreement"].to_numpy(float)
    low = overall.loc[order, "mean_pairwise_disagreement_ci95_low"].to_numpy(float)
    high = overall.loc[order, "mean_pairwise_disagreement_ci95_high"].to_numpy(float)
    fig, ax = plt.subplots(figsize=(6.9, 4.2))
    bars = ax.bar(
        range(2), values, yerr=np.vstack([values - low, high - values]), capsize=5,
        color=[CONDITION_COLORS[c] for c in order], edgecolor="black", linewidth=0.6, zorder=3,
    )
    ax.set_xticks(range(2), [CONDITION_LABELS[c] for c in order])
    percent_axis(ax, 0.5)
    ax.set_title("Mean pairwise disagreement across repeats", loc="left", fontweight="bold")
    add_value_labels(ax, bars, values, 0.018)
    fig.tight_layout()
    return save_figure(fig, out_dir, "figure_04_pairwise_disagreement")


def _short_task(task_id: str, task_type: str) -> str:
    core = task_id.rsplit("_", 1)[0]
    if "_" in core:
        core = core.split("_", 1)[1]
    core = core.replace("-", " ")
    if len(core) > 43:
        core = core[:40].rstrip() + "…"
    return f"{'C' if task_type == 'classification' else 'R'} · {core}"


def figure_task_movements(data: dict, out_dir: Path) -> list[Path]:
    frame = data["movement"].copy().sort_values(
        ["difference", "task_type", "task_id"], ascending=[True, True, True]
    )
    y = np.arange(len(frame))
    movement_color = {"improved": TEAL, "declined": RED, "tied": GREY}
    fig, ax = plt.subplots(figsize=(9.0, 9.2))
    for index, row in enumerate(frame.itertuples(index=False)):
        color = movement_color[row.movement]
        ax.plot([row.turns_3, row.turns_5], [index, index], color=color,
                linewidth=1.8 if row.movement != "tied" else 1.0, alpha=0.85, zorder=1)
    ax.scatter(frame["turns_3"], y, s=35, color=BLUE, marker="o", edgecolor="black",
               linewidth=0.4, label="3 turns", zorder=3)
    ax.scatter(frame["turns_5"], y, s=44, facecolor="none", marker="s", edgecolor=ORANGE,
               linewidth=1.5, label="5 turns", zorder=4)
    for index, row in enumerate(frame.itertuples(index=False)):
        ax.text(1.035, index, f"{row.difference:+.0%}", va="center", ha="left",
                color=movement_color[row.movement], fontsize=7.5, fontweight="bold")
    ax.set_yticks(y, [_short_task(row.task_id, row.task_type) for row in frame.itertuples(index=False)])
    ax.set_xlim(-0.02, 1.11)
    ax.set_xticks(np.arange(0, 1.01, 0.2), [f"{v:.0%}" for v in np.arange(0, 1.01, 0.2)])
    ax.set_xlabel("Exact-match success across five repeats")
    ax.set_title("Per-task movement from three to five turns", loc="left", fontweight="bold")
    ax.text(1.035, len(frame) - 0.15, "Δ", ha="left", va="bottom", fontsize=8,
            color=DARK_GREY, fontweight="bold")
    ax.grid(axis="x", color=LIGHT_GREY, linewidth=0.7, zorder=0)
    ax.legend(frameon=False, ncol=2, loc="lower right", bbox_to_anchor=(1.0, 1.002))
    fig.text(0.21, 0.012, "C = Classification-IF; R = Regression-IF. Lines show all 24 paired tasks.",
             fontsize=8, color=DARK_GREY)
    fig.tight_layout(rect=(0, 0.035, 1, 1))
    return save_figure(fig, out_dir, "figure_05_task_movements")


def figure_failures(data: dict, out_dir: Path) -> list[Path]:
    frame = data["failure_modes"].groupby(["condition", "failure_category"])["count"].sum().unstack(
        fill_value=0
    )
    order = ["turns_3", "turns_5"]
    categories = [
        "code_error", "wrong_prediction_unclassified", "malformed_prediction",
        "max_turn_or_token_limit",
    ]
    labels = ["Code error", "Wrong prediction\n(unclassified)", "Malformed\nprediction", "Turn/token\nlimit"]
    colors = [RED, BLUE, ORANGE, GREY]
    x = np.arange(len(categories))
    width = 0.35
    fig, ax = plt.subplots(figsize=(7.4, 4.5))
    for offset, condition in zip((-width / 2, width / 2), order):
        values = [int(frame.loc[condition].get(category, 0)) for category in categories]
        bars = ax.bar(
            x + offset, values, width, label=CONDITION_LABELS[condition],
            color=CONDITION_COLORS[condition], edgecolor="black", linewidth=0.5,
            hatch="" if condition == "turns_3" else "//", zorder=3,
        )
        for bar, value in zip(bars, values):
            if value:
                ax.text(bar.get_x() + bar.get_width() / 2, value + 0.8, str(value),
                        ha="center", va="bottom", fontsize=8, fontweight="bold")
    ax.set_xticks(x, labels)
    ax.set_ylabel("Failed runs")
    ax.set_ylim(0, 72)
    ax.grid(axis="y", color=LIGHT_GREY, linewidth=0.7, zorder=0)
    ax.set_title("Mechanically supported failure categories", loc="left", fontweight="bold")
    ax.legend(frameon=False, ncol=2, loc="upper right")
    ax.text(0.99, 0.88, "101 passes; 139 failures", transform=ax.transAxes,
            ha="right", va="top", color=DARK_GREY, fontsize=8)
    fig.tight_layout()
    return save_figure(fig, out_dir, "figure_06_failure_categories")


def figure_main_result(data: dict, out_dir: Path) -> list[Path]:
    overall = data["conditions"][data["conditions"]["stratum"] == "overall"].set_index(
        "condition"
    )
    order = ["turns_3", "turns_5"]
    x = np.arange(2)
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.4), sharey=True)

    success = overall.loc[order, "run_success_rate"].to_numpy(float)
    bars = axes[0].bar(x, success, color=[CONDITION_COLORS[c] for c in order],
                       edgecolor="black", linewidth=0.6, zorder=3)
    axes[0].set_xticks(x, [CONDITION_LABELS[c] for c in order])
    axes[0].set_title("A  Average capability", loc="left", fontweight="bold")
    axes[0].set_ylabel("Rate")
    add_value_labels(axes[0], bars, success, 0.018)

    metrics = [
        ("flaky_task_rate", "Flaky tasks", "o", ORANGE),
        ("mean_pairwise_disagreement", "Pairwise disagreement", "s", RED),
    ]
    for column, label, marker, color in metrics:
        values = overall.loc[order, column].to_numpy(float)
        axes[1].plot(x, values, marker=marker, color=color, linewidth=2.0,
                     markersize=7, label=label, zorder=3)
        for xpos, value in zip(x, values):
            axes[1].text(xpos, value + 0.018, f"{value:.1%}", ha="center", va="bottom",
                         fontsize=8, fontweight="bold", color=color)
    axes[1].set_xticks(x, [CONDITION_LABELS[c] for c in order])
    axes[1].set_title("B  Run-to-run variability", loc="left", fontweight="bold")
    axes[1].legend(frameon=False, loc="upper left")

    for ax in axes:
        ax.set_ylim(0, 0.65)
        ax.set_yticks(np.arange(0, 0.61, 0.1), [f"{v:.0%}" for v in np.arange(0, 0.61, 0.1)])
        ax.grid(axis="y", color=LIGHT_GREY, linewidth=0.7, zorder=0)
    fig.suptitle("More turns improved success but increased measured variability",
                 x=0.06, ha="left", fontsize=13, fontweight="bold")
    fig.text(
        0.06, 0.005,
        "Shared percentage scale; no dual axis. Success is run-level; flaky-task rate and disagreement are task-level.",
        fontsize=8, color=DARK_GREY,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.94))
    return save_figure(fig, out_dir, "figure_07_main_result")


FIGURES: list[tuple[str, str, Callable[[dict, Path], list[Path]]]] = [
    ("figure_01_success_rate_ci", "Exact-match success with task-bootstrap 95% confidence intervals", figure_success_ci),
    ("figure_02_task_type_success", "Classification and regression success by condition", figure_task_type),
    ("figure_03_reliability_composition", "Always-pass, flaky, and never-pass task composition", figure_composition),
    ("figure_04_pairwise_disagreement", "Pairwise disagreement with task-bootstrap 95% confidence intervals", figure_disagreement),
    ("figure_05_task_movements", "Paired movement of all 24 tasks", figure_task_movements),
    ("figure_06_failure_categories", "Mechanically supported failure categories", figure_failures),
    ("figure_07_main_result", "Success and variability on a shared percentage scale", figure_main_result),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--derived-dir", type=Path, default=Path("results/derived"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/figures"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data = load_publication_data(args.derived_dir)
    outputs: list[Path] = []
    figure_records = []
    for stem, description, function in FIGURES:
        paths = function(data, args.output_dir)
        outputs.extend(paths)
        figure_records.append(
            {
                "id": stem,
                "description": description,
                "files": {path.suffix.lstrip("."): path.as_posix() for path in paths},
            }
        )

    manifest = {
        "schema_version": 1,
        "generator": "src/generate_figures.py",
        "matplotlib_version": matplotlib.__version__,
        "input_directory": args.derived_dir.as_posix(),
        "source_sha256": data["source_hashes"],
        "figures": figure_records,
    }
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Generated {len(FIGURES)} figures in PNG and PDF from {args.derived_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
