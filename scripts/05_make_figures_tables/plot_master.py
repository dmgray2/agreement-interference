#!/usr/bin/env python3
"""
Make paper figures from the master score file.

Main input:
    data/scores/master_score_lists/master_scores_full.csv

Optional input:
    data/scores/summaries/stats_summary.csv

Output folder:
    paper/figures/

Run from the repository root:

    python3 scripts/05_make_figures_tables/plot_master.py

The figures use critical-region accuracy (`correct_crit`). The V2 scaling
figure uses `stats_summary.csv` when available, so its confidence intervals
match the main reported bootstrap intervals. COPULA scaling
confidence intervals are computed inside this script by resampling one COPULA item per sentence family.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_MASTER = (
    REPO_ROOT
    / "data"
    / "scores"
    / "master_score_lists"
    / "master_scores_full.csv"
)

DEFAULT_STATS = (
    REPO_ROOT
    / "data"
    / "scores"
    / "summaries"
    / "stats_summary.csv"
)

DEFAULT_OUTDIR = REPO_ROOT / "paper" / "figures"

MODEL_ORDER = [
    "gpt2",
    "pythia410",
    "llama32_1b",
    "qwen25_1.5b",
    "qwen25_3b",
    "llama32_3b",
    "qwen25_7b",
    "qwen25_14b",
    "qwen25_32b",
    "llama31_70b",
    "qwen25_72b",
]

MODEL_LABELS = {
    "gpt2": "GPT-2",
    "pythia410": "Pythia-410M",
    "llama32_1b": "Llama-3.2-1B",
    "qwen25_1.5b": "Qwen2.5-1.5B",
    "qwen25_3b": "Qwen2.5-3B",
    "llama32_3b": "Llama-3.2-3B",
    "qwen25_7b": "Qwen2.5-7B",
    "qwen25_14b": "Qwen2.5-14B",
    "qwen25_32b": "Qwen2.5-32B",
    "llama31_70b": "Llama-3.1-70B",
    "qwen25_72b": "Qwen2.5-72B",
}

# Approximate model sizes in millions of parameters. These values are used
# only to place points on the log-scale scatter plots.
MODEL_SIZES_M = {
    "gpt2": 124,
    "pythia410": 410,
    "llama32_1b": 1230,
    "qwen25_1.5b": 1500,
    "qwen25_3b": 3000,
    "llama32_3b": 3210,
    "qwen25_7b": 7000,
    "qwen25_14b": 14000,
    "qwen25_32b": 32000,
    "llama31_70b": 70000,
    "qwen25_72b": 72000,
}

PATTERN_ORDER = ["SSS", "PPP", "SSP", "PPS", "PSS", "SPP", "SPS", "PSP"]

SUB = {
    "1": "\u2081",
    "2": "\u2082",
    "3": "\u2083",
    "s": "\u209b",
    "p": "\u209a",
}

REQUIRED_MASTER_COLUMNS = [
    "model",
    "family_id",
    "checkpoint",
    "correct_crit",
    "v1_attractor_incongruent",
    "v2_attractor_incongruent",
    "np_pattern",
    "np1_num",
    "np2_num",
    "np3_num",
]


def model_label(model: str) -> str:
    return MODEL_LABELS.get(model, model)


def ordered_models(df: pd.DataFrame) -> list[str]:
    available = set(df["model"].unique())
    return [model for model in MODEL_ORDER if model in available]


def validate_master(df: pd.DataFrame) -> None:
    missing = [column for column in REQUIRED_MASTER_COLUMNS if column not in df.columns]

    if missing:
        raise ValueError(f"Master score file is missing required columns: {missing}")

    missing_models = [model for model in MODEL_ORDER if model not in set(df["model"])]

    if missing_models:
        raise ValueError(f"Master score file is missing expected models: {missing_models}")

    needed_checkpoints = {"V3", "V2", "COPULA"}
    found_checkpoints = set(df["checkpoint"].dropna().unique())

    if not needed_checkpoints.issubset(found_checkpoints):
        raise ValueError(
            "Master score file is missing expected checkpoints: "
            f"{sorted(needed_checkpoints - found_checkpoints)}"
        )


def save_figure(fig, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


def format_np(index: str, number: str) -> str:
    number_symbol = "s" if number == "S" else "p"
    return f"NP{SUB[index]}{SUB[number_symbol]}"


def make_pattern_label(row: pd.Series) -> str:
    return " ".join(
        [
            format_np("1", row["np1_num"]),
            format_np("2", row["np2_num"]),
            format_np("3", row["np3_num"]),
        ]
    )


def read_stats(path: Path | None) -> pd.DataFrame | None:
    if path is None:
        return None

    if not path.exists():
        print(f"Stats file not found: {path}")
        return None

    return pd.read_csv(path)


def require_one_value(row: pd.DataFrame, context: str) -> float:
    if len(row) != 1:
        raise ValueError(f"Expected exactly one value for {context}; found {len(row)}")

    return float(row["correct_crit"].iloc[0])


def plot_congruence_bars(
    df: pd.DataFrame,
    checkpoint: str,
    flag_column: str,
    congruent_label: str,
    incongruent_label: str,
    ylabel: str,
    title: str,
    out_path: Path,
) -> None:
    sub = df[df["checkpoint"] == checkpoint].copy()

    grouped = (
        sub.groupby(["model", flag_column])["correct_crit"]
        .mean()
        .reset_index()
    )

    models = ordered_models(grouped)
    congruent_values = []
    incongruent_values = []

    for model in models:
        model_rows = grouped[grouped["model"] == model]

        congruent_values.append(
            require_one_value(
                model_rows[model_rows[flag_column] == 0],
                f"{model}, {checkpoint}, {flag_column}=0",
            )
        )

        incongruent_values.append(
            require_one_value(
                model_rows[model_rows[flag_column] == 1],
                f"{model}, {checkpoint}, {flag_column}=1",
            )
        )

    x = np.arange(len(models))
    width = 0.4

    fig, ax = plt.subplots(figsize=(9.5, 4.2))

    ax.bar(x - width / 2, congruent_values, width, label=congruent_label)
    ax.bar(x + width / 2, incongruent_values, width, label=incongruent_label)

    ax.set_xticks(x)
    ax.set_xticklabels([model_label(m) for m in models], rotation=30, ha="right")
    ax.set_ylim(0, 1)
    ax.axhline(0.5, linewidth=0.8, linestyle="--")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=9)

    fig.tight_layout()
    save_figure(fig, out_path)


def plot_checkpoint_accuracy(df: pd.DataFrame, out_path: Path) -> None:
    """Compare checkpoints within model families using a shared accuracy axis."""
    families = {
        "GPT-2": ["gpt2"],
        "Pythia": ["pythia410"],
        "Llama": ["llama32_1b", "llama32_3b", "llama31_70b"],
        "Qwen2.5": [
            "qwen25_1.5b", "qwen25_3b", "qwen25_7b",
            "qwen25_14b", "qwen25_32b", "qwen25_72b",
        ],
    }
    size_labels = {
        "gpt2": "124M", "pythia410": "410M",
        "llama32_1b": "3.2 1B", "llama32_3b": "3.2 3B",
        "llama31_70b": "3.1 70B",
        "qwen25_1.5b": "1.5B", "qwen25_3b": "3B",
        "qwen25_7b": "7B", "qwen25_14b": "14B",
        "qwen25_32b": "32B", "qwen25_72b": "72B",
    }
    grouped = df.groupby(["model", "checkpoint"])["correct_crit"].agg(["mean", "size"])
    expected = {(model, checkpoint) for models in families.values()
                for model in models for checkpoint in ("V3", "V2", "COPULA")}
    if set(grouped.index) != expected or not grouped["size"].eq(2000).all():
        raise ValueError("Expected 2,000 items for each of the 33 model/checkpoint cells.")

    fig, axes = plt.subplots(2, 2, figsize=(7.0, 4.7), sharey=True)
    styles = [("V3", "#2368AD", "o"), ("V2", "#BD5B10", "s"),
              ("COPULA", "#37834C", "^")]
    for ax, (family, models) in zip(axes.flat, families.items()):
        x = np.arange(len(models))
        for index, (checkpoint, color, marker) in enumerate(styles):
            positions = x if len(models) > 1 else x + (index - 1) * 0.12
            accuracy = [grouped.loc[(model, checkpoint), "mean"] for model in models]
            ax.plot(positions, accuracy, color=color, marker=marker, markersize=4,
                    linewidth=1.2, linestyle="-" if len(models) > 1 else "none",
                    label=checkpoint)
        ax.set_title(family, fontsize=10, pad=7)
        ax.set_xticks(x, [size_labels[model] for model in models], fontsize=8)
        ax.set_xlim(-0.4, len(models) - 0.6)
        ax.set_ylim(0, 1)
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
        ax.tick_params(axis="y", labelsize=8)
        ax.axhline(0.5, color="#888888", linewidth=0.7, linestyle="--")
        ax.spines[["top", "right"]].set_visible(False)
    axes[0, 0].set_ylabel("Critical-region accuracy", fontsize=9)
    axes[1, 0].set_ylabel("Critical-region accuracy", fontsize=9)
    axes[1, 0].set_xlabel("Llama version and model size", fontsize=8)
    axes[1, 1].set_xlabel("Model size (equally spaced categories)", fontsize=8)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.91), h_pad=2.1, w_pad=1.7)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=250, bbox_inches="tight")
    save_figure(fig, out_path.with_name("checkpoint_accuracy_by_family.pdf"))


def plot_tokenization_robustness(audit_dir: Path, outdir: Path) -> None:
    """Plot the archived subset estimates and write their appendix table."""
    subsets = [
        ("Full (2,000)", "results_full_v2_2000.csv", 2000),
        ("Shared by all 11 (1,094)", "results_common_all_models.csv", 1094),
        ("Shared by 10, excluding Pythia (1,636)", "results_common_non_pythia.csv", 1636),
    ]
    models = [
        "gpt2", "pythia410", "llama32_1b", "llama32_3b", "llama31_70b",
        "qwen25_1.5b", "qwen25_3b", "qwen25_7b", "qwen25_14b",
        "qwen25_32b", "qwen25_72b",
    ]
    tables = []
    for _, filename, size in subsets:
        table = pd.read_csv(audit_dir / filename).set_index("model")
        expected = set(models) - ({"pythia410"} if size == 1636 else set())
        if not table.index.is_unique or set(table.index) != expected:
            raise ValueError(f"Unexpected model membership in {filename}")
        if not table["n_families"].eq(size).all():
            raise ValueError(f"Unexpected sample size in {filename}")
        tables.append(table)
    fig, ax = plt.subplots(figsize=(8.1, 5.8))
    styles = [("#2368AD", "o"), ("#BD5B10", "s"), ("#37834C", "^")]
    metric = "drop_congruent_minus_incongruent"
    for index, ((label, _, _), table, (color, marker)) in enumerate(zip(subsets, tables, styles)):
        selected = [model for model in models if model in table.index]
        values = table.loc[selected, metric].to_numpy()
        lower = table.loc[selected, "bootstrap_ci95_low"].to_numpy()
        upper = table.loc[selected, "bootstrap_ci95_high"].to_numpy()
        if not np.all((lower <= values) & (values <= upper)):
            raise ValueError("Stored intervals do not contain the point estimates.")
        y = np.array([models.index(model) for model in selected]) + (index - 1) * 0.21
        ax.errorbar(values, y, xerr=np.array([values - lower, upper - values]),
                    fmt=marker, color=color, markersize=4, linewidth=1,
                    capsize=2, label=label)
    ax.set_yticks(range(len(models)), [model_label(model) for model in models], fontsize=9)
    ax.invert_yaxis()
    ax.axvline(0, color="#777777", linestyle="--", linewidth=0.8)
    ax.set_xlabel("V2 accuracy gap (NP3 matches NP2 minus NP3 mismatches NP2)", fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.01), fontsize=8, frameon=False)
    fig.tight_layout()
    outdir.mkdir(parents=True, exist_ok=True)
    fig.savefig(outdir / "tokenization_robustness_v2.png", dpi=250, bbox_inches="tight")
    save_figure(fig, outdir / "tokenization_robustness_v2.pdf")

    table_dir = outdir.parent / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)
    lines = [
        r"\begin{table*}[tp]", r"\centering\small",
        r"\begin{tabular}{lccc}", r"\toprule",
        r"Model & Full (2,000) & Shared 11 (1,094) & Shared 10 (1,636) \\",
        r"\midrule",
    ]
    for model in models:
        cells = []
        for table in tables:
            if model not in table.index:
                cells.append("--")
                continue
            row = table.loc[model]
            cells.append(
                f"{row[metric]:.3f} [{row['bootstrap_ci95_low']:.3f}, "
                f"{row['bootstrap_ci95_high']:.3f}]"
            )
        lines.append(" & ".join([model_label(model), *cells]) + r" \\")
    lines.extend([
        r"\bottomrule", r"\end{tabular}",
        r"\caption{V2 critical-region accuracy gap for NP3--NP2 match versus mismatch. "
        r"Brackets give the archived 95\% bootstrap intervals (20,000 resamples). "
        r"The restricted sets contain families whose recorded critical span is exactly one token "
        r"for every included tokenizer. Shared 11 includes all models. Shared 10 omits Pythia. "
        r"All results reuse the original scores.}",
        r"\label{tab:tokenization-robustness}", r"\end{table*}",
    ])
    (table_dir / "table_tokenization_robustness.tex").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def bootstrap_drop_ci(
    df_checkpoint: pd.DataFrame,
    flag_column: str,
    n_bootstrap: int,
    seed: int,
) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)

    y = df_checkpoint["correct_crit"].to_numpy(dtype=float)
    label = df_checkpoint[flag_column].to_numpy(dtype=int)

    if len(np.unique(label)) != 2:
        raise ValueError(f"Cannot compute drop; {flag_column} does not contain both 0 and 1")

    observed = y[label == 0].mean() - y[label == 1].mean()

    drops = []
    n_rows = len(y)

    while len(drops) < n_bootstrap:
        sample = rng.choice(n_rows, size=n_rows, replace=True)
        y_sample = y[sample]
        label_sample = label[sample]

        if len(np.unique(label_sample)) != 2:
            continue

        drops.append(y_sample[label_sample == 0].mean() - y_sample[label_sample == 1].mean())

    ci_low = np.quantile(drops, 0.025)
    ci_high = np.quantile(drops, 0.975)

    return float(observed), float(ci_low), float(ci_high)


def make_drop_summary(
    df: pd.DataFrame,
    checkpoint: str,
    flag_column: str,
    n_bootstrap: int,
    seed: int,
) -> pd.DataFrame:
    rows = []
    models = ordered_models(df)

    for i, model in enumerate(models):
        sub = df[(df["model"] == model) & (df["checkpoint"] == checkpoint)].copy()
        model_seed = seed + (i * 10000)

        drop, ci_low, ci_high = bootstrap_drop_ci(
            df_checkpoint=sub,
            flag_column=flag_column,
            n_bootstrap=n_bootstrap,
            seed=model_seed,
        )

        rows.append(
            {
                "model": model,
                "size_m": MODEL_SIZES_M[model],
                "drop": drop,
                "ci_low": ci_low,
                "ci_high": ci_high,
            }
        )

    return pd.DataFrame(rows)


def make_v2_drop_summary(
    df: pd.DataFrame,
    stats: pd.DataFrame | None,
    n_bootstrap: int,
    seed: int,
) -> pd.DataFrame:
    needed = {"model", "v2_drop_crit", "ci95_lo", "ci95_hi"}

    if stats is not None and needed.issubset(set(stats.columns)):
        rows = []

        for model in MODEL_ORDER:
            sub = stats[stats["model"] == model]

            if len(sub) == 0:
                continue

            row = sub.iloc[0]

            rows.append(
                {
                    "model": model,
                    "size_m": MODEL_SIZES_M[model],
                    "drop": float(row["v2_drop_crit"]),
                    "ci_low": float(row["ci95_lo"]),
                    "ci_high": float(row["ci95_hi"]),
                }
            )

        return pd.DataFrame(rows)

    return make_drop_summary(
        df=df,
        checkpoint="V2",
        flag_column="v2_attractor_incongruent",
        n_bootstrap=n_bootstrap,
        seed=seed,
    )


def plot_scaling_scatter(
    summary: pd.DataFrame,
    ylabel: str,
    title: str,
    out_path: Path,
) -> None:
    summary = summary.sort_values("size_m").copy()

    lower_error = summary["drop"] - summary["ci_low"]
    upper_error = summary["ci_high"] - summary["drop"]
    errors = np.vstack([lower_error, upper_error])

    fig, ax = plt.subplots(figsize=(8.2, 4.4))

    ax.errorbar(
        summary["size_m"],
        summary["drop"],
        yerr=errors,
        fmt="o",
        capsize=4,
        linestyle="none",
    )

    for _, row in summary.iterrows():
        ax.annotate(
            model_label(row["model"]),
            (row["size_m"], row["drop"]),
            textcoords="offset points",
            xytext=(6, 6),
            fontsize=8,
        )

    ax.set_xscale("log")
    ax.axhline(0, linewidth=0.8, linestyle="--")
    ax.set_xlabel("Parameters, millions, log scale")
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    fig.tight_layout()
    save_figure(fig, out_path)


def pattern_info(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df[["np_pattern", "np1_num", "np2_num", "np3_num"]]
        .dropna(subset=["np_pattern"])
        .drop_duplicates("np_pattern")
        .set_index("np_pattern")
    )


def plot_np_pattern_by_pattern(
    df: pd.DataFrame,
    checkpoint: str,
    title: str,
    out_path: Path,
) -> None:
    df = df.dropna(subset=["np_pattern"]).copy()
    sub = df[df["checkpoint"] == checkpoint].copy()
    info = pattern_info(df)

    patterns = [p for p in PATTERN_ORDER if p in set(sub["np_pattern"])]
    models = ordered_models(sub)

    grouped = (
        sub.groupby(["model", "np_pattern"])["correct_crit"]
        .mean()
        .reset_index()
    )

    x = np.arange(len(patterns))
    width = 0.85 / len(models)

    fig, ax = plt.subplots(figsize=(17, 7))

    for j, model in enumerate(models):
        values = []

        for pattern in patterns:
            row = grouped[
                (grouped["model"] == model)
                & (grouped["np_pattern"] == pattern)
            ]
            values.append(require_one_value(row, f"{model}, {checkpoint}, {pattern}"))

        offset = (j - len(models) / 2 + 0.5) * width
        ax.bar(x + offset, values, width, label=model_label(model))

    labels = [make_pattern_label(info.loc[pattern]) for pattern in patterns]

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylim(0, 1)
    ax.axhline(0.5, linewidth=0.8, linestyle="--")
    ax.set_ylabel("Critical-region accuracy")
    ax.set_title(title)
    ax.legend(title="Model", bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=9)

    fig.subplots_adjust(right=0.80)
    save_figure(fig, out_path)


def plot_np_pattern_by_model(
    df: pd.DataFrame,
    checkpoint: str,
    title: str,
    out_path: Path,
) -> None:
    df = df.dropna(subset=["np_pattern"]).copy()
    sub = df[df["checkpoint"] == checkpoint].copy()
    info = pattern_info(df)

    patterns = [p for p in PATTERN_ORDER if p in set(sub["np_pattern"])]
    models = ordered_models(sub)

    grouped = (
        sub.groupby(["model", "np_pattern"])["correct_crit"]
        .mean()
        .reset_index()
    )

    x = np.arange(len(models))
    width = 0.85 / len(patterns)

    fig, ax = plt.subplots(figsize=(18, 7))

    for k, pattern in enumerate(patterns):
        values = []

        for model in models:
            row = grouped[
                (grouped["model"] == model)
                & (grouped["np_pattern"] == pattern)
            ]
            values.append(require_one_value(row, f"{model}, {checkpoint}, {pattern}"))

        label = make_pattern_label(info.loc[pattern])
        offset = (k - len(patterns) / 2 + 0.5) * width
        ax.bar(x + offset, values, width, label=label)

    ax.set_xticks(x)
    ax.set_xticklabels([model_label(m) for m in models], rotation=30, ha="right")
    ax.set_ylim(0, 1)
    ax.axhline(0.5, linewidth=0.8, linestyle="--")
    ax.set_ylabel("Critical-region accuracy")
    ax.set_title(title)
    ax.legend(title="NP pattern", bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=9)

    fig.subplots_adjust(right=0.80)
    save_figure(fig, out_path)


def make_all_figures(
    df: pd.DataFrame,
    stats: pd.DataFrame | None,
    outdir: Path,
    n_bootstrap: int,
    seed: int,
) -> None:
    plot_congruence_bars(
        df=df,
        checkpoint="V2",
        flag_column="v2_attractor_incongruent",
        congruent_label="NP\u2083 matches NP\u2082",
        incongruent_label="NP\u2083 mismatches NP\u2082",
        ylabel="V2 critical-region accuracy",
        title="V2 accuracy by NP\u2083 relation to NP\u2082",
        out_path=outdir / "v2_accuracy_by_attractor.png",
    )

    plot_congruence_bars(
        df=df,
        checkpoint="COPULA",
        flag_column="v1_attractor_incongruent",
        congruent_label="NP\u2083 matches NP\u2081",
        incongruent_label="NP\u2083 mismatches NP\u2081",
        ylabel="COPULA critical-region accuracy",
        title="COPULA accuracy by NP\u2083 relation to NP\u2081",
        out_path=outdir / "copula_accuracy_by_attractor.png",
    )

    plot_checkpoint_accuracy(df=df, out_path=outdir / "checkpoint_accuracy.png")

    v2_summary = make_v2_drop_summary(
        df=df,
        stats=stats,
        n_bootstrap=n_bootstrap,
        seed=seed,
    )

    plot_scaling_scatter(
        summary=v2_summary,
        ylabel="V2 drop: congruent minus incongruent",
        title="Model scale and the V2 attractor effect",
        out_path=outdir / "scaling_v2_scatter.png",
    )

    copula_summary = make_drop_summary(
        df=df,
        checkpoint="COPULA",
        flag_column="v1_attractor_incongruent",
        n_bootstrap=n_bootstrap,
        seed=seed + 500000,
    )

    plot_scaling_scatter(
        summary=copula_summary,
        ylabel="COPULA drop: congruent minus incongruent",
        title="Model scale and the COPULA NP\u2083 effect",
        out_path=outdir / "scaling_copula_scatter.png",
    )

    plot_np_pattern_by_pattern(
        df=df,
        checkpoint="V2",
        title="V2 accuracy by NP number pattern",
        out_path=outdir / "appendix_np_pattern_v2.png",
    )

    plot_np_pattern_by_pattern(
        df=df,
        checkpoint="COPULA",
        title="COPULA accuracy by NP number pattern",
        out_path=outdir / "appendix_np_pattern_copula.png",
    )

    plot_np_pattern_by_model(
        df=df,
        checkpoint="V2",
        title="V2 accuracy by model and NP number pattern",
        out_path=outdir / "appendix_np_pattern_by_model_v2.png",
    )

    plot_np_pattern_by_model(
        df=df,
        checkpoint="COPULA",
        title="COPULA accuracy by model and NP number pattern",
        out_path=outdir / "appendix_np_pattern_by_model_copula.png",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Make paper figures from the master score file."
    )

    parser.add_argument(
        "--master",
        type=Path,
        default=DEFAULT_MASTER,
        help="Input master score CSV.",
    )

    parser.add_argument(
        "--stats",
        type=Path,
        default=DEFAULT_STATS,
        help="Optional stats summary CSV.",
    )

    parser.add_argument(
        "--outdir",
        type=Path,
        default=DEFAULT_OUTDIR,
        help="Output figure folder.",
    )

    parser.add_argument(
        "--camera-ready-only", action="store_true",
        help="Generate the family checkpoint figure and archived tokenization appendix only.",
    )

    parser.add_argument(
        "--bootstrap",
        type=int,
        default=10000,
        help="Bootstrap samples for COPULA scaling CIs when needed.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed.",
    )

    args = parser.parse_args()

    if not args.master.exists():
        raise FileNotFoundError(f"Could not find input file: {args.master}")

    print(f"Reading {args.master}")
    df = pd.read_csv(args.master)
    validate_master(df)

    if args.camera_ready_only:
        plot_checkpoint_accuracy(df, args.outdir / "checkpoint_accuracy.png")
        plot_tokenization_robustness(REPO_ROOT / "tokenization_robustness_audit", args.outdir)
        return

    stats = read_stats(args.stats)
    args.outdir.mkdir(parents=True, exist_ok=True)

    make_all_figures(
        df=df,
        stats=stats,
        outdir=args.outdir,
        n_bootstrap=args.bootstrap,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
