#!/usr/bin/env python3
"""
Make compact LaTeX tables for the main paper.

Inputs:
    data/scores/summaries/stats_summary.csv
    data/scores/summaries/regime_contrasts.csv

Outputs:
    paper/tables/table_v2_main_results.tex
    paper/tables/table_v2_regime_contrasts.tex

The first table reports the main V2 attractor effect.
The second table reports the V2 diagonal regime contrast.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_STATS = (
    REPO_ROOT
    / "data"
    / "scores"
    / "summaries"
    / "stats_summary.csv"
)

DEFAULT_REGIME = (
    REPO_ROOT
    / "data"
    / "scores"
    / "summaries"
    / "regime_contrasts.csv"
)

DEFAULT_OUT_MAIN = (
    REPO_ROOT
    / "paper"
    / "tables"
    / "table_v2_main_results.tex"
)

DEFAULT_OUT_REGIME = (
    REPO_ROOT
    / "paper"
    / "tables"
    / "table_v2_regime_contrasts.tex"
)

MODEL_ORDER = [
    "gpt2",
    "pythia410",
    "llama32_1b",
    "qwen25_1.5b",
    "llama32_3b",
    "qwen25_3b",
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
    "llama32_3b": "Llama-3.2-3B",
    "qwen25_1.5b": "Qwen2.5-1.5B",
    "qwen25_3b": "Qwen2.5-3B",
    "qwen25_7b": "Qwen2.5-7B",
    "qwen25_14b": "Qwen2.5-14B",
    "qwen25_32b": "Qwen2.5-32B",
    "llama31_70b": "Llama-3.1-70B",
    "qwen25_72b": "Qwen2.5-72B",
}


def fmt3(x: float) -> str:
    value = f"{float(x):.3f}"

    if value.startswith("0"):
        value = value[1:]

    if value.startswith("-0"):
        value = "-" + value[2:]

    return value


def fmt_ci(lo: float, hi: float) -> str:
    return f"[{fmt3(lo)}, {fmt3(hi)}]"


def check_columns(df: pd.DataFrame, required: list[str], name: str) -> None:
    missing = [column for column in required if column not in df.columns]

    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def order_models(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["model"] = pd.Categorical(df["model"], categories=MODEL_ORDER, ordered=True)
    return df.sort_values("model").reset_index(drop=True)


def build_v2_main_table(stats: pd.DataFrame) -> str:
    check_columns(
        stats,
        [
            "model",
            "acc_v2_cong_crit",
            "acc_v2_incong_crit",
            "v2_drop_crit",
            "ci95_lo",
            "ci95_hi",
        ],
        "stats_summary.csv",
    )

    stats = order_models(stats)

    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"    \centering")
    lines.append(r"    \footnotesize")
    lines.append(r"    \setlength{\tabcolsep}{4.5pt}")
    lines.append(r"    \begin{tabular}{lrrrr}")
    lines.append(r"        \toprule")
    lines.append(
        r"        \textbf{Model} & \textbf{Cong.} & \textbf{Incong.} & \textbf{Drop} & \textbf{95\% CI} \\"
    )
    lines.append(r"        \midrule")

    for _, row in stats.iterrows():
        model = MODEL_LABELS[str(row["model"])]
        cong = fmt3(row["acc_v2_cong_crit"])
        incong = fmt3(row["acc_v2_incong_crit"])
        drop = fmt3(row["v2_drop_crit"])
        ci = fmt_ci(row["ci95_lo"], row["ci95_hi"])

        lines.append(
            f"        {model} & {cong} & {incong} & {drop} & ${ci}$ \\\\"
        )

    lines.append(r"        \bottomrule")
    lines.append(r"    \end{tabular}")
    lines.append(
        r"    \caption{V2 critical-region accuracy by NP\textsubscript{3} attractor congruence. "
        r"Cong. = NP\textsubscript{3} matches the V2 controller NP\textsubscript{2}; "
        r"Incong. = NP\textsubscript{3} mismatches NP\textsubscript{2}. "
        r"Drop = congruent accuracy minus incongruent accuracy. "
        r"Confidence intervals are 95\% bootstrap intervals.}"
    )
    lines.append(r"    \label{tab:v2_results}")
    lines.append(r"\end{table}")

    return "\n".join(lines) + "\n"


def build_regime_table(regime: pd.DataFrame) -> str:
    check_columns(
        regime,
        [
            "model",
            "checkpoint",
            "acc_00",
            "acc_01",
            "acc_10",
            "acc_11",
            "diagonal_contrast",
            "ci95_lo",
            "ci95_hi",
        ],
        "regime_contrasts.csv",
    )

    regime = regime[regime["checkpoint"] == "V2"].copy()
    regime = order_models(regime)

    lines = []
    lines.append(r"\begin{table*}[t]")
    lines.append(r"    \centering")
    lines.append(r"    \footnotesize")
    lines.append(r"    \setlength{\tabcolsep}{5pt}")
    lines.append(r"    \begin{tabular}{lrrrrrr}")
    lines.append(r"        \toprule")
    lines.append(
        r"        \textbf{Model} & \textbf{(0,0)} & \textbf{(0,1)} & \textbf{(1,0)} & \textbf{(1,1)} & \textbf{Diag. diff.} & \textbf{95\% CI} \\"
    )
    lines.append(r"        \midrule")

    for _, row in regime.iterrows():
        model = MODEL_LABELS[str(row["model"])]
        row_values = [
            fmt3(row["acc_00"]),
            fmt3(row["acc_01"]),
            fmt3(row["acc_10"]),
            fmt3(row["acc_11"]),
            fmt3(row["diagonal_contrast"]),
            f"${fmt_ci(row['ci95_lo'], row['ci95_hi'])}$",
        ]

        lines.append("        " + model + " & " + " & ".join(row_values) + r" \\")

    lines.append(r"        \bottomrule")
    lines.append(r"    \end{tabular}")
    lines.append(
        r"    \caption{V2 critical-region accuracy by attractor regime. "
        r"The diagonal difference is $((0,0)+(1,1))/2 - ((0,1)+(1,0))/2$. "
        r"For V2, the diagonal regimes are cases where NP\textsubscript{1} and "
        r"NP\textsubscript{2} match; the off-diagonal regimes are cases where "
        r"NP\textsubscript{1} and NP\textsubscript{2} mismatch. "
        r"Confidence intervals are 95\% bootstrap intervals.}"
    )
    lines.append(r"    \label{tab:v2_regime_contrasts}")
    lines.append(r"\end{table*}")

    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Make compact LaTeX tables for the main paper."
    )

    parser.add_argument(
        "--stats",
        type=Path,
        default=DEFAULT_STATS,
        help="Input stats_summary.csv file.",
    )

    parser.add_argument(
        "--regime",
        type=Path,
        default=DEFAULT_REGIME,
        help="Input regime_contrasts.csv file.",
    )

    parser.add_argument(
        "--out-main",
        "--out_main",
        dest="out_main",
        type=Path,
        default=DEFAULT_OUT_MAIN,
        help="Output path for the V2 main table.",
    )

    parser.add_argument(
        "--out-regime",
        "--out_regime",
        dest="out_regime",
        type=Path,
        default=DEFAULT_OUT_REGIME,
        help="Output path for the V2 regime table.",
    )

    args = parser.parse_args()

    if not args.stats.exists():
        raise FileNotFoundError(f"Could not find input file: {args.stats}")

    if not args.regime.exists():
        raise FileNotFoundError(f"Could not find input file: {args.regime}")

    print(f"Reading {args.stats}")
    stats = pd.read_csv(args.stats)

    print(f"Reading {args.regime}")
    regime = pd.read_csv(args.regime)

    args.out_main.parent.mkdir(parents=True, exist_ok=True)
    args.out_regime.parent.mkdir(parents=True, exist_ok=True)

    args.out_main.write_text(build_v2_main_table(stats), encoding="utf-8")
    args.out_regime.write_text(build_regime_table(regime), encoding="utf-8")

    print(f"Wrote {args.out_main}")
    print(f"Wrote {args.out_regime}")


if __name__ == "__main__":
    main()
