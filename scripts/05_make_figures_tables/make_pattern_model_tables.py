#!/usr/bin/env python3
"""
Make compact NP-pattern tables for the paper and appendix.

Input:
    data/scores/summaries/pattern_model_pvalues.csv

Outputs:
    paper/tables/table_pattern_v2_stacked.tex
    paper/tables/table_pattern_copula_stacked.tex
    paper/tables/table_pattern_v2_wide.tex

The stacked appendix tables preserve the regime grouping while keeping the
appendix tables narrow enough for a two-column paper. Each pattern-pair column
contains two patterns stacked vertically. Each model cell contains the two
corresponding accuracies stacked in the same order.

Stars come from the p-values in pattern_model_pvalues.csv. The current pipeline
uses exact binomial tests against chance accuracy = 0.5 using one checkpoint item per sentence family.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_PVALUES = (
    REPO_ROOT / "data" / "scores" / "summaries" / "pattern_model_pvalues.csv"
)

DEFAULT_OUT_DIR = REPO_ROOT / "paper" / "tables"

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

# V2 controller is NP2. These pairings preserve the four V2 regimes.
V2_PATTERN_PAIRS = [
    ("SSS", "PPP"),  # no mismatch
    ("SSP", "PPS"),  # NP3 mismatch only
    ("PSS", "SPP"),  # NP1 mismatch only
    ("SPS", "PSP"),  # NP1 and NP3 mismatch
]

# COPULA controller is NP1. These pairings preserve the four COPULA regimes.
COPULA_PATTERN_PAIRS = [
    ("SSS", "PPP"),  # no attractor
    ("SSP", "PPS"),  # NP3 attractor only
    ("SPS", "PSP"),  # NP2 attractor only
    ("SPP", "PSS"),  # NP2 and NP3 attractors
]

V2_WIDE_PATTERN_ORDER = ["SSS", "PPP", "SSP", "PPS", "PSS", "SPP", "SPS", "PSP"]


def latex_escape(text):
    """Escape only the characters that occur in model names."""
    return str(text).replace("_", r"\_")


def check_required_columns(df):
    required = [
        "checkpoint",
        "np_pattern",
        "n_families",
        "model",
        "acc",
        "sig_label",
    ]

    missing = [column for column in required if column not in df.columns]

    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def format_accuracy(acc, sig_label):
    """Format accuracy with significance stars."""
    if pd.isna(acc):
        return "--"

    value = f"{float(acc):.3f}"
    if value.startswith("0"):
        value = value[1:]

    stars = ""
    if isinstance(sig_label, str) and sig_label != "ns":
        stars = sig_label

    # Bold means above chance. Stars indicate the exact binomial-test result.
    if float(acc) > 0.5:
        return f"\\textbf{{{value}}}{stars}"

    return f"{value}{stars}"


def controller_label(pattern, controller_index):
    """
    Return a LaTeX pattern label with the controller bolded.

    controller_index is zero-based:
        0 = NP1
        1 = NP2
        2 = NP3
    """
    chars = list(pattern)
    chars[controller_index] = r"\mathbf{" + chars[controller_index] + "}"
    return "$" + "".join(chars) + "$"


def get_cell_map(df_checkpoint):
    cell_map = {}

    for _, row in df_checkpoint.iterrows():
        key = (row["model"], row["np_pattern"])
        cell_map[key] = (row["acc"], row["sig_label"])

    return cell_map


def get_family_counts(df_checkpoint, patterns):
    counts = {}

    for pattern in patterns:
        sub = df_checkpoint[df_checkpoint["np_pattern"] == pattern]

        if len(sub) == 0:
            raise ValueError(f"No rows found for pattern {pattern}")

        unique_counts = sorted(sub["n_families"].dropna().unique())

        if len(unique_counts) != 1:
            raise ValueError(
                f"Different family counts found for pattern {pattern}: {unique_counts}"
            )

        counts[pattern] = int(unique_counts[0])

    return counts


def format_n_counts(counts, pattern_pairs):
    pieces = []

    for top, bottom in pattern_pairs:
        pieces.append(f"{top}={counts[top]}")
        pieces.append(f"{bottom}={counts[bottom]}")

    return ", ".join(pieces)


def make_pair_header(top, bottom, controller_index):
    top_label = controller_label(top, controller_index)
    bottom_label = controller_label(bottom, controller_index)
    return rf"\shortstack{{{top_label}\\{bottom_label}}}"


def make_stacked_cell(cell_map, model, top, bottom):
    top_acc, top_sig = cell_map.get((model, top), (np.nan, "ns"))
    bottom_acc, bottom_sig = cell_map.get((model, bottom), (np.nan, "ns"))

    top_text = format_accuracy(top_acc, top_sig)
    bottom_text = format_accuracy(bottom_acc, bottom_sig)

    return rf"\shortstack{{{top_text}\\{bottom_text}}}"


def make_stacked_caption(checkpoint, counts, pattern_pairs):
    n_text = format_n_counts(counts, pattern_pairs)

    if checkpoint == "V2":
        return (
            "V2 critical-region accuracy by NP-number pattern and model. "
            "Columns preserve the four V2 regimes; the two patterns in each "
            "column share the same mismatch configuration but differ in absolute "
            "number values. The controller NP$_2$ is bolded in each pattern label. "
            f"Family counts by pattern: {n_text}. "
            "Bold accuracies are above chance; stars mark two-sided exact "
            "binomial tests against chance accuracy = 0.5 using one checkpoint item per sentence family "
            "({*} $p < .05$, {**} $p < .01$, {***} $p < .001$)."
        )

    if checkpoint == "COPULA":
        return (
            "COPULA critical-region accuracy by NP-number pattern and model. "
            "Columns preserve the four COPULA regimes; the two patterns in each "
            "column share the same attractor configuration but differ in absolute "
            "number values. The controller NP$_1$ is bolded in each pattern label. "
            f"Family counts by pattern: {n_text}. "
            "Bold accuracies are above chance; stars mark two-sided exact "
            "binomial tests against chance accuracy = 0.5 using one checkpoint item per sentence family "
            "({*} $p < .05$, {**} $p < .01$, {***} $p < .001$)."
        )

    raise ValueError(f"Unknown checkpoint: {checkpoint}")


def build_stacked_table(df, checkpoint, pattern_pairs, controller_index, label):
    df_checkpoint = df[df["checkpoint"] == checkpoint].copy()

    if df_checkpoint.empty:
        raise ValueError(f"No rows found for checkpoint {checkpoint}")

    patterns = [p for pair in pattern_pairs for p in pair]
    counts = get_family_counts(df_checkpoint, patterns)
    cell_map = get_cell_map(df_checkpoint)

    lines = []
    lines.append(r"\begin{table}[!t]")
    lines.append(r"    \centering")
    lines.append(r"    \scriptsize")
    lines.append(r"    \setlength{\tabcolsep}{4pt}")
    lines.append(r"    \renewcommand{\arraystretch}{1.12}")
    lines.append(r"    \begin{tabular}{lcccc}")
    lines.append(r"        \toprule")

    headers = [r"\textbf{Model}"]
    for top, bottom in pattern_pairs:
        headers.append(make_pair_header(top, bottom, controller_index))

    lines.append("        " + " & ".join(headers) + r" \\")
    lines.append(r"        \midrule")

    for model in MODEL_ORDER:
        row = [MODEL_LABELS[model]]
        for top, bottom in pattern_pairs:
            row.append(make_stacked_cell(cell_map, model, top, bottom))
        lines.append("        " + " & ".join(row) + r" \\")

    lines.append(r"        \bottomrule")
    lines.append(r"    \end{tabular}")
    lines.append(f"    \\caption{{{make_stacked_caption(checkpoint, counts, pattern_pairs)}}}")
    lines.append(f"    \\label{{{label}}}")
    lines.append(r"\end{table}")

    return "\n".join(lines) + "\n"


def make_wide_v2_caption(counts):
    n_text = ", ".join([f"{p}={counts[p]}" for p in V2_WIDE_PATTERN_ORDER])
    return (
        "V2 critical-region accuracy by NP-number pattern and model. "
        "The controller NP$_2$ is bolded in each pattern label. "
        f"Family counts by pattern: {n_text}. "
        "Bold accuracies are above chance; stars mark two-sided exact "
        "binomial tests against chance accuracy = 0.5 using one checkpoint item per sentence family "
        "({*} $p < .05$, {**} $p < .01$, {***} $p < .001$)."
    )


def build_wide_v2_table(df):
    checkpoint = "V2"
    df_checkpoint = df[df["checkpoint"] == checkpoint].copy()
    counts = get_family_counts(df_checkpoint, V2_WIDE_PATTERN_ORDER)
    cell_map = get_cell_map(df_checkpoint)

    lines = []
    lines.append(r"\begin{table*}[!t]")
    lines.append(r"    \centering")
    lines.append(r"    \scriptsize")
    lines.append(r"    \setlength{\tabcolsep}{4pt}")
    lines.append(r"    \renewcommand{\arraystretch}{1.08}")
    lines.append(r"    \begin{tabular}{lrrrrrrrr}")
    lines.append(r"        \toprule")

    headers = [r"\textbf{Model}"]
    for pattern in V2_WIDE_PATTERN_ORDER:
        headers.append(controller_label(pattern, controller_index=1))

    lines.append("        " + " & ".join(headers) + r" \\")
    lines.append(r"        \midrule")

    for model in MODEL_ORDER:
        row = [MODEL_LABELS[model]]
        for pattern in V2_WIDE_PATTERN_ORDER:
            acc, sig = cell_map.get((model, pattern), (np.nan, "ns"))
            row.append(format_accuracy(acc, sig))
        lines.append("        " + " & ".join(row) + r" \\")

    lines.append(r"        \bottomrule")
    lines.append(r"    \end{tabular}")
    lines.append(f"    \\caption{{{make_wide_v2_caption(counts)}}}")
    lines.append(r"    \label{tab:v2_pattern_wide}")
    lines.append(r"\end{table*}")

    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(
        description="Make compact stacked NP-pattern tables."
    )
    parser.add_argument(
        "--pvalues",
        type=Path,
        default=DEFAULT_PVALUES,
        help="Input pattern_model_pvalues.csv file.",
    )
    parser.add_argument(
        "--out-dir",
        "--out_dir",
        dest="out_dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="Directory for output LaTeX tables.",
    )
    args = parser.parse_args()

    if not args.pvalues.exists():
        raise FileNotFoundError(f"Could not find input file: {args.pvalues}")

    print(f"Reading {args.pvalues}")
    df = pd.read_csv(args.pvalues)
    check_required_columns(df)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    outputs = {
        "table_pattern_v2_stacked.tex": build_stacked_table(
            df=df,
            checkpoint="V2",
            pattern_pairs=V2_PATTERN_PAIRS,
            controller_index=1,
            label="tab:pattern_v2_stacked",
        ),
        "table_pattern_copula_stacked.tex": build_stacked_table(
            df=df,
            checkpoint="COPULA",
            pattern_pairs=COPULA_PATTERN_PAIRS,
            controller_index=0,
            label="tab:pattern_copula_stacked",
        ),
        "table_pattern_v2_wide.tex": build_wide_v2_table(df),
    }

    for filename, text in outputs.items():
        out_path = args.out_dir / filename
        out_path.write_text(text, encoding="utf-8")
        print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
