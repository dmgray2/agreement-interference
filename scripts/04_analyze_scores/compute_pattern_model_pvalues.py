#!/usr/bin/env python3
"""
Compute exact binomial p-values for the NP-pattern appendix tables.

Input:
    data/scores/master_score_lists/master_scores_full.csv

Output:
    data/scores/summaries/pattern_model_pvalues.csv

For each model, checkpoint, and NP-number pattern, this script computes:

    critical-region accuracy
    number of items
    number of families
    exact two-sided binomial p-value against chance accuracy = 0.5
    whether accuracy is above or below chance
    a significance label for the LaTeX tables

The test uses one checkpoint item per sentence family. In the released benchmark, each
model/checkpoint/pattern/family cell should contribute exactly one binary
critical-region score. The script checks this assumption before using the
binomial test.

Run from the repository root:

    python3 scripts/04_analyze_scores/compute_pattern_model_pvalues.py

Optional:

    python3 scripts/04_analyze_scores/compute_pattern_model_pvalues.py \
      --checkpoints V2 COPULA
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest


REPO_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_MASTER = (
    REPO_ROOT
    / "data"
    / "scores"
    / "master_score_lists"
    / "master_scores_full.csv"
)

DEFAULT_OUT = (
    REPO_ROOT
    / "data"
    / "scores"
    / "summaries"
    / "pattern_model_pvalues.csv"
)

PATTERN_ORDER = ["SSS", "PPP", "SSP", "PPS", "PSS", "SPP", "SPS", "PSP"]
CHECKPOINT_ORDER = ["V2", "COPULA"]


def check_required_columns(df: pd.DataFrame, required_columns: list[str]) -> None:
    """Stop with a clear message if an expected column is missing."""
    missing = [column for column in required_columns if column not in df.columns]

    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def significance_label(p_value: float) -> str | float:
    """Convert a p-value into the star label used in the appendix tables."""
    if pd.isna(p_value):
        return np.nan

    if p_value < 0.001:
        return "***"

    if p_value < 0.01:
        return "**"

    if p_value < 0.05:
        return "*"

    return "ns"


def direction_label(accuracy: float) -> str | float:
    """Say whether the cell accuracy is above or below chance."""
    if pd.isna(accuracy):
        return np.nan

    if accuracy > 0.5:
        return "above"

    if accuracy < 0.5:
        return "below"

    return "chance"


def summarize_cell(
    checkpoint: str,
    pattern: str,
    model: str,
    sub: pd.DataFrame,
) -> dict[str, object]:
    """Compute accuracy and exact binomial p-value for one table cell."""
    if len(sub) == 0:
        return {
            "checkpoint": checkpoint,
            "np_pattern": pattern,
            "n_items": 0,
            "n_families": 0,
            "model": model,
            "acc": np.nan,
            "n_correct_families": 0,
            "p_two_sided": np.nan,
            "direction": np.nan,
            "sig_label": np.nan,
        }

    family_scores = (
        sub.groupby("family_id")["correct_crit"]
        .mean()
        .reset_index(name="family_score")
    )

    non_binary = family_scores[
        ~family_scores["family_score"].isin([0, 1])
    ]

    if len(non_binary) > 0:
        raise ValueError(
            "The exact binomial test requires one binary score per family. "
            f"Found non-binary family means for {checkpoint} | {pattern} | {model}."
        )

    n_families = len(family_scores)
    n_correct = int(family_scores["family_score"].sum())
    accuracy = n_correct / n_families

    p_value = binomtest(
        k=n_correct,
        n=n_families,
        p=0.5,
        alternative="two-sided",
    ).pvalue

    return {
        "checkpoint": checkpoint,
        "np_pattern": pattern,
        "n_items": int(len(sub)),
        "n_families": int(n_families),
        "model": model,
        "acc": round(float(accuracy), 4),
        "n_correct_families": int(n_correct),
        "p_two_sided": round(float(p_value), 6),
        "direction": direction_label(accuracy),
        "sig_label": significance_label(p_value),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute exact binomial p-values for NP-pattern appendix tables."
    )

    parser.add_argument(
        "--master",
        type=Path,
        default=DEFAULT_MASTER,
        help="Input master score CSV.",
    )

    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT,
        help="Output CSV.",
    )

    parser.add_argument(
        "--checkpoints",
        nargs="+",
        default=CHECKPOINT_ORDER,
        help="Checkpoints to summarize.",
    )

    # Kept only so older run commands do not fail. These are not used because
    # the revised script uses scipy's exact binomial test instead of simulation.
    parser.add_argument(
        "--perm",
        type=int,
        default=None,
        help="Ignored. Kept for compatibility with older run commands.",
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Ignored. Kept for compatibility with older run commands.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Ignored. Kept for compatibility with older run commands.",
    )

    args = parser.parse_args()

    if not args.master.exists():
        raise FileNotFoundError(f"Could not find input file: {args.master}")

    print(f"Reading {args.master}")
    df = pd.read_csv(args.master)

    required_columns = [
        "model",
        "family_id",
        "checkpoint",
        "np_pattern",
        "correct_crit",
    ]

    check_required_columns(df, required_columns)

    df = df.dropna(subset=["np_pattern", "correct_crit"]).copy()

    checkpoints = [
        checkpoint
        for checkpoint in args.checkpoints
        if checkpoint in set(df["checkpoint"])
    ]

    patterns = [
        pattern
        for pattern in PATTERN_ORDER
        if pattern in set(df["np_pattern"])
    ]

    models = sorted(df["model"].unique())

    rows = []
    total_cells = len(checkpoints) * len(patterns) * len(models)
    cell_number = 0

    for checkpoint in checkpoints:
        df_checkpoint = df[df["checkpoint"] == checkpoint]

        for pattern in patterns:
            df_pattern = df_checkpoint[df_checkpoint["np_pattern"] == pattern]

            for model in models:
                cell_number += 1
                print(f"[{cell_number}/{total_cells}] {checkpoint} | {pattern} | {model}")

                sub = df_pattern[df_pattern["model"] == model]

                row = summarize_cell(
                    checkpoint=checkpoint,
                    pattern=pattern,
                    model=model,
                    sub=sub,
                )

                rows.append(row)

    out_df = pd.DataFrame(rows)

    column_order = [
        "checkpoint",
        "np_pattern",
        "n_items",
        "n_families",
        "model",
        "acc",
        "n_correct_families",
        "p_two_sided",
        "direction",
        "sig_label",
    ]

    out_df = out_df[column_order]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.out, index=False)

    print()
    print(f"Wrote {args.out}")
    print(f"Rows: {len(out_df)}")

    print()
    print("Significance labels:")
    print(out_df["sig_label"].value_counts(dropna=False).to_string())

    print()
    print("Directions:")
    print(out_df["direction"].value_counts(dropna=False).to_string())


if __name__ == "__main__":
    main()
