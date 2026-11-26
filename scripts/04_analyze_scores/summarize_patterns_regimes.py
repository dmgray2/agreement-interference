#!/usr/bin/env python3
"""
Make pattern and regime summary files.

Input:
    data/scores/master_score_lists/master_scores_full.csv

Outputs:
    data/scores/summaries/pattern_breakdown.csv
    data/scores/summaries/regime_breakdown.csv

The master score file should already have these metadata columns:

    np1_num
    np2_num
    np3_num
    np_pattern
    np2_np1_incongruent
    regime_v1v2

This script does not change the master score file. It only makes
summary files.
"""

import argparse
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_MASTER = (
    REPO_ROOT
    / "data"
    / "scores"
    / "master_score_lists"
    / "master_scores_full.csv"
)

DEFAULT_OUT_PATTERN_BREAKDOWN = (
    REPO_ROOT
    / "data"
    / "scores"
    / "summaries"
    / "pattern_breakdown.csv"
)

DEFAULT_OUT_REGIME_BREAKDOWN = (
    REPO_ROOT
    / "data"
    / "scores"
    / "summaries"
    / "regime_breakdown.csv"
)


PATTERN_ORDER = ["SSS", "PPP", "SSP", "PPS", "PSS", "SPP", "SPS", "PSP"]
REGIME_ORDER = ["(0,0)", "(0,1)", "(1,0)", "(1,1)"]


def check_required_columns(df, required_columns):
    """
    Stop with a clear message if an expected column is missing.
    """
    missing = []

    for column in required_columns:
        if column not in df.columns:
            missing.append(column)

    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def make_pattern_flags(df):
    """
    Make one row per NP pattern with number values and attractor flags.
    """
    pattern_flags = (
        df[["np_pattern", "np1_num", "np2_num", "np3_num"]]
        .drop_duplicates("np_pattern")
        .copy()
    )

    pattern_flags["np3_v2_attractor"] = (
        pattern_flags["np3_num"] != pattern_flags["np2_num"]
    ).astype(int)

    pattern_flags["np3_copula_attractor"] = (
        pattern_flags["np3_num"] != pattern_flags["np1_num"]
    ).astype(int)

    pattern_flags["np2_copula_attractor"] = (
        pattern_flags["np2_num"] != pattern_flags["np1_num"]
    ).astype(int)

    return pattern_flags


def make_regime_flags(df):
    """
    Make one row per regime with the regime meaning.
    """
    regime_flags = (
        df[
            [
                "regime_v1v2",
                "v1_attractor_incongruent",
                "v2_attractor_incongruent",
            ]
        ]
        .drop_duplicates("regime_v1v2")
        .copy()
    )

    return regime_flags


def make_pattern_breakdown(df):
    """
    Compute critical-region accuracy by model, checkpoint, and NP pattern.
    """
    df_patterns = df.dropna(subset=["np_pattern"]).copy()

    breakdown = (
        df_patterns
        .groupby(["model", "checkpoint", "np_pattern"], as_index=False)
        .agg(
            n_items=("correct_crit", "count"),
            n_families=("family_id", "nunique"),
            acc_crit=("correct_crit", "mean"),
        )
    )

    pattern_flags = make_pattern_flags(df_patterns)

    breakdown = breakdown.merge(
        pattern_flags,
        on="np_pattern",
        how="left",
    )

    breakdown["np_pattern"] = pd.Categorical(
        breakdown["np_pattern"],
        categories=PATTERN_ORDER,
        ordered=True,
    )

    breakdown = breakdown.sort_values(
        ["model", "checkpoint", "np_pattern"]
    ).reset_index(drop=True)

    breakdown["np_pattern"] = breakdown["np_pattern"].astype(str)

    return breakdown


def make_regime_breakdown(df):
    """
    Compute critical-region accuracy by model, checkpoint, and regime.
    """
    df_regimes = df.dropna(subset=["regime_v1v2"]).copy()

    breakdown = (
        df_regimes
        .groupby(["model", "checkpoint", "regime_v1v2"], as_index=False)
        .agg(
            n_items=("correct_crit", "count"),
            n_families=("family_id", "nunique"),
            acc_crit=("correct_crit", "mean"),
        )
    )

    regime_flags = make_regime_flags(df_regimes)

    breakdown = breakdown.merge(
        regime_flags,
        on="regime_v1v2",
        how="left",
    )

    breakdown["regime_v1v2"] = pd.Categorical(
        breakdown["regime_v1v2"],
        categories=REGIME_ORDER,
        ordered=True,
    )

    breakdown = breakdown.sort_values(
        ["model", "checkpoint", "regime_v1v2"]
    ).reset_index(drop=True)

    breakdown["regime_v1v2"] = breakdown["regime_v1v2"].astype(str)

    return breakdown


def write_csv(df, path):
    """
    Write a CSV file.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"Wrote {path} ({len(df)} rows)")


def print_checks(df, pattern_breakdown, regime_breakdown):
    """
    Print checks so mistakes are easy to spot.
    """
    print()
    print("Master score rows:")
    print(len(df))

    print()
    print("Rows by checkpoint:")
    print(df["checkpoint"].value_counts().sort_index().to_string())

    print()
    print("Pattern breakdown rows:")
    print(len(pattern_breakdown))

    print()
    print("Regime breakdown rows:")
    print(len(regime_breakdown))

    print()
    print("Families per NP pattern:")
    pattern_counts = (
        df[["np_pattern", "family_id"]]
        .drop_duplicates()
        .groupby("np_pattern")["family_id"]
        .nunique()
        .reindex(PATTERN_ORDER)
    )
    print(pattern_counts.to_string())

    print()
    print("Families per regime:")
    regime_counts = (
        df[["regime_v1v2", "family_id"]]
        .drop_duplicates()
        .groupby("regime_v1v2")["family_id"]
        .nunique()
        .reindex(REGIME_ORDER)
    )
    print(regime_counts.to_string())


def main():
    parser = argparse.ArgumentParser(
        description="Make pattern and regime summary files."
    )

    parser.add_argument(
        "--master",
        type=Path,
        default=DEFAULT_MASTER,
        help="Input master score CSV.",
    )

    parser.add_argument(
        "--out-pattern-breakdown",
        "--out_pattern_breakdown",
        dest="out_pattern_breakdown",
        type=Path,
        default=DEFAULT_OUT_PATTERN_BREAKDOWN,
        help="Output path for pattern_breakdown.csv.",
    )

    parser.add_argument(
        "--out-regime-breakdown",
        "--out_regime_breakdown",
        dest="out_regime_breakdown",
        type=Path,
        default=DEFAULT_OUT_REGIME_BREAKDOWN,
        help="Output path for regime_breakdown.csv.",
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
        "correct_crit",
        "np1_num",
        "np2_num",
        "np3_num",
        "np_pattern",
        "np2_np1_incongruent",
        "v1_attractor_incongruent",
        "v2_attractor_incongruent",
        "regime_v1v2",
    ]

    check_required_columns(df, required_columns)

    pattern_breakdown = make_pattern_breakdown(df)
    regime_breakdown = make_regime_breakdown(df)

    write_csv(pattern_breakdown, args.out_pattern_breakdown)
    write_csv(regime_breakdown, args.out_regime_breakdown)

    print_checks(df, pattern_breakdown, regime_breakdown)


if __name__ == "__main__":
    main()
