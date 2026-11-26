#!/usr/bin/env python3
"""
Compute regime-level diagonal contrasts with bootstrap CIs.

Input:
    data/scores/master_score_lists/master_scores_full.csv

Output:
    data/scores/summaries/regime_contrasts.csv

For each model and checkpoint, this script computes accuracy in the four
regimes:

    (0,0), (0,1), (1,0), (1,1)

where the first value is whether NP3 mismatches NP1 and the second value is
whether NP3 mismatches NP2.

The main contrast is:

    diagonal_contrast =
        ((acc(0,0) + acc(1,1)) / 2)
        -
        ((acc(0,1) + acc(1,0)) / 2)

For V2, the diagonal regimes are the regimes where NP1 and NP2 match each
other. The off-diagonal regimes are the regimes where NP1 and NP2 mismatch.
So, at V2, this contrast estimates how much performance changes when the
structurally higher NP1 matches versus mismatches the V2 controller NP2,
averaging over whether NP3 matches or mismatches NP2.

The confidence interval uses a bootstrap: it resamples
families with replacement within each model and checkpoint, then recomputes
the contrast.

To reproduce the archived intervals, use the full released master CSV,
the default checkpoint order, and the worker count and seed below.
The worker count and the order of model/checkpoint cells affect the
bootstrap draws.

Run from the repository root:

    python3 scripts/04_analyze_scores/compute_regime_contrasts.py \
      --bootstrap 10000 \
      --workers 12 \
      --seed 0
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

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

DEFAULT_OUT = (
    REPO_ROOT
    / "data"
    / "scores"
    / "summaries"
    / "regime_contrasts.csv"
)

REGIME_ORDER = ["(0,0)", "(0,1)", "(1,0)", "(1,1)"]
DEFAULT_CHECKPOINTS = ["V2", "COPULA", "V3"]


def check_required_columns(df: pd.DataFrame, required_columns: list[str]) -> None:
    """Stop with a clear message if an expected column is missing."""
    missing = []

    for column in required_columns:
        if column not in df.columns:
            missing.append(column)

    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def split_work(n_total: int, n_workers: int) -> list[int]:
    """Split a number of bootstrap trials across workers."""
    n_workers = max(1, int(n_workers))

    base = n_total // n_workers
    extra = n_total % n_workers

    chunks = []

    for i in range(n_workers):
        n = base
        if i < extra:
            n += 1
        if n > 0:
            chunks.append(n)

    return chunks


def make_worker_seeds(seed: int, n_workers: int) -> list[int]:
    """Make separate repeatable seeds for worker processes."""
    seed_sequence = np.random.SeedSequence(seed)
    child_seeds = seed_sequence.spawn(n_workers)

    seeds = []
    for child in child_seeds:
        seeds.append(int(child.generate_state(1)[0]))

    return seeds


def regime_accuracy_from_arrays(regimes: np.ndarray, correct: np.ndarray) -> dict[str, float]:
    """Return mean critical-region accuracy for each regime."""
    out = {}

    for regime in REGIME_ORDER:
        mask = regimes == regime
        if mask.sum() == 0:
            out[regime] = np.nan
        else:
            out[regime] = float(correct[mask].mean())

    return out


def diagonal_contrast_from_map(acc: dict[str, float]) -> float:
    """Compute the diagonal-minus-off-diagonal regime contrast."""
    diagonal_mean = (acc["(0,0)"] + acc["(1,1)"]) / 2.0
    off_diagonal_mean = (acc["(0,1)"] + acc["(1,0)"]) / 2.0

    return float(diagonal_mean - off_diagonal_mean)


def make_family_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse to one row per family and regime.

    In the current benchmark, each model/checkpoint/family cell should already
    have one row. The CLI validates that assumption before analysis.
    """
    family_df = (
        df.groupby(["family_id", "regime_v1v2"], as_index=False)
        .agg(correct_crit=("correct_crit", "mean"))
    )

    return family_df


def bootstrap_worker(task: tuple[np.ndarray, np.ndarray, int, int]) -> np.ndarray:
    """Run part of the bootstrap for one model/checkpoint cell."""
    regimes, correct, n_trials, seed = task

    rng = np.random.default_rng(seed)
    n_families = len(correct)
    values = []

    for _ in range(n_trials):
        sample = rng.choice(n_families, size=n_families, replace=True)

        sampled_regimes = regimes[sample]
        sampled_correct = correct[sample]

        acc = regime_accuracy_from_arrays(sampled_regimes, sampled_correct)
        contrast = diagonal_contrast_from_map(acc)

        if not np.isnan(contrast):
            values.append(contrast)

    return np.asarray(values, dtype=float)


def bootstrap_ci(
    regimes: np.ndarray,
    correct: np.ndarray,
    n_bootstrap: int,
    n_workers: int,
    seed: int,
) -> tuple[float, float]:
    """Compute a 95 percent bootstrap CI."""
    if n_bootstrap <= 0:
        return np.nan, np.nan

    chunks = split_work(n_bootstrap, n_workers)
    seeds = make_worker_seeds(seed, len(chunks))

    tasks = []
    for n_trials, worker_seed in zip(chunks, seeds):
        tasks.append((regimes, correct, n_trials, worker_seed))

    all_values = []

    with ProcessPoolExecutor(max_workers=len(tasks)) as executor:
        futures = []
        for task in tasks:
            futures.append(executor.submit(bootstrap_worker, task))

        for future in as_completed(futures):
            values = future.result()
            if len(values) > 0:
                all_values.append(values)

    if not all_values:
        return np.nan, np.nan

    boot_values = np.concatenate(all_values)

    ci_low = float(np.quantile(boot_values, 0.025))
    ci_high = float(np.quantile(boot_values, 0.975))

    return ci_low, ci_high


def summarize_model_checkpoint(
    model: str,
    checkpoint: str,
    df_sub: pd.DataFrame,
    n_bootstrap: int,
    n_workers: int,
    seed: int,
) -> dict[str, object]:
    """Compute the regime contrast for one model/checkpoint pair."""
    missing = [regime for regime in REGIME_ORDER if regime not in set(df_sub["regime_v1v2"])]
    if missing:
        raise ValueError(f"{model}/{checkpoint} is missing regimes: {missing}")

    family_df = make_family_table(df_sub)

    regimes = family_df["regime_v1v2"].astype(str).to_numpy()
    correct = family_df["correct_crit"].to_numpy(dtype=float)

    acc = regime_accuracy_from_arrays(regimes, correct)
    contrast = diagonal_contrast_from_map(acc)

    ci_low, ci_high = bootstrap_ci(
        regimes=regimes,
        correct=correct,
        n_bootstrap=n_bootstrap,
        n_workers=n_workers,
        seed=seed,
    )

    n_by_regime = (
        family_df.groupby("regime_v1v2")["family_id"]
        .nunique()
        .to_dict()
    )

    row = {
        "model": model,
        "checkpoint": checkpoint,
        "n_families": int(family_df["family_id"].nunique()),
        "acc_00": acc["(0,0)"],
        "acc_01": acc["(0,1)"],
        "acc_10": acc["(1,0)"],
        "acc_11": acc["(1,1)"],
        "n_00": int(n_by_regime.get("(0,0)", 0)),
        "n_01": int(n_by_regime.get("(0,1)", 0)),
        "n_10": int(n_by_regime.get("(1,0)", 0)),
        "n_11": int(n_by_regime.get("(1,1)", 0)),
        "diagonal_mean": (acc["(0,0)"] + acc["(1,1)"]) / 2.0,
        "off_diagonal_mean": (acc["(0,1)"] + acc["(1,0)"]) / 2.0,
        "diagonal_contrast": contrast,
        "ci95_lo": ci_low,
        "ci95_hi": ci_high,
    }

    return row


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute regime-level diagonal contrasts."
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
        default=DEFAULT_CHECKPOINTS,
        help="Checkpoints to summarize.",
    )

    parser.add_argument(
        "--bootstrap",
        type=int,
        default=5000,
        help="Number of bootstrap samples.",
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=12,
        help="Number of worker processes.",
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

    required_columns = [
        "model",
        "family_id",
        "checkpoint",
        "correct_crit",
        "regime_v1v2",
    ]

    check_required_columns(df, required_columns)

    missing = df[required_columns].isna().sum()
    if missing.any():
        raise ValueError(f"Missing required values: {missing[missing > 0].to_dict()}")

    unexpected = sorted(set(df["regime_v1v2"]) - set(REGIME_ORDER), key=str)
    if unexpected:
        raise ValueError(f"Unexpected regime labels: {unexpected!r}. Expected {REGIME_ORDER}.")

    invalid = ~df["correct_crit"].isin([0, 1])
    if invalid.any():
        raise ValueError(f"correct_crit must be 0 or 1; found {invalid.sum()} invalid rows.")

    family_keys = ["model", "checkpoint", "family_id"]
    duplicates = df.duplicated(family_keys, keep=False)
    if duplicates.any():
        example = df.loc[duplicates, family_keys].iloc[0].to_dict()
        raise ValueError(f"Duplicate model/checkpoint/family rows: {example}")

    rows = []
    models = sorted(df["model"].unique())

    cell_number = 0

    for checkpoint in args.checkpoints:
        df_checkpoint = df[df["checkpoint"] == checkpoint].copy()

        if len(df_checkpoint) == 0:
            print(f"Skipping missing checkpoint: {checkpoint}")
            continue

        for model in models:
            df_sub = df_checkpoint[df_checkpoint["model"] == model].copy()

            if len(df_sub) == 0:
                continue

            cell_number += 1
            cell_seed = args.seed + (cell_number * 10000)

            print(f"[{cell_number}] {checkpoint} | {model}")

            row = summarize_model_checkpoint(
                model=model,
                checkpoint=checkpoint,
                df_sub=df_sub,
                n_bootstrap=args.bootstrap,
                n_workers=args.workers,
                seed=cell_seed,
            )

            rows.append(row)

    out_df = pd.DataFrame(rows)

    column_order = [
        "model",
        "checkpoint",
        "n_families",
        "acc_00",
        "acc_01",
        "acc_10",
        "acc_11",
        "n_00",
        "n_01",
        "n_10",
        "n_11",
        "diagonal_mean",
        "off_diagonal_mean",
        "diagonal_contrast",
        "ci95_lo",
        "ci95_hi",
    ]

    out_df = out_df[column_order]

    numeric_columns = [
        "acc_00",
        "acc_01",
        "acc_10",
        "acc_11",
        "diagonal_mean",
        "off_diagonal_mean",
        "diagonal_contrast",
        "ci95_lo",
        "ci95_hi",
    ]

    for column in numeric_columns:
        out_df[column] = out_df[column].round(4)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.out, index=False)

    print()
    print(f"Wrote {args.out}")
    print(f"Rows: {len(out_df)}")
    print()
    print(out_df.to_string(index=False))


if __name__ == "__main__":
    main()
