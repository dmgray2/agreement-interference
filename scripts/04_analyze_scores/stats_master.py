#!/usr/bin/env python3
"""
Compute the main V2 attractor-interference summary table.

Input:
    data/scores/master_score_lists/master_scores_full.csv

Output:
    data/scores/summaries/stats_summary.csv

For each model, this script computes:

    full-continuation accuracy across all checkpoints
    critical-region accuracy across all checkpoints
    V2 critical-region accuracy when NP3 matches NP2
    V2 critical-region accuracy when NP3 mismatches NP2
    V2 drop: congruent accuracy - incongruent accuracy
    95% bootstrap confidence interval for the drop
    one-sided permutation p-value for the drop

The V2 analysis uses one item per sentence family. Duplicate V2 rows and
invalid V2 labels are rejected before inference. The default run uses a
stable seed per model and writes stats_summary_worker_independent.csv.

To reproduce the camera-ready archive, run from the repository root
with the full released master file:

    python3 scripts/04_analyze_scores/stats_master.py \
      --reproduce-archived --bootstrap 10000 --perm 10000 --workers 12 --seed 0
"""

from __future__ import annotations

import argparse
import hashlib
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
    / "stats_summary.csv"
)


def check_required_columns(df: pd.DataFrame, required_columns: list[str]) -> None:
    """Stop with a clear message if an expected column is missing."""
    missing = []

    for column in required_columns:
        if column not in df.columns:
            missing.append(column)

    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def split_work(n_total: int, n_workers: int) -> list[int]:
    """Split a number of trials across worker processes."""
    n_workers = max(1, int(n_workers))

    if n_total <= 0:
        return []

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
    """Make separate repeatable seeds for workers."""
    seed_sequence = np.random.SeedSequence(seed)
    child_seeds = seed_sequence.spawn(n_workers)

    seeds = []

    for child in child_seeds:
        seeds.append(int(child.generate_state(1)[0]))

    return seeds


def drop_score(y: np.ndarray, label: np.ndarray) -> float:
    """
    Compute congruent accuracy minus incongruent accuracy.

    label = 0 means NP3 matches NP2.
    label = 1 means NP3 mismatches NP2.
    """
    if not np.any(label == 0):
        return np.nan

    if not np.any(label == 1):
        return np.nan

    congruent = y[label == 0].mean()
    incongruent = y[label == 1].mean()

    return float(congruent - incongruent)


def bootstrap_worker(task: tuple[np.ndarray, np.ndarray, int, int]) -> np.ndarray:
    """Run part of the bootstrap."""
    y, label, n_trials, seed = task

    rng = np.random.default_rng(seed)
    n_rows = len(y)

    drops = np.empty(n_trials, dtype=float)

    for i in range(n_trials):
        sample = rng.choice(n_rows, size=n_rows, replace=True)
        drops[i] = drop_score(y[sample], label[sample])

    return drops


def permutation_worker(task: tuple[np.ndarray, np.ndarray, float, int, int]) -> int:
    """Run part of the permutation test."""
    y, label, observed_drop, n_trials, seed = task

    rng = np.random.default_rng(seed)

    hits = 0

    for _ in range(n_trials):
        shuffled_label = label.copy()
        rng.shuffle(shuffled_label)

        permuted_drop = drop_score(y, shuffled_label)

        if permuted_drop >= observed_drop:
            hits += 1

    return hits


def bootstrap_ci(
    y: np.ndarray,
    label: np.ndarray,
    n_bootstrap: int,
    n_workers: int | None,
    seed: int,
) -> tuple[float, float]:
    """Compute a 95% bootstrap confidence interval."""
    if n_bootstrap <= 0:
        return np.nan, np.nan

    if n_workers is None:
        all_drops = [bootstrap_worker((y, label, n_bootstrap, seed))]
    else:
        chunks = split_work(n_bootstrap, n_workers)
        seeds = make_worker_seeds(seed, len(chunks))

        tasks = []

        for n_trials, worker_seed in zip(chunks, seeds):
            tasks.append((y, label, n_trials, worker_seed))

        all_drops = []

        with ProcessPoolExecutor(max_workers=len(tasks)) as executor:
            futures = []

            for task in tasks:
                futures.append(executor.submit(bootstrap_worker, task))

            for future in as_completed(futures):
                all_drops.append(future.result())

    drops = np.concatenate(all_drops)
    drops = drops[~np.isnan(drops)]

    if len(drops) == 0:
        return np.nan, np.nan

    ci_low = float(np.quantile(drops, 0.025))
    ci_high = float(np.quantile(drops, 0.975))

    return ci_low, ci_high


def permutation_p_value(
    y: np.ndarray,
    label: np.ndarray,
    observed_drop: float,
    n_permutations: int,
    n_workers: int | None,
    seed: int,
) -> float:
    """
    Compute a one-sided permutation p-value.

    The p-value is the share of shuffled datasets with a V2 drop at least
    as large as the observed V2 drop.
    """
    if n_permutations <= 0:
        return np.nan

    if np.isnan(observed_drop):
        return np.nan

    if n_workers is None:
        hits = permutation_worker((y, label, observed_drop, n_permutations, seed))
        return (hits + 1) / (n_permutations + 1)

    chunks = split_work(n_permutations, n_workers)
    seeds = make_worker_seeds(seed, len(chunks))

    tasks = []

    for n_trials, worker_seed in zip(chunks, seeds):
        tasks.append((y, label, observed_drop, n_trials, worker_seed))

    hits = 0

    with ProcessPoolExecutor(max_workers=len(tasks)) as executor:
        futures = []

        for task in tasks:
            futures.append(executor.submit(permutation_worker, task))

        for future in as_completed(futures):
            hits += future.result()

    return (hits + 1) / (n_permutations + 1)


def make_v2_family_table(df_model: pd.DataFrame) -> pd.DataFrame:
    """Validate one V2 row per family and preserve the historical row order."""
    df_v2 = df_model.loc[df_model["checkpoint"] == "V2",
                         ["family_id", "correct_crit", "v2_attractor_incongruent"]]
    model = df_model["model"].iloc[0]
    duplicates = df_v2["family_id"].duplicated()
    if duplicates.any():
        family = df_v2.loc[duplicates, "family_id"].iloc[0]
        raise ValueError(f"{model}: duplicate V2 rows for family {family!r}")
    labels = df_v2["v2_attractor_incongruent"]
    if not labels.isin([0, 1]).all() or set(labels) != {0, 1}:
        raise ValueError(f"{model}: V2 labels must be binary and include both 0 and 1")
    return df_v2.sort_values("family_id").reset_index(drop=True)


def summarize_model(
    model_name: str,
    df_model: pd.DataFrame,
    n_bootstrap: int,
    n_permutations: int,
    n_workers: int | None,
    seed: int,
) -> dict[str, object]:
    """Compute all summary values for one model."""
    overall_accuracy = float(df_model["correct"].mean())
    overall_critical_accuracy = float(df_model["correct_crit"].mean())

    v2_family = make_v2_family_table(df_model)

    y = v2_family["correct_crit"].to_numpy(dtype=float)
    label = v2_family["v2_attractor_incongruent"].to_numpy(dtype=int)

    congruent_accuracy = float(y[label == 0].mean())
    incongruent_accuracy = float(y[label == 1].mean())
    observed_drop = drop_score(y, label)

    ci_low, ci_high = bootstrap_ci(
        y=y,
        label=label,
        n_bootstrap=n_bootstrap,
        n_workers=n_workers,
        seed=seed + 1000,
    )

    p_value = permutation_p_value(
        y=y,
        label=label,
        observed_drop=observed_drop,
        n_permutations=n_permutations,
        n_workers=n_workers,
        seed=seed + 2000,
    )

    return {
        "model": model_name,
        "n_items": int(len(df_model)),
        "acc_full": overall_accuracy,
        "acc_crit": overall_critical_accuracy,
        "n_v2": int(len(v2_family)),
        "acc_v2_cong_crit": congruent_accuracy,
        "acc_v2_incong_crit": incongruent_accuracy,
        "v2_drop_crit": observed_drop,
        "ci95_lo": ci_low,
        "ci95_hi": ci_high,
        "perm_p_one_sided": p_value,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute the main V2 attractor-interference summary table."
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
        help="Output summary CSV. Defaults to a name specific to the RNG mode.",
    )

    parser.add_argument(
        "--bootstrap",
        type=int,
        default=10000,
        help="Number of bootstrap samples.",
    )

    parser.add_argument(
        "--perm",
        type=int,
        default=10000,
        help="Number of permutations.",
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=12,
        help="Number of worker processes in archived reproduction mode.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed.",
    )

    parser.add_argument(
        "--reproduce-archived",
        action="store_true",
        help="Use the historical worker-dependent RNG for the camera-ready results.",
    )
    args = parser.parse_args()
    if args.out is None:
        args.out = (
            DEFAULT_OUT if args.reproduce_archived
            else DEFAULT_OUT.with_name("stats_summary_worker_independent.csv")
        )

    if not args.master.exists():
        raise FileNotFoundError(f"Could not find input file: {args.master}")

    print(f"Reading {args.master}")
    df = pd.read_csv(args.master)

    required_columns = [
        "model",
        "family_id",
        "checkpoint",
        "correct",
        "correct_crit",
        "v2_attractor_incongruent",
    ]

    check_required_columns(df, required_columns)

    df = df.dropna(subset=["model", "family_id", "checkpoint"]).copy()

    models = sorted(df["model"].unique())
    rows = []

    for i, model_name in enumerate(models):
        df_model = df[df["model"] == model_name].copy()
        if args.reproduce_archived:
            model_seed = args.seed + (i * 10000)
        else:
            digest = hashlib.sha256(f"{args.seed}:{model_name}".encode()).digest()
            model_seed = int.from_bytes(digest[:16], "big")

        print(
            f"Computing {model_name} "
            f"with {len(df_model)} item rows."
        )

        row = summarize_model(
            model_name=model_name,
            df_model=df_model,
            n_bootstrap=args.bootstrap,
            n_permutations=args.perm,
            n_workers=args.workers if args.reproduce_archived else None,
            seed=model_seed,
        )

        rows.append(row)

    out_df = pd.DataFrame(rows)

    column_order = [
        "model",
        "n_items",
        "acc_full",
        "acc_crit",
        "n_v2",
        "acc_v2_cong_crit",
        "acc_v2_incong_crit",
        "v2_drop_crit",
        "ci95_lo",
        "ci95_hi",
        "perm_p_one_sided",
    ]

    out_df = out_df[column_order]

    numeric_columns = [
        "acc_full",
        "acc_crit",
        "acc_v2_cong_crit",
        "acc_v2_incong_crit",
        "v2_drop_crit",
        "ci95_lo",
        "ci95_hi",
        "perm_p_one_sided",
    ]

    for column in numeric_columns:
        out_df[column] = out_df[column].round(6)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.out, index=False)

    print()
    print(f"Wrote {args.out}")
    print()
    print(out_df.to_string(index=False))


if __name__ == "__main__":
    main()
