#!/usr/bin/env python3
"""
analyze_common_token_items.py

Reproducible tokenization-robustness analysis for the CE2 V2 results.

This script DOES NOT load or rerun any language model. It DOES NOT recompute
log probabilities. It reads the already-scored item-level master CSV and uses:

    correct_crit
    crit_start
    crit_end
    v2_attractor_incongruent

The tokenization criterion is model-specific:

    critical_span_length = crit_end - crit_start
    single_token_eligible = 1 when critical_span_length == 1

It constructs three analysis sets:

1. full_v2_2000
   All 2,000 V2 families; no tokenization exclusion.

2. common_all_models
   Families that are single-token eligible for every model in the master file.

3. common_non_pythia
   Families that are single-token eligible for every model except pythia410.
   Pythia is not analyzed in this subset.

For each model and subset, the script computes:
    congruent accuracy
    incongruent accuracy
    drop = congruent - incongruent
    percentile family bootstrap 95% CI (20,000 resamples by default)
    exact one-sided conditional permutation p-value
    Holm-adjusted p-value within the subset

The exact p-value conditions on:
    total number of correct responses,
    number of congruent families,
    number of incongruent families.

Under label permutation, the number of correct responses assigned to the
congruent group has a hypergeometric distribution.

Outputs give the exact included and excluded family IDs, so the filter is
fully auditable.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Set, Tuple

import numpy as np
from scipy.stats import hypergeom


DEFAULT_EXCLUDED_MODEL = "pythia410"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_v2_rows(master_path: Path) -> Tuple[List[str], Dict[str, Dict[str, dict]]]:
    """Read one V2 row per model and family, with strict validation."""
    rows_by_model: Dict[str, Dict[str, dict]] = {}

    with master_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {
            "model",
            "family_id",
            "checkpoint",
            "correct_crit",
            "crit_start",
            "crit_end",
            "v2_attractor_incongruent",
            "np_pattern",
        }
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing required columns: {sorted(missing)}")

        for raw in reader:
            if raw["checkpoint"] != "V2":
                continue

            model = raw["model"]
            family_id = raw["family_id"]

            correct = int(float(raw["correct_crit"]))
            if correct not in (0, 1):
                raise ValueError(
                    f"correct_crit must be binary: {model}, {family_id}, {correct}"
                )

            label = int(float(raw["v2_attractor_incongruent"]))
            if label not in (0, 1):
                raise ValueError(
                    f"v2_attractor_incongruent must be binary: "
                    f"{model}, {family_id}, {label}"
                )

            crit_start = int(float(raw["crit_start"]))
            crit_end = int(float(raw["crit_end"]))
            span_len = crit_end - crit_start

            model_rows = rows_by_model.setdefault(model, {})
            if family_id in model_rows:
                raise ValueError(
                    f"Duplicate V2 row for model={model}, family={family_id}"
                )

            model_rows[family_id] = {
                "correct_crit": correct,
                "label": label,
                "crit_start": crit_start,
                "crit_end": crit_end,
                "span_len": span_len,
                "np_pattern": raw["np_pattern"],
            }

    models = sorted(rows_by_model)
    if not models:
        raise ValueError("No V2 rows found.")

    family_sets = {m: set(rows_by_model[m]) for m in models}
    first_families = family_sets[models[0]]

    for model in models:
        if family_sets[model] != first_families:
            missing_here = sorted(first_families - family_sets[model])
            extra_here = sorted(family_sets[model] - first_families)
            raise ValueError(
                f"Models do not have identical V2 family IDs. {model}: "
                f"missing={len(missing_here)}, extra={len(extra_here)}"
            )

    # Metadata should agree across models for the same family.
    for family_id in sorted(first_families):
        labels = {rows_by_model[m][family_id]["label"] for m in models}
        patterns = {rows_by_model[m][family_id]["np_pattern"] for m in models}
        if len(labels) != 1:
            raise ValueError(f"Label disagreement across models: {family_id}")
        if len(patterns) != 1:
            raise ValueError(f"Pattern disagreement across models: {family_id}")

    return models, rows_by_model


def intersection(sets: Iterable[Set[str]]) -> Set[str]:
    sets = list(sets)
    if not sets:
        return set()
    return set.intersection(*sets)


def exact_one_sided_permutation_p(y: np.ndarray, label: np.ndarray) -> float:
    """
    Exact conditional label-permutation p-value for:
        mean(y | label=0) - mean(y | label=1) > 0
    """
    y = np.asarray(y, dtype=int)
    label = np.asarray(label, dtype=int)

    n_congruent = int((label == 0).sum())
    total_correct = int(y.sum())
    correct_congruent = int(y[label == 0].sum())
    n_total = len(y)

    return float(
        hypergeom.sf(
            correct_congruent - 1,
            n_total,
            total_correct,
            n_congruent,
        )
    )


def monte_carlo_label_permutation_p(
    y: np.ndarray,
    label: np.ndarray,
    observed_drop: float,
    n_permutations: int,
    seed: int,
) -> float:
    """Monte Carlo one-sided label-permutation p-value.

    This mirrors the logic of stats_master.py: shuffle the congruence labels,
    recompute congruent minus incongruent accuracy, and count shuffled drops
    at least as large as the observed drop.
    """
    if n_permutations <= 0:
        return math.nan

    y = np.asarray(y, dtype=float)
    label = np.asarray(label, dtype=int)
    rng = np.random.default_rng(seed)
    hits = 0

    for _ in range(n_permutations):
        shuffled = label.copy()
        rng.shuffle(shuffled)
        congruent = y[shuffled == 0]
        incongruent = y[shuffled == 1]
        permuted_drop = float(congruent.mean() - incongruent.mean())
        if permuted_drop >= observed_drop:
            hits += 1

    return (hits + 1) / (n_permutations + 1)


def bootstrap_ci(
    y: np.ndarray,
    label: np.ndarray,
    n_bootstrap: int,
    seed: int,
    batch_size: int = 2000,
) -> Tuple[float, float]:
    """Percentile family bootstrap CI for congruent minus incongruent accuracy."""
    y = np.asarray(y, dtype=float)
    label = np.asarray(label, dtype=int)
    n = len(y)
    rng = np.random.default_rng(seed)
    values: List[np.ndarray] = []

    for start in range(0, n_bootstrap, batch_size):
        batch = min(batch_size, n_bootstrap - start)
        indices = rng.integers(0, n, size=(batch, n))
        sampled_y = y[indices]
        sampled_label = label[indices]

        congruent_mask = sampled_label == 0
        incongruent_mask = sampled_label == 1

        n_congruent = congruent_mask.sum(axis=1)
        n_incongruent = incongruent_mask.sum(axis=1)

        valid = (n_congruent > 0) & (n_incongruent > 0)

        drops = np.full(batch, np.nan, dtype=float)
        drops[valid] = (
            (sampled_y * congruent_mask).sum(axis=1)[valid] / n_congruent[valid]
            -
            (sampled_y * incongruent_mask).sum(axis=1)[valid]
            / n_incongruent[valid]
        )
        values.append(drops)

    boot = np.concatenate(values)
    boot = boot[~np.isnan(boot)]
    if len(boot) == 0:
        return math.nan, math.nan

    lo, hi = np.quantile(boot, [0.025, 0.975])
    return float(lo), float(hi)


def holm_adjust(p_values: Sequence[float]) -> List[float]:
    """Holm family-wise adjusted p-values."""
    n = len(p_values)
    order = sorted(range(n), key=lambda i: p_values[i])
    adjusted_sorted: List[float] = []
    running_max = 0.0

    for rank, index in enumerate(order):
        adjusted = (n - rank) * p_values[index]
        running_max = max(running_max, adjusted)
        adjusted_sorted.append(min(1.0, running_max))

    output = [math.nan] * n
    for index, adjusted in zip(order, adjusted_sorted):
        output[index] = adjusted
    return output


def analyze_subset(
    subset_name: str,
    family_ids: Set[str],
    analysis_models: List[str],
    rows_by_model: Dict[str, Dict[str, dict]],
    n_bootstrap: int,
    n_permutations: int,
    base_seed: int,
) -> List[dict]:
    rows: List[dict] = []

    for model_index, model in enumerate(analysis_models):
        ordered_families = sorted(family_ids)
        y = np.asarray(
            [rows_by_model[model][fam]["correct_crit"] for fam in ordered_families],
            dtype=int,
        )
        label = np.asarray(
            [rows_by_model[model][fam]["label"] for fam in ordered_families],
            dtype=int,
        )

        congruent = y[label == 0]
        incongruent = y[label == 1]

        acc_congruent = float(congruent.mean())
        acc_incongruent = float(incongruent.mean())
        drop = acc_congruent - acc_incongruent

        ci_low, ci_high = bootstrap_ci(
            y=y,
            label=label,
            n_bootstrap=n_bootstrap,
            seed=base_seed + model_index,
        )
        p_exact = exact_one_sided_permutation_p(y, label)
        p_monte_carlo = monte_carlo_label_permutation_p(
            y=y,
            label=label,
            observed_drop=drop,
            n_permutations=n_permutations,
            seed=base_seed + 10000 + model_index,
        )

        rows.append(
            {
                "subset": subset_name,
                "model": model,
                "n_families": len(family_ids),
                "n_congruent": len(congruent),
                "n_incongruent": len(incongruent),
                "correct_congruent": int(congruent.sum()),
                "correct_incongruent": int(incongruent.sum()),
                "acc_congruent": acc_congruent,
                "acc_incongruent": acc_incongruent,
                "drop_congruent_minus_incongruent": drop,
                "bootstrap_ci95_low": ci_low,
                "bootstrap_ci95_high": ci_high,
                "p_exact_one_sided": p_exact,
                "p_monte_carlo_one_sided": p_monte_carlo,
            }
        )

    adjusted_exact = holm_adjust([row["p_exact_one_sided"] for row in rows])
    adjusted_mc = holm_adjust([row["p_monte_carlo_one_sided"] for row in rows])
    for row, p_holm_exact, p_holm_mc in zip(rows, adjusted_exact, adjusted_mc):
        row["p_holm_exact_within_subset"] = p_holm_exact
        row["p_holm_monte_carlo_within_subset"] = p_holm_mc
        row["significant_exact_unadjusted_0_05"] = int(
            row["p_exact_one_sided"] < 0.05
        )
        row["significant_exact_holm_0_05"] = int(p_holm_exact < 0.05)
        row["significant_monte_carlo_unadjusted_0_05"] = int(
            row["p_monte_carlo_one_sided"] < 0.05
        )
        row["significant_monte_carlo_holm_0_05"] = int(p_holm_mc < 0.05)

    return rows


def write_csv(path: Path, rows: List[dict], fieldnames: List[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        if not rows:
            raise ValueError(f"Cannot infer columns for empty output: {path}")
        fieldnames = list(rows[0].keys())

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--master", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--excluded-model", default=DEFAULT_EXCLUDED_MODEL)
    parser.add_argument("--bootstrap", type=int, default=20000)
    parser.add_argument("--permutations", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260712)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    models, rows_by_model = read_v2_rows(args.master)
    all_families = set(rows_by_model[models[0]])

    eligible_by_model = {
        model: {
            family_id
            for family_id, row in rows_by_model[model].items()
            if row["span_len"] == 1
        }
        for model in models
    }

    common_all = intersection(eligible_by_model[m] for m in models)

    non_pythia_models = [m for m in models if m != args.excluded_model]
    if args.excluded_model not in models:
        raise ValueError(
            f"Excluded model {args.excluded_model!r} is not present. Models: {models}"
        )
    common_non_pythia = intersection(
        eligible_by_model[m] for m in non_pythia_models
    )

    # Model-level counts.
    model_count_rows = []
    for model in models:
        eligible = eligible_by_model[model]
        model_count_rows.append(
            {
                "model": model,
                "v2_total_families": len(all_families),
                "single_token_eligible": len(eligible),
                "not_single_token_for_this_model": len(all_families - eligible),
                "eligible_retention_rate": len(eligible) / len(all_families),
                "in_all_model_analysis": 1,
                "in_non_pythia_analysis": int(model in non_pythia_models),
            }
        )
    write_csv(args.out_dir / "model_filter_counts.csv", model_count_rows)

    # One row per family, including every model-specific span length.
    family_status_rows = []
    for family_id in sorted(all_families):
        representative = rows_by_model[models[0]][family_id]
        row = {
            "family_id": family_id,
            "np_pattern": representative["np_pattern"],
            "v2_attractor_incongruent": representative["label"],
            "included_full_v2_2000": 1,
            "included_common_all_models": int(family_id in common_all),
            "included_common_non_pythia": int(family_id in common_non_pythia),
        }
        failing_models = []
        for model in models:
            span_len = rows_by_model[model][family_id]["span_len"]
            row[f"span_len__{model}"] = span_len
            row[f"single_token__{model}"] = int(span_len == 1)
            if span_len != 1:
                failing_models.append(model)
        row["models_failing_single_token"] = ";".join(failing_models)
        family_status_rows.append(row)

    family_fields = [
        "family_id",
        "np_pattern",
        "v2_attractor_incongruent",
        "included_full_v2_2000",
        "included_common_all_models",
        "included_common_non_pythia",
    ]
    for model in models:
        family_fields.extend([f"span_len__{model}", f"single_token__{model}"])
    family_fields.append("models_failing_single_token")
    write_csv(
        args.out_dir / "family_filter_status.csv",
        family_status_rows,
        family_fields,
    )

    # Explicit included-family lists.
    common_all_rows = [
        {
            "family_id": fam,
            "np_pattern": rows_by_model[models[0]][fam]["np_pattern"],
            "v2_attractor_incongruent": rows_by_model[models[0]][fam]["label"],
        }
        for fam in sorted(common_all)
    ]
    write_csv(
        args.out_dir / "included_families_common_all_models.csv",
        common_all_rows,
    )

    common_non_pythia_rows = [
        {
            "family_id": fam,
            "np_pattern": rows_by_model[models[0]][fam]["np_pattern"],
            "v2_attractor_incongruent": rows_by_model[models[0]][fam]["label"],
        }
        for fam in sorted(common_non_pythia)
    ]
    write_csv(
        args.out_dir / "included_families_common_non_pythia.csv",
        common_non_pythia_rows,
    )

    excluded_all_rows = [
        {
            "family_id": row["family_id"],
            "np_pattern": row["np_pattern"],
            "v2_attractor_incongruent": row["v2_attractor_incongruent"],
            "models_failing_single_token": row["models_failing_single_token"],
        }
        for row in family_status_rows
        if not row["included_common_all_models"]
    ]
    write_csv(
        args.out_dir / "excluded_from_common_all_models.csv",
        excluded_all_rows,
    )

    excluded_non_pythia_rows = [
        {
            "family_id": row["family_id"],
            "np_pattern": row["np_pattern"],
            "v2_attractor_incongruent": row["v2_attractor_incongruent"],
            "models_failing_single_token": ";".join(
                model
                for model in row["models_failing_single_token"].split(";")
                if model and model != args.excluded_model
            ),
        }
        for row in family_status_rows
        if not row["included_common_non_pythia"]
    ]
    write_csv(
        args.out_dir / "excluded_from_common_non_pythia.csv",
        excluded_non_pythia_rows,
    )

    # Pattern and congruence counts for each family set.
    subset_count_rows = []
    subset_definitions = [
        ("full_v2_2000", all_families),
        ("common_all_models", common_all),
        ("common_non_pythia", common_non_pythia),
    ]
    for subset_name, families in subset_definitions:
        pattern_counts = Counter(
            rows_by_model[models[0]][fam]["np_pattern"] for fam in families
        )
        label_counts = Counter(
            rows_by_model[models[0]][fam]["label"] for fam in families
        )
        for pattern in sorted(pattern_counts):
            subset_count_rows.append(
                {
                    "subset": subset_name,
                    "count_type": "np_pattern",
                    "level": pattern,
                    "n_families": pattern_counts[pattern],
                }
            )
        subset_count_rows.extend(
            [
                {
                    "subset": subset_name,
                    "count_type": "v2_attractor",
                    "level": "congruent_0",
                    "n_families": label_counts[0],
                },
                {
                    "subset": subset_name,
                    "count_type": "v2_attractor",
                    "level": "incongruent_1",
                    "n_families": label_counts[1],
                },
            ]
        )
    write_csv(args.out_dir / "subset_composition_counts.csv", subset_count_rows)

    # Statistical results.
    full_results = analyze_subset(
        subset_name="full_v2_2000",
        family_ids=all_families,
        analysis_models=models,
        rows_by_model=rows_by_model,
        n_bootstrap=args.bootstrap,
        n_permutations=args.permutations,
        base_seed=args.seed,
    )
    common_all_results = analyze_subset(
        subset_name="common_all_models",
        family_ids=common_all,
        analysis_models=models,
        rows_by_model=rows_by_model,
        n_bootstrap=args.bootstrap,
        n_permutations=args.permutations,
        base_seed=args.seed + 1000,
    )
    common_non_pythia_results = analyze_subset(
        subset_name="common_non_pythia",
        family_ids=common_non_pythia,
        analysis_models=non_pythia_models,
        rows_by_model=rows_by_model,
        n_bootstrap=args.bootstrap,
        n_permutations=args.permutations,
        base_seed=args.seed + 2000,
    )

    write_csv(args.out_dir / "results_full_v2_2000.csv", full_results)
    write_csv(
        args.out_dir / "results_common_all_models.csv",
        common_all_results,
    )
    write_csv(
        args.out_dir / "results_common_non_pythia.csv",
        common_non_pythia_results,
    )

    # Machine-readable run summary.
    run_summary_rows = [
        {"key": "input_master", "value": str(args.master)},
        {"key": "input_master_sha256", "value": sha256_file(args.master)},
        {"key": "checkpoint_analyzed", "value": "V2"},
        {"key": "models_in_master", "value": ";".join(models)},
        {"key": "n_models", "value": len(models)},
        {"key": "n_v2_families_full", "value": len(all_families)},
        {"key": "single_token_rule", "value": "crit_end - crit_start == 1"},
        {"key": "n_common_all_models", "value": len(common_all)},
        {
            "key": "non_pythia_excluded_model",
            "value": args.excluded_model,
        },
        {
            "key": "n_common_non_pythia",
            "value": len(common_non_pythia),
        },
        {"key": "bootstrap_resamples", "value": args.bootstrap},
        {"key": "monte_carlo_permutations", "value": args.permutations},
        {"key": "base_seed", "value": args.seed},
        {
            "key": "p_value_methods",
            "value": (
                "exact one-sided conditional label permutation (hypergeometric); "
                "Monte Carlo one-sided label permutation"
            ),
        },
        {
            "key": "rescored_models",
            "value": "0; existing correct_crit/log-probability columns only",
        },
        {
            "key": "source_master_modified",
            "value": "no",
        },
    ]
    write_csv(args.out_dir / "run_summary.csv", run_summary_rows)

    # Output checks printed to terminal.
    print(f"Input SHA256: {sha256_file(args.master)}")
    print(f"Models: {len(models)}")
    print(f"Full V2 families: {len(all_families)}")
    print(f"Common to all models: {len(common_all)}")
    print(
        f"Common after excluding {args.excluded_model}: "
        f"{len(common_non_pythia)}"
    )
    print(f"Wrote outputs to {args.out_dir}")


if __name__ == "__main__":
    main()
