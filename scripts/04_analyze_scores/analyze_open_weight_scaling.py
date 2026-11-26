#!/usr/bin/env python3

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


INPUT = Path("results/final/master_forced_choice_scores_v1.csv")
OUT_DIR = Path("results/analysis/open_weight_scaling_v1")

N_BOOT = 10_000
SEED = 20260819


PARAMETERS = {
    "gpt2": 124_000_000,
    "pythia-410m": 405_334_016,
    "llama-3.2-1b": 1_000_000_000,
    "llama-3.2-3b": 3_000_000_000,
    "llama-3.1-70b": 70_000_000_000,
    "qwen2.5-1.5b": 1_540_000_000,
    "qwen2.5-3b": 3_090_000_000,
    "qwen2.5-7b": 7_610_000_000,
    "qwen2.5-14b": 14_700_000_000,
    "qwen2.5-32b": 32_500_000_000,
    "qwen2.5-72b": 72_700_000_000,
}

QWEN_MODELS = [
    "qwen2.5-1.5b",
    "qwen2.5-3b",
    "qwen2.5-7b",
    "qwen2.5-14b",
    "qwen2.5-32b",
    "qwen2.5-72b",
]

ALL_MODELS = list(PARAMETERS.keys())

METRICS = [
    "congruent_accuracy",
    "incongruent_accuracy",
    "global_drop",
    "nearest_attractor_effect",
]


def percentile_ci(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if len(values) == 0:
        return np.nan, np.nan

    low, high = np.percentile(values, [2.5, 97.5])
    return float(low), float(high)


def compute_model_metrics(df):
    out = {}

    congruent = df.loc[
        df["global_congruence"] == "congruent",
        "counterbalanced_accuracy",
    ].mean()

    incongruent = df.loc[
        df["global_congruence"] == "incongruent",
        "counterbalanced_accuracy",
    ].mean()

    n3_match = df.loc[
        df["nearest_attractor_congruence"] == "N3_matches_N2",
        "counterbalanced_accuracy",
    ].mean()

    n3_mismatch = df.loc[
        df["nearest_attractor_congruence"] == "N3_mismatches_N2",
        "counterbalanced_accuracy",
    ].mean()

    out["congruent_accuracy"] = float(congruent)
    out["incongruent_accuracy"] = float(incongruent)
    out["global_drop"] = float(congruent - incongruent)
    out["nearest_attractor_effect"] = float(n3_match - n3_mismatch)

    return out


def rho_for_subset(metric_table, models, metric):
    x = np.array(
        [np.log10(PARAMETERS[m]) for m in models],
        dtype=float,
    )
    y = np.array(
        [metric_table[m][metric] for m in models],
        dtype=float,
    )

    result = spearmanr(x, y)

    return float(result.statistic)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(INPUT, low_memory=False)

    df = df.loc[df["provider"] == "open_weight"].copy()

    found_models = set(df["model"].unique())
    expected_models = set(ALL_MODELS)

    if found_models != expected_models:
        raise ValueError(
            "Open-weight model mismatch.\n"
            f"Expected: {sorted(expected_models)}\n"
            f"Found: {sorted(found_models)}"
        )

    counts = df.groupby("model").size()

    if not (counts == 2000).all():
        raise ValueError(
            "Expected exactly 2000 items per open-weight model:\n"
            f"{counts}"
        )

    if df["counterbalanced_accuracy"].isna().any():
        raise ValueError(
            "Missing counterbalanced_accuracy values in open-weight data."
        )

    # ------------------------------------------------------------------
    # Validate matched item structure across models.
    # ------------------------------------------------------------------

    reference_model = ALL_MODELS[0]

    reference = (
        df.loc[
            df["model"] == reference_model,
            ["base_item_id", "number_pattern"],
        ]
        .sort_values("base_item_id")
        .reset_index(drop=True)
    )

    for model in ALL_MODELS[1:]:
        current = (
            df.loc[
                df["model"] == model,
                ["base_item_id", "number_pattern"],
            ]
            .sort_values("base_item_id")
            .reset_index(drop=True)
        )

        if not current.equals(reference):
            raise ValueError(
                f"Item IDs / number patterns do not match for {model}"
            )

    patterns = sorted(reference["number_pattern"].unique())

    if len(patterns) != 8:
        raise ValueError(
            f"Expected 8 number patterns, found {len(patterns)}: {patterns}"
        )

    # ------------------------------------------------------------------
    # Observed model-level metrics.
    # ------------------------------------------------------------------

    observed_metrics = {}

    for model in ALL_MODELS:
        model_df = df.loc[df["model"] == model]
        observed_metrics[model] = compute_model_metrics(model_df)

    metric_rows = []

    for model in sorted(
        ALL_MODELS,
        key=lambda m: PARAMETERS[m],
    ):
        row = {
            "model": model,
            "parameters": PARAMETERS[model],
            "log10_parameters": np.log10(PARAMETERS[model]),
            "family": (
                "qwen2.5"
                if model.startswith("qwen2.5")
                else (
                    "llama"
                    if model.startswith("llama")
                    else (
                        "pythia"
                        if model.startswith("pythia")
                        else "gpt2"
                    )
                )
            ),
        }

        row.update(observed_metrics[model])
        metric_rows.append(row)

    metrics_df = pd.DataFrame(metric_rows)

    metrics_path = OUT_DIR / "open_weight_scaling_model_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)

    # ------------------------------------------------------------------
    # Observed correlations.
    # ------------------------------------------------------------------

    subsets = {
        "qwen_only": QWEN_MODELS,
        "all_open_weight": ALL_MODELS,
    }

    observed_rhos = {}

    for subset_name, models in subsets.items():
        for metric in METRICS:
            observed_rhos[(subset_name, metric)] = rho_for_subset(
                observed_metrics,
                models,
                metric,
            )

    # ------------------------------------------------------------------
    # Prepare fast matched bootstrap.
    #
    # Each item is represented once in the reference-model item table.
    # For every bootstrap replicate, item IDs are sampled WITH replacement
    # separately within each number pattern. The same sampled item IDs are
    # then used for every model.
    # ------------------------------------------------------------------

    item_meta = reference.copy()

    pattern_indices = {}

    for pattern in patterns:
        pattern_indices[pattern] = np.flatnonzero(
            item_meta["number_pattern"].to_numpy() == pattern
        )

    # score_matrix:
    # rows = matched base items
    # columns = models
    score_matrix = np.empty(
        (len(item_meta), len(ALL_MODELS)),
        dtype=np.float64,
    )

    # Boolean masks defining each metric.
    reference_full = (
        df.loc[
            df["model"] == reference_model,
            [
                "base_item_id",
                "global_congruence",
                "nearest_attractor_congruence",
            ],
        ]
        .sort_values("base_item_id")
        .reset_index(drop=True)
    )

    if not np.array_equal(
        reference_full["base_item_id"].to_numpy(),
        item_meta["base_item_id"].to_numpy(),
    ):
        raise ValueError("Reference metadata alignment failed.")

    congruent_mask = (
        reference_full["global_congruence"].to_numpy() == "congruent"
    )

    incongruent_mask = (
        reference_full["global_congruence"].to_numpy() == "incongruent"
    )

    n3_match_mask = (
        reference_full["nearest_attractor_congruence"].to_numpy()
        == "N3_matches_N2"
    )

    n3_mismatch_mask = (
        reference_full["nearest_attractor_congruence"].to_numpy()
        == "N3_mismatches_N2"
    )

    for j, model in enumerate(ALL_MODELS):
        current = (
            df.loc[
                df["model"] == model,
                ["base_item_id", "counterbalanced_accuracy"],
            ]
            .sort_values("base_item_id")
            .reset_index(drop=True)
        )

        if not np.array_equal(
            current["base_item_id"].to_numpy(),
            item_meta["base_item_id"].to_numpy(),
        ):
            raise ValueError(
                f"Score alignment failed for model {model}"
            )

        score_matrix[:, j] = current[
            "counterbalanced_accuracy"
        ].to_numpy(dtype=float)

    # ------------------------------------------------------------------
    # Bootstrap correlations.
    # ------------------------------------------------------------------

    rng = np.random.default_rng(SEED)

    bootstrap_rhos = {
        (subset_name, metric): []
        for subset_name in subsets
        for metric in METRICS
    }

    model_index = {
        model: i
        for i, model in enumerate(ALL_MODELS)
    }

    for b in range(N_BOOT):
        sampled_parts = []

        for pattern in patterns:
            idx = pattern_indices[pattern]

            sampled = rng.choice(
                idx,
                size=len(idx),
                replace=True,
            )

            sampled_parts.append(sampled)

        sampled_idx = np.concatenate(sampled_parts)

        sampled_scores = score_matrix[sampled_idx, :]

        sampled_congruent_mask = congruent_mask[sampled_idx]
        sampled_incongruent_mask = incongruent_mask[sampled_idx]
        sampled_n3_match_mask = n3_match_mask[sampled_idx]
        sampled_n3_mismatch_mask = n3_mismatch_mask[sampled_idx]

        boot_metrics = {}

        for j, model in enumerate(ALL_MODELS):
            scores = sampled_scores[:, j]

            congruent = scores[sampled_congruent_mask].mean()
            incongruent = scores[sampled_incongruent_mask].mean()

            n3_match = scores[sampled_n3_match_mask].mean()
            n3_mismatch = scores[sampled_n3_mismatch_mask].mean()

            boot_metrics[model] = {
                "congruent_accuracy": float(congruent),
                "incongruent_accuracy": float(incongruent),
                "global_drop": float(congruent - incongruent),
                "nearest_attractor_effect": float(
                    n3_match - n3_mismatch
                ),
            }

        for subset_name, models in subsets.items():
            for metric in METRICS:
                rho = rho_for_subset(
                    boot_metrics,
                    models,
                    metric,
                )

                bootstrap_rhos[(subset_name, metric)].append(rho)

        if (b + 1) % 1000 == 0:
            print(f"Bootstrap {b + 1}/{N_BOOT}")

    # ------------------------------------------------------------------
    # Final summary.
    # ------------------------------------------------------------------

    summary_rows = []

    for subset_name, models in subsets.items():
        for metric in METRICS:
            values = bootstrap_rhos[(subset_name, metric)]

            ci_low, ci_high = percentile_ci(values)

            summary_rows.append({
                "subset": subset_name,
                "n_models": len(models),
                "metric": metric,
                "spearman_rho": observed_rhos[
                    (subset_name, metric)
                ],
                "ci_lower": ci_low,
                "ci_upper": ci_high,
                "bootstrap_replicates": N_BOOT,
                "bootstrap_seed": SEED,
                "bootstrap_stratified": True,
                "bootstrap_stratum_field": "number_pattern",
                "matched_item_ids_across_models": True,
                "ci_method": "percentile",
                "confidence_level": 0.95,
            })

    summary_df = pd.DataFrame(summary_rows)

    summary_path = OUT_DIR / "open_weight_scaling_correlations.csv"
    summary_df.to_csv(summary_path, index=False)

    # Save bootstrap draws for auditability.
    draw_rows = []

    for subset_name in subsets:
        for metric in METRICS:
            vals = bootstrap_rhos[(subset_name, metric)]

            for i, rho in enumerate(vals, start=1):
                draw_rows.append({
                    "bootstrap_replicate": i,
                    "subset": subset_name,
                    "metric": metric,
                    "spearman_rho": rho,
                })

    draws_path = OUT_DIR / "open_weight_scaling_bootstrap_draws.csv"

    pd.DataFrame(draw_rows).to_csv(
        draws_path,
        index=False,
    )

    manifest = {
        "analysis": "open_weight_scaling_v1",
        "input": str(INPUT),
        "models_all_open_weight": ALL_MODELS,
        "models_qwen_only": QWEN_MODELS,
        "parameter_counts": PARAMETERS,
        "size_reporting_variable": "log10(total_parameters)",
        "correlation": "Spearman rank correlation",
        "metrics": METRICS,
        "bootstrap_replicates": N_BOOT,
        "bootstrap_seed": SEED,
        "bootstrap_method": (
            "Nonparametric matched-item bootstrap. "
            "Base item IDs are resampled with replacement separately "
            "within each of the eight number_pattern strata. "
            "The same sampled item IDs are used for every model "
            "within each bootstrap replicate."
        ),
        "confidence_interval": "95% percentile bootstrap interval",
        "interpretation": {
            "qwen_only": (
                "Primary within-family scaling analysis; reduces "
                "confounding between model size and model family."
            ),
            "all_open_weight": (
                "Cross-family descriptive robustness analysis; "
                "does not isolate a causal effect of parameter count."
            ),
        },
    }

    manifest_path = OUT_DIR / "open_weight_scaling_manifest.json"

    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    # ------------------------------------------------------------------
    # Console report.
    # ------------------------------------------------------------------

    print()
    print("Validation PASS")
    print(f"Open-weight models: {len(ALL_MODELS)}")
    print(f"Qwen models: {len(QWEN_MODELS)}")
    print(f"Items/model: {len(item_meta)}")
    print(f"Number-pattern strata: {len(patterns)}")
    print(f"Bootstrap replicates: {N_BOOT}")
    print(f"Seed: {SEED}")
    print()

    print(
        f"{'subset':<18}"
        f"{'metric':<28}"
        f"{'rho':>8}"
        f"{'95% CI':>23}"
    )

    for _, r in summary_df.iterrows():
        ci = (
            f"[{r['ci_lower']:.4f}, "
            f"{r['ci_upper']:.4f}]"
        )

        print(
            f"{r['subset']:<18}"
            f"{r['metric']:<28}"
            f"{r['spearman_rho']:>8.4f}"
            f"{ci:>23}"
        )

    print()
    print(f"Wrote: {metrics_path}")
    print(f"Wrote: {summary_path}")
    print(f"Wrote: {draws_path}")
    print(f"Wrote: {manifest_path}")


if __name__ == "__main__":
    main()
