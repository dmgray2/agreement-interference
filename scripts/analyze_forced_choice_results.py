#!/usr/bin/env python3

import argparse
import csv
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Callable


DEFAULT_BOOTSTRAP_REPLICATES = 10_000
DEFAULT_BOOTSTRAP_SEED = 20260728


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue

            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON on line {line_number} of {path}"
                ) from exc

    return rows


def write_json(
    path: Path,
    value: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as output:
        json.dump(
            value,
            output,
            ensure_ascii=False,
            indent=2,
        )
        output.write("\n")


def write_csv(
    path: Path,
    rows: list[dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        raise ValueError("Cannot write an empty CSV file")

    fieldnames: list[str] = []

    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)

    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as output:
        writer = csv.DictWriter(
            output,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(rows)


def validate_item_rows(
    rows: list[dict[str, Any]],
) -> tuple[str, str]:
    if len(rows) != 2000:
        raise ValueError(
            f"Expected 2,000 item rows, found {len(rows)}"
        )

    base_item_ids = [
        row.get("base_item_id")
        for row in rows
    ]

    if len(base_item_ids) != len(set(base_item_ids)):
        raise ValueError(
            "Duplicate base_item_id values found"
        )

    providers = {
        row.get("provider")
        for row in rows
    }

    models = {
        row.get("requested_model")
        for row in rows
    }

    if len(providers) != 1:
        raise ValueError(
            f"Expected one provider, found {sorted(providers)}"
        )

    if len(models) != 1:
        raise ValueError(
            f"Expected one requested model, found {sorted(models)}"
        )

    required_fields = (
        "base_item_id",
        "source_family_id",
        "number_pattern",
        "cue_regime",
        "global_congruence",
        "nearest_attractor_congruence",
        "counterbalanced_accuracy",
        "strict_paired_correct",
        "order_inconsistent",
        "technical_missing",
    )

    for row_number, row in enumerate(rows, start=1):
        missing = [
            field
            for field in required_fields
            if field not in row
        ]

        if missing:
            raise ValueError(
                f"Row {row_number} is missing fields: {missing}"
            )

        score = row["counterbalanced_accuracy"]

        if score not in (0, 0.5, 1, None):
            raise ValueError(
                f"Unexpected score {score!r} in row {row_number}"
            )

    provider = next(iter(providers))
    model = next(iter(models))

    return provider, model


def nonmissing_rows(
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    return [
        row
        for row in rows
        if row["counterbalanced_accuracy"] is not None
    ]


def safe_mean(values: list[float]) -> float | None:
    if not values:
        return None

    return mean(values)


def proportion(
    rows: list[dict[str, Any]],
    field: str,
) -> float | None:
    values = [
        row[field]
        for row in rows
        if row[field] is not None
    ]

    if not values:
        return None

    return mean(values)


def summarize_subset(
    rows: list[dict[str, Any]],
    provider: str,
    model: str,
    grouping_variable: str,
    grouping_value: str,
) -> dict[str, Any]:
    available = nonmissing_rows(rows)

    scores = [
        float(row["counterbalanced_accuracy"])
        for row in available
    ]

    return {
        "provider": provider,
        "requested_model": model,
        "grouping_variable": grouping_variable,
        "grouping_value": grouping_value,
        "n_total": len(rows),
        "n_available": len(available),
        "n_technical_missing": len(rows) - len(available),
        "mean_counterbalanced_accuracy": safe_mean(scores),
        "mean_strict_paired_correct": proportion(
            available,
            "strict_paired_correct",
        ),
        "mean_order_inconsistent": proportion(
            available,
            "order_inconsistent",
        ),
    }


def make_descriptive_rows(
    rows: list[dict[str, Any]],
    provider: str,
    model: str,
) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []

    summaries.append(
        summarize_subset(
            rows=rows,
            provider=provider,
            model=model,
            grouping_variable="overall",
            grouping_value="all",
        )
    )

    grouping_fields = (
        "global_congruence",
        "cue_regime",
        "number_pattern",
        "nearest_attractor_congruence",
    )

    for field in grouping_fields:
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)

        for row in rows:
            grouped[str(row[field])].append(row)

        for value in sorted(grouped):
            summaries.append(
                summarize_subset(
                    rows=grouped[value],
                    provider=provider,
                    model=model,
                    grouping_variable=field,
                    grouping_value=value,
                )
            )

    return summaries


def infer_congruence_labels(
    rows: list[dict[str, Any]],
) -> tuple[str, str]:
    values = {
        str(row["global_congruence"])
        for row in rows
    }

    candidate_pairs = (
        ("congruent", "incongruent"),
        ("globally_congruent", "globally_incongruent"),
        ("True", "False"),
        ("1", "0"),
    )

    for congruent_label, incongruent_label in candidate_pairs:
        if {
            congruent_label,
            incongruent_label,
        }.issubset(values):
            return congruent_label, incongruent_label

    raise ValueError(
        "Could not infer the congruent and incongruent labels. "
        f"Observed values: {sorted(values)}"
    )


def condition_mean(
    rows: list[dict[str, Any]],
    field: str,
    value: str,
) -> float | None:
    scores = [
        float(row["counterbalanced_accuracy"])
        for row in rows
        if (
            row["counterbalanced_accuracy"] is not None
            and str(row[field]) == value
        )
    ]

    return safe_mean(scores)


def global_drop(
    rows: list[dict[str, Any]],
    congruent_label: str,
    incongruent_label: str,
) -> float | None:
    congruent_mean = condition_mean(
        rows,
        field="global_congruence",
        value=congruent_label,
    )

    incongruent_mean = condition_mean(
        rows,
        field="global_congruence",
        value=incongruent_label,
    )

    if (
        congruent_mean is None
        or incongruent_mean is None
    ):
        return None

    return congruent_mean - incongruent_mean


def percentile(
    sorted_values: list[float],
    probability: float,
) -> float:
    if not sorted_values:
        raise ValueError(
            "Cannot calculate a percentile of an empty list"
        )

    if not 0 <= probability <= 1:
        raise ValueError(
            "Probability must be between 0 and 1"
        )

    position = probability * (len(sorted_values) - 1)
    lower_index = math.floor(position)
    upper_index = math.ceil(position)

    if lower_index == upper_index:
        return sorted_values[lower_index]

    lower_value = sorted_values[lower_index]
    upper_value = sorted_values[upper_index]
    weight = position - lower_index

    return lower_value + weight * (
        upper_value - lower_value
    )

def stratified_item_bootstrap(
    rows: list[dict[str, Any]],
    statistic: Callable[[list[dict[str, Any]]], float | None],
    stratum_field: str,
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    """
    Preregistered bootstrap for forced-choice analyses.

    Resample complete item pairs with replacement separately
    within each of the eight N1-N2-N3 number configurations,
    preserving the observed number of available items in each
    configuration.
    """
    available_rows = nonmissing_rows(rows)

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for row in available_rows:
        grouped[str(row[stratum_field])].append(row)

    strata = sorted(grouped)

    if len(strata) != 8:
        raise ValueError(
            f"Expected 8 bootstrap strata in {stratum_field}, "
            f"found {len(strata)}: {strata}"
        )

    rng = random.Random(seed)
    bootstrap_values: list[float] = []
    skipped_replicates = 0

    for _ in range(replicates):
        sampled_rows: list[dict[str, Any]] = []

        for stratum in strata:
            stratum_rows = grouped[stratum]

            sampled_rows.extend(
                rng.choices(
                    stratum_rows,
                    k=len(stratum_rows),
                )
            )

        value = statistic(sampled_rows)

        if value is None:
            skipped_replicates += 1
            continue

        bootstrap_values.append(value)

    if not bootstrap_values:
        raise ValueError(
            "All bootstrap replicates were undefined"
        )

    bootstrap_values.sort()

    return {
        "resampling_unit": "item",
        "stratified": True,
        "stratum_field": stratum_field,
        "strata": strata,
        "stratum_counts": {
            stratum: len(grouped[stratum])
            for stratum in strata
        },
        "requested_replicates": replicates,
        "successful_replicates": len(bootstrap_values),
        "skipped_replicates": skipped_replicates,
        "seed": seed,
        "ci_method": "percentile",
        "confidence_level": 0.95,
        "ci_lower": percentile(
            bootstrap_values,
            0.025,
        ),
        "ci_upper": percentile(
            bootstrap_values,
            0.975,
        ),
    }
def build_secondary_bootstrap_analysis(
    rows: list[dict[str, Any]],
    provider: str,
    model: str,
    replicates: int,
    seed: int,
) -> tuple[
    list[dict[str, Any]],
    dict[str, Any],
]:
    """
    Planned secondary forced-choice analyses.

    Uses one shared set of stratified item bootstrap resamples
    for all subgroup means and the N3-match contrast.

    Complete item pairs are resampled with replacement
    separately within each of the eight N1-N2-N3 number
    configurations.
    """

    descriptive_rows = make_descriptive_rows(
        rows=rows,
        provider=provider,
        model=model,
    )
    group_specs = [
        (
            row["grouping_variable"],
            row["grouping_value"],
        )
        for row in descriptive_rows
    ]

    group_to_index = {
        spec: index
        for index, spec in enumerate(
            group_specs
        )
    }

    metric_fields = (
        "counterbalanced_accuracy",
        "strict_paired_correct",
        "order_inconsistent",
    )

    output_metric_names = (
        "mean_counterbalanced_accuracy",
        "mean_strict_paired_correct",
        "mean_order_inconsistent",
    )

    available_rows = nonmissing_rows(rows)

    prepared_by_stratum: dict[
        str,
        list[
            tuple[
                tuple[float, float, float],
                tuple[int, ...],
            ]
        ],
    ] = defaultdict(list)

    for row in available_rows:
        metric_values: list[float] = []

        for field in metric_fields:
            value = row[field]

            if value is None:
                raise ValueError(
                    f"{row['base_item_id']} has "
                    f"missing {field} despite having "
                    "an available counterbalanced score"
                )

            metric_values.append(
                float(value)
            )

        row_group_specs = (
            ("overall", "all"),
            (
                "global_congruence",
                str(row["global_congruence"]),
            ),
            (
                "cue_regime",
                str(row["cue_regime"]),
            ),
            (
                "number_pattern",
                str(row["number_pattern"]),
            ),
            (
                "nearest_attractor_congruence",
                str(
                    row[
                        "nearest_attractor_congruence"
                    ]
                ),
            ),
        )

        group_indices = tuple(
            group_to_index[spec]
            for spec in row_group_specs
        )

        prepared_by_stratum[
            str(row["number_pattern"])
        ].append(
            (
                (
                    metric_values[0],
                    metric_values[1],
                    metric_values[2],
                ),
                group_indices,
            )
        )

    strata = sorted(
        prepared_by_stratum
    )

    if len(strata) != 8:
        raise ValueError(
            "Expected 8 number-pattern bootstrap "
            f"strata, found {len(strata)}: "
            f"{strata}"
        )

    expected_nearest_labels = {
        "N3_matches_N2",
        "N3_mismatches_N2",
    }

    observed_nearest_labels = {
        value
        for field, value in group_specs
        if field
        == "nearest_attractor_congruence"
    }

    if (
        observed_nearest_labels
        != expected_nearest_labels
    ):
        raise ValueError(
            "Unexpected nearest-attractor labels: "
            f"{sorted(observed_nearest_labels)}"
        )

    n3_match_index = group_to_index[
        (
            "nearest_attractor_congruence",
            "N3_matches_N2",
        )
    ]

    n3_mismatch_index = group_to_index[
        (
            "nearest_attractor_congruence",
            "N3_mismatches_N2",
        )
    ]
    both_match_index = group_to_index[
        (
            "cue_regime",
            "both_match",
        )
    ]

    n1_mismatch_index = group_to_index[
        (
            "cue_regime",
            "N1_mismatch_only",
        )
    ]

    n3_mismatch_only_index = group_to_index[
        (
            "cue_regime",
            "N3_mismatch_only",
        )
    ]
    bootstrap_values: list[
        list[list[float]]
    ] = [
        [
            []
            for _ in metric_fields
        ]
        for _ in group_specs
    ]

    n3_contrast_values: list[float] = []

    n1_drop_values: list[float] = []
    n3_drop_values: list[float] = []
    n1_vs_n3_asymmetry_values: list[float] = []
    
    rng = random.Random(seed)

    for _ in range(replicates):
        sums = [
            [0.0, 0.0, 0.0]
            for _ in group_specs
        ]

        counts = [
            0
            for _ in group_specs
        ]

        for stratum in strata:
            stratum_rows = (
                prepared_by_stratum[
                    stratum
                ]
            )

            sampled = rng.choices(
                stratum_rows,
                k=len(stratum_rows),
            )

            for (
                metric_values,
                group_indices,
            ) in sampled:
                for group_index in group_indices:
                    counts[group_index] += 1

                    sums[group_index][0] += (
                        metric_values[0]
                    )

                    sums[group_index][1] += (
                        metric_values[1]
                    )

                    sums[group_index][2] += (
                        metric_values[2]
                    )

        replicate_means: list[
            list[float]
        ] = [
            [0.0, 0.0, 0.0]
            for _ in group_specs
        ]

        for group_index in range(
            len(group_specs)
        ):
            if counts[group_index] == 0:
                raise ValueError(
                    "Bootstrap replicate produced "
                    "an empty descriptive subgroup"
                )

            for metric_index in range(
                len(metric_fields)
            ):
                value = (
                    sums[group_index][metric_index]
                    / counts[group_index]
                )

                replicate_means[
                    group_index
                ][metric_index] = value

                bootstrap_values[
                    group_index
                ][metric_index].append(
                    value
                )

        n3_contrast_values.append(
            replicate_means[
                n3_match_index
            ][0]
            - replicate_means[
                n3_mismatch_index
            ][0]
        )
        both_match_mean = replicate_means[
            both_match_index
        ][0]

        n1_mismatch_mean = replicate_means[
            n1_mismatch_index
        ][0]

        n3_mismatch_only_mean = replicate_means[
            n3_mismatch_only_index
        ][0]

        n1_drop_values.append(
            both_match_mean
            - n1_mismatch_mean
        )

        n3_drop_values.append(
            both_match_mean
            - n3_mismatch_only_mean
        )

        n1_vs_n3_asymmetry_values.append(
            n3_mismatch_only_mean
            - n1_mismatch_mean
        )
    secondary_rows: list[
        dict[str, Any]
    ] = []

    for group_index, row in enumerate(
        descriptive_rows
    ):
        output_row = dict(row)

        for (
            metric_index,
            output_name,
        ) in enumerate(
            output_metric_names
        ):
            values = sorted(
                bootstrap_values[
                    group_index
                ][metric_index]
            )

            output_row[
                f"{output_name}_ci_lower"
            ] = percentile(
                values,
                0.025,
            )

            output_row[
                f"{output_name}_ci_upper"
            ] = percentile(
                values,
                0.975,
            )

        output_row[
            "bootstrap_replicates"
        ] = replicates

        output_row[
            "bootstrap_seed"
        ] = seed

        secondary_rows.append(
            output_row
        )

    n3_match_mean = condition_mean(
        rows,
        field=(
            "nearest_attractor_congruence"
        ),
        value="N3_matches_N2",
    )

    n3_mismatch_mean = condition_mean(
        rows,
        field=(
            "nearest_attractor_congruence"
        ),
        value="N3_mismatches_N2",
    )

    if (
        n3_match_mean is None
        or n3_mismatch_mean is None
    ):
        raise ValueError(
            "N3 congruence contrast is undefined"
        )

    sorted_n3_contrasts = sorted(
        n3_contrast_values
    )

    n3_contrast = {
        "provider": provider,
        "requested_model": model,
        "contrast": (
            "N3_matches_N2_minus_"
            "N3_mismatches_N2"
        ),
        "match_mean": n3_match_mean,
        "mismatch_mean": n3_mismatch_mean,
        "observed_difference": (
            n3_match_mean
            - n3_mismatch_mean
        ),
        "positive_values_mean": (
            "higher accuracy when N3 matches N2"
        ),
        "bootstrap": {
            "resampling_unit": "item",
            "stratified": True,
            "stratum_field": (
                "number_pattern"
            ),
            "strata": strata,
            "stratum_counts": {
                stratum: len(
                    prepared_by_stratum[
                        stratum
                    ]
                )
                for stratum in strata
            },
            "requested_replicates": (
                replicates
            ),
            "successful_replicates": (
                replicates
            ),
            "skipped_replicates": 0,
            "seed": seed,
            "ci_method": "percentile",
            "confidence_level": 0.95,
            "ci_lower": percentile(
                sorted_n3_contrasts,
                0.025,
            ),
            "ci_upper": percentile(
                sorted_n3_contrasts,
                0.975,
            ),
        },
    }
    both_match_observed = condition_mean(
        rows,
        field="cue_regime",
        value="both_match",
    )

    n1_mismatch_observed = condition_mean(
        rows,
        field="cue_regime",
        value="N1_mismatch_only",
    )

    n3_mismatch_only_observed = condition_mean(
        rows,
        field="cue_regime",
        value="N3_mismatch_only",
    )

    if (
        both_match_observed is None
        or n1_mismatch_observed is None
        or n3_mismatch_only_observed is None
    ):
        raise ValueError(
            "Cue-regime contrasts are undefined"
        )

    sorted_n1_drops = sorted(
        n1_drop_values
    )

    sorted_n3_drops = sorted(
        n3_drop_values
    )

    sorted_asymmetry_values = sorted(
        n1_vs_n3_asymmetry_values
    )

    n3_contrast["regime_contrasts"] = [
        {
            "contrast": (
                "both_match_minus_"
                "N1_mismatch_only"
            ),
            "both_match_mean": (
                both_match_observed
            ),
            "mismatch_mean": (
                n1_mismatch_observed
            ),
            "observed_difference": (
                both_match_observed
                - n1_mismatch_observed
            ),
            "positive_values_mean": (
                "greater interference from "
                "an N1 mismatch"
            ),
            "ci_lower": percentile(
                sorted_n1_drops,
                0.025,
            ),
            "ci_upper": percentile(
                sorted_n1_drops,
                0.975,
            ),
        },
        {
            "contrast": (
                "both_match_minus_"
                "N3_mismatch_only"
            ),
            "both_match_mean": (
                both_match_observed
            ),
            "mismatch_mean": (
                n3_mismatch_only_observed
            ),
            "observed_difference": (
                both_match_observed
                - n3_mismatch_only_observed
            ),
            "positive_values_mean": (
                "greater interference from "
                "an N3 mismatch"
            ),
            "ci_lower": percentile(
                sorted_n3_drops,
                0.025,
            ),
            "ci_upper": percentile(
                sorted_n3_drops,
                0.975,
            ),
        },
        {
            "contrast": (
                "N3_mismatch_only_minus_"
                "N1_mismatch_only"
            ),
            "N3_mismatch_only_mean": (
                n3_mismatch_only_observed
            ),
            "N1_mismatch_only_mean": (
                n1_mismatch_observed
            ),
            "observed_difference": (
                n3_mismatch_only_observed
                - n1_mismatch_observed
            ),
            "positive_values_mean": (
                "N1 mismatch causes greater "
                "interference than N3 mismatch"
            ),
            "ci_lower": percentile(
                sorted_asymmetry_values,
                0.025,
            ),
            "ci_upper": percentile(
                sorted_asymmetry_values,
                0.975,
            ),
        },
    ]
    return (
        secondary_rows,
        n3_contrast,
    )

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze one model's paired counterbalanced "
            "CE-Agree forced-choice results."
        )
    )

    parser.add_argument(
        "--item-results",
        required=True,
        type=Path,
        help=(
            "Item-level JSONL file produced by "
            "score_forced_choice_results.py."
        ),
    )

    parser.add_argument(
        "--output-prefix",
        required=True,
        type=Path,
        help=(
            "Prefix for the descriptive CSV, contrast JSON, "
            "and bootstrap JSON files."
        ),
    )

    parser.add_argument(
        "--bootstrap-replicates",
        type=int,
        default=DEFAULT_BOOTSTRAP_REPLICATES,
    )

    parser.add_argument(
        "--bootstrap-seed",
        type=int,
        default=DEFAULT_BOOTSTRAP_SEED,
    )

    args = parser.parse_args()

    if args.bootstrap_replicates <= 0:
        raise ValueError(
            "--bootstrap-replicates must be positive"
        )

    rows = load_jsonl(args.item_results)
    provider, model = validate_item_rows(rows)

    congruent_label, incongruent_label = (
        infer_congruence_labels(rows)
    )

    descriptive_rows = make_descriptive_rows(
        rows=rows,
        provider=provider,
        model=model,
    )
    (
        secondary_descriptive_rows,
        secondary_n3_contrast,
    ) = build_secondary_bootstrap_analysis(
        rows=rows,
        provider=provider,
        model=model,
        replicates=args.bootstrap_replicates,
        seed=args.bootstrap_seed,
    )
    congruent_mean = condition_mean(
        rows,
        field="global_congruence",
        value=congruent_label,
    )

    incongruent_mean = condition_mean(
        rows,
        field="global_congruence",
        value=incongruent_label,
    )

    observed_drop = global_drop(
        rows,
        congruent_label=congruent_label,
        incongruent_label=incongruent_label,
    )

    bootstrap = stratified_item_bootstrap(
        rows=rows,
        statistic=lambda sampled_rows: global_drop(
            sampled_rows,
            congruent_label=congruent_label,
            incongruent_label=incongruent_label,
        ),
        stratum_field="number_pattern",
        replicates=args.bootstrap_replicates,
        seed=args.bootstrap_seed,
    )

    contrast = {
        "provider": provider,
        "requested_model": model,
        "contrast": (
            "global_congruent_minus_global_incongruent"
        ),
        "congruent_label": congruent_label,
        "incongruent_label": incongruent_label,
        "congruent_mean": congruent_mean,
        "incongruent_mean": incongruent_mean,
        "observed_drop": observed_drop,
        "positive_values_mean": (
            "higher accuracy in the globally congruent condition"
        ),
        "technical_missing_items": sum(
            row["counterbalanced_accuracy"] is None
            for row in rows
        ),
        "bootstrap": bootstrap,
    }

    descriptive_path = Path(
        str(args.output_prefix) + ".descriptive.csv"
    )

    contrast_path = Path(
        str(args.output_prefix) + ".global_contrast.json"
    )

    secondary_descriptive_path = Path(
        str(args.output_prefix)
        + ".secondary_descriptive.csv"
    )

    secondary_contrasts_path = Path(
        str(args.output_prefix)
        + ".secondary_contrasts.json"
    )

    write_csv(
        descriptive_path,
        descriptive_rows,
    )

    write_json(
        contrast_path,
        contrast,
    )

    write_csv(
        secondary_descriptive_path,
        secondary_descriptive_rows,
    )

    write_json(
        secondary_contrasts_path,
        secondary_n3_contrast,
    )
    print(f"Provider: {provider}")
    print(f"Model: {model}")
    print(f"Item rows: {len(rows)}")
    print(
        "Technical missing items: "
        f"{contrast['technical_missing_items']}"
    )

    print("\nPrimary global contrast:")
    print(
        f"  Congruent mean:   {congruent_mean:.6f}"
        if congruent_mean is not None
        else "  Congruent mean:   undefined"
    )
    print(
        f"  Incongruent mean: {incongruent_mean:.6f}"
        if incongruent_mean is not None
        else "  Incongruent mean: undefined"
    )
    print(
        f"  Congruent minus incongruent: {observed_drop:.6f}"
        if observed_drop is not None
        else "  Congruent minus incongruent: undefined"
    )
    print(
        "  95% stratified item bootstrap CI: "
        f"[{bootstrap['ci_lower']:.6f}, "
        f"{bootstrap['ci_upper']:.6f}]"
    )

    print(f"\nSaved descriptive results to:")
    print(f"  {descriptive_path}")

    print("Saved global contrast to:")
    print(f"  {contrast_path}")


if __name__ == "__main__":
    main()
