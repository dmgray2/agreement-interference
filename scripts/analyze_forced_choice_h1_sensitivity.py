#!/usr/bin/env python3

"""
Preregistered H1 calibration analysis.

Compare the global congruent-minus-incongruent V2 contrast
between:

1. critical-region accuracy, and
2. counterbalanced forced-choice accuracy

for the fixed set of eleven open-weight models.

The bootstrap resamples complete item IDs with replacement
separately within each of the eight N1-N2-N3 number patterns.
The same sampled item IDs are used across both methods and all
models in each replicate.

Outputs:
    results/final/preregistered_h1_v1.model_contrasts.csv
    results/final/preregistered_h1_v1.calibration.json
"""

import csv
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]

CRITICAL_PATH = (
    REPO_ROOT
    / "data"
    / "scores"
    / "master_score_lists"
    / "master_scores_full.csv"
)

FORCED_PATH = (
    REPO_ROOT
    / "results"
    / "final"
    / "master_forced_choice_scores_v1.csv"
)

OUTPUT_PREFIX = (
    REPO_ROOT
    / "results"
    / "final"
    / "h1_sensitivity_excluding_gpt2_pythia_v1"
)

BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 20260731


MODEL_MAP = {
    "llama-3.2-1b": "llama32_1b",
    "llama-3.2-3b": "llama32_3b",
    "llama-3.1-70b": "llama31_70b",
    "qwen2.5-1.5b": "qwen25_1.5b",
    "qwen2.5-3b": "qwen25_3b",
    "qwen2.5-7b": "qwen25_7b",
    "qwen2.5-14b": "qwen25_14b",
    "qwen2.5-32b": "qwen25_32b",
    "qwen2.5-72b": "qwen25_72b",
}


def load_csv(
    path: Path,
) -> list[dict[str, str]]:
    with path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as source:
        return list(csv.DictReader(source))


def write_csv(
    path: Path,
    rows: list[dict[str, Any]],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not rows:
        raise ValueError(
            "Cannot write empty CSV"
        )

    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as output:
        writer = csv.DictWriter(
            output,
            fieldnames=list(rows[0]),
        )

        writer.writeheader()
        writer.writerows(rows)


def write_json(
    path: Path,
    value: Any,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as output:
        json.dump(
            value,
            output,
            ensure_ascii=False,
            indent=2,
        )
        output.write("\n")


def is_global_congruent(
    row: dict[str, str],
) -> bool:
    return (
        row["np1_num"] == row["np2_num"]
        and row["np3_num"] == row["np2_num"]
    )


def mean_or_none(
    values: list[float],
) -> float | None:
    if not values:
        return None

    return mean(values)


def critical_drop(
    indexed_rows: dict[str, dict[str, str]],
    sampled_item_ids: list[str],
) -> float:
    congruent: list[float] = []
    incongruent: list[float] = []

    for item_id in sampled_item_ids:
        row = indexed_rows[item_id]
        value = float(row["correct_crit"])

        if is_global_congruent(row):
            congruent.append(value)
        else:
            incongruent.append(value)

    congruent_mean = mean_or_none(congruent)
    incongruent_mean = mean_or_none(incongruent)

    if (
        congruent_mean is None
        or incongruent_mean is None
    ):
        raise ValueError(
            "Critical-region contrast undefined"
        )

    return congruent_mean - incongruent_mean


def forced_drop(
    indexed_rows: dict[str, dict[str, str]],
    sampled_item_ids: list[str],
) -> float:
    congruent: list[float] = []
    incongruent: list[float] = []

    for item_id in sampled_item_ids:
        row = indexed_rows[item_id]

        score_text = row[
            "counterbalanced_accuracy"
        ]

        if score_text == "":
            raise ValueError(
                "Unexpected missing forced-choice score "
                f"for {item_id}"
            )

        value = float(score_text)

        if row["global_congruence"] == "congruent":
            congruent.append(value)
        elif (
            row["global_congruence"]
            == "incongruent"
        ):
            incongruent.append(value)
        else:
            raise ValueError(
                "Unexpected global_congruence value: "
                f"{row['global_congruence']!r}"
            )

    congruent_mean = mean_or_none(congruent)
    incongruent_mean = mean_or_none(incongruent)

    if (
        congruent_mean is None
        or incongruent_mean is None
    ):
        raise ValueError(
            "Forced-choice contrast undefined"
        )

    return congruent_mean - incongruent_mean


def rankdata(
    values: list[float],
) -> list[float]:
    """
    Average ranks for ties.
    """
    indexed = sorted(
        enumerate(values),
        key=lambda pair: pair[1],
    )

    ranks = [0.0] * len(values)

    position = 0

    while position < len(indexed):
        end = position + 1

        while (
            end < len(indexed)
            and indexed[end][1]
            == indexed[position][1]
        ):
            end += 1

        average_rank = (
            (position + 1) + end
        ) / 2

        for index in range(
            position,
            end,
        ):
            original_index = indexed[index][0]
            ranks[original_index] = average_rank

        position = end

    return ranks


def pearson(
    x: list[float],
    y: list[float],
) -> float:
    if len(x) != len(y):
        raise ValueError(
            "Pearson vectors differ in length"
        )

    if len(x) < 2:
        raise ValueError(
            "Need at least two values"
        )

    x_mean = mean(x)
    y_mean = mean(y)

    numerator = sum(
        (x_value - x_mean)
        * (y_value - y_mean)
        for x_value, y_value
        in zip(x, y)
    )

    x_ss = sum(
        (value - x_mean) ** 2
        for value in x
    )

    y_ss = sum(
        (value - y_mean) ** 2
        for value in y
    )

    denominator = math.sqrt(
        x_ss * y_ss
    )

    if denominator == 0:
        raise ValueError(
            "Correlation undefined because "
            "one vector has zero variance"
        )

    return numerator / denominator


def spearman(
    x: list[float],
    y: list[float],
) -> float:
    return pearson(
        rankdata(x),
        rankdata(y),
    )


def sign_label(
    value: float,
) -> str:
    if value > 0:
        return "positive"

    if value < 0:
        return "negative"

    return "zero"


def percentile(
    sorted_values: list[float],
    probability: float,
) -> float:
    if not sorted_values:
        raise ValueError(
            "Cannot calculate percentile "
            "of empty list"
        )

    position = probability * (
        len(sorted_values) - 1
    )

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


def main() -> None:
    critical_all = load_csv(
        CRITICAL_PATH
    )

    forced_all = load_csv(
        FORCED_PATH
    )

    critical_rows = [
        row
        for row in critical_all
        if (
            row["checkpoint"] == "V2"
            and row["model"]
            in set(MODEL_MAP.values())
        )
    ]

    forced_rows = [
        row
        for row in forced_all
        if row["model"] in MODEL_MAP
    ]

    critical_by_model: dict[
        str,
        dict[str, dict[str, str]]
    ] = {}

    forced_by_model: dict[
        str,
        dict[str, dict[str, str]]
    ] = {}

    for forced_model, critical_model in (
        MODEL_MAP.items()
    ):
        critical_model_rows = [
            row
            for row in critical_rows
            if row["model"] == critical_model
        ]

        forced_model_rows = [
            row
            for row in forced_rows
            if row["model"] == forced_model
        ]

        if len(critical_model_rows) != 2000:
            raise ValueError(
                f"{critical_model}: expected "
                f"2,000 critical V2 rows, found "
                f"{len(critical_model_rows)}"
            )

        if len(forced_model_rows) != 2000:
            raise ValueError(
                f"{forced_model}: expected "
                f"2,000 forced-choice rows, found "
                f"{len(forced_model_rows)}"
            )

        critical_index = {
            row["pair_id"]: row
            for row in critical_model_rows
        }

        forced_index = {
            row["base_item_id"]: row
            for row in forced_model_rows
        }

        if len(critical_index) != 2000:
            raise ValueError(
                f"{critical_model}: duplicate "
                "critical pair_id values"
            )

        if len(forced_index) != 2000:
            raise ValueError(
                f"{forced_model}: duplicate "
                "forced base_item_id values"
            )

        if (
            set(critical_index)
            != set(forced_index)
        ):
            raise ValueError(
                f"{forced_model}: critical and "
                "forced item IDs differ"
            )

        critical_by_model[
            forced_model
        ] = critical_index

        forced_by_model[
            forced_model
        ] = forced_index

    model_names = list(MODEL_MAP)

    reference_model = model_names[0]

    item_ids = sorted(
        critical_by_model[
            reference_model
        ]
    )

    # Validate item metadata across the two methods.
    for item_id in item_ids:
        critical_row = critical_by_model[
            reference_model
        ][item_id]

        forced_row = forced_by_model[
            reference_model
        ][item_id]

        expected_global = (
            "congruent"
            if is_global_congruent(
                critical_row
            )
            else "incongruent"
        )

        if (
            forced_row["global_congruence"]
            != expected_global
        ):
            raise ValueError(
                f"{item_id}: global congruence "
                "does not match across methods"
            )

        if (
            forced_row["number_pattern"]
            != critical_row["np_pattern"]
        ):
            raise ValueError(
                f"{item_id}: number pattern "
                "does not match across methods"
            )

    observed_rows: list[
        dict[str, Any]
    ] = []

    critical_drops: list[float] = []
    forced_drops: list[float] = []

    for forced_model in model_names:
        critical_value = critical_drop(
            critical_by_model[
                forced_model
            ],
            item_ids,
        )

        forced_value = forced_drop(
            forced_by_model[
                forced_model
            ],
            item_ids,
        )

        critical_sign = sign_label(
            critical_value
        )

        forced_sign = sign_label(
            forced_value
        )

        critical_drops.append(
            critical_value
        )

        forced_drops.append(
            forced_value
        )

        observed_rows.append(
            {
                "model": forced_model,
                "critical_model_name": (
                    MODEL_MAP[forced_model]
                ),
                "critical_region_drop": (
                    critical_value
                ),
                "forced_choice_drop": (
                    forced_value
                ),
                "critical_region_sign": (
                    critical_sign
                ),
                "forced_choice_sign": (
                    forced_sign
                ),
                "sign_agreement": int(
                    critical_sign
                    == forced_sign
                ),
            }
        )

    observed_rho = spearman(
        critical_drops,
        forced_drops,
    )

    sign_agreement_count = sum(
        row["sign_agreement"]
        for row in observed_rows
    )

    # Build eight item-level bootstrap strata.
    strata: dict[
        str,
        list[str]
    ] = defaultdict(list)

    for item_id in item_ids:
        pattern = critical_by_model[
            reference_model
        ][item_id]["np_pattern"]

        strata[pattern].append(
            item_id
        )

    if len(strata) != 8:
        raise ValueError(
            f"Expected 8 number-pattern strata, "
            f"found {len(strata)}: "
            f"{sorted(strata)}"
        )

    rng = random.Random(
        BOOTSTRAP_SEED
    )

    bootstrap_rhos: list[float] = []
    skipped = 0

    for _ in range(
        BOOTSTRAP_REPLICATES
    ):
        sampled_item_ids: list[str] = []

        for pattern in sorted(strata):
            pattern_ids = strata[
                pattern
            ]

            sampled_item_ids.extend(
                rng.choices(
                    pattern_ids,
                    k=len(pattern_ids),
                )
            )

        sampled_critical_drops: list[
            float
        ] = []

        sampled_forced_drops: list[
            float
        ] = []

        for forced_model in model_names:
            sampled_critical_drops.append(
                critical_drop(
                    critical_by_model[
                        forced_model
                    ],
                    sampled_item_ids,
                )
            )

            sampled_forced_drops.append(
                forced_drop(
                    forced_by_model[
                        forced_model
                    ],
                    sampled_item_ids,
                )
            )

        try:
            rho = spearman(
                sampled_critical_drops,
                sampled_forced_drops,
            )
        except ValueError:
            skipped += 1
            continue

        bootstrap_rhos.append(
            rho
        )

    if not bootstrap_rhos:
        raise ValueError(
            "All bootstrap correlations "
            "were undefined"
        )

    bootstrap_rhos.sort()

    result = {
        "analysis": (
            "h1_sensitivity_excluding_gpt2_pythia"
            "method_calibration"
        ),
        "number_of_models": len(
            model_names
        ),
        "models": model_names,
        "contrast": (
            "global_congruent_minus_"
            "global_incongruent"
        ),
        "critical_region_metric": (
            "correct_crit"
        ),
        "forced_choice_metric": (
            "counterbalanced_accuracy"
        ),
        "observed_spearman_rho": (
            observed_rho
        ),
        "sign_agreement_count": (
            sign_agreement_count
        ),
        "sign_agreement_denominator": (
            len(model_names)
        ),
        "bootstrap": {
            "resampling_unit": "item",
            "paired_across_methods": True,
            "paired_across_models": True,
            "stratified": True,
            "stratum_field": (
                "number_pattern"
            ),
            "strata": sorted(strata),
            "stratum_counts": {
                pattern: len(
                    strata[pattern]
                )
                for pattern
                in sorted(strata)
            },
            "requested_replicates": (
                BOOTSTRAP_REPLICATES
            ),
            "successful_replicates": (
                len(bootstrap_rhos)
            ),
            "skipped_replicates": (
                skipped
            ),
            "seed": BOOTSTRAP_SEED,
            "ci_method": "percentile",
            "confidence_level": 0.95,
            "ci_lower": percentile(
                bootstrap_rhos,
                0.025,
            ),
            "ci_upper": percentile(
                bootstrap_rhos,
                0.975,
            ),
        },
    }

    model_path = Path(
        str(OUTPUT_PREFIX)
        + ".model_contrasts.csv"
    )

    calibration_path = Path(
        str(OUTPUT_PREFIX)
        + ".calibration.json"
    )

    write_csv(
        model_path,
        observed_rows,
    )

    write_json(
        calibration_path,
        result,
    )

    print(
        f"Models: {len(model_names)}"
    )
    print(
        f"Shared V2 items: {len(item_ids)}"
    )
    print()
    print("Model-level contrasts:")

    for row in observed_rows:
        print(
            f"  {row['model']:<16} "
            f"critical="
            f"{row['critical_region_drop']:.6f}  "
            f"forced="
            f"{row['forced_choice_drop']:.6f}  "
            f"sign agreement="
            f"{row['sign_agreement']}"
        )

    print()
    print(
        "Sign agreement: "
        f"{sign_agreement_count}/"
        f"{len(model_names)}"
    )

    print(
        "Spearman rho: "
        f"{observed_rho:.6f}"
    )

    print(
        "95% stratified item bootstrap CI: "
        f"[{result['bootstrap']['ci_lower']:.6f}, "
        f"{result['bootstrap']['ci_upper']:.6f}]"
    )

    print(
        "Successful bootstrap replicates: "
        f"{len(bootstrap_rhos)}/"
        f"{BOOTSTRAP_REPLICATES}"
    )

    print()
    print(
        f"Saved model contrasts to:"
    )
    print(
        f"  {model_path}"
    )

    print(
        "Saved calibration result to:"
    )
    print(
        f"  {calibration_path}"
    )


if __name__ == "__main__":
    main()
