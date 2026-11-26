#!/usr/bin/env python3

import argparse
import csv
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


DEFAULT_BOOTSTRAP_REPLICATES = 10_000
DEFAULT_BOOTSTRAP_SEED = 20260729


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
    value: Any,
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
        raise ValueError("Cannot write an empty CSV")

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


def parse_named_file(value: str) -> tuple[str, Path]:
    """
    Parse command-line values of the form:

        short_name=path/to/file.jsonl
    """
    if "=" not in value:
        raise argparse.ArgumentTypeError(
            "Expected NAME=PATH"
        )

    name, path_text = value.split("=", 1)

    name = name.strip()
    path_text = path_text.strip()

    if not name:
        raise argparse.ArgumentTypeError(
            "Model name cannot be empty"
        )

    if not path_text:
        raise argparse.ArgumentTypeError(
            "File path cannot be empty"
        )

    return name, Path(path_text)


def validate_model_rows(
    model_name: str,
    rows: list[dict[str, Any]],
) -> None:
    if len(rows) != 2000:
        raise ValueError(
            f"{model_name}: expected 2,000 rows, found {len(rows)}"
        )

    base_item_ids = [
        row.get("base_item_id")
        for row in rows
    ]

    if len(base_item_ids) != len(set(base_item_ids)):
        raise ValueError(
            f"{model_name}: duplicate base_item_id values"
        )

    for row_number, row in enumerate(rows, start=1):
        score = row.get("counterbalanced_accuracy")

        if score not in (0, 0.5, 1, None):
            raise ValueError(
                f"{model_name}: unexpected score {score!r} "
                f"in row {row_number}"
            )


def index_by_item(
    rows: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    return {
        row["base_item_id"]: row
        for row in rows
    }


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
        "Could not infer congruence labels. "
        f"Observed values: {sorted(values)}"
    )


def mean_score(
    rows: list[dict[str, Any]],
) -> float | None:
    scores = [
        float(row["counterbalanced_accuracy"])
        for row in rows
        if row["counterbalanced_accuracy"] is not None
    ]

    if not scores:
        return None

    return mean(scores)


def subset_rows(
    rows: list[dict[str, Any]],
    field: str,
    value: str,
) -> list[dict[str, Any]]:
    return [
        row
        for row in rows
        if str(row[field]) == value
    ]


def model_summary(
    model_name: str,
    rows: list[dict[str, Any]],
    congruent_label: str,
    incongruent_label: str,
) -> dict[str, Any]:
    congruent_rows = subset_rows(
        rows,
        field="global_congruence",
        value=congruent_label,
    )

    incongruent_rows = subset_rows(
        rows,
        field="global_congruence",
        value=incongruent_label,
    )

    overall_mean = mean_score(rows)
    congruent_mean = mean_score(congruent_rows)
    incongruent_mean = mean_score(incongruent_rows)

    if (
        congruent_mean is None
        or incongruent_mean is None
    ):
        drop = None
    else:
        drop = congruent_mean - incongruent_mean

    return {
        "model_name": model_name,
        "provider": rows[0].get("provider"),
        "requested_model": rows[0].get(
            "requested_model"
        ),
        "n_items": len(rows),
        "n_available_overall": sum(
            row["counterbalanced_accuracy"] is not None
            for row in rows
        ),
        "n_technical_missing": sum(
            row["counterbalanced_accuracy"] is None
            for row in rows
        ),
        "overall_accuracy": overall_mean,
        "fully_congruent_accuracy": congruent_mean,
        "globally_incongruent_accuracy": incongruent_mean,
        "congruent_minus_incongruent_drop": drop,
    }


def percentile(
    sorted_values: list[float],
    probability: float,
) -> float:
    if not sorted_values:
        raise ValueError(
            "Cannot calculate percentile of an empty list"
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


def equal_weight_model_mean(
    model_rows: dict[str, list[dict[str, Any]]],
    model_names: list[str],
    item_ids: list[str],
    condition_field: str,
    condition_value: str,
) -> float | None:
    per_model_means: list[float] = []

    for model_name in model_names:
        indexed = index_by_item(model_rows[model_name])

        scores = [
            float(indexed[item_id][
                "counterbalanced_accuracy"
            ])
            for item_id in item_ids
            if (
                item_id in indexed
                and str(
                    indexed[item_id][condition_field]
                ) == condition_value
                and indexed[item_id][
                    "counterbalanced_accuracy"
                ] is not None
            )
        ]

        if not scores:
            return None

        per_model_means.append(mean(scores))

    return mean(per_model_means)


def frontier_minus_comparator(
    model_rows: dict[str, list[dict[str, Any]]],
    frontier_name: str,
    comparator_names: list[str],
    item_ids: list[str],
    congruent_label: str,
) -> float | None:
    frontier_index = index_by_item(
        model_rows[frontier_name]
    )

    frontier_scores = [
        float(
            frontier_index[item_id][
                "counterbalanced_accuracy"
            ]
        )
        for item_id in item_ids
        if (
            item_id in frontier_index
            and str(
                frontier_index[item_id][
                    "global_congruence"
                ]
            ) == congruent_label
            and frontier_index[item_id][
                "counterbalanced_accuracy"
            ] is not None
        )
    ]

    if not frontier_scores:
        return None

    comparator_mean = equal_weight_model_mean(
        model_rows=model_rows,
        model_names=comparator_names,
        item_ids=item_ids,
        condition_field="global_congruence",
        condition_value=congruent_label,
    )

    if comparator_mean is None:
        return None

    return mean(frontier_scores) - comparator_mean


def stratified_item_bootstrap_frontier_comparison(
    model_rows: dict[str, list[dict[str, Any]]],
    frontier_name: str,
    comparator_names: list[str],
    congruent_label: str,
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    """
    Preregistered H2 bootstrap.

    Resample paired item IDs with replacement separately within
    each of the eight N1-N2-N3 number configurations. The same
    sampled item IDs are then used for the frontier model and
    every open-weight comparator model.
    """
    reference_rows = model_rows[frontier_name]

    grouped: dict[str, list[str]] = defaultdict(list)

    for row in reference_rows:
        if row["counterbalanced_accuracy"] is None:
            continue

        grouped[str(row["number_pattern"])].append(
            row["base_item_id"]
        )

    strata = sorted(grouped)

    if len(strata) != 8:
        raise ValueError(
            f"Expected 8 bootstrap strata, found "
            f"{len(strata)}: {strata}"
        )

    rng = random.Random(seed)
    values: list[float] = []
    skipped = 0

    for _ in range(replicates):
        sampled_item_ids: list[str] = []

        for stratum in strata:
            stratum_ids = grouped[stratum]

            sampled_item_ids.extend(
                rng.choices(
                    stratum_ids,
                    k=len(stratum_ids),
                )
            )

        value = frontier_minus_comparator(
            model_rows=model_rows,
            frontier_name=frontier_name,
            comparator_names=comparator_names,
            item_ids=sampled_item_ids,
            congruent_label=congruent_label,
        )

        if value is None:
            skipped += 1
            continue

        values.append(value)

    if not values:
        raise ValueError(
            "All bootstrap replicates were undefined"
        )

    values.sort()

    return {
        "resampling_unit": "item",
        "paired_across_models": True,
        "stratified": True,
        "stratum_field": "number_pattern",
        "strata": strata,
        "stratum_counts": {
            stratum: len(grouped[stratum])
            for stratum in strata
        },
        "requested_replicates": replicates,
        "successful_replicates": len(values),
        "skipped_replicates": skipped,
        "seed": seed,
        "ci_method": "percentile",
        "confidence_level": 0.95,
        "ci_lower": percentile(values, 0.025),
        "ci_upper": percentile(values, 0.975),
    }

def validate_common_items(
    model_rows: dict[str, list[dict[str, Any]]],
) -> list[str]:
    model_names = list(model_rows)

    reference_ids = {
        row["base_item_id"]
        for row in model_rows[model_names[0]]
    }

    for model_name in model_names[1:]:
        current_ids = {
            row["base_item_id"]
            for row in model_rows[model_name]
        }

        if current_ids != reference_ids:
            raise ValueError(
                f"{model_name} does not contain the same "
                "2,000 base items as the reference model"
            )

    return sorted(reference_ids)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare frontier and open-weight CE-Agree "
            "forced-choice results."
        )
    )

    parser.add_argument(
        "--model-file",
        action="append",
        required=True,
        type=parse_named_file,
        metavar="NAME=PATH",
        help=(
            "Item-level result file. Repeat once per model."
        ),
    )

    parser.add_argument(
        "--comparator-model",
        action="append",
        required=True,
        help=(
            "Short name of an open-weight model included in "
            "the equal-weight comparator. Repeat once per model."
        ),
    )

    parser.add_argument(
        "--frontier-model",
        action="append",
        required=True,
        help=(
            "Short name of a frontier endpoint to compare with "
            "the open-weight comparator. Repeat if needed."
        ),
    )

    parser.add_argument(
        "--output-prefix",
        required=True,
        type=Path,
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

    model_rows: dict[str, list[dict[str, Any]]] = {}

    for model_name, path in args.model_file:
        if model_name in model_rows:
            raise ValueError(
                f"Duplicate model short name: {model_name}"
            )

        rows = load_jsonl(path)
        validate_model_rows(model_name, rows)
        model_rows[model_name] = rows

    all_model_names = set(model_rows)

    missing_comparators = (
        set(args.comparator_model) - all_model_names
    )

    missing_frontiers = (
        set(args.frontier_model) - all_model_names
    )

    if missing_comparators:
        raise ValueError(
            "Comparator models missing from --model-file: "
            f"{sorted(missing_comparators)}"
        )

    if missing_frontiers:
        raise ValueError(
            "Frontier models missing from --model-file: "
            f"{sorted(missing_frontiers)}"
        )

    overlap = (
        set(args.comparator_model)
        & set(args.frontier_model)
    )

    if overlap:
        raise ValueError(
            "Models cannot be both comparator and frontier: "
            f"{sorted(overlap)}"
        )

    common_item_ids = validate_common_items(
        model_rows
    )

    first_rows = next(iter(model_rows.values()))

    congruent_label, incongruent_label = (
        infer_congruence_labels(first_rows)
    )

    summary_rows = [
        model_summary(
            model_name=model_name,
            rows=model_rows[model_name],
            congruent_label=congruent_label,
            incongruent_label=incongruent_label,
        )
        for model_name in model_rows
    ]

    comparator_congruent_mean = (
        equal_weight_model_mean(
            model_rows=model_rows,
            model_names=args.comparator_model,
            item_ids=common_item_ids,
            condition_field="global_congruence",
            condition_value=congruent_label,
        )
    )

    comparisons: list[dict[str, Any]] = []

    for index, frontier_name in enumerate(
        args.frontier_model
    ):
        frontier_summary = next(
            row
            for row in summary_rows
            if row["model_name"] == frontier_name
        )

        observed_difference = (
            frontier_minus_comparator(
                model_rows=model_rows,
                frontier_name=frontier_name,
                comparator_names=args.comparator_model,
                item_ids=common_item_ids,
                congruent_label=congruent_label,
            )
        )

        bootstrap = (
            stratified_item_bootstrap_frontier_comparison(
                model_rows=model_rows,
                frontier_name=frontier_name,
                comparator_names=args.comparator_model,
                congruent_label=congruent_label,
                replicates=args.bootstrap_replicates,
                seed=args.bootstrap_seed + index,
            )
        )

        comparisons.append(
            {
                "frontier_model": frontier_name,
                "frontier_fully_congruent_accuracy": (
                    frontier_summary[
                        "fully_congruent_accuracy"
                    ]
                ),
                "comparator_type": (
                    "equal_weight_mean_across_open_weight_models"
                ),
                "comparator_models": list(
                    args.comparator_model
                ),
                "comparator_fully_congruent_accuracy": (
                    comparator_congruent_mean
                ),
                "frontier_minus_comparator": (
                    observed_difference
                ),
                "positive_values_mean": (
                    "higher fully congruent V2 accuracy "
                    "for the frontier model"
                ),
                "bootstrap": bootstrap,
            }
        )

    summary_path = Path(
        str(args.output_prefix) + ".model_summary.csv"
    )

    comparison_path = Path(
        str(args.output_prefix)
        + ".frontier_comparisons.json"
    )

    write_csv(summary_path, summary_rows)
    write_json(comparison_path, comparisons)

    print(f"Models loaded: {len(model_rows)}")
    print(
        "Open-weight comparator models: "
        + ", ".join(args.comparator_model)
    )
    print(
        "Equal-weight comparator fully congruent accuracy: "
        f"{comparator_congruent_mean:.6f}"
    )

    print("\nFrontier comparisons:")

    for comparison in comparisons:
        bootstrap = comparison["bootstrap"]

        print(
            f"  {comparison['frontier_model']}: "
            f"difference="
            f"{comparison['frontier_minus_comparator']:.6f}, "
            f"95% CI=[{bootstrap['ci_lower']:.6f}, "
            f"{bootstrap['ci_upper']:.6f}]"
        )

    print(f"\nSaved model summaries to:")
    print(f"  {summary_path}")

    print("Saved frontier comparisons to:")
    print(f"  {comparison_path}")


if __name__ == "__main__":
    main()
