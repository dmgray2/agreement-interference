#!/usr/bin/env python3

"""
Consolidate CE-Agree forced-choice secondary analyses.

Outputs:
    results/final/master_forced_choice_secondary_descriptive_v1.csv
    results/final/master_forced_choice_secondary_contrasts_v1.csv

Expected:
    17 substantively interpretable models
    17 descriptive rows per model = 289 rows
    1 N3 contrast per model = 17 rows

GPT-3.5 is intentionally excluded because its outputs failed the
preregistered response-format requirement and are not interpretable
as grammatical-performance estimates.
"""

import csv
import json
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]

FINAL_DIR = (
    REPO_ROOT
    / "results"
    / "final"
)

DESCRIPTIVE_OUTPUT = (
    FINAL_DIR
    / "master_forced_choice_secondary_descriptive_v1.csv"
)

CONTRAST_OUTPUT = (
    FINAL_DIR
    / "master_forced_choice_secondary_contrasts_v1.csv"
)
REGIME_CONTRAST_OUTPUT = (
    FINAL_DIR
    / "master_forced_choice_regime_contrasts_v1.csv"
)

MODEL_FILES = {
    "gpt2": "gpt2_analysis_v1",
    "pythia-410m": "pythia-410m_analysis_v1",
    "llama-3.2-1b": "llama-3.2-1b_analysis_v1",
    "llama-3.2-3b": "llama-3.2-3b_analysis_v1",
    "llama-3.1-70b": "llama-3.1-70b_analysis_v1",
    "qwen2.5-1.5b": "qwen2.5-1.5b_analysis_v1",
    "qwen2.5-3b": "qwen2.5-3b_analysis_v1",
    "qwen2.5-7b": "qwen2.5-7b_analysis_v1",
    "qwen2.5-14b": "qwen2.5-14b_analysis_v1",
    "qwen2.5-32b": "qwen2.5-32b_analysis_v1",
    "qwen2.5-72b": "qwen2.5-72b_analysis_v1",
    "gpt-4.1": "openai_gpt-4.1_analysis_v1",
    "gpt-5": "openai_gpt-5_analysis_v1",
    "gpt-5.6-sol": "openai_gpt-5.6-sol_analysis_v1",
    "opus-4.1": "anthropic_opus-4.1_analysis_v1",
    "opus-4.5": "anthropic_opus-4.5_analysis_v1",
    "opus-5": "anthropic_opus-5_analysis_v1",
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


def load_json(
    path: Path,
) -> dict[str, Any]:
    with path.open(
        "r",
        encoding="utf-8",
    ) as source:
        return json.load(source)


def write_csv(
    path: Path,
    rows: list[dict[str, Any]],
) -> None:
    if not rows:
        raise ValueError(
            "Cannot write an empty CSV"
        )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

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


def flatten_contrast(
    model: str,
    value: dict[str, Any],
) -> dict[str, Any]:
    bootstrap = value["bootstrap"]

    return {
        "model": model,
        "provider": value["provider"],
        "requested_model": value[
            "requested_model"
        ],
        "contrast": value["contrast"],
        "match_mean": value["match_mean"],
        "mismatch_mean": value[
            "mismatch_mean"
        ],
        "observed_difference": value[
            "observed_difference"
        ],
        "positive_values_mean": value[
            "positive_values_mean"
        ],
        "bootstrap_replicates": bootstrap[
            "requested_replicates"
        ],
        "bootstrap_successful": bootstrap[
            "successful_replicates"
        ],
        "bootstrap_skipped": bootstrap[
            "skipped_replicates"
        ],
        "bootstrap_seed": bootstrap[
            "seed"
        ],
        "bootstrap_stratified": bootstrap[
            "stratified"
        ],
        "bootstrap_stratum_field": bootstrap[
            "stratum_field"
        ],
        "ci_method": bootstrap[
            "ci_method"
        ],
        "confidence_level": bootstrap[
            "confidence_level"
        ],
        "ci_lower": bootstrap[
            "ci_lower"
        ],
        "ci_upper": bootstrap[
            "ci_upper"
        ],
    }


def main() -> None:
    descriptive_master: list[
        dict[str, Any]
    ] = []

    regime_contrast_master: list[
        dict[str, Any]
    ] = []

    contrast_master: list[
        dict[str, Any]
    ] = []

    print(
        "Consolidating forced-choice "
        "secondary analyses."
    )
    print()

    for model, prefix in MODEL_FILES.items():
        descriptive_path = FINAL_DIR / (
            prefix
            + ".secondary_descriptive.csv"
        )

        contrast_path = FINAL_DIR / (
            prefix
            + ".secondary_contrasts.json"
        )

        if not descriptive_path.exists():
            raise FileNotFoundError(
                f"Missing descriptive file: "
                f"{descriptive_path}"
            )

        if not contrast_path.exists():
            raise FileNotFoundError(
                f"Missing contrast file: "
                f"{contrast_path}"
            )

        descriptive_rows = load_csv(
            descriptive_path
        )

        if len(descriptive_rows) != 17:
            raise ValueError(
                f"{model}: expected 17 "
                f"descriptive rows, found "
                f"{len(descriptive_rows)}"
            )

        for row in descriptive_rows:
            output_row: dict[
                str,
                Any
            ] = {
                "model": model,
            }

            output_row.update(row)

            descriptive_master.append(
                output_row
            )

        contrast = load_json(
            contrast_path
        )

        if (
            contrast.get("contrast")
            !=
            "N3_matches_N2_minus_"
            "N3_mismatches_N2"
        ):
            raise ValueError(
                f"{model}: unexpected "
                "secondary contrast"
            )

        contrast_master.append(
            flatten_contrast(
                model=model,
                value=contrast,
            )
        )
        regime_contrasts = contrast.get(
            "regime_contrasts",
            [],
        )

        if len(regime_contrasts) != 3:
            raise ValueError(
                f"{model}: expected 3 regime "
                f"contrasts, found "
                f"{len(regime_contrasts)}"
            )

        for regime_contrast in regime_contrasts:
            regime_contrast_master.append(
                {
                    "model": model,
                    "provider": contrast[
                        "provider"
                    ],
                    "requested_model": contrast[
                        "requested_model"
                    ],
                    **regime_contrast,
                }
            )
        print(
            f"Loaded {model:<16} "
            f"descriptive=17  contrasts=1"
        )

    expected_descriptive_rows = (
        len(MODEL_FILES) * 17
    )

    if (
        len(descriptive_master)
        != expected_descriptive_rows
    ):
        raise ValueError(
            "Unexpected descriptive master "
            f"row count: "
            f"{len(descriptive_master)}"
        )

    if (
        len(contrast_master)
        != len(MODEL_FILES)
    ):
        raise ValueError(
            "Unexpected contrast master "
            f"row count: "
            f"{len(contrast_master)}"
        )

    expected_regime_rows = (
        len(MODEL_FILES) * 3
    )

    if (
        len(regime_contrast_master)
        != expected_regime_rows
    ):
        raise ValueError(
            "Unexpected regime contrast "
            f"master row count: "
            f"{len(regime_contrast_master)}"
        )

    write_csv(
        DESCRIPTIVE_OUTPUT,
        descriptive_master,
    )

    write_csv(
        CONTRAST_OUTPUT,
        contrast_master,
    )

    write_csv(
        REGIME_CONTRAST_OUTPUT,
        regime_contrast_master,
    )
    print()
    print("Master checks:")
    print(
        f"Models: {len(MODEL_FILES)}"
    )
    print(
        "Secondary descriptive rows: "
        f"{len(descriptive_master)}"
    )
    print(
        "Secondary contrast rows: "
        f"{len(contrast_master)}"
    )

    print()
    print(
        "Wrote:"
    )
    print(
        f"  {DESCRIPTIVE_OUTPUT}"
    )
    print(
        f"  {CONTRAST_OUTPUT}"
    )


if __name__ == "__main__":
    main()
