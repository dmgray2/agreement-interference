#!/usr/bin/env python3
"""
Validate the eleven primary BF16 open-weight forced-choice result files.

Checks performed
----------------
For each model:
1. Exactly 4,000 prompt-level rows.
2. Exactly 4,000 unique prompt IDs.
3. Exactly 2,000 unique base-item IDs.
4. Every base item occurs exactly twice.
5. Each base item contains order versions 1 and 2.
6. Each base item has the grammatical answer assigned once to A and once to B.
7. Answers are only A or B.
8. The stored correctness variable agrees with answer == grammatical_label.
9. The stored A-minus-B log-probability difference agrees with
   logprob_a - logprob_b.
10. The reported answer agrees with the preregistered decision rule:
       A if logprob_a >= logprob_b, otherwise B.
    Thus exact ties must be assigned to A.
11. Required experimental metadata are complete.

Across models:
12. Every model contains exactly the same prompt IDs.
13. Stimulus and condition metadata agree exactly for every prompt.

The two FP32 precision-validation files are intentionally excluded from this
primary-data validation.

Run from the repository root:
    python scripts/validate_forced_choice_open_weight.py
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd


SCORES_DIR = Path(
    "data/scores/model_individual_level_forced_choice"
)

MODEL_FILES = {
    "gpt2": "gpt2.forced_choice.bfloat16.csv",
    "pythia-410m": "pythia-410m.forced_choice.bfloat16.csv",
    "llama-3.2-1b": "llama-3.2-1b.forced_choice.bfloat16.csv",
    "llama-3.2-3b": "llama-3.2-3b.forced_choice.bfloat16.csv",
    "llama-3.1-70b": "llama-3.1-70b.forced_choice.bfloat16.csv",
    "qwen2.5-1.5b": "qwen2.5-1.5b.forced_choice.bfloat16.csv",
    "qwen2.5-3b": "qwen2.5-3b.forced_choice.bfloat16.csv",
    "qwen2.5-7b": "qwen2.5-7b.forced_choice.bfloat16.csv",
    "qwen2.5-14b": "qwen2.5-14b.forced_choice.bfloat16.csv",
    "qwen2.5-32b": "qwen2.5-32b.forced_choice.bfloat16.csv",
    "qwen2.5-72b": "qwen2.5-72b.forced_choice.bfloat16.csv",
}

REQUIRED_COLUMNS = [
    "model",
    "prompt_id",
    "base_item_id",
    "source_family_id",
    "condition",
    "checkpoint",
    "order_version",
    "grammatical_label",
    "answer",
    "correct",
    "logprob_a",
    "logprob_b",
    "logprob_a_minus_b",
    "prefix",
    "option_a",
    "option_b",
    "prompt",
    "number_pattern",
    "cue_regime",
    "global_congruence",
    "nearest_attractor_congruence",
    "np1_number",
    "np2_number",
    "np3_number",
    "np1_lexeme",
    "np2_lexeme",
    "np3_lexeme",
    "v2_lexeme",
    "v3_lexeme",
    "source_good",
    "source_bad",
    "response_interface",
]

# These should be identical for a given prompt across all eleven models.
CROSS_MODEL_COLUMNS = [
    "prompt_id",
    "base_item_id",
    "source_family_id",
    "condition",
    "checkpoint",
    "order_version",
    "grammatical_label",
    "prefix",
    "option_a",
    "option_b",
    "prompt",
    "number_pattern",
    "cue_regime",
    "global_congruence",
    "nearest_attractor_congruence",
    "np1_number",
    "np2_number",
    "np3_number",
    "np1_lexeme",
    "np2_lexeme",
    "np3_lexeme",
    "v2_lexeme",
    "v3_lexeme",
    "source_good",
    "source_bad",
]


def fail(model: str, message: str) -> None:
    """Print a validation failure and terminate with a nonzero exit status."""
    print(f"\nFAIL [{model}]: {message}")
    sys.exit(1)


def validate_one(model: str, path: Path) -> pd.DataFrame:
    """Validate one prompt-level forced-choice result file."""
    if not path.exists():
        fail(model, f"File not found: {path}")

    df = pd.read_csv(path)

    missing_columns = [
        column for column in REQUIRED_COLUMNS if column not in df.columns
    ]
    if missing_columns:
        fail(model, f"Missing columns: {missing_columns}")

    if len(df) != 4000:
        fail(model, f"Expected 4000 rows, found {len(df)}")

    if df["prompt_id"].nunique() != 4000:
        fail(
            model,
            f"Expected 4000 unique prompt IDs, found "
            f"{df['prompt_id'].nunique()}",
        )

    if df["base_item_id"].nunique() != 2000:
        fail(
            model,
            f"Expected 2000 unique base-item IDs, found "
            f"{df['base_item_id'].nunique()}",
        )

    pair_counts = df.groupby("base_item_id").size()
    if not (pair_counts == 2).all():
        bad = pair_counts[pair_counts != 2]
        fail(
            model,
            f"{len(bad)} base items do not occur exactly twice",
        )

    order_sets = (
        df.groupby("base_item_id")["order_version"]
        .apply(lambda x: set(x.tolist()))
    )
    bad_orders = order_sets[order_sets != {1, 2}]
    if len(bad_orders):
        fail(
            model,
            f"{len(bad_orders)} base items do not contain orders {{1, 2}}",
        )

    label_sets = (
        df.groupby("base_item_id")["grammatical_label"]
        .apply(lambda x: set(x.tolist()))
    )
    bad_labels = label_sets[label_sets != {"A", "B"}]
    if len(bad_labels):
        fail(
            model,
            f"{len(bad_labels)} base items are not counterbalanced A/B",
        )

    answer_values = set(df["answer"].dropna().unique())
    if not answer_values.issubset({"A", "B"}):
        fail(model, f"Invalid answer values: {sorted(answer_values)}")

    if df["answer"].isna().any():
        fail(model, "Missing answer values found")

    expected_correct = (
        df["answer"].astype(str) == df["grammatical_label"].astype(str)
    ).astype(int)

    stored_correct = pd.to_numeric(df["correct"], errors="coerce")
    if stored_correct.isna().any():
        fail(model, "Non-numeric or missing correctness values found")

    correctness_mismatches = (stored_correct != expected_correct).sum()
    if correctness_mismatches:
        fail(
            model,
            f"{correctness_mismatches} rows have inconsistent correctness",
        )

    for column in ["logprob_a", "logprob_b", "logprob_a_minus_b"]:
        values = pd.to_numeric(df[column], errors="coerce")
        if values.isna().any():
            fail(model, f"Missing or non-numeric values in {column}")
        if not np.isfinite(values).all():
            fail(model, f"Non-finite values in {column}")

    calculated_diff = df["logprob_a"] - df["logprob_b"]
    diff_matches = np.isclose(
        calculated_diff,
        df["logprob_a_minus_b"],
        rtol=1e-7,
        atol=1e-7,
    )
    if not diff_matches.all():
        fail(
            model,
            f"{(~diff_matches).sum()} stored log-probability differences "
            f"do not equal logprob_a - logprob_b",
        )

    expected_answer = np.where(
        df["logprob_a_minus_b"] >= 0,
        "A",
        "B",
    )
    answer_mismatches = (df["answer"].to_numpy() != expected_answer).sum()
    if answer_mismatches:
        fail(
            model,
            f"{answer_mismatches} answers violate the A/B decision rule",
        )

    essential_metadata = [
        column
        for column in CROSS_MODEL_COLUMNS
        if column != "prompt_id"
    ]
    null_counts = df[essential_metadata].isna().sum()
    bad_nulls = null_counts[null_counts > 0]
    if len(bad_nulls):
        fail(
            model,
            "Missing experimental metadata: "
            + ", ".join(
                f"{column}={count}"
                for column, count in bad_nulls.items()
            ),
        )

    ties = int((df["logprob_a_minus_b"] == 0).sum())
    a_count = int((df["answer"] == "A").sum())
    b_count = int((df["answer"] == "B").sum())

    print(
        f"PASS  {model:15s} "
        f"rows=4000  items=2000  "
        f"A={a_count:4d}  B={b_count:4d}  ties={ties:4d}"
    )

    return df.sort_values("prompt_id").reset_index(drop=True)


def validate_across_models(data: dict[str, pd.DataFrame]) -> None:
    """Verify that all eleven files contain identical stimuli and metadata."""
    reference_model = next(iter(data))
    reference = data[reference_model]

    for model, df in data.items():
        if model == reference_model:
            continue

        if not df["prompt_id"].equals(reference["prompt_id"]):
            missing = set(reference["prompt_id"]) - set(df["prompt_id"])
            extra = set(df["prompt_id"]) - set(reference["prompt_id"])
            fail(
                model,
                f"Prompt IDs differ from {reference_model}: "
                f"{len(missing)} missing, {len(extra)} extra",
            )

        for column in CROSS_MODEL_COLUMNS:
            left = reference[column].astype(str)
            right = df[column].astype(str)

            mismatches = left != right
            if mismatches.any():
                fail(
                    model,
                    f"{mismatches.sum()} values in '{column}' differ "
                    f"from {reference_model}",
                )


def main() -> None:
    """Run all within-model and across-model validation checks."""
    print("Validating primary BF16 forced-choice files...\n")

    validated = {}

    for model, filename in MODEL_FILES.items():
        path = SCORES_DIR / filename
        validated[model] = validate_one(model, path)

    print("\nChecking cross-model stimulus identity...")
    validate_across_models(validated)

    print("\nPASS: all 11 primary BF16 files validated successfully.")
    print("The two FP32 precision-check files were not included.")


if __name__ == "__main__":
    main()
