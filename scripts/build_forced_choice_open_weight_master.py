#!/usr/bin/env python3
"""
Build the canonical prompt-level master file for the eleven preregistered
open-weight forced-choice BF16 runs.

Input:
    data/scores/model_individual_level_forced_choice/
        eleven *.forced_choice.bfloat16.csv files

Output:
    data/scores/master_score_lists/
        master_forced_choice_open_weight_bfloat16.v1.csv

The two auxiliary float32 precision-validation files are intentionally excluded.

This script does not calculate inferential statistics or item-level outcomes.
It concatenates the validated primary prompt-level files into one canonical
master table and performs structural checks before writing.

Expected output:
    11 models x 4,000 prompts = 44,000 rows
    11 models x 2,000 base items = 22,000 model-item combinations
"""

import argparse
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_INPUT_DIR = (
    REPO_ROOT
    / "data"
    / "scores"
    / "model_individual_level_forced_choice"
)

DEFAULT_OUTPUT = (
    REPO_ROOT
    / "data"
    / "scores"
    / "master_score_lists"
    / "master_forced_choice_open_weight_bfloat16.v1.csv"
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


def read_model_file(
    model_key: str,
    path: Path,
) -> pd.DataFrame:
    """Read one primary model file and perform basic structural checks."""
    if not path.exists():
        raise FileNotFoundError(f"Missing input file: {path}")

    df = pd.read_csv(path)

    if len(df) != 4000:
        raise ValueError(
            f"{model_key}: expected 4000 rows, found {len(df)}"
        )

    if df["prompt_id"].nunique() != 4000:
        raise ValueError(
            f"{model_key}: prompt_id is not unique within model"
        )

    if df["base_item_id"].nunique() != 2000:
        raise ValueError(
            f"{model_key}: expected 2000 base items, found "
            f"{df['base_item_id'].nunique()}"
        )

    pair_counts = df.groupby("base_item_id").size()
    if not (pair_counts == 2).all():
        raise ValueError(
            f"{model_key}: every base item must occur exactly twice"
        )

    # Use one stable model naming scheme in the master file.
    # Preserve the scorer's original model string for provenance.
    if "model" in df.columns:
        df = df.rename(columns={"model": "model_reported_by_scorer"})

    df.insert(0, "model", model_key)
    df.insert(1, "precision", "bfloat16")
    df.insert(2, "source_file", path.name)

    return df


def validate_master(master: pd.DataFrame) -> None:
    """Check expected dimensions and cross-model prompt identity."""
    expected_rows = 11 * 4000
    if len(master) != expected_rows:
        raise ValueError(
            f"Expected {expected_rows} master rows, found {len(master)}"
        )

    if master["model"].nunique() != 11:
        raise ValueError(
            f"Expected 11 models, found {master['model'].nunique()}"
        )

    model_item_count = (
        master[["model", "base_item_id"]]
        .drop_duplicates()
        .shape[0]
    )
    if model_item_count != 11 * 2000:
        raise ValueError(
            "Expected 22,000 unique model-item combinations, found "
            f"{model_item_count}"
        )

    reference_model = next(iter(MODEL_FILES))
    reference_ids = set(
        master.loc[
            master["model"] == reference_model,
            "prompt_id",
        ]
    )

    for model_key in MODEL_FILES:
        model_ids = set(
            master.loc[
                master["model"] == model_key,
                "prompt_id",
            ]
        )

        if model_ids != reference_ids:
            missing = reference_ids - model_ids
            extra = model_ids - reference_ids
            raise ValueError(
                f"{model_key}: prompt IDs differ from {reference_model}; "
                f"{len(missing)} missing, {len(extra)} extra"
            )

    duplicate_keys = master.duplicated(
        subset=["model", "prompt_id"]
    ).sum()

    if duplicate_keys:
        raise ValueError(
            f"Found {duplicate_keys} duplicate model-prompt rows"
        )


def main() -> None:
    """Build and write the canonical BF16 forced-choice master."""
    parser = argparse.ArgumentParser(
        description=(
            "Combine the eleven primary open-weight forced-choice BF16 "
            "files into one canonical prompt-level master CSV."
        )
    )

    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help="Directory containing individual forced-choice CSV files.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output master CSV.",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow an existing output file to be replaced.",
    )

    args = parser.parse_args()

    if args.output.exists() and not args.overwrite:
        raise FileExistsError(
            f"Output already exists: {args.output}\n"
            "Use --overwrite only if you intentionally want to replace it."
        )

    frames = []

    for model_key, filename in MODEL_FILES.items():
        path = args.input_dir / filename
        df = read_model_file(model_key, path)
        frames.append(df)

        print(
            f"Loaded {model_key:15s} "
            f"rows={len(df):4d} "
            f"items={df['base_item_id'].nunique():4d}"
        )

    master = pd.concat(
        frames,
        ignore_index=True,
        sort=False,
    )

    validate_master(master)

    master = master.sort_values(
        ["model", "base_item_id", "order_version"]
    ).reset_index(drop=True)

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    master.to_csv(
        args.output,
        index=False,
    )

    print()
    print("PASS: master validation succeeded.")
    print(f"Models:                  {master['model'].nunique()}")
    print(f"Prompt-level rows:       {len(master)}")
    print(
        "Model-item combinations:",
        master[["model", "base_item_id"]]
        .drop_duplicates()
        .shape[0],
    )
    print(f"Wrote: {args.output}")


if __name__ == "__main__":
    main()
