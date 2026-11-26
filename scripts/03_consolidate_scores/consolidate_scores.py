#!/usr/bin/env python3
"""
Make the master score files.

Inputs:
    data/stimuli/minpairs.ce2fam.generated.v0.7_nf2000.jsonl
    data/scores/model_individual_level/
    data/scores/model_family_level/

Outputs:
    data/scores/master_score_lists/master_scores_full.csv
    data/scores/master_score_lists/master_scores_family.csv

The item master has one row per model and benchmark item.

The family master has one row per model and sentence family.

This script also adds NP number-pattern columns from the stimulus JSONL:
It also adds a regime score column

    np1_num
    np2_num
    np3_num
    np_pattern
    np2_np1_incongruent
    regime_v1v2

This step only adds metadata. It does not change model scores.
"""

import argparse
import json
import re
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_STIMULI = (
    REPO_ROOT
    / "data"
    / "stimuli"
    / "minpairs.ce2fam.generated.v0.7_nf2000.jsonl"
)

DEFAULT_ITEM_DIR = (
    REPO_ROOT
    / "data"
    / "scores"
    / "model_individual_level"
)

DEFAULT_FAMILY_DIR = (
    REPO_ROOT
    / "data"
    / "scores"
    / "model_family_level"
)

DEFAULT_OUT_ITEMS = (
    REPO_ROOT
    / "data"
    / "scores"
    / "master_score_lists"
    / "master_scores_full.csv"
)

DEFAULT_OUT_FAMILY = (
    REPO_ROOT
    / "data"
    / "scores"
    / "master_score_lists"
    / "master_scores_family.csv"
)


def model_name_from_filename(path):
    """
    Parse the model name from a score filename.
    """
    name = path.name

    match = re.match(r"scores\.(.+?)\.ce2fam", name)

    if not match:
        raise ValueError(f"Could not parse model name from filename: {name}")

    return match.group(1)


def find_item_files(item_dir):
    """
    Find per-item score files.
    """
    paths = sorted(item_dir.glob("scores.*.csv"))

    item_paths = []

    for path in paths:
        name = path.name

        if ".family." in name:
            continue

        if "summary" in name:
            continue

        if name.endswith(".csv"):
            item_paths.append(path)

    return item_paths


def find_family_files(family_dir):
    """
    Find family-level score files.
    """
    paths = sorted(family_dir.glob("scores.*.family.csv"))

    if paths:
        return paths

    paths = sorted(family_dir.glob("scores.*.csv"))

    family_paths = []

    for path in paths:
        if ".family." in path.name:
            family_paths.append(path)

    return family_paths


def load_score_files(paths, label):
    """
    Read score files and add the model column.
    """
    frames = []

    for path in paths:
        model = model_name_from_filename(path)

        df = pd.read_csv(path)

        if "model" in df.columns:
            df = df.drop(columns=["model"])

        df.insert(0, "model", model)

        frames.append(df)

        print(f"Loaded {len(df):>6} rows from {path.name} ({model})")

    if not frames:
        raise FileNotFoundError(f"No {label} files found.")

    return pd.concat(frames, ignore_index=True)


def clean_nonbenchmark_rows(df):
    """
    Remove SANITY and TEST rows if they are present.
    """
    df = df.copy()

    start_rows = len(df)

    if "condition" in df.columns:
        df = df[~df["condition"].isin(["SANITY", "TEST"])].copy()

    if "family_id" in df.columns:
        family_text = df["family_id"].astype(str).str.lower()
        df = df[~family_text.str.startswith("sanity")].copy()
        df = df[~family_text.str.startswith("test")].copy()

    removed = start_rows - len(df)

    return df.reset_index(drop=True), removed


def number_label(value):
    """
    Convert a number value from the stimulus file into S or P.
    """
    if value in ["sg", "singular", "S"]:
        return "S"

    if value in ["pl", "plural", "P"]:
        return "P"

    raise ValueError(f"Unknown number value: {value}")


def load_np_patterns(stimulus_path):
    """
    Read NP number patterns from the stimulus JSONL file.
    """
    patterns = {}

    with stimulus_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()

            if not line:
                continue

            item = json.loads(line)

            family_id = item.get("family_id")

            if not family_id:
                family_id = item.get("meta", {}).get("family_id")

            if not family_id:
                continue

            if family_id in patterns:
                continue

            meta = item.get("meta", {})
            nums = meta.get("nums", {})

            if not nums:
                continue

            np1 = number_label(nums.get("NP1"))
            np2 = number_label(nums.get("NP2"))
            np3 = number_label(nums.get("NP3"))

            v1_attractor_incongruent = int(np3 != np1)
            v2_attractor_incongruent = int(np3 != np2)
            np2_np1_incongruent = int(np2 != np1)

            patterns[family_id] = {
                "family_id": family_id,
                "np1_num": np1,
                "np2_num": np2,
                "np3_num": np3,
                "np_pattern": np1 + np2 + np3,
                "np2_np1_incongruent": np2_np1_incongruent,
                "regime_v1v2": f"({v1_attractor_incongruent},{v2_attractor_incongruent})",
            }

    if not patterns:
        raise ValueError(f"No NP patterns found in {stimulus_path}")

    return pd.DataFrame(patterns.values())

def add_np_patterns(df, pattern_df):
    """
    Add NP number columns to a score dataframe.
    """
    df = df.copy()

    pattern_columns = [
        "np1_num",
        "np2_num",
        "np3_num",
        "np_pattern",
        "np2_np1_incongruent",
        "regime_v1v2",
    ]

    for column in pattern_columns:
        if column in df.columns:
            df = df.drop(columns=[column])

    merged = df.merge(pattern_df, on="family_id", how="left")

    missing = merged["np_pattern"].isna().sum()

    if missing:
        raise ValueError(
            f"{missing} rows did not match a family_id in the stimulus file."
        )

    return merged

def check_item_master(df):
    """
    Print checks for the item master file.
    """
    print()
    print("Item master checks:")
    print(f"Rows: {len(df)}")
    print(f"Models: {df['model'].nunique()}")

    if "checkpoint" in df.columns:
        print()
        print("Rows by checkpoint:")
        print(df["checkpoint"].value_counts().sort_index().to_string())

    if "np_pattern" in df.columns:
        print()
        print("Families by NP pattern, using first model:")
        first_model = sorted(df["model"].unique())[0]
        one_model = df[df["model"] == first_model]

        counts = (
            one_model
            .groupby("np_pattern")["family_id"]
            .nunique()
            .sort_index()
        )

        print(counts.to_string())


def check_family_master(df):
    """
    Print checks for the family master file.
    """
    print()
    print("Family master checks:")
    print(f"Rows: {len(df)}")
    print(f"Models: {df['model'].nunique()}")

    if "family_id" in df.columns:
        rows_per_model = df.groupby("model")["family_id"].nunique()
        print()
        print("Families per model:")
        print(rows_per_model.to_string())


def write_csv(df, path):
    """
    Write a CSV file.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"Wrote {path} ({len(df)} rows)")


def main():
    parser = argparse.ArgumentParser(
        description="Make master score files from per-model score files."
    )

    parser.add_argument(
        "--stimuli",
        type=Path,
        default=DEFAULT_STIMULI,
        help="Stimulus JSONL file.",
    )

    parser.add_argument(
        "--item-dir",
        "--item_dir",
        dest="item_dir",
        type=Path,
        default=DEFAULT_ITEM_DIR,
        help="Folder with per-item score CSVs.",
    )

    parser.add_argument(
        "--family-dir",
        "--family_dir",
        dest="family_dir",
        type=Path,
        default=DEFAULT_FAMILY_DIR,
        help="Folder with family-level score CSVs.",
    )

    parser.add_argument(
        "--out-items",
        "--out_items",
        dest="out_items",
        type=Path,
        default=DEFAULT_OUT_ITEMS,
        help="Output item master CSV.",
    )

    parser.add_argument(
        "--out-family",
        "--out_family",
        dest="out_family",
        type=Path,
        default=DEFAULT_OUT_FAMILY,
        help="Output family master CSV.",
    )

    args = parser.parse_args()

    if not args.stimuli.exists():
        raise FileNotFoundError(f"Could not find stimulus file: {args.stimuli}")

    if not args.item_dir.exists():
        raise FileNotFoundError(f"Could not find item score folder: {args.item_dir}")

    if not args.family_dir.exists():
        raise FileNotFoundError(
            f"Could not find family score folder: {args.family_dir}"
        )

    print(f"Reading NP patterns from {args.stimuli}")
    pattern_df = load_np_patterns(args.stimuli)
    print(f"Found NP patterns for {len(pattern_df)} families.")

    item_paths = find_item_files(args.item_dir)
    family_paths = find_family_files(args.family_dir)

    print()
    print(f"Found {len(item_paths)} item score files.")
    print(f"Found {len(family_paths)} family score files.")

    print()
    print("Reading item score files:")
    item_master = load_score_files(item_paths, "item score")

    print()
    print("Reading family score files:")
    family_master = load_score_files(family_paths, "family score")

    item_master, removed_items = clean_nonbenchmark_rows(item_master)
    family_master, removed_families = clean_nonbenchmark_rows(family_master)

    print()
    print(f"Removed {removed_items} non-benchmark item rows.")
    print(f"Removed {removed_families} non-benchmark family rows.")

    item_master = add_np_patterns(item_master, pattern_df)
    family_master = add_np_patterns(family_master, pattern_df)

    write_csv(item_master, args.out_items)
    write_csv(family_master, args.out_family)

    check_item_master(item_master)
    check_family_master(family_master)


if __name__ == "__main__":
    main()
