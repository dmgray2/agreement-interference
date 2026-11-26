#!/usr/bin/env python3

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def number_pattern(nums: dict[str, str]) -> str:
    """Return a pattern such as SPP from NP1, NP2, and NP3 number values."""
    symbol = {"sg": "S", "pl": "P"}
    return "".join(symbol[nums[key]] for key in ("NP1", "NP2", "NP3"))


def cue_regime(nums: dict[str, str]) -> tuple[str, str, str]:
    """
    Return:
      four-level cue regime,
      global congruence label,
      nearest-attractor congruence label.
    """
    n1_matches = nums["NP1"] == nums["NP2"]
    n3_matches = nums["NP3"] == nums["NP2"]

    if n1_matches and n3_matches:
        regime = "both_match"
    elif not n1_matches and n3_matches:
        regime = "N1_mismatch_only"
    elif n1_matches and not n3_matches:
        regime = "N3_mismatch_only"
    else:
        regime = "both_mismatch"

    global_congruence = (
        "congruent" if n1_matches and n3_matches else "incongruent"
    )
    nearest_attractor = (
        "N3_matches_N2" if n3_matches else "N3_mismatches_N2"
    )

    return regime, global_congruence, nearest_attractor


def build_prompt(prefix: str, option_a: str, option_b: str) -> str:
    """Construct the plain-text forced-choice prompt."""
    return (
        "Choose the grammatically correct continuation.\n\n"
        f"{prefix.rstrip()}\n\n"
        f"A. {option_a.strip()}\n"
        f"B. {option_b.strip()}\n\n"
        "Answer:"
    )
def make_record(
    source: dict[str, Any],
    order_version: int,
    grammatical_label: str,
) -> dict[str, Any]:
    nums = source["meta"]["nums"]
    pattern = number_pattern(nums)
    regime, global_congruence, nearest_attractor = cue_regime(nums)

    good = source["good"].strip()
    bad = source["bad"].strip()

    if grammatical_label == "A":
        option_a = good
        option_b = bad
    elif grammatical_label == "B":
        option_a = bad
        option_b = good
    else:
        raise ValueError(f"Unexpected grammatical label: {grammatical_label}")

    return {
        "prompt_id": f'{source["pair_id"]}_order{order_version}',
        "base_item_id": source["pair_id"],
        "source_family_id": source["meta"]["family_id"],
        "condition": source["condition"],
        "checkpoint": source["meta"]["checkpoint"],
        "order_version": order_version,
        "grammatical_label": grammatical_label,
        "prefix": source["prefix"],
        "option_a": option_a,
        "option_b": option_b,
        "prompt": build_prompt(source["prefix"], option_a, option_b),
        "number_pattern": pattern,
        "cue_regime": regime,
        "global_congruence": global_congruence,
        "nearest_attractor_congruence": nearest_attractor,
        "nums": nums,
        "lex": source["meta"]["lex"],
        "source_good": source["good"],
        "source_bad": source["bad"],
    }


def validate(records: list[dict[str, Any]]) -> None:
    if len(records) != 4000:
        raise ValueError(f"Expected 4000 records, found {len(records)}")

    prompt_ids = [row["prompt_id"] for row in records]
    if len(prompt_ids) != len(set(prompt_ids)):
        raise ValueError("Duplicate prompt_id values found")

    base_counts = Counter(row["base_item_id"] for row in records)
    bad_base_counts = {
        item_id: count for item_id, count in base_counts.items() if count != 2
    }
    if bad_base_counts:
        raise ValueError(
            f"Each base item must occur twice. Problems: {bad_base_counts}"
        )

    label_counts = Counter(row["grammatical_label"] for row in records)
    if label_counts != Counter({"A": 2000, "B": 2000}):
        raise ValueError(
            f"Unexpected grammatical-label counts: {dict(label_counts)}"
        )

    for item_id in base_counts:
        item_rows = [row for row in records if row["base_item_id"] == item_id]
        labels = {row["grammatical_label"] for row in item_rows}
        if labels != {"A", "B"}:
            raise ValueError(
                f"{item_id} does not have both grammatical-label orders"
            )

    expected_patterns = {
        "PPP": 508,
        "PPS": 464,
        "PSP": 514,
        "PSS": 514,
        "SPP": 498,
        "SPS": 496,
        "SSP": 520,
        "SSS": 486,
    }
    observed_patterns = Counter(row["number_pattern"] for row in records)
    if dict(observed_patterns) != expected_patterns:
        raise ValueError(
            "Unexpected pattern counts.\n"
            f"Observed: {dict(observed_patterns)}\n"
            f"Expected: {expected_patterns}"
        )

    expected_regimes = {
        "both_match": 994,
        "N1_mismatch_only": 1012,
        "N3_mismatch_only": 984,
        "both_mismatch": 1010,
    }
    observed_regimes = Counter(row["cue_regime"] for row in records)
    if dict(observed_regimes) != expected_regimes:
        raise ValueError(
            "Unexpected regime counts.\n"
            f"Observed: {dict(observed_regimes)}\n"
            f"Expected: {expected_regimes}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert CE2 V2 items to counterbalanced forced-choice prompts."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(
            "data/stimuli/minpairs.ce2fam.generated.v0.7_nf2000.jsonl"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "data/stimuli/minpairs.ce2.v2.forced_choice.counterbalanced.v1.jsonl"
        ),
    )
    args = parser.parse_args()

    records: list[dict[str, Any]] = []

    with args.input.open("r", encoding="utf-8") as source_file:
        for line_number, line in enumerate(source_file, start=1):
            source = json.loads(line)

            if source["meta"]["checkpoint"] != "V2":
                continue

            records.append(
                make_record(
                    source=source,
                    order_version=1,
                    grammatical_label="A",
                )
            )
            records.append(
                make_record(
                    source=source,
                    order_version=2,
                    grammatical_label="B",
                )
            )

    validate(records)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as output_file:
        for record in records:
            output_file.write(
                json.dumps(record, ensure_ascii=False) + "\n"
            )

    print(f"Wrote {len(records)} records to {args.output}")

    print("\nNumber-pattern counts:")
    for key, value in sorted(Counter(
        row["number_pattern"] for row in records
    ).items()):
        print(f"  {key}: {value}")

    print("\nCue-regime counts:")
    for key, value in sorted(Counter(
        row["cue_regime"] for row in records
    ).items()):
        print(f"  {key}: {value}")

    print("\nGlobal-congruence counts:")
    for key, value in sorted(Counter(
        row["global_congruence"] for row in records
    ).items()):
        print(f"  {key}: {value}")

    print("\nGrammatical-label counts:")
    for key, value in sorted(Counter(
        row["grammatical_label"] for row in records
    ).items()):
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
