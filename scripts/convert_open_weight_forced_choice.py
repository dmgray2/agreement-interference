#!/usr/bin/env python3

import argparse
import csv
from pathlib import Path
from typing import Any

from score_forced_choice_results import (
    combine_counterbalanced_rows,
    write_jsonl,
)


def load_open_weight_csv(
    path: Path,
) -> list[dict[str, Any]]:
    with path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as source:
        return list(csv.DictReader(source))


def convert_prompt_rows(
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if len(rows) != 4000:
        raise ValueError(
            f"Expected 4,000 prompt rows, found {len(rows)}"
        )

    converted: list[dict[str, Any]] = []

    model_names = {
        row["model"]
        for row in rows
    }

    if len(model_names) != 1:
        raise ValueError(
            f"Expected one model, found {sorted(model_names)}"
        )

    model_name = next(iter(model_names))

    for row in rows:
        answer = row["answer"]

        if answer not in {"A", "B"}:
            raise ValueError(
                f"Unexpected constrained answer: {answer!r}"
            )

        correct = int(row["correct"])

        if correct not in {0, 1}:
            raise ValueError(
                f"Unexpected correctness value: {correct!r}"
            )

        converted.append(
            {
                "provider": "open_weight",
                "requested_model": model_name,
                "prompt_id": row["prompt_id"],
                "base_item_id": row["base_item_id"],
                "source_family_id": row[
                    "source_family_id"
                ],
                "order_version": row["order_version"],
                "grammatical_label": row[
                    "grammatical_label"
                ],
                "number_pattern": row["number_pattern"],
                "cue_regime": row["cue_regime"],
                "global_congruence": row[
                    "global_congruence"
                ],
                "nearest_attractor_congruence": row[
                    "nearest_attractor_congruence"
                ],
                "nums": {
                    "NP1": row["np1_number"],
                    "NP2": row["np2_number"],
                    "NP3": row["np3_number"],
                },
                "lex": {
                    "NP1": row["np1_lexeme"],
                    "NP2": row["np2_lexeme"],
                    "NP3": row["np3_lexeme"],
                    "V2": row["v2_lexeme"],
                    "V3": row["v3_lexeme"],
                },
                "result_status": "success",
                "raw_response": answer,
                "response_valid": True,
                "parsed_label": answer,
                "response_source": (
                    "constrained_next_token_logprob"
                ),
                "correct": correct,
                "technical_missing": False,
                "returned_model": model_name,
                "input_tokens": None,
                "output_tokens": None,
                "reasoning_tokens": None,
                "stop_reason": None,
                "incomplete_reason": None,
            }
        )

    return converted


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Convert an open-weight forced-choice CSV "
            "to the common paired-item JSONL format."
        )
    )

    parser.add_argument(
        "--input",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--prompt-output",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--item-output",
        required=True,
        type=Path,
    )

    args = parser.parse_args()

    source_rows = load_open_weight_csv(args.input)

    prompt_rows = convert_prompt_rows(source_rows)

    item_rows = combine_counterbalanced_rows(
        prompt_rows
    )

    if len(item_rows) != 2000:
        raise ValueError(
            f"Expected 2,000 paired items, "
            f"found {len(item_rows)}"
        )

    write_jsonl(args.prompt_output, prompt_rows)
    write_jsonl(args.item_output, item_rows)

    print(f"Model: {prompt_rows[0]['requested_model']}")
    print(f"Prompt rows: {len(prompt_rows)}")
    print(f"Paired item rows: {len(item_rows)}")

    score_counts = {
        value: sum(
            row["counterbalanced_accuracy"] == value
            for row in item_rows
        )
        for value in (0.0, 0.5, 1.0)
    }

    print(
        "Paired scores: "
        f"0.0={score_counts[0.0]}, "
        f"0.5={score_counts[0.5]}, "
        f"1.0={score_counts[1.0]}"
    )

    print(f"Saved prompt rows to: {args.prompt_output}")
    print(f"Saved paired items to: {args.item_output}")


if __name__ == "__main__":
    main()
