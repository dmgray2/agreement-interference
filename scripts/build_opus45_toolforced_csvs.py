#!/usr/bin/env python3

import csv
import json
from pathlib import Path
from typing import Any


STIMULUS_FILES = {
    "A": Path(
        "data/stimuli/"
        "minpairs.ce2.v2.forced_choice.orderA.v1.jsonl"
    ),
    "B": Path(
        "data/stimuli/"
        "minpairs.ce2.v2.forced_choice.orderB.v1.jsonl"
    ),
}

RESULT_FILES = {
    "A": Path(
        "results/parsed_batches/anthropic/"
        "anthropic_opus-4.5_toolforced_orderA_v1.jsonl"
    ),
    "B": Path(
        "results/parsed_batches/anthropic/"
        "anthropic_opus-4.5_toolforced_orderB_v1.jsonl"
    ),
}

OUTPUT_DIRECTORY = Path("results/final")

PROMPT_OUTPUT = (
    OUTPUT_DIRECTORY
    / "anthropic_opus-4.5_toolforced_prompt_level_v1.csv"
)

PAIRED_OUTPUT = (
    OUTPUT_DIRECTORY
    / "anthropic_opus-4.5_toolforced_paired_v1.csv"
)


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


def write_csv(
    path: Path,
    rows: list[dict[str, Any]],
    fieldnames: list[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as output:
        writer = csv.DictWriter(
            output,
            fieldnames=fieldnames,
            extrasaction="raise",
        )

        writer.writeheader()
        writer.writerows(rows)


def flatten_stimulus(
    row: dict[str, Any],
) -> dict[str, Any]:
    nums = row.get("nums", {})
    lex = row.get("lex", {})

    return {
        "prompt_id": row["prompt_id"],
        "base_item_id": row["base_item_id"],
        "source_family_id": row["source_family_id"],
        "condition": row["condition"],
        "checkpoint": row["checkpoint"],
        "order_version": row["order_version"],
        "grammatical_label": row["grammatical_label"],
        "prefix": row["prefix"],
        "option_a": row["option_a"],
        "option_b": row["option_b"],
        "prompt": row["prompt"],
        "number_pattern": row["number_pattern"],
        "cue_regime": row["cue_regime"],
        "global_congruence": row["global_congruence"],
        "nearest_attractor_congruence": (
            row["nearest_attractor_congruence"]
        ),
        "np1_number": nums.get("NP1"),
        "np2_number": nums.get("NP2"),
        "np3_number": nums.get("NP3"),
        "np1_lexeme": lex.get("NP1"),
        "np2_lexeme": lex.get("NP2"),
        "np3_lexeme": lex.get("NP3"),
        "v2_lexeme": lex.get("V2"),
        "v3_lexeme": lex.get("V3"),
        "source_good": row["source_good"],
        "source_bad": row["source_bad"],
    }


def main() -> None:
    prompt_rows: list[dict[str, Any]] = []
    prompt_rows_by_base: dict[
        str,
        dict[str, dict[str, Any]],
    ] = {}

    for order in ("A", "B"):
        stimulus_rows = load_jsonl(
            STIMULUS_FILES[order]
        )

        result_rows = load_jsonl(
            RESULT_FILES[order]
        )

        if len(stimulus_rows) != 2000:
            raise ValueError(
                f"Order {order}: expected 2,000 stimulus rows, "
                f"found {len(stimulus_rows)}"
            )

        if len(result_rows) != 2000:
            raise ValueError(
                f"Order {order}: expected 2,000 result rows, "
                f"found {len(result_rows)}"
            )

        stimuli_by_prompt_id = {
            row["prompt_id"]: row
            for row in stimulus_rows
        }

        results_by_prompt_id = {
            row["custom_id"]: row
            for row in result_rows
        }

        if len(stimuli_by_prompt_id) != 2000:
            raise ValueError(
                f"Order {order}: duplicate stimulus prompt IDs"
            )

        if len(results_by_prompt_id) != 2000:
            raise ValueError(
                f"Order {order}: duplicate result custom IDs"
            )

        missing_results = (
            set(stimuli_by_prompt_id)
            - set(results_by_prompt_id)
        )

        extra_results = (
            set(results_by_prompt_id)
            - set(stimuli_by_prompt_id)
        )

        if missing_results:
            raise ValueError(
                f"Order {order}: "
                f"{len(missing_results)} missing results"
            )

        if extra_results:
            raise ValueError(
                f"Order {order}: "
                f"{len(extra_results)} unexpected results"
            )

        for prompt_id, stimulus in (
            stimuli_by_prompt_id.items()
        ):
            result = results_by_prompt_id[prompt_id]

            if result.get("status") != "success":
                raise ValueError(
                    f"{prompt_id}: result status is "
                    f"{result.get('status')!r}"
                )

            answer = result.get("answer")

            if answer not in ("A", "B"):
                raise ValueError(
                    f"{prompt_id}: invalid answer {answer!r}"
                )

            grammatical_label = stimulus[
                "grammatical_label"
            ]

            is_correct = int(
                answer == grammatical_label
            )

            prompt_row = {
                **flatten_stimulus(stimulus),
                "model": (
                    "claude-opus-4-5-20251101"
                ),
                "answer_order": order,
                "model_answer": answer,
                "is_correct": is_correct,
                "status": result["status"],
                "stop_reason": result.get(
                    "stop_reason"
                ),
                "input_tokens": result.get(
                    "input_tokens"
                ),
                "output_tokens": result.get(
                    "output_tokens"
                ),
                "visible_text": result.get(
                    "visible_text",
                    "",
                ),
                "response_interface": (
                    "forced_tool_call"
                ),
            }

            prompt_rows.append(prompt_row)

            base_item_id = stimulus[
                "base_item_id"
            ]

            prompt_rows_by_base.setdefault(
                base_item_id,
                {},
            )

            if order in prompt_rows_by_base[
                base_item_id
            ]:
                raise ValueError(
                    f"{base_item_id}: duplicate order {order}"
                )

            prompt_rows_by_base[
                base_item_id
            ][order] = prompt_row

    if len(prompt_rows) != 4000:
        raise ValueError(
            f"Expected 4,000 prompt rows, "
            f"found {len(prompt_rows)}"
        )

    if len(prompt_rows_by_base) != 2000:
        raise ValueError(
            f"Expected 2,000 paired base items, "
            f"found {len(prompt_rows_by_base)}"
        )

    prompt_rows.sort(
        key=lambda row: (
            row["base_item_id"],
            row["order_version"],
        )
    )

    paired_rows: list[dict[str, Any]] = []

    for base_item_id in sorted(
        prompt_rows_by_base
    ):
        pair = prompt_rows_by_base[base_item_id]

        if set(pair) != {"A", "B"}:
            raise ValueError(
                f"{base_item_id}: incomplete pair "
                f"{sorted(pair)}"
            )

        order_a = pair["A"]
        order_b = pair["B"]

        metadata_fields = [
            "source_family_id",
            "condition",
            "checkpoint",
            "prefix",
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

        for field in metadata_fields:
            if order_a[field] != order_b[field]:
                raise ValueError(
                    f"{base_item_id}: order metadata "
                    f"disagree for {field}"
                )

        correct_a = int(order_a["is_correct"])
        correct_b = int(order_b["is_correct"])

        paired_score = (
            correct_a + correct_b
        ) / 2

        paired_rows.append(
            {
                "model": (
                    "claude-opus-4-1-20250805"
                ),
                "base_item_id": base_item_id,
                "source_family_id": order_a[
                    "source_family_id"
                ],
                "condition": order_a["condition"],
                "checkpoint": order_a["checkpoint"],
                "prefix": order_a["prefix"],
                "number_pattern": order_a[
                    "number_pattern"
                ],
                "cue_regime": order_a[
                    "cue_regime"
                ],
                "global_congruence": order_a[
                    "global_congruence"
                ],
                "nearest_attractor_congruence": (
                    order_a[
                        "nearest_attractor_congruence"
                    ]
                ),
                "np1_number": order_a["np1_number"],
                "np2_number": order_a["np2_number"],
                "np3_number": order_a["np3_number"],
                "np1_lexeme": order_a["np1_lexeme"],
                "np2_lexeme": order_a["np2_lexeme"],
                "np3_lexeme": order_a["np3_lexeme"],
                "v2_lexeme": order_a["v2_lexeme"],
                "v3_lexeme": order_a["v3_lexeme"],
                "source_good": order_a["source_good"],
                "source_bad": order_a["source_bad"],
                "order_a_prompt_id": order_a[
                    "prompt_id"
                ],
                "order_a_option_a": order_a[
                    "option_a"
                ],
                "order_a_option_b": order_a[
                    "option_b"
                ],
                "order_a_answer": order_a[
                    "model_answer"
                ],
                "order_a_correct": correct_a,
                "order_b_prompt_id": order_b[
                    "prompt_id"
                ],
                "order_b_option_a": order_b[
                    "option_a"
                ],
                "order_b_option_b": order_b[
                    "option_b"
                ],
                "order_b_answer": order_b[
                    "model_answer"
                ],
                "order_b_correct": correct_b,
                "paired_score": paired_score,
                "response_pattern": (
                    f"{order_a['model_answer']}"
                    f"{order_b['model_answer']}"
                ),
                "response_interface": (
                    "forced_tool_call"
                ),
            }
        )

    prompt_fieldnames = list(
        prompt_rows[0].keys()
    )

    paired_fieldnames = list(
        paired_rows[0].keys()
    )

    write_csv(
        PROMPT_OUTPUT,
        prompt_rows,
        prompt_fieldnames,
    )

    write_csv(
        PAIRED_OUTPUT,
        paired_rows,
        paired_fieldnames,
    )

    score_counts = {
        0.0: 0,
        0.5: 0,
        1.0: 0,
    }

    for row in paired_rows:
        score_counts[row["paired_score"]] += 1

    print(f"Prompt-level rows: {len(prompt_rows)}")
    print(f"Paired rows: {len(paired_rows)}")
    print(f"Paired score counts: {score_counts}")
    print()
    print("Saved prompt-level CSV:")
    print(f"  {PROMPT_OUTPUT}")
    print("Saved paired CSV:")
    print(f"  {PAIRED_OUTPUT}")


if __name__ == "__main__":
    main()
