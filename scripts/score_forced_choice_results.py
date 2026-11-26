#!/usr/bin/env python3

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


VALID_RESPONSE_PATTERN = re.compile(r"^\s*([AaBb])\.?\s*$")


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


def write_jsonl(
    path: Path,
    rows: list[dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as output:
        for row in rows:
            output.write(
                json.dumps(row, ensure_ascii=False) + "\n"
            )
def parse_visible_response(raw_response: Any) -> tuple[bool, str | None]:
    """
    Apply the preregistered visible-response rule.

    Valid:
      A
      a
      B
      b
      optionally followed by one period
      with surrounding whitespace allowed
    """
    if not isinstance(raw_response, str):
        return False, None

    match = VALID_RESPONSE_PATTERN.fullmatch(raw_response)

    if match is None:
        return False, None

    return True, match.group(1).upper()


def parse_result_response(
    result: dict[str, Any],
) -> tuple[bool, str | None, str]:
    """
    Extract a preregistered A/B response from either a forced
    tool call or ordinary visible text.

    Priority:
      1. OpenAI forced-tool field: tool_answer
      2. Anthropic forced-tool field: answer
      3. Ordinary visible-text response
    """
    tool_answer = result.get("tool_answer")

    if tool_answer in {"A", "B"}:
        return True, tool_answer, "openai_tool"

    anthropic_answer = result.get("answer")

    if anthropic_answer in {"A", "B"}:
        return True, anthropic_answer, "anthropic_tool"

    visible_text = result.get("raw_response")

    if visible_text is None:
        visible_text = result.get("visible_text")

    response_valid, parsed_label = parse_visible_response(
        visible_text
    )

    return response_valid, parsed_label, "visible_text"


def index_unique(
    rows: list[dict[str, Any]],
    field: str,
    source_name: str,
) -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}

    for row in rows:
        key = row.get(field)

        if not isinstance(key, str) or not key:
            raise ValueError(
                f"Missing or invalid {field} in {source_name}"
            )

        if key in index:
            raise ValueError(
                f"Duplicate {field}={key!r} in {source_name}"
            )

        index[key] = row

    return index


def score_prompt_rows(
    stimulus_rows: list[dict[str, Any]],
    result_rows: list[dict[str, Any]],
    provider: str,
    requested_model: str,
) -> list[dict[str, Any]]:
    stimuli = index_unique(
        stimulus_rows,
        field="prompt_id",
        source_name="stimulus file",
    )

    results = index_unique(
        result_rows,
        field="custom_id",
        source_name="result file",
    )

    missing_results = sorted(set(stimuli) - set(results))
    unexpected_results = sorted(set(results) - set(stimuli))

    if missing_results:
        raise ValueError(
            f"{len(missing_results)} stimulus prompts have no result"
        )

    if unexpected_results:
        raise ValueError(
            f"{len(unexpected_results)} results have no stimulus row"
        )

    scored_rows: list[dict[str, Any]] = []

    for prompt_id, stimulus in stimuli.items():
        result = results[prompt_id]
        result_status = result.get("status")

        technical_missing = result_status != "success"

        if technical_missing:
            response_valid = None
            parsed_label = None
            response_source = None
            correct = None
        else:
            response_valid, parsed_label, response_source = (
                parse_result_response(result)
            )

            if response_valid:
                correct = int(
                    parsed_label == stimulus["grammatical_label"]
                )
            else:
                # A substantive malformed output is invalid and incorrect.
                correct = 0

        scored_rows.append(
            {
                "provider": provider,
                "requested_model": requested_model,
                "prompt_id": prompt_id,
                "base_item_id": stimulus["base_item_id"],
                "source_family_id": stimulus[
                    "source_family_id"
                ],
                "order_version": stimulus["order_version"],
                "grammatical_label": stimulus[
                    "grammatical_label"
                ],
                "number_pattern": stimulus["number_pattern"],
                "cue_regime": stimulus["cue_regime"],
                "global_congruence": stimulus[
                    "global_congruence"
                ],
                "nearest_attractor_congruence": stimulus[
                    "nearest_attractor_congruence"
                ],
                "nums": stimulus["nums"],
                "lex": stimulus["lex"],
                "result_status": result_status,
                "raw_response": result.get("raw_response"),
                "response_valid": response_valid,
                "parsed_label": parsed_label,
                "response_source": response_source,
                "correct": correct,
                "technical_missing": technical_missing,
                "returned_model": result.get("returned_model"),
                "input_tokens": result.get("input_tokens"),
                "output_tokens": result.get("output_tokens"),
                "reasoning_tokens": result.get(
                    "reasoning_tokens"
                ),
                "stop_reason": result.get("stop_reason"),
                "incomplete_reason": result.get(
                    "incomplete_reason"
                ),
            }
        )

    return scored_rows


def combine_counterbalanced_rows(
    prompt_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}

    for row in prompt_rows:
        grouped.setdefault(
            row["base_item_id"],
            [],
        ).append(row)

    item_rows: list[dict[str, Any]] = []

    for base_item_id, pair in grouped.items():
        if len(pair) != 2:
            raise ValueError(
                f"{base_item_id} has {len(pair)} prompt rows, expected 2"
            )

        by_label = {
            row["grammatical_label"]: row
            for row in pair
        }

        if set(by_label) != {"A", "B"}:
            raise ValueError(
                f"{base_item_id} does not contain one A-order "
                "and one B-order presentation"
            )

        row_a = by_label["A"]
        row_b = by_label["B"]

        for field in (
            "number_pattern",
            "cue_regime",
            "global_congruence",
            "nearest_attractor_congruence",
            "nums",
            "lex",
        ):
            if row_a[field] != row_b[field]:
                raise ValueError(
                    f"{base_item_id} differs across orders in {field}"
                )

        technical_missing = bool(
            row_a["technical_missing"]
            or row_b["technical_missing"]
        )

        if technical_missing:
            counterbalanced_accuracy = None
            strict_paired_correct = None
            order_inconsistent = None
        else:
            correct_a = int(row_a["correct"])
            correct_b = int(row_b["correct"])

            counterbalanced_accuracy = (
                correct_a + correct_b
            ) / 2

            strict_paired_correct = int(
                correct_a == 1 and correct_b == 1
            )

            order_inconsistent = int(
                correct_a != correct_b
            )

        item_rows.append(
            {
                "provider": row_a["provider"],
                "requested_model": row_a[
                    "requested_model"
                ],
                "base_item_id": base_item_id,
                "source_family_id": row_a[
                    "source_family_id"
                ],
                "number_pattern": row_a[
                    "number_pattern"
                ],
                "cue_regime": row_a["cue_regime"],
                "global_congruence": row_a[
                    "global_congruence"
                ],
                "nearest_attractor_congruence": row_a[
                    "nearest_attractor_congruence"
                ],
                "nums": row_a["nums"],
                "lex": row_a["lex"],
                "correct_order_A": row_a["correct"],
                "correct_order_B": row_b["correct"],
                "valid_order_A": row_a["response_valid"],
                "valid_order_B": row_b["response_valid"],
                "raw_response_order_A": row_a[
                    "raw_response"
                ],
                "raw_response_order_B": row_b[
                    "raw_response"
                ],
                "technical_missing": technical_missing,
                "counterbalanced_accuracy": (
                    counterbalanced_accuracy
                ),
                "strict_paired_correct": (
                    strict_paired_correct
                ),
                "order_inconsistent": order_inconsistent,
            }
        )

    return item_rows


def print_summary(
    prompt_rows: list[dict[str, Any]],
    item_rows: list[dict[str, Any]],
) -> None:
    status_counts = Counter(
        row["result_status"]
        for row in prompt_rows
    )

    valid_counts = Counter(
        str(row["response_valid"])
        for row in prompt_rows
    )

    score_counts = Counter(
        str(row["counterbalanced_accuracy"])
        for row in item_rows
    )

    print("Prompt-level result status:")
    for key, value in sorted(status_counts.items()):
        print(f"  {key}: {value}")

    print("\nPrompt-level validity:")
    for key, value in sorted(valid_counts.items()):
        print(f"  {key}: {value}")

    print("\nItem-level counterbalanced scores:")
    for key, value in sorted(score_counts.items()):
        print(f"  {key}: {value}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Score paired counterbalanced CE-Agree forced-choice results."
        )
    )

    parser.add_argument(
        "--provider",
        required=True,
        choices=("anthropic", "openai", "test"),
    )

    parser.add_argument(
        "--model",
        required=True,
        help="Exact requested model identifier.",
    )

    parser.add_argument(
        "--order-a-stimuli",
        type=Path,
        default=Path(
            "data/stimuli/"
            "minpairs.ce2.v2.forced_choice.orderA.v1.jsonl"
        ),
    )

    parser.add_argument(
        "--order-b-stimuli",
        type=Path,
        default=Path(
            "data/stimuli/"
            "minpairs.ce2.v2.forced_choice.orderB.v1.jsonl"
        ),
    )

    parser.add_argument(
        "--order-a-results",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--order-b-results",
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

    stimuli_a = load_jsonl(args.order_a_stimuli)
    stimuli_b = load_jsonl(args.order_b_stimuli)

    results_a = load_jsonl(args.order_a_results)
    results_b = load_jsonl(args.order_b_results)

    prompt_rows_a = score_prompt_rows(
        stimulus_rows=stimuli_a,
        result_rows=results_a,
        provider=args.provider,
        requested_model=args.model,
    )

    prompt_rows_b = score_prompt_rows(
        stimulus_rows=stimuli_b,
        result_rows=results_b,
        provider=args.provider,
        requested_model=args.model,
    )

    prompt_rows = prompt_rows_a + prompt_rows_b

    item_rows = combine_counterbalanced_rows(
        prompt_rows
    )

    if len(prompt_rows) != 4000:
        raise ValueError(
            f"Expected 4,000 prompt rows, found {len(prompt_rows)}"
        )

    if len(item_rows) != 2000:
        raise ValueError(
            f"Expected 2,000 item rows, found {len(item_rows)}"
        )

    write_jsonl(args.prompt_output, prompt_rows)
    write_jsonl(args.item_output, item_rows)

    print_summary(prompt_rows, item_rows)

    print(f"\nSaved prompt-level scores to {args.prompt_output}")
    print(f"Saved item-level scores to {args.item_output}")


if __name__ == "__main__":
    main()
