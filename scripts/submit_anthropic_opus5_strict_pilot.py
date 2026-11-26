#!/usr/bin/env python3

import argparse
import json
import os
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import anthropic


MODEL_ID = "claude-opus-5"
MAX_TOKENS = 1000
PILOT_ITEMS_PER_ORDER = 5

ORDER_FILES = {
    "A": Path(
        "data/stimuli/"
        "minpairs.ce2.v2.forced_choice.orderA.v1.jsonl"
    ),
    "B": Path(
        "data/stimuli/"
        "minpairs.ce2.v2.forced_choice.orderB.v1.jsonl"
    ),
}

SEEDS = {
    "A": 50001,
    "B": 50002,
}

STRICT_INSTRUCTION = (
    "Respond with exactly one letter, A or B. "
    "Do not include punctuation or an explanation."
)

RUN_NAME = "anthropic_opus-5_strictformat_pilot10_v1"


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


def validate_source_rows(
    rows: list[dict[str, Any]],
    order: str,
) -> None:
    if len(rows) != 2000:
        raise ValueError(
            f"Expected 2,000 source rows, found {len(rows)}"
        )

    prompt_ids = [row["prompt_id"] for row in rows]

    if len(prompt_ids) != len(set(prompt_ids)):
        raise ValueError("Duplicate prompt_id values found")

    wrong_labels = [
        row["prompt_id"]
        for row in rows
        if row["grammatical_label"] != order
    ]

    if wrong_labels:
        raise ValueError(
            f"{len(wrong_labels)} rows have an unexpected "
            f"grammatical label for order {order}"
        )

    empty_prompts = [
        row["prompt_id"]
        for row in rows
        if not row.get("prompt", "").strip()
    ]

    if empty_prompts:
        raise ValueError(
            f"{len(empty_prompts)} rows contain empty prompts"
        )


def add_strict_instruction(prompt: str) -> str:
    """
    Preserve the original prompt, but place the new formatting
    instruction immediately before the final Answer: cue.
    """
    text = prompt.rstrip()

    if text.endswith("Answer:"):
        text = text[: -len("Answer:")].rstrip()

    return (
        f"{text}\n\n"
        f"{STRICT_INSTRUCTION}\n\n"
        "Answer:"
    )


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


def write_json(
    path: Path,
    value: dict[str, Any],
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


def load_existing_metadata(
    path: Path,
) -> dict[str, Any] | None:
    if not path.exists():
        return None

    with path.open("r", encoding="utf-8") as source:
        return json.load(source)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare or submit a ten-item Opus 5 pilot "
            "testing an explicit strict-format instruction."
        )
    )

    parser.add_argument(
        "--submit",
        action="store_true",
        help=(
            "Actually submit the pilot batch. Without this "
            "flag, the script performs a dry run only."
        ),
    )

    args = parser.parse_args()

    requests: list[dict[str, Any]] = []
    selections: list[dict[str, Any]] = []

    for order in ("A", "B"):
        source_path = ORDER_FILES[order]
        rows = load_jsonl(source_path)
        validate_source_rows(rows, order)

        shuffled_rows = list(rows)
        random.Random(SEEDS[order]).shuffle(shuffled_rows)

        selected_rows = shuffled_rows[
            :PILOT_ITEMS_PER_ORDER
        ]

        for row in selected_rows:
            custom_id = (
                f"strictpilot_{order}_{row['prompt_id']}"
            )

            if len(custom_id) > 64:
                raise ValueError(
                    f"custom_id exceeds 64 characters: {custom_id}"
                )

            strict_prompt = add_strict_instruction(
                row["prompt"]
            )

            requests.append(
                {
                    "custom_id": custom_id,
                    "params": {
                        "model": MODEL_ID,
                        "max_tokens": MAX_TOKENS,
                        "messages": [
                            {
                                "role": "user",
                                "content": strict_prompt,
                            }
                        ],
                    },
                }
            )

            selections.append(
                {
                    "custom_id": custom_id,
                    "source_prompt_id": row["prompt_id"],
                    "answer_order": order,
                    "grammatical_label": (
                        row["grammatical_label"]
                    ),
                    "source_file": str(source_path),
                    "strict_prompt": strict_prompt,
                }
            )

    if len(requests) != 10:
        raise ValueError(
            f"Expected 10 pilot requests, found {len(requests)}"
        )

    request_manifest_path = Path(
        "data/batch_requests/anthropic"
    ) / f"{RUN_NAME}.jsonl"

    selection_manifest_path = Path(
        "data/batch_requests/anthropic"
    ) / f"{RUN_NAME}_selections.jsonl"

    metadata_path = Path(
        "results/batch_manifests/anthropic"
    ) / f"{RUN_NAME}.json"

    existing_metadata = load_existing_metadata(
        metadata_path
    )

    if (
        existing_metadata is not None
        and existing_metadata.get("submitted")
    ):
        raise RuntimeError(
            "This pilot has already been submitted. "
            f"Recorded batch ID: "
            f"{existing_metadata.get('batch_id')}"
        )

    write_jsonl(
        request_manifest_path,
        requests,
    )

    write_jsonl(
        selection_manifest_path,
        selections,
    )

    metadata: dict[str, Any] = {
        "provider": "anthropic",
        "run_name": RUN_NAME,
        "requested_model": MODEL_ID,
        "source_files": {
            order: str(path)
            for order, path in ORDER_FILES.items()
        },
        "request_manifest_file": str(
            request_manifest_path
        ),
        "selection_manifest_file": str(
            selection_manifest_path
        ),
        "number_of_requests": len(requests),
        "items_per_answer_order": (
            PILOT_ITEMS_PER_ORDER
        ),
        "random_seeds": SEEDS,
        "max_tokens": MAX_TOKENS,
        "temperature_parameter": "omitted",
        "thinking_parameter": "omitted",
        "tools": "none",
        "protocol_status": (
            "formatting_pilot_not_for_analysis"
        ),
        "included_in_confirmatory_analysis": False,
        "strict_instruction": STRICT_INSTRUCTION,
        "pilot_reason": (
            "The preceding Opus 5 run produced elaborated "
            "answers because the original prompt did not "
            "explicitly require a bare A/B response. This "
            "ten-item pilot tests an explicit response-format "
            "instruction before any further full-model run."
        ),
        "created_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "submitted": False,
        "batch_id": None,
    }

    print(f"Model: {MODEL_ID}")
    print(f"Pilot requests: {len(requests)}")
    print(
        "Answer orders: "
        f"{PILOT_ITEMS_PER_ORDER} A and "
        f"{PILOT_ITEMS_PER_ORDER} B"
    )
    print(f"Maximum output tokens: {MAX_TOKENS}")
    print(
        "Protocol status: "
        "formatting pilot, not for analysis"
    )
    print(f"Request manifest: {request_manifest_path}")
    print(
        f"Selection manifest: {selection_manifest_path}"
    )
    print(f"Metadata file: {metadata_path}")

    print("\nSelected requests:")

    for selection in selections:
        print(
            f"  {selection['custom_id']} "
            f"(correct label "
            f"{selection['grammatical_label']})"
        )

    if not args.submit:
        write_json(metadata_path, metadata)

        print(
            "\nDRY RUN ONLY: no API requests "
            "were submitted."
        )
        return

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print(
            "ERROR: ANTHROPIC_API_KEY is not set.",
            file=sys.stderr,
        )
        sys.exit(1)

    print("\nSubmitting strict-format pilot...")

    client = anthropic.Anthropic()

    batch = client.messages.batches.create(
        requests=requests
    )

    metadata.update(
        {
            "submitted": True,
            "batch_id": batch.id,
            "processing_status_at_submission": (
                batch.processing_status
            ),
            "submitted_at_utc": datetime.now(
                timezone.utc
            ).isoformat(),
        }
    )

    write_json(metadata_path, metadata)

    print(f"Submitted batch ID: {batch.id}")
    print(
        f"Processing status: "
        f"{batch.processing_status}"
    )
    print(f"Updated metadata: {metadata_path}")


if __name__ == "__main__":
    main()
