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


MODEL_ID = "claude-opus-4-5-20251101"
MAX_TOKENS = 100

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

ANSWER_TOOL = {
    "name": "record_answer",
    "description": (
        "Record the selected answer to the forced-choice "
        "grammaticality question."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "answer": {
                "type": "string",
                "enum": ["A", "B"],
            }
        },
        "required": ["answer"],
        "additionalProperties": False,
    },
}

TOOL_CHOICE = {
    "type": "tool",
    "name": "record_answer",
}


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


def make_batch_requests(
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    requests: list[dict[str, Any]] = []

    for row in rows:
        requests.append(
            {
                "custom_id": row["prompt_id"],
                "params": {
                    "model": MODEL_ID,
                    "max_tokens": MAX_TOKENS,
                    "tools": [ANSWER_TOOL],
                    "tool_choice": TOOL_CHOICE,
                    "messages": [
                        {
                            "role": "user",
                            "content": row["prompt"],
                        }
                    ],
                },
            }
        )

    return requests


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


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare or submit an Opus 4.1 forced-choice "
            "batch using an enum-constrained answer tool."
        )
    )

    parser.add_argument(
        "--order",
        required=True,
        choices=("A", "B"),
        help=(
            "A means the grammatical continuation is labelled A; "
            "B means it is labelled B."
        ),
    )

    parser.add_argument(
        "--submit",
        action="store_true",
        help=(
            "Submit the batch. Without this flag, "
            "perform a dry run only."
        ),
    )

    args = parser.parse_args()

    order = args.order
    seed = SEEDS[order]
    source_path = ORDER_FILES[order]

    rows = load_jsonl(source_path)
    validate_source_rows(rows, order)

    shuffled_rows = list(rows)
    random.Random(seed).shuffle(shuffled_rows)

    requests = make_batch_requests(shuffled_rows)

    run_name = (
        f"anthropic_opus-4.5_toolforced_"
        f"order{order}_v1"
    )

    request_manifest_path = Path(
        "data/batch_requests/anthropic"
    ) / f"{run_name}.jsonl"

    metadata_path = Path(
        "results/batch_manifests/anthropic"
    ) / f"{run_name}.json"

    if metadata_path.exists():
        existing = json.loads(
            metadata_path.read_text(encoding="utf-8")
        )

        if existing.get("submitted"):
            raise RuntimeError(
                "This batch has already been submitted. "
                f"Recorded batch ID: "
                f"{existing.get('batch_id')}"
            )

    write_jsonl(request_manifest_path, requests)

    metadata: dict[str, Any] = {
        "provider": "anthropic",
        "run_name": run_name,
        "requested_model": MODEL_ID,
        "answer_order": order,
        "source_file": str(source_path),
        "request_manifest_file": str(
            request_manifest_path
        ),
        "number_of_requests": len(requests),
        "random_seed": seed,
        "max_tokens": MAX_TOKENS,
        "temperature_parameter": "omitted",
        "thinking_parameter": "omitted",
        "response_interface": "forced_tool_call",
        "tool_name": ANSWER_TOOL["name"],
        "allowed_answers": ["A", "B"],
        "tool_choice": TOOL_CHOICE,
        "source_prompt_modified": False,
        "protocol_status": (
            "corrected_response_interface_before_analysis"
        ),
        "included_in_confirmatory_analysis": True,
        "correction_reason": (
            "Preliminary setup runs showed that free-text "
            "responses could contain extended reasoning and reach "
            "the output-token ceiling before returning a scorable "
            "choice. Before analyzing model performance, the "
            "response interface was changed to a forced tool call "
            "whose answer field permits only A or B. The stimulus "
            "prompts themselves were not changed."
        ),
        "preliminary_runs_excluded": True,
        "created_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "submitted": False,
        "batch_id": None,
    }

    print(f"Model: {MODEL_ID}")
    print(f"Answer order: {order}")
    print(f"Source rows: {len(rows)}")
    print(f"Random seed: {seed}")
    print(f"Maximum output tokens: {MAX_TOKENS}")
    print("Response interface: forced tool call")
    print("Allowed answers: A, B")
    print("Source prompts modified: no")
    print(f"Request manifest: {request_manifest_path}")
    print(f"Metadata file: {metadata_path}")

    print("\nFirst three shuffled request IDs:")

    for request in requests[:3]:
        print(f"  {request['custom_id']}")

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

    print("\nSubmitting tool-forced Opus 4.5 batch...")

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
