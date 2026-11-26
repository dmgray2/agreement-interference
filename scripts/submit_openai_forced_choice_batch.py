#!/usr/bin/env python3

import argparse
import json
import os
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from openai import OpenAI


MODELS = {
    "gpt-3.5": "gpt-3.5-turbo-0125",
    "gpt-4.1": "gpt-4.1-2025-04-14",
    "gpt-5": "gpt-5-2025-08-07",
    "gpt-5.6-sol": "gpt-5.6-sol",
}

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
    ("gpt-3.5", "A"): 35001,
    ("gpt-3.5", "B"): 35002,
    ("gpt-4.1", "A"): 41011,
    ("gpt-4.1", "B"): 41012,
    ("gpt-5", "A"): 50011,
    ("gpt-5", "B"): 50012,
    ("gpt-5.6-sol", "A"): 56001,
    ("gpt-5.6-sol", "B"): 56002,
}

MAX_OUTPUT_TOKENS = 250

ANSWER_TOOL = {
    "type": "function",
    "name": "record_answer",
    "description": "Record the selected answer.",
    "parameters": {
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
    "strict": True,
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
    model_key: str,
    model_id: str,
) -> list[dict[str, Any]]:
    requests: list[dict[str, Any]] = []

    for row in rows:
        body: dict[str, Any] = {
            "model": model_id,
            "input": row["prompt"],
            "max_output_tokens": MAX_OUTPUT_TOKENS,
            "store": False,
        }

        if model_key in {
            "gpt-4.1",
            "gpt-5",
            "gpt-5.6-sol",
        }:
            body["tools"] = [ANSWER_TOOL]
            body["tool_choice"] = {
                "type": "function",
                "name": "record_answer",
            }

        if model_key == "gpt-5":
            body["reasoning"] = {
                "effort": "minimal"
            }

        if model_key == "gpt-5.6-sol":
            body["reasoning"] = {
                "effort": "none"
            }

        requests.append(
            {
                "custom_id": row["prompt_id"],
                "method": "POST",
                "url": "/v1/responses",
                "body": body,
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
            "Prepare or submit one OpenAI CE-Agree "
            "forced-choice batch."
        )
    )

    parser.add_argument(
        "--model",
        required=True,
        choices=sorted(MODELS),
    )

    parser.add_argument(
        "--order",
        required=True,
        choices=("A", "B"),
    )

    parser.add_argument(
        "--submit",
        action="store_true",
        help=(
            "Actually upload and submit the batch. "
            "Without this flag, perform a dry run only."
        ),
    )

    args = parser.parse_args()

    model_key = args.model
    model_id = MODELS[model_key]
    order = args.order
    seed = SEEDS[(model_key, order)]
    source_path = ORDER_FILES[order]

    rows = load_jsonl(source_path)
    validate_source_rows(rows, order)

    shuffled_rows = list(rows)
    random.Random(seed).shuffle(shuffled_rows)

    requests = make_batch_requests(
        rows=shuffled_rows,
        model_key=model_key,
        model_id=model_id,
    )
    run_name = f"openai_{model_key}_order{order}_v1"

    request_manifest_path = Path(
        "data/batch_requests/openai"
    ) / f"{run_name}.jsonl"

    metadata_path = Path(
        "results/batch_manifests/openai"
    ) / f"{run_name}.json"

    write_jsonl(request_manifest_path, requests)

    metadata: dict[str, Any] = {
        "provider": "openai",
        "run_name": run_name,
        "requested_model": model_id,
        "answer_order": order,
        "source_file": str(source_path),
        "request_manifest_file": str(request_manifest_path),
        "number_of_requests": len(requests),
        "random_seed": seed,
        "max_output_tokens": MAX_OUTPUT_TOKENS,
        "endpoint": "/v1/responses",
        "store": False,
        "response_mode": (
            "free_text"
            if model_key == "gpt-3.5"
            else "forced_record_answer_tool"
        ),
        "reasoning_effort": (
        "minimal"
        if model_key == "gpt-5"
        else "none"
        if model_key == "gpt-5.6-sol"
        else None
        ),
        "created_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "submitted": False,
        "input_file_id": None,
        "batch_id": None,
    }

    print(f"Model: {model_id}")
    print(f"Answer order: {order}")
    print(f"Source rows: {len(rows)}")
    print(f"Random seed: {seed}")
    print(
        f"Maximum output tokens: {MAX_OUTPUT_TOKENS}"
    )
    print(f"Request manifest: {request_manifest_path}")
    print(f"Metadata file: {metadata_path}")

    print("\nFirst three shuffled request IDs:")
    for request in requests[:3]:
        print(f"  {request['custom_id']}")

    if not args.submit:
        write_json(metadata_path, metadata)

        print(
            "\nDRY RUN ONLY: no experimental API requests "
            "were submitted."
        )
        return

    if not os.environ.get("OPENAI_API_KEY"):
        print(
            "ERROR: OPENAI_API_KEY is not set.",
            file=sys.stderr,
        )
        sys.exit(1)

    client = OpenAI()

    print("\nUploading batch input file...")

    with request_manifest_path.open("rb") as input_file:
        uploaded_file = client.files.create(
            file=input_file,
            purpose="batch",
        )

    print(f"Uploaded file ID: {uploaded_file.id}")
    print("Submitting experimental batch...")

    batch = client.batches.create(
        input_file_id=uploaded_file.id,
        endpoint="/v1/responses",
        completion_window="24h",
        metadata={
            "run_name": run_name,
            "answer_order": order,
        },
    )

    metadata.update(
        {
            "submitted": True,
            "input_file_id": uploaded_file.id,
            "batch_id": batch.id,
            "batch_status": batch.status,
            "submitted_at_utc": datetime.now(
                timezone.utc
            ).isoformat(),
        }
    )

    write_json(metadata_path, metadata)

    print(f"Submitted batch ID: {batch.id}")
    print(f"Batch status: {batch.status}")
    print(f"Updated metadata: {metadata_path}")


if __name__ == "__main__":
    main()
