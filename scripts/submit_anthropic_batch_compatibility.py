#!/usr/bin/env python3

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import anthropic


MODELS = {
    "opus-4.1": "claude-opus-4-1-20250805",
    "opus-4.5": "claude-opus-4-5-20251101",
    "opus-5": "claude-opus-5",
}

PROMPT_FILE = Path(
    "data/stimuli/api_compatibility_prompts.v1.jsonl"
)

OUTPUT_DIR = Path(
    "results/api_compatibility/anthropic_batch"
)

MAX_TOKENS = 100


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
            "Submit two harmless compatibility prompts through "
            "the Anthropic Message Batches API."
        )
    )

    parser.add_argument(
        "--model",
        required=True,
        choices=sorted(MODELS),
    )

    parser.add_argument(
        "--submit",
        action="store_true",
    )

    args = parser.parse_args()

    rows = load_jsonl(PROMPT_FILE)

    if len(rows) != 2:
        raise ValueError(
            f"Expected two compatibility prompts, found {len(rows)}"
        )

    model_id = MODELS[args.model]

    requests = [
        {
            "custom_id": row["prompt_id"],
            "params": {
                "model": model_id,
                "max_tokens": MAX_TOKENS,
                "messages": [
                    {
                        "role": "user",
                        "content": row["prompt"],
                    }
                ],
            },
        }
        for row in rows
    ]

    metadata_path = (
        OUTPUT_DIR
        / f"anthropic_{args.model}_batch_compatibility.json"
    )

    metadata: dict[str, Any] = {
        "provider": "anthropic",
        "model_key": args.model,
        "requested_model": model_id,
        "prompt_file": str(PROMPT_FILE),
        "number_of_requests": len(requests),
        "max_tokens": MAX_TOKENS,
        "temperature_parameter": "omitted",
        "thinking_parameter": "omitted",
        "submitted": False,
        "batch_id": None,
        "created_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
    }

    print(f"Model: {model_id}")
    print(f"Requests: {len(requests)}")
    print(f"Maximum output tokens: {MAX_TOKENS}")

    if not args.submit:
        write_json(metadata_path, metadata)
        print("DRY RUN ONLY: no API requests were submitted.")
        return

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print(
            "ERROR: ANTHROPIC_API_KEY is not set.",
            file=sys.stderr,
        )
        sys.exit(1)

    client = anthropic.Anthropic()

    batch = client.messages.batches.create(
        requests=requests
    )

    metadata.update(
        {
            "submitted": True,
            "batch_id": batch.id,
            "processing_status": batch.processing_status,
            "submitted_at_utc": datetime.now(
                timezone.utc
            ).isoformat(),
        }
    )

    write_json(metadata_path, metadata)

    print(f"Submitted batch ID: {batch.id}")
    print(f"Processing status: {batch.processing_status}")
    print(f"Metadata saved to: {metadata_path}")


if __name__ == "__main__":
    main()
