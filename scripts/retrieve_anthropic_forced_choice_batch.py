#!/usr/bin/env python3

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import anthropic


MODEL_KEYS = (
    "opus-4.1",
    "opus-4.5",
    "opus-5",
)


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as source:
        return json.load(source)


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


def serialize_content_block(block: Any) -> dict[str, Any]:
    block_type = getattr(block, "type", None)

    serialized: dict[str, Any] = {
        "type": block_type,
    }

    if block_type == "text":
        serialized["text"] = getattr(block, "text", "")

    elif block_type == "thinking":
        # Do not retain large provider signatures.
        serialized["thinking_text"] = getattr(
            block,
            "thinking",
            "",
        )
        serialized["signature_present"] = bool(
            getattr(block, "signature", None)
        )

    else:
        # Preserve a compact representation of unexpected block types.
        serialized["representation"] = repr(block)

    return serialized


def extract_visible_text(content: list[Any]) -> str:
    return "".join(
        getattr(block, "text", "")
        for block in content
        if getattr(block, "type", None) == "text"
    )


def parse_result(
    batch_result: Any,
    requested_model: str,
) -> dict[str, Any]:
    custom_id = batch_result.custom_id
    result = batch_result.result
    result_type = getattr(result, "type", None)

    parsed: dict[str, Any] = {
        "custom_id": custom_id,
        "result_type": result_type,
        "requested_model": requested_model,
    }

    if result_type == "succeeded":
        message = result.message
        content_blocks = [
            serialize_content_block(block)
            for block in message.content
        ]

        parsed.update(
            {
                "status": "success",
                "returned_model": message.model,
                "raw_response": extract_visible_text(
                    message.content
                ),
                "content_blocks": content_blocks,
                "stop_reason": message.stop_reason,
                "stop_sequence": message.stop_sequence,
                "input_tokens": message.usage.input_tokens,
                "output_tokens": message.usage.output_tokens,
            }
        )

    elif result_type == "errored":
        error = result.error

        parsed.update(
            {
                "status": "error",
                "error_type": getattr(error, "type", None),
                "error_message": getattr(
                    error,
                    "message",
                    str(error),
                ),
            }
        )

    elif result_type == "expired":
        parsed.update(
            {
                "status": "expired",
            }
        )

    elif result_type == "canceled":
        parsed.update(
            {
                "status": "canceled",
            }
        )

    else:
        parsed.update(
            {
                "status": "unknown",
                "result_representation": repr(result),
            }
        )

    return parsed


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Retrieve one completed Anthropic CE-Agree batch."
        )
    )

    parser.add_argument(
        "--model",
        required=True,
        choices=MODEL_KEYS,
    )

    parser.add_argument(
        "--order",
        required=True,
        choices=("A", "B"),
    )

    args = parser.parse_args()

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print(
            "ERROR: ANTHROPIC_API_KEY is not set.",
            file=sys.stderr,
        )
        sys.exit(1)

    run_name = (
        f"anthropic_{args.model}_order{args.order}_v1"
    )

    metadata_path = Path(
        "results/batch_manifests/anthropic"
    ) / f"{run_name}.json"

    raw_output_path = Path(
        "results/raw_batches/anthropic"
    ) / f"{run_name}.jsonl"

    parsed_output_path = Path(
        "results/parsed_batches/anthropic"
    ) / f"{run_name}.jsonl"

    metadata = load_json(metadata_path)

    if not metadata.get("submitted"):
        raise ValueError(
            f"{metadata_path} does not describe a submitted batch."
        )

    batch_id = metadata.get("batch_id")
    if not batch_id:
        raise ValueError(
            f"No batch_id found in {metadata_path}"
        )

    client = anthropic.Anthropic()

    batch = client.messages.batches.retrieve(batch_id)

    print(f"Batch ID: {batch.id}")
    print(f"Processing status: {batch.processing_status}")
    print(f"Request counts: {batch.request_counts}")

    metadata["last_checked_at_utc"] = datetime.now(
        timezone.utc
    ).isoformat()
    metadata["batch_processing_status"] = (
        batch.processing_status
    )

    if batch.processing_status != "ended":
        write_json(metadata_path, metadata)

        print(
            "\nBatch is not complete. No result files were written."
        )
        return

    raw_rows: list[dict[str, Any]] = []
    parsed_rows: list[dict[str, Any]] = []

    for batch_result in client.messages.batches.results(
        batch_id
    ):
        raw_rows.append(
            batch_result.model_dump(
                mode="json",
            )
        )

        parsed_rows.append(
            parse_result(
                batch_result=batch_result,
                requested_model=metadata[
                    "requested_model"
                ],
            )
        )

    if len(raw_rows) != metadata["number_of_requests"]:
        raise ValueError(
            "Retrieved result count does not match the "
            f"manifest: {len(raw_rows)} versus "
            f"{metadata['number_of_requests']}"
        )

    write_jsonl(raw_output_path, raw_rows)
    write_jsonl(parsed_output_path, parsed_rows)

    metadata.update(
        {
            "retrieved_at_utc": datetime.now(
                timezone.utc
            ).isoformat(),
            "raw_output_file": str(raw_output_path),
            "parsed_output_file": str(
                parsed_output_path
            ),
            "retrieved_result_count": len(raw_rows),
        }
    )

    write_json(metadata_path, metadata)

    print(f"\nSaved {len(raw_rows)} raw results to:")
    print(f"  {raw_output_path}")

    print("Saved parsed results to:")
    print(f"  {parsed_output_path}")

    print("Updated metadata:")
    print(f"  {metadata_path}")


if __name__ == "__main__":
    main()
