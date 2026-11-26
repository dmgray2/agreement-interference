#!/usr/bin/env python3

import argparse
import json
import os
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import anthropic


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as source:
        return json.load(source)


def write_json(path: Path, value: dict[str, Any]) -> None:
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


def sanitize_raw_result(
    batch_result: Any,
) -> dict[str, Any]:
    raw = batch_result.model_dump(mode="json")
    result = raw.get("result", {})

    if result.get("type") != "succeeded":
        return raw

    message = result.get("message", {})
    content = message.get("content", [])
    sanitized: list[dict[str, Any]] = []

    for block in content:
        block_type = block.get("type")

        if block_type == "tool_use":
            sanitized.append(
                {
                    "type": "tool_use",
                    "id": block.get("id"),
                    "name": block.get("name"),
                    "input": block.get("input"),
                }
            )

        elif block_type == "text":
            sanitized.append(
                {
                    "type": "text",
                    "text": block.get("text", ""),
                }
            )

        elif block_type == "thinking":
            sanitized.append(
                {
                    "type": "thinking",
                    "signature_present": bool(
                        block.get("signature")
                    ),
                }
            )

        else:
            sanitized.append(
                {
                    "type": block_type,
                }
            )

    message["content"] = sanitized
    return raw


def extract_error(
    result: Any,
) -> tuple[str | None, str | None, str | None]:
    wrapper = getattr(result, "error", None)

    if wrapper is None:
        return None, None, None

    inner = getattr(wrapper, "error", None)

    if inner is None:
        inner = wrapper

    return (
        getattr(inner, "type", None),
        getattr(inner, "message", str(inner)),
        getattr(wrapper, "request_id", None),
    )


def parse_result(
    batch_result: Any,
    requested_model: str,
) -> dict[str, Any]:
    result = batch_result.result
    result_type = getattr(result, "type", None)

    parsed: dict[str, Any] = {
        "custom_id": batch_result.custom_id,
        "result_type": result_type,
        "requested_model": requested_model,
    }

    if result_type == "succeeded":
        message = result.message
        tool_blocks = [
            block
            for block in message.content
            if getattr(block, "type", None) == "tool_use"
            and getattr(block, "name", None)
            == "record_answer"
        ]

        answer = None
        tool_error = None

        if len(tool_blocks) != 1:
            tool_error = (
                f"Expected exactly one record_answer "
                f"tool call, found {len(tool_blocks)}"
            )
        else:
            tool_input = getattr(
                tool_blocks[0],
                "input",
                None,
            )

            if isinstance(tool_input, dict):
                candidate = tool_input.get("answer")

                if candidate in ("A", "B"):
                    answer = candidate
                else:
                    tool_error = (
                        f"Invalid answer value: "
                        f"{candidate!r}"
                    )
            else:
                tool_error = (
                    "Tool input was not an object"
                )

        text_blocks = [
            getattr(block, "text", "")
            for block in message.content
            if getattr(block, "type", None) == "text"
        ]

        parsed.update(
            {
                "status": (
                    "success"
                    if answer is not None
                    else "malformed"
                ),
                "returned_model": message.model,
                "answer": answer,
                "tool_error": tool_error,
                "visible_text": "".join(text_blocks),
                "stop_reason": message.stop_reason,
                "stop_sequence": message.stop_sequence,
                "input_tokens": (
                    message.usage.input_tokens
                ),
                "output_tokens": (
                    message.usage.output_tokens
                ),
            }
        )

    elif result_type == "errored":
        error_type, error_message, request_id = (
            extract_error(result)
        )

        parsed.update(
            {
                "status": "error",
                "error_type": error_type,
                "error_message": error_message,
                "request_id": request_id,
            }
        )

    elif result_type == "expired":
        parsed["status"] = "expired"

    elif result_type == "canceled":
        parsed["status"] = "canceled"

    else:
        parsed["status"] = "unknown"

    return parsed


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Retrieve an Anthropic forced-tool batch and "
            "extract the enum-constrained A/B answer."
        )
    )

    parser.add_argument(
        "--run-name",
        required=True,
        help=(
            "Exact metadata filename stem without .json."
        ),
    )

    args = parser.parse_args()
    run_name = args.run_name

    if "/" in run_name or "\\" in run_name:
        raise ValueError(
            "--run-name must be a filename stem"
        )

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print(
            "ERROR: ANTHROPIC_API_KEY is not set.",
            file=sys.stderr,
        )
        sys.exit(1)

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

    batch_id = metadata.get("batch_id")

    if not batch_id:
        raise ValueError(
            f"No batch_id found in {metadata_path}"
        )

    expected_count = metadata.get(
        "number_of_requests"
    )

    requested_model = metadata.get(
        "requested_model",
        "unknown",
    )

    client = anthropic.Anthropic()
    batch = client.messages.batches.retrieve(
        batch_id
    )

    print(f"Run name: {run_name}")
    print(f"Batch ID: {batch.id}")
    print(f"Status: {batch.processing_status}")
    print(f"Counts: {batch.request_counts}")

    if batch.processing_status != "ended":
        print(
            "\nBatch is not complete. "
            "No result files were written."
        )
        return

    raw_rows: list[dict[str, Any]] = []
    parsed_rows: list[dict[str, Any]] = []

    for batch_result in (
        client.messages.batches.results(batch_id)
    ):
        raw_rows.append(
            sanitize_raw_result(batch_result)
        )

        parsed_rows.append(
            parse_result(
                batch_result,
                requested_model,
            )
        )

    if len(parsed_rows) != expected_count:
        raise ValueError(
            f"Retrieved {len(parsed_rows)} rows; "
            f"expected {expected_count}"
        )

    custom_ids = [
        row["custom_id"]
        for row in parsed_rows
    ]

    if len(custom_ids) != len(set(custom_ids)):
        raise ValueError(
            "Duplicate custom_id values found"
        )

    write_jsonl(raw_output_path, raw_rows)
    write_jsonl(parsed_output_path, parsed_rows)

    statuses = Counter(
        row["status"]
        for row in parsed_rows
    )

    answers = Counter(
        row.get("answer")
        for row in parsed_rows
        if row.get("answer") in ("A", "B")
    )

    metadata.update(
        {
            "retrieved_at_utc": datetime.now(
                timezone.utc
            ).isoformat(),
            "raw_output_file": str(
                raw_output_path
            ),
            "parsed_output_file": str(
                parsed_output_path
            ),
            "retrieved_result_count": len(
                parsed_rows
            ),
            "retrieved_status_counts": dict(
                statuses
            ),
            "retrieved_answer_counts": dict(
                answers
            ),
            "hidden_thinking_text_retained": False,
            "thinking_signatures_retained": False,
            "retrieval_script": (
                "scripts/"
                "retrieve_anthropic_toolforced_batch.py"
            ),
        }
    )

    write_json(metadata_path, metadata)

    print(f"\nSaved raw results to:")
    print(f"  {raw_output_path}")

    print("Saved parsed results to:")
    print(f"  {parsed_output_path}")

    print(f"Statuses: {dict(statuses)}")
    print(f"Answers: {dict(answers)}")
    print(f"Updated metadata: {metadata_path}")


if __name__ == "__main__":
    main()
