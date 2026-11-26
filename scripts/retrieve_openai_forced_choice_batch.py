#!/usr/bin/env python3

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from openai import OpenAI


MODEL_KEYS = (
    "gpt-3.5",
    "gpt-4.1",
    "gpt-5",
    "gpt-5.6-sol",
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


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as output:
        output.write(text)


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


def read_jsonl_text(text: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    for line_number, line in enumerate(
        text.splitlines(),
        start=1,
    ):
        if not line.strip():
            continue

        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Invalid JSON on downloaded line {line_number}"
            ) from exc

    return rows


def extract_visible_text(response_body: dict[str, Any]) -> str:
    """
    Extract visible output text from a Responses API response body.
    """
    text_parts: list[str] = []

    for output_item in response_body.get("output", []):
        if output_item.get("type") != "message":
            continue

        for content_item in output_item.get("content", []):
            if content_item.get("type") == "output_text":
                text_parts.append(
                    content_item.get("text", "")
                )

    return "".join(text_parts)

def extract_tool_answer(response_body: dict[str, Any]) -> str | None:
    """
    Extract the A/B answer from the forced record_answer tool call.
    """
    for output_item in response_body.get("output", []):
        if output_item.get("type") != "function_call":
            continue

        if output_item.get("name") != "record_answer":
            continue

        arguments = output_item.get("arguments")

        if not arguments:
            return None

        try:
            parsed = json.loads(arguments)
        except json.JSONDecodeError:
            return None

        answer = parsed.get("answer")

        if answer in {"A", "B"}:
            return answer

        return None

    return None

def compact_output_items(
    response_body: dict[str, Any],
) -> list[dict[str, Any]]:
    """
    Retain compact information about output item and content types.
    """
    compact: list[dict[str, Any]] = []

    for output_item in response_body.get("output", []):
        item: dict[str, Any] = {
            "type": output_item.get("type"),
        }

        if output_item.get("type") == "message":
            item["role"] = output_item.get("role")
            item["status"] = output_item.get("status")
            item["content"] = [
                {
                    "type": content_item.get("type"),
                    "text": (
                        content_item.get("text")
                        if content_item.get("type")
                        == "output_text"
                        else None
                    ),
                }
                for content_item in output_item.get(
                    "content",
                    [],
                )
            ]

        elif output_item.get("type") == "reasoning":
            item["summary_count"] = len(
                output_item.get("summary", [])
            )

        compact.append(item)

    return compact


def parse_success_row(
    row: dict[str, Any],
    requested_model: str,
) -> dict[str, Any]:
    custom_id = row.get("custom_id")
    response_wrapper = row.get("response") or {}
    status_code = response_wrapper.get("status_code")
    response_body = response_wrapper.get("body") or {}

    usage = response_body.get("usage") or {}
    output_details = (
        usage.get("output_tokens_details") or {}
    )

    incomplete_details = (
        response_body.get("incomplete_details") or {}
    )

    return {
        "custom_id": custom_id,
        "status": "success",
        "http_status_code": status_code,
        "requested_model": requested_model,
        "returned_model": response_body.get("model"),
        "response_id": response_body.get("id"),
        "response_status": response_body.get("status"),
        "raw_response": extract_visible_text(
            response_body
        ),
        "tool_answer": extract_tool_answer(
            response_body
        ),
        "output_items": compact_output_items(
            response_body
        ),
        "incomplete_reason": incomplete_details.get(
            "reason"
        ),
        "input_tokens": usage.get("input_tokens"),
        "output_tokens": usage.get("output_tokens"),
        "reasoning_tokens": output_details.get(
            "reasoning_tokens"
        ),
        "total_tokens": usage.get("total_tokens"),
        "request_id": row.get("id"),
    }


def parse_error_row(
    row: dict[str, Any],
    requested_model: str,
) -> dict[str, Any]:
    error = row.get("error") or {}

    return {
        "custom_id": row.get("custom_id"),
        "status": "error",
        "requested_model": requested_model,
        "error_code": error.get("code"),
        "error_message": error.get("message"),
        "error_param": error.get("param"),
        "request_id": row.get("id"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Retrieve one completed OpenAI CE-Agree batch."
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

    if not os.environ.get("OPENAI_API_KEY"):
        print(
            "ERROR: OPENAI_API_KEY is not set.",
            file=sys.stderr,
        )
        sys.exit(1)

    run_name = (
        f"openai_{args.model}_order{args.order}_v1"
    )

    metadata_path = Path(
        "results/batch_manifests/openai"
    ) / f"{run_name}.json"

    raw_output_path = Path(
        "results/raw_batches/openai"
    ) / f"{run_name}.jsonl"

    raw_error_path = Path(
        "results/raw_batches/openai"
    ) / f"{run_name}.errors.jsonl"

    parsed_output_path = Path(
        "results/parsed_batches/openai"
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

    client = OpenAI()

    batch = client.batches.retrieve(batch_id)

    print(f"Batch ID: {batch.id}")
    print(f"Batch status: {batch.status}")
    print(f"Request counts: {batch.request_counts}")

    metadata["last_checked_at_utc"] = datetime.now(
        timezone.utc
    ).isoformat()
    metadata["batch_status"] = batch.status
    metadata["output_file_id"] = batch.output_file_id
    metadata["error_file_id"] = batch.error_file_id

    if batch.status != "completed":
        write_json(metadata_path, metadata)

        print(
            "\nBatch is not complete. No result files were written."
        )
        return

    success_rows: list[dict[str, Any]] = []
    error_rows: list[dict[str, Any]] = []

    if batch.output_file_id:
        output_response = client.files.content(
            batch.output_file_id
        )
        output_text = output_response.text

        write_text(raw_output_path, output_text)
        success_rows = read_jsonl_text(output_text)

    if batch.error_file_id:
        error_response = client.files.content(
            batch.error_file_id
        )
        error_text = error_response.text

        write_text(raw_error_path, error_text)
        error_rows = read_jsonl_text(error_text)

    parsed_rows = [
        parse_success_row(
            row=row,
            requested_model=metadata["requested_model"],
        )
        for row in success_rows
    ]

    parsed_rows.extend(
        parse_error_row(
            row=row,
            requested_model=metadata["requested_model"],
        )
        for row in error_rows
    )

    expected_count = metadata["number_of_requests"]

    if len(parsed_rows) != expected_count:
        raise ValueError(
            "Retrieved result count does not match the "
            f"manifest: {len(parsed_rows)} versus "
            f"{expected_count}"
        )

    custom_ids = [
        row.get("custom_id")
        for row in parsed_rows
    ]

    if len(custom_ids) != len(set(custom_ids)):
        raise ValueError(
            "Duplicate custom_id values found in retrieved results."
        )

    write_jsonl(parsed_output_path, parsed_rows)

    metadata.update(
        {
            "retrieved_at_utc": datetime.now(
                timezone.utc
            ).isoformat(),
            "raw_output_file": (
                str(raw_output_path)
                if success_rows
                else None
            ),
            "raw_error_file": (
                str(raw_error_path)
                if error_rows
                else None
            ),
            "parsed_output_file": str(
                parsed_output_path
            ),
            "retrieved_success_count": len(
                success_rows
            ),
            "retrieved_error_count": len(
                error_rows
            ),
            "retrieved_result_count": len(
                parsed_rows
            ),
        }
    )

    write_json(metadata_path, metadata)

    print(
        f"\nRetrieved {len(success_rows)} successful results "
        f"and {len(error_rows)} errors."
    )

    if success_rows:
        print(f"Raw successful output: {raw_output_path}")

    if error_rows:
        print(f"Raw error output: {raw_error_path}")

    print(f"Parsed output: {parsed_output_path}")
    print(f"Updated metadata: {metadata_path}")


if __name__ == "__main__":
    main()
