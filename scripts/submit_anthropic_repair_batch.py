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


MODELS = {
    "opus-4.1": "claude-opus-4-1-20250805",
    "opus-4.5": "claude-opus-4-5-20251101",
    "opus-5": "claude-opus-5",
}

# These correspond to the technical failures allowed for retry
# under the preregistered procedure.
RETRYABLE_ERROR_TYPES = {
    "api_error",
    "overloaded_error",
    "rate_limit_error",
    "timeout_error",
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


def get_error_details(
    record: dict[str, Any],
) -> tuple[str | None, str | None]:
    result = record.get("result", {})

    if result.get("type") != "errored":
        return None, None

    outer_error = result.get("error", {})
    inner_error = outer_error.get("error", {})

    return (
        inner_error.get("type"),
        inner_error.get("message"),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare or submit a repair batch containing only "
            "retry-eligible Anthropic technical failures."
        )
    )

    parser.add_argument(
        "--model",
        required=True,
        choices=sorted(MODELS),
        help="Preregistered Anthropic model.",
    )

    parser.add_argument(
        "--order",
        required=True,
        choices=("A", "B"),
        help="Counterbalanced answer order.",
    )

    parser.add_argument(
        "--attempt",
        required=True,
        type=int,
        choices=(1, 2),
        help=(
            "Repair-attempt number. Attempt 1 follows the "
            "original request; attempt 2 follows repair 1."
        ),
    )

    parser.add_argument(
        "--submit",
        action="store_true",
        help=(
            "Actually submit the repair batch. Without this "
            "flag, the script performs a dry run only."
        ),
    )

    args = parser.parse_args()

    model_key = args.model
    model_id = MODELS[model_key]
    order = args.order
    attempt = args.attempt

    base_name = f"anthropic_{model_key}_order{order}"

    original_manifest_path = Path(
        "data/batch_requests/anthropic"
    ) / f"{base_name}_v1.jsonl"

    if attempt == 1:
        source_results_path = Path(
            "results/raw_batches/anthropic"
        ) / f"{base_name}_v1.jsonl"
    else:
        source_results_path = Path(
            "results/raw_batches/anthropic"
        ) / f"{base_name}_repair1_v1.jsonl"

    repair_name = f"{base_name}_repair{attempt}_v1"

    repair_manifest_path = Path(
        "data/batch_requests/anthropic"
    ) / f"{repair_name}.jsonl"

    metadata_path = Path(
        "results/batch_manifests/anthropic"
    ) / f"{repair_name}.json"

    if not original_manifest_path.exists():
        raise FileNotFoundError(
            f"Original request manifest not found: "
            f"{original_manifest_path}"
        )

    if not source_results_path.exists():
        raise FileNotFoundError(
            f"Source result file not found: "
            f"{source_results_path}"
        )

    original_requests = load_jsonl(original_manifest_path)
    source_results = load_jsonl(source_results_path)

    request_ids = [
        request["custom_id"]
        for request in original_requests
    ]

    if len(request_ids) != len(set(request_ids)):
        raise ValueError(
            "Duplicate custom_id values in original manifest"
        )

    failed_ids: set[str] = set()
    error_details: dict[str, tuple[str, str]] = {}
    nonretryable_failures: list[tuple[str, str, str]] = []

    for record in source_results:
        result_type = record.get("result", {}).get("type")

        if result_type == "succeeded":
            continue

        custom_id = record.get("custom_id", "")
        error_type, message = get_error_details(record)

        if (
            result_type == "errored"
            and error_type in RETRYABLE_ERROR_TYPES
        ):
            failed_ids.add(custom_id)
            error_details[custom_id] = (
                error_type or "unknown",
                message or "",
            )
        else:
            nonretryable_failures.append(
                (
                    custom_id,
                    str(result_type),
                    str(error_type),
                )
            )

    if nonretryable_failures:
        print(
            "ERROR: The source file contains unsuccessful "
            "results that are not automatically retryable:",
            file=sys.stderr,
        )

        for custom_id, result_type, error_type in (
            nonretryable_failures
        ):
            print(
                f"  {custom_id}: result_type={result_type}, "
                f"error_type={error_type}",
                file=sys.stderr,
            )

        sys.exit(1)

    if not failed_ids:
        print(
            "No retry-eligible technical failures were found."
        )
        return

    repair_requests = [
        request
        for request in original_requests
        if request["custom_id"] in failed_ids
    ]

    selected_ids = {
        request["custom_id"]
        for request in repair_requests
    }

    if selected_ids != failed_ids:
        missing_ids = sorted(failed_ids - selected_ids)
        raise ValueError(
            "Some failed IDs were absent from the original "
            f"manifest: {missing_ids}"
        )

    error_type_counts = Counter(
        error_type
        for error_type, _ in error_details.values()
    )

    write_jsonl(
        repair_manifest_path,
        repair_requests,
    )

    metadata: dict[str, Any] = {
        "provider": "anthropic",
        "run_name": repair_name,
        "requested_model": model_id,
        "answer_order": order,
        "repair_attempt": attempt,
        "source_result_file": str(source_results_path),
        "original_request_manifest_file": str(
            original_manifest_path
        ),
        "repair_request_manifest_file": str(
            repair_manifest_path
        ),
        "number_of_requests": len(repair_requests),
        "retryable_error_type_counts": dict(
            error_type_counts
        ),
        "max_tokens": 100,
        "temperature_parameter": "omitted",
        "thinking_parameter": "omitted",
        "tools": "none",
        "created_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "submitted": False,
        "batch_id": None,
    }

    print(f"Model: {model_id}")
    print(f"Answer order: {order}")
    print(f"Repair attempt: {attempt}")
    print(f"Source results: {source_results_path}")
    print(
        f"Retry-eligible requests: "
        f"{len(repair_requests)}"
    )
    print(
        "Error types: "
        f"{dict(error_type_counts)}"
    )
    print(f"Repair manifest: {repair_manifest_path}")
    print(f"Metadata file: {metadata_path}")

    print("\nRepair request IDs:")
    for request in repair_requests:
        custom_id = request["custom_id"]
        error_type, message = error_details[custom_id]
        print(
            f"  {custom_id}: "
            f"{error_type}: {message}"
        )

    if not args.submit:
        write_json(metadata_path, metadata)

        print(
            "\nDRY RUN ONLY: no repair requests "
            "were submitted."
        )
        return

    if metadata_path.exists():
        try:
            existing_metadata = json.loads(
                metadata_path.read_text(
                    encoding="utf-8"
                )
            )
        except json.JSONDecodeError:
            existing_metadata = {}

        if existing_metadata.get("submitted"):
            print(
                "ERROR: This repair batch has already been "
                "submitted according to its metadata.",
                file=sys.stderr,
            )
            sys.exit(1)

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print(
            "ERROR: ANTHROPIC_API_KEY is not set.",
            file=sys.stderr,
        )
        sys.exit(1)

    client = anthropic.Anthropic()

    print("\nSubmitting repair batch...")

    batch = client.messages.batches.create(
        requests=repair_requests
    )

    metadata.update(
        {
            "submitted": True,
            "batch_id": batch.id,
            "batch_processing_status": (
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
        "Processing status: "
        f"{batch.processing_status}"
    )
    print(f"Updated metadata: {metadata_path}")


if __name__ == "__main__":
    main()
