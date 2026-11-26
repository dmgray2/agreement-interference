#!/usr/bin/env python3

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import anthropic


MODEL_ID = "claude-opus-4-5-20251101"
MAX_TOKENS = 100

LOW_CREDIT_MESSAGE = (
    "Your credit balance is too low to access the Anthropic API."
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


def extract_error_details(
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
            "Prepare or submit the documented Opus 4.5 "
            "credit-balance completion batch."
        )
    )

    parser.add_argument(
        "--order",
        required=True,
        choices=("A", "B"),
        help="Counterbalanced answer order.",
    )

    parser.add_argument(
        "--submit",
        action="store_true",
        help=(
            "Actually submit the completion batch. "
            "Without this flag, perform a dry run only."
        ),
    )

    args = parser.parse_args()
    order = args.order

    base_name = f"anthropic_opus-4.5_order{order}_v1"

    original_manifest_path = Path(
        "data/batch_requests/anthropic"
    ) / f"{base_name}.jsonl"

    original_results_path = Path(
        "results/raw_batches/anthropic"
    ) / f"{base_name}.jsonl"

    run_name = (
        f"anthropic_opus-4.5_order{order}_"
        "credit_completion_v1"
    )

    completion_manifest_path = Path(
        "data/batch_requests/anthropic"
    ) / f"{run_name}.jsonl"

    metadata_path = Path(
        "results/batch_manifests/anthropic"
    ) / f"{run_name}.json"

    if not original_manifest_path.exists():
        raise FileNotFoundError(
            f"Original manifest not found: "
            f"{original_manifest_path}"
        )

    if not original_results_path.exists():
        raise FileNotFoundError(
            f"Original results not found: "
            f"{original_results_path}"
        )

    original_requests = load_jsonl(
        original_manifest_path
    )

    original_results = load_jsonl(
        original_results_path
    )

    request_by_id = {
        request["custom_id"]: request
        for request in original_requests
    }

    if len(request_by_id) != len(original_requests):
        raise ValueError(
            "Duplicate custom_id values found "
            "in the original request manifest"
        )

    credit_failure_ids: list[str] = []
    unexpected_failures: list[
        tuple[str, str | None, str | None]
    ] = []

    for record in original_results:
        result = record.get("result", {})
        result_type = result.get("type")

        if result_type == "succeeded":
            continue

        custom_id = record.get("custom_id", "")
        error_type, message = extract_error_details(
            record
        )

        if (
            result_type == "errored"
            and error_type == "invalid_request_error"
            and message is not None
            and LOW_CREDIT_MESSAGE in message
        ):
            credit_failure_ids.append(custom_id)
        else:
            unexpected_failures.append(
                (
                    custom_id,
                    error_type,
                    message,
                )
            )

    if unexpected_failures:
        print(
            "ERROR: Unexpected unsuccessful results were found:",
            file=sys.stderr,
        )

        for custom_id, error_type, message in (
            unexpected_failures
        ):
            print(
                f"  {custom_id}: "
                f"{error_type}: {message}",
                file=sys.stderr,
            )

        sys.exit(1)

    if not credit_failure_ids:
        print(
            "No insufficient-credit failures were found."
        )
        return

    completion_requests: list[
        dict[str, Any]
    ] = []

    for custom_id in credit_failure_ids:
        if custom_id not in request_by_id:
            raise ValueError(
                f"Failed request ID not found in "
                f"original manifest: {custom_id}"
            )

        request = request_by_id[custom_id]

        params = request.get("params", {})

        if params.get("model") != MODEL_ID:
            raise ValueError(
                f"Unexpected model for {custom_id}: "
                f"{params.get('model')}"
            )

        if params.get("max_tokens") != MAX_TOKENS:
            raise ValueError(
                f"Unexpected max_tokens for {custom_id}: "
                f"{params.get('max_tokens')}"
            )

        completion_requests.append(request)

    write_jsonl(
        completion_manifest_path,
        completion_requests,
    )

    metadata: dict[str, Any] = {
        "provider": "anthropic",
        "run_name": run_name,
        "requested_model": MODEL_ID,
        "answer_order": order,
        "protocol_status": "documented_deviation",
        "deviation_type": (
            "completion_of_requests_rejected_"
            "before_inference_for_insufficient_credit"
        ),
        "deviation_reason": (
            "The original requests were rejected before "
            "model inference because the Anthropic prepaid "
            "credit balance was exhausted. They are being "
            "resubmitted after credits were replenished with "
            "the original model, prompt, max_tokens, and all "
            "other inference settings unchanged."
        ),
        "original_results_file": str(
            original_results_path
        ),
        "original_request_manifest_file": str(
            original_manifest_path
        ),
        "completion_request_manifest_file": str(
            completion_manifest_path
        ),
        "number_of_requests": len(
            completion_requests
        ),
        "max_tokens": MAX_TOKENS,
        "temperature_parameter": "omitted",
        "thinking_parameter": "omitted",
        "tools": "none",
        "original_errors_preserved": True,
        "created_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "submitted": False,
        "batch_id": None,
    }

    print(f"Model: {MODEL_ID}")
    print(f"Answer order: {order}")
    print(
        "Credit-rejected requests: "
        f"{len(completion_requests)}"
    )
    print(f"Maximum output tokens: {MAX_TOKENS}")
    print("Protocol status: documented deviation")
    print(
        f"Source results: {original_results_path}"
    )
    print(
        "Completion manifest: "
        f"{completion_manifest_path}"
    )
    print(f"Metadata file: {metadata_path}")

    print("\nCompletion request IDs:")

    for request in completion_requests:
        print(f"  {request['custom_id']}")

    if not args.submit:
        write_json(metadata_path, metadata)

        print(
            "\nDRY RUN ONLY: no completion requests "
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
                "ERROR: This completion batch has already "
                "been submitted according to its metadata.",
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

    print("\nSubmitting completion batch...")

    batch = client.messages.batches.create(
        requests=completion_requests
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
