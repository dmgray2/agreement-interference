#!/usr/bin/env python3

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from openai import OpenAI


PROMPT_PATH = Path(
    os.environ.get(
        "OPENAI_COMPAT_PROMPT_PATH",
        "data/stimuli/api_compatibility_prompts.v1.jsonl",
    )
)

OUTPUT_PATH = Path(
    os.environ.get(
        "OPENAI_COMPAT_OUTPUT_PATH",
        "results/api_compatibility/openai_compatibility_results.v1.jsonl",
    )
)

MODELS = [
    "gpt-3.5-turbo-0125",
    "gpt-4.1-2025-04-14",
    "gpt-5-2025-08-07",
    "gpt-5.6-sol",
]


def load_prompts(path: Path) -> list[dict[str, Any]]:
    prompts = []

    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            if not line.strip():
                continue

            try:
                prompts.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON on line {line_number} of {path}"
                ) from exc

    if len(prompts) != 2:
        raise ValueError(
            f"Expected 2 compatibility prompts, found {len(prompts)}"
        )

    return prompts


def get_usage_value(obj: Any, field: str) -> Any:
    if obj is None:
        return None
    return getattr(obj, field, None)


def main() -> None:
    if not os.environ.get("OPENAI_API_KEY"):
        print(
            "ERROR: OPENAI_API_KEY is not set.",
            file=sys.stderr,
        )
        sys.exit(1)

    prompts = load_prompts(PROMPT_PATH)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    client = OpenAI()
    results = []

    for model in MODELS:
        for prompt_row in prompts:
            print(
                f"Testing {model} on {prompt_row['prompt_id']}...",
                flush=True,
            )

            started_at = datetime.now(timezone.utc).isoformat()

            result: dict[str, Any] = {
                "provider": "openai",
                "requested_model": model,
                "prompt_id": prompt_row["prompt_id"],
                "correct_label": prompt_row["correct_label"],
                "started_at_utc": started_at,
            }

            try:
                response = client.responses.create(
                    model=model,
                    input=prompt_row["prompt"],
                    max_output_tokens=250,
                    store=False,
                )

                usage = response.usage
                output_details = (
                    getattr(usage, "output_tokens_details", None)
                    if usage is not None
                    else None
                )

                result.update(
                    {
                        "status": "success",
                        "response_id": response.id,
                        "returned_model": response.model,
                        "response_status": response.status,
                        "raw_response": response.output_text,
                        "incomplete_reason": (
                            getattr(
                                response.incomplete_details,
                                "reason",
                                None,
                            )
                            if response.incomplete_details is not None
                            else None
                        ),
                        "input_tokens": get_usage_value(
                            usage,
                            "input_tokens",
                        ),
                        "output_tokens": get_usage_value(
                            usage,
                            "output_tokens",
                        ),
                        "reasoning_tokens": get_usage_value(
                            output_details,
                            "reasoning_tokens",
                        ),
                        "completed_at_utc": datetime.now(
                            timezone.utc
                        ).isoformat(),
                    }
                )

                print(
                    f"  Response: {response.output_text!r} "
                    f"(status={response.status}, "
                    f"output_tokens={result['output_tokens']}, "
                    f"reasoning_tokens={result['reasoning_tokens']})"
                )

            except Exception as exc:
                result.update(
                    {
                        "status": "error",
                        "error_type": type(exc).__name__,
                        "error_message": str(exc),
                        "completed_at_utc": datetime.now(
                            timezone.utc
                        ).isoformat(),
                    }
                )

                print(
                    f"  ERROR: {type(exc).__name__}: {exc}",
                    file=sys.stderr,
                )

            results.append(result)

    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

    print(f"\nSaved {len(results)} results to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
