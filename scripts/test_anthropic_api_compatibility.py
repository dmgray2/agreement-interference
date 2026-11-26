#!/usr/bin/env python3

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import anthropic


PROMPT_PATH = Path(
    "data/stimuli/api_compatibility_prompts.v1.jsonl"
)

OUTPUT_PATH = Path(
    "results/api_compatibility/anthropic_compatibility_results.v1.jsonl"
)

MODELS = [
    "claude-opus-4-1-20250805",
    "claude-opus-4-5-20251101",
    "claude-opus-5",
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


def extract_text(message: Any) -> str:
    text_blocks = [
        block.text
        for block in message.content
        if getattr(block, "type", None) == "text"
    ]
    return "".join(text_blocks)


def main() -> None:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print(
            "ERROR: ANTHROPIC_API_KEY is not set.",
            file=sys.stderr,
        )
        sys.exit(1)

    prompts = load_prompts(PROMPT_PATH)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    client = anthropic.Anthropic()

    results = []

    for model in MODELS:
        for prompt_row in prompts:
            print(
                f"Testing {model} on {prompt_row['prompt_id']}...",
                flush=True,
            )

            started_at = datetime.now(timezone.utc).isoformat()

            result: dict[str, Any] = {
                "provider": "anthropic",
                "requested_model": model,
                "prompt_id": prompt_row["prompt_id"],
                "correct_label": prompt_row["correct_label"],
                "started_at_utc": started_at,
            }

            try:
                message = client.messages.create(
                    model=model,
                    max_tokens=100,
                    messages=[
                        {
                            "role": "user",
                            "content": prompt_row["prompt"],
                        }
                    ],
                )

                raw_text = extract_text(message)
                print(f"  Content blocks: {message.content!r}")

                result.update(
                    {
                        "status": "success",
                        "returned_model": message.model,
                        "raw_response": raw_text,
                        "stop_reason": message.stop_reason,
                        "input_tokens": message.usage.input_tokens,
                        "output_tokens": message.usage.output_tokens,
                        "completed_at_utc": datetime.now(
                            timezone.utc
                        ).isoformat(),
                    }
                )

                print(
                    f"  Response: {raw_text!r} "
                    f"(stop_reason={message.stop_reason})"
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
