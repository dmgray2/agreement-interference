#!/usr/bin/env python3

import json
import os
import sys
from pathlib import Path
from typing import Any

import anthropic


MODEL_ID = "claude-opus-4-1-20250805"
MAX_TOKENS = 100

TEST_ITEMS = {
    "A": {
        "path": Path(
            "data/stimuli/"
            "minpairs.ce2.v2.forced_choice.orderA.v1.jsonl"
        ),
        "prompt_id": "ce2fam_01008_V2_order1",
    },
    "B": {
        "path": Path(
            "data/stimuli/"
            "minpairs.ce2.v2.forced_choice.orderB.v1.jsonl"
        ),
        "prompt_id": "ce2fam_00351_V2_order2",
    },
}

STRICT_INSTRUCTION = "Answer only A or B."


def load_item(
    path: Path,
    prompt_id: str,
) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as source:
        for line in source:
            if not line.strip():
                continue

            row = json.loads(line)

            if row["prompt_id"] == prompt_id:
                return row

    raise ValueError(
        f"Could not find {prompt_id} in {path}"
    )


def revise_prompt(prompt: str) -> str:
    text = prompt.rstrip()

    if text.endswith("Answer:"):
        text = text[: -len("Answer:")].rstrip()

    return (
        f"{text}\n\n"
        f"{STRICT_INSTRUCTION}\n\n"
        "Answer:"
    )


def visible_text(message: Any) -> str:
    return "".join(
        block.text
        for block in message.content
        if getattr(block, "type", None) == "text"
    )


def main() -> None:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print(
            "ERROR: ANTHROPIC_API_KEY is not set.",
            file=sys.stderr,
        )
        sys.exit(1)

    client = anthropic.Anthropic()

    for expected_label, item_info in TEST_ITEMS.items():
        row = load_item(
            path=item_info["path"],
            prompt_id=item_info["prompt_id"],
        )

        prompt = revise_prompt(row["prompt"])

        print("\n" + "=" * 70)
        print(f"Prompt ID: {row['prompt_id']}")
        print(f"Correct label: {expected_label}")
        print(f"Instruction: {STRICT_INSTRUCTION}")
        print("=" * 70)

        message = client.messages.create(
            model=MODEL_ID,
            max_tokens=MAX_TOKENS,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
        )

        response = visible_text(message)

        print(f"Response: {response!r}")
        print(f"Stop reason: {message.stop_reason}")
        print(f"Input tokens: {message.usage.input_tokens}")
        print(f"Output tokens: {message.usage.output_tokens}")


if __name__ == "__main__":
    main()
