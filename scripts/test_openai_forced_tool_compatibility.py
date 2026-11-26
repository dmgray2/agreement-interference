#!/usr/bin/env python3

import json
import os
from pathlib import Path

from openai import OpenAI


PROMPT_PATH = Path(
    "data/stimuli/api_compatibility_prompts.ce2.v1.jsonl"
)

MODELS = [
    {
        "name": "gpt-4.1-2025-04-14",
        "reasoning": None,
    },
    {
        "name": "gpt-5-2025-08-07",
        "reasoning": {"effort": "minimal"},
    },
    {
        "name": "gpt-5.6-sol",
        "reasoning": {"effort": "none"},
    },
]


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


def load_prompts():
    rows = []
    with PROMPT_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def extract_tool_answer(response):
    for item in response.output:
        if item.type == "function_call" and item.name == "record_answer":
            args = json.loads(item.arguments)
            return args.get("answer")
    return None


def main():
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set.")

    client = OpenAI()
    prompts = load_prompts()

    for model_info in MODELS:
        model = model_info["name"]

        for row in prompts:
            print(f"\nTesting {model} on {row['prompt_id']}...")

            kwargs = {
                "model": model,
                "input": row["prompt"],
                "tools": [ANSWER_TOOL],
                "tool_choice": {
                    "type": "function",
                    "name": "record_answer",
                },
                "max_output_tokens": 250,
                "store": False,
            }

            if model_info["reasoning"] is not None:
                kwargs["reasoning"] = model_info["reasoning"]

            try:
                response = client.responses.create(**kwargs)

                answer = extract_tool_answer(response)

                reasoning_tokens = (
                    response.usage.output_tokens_details.reasoning_tokens
                    if response.usage
                    and response.usage.output_tokens_details
                    else None
                )

                print(f"  Answer: {answer!r}")
                print(f"  Correct label: {row['correct_label']}")
                print(f"  Status: {response.status}")
                print(f"  Output tokens: {response.usage.output_tokens if response.usage else None}")
                print(f"  Reasoning tokens: {reasoning_tokens}")

            except Exception as e:
                print(f"  ERROR: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
