#!/usr/bin/env python3

import json
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer


MODELS = {
    "gpt2": "gpt2",
    "pythia-410m": "EleutherAI/pythia-410m",
    "llama-3.2-1b": "meta-llama/Llama-3.2-1B",
    "llama-3.2-3b": "meta-llama/Llama-3.2-3B",
    "llama-3.1-70b": "meta-llama/Llama-3.1-70B",
    "qwen2.5-1.5b": "Qwen/Qwen2.5-1.5B",
    "qwen2.5-3b": "Qwen/Qwen2.5-3B",
    "qwen2.5-7b": "Qwen/Qwen2.5-7B",
    "qwen2.5-14b": "Qwen/Qwen2.5-14B",
    "qwen2.5-32b": "Qwen/Qwen2.5-32B",
    "qwen2.5-72b": "Qwen/Qwen2.5-72B",
}

STIMULUS_FILE = Path(
    "data/stimuli/"
    "minpairs.ce2.v2.forced_choice.orderA.v1.jsonl"
)

OUTPUT_FILE = Path(
    "results/tokenization/"
    "open_weight_answer_token_audit.v1.json"
)

CANDIDATES = {
    "A": " A",
    "B": " B",
}


def load_prompts(path: Path) -> list[str]:
    prompts: list[str] = []

    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue

            row = json.loads(line)

            prompt = row.get("prompt")
            if not isinstance(prompt, str):
                raise ValueError(
                    f"Missing prompt on line {line_number}"
                )

            prompts.append(prompt)

    if len(prompts) != 2000:
        raise ValueError(
            f"Expected 2,000 prompts, found {len(prompts)}"
        )

    return prompts


def audit_candidate(
    tokenizer: Any,
    prompts: list[str],
    candidate: str,
) -> dict[str, Any]:
    tokenizations: dict[tuple[int, ...], int] = {}
    prefix_failures = 0

    for prompt in prompts:
        prompt_ids = tokenizer.encode(
            prompt,
            add_special_tokens=False,
        )

        complete_ids = tokenizer.encode(
            prompt + candidate,
            add_special_tokens=False,
        )

        if complete_ids[: len(prompt_ids)] != prompt_ids:
            prefix_failures += 1
            continue

        candidate_ids = complete_ids[len(prompt_ids):]

        key = tuple(candidate_ids)

        tokenizations[key] = tokenizations.get(key, 0) + 1

    return {
        "candidate_string": candidate,
        "prefix_failures": prefix_failures,
        "unique_tokenizations": [
            {
                "token_ids": list(token_ids),
                "tokens": tokenizer.convert_ids_to_tokens(
                    list(token_ids)
                ),
                "token_count": len(token_ids),
                "number_of_prompts": count,
            }
            for token_ids, count in sorted(
                tokenizations.items()
            )
        ],
    }


def main() -> None:
    prompts = load_prompts(STIMULUS_FILE)

    results: list[dict[str, Any]] = []

    for short_name, model_id in MODELS.items():
        print()
        print(f"===== {short_name} =====")
        print(f"Loading tokenizer: {model_id}")

        try:
            tokenizer = AutoTokenizer.from_pretrained(
                model_id,
                trust_remote_code=True,
            )

            model_result: dict[str, Any] = {
                "model_name": short_name,
                "model_id": model_id,
                "status": "success",
            }

            for label, candidate in CANDIDATES.items():
                audit = audit_candidate(
                    tokenizer=tokenizer,
                    prompts=prompts,
                    candidate=candidate,
                )

                model_result[label] = audit

                print(
                    f"{label}: "
                    f"{audit['unique_tokenizations']}"
                )

                if audit["prefix_failures"]:
                    print(
                        f"WARNING: {audit['prefix_failures']} "
                        f"prefix failures for {label}"
                    )

            results.append(model_result)

        except Exception as exc:
            print(
                f"ERROR: {type(exc).__name__}: {exc}"
            )

            results.append(
                {
                    "model_name": short_name,
                    "model_id": model_id,
                    "status": "error",
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                }
            )

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with OUTPUT_FILE.open(
        "w",
        encoding="utf-8",
    ) as output:
        json.dump(
            results,
            output,
            ensure_ascii=False,
            indent=2,
        )
        output.write("\n")

    successful = sum(
        row["status"] == "success"
        for row in results
    )

    failed = len(results) - successful

    print()
    print(
        f"Finished: {successful} successful, "
        f"{failed} failed."
    )
    print(f"Saved audit to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
