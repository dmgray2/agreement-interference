#!/usr/bin/env python3

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)


ORDER_FILES = {
    "A": Path(
        "data/stimuli/"
        "minpairs.ce2.v2.forced_choice.orderA.v1.jsonl"
    ),
    "B": Path(
        "data/stimuli/"
        "minpairs.ce2.v2.forced_choice.orderB.v1.jsonl"
    ),
}

DEFAULT_MAX_NEW_TOKENS = 16


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


def append_jsonl(
    path: Path,
    row: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("a", encoding="utf-8") as output:
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


def validate_rows(
    rows: list[dict[str, Any]],
    order: str,
) -> None:
    if len(rows) != 2000:
        raise ValueError(
            f"Expected 2,000 rows, found {len(rows)}"
        )

    prompt_ids = [
        row["prompt_id"]
        for row in rows
    ]

    if len(prompt_ids) != len(set(prompt_ids)):
        raise ValueError(
            "Duplicate prompt_id values found"
        )

    wrong_labels = [
        row["prompt_id"]
        for row in rows
        if row["grammatical_label"] != order
    ]

    if wrong_labels:
        raise ValueError(
            f"{len(wrong_labels)} rows have the wrong "
            f"grammatical label for order {order}"
        )


def resolve_dtype(
    dtype_name: str,
) -> torch.dtype | str:
    if dtype_name == "auto":
        return "auto"

    mapping = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }

    return mapping[dtype_name]


def choose_device() -> str:
    if torch.cuda.is_available():
        return "cuda"

    if hasattr(torch.backends, "mps"):
        if torch.backends.mps.is_available():
            return "mps"

    return "cpu"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run one open-weight model on one counterbalanced "
            "CE-Agree forced-choice order."
        )
    )

    parser.add_argument(
        "--model-id",
        required=True,
        help=(
            "Hugging Face model ID or local model directory."
        ),
    )

    parser.add_argument(
        "--model-name",
        required=True,
        help=(
            "Short filesystem-safe name for output files."
        ),
    )

    parser.add_argument(
        "--order",
        required=True,
        choices=("A", "B"),
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=DEFAULT_MAX_NEW_TOKENS,
    )

    parser.add_argument(
        "--dtype",
        choices=(
            "auto",
            "float32",
            "float16",
            "bfloat16",
        ),
        default="auto",
    )

    parser.add_argument(
        "--device-map",
        default="auto",
        help=(
            "Transformers device_map value. Default: auto."
        ),
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help=(
            "Run only the first N prompts for a smoke test."
        ),
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Replace an existing result file."
        ),
    )

    args = parser.parse_args()

    if args.batch_size <= 0:
        raise ValueError(
            "--batch-size must be positive"
        )

    if args.max_new_tokens <= 0:
        raise ValueError(
            "--max-new-tokens must be positive"
        )

    source_path = ORDER_FILES[args.order]
    rows = load_jsonl(source_path)
    validate_rows(rows, args.order)

    if args.limit is not None:
        if args.limit <= 0:
            raise ValueError(
                "--limit must be positive"
            )

        rows = rows[:args.limit]

    run_name = (
        f"open_weight_{args.model_name}_"
        f"order{args.order}_v1"
    )

    output_path = Path(
        "results/parsed_batches/open_weight"
    ) / f"{run_name}.jsonl"

    metadata_path = Path(
        "results/batch_manifests/open_weight"
    ) / f"{run_name}.json"

    if output_path.exists():
        if not args.overwrite:
            raise FileExistsError(
                f"{output_path} already exists. "
                "Use --overwrite to replace it."
            )

        output_path.unlink()

    started_at = datetime.now(
        timezone.utc
    ).isoformat()

    metadata = {
        "provider": "open_weight",
        "run_name": run_name,
        "requested_model": args.model_id,
        "model_name": args.model_name,
        "answer_order": args.order,
        "source_file": str(source_path),
        "number_of_prompts": len(rows),
        "batch_size": args.batch_size,
        "max_new_tokens": args.max_new_tokens,
        "decoding": "greedy",
        "do_sample": False,
        "temperature": None,
        "dtype": args.dtype,
        "device_map": args.device_map,
        "started_at_utc": started_at,
        "completed": False,
        "output_file": str(output_path),
    }

    write_json(metadata_path, metadata)

    print(f"Loading tokenizer: {args.model_id}")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_id,
        trust_remote_code=True,
    )

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "left"

    print(f"Loading model: {args.model_id}")

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=resolve_dtype(args.dtype),
        device_map=args.device_map,
        trust_remote_code=True,
    )

    model.eval()

    model_device = next(
        model.parameters()
    ).device

    print(f"Model device: {model_device}")
    print(f"Prompts to run: {len(rows)}")
    print(f"Batch size: {args.batch_size}")
    print(
        f"Maximum new tokens: {args.max_new_tokens}"
    )
    print(f"Output file: {output_path}")

    total_started = time.time()
    completed_count = 0

    for start_index in range(
        0,
        len(rows),
        args.batch_size,
    ):
        batch_rows = rows[
            start_index:
            start_index + args.batch_size
        ]

        prompts = [
            row["prompt"]
            for row in batch_rows
        ]

        encoded = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=False,
        )

        encoded = {
            key: value.to(model_device)
            for key, value in encoded.items()
        }

        input_length = encoded[
            "input_ids"
        ].shape[1]

        call_started = time.time()

        try:
            with torch.inference_mode():
                generated = model.generate(
                    **encoded,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )

            generated_tokens = generated[
                :,
                input_length:
            ]

            decoded_outputs = tokenizer.batch_decode(
                generated_tokens,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )

            call_seconds = (
                time.time() - call_started
            )

            for row, generated_ids, text in zip(
                batch_rows,
                generated_tokens,
                decoded_outputs,
            ):
                output_token_count = int(
                    generated_ids.numel()
                )

                result = {
                    "custom_id": row["prompt_id"],
                    "status": "success",
                    "requested_model": args.model_id,
                    "returned_model": args.model_id,
                    "raw_response": text,
                    "input_tokens": int(
                        encoded["attention_mask"][
                            batch_rows.index(row)
                        ].sum().item()
                    ),
                    "output_tokens": output_token_count,
                    "reasoning_tokens": None,
                    "stop_reason": (
                        "eos_or_limit"
                    ),
                    "incomplete_reason": None,
                    "batch_seconds": call_seconds,
                }

                append_jsonl(
                    output_path,
                    result,
                )

                completed_count += 1

        except Exception as exc:
            call_seconds = (
                time.time() - call_started
            )

            for row in batch_rows:
                result = {
                    "custom_id": row["prompt_id"],
                    "status": "error",
                    "requested_model": args.model_id,
                    "returned_model": None,
                    "raw_response": None,
                    "input_tokens": None,
                    "output_tokens": None,
                    "reasoning_tokens": None,
                    "stop_reason": None,
                    "incomplete_reason": None,
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                    "batch_seconds": call_seconds,
                }

                append_jsonl(
                    output_path,
                    result,
                )

                completed_count += 1

        print(
            f"\rCompleted {completed_count}/{len(rows)}",
            end="",
            flush=True,
        )

    elapsed_seconds = (
        time.time() - total_started
    )

    metadata.update(
        {
            "completed": True,
            "completed_at_utc": datetime.now(
                timezone.utc
            ).isoformat(),
            "elapsed_seconds": elapsed_seconds,
            "completed_prompt_count": (
                completed_count
            ),
        }
    )

    write_json(metadata_path, metadata)

    print()
    print(
        f"Finished {completed_count} prompts in "
        f"{elapsed_seconds:.2f} seconds."
    )
    print(f"Results saved to: {output_path}")
    print(f"Metadata saved to: {metadata_path}")


if __name__ == "__main__":
    main()
