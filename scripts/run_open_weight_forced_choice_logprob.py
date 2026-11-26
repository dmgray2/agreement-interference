#!/usr/bin/env python3
"""Constrained A/B scoring with float32 log-softmax (v2).

Model inference precision still follows --dtype. New output names use
logprob_fp32_v2, and metadata records the log-probability precision.
The registered exact-tie rule remains "select A". The pre-fix script is
preserved under scripts/archive/pre_float32_logsoftmax/.
"""

import argparse
import json
import math
import random
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


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

DEFAULT_SEEDS = {
    "A": 71001,
    "B": 71002,
}

CANDIDATES = {
    "A": " A",
    "B": " B",
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
    if not rows:
        raise ValueError(
            f"No rows found for grammatical order {order}"
        )

    prompt_ids = [
        row["prompt_id"]
        for row in rows
    ]

    if len(prompt_ids) != len(set(prompt_ids)):
        raise ValueError(
            "Duplicate prompt_id values found"
        )

    for row in rows:
        if row["grammatical_label"] != order:
            raise ValueError(
                f"{row['prompt_id']} has grammatical label "
                f"{row['grammatical_label']}, expected {order}"
            )

        if not row["prompt"].endswith("Answer:"):
            raise ValueError(
                f"{row['prompt_id']} does not end with 'Answer:'"
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


def candidate_token_ids(
    tokenizer: Any,
    prompt: str,
    candidate: str,
) -> tuple[list[int], list[int]]:
    """
    Return the prompt token IDs and candidate token IDs.

    The complete prompt-plus-candidate tokenization must begin
    with exactly the prompt tokenization. This verifies that the
    response boundary is represented cleanly for this tokenizer.
    """
    prompt_ids = tokenizer.encode(
        prompt,
        add_special_tokens=False,
    )

    complete_ids = tokenizer.encode(
        prompt + candidate,
        add_special_tokens=False,
    )

    if complete_ids[: len(prompt_ids)] != prompt_ids:
        raise ValueError(
            "Prompt-plus-candidate tokenization does not preserve "
            "the prompt as an exact token prefix."
        )

    answer_ids = complete_ids[len(prompt_ids):]

    if not answer_ids:
        raise ValueError(
            f"Candidate {candidate!r} produced no answer tokens"
        )

    return prompt_ids, answer_ids


def audit_candidates(
    tokenizer: Any,
    prompts: list[str],
) -> dict[str, Any]:
    """
    Audit candidate tokenization across all prompts.
    """
    audit: dict[str, Any] = {}

    for label, candidate in CANDIDATES.items():
        token_sequences: dict[tuple[int, ...], int] = {}

        for prompt in prompts:
            _, answer_ids = candidate_token_ids(
                tokenizer,
                prompt,
                candidate,
            )

            key = tuple(answer_ids)
            token_sequences[key] = (
                token_sequences.get(key, 0) + 1
            )

        audit[label] = {
            "candidate_string": candidate,
            "unique_tokenizations": [
                {
                    "token_ids": list(token_ids),
                    "tokens": tokenizer.convert_ids_to_tokens(
                        list(token_ids)
                    ),
                    "number_of_prompts": count,
                }
                for token_ids, count in sorted(
                    token_sequences.items()
                )
            ],
        }

    return audit


def score_candidate_batch(
    model: Any,
    tokenizer: Any,
    prompts: list[str],
    candidate: str,
    model_device: torch.device,
) -> tuple[list[float], list[list[int]]]:
    """
    Calculate the summed conditional log probability of one
    candidate string for each prompt.
    """
    complete_sequences: list[list[int]] = []
    prompt_lengths: list[int] = []
    answer_token_ids: list[list[int]] = []

    for prompt in prompts:
        prompt_ids, answer_ids = candidate_token_ids(
            tokenizer,
            prompt,
            candidate,
        )

        complete_sequences.append(
            prompt_ids + answer_ids
        )
        prompt_lengths.append(len(prompt_ids))
        answer_token_ids.append(answer_ids)

    encoded = tokenizer.pad(
        {
            "input_ids": complete_sequences,
        },
        padding=True,
        return_tensors="pt",
    )

    input_ids = encoded["input_ids"].to(model_device)
    attention_mask = encoded["attention_mask"].to(
        model_device
    )

    with torch.inference_mode():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

    logits = outputs.logits
    # Cast before log-softmax to avoid additional bfloat16 output rounding.
    # Already quantized logits can still produce exact ties.
    log_probabilities = torch.log_softmax(
        logits,
        dim=-1,
        dtype=torch.float32,
    )

    scores: list[float] = []

    for batch_index, (
        prompt_length,
        answer_ids,
        complete_ids,
    ) in enumerate(
        zip(
            prompt_lengths,
            answer_token_ids,
            complete_sequences,
        )
    ):
        score = 0.0

        for answer_offset, token_id in enumerate(
            answer_ids
        ):
            token_position = (
                prompt_length + answer_offset
            )

            prediction_position = token_position - 1

            token_log_probability = log_probabilities[
                batch_index,
                prediction_position,
                token_id,
            ]

            score += float(
                token_log_probability.item()
            )

        scores.append(score)

    return scores, answer_token_ids


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run one open-weight causal language model on one "
            "CE-Agree answer order using constrained A/B "
            "label-probability readout."
        )
    )

    parser.add_argument(
        "--model-id",
        required=True,
        help="Hugging Face model ID or local model directory.",
    )

    parser.add_argument(
        "--model-name",
        required=True,
        help="Short filesystem-safe model name.",
    )

    parser.add_argument(
        "--order",
        required=True,
        choices=("A", "B"),
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help=(
            "Optional combined forced-choice JSONL file. "
            "If supplied, rows are selected by grammatical_label "
            "for the requested --order. If omitted, the legacy "
            "CE2 order-specific file is used."
        ),
    )

    parser.add_argument(
        "--run-tag",
        default="",
        help=(
            "Optional short label added to output filenames, "
            "for example ce3past or ce3present."
        ),
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
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
    )

    parser.add_argument(
        "--shuffle-seed",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Run only the first N shuffled prompts.",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
    )

    args = parser.parse_args()

    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")

    if args.input is None:
        source_path = ORDER_FILES[args.order]
        rows = load_jsonl(source_path)
    else:
        source_path = args.input

        all_rows = load_jsonl(
            source_path
        )

        rows = [
            row
            for row in all_rows
            if (
                row["grammatical_label"]
                == args.order
            )
        ]

    validate_rows(
        rows,
        args.order,
    )
    seed = (
        args.shuffle_seed
        if args.shuffle_seed is not None
        else DEFAULT_SEEDS[args.order]
    )

    shuffled_rows = list(rows)
    random.Random(seed).shuffle(shuffled_rows)

    if args.limit is not None:
        if args.limit <= 0:
            raise ValueError("--limit must be positive")

        shuffled_rows = shuffled_rows[: args.limit]

    tag_part = (
        f"_{args.run_tag}"
        if args.run_tag
        else ""
    )

    run_name = (
        f"open_weight_{args.model_name}"
        f"{tag_part}_"
        f"order{args.order}_logprob_fp32_v2"
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

    print(f"Loading tokenizer: {args.model_id}")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_id,
        trust_remote_code=True,
    )

    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError(
                "Tokenizer has neither a pad token nor an EOS token"
            )

        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "right"

    print("Auditing candidate tokenization...")

    tokenization_audit = audit_candidates(
        tokenizer,
        [row["prompt"] for row in rows],
    )

    for label in ("A", "B"):
        print(
            f"Candidate {label!r}: "
            f"{tokenization_audit[label]['unique_tokenizations']}"
        )

    print(f"Loading model: {args.model_id}")

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=resolve_dtype(args.dtype),
        device_map=args.device_map,
        trust_remote_code=True,
    )

    model.eval()
    model_device = next(model.parameters()).device

    metadata: dict[str, Any] = {
        "provider": "open_weight",
        "run_name": run_name,
        "requested_model": args.model_id,
        "model_name": args.model_name,
        "answer_order": args.order,
        "source_file": str(source_path),
        "number_of_prompts": len(shuffled_rows),
        "batch_size": args.batch_size,
        "dtype": args.dtype,
        "resolved_model_dtype": str(next(model.parameters()).dtype),
        "logprob_dtype": "float32",
        "scoring_version": "float32_logsoftmax_v2",
        "device_map": args.device_map,
        "response_readout": (
            "summed conditional log probability of candidate "
            "strings ' A' and ' B'"
        ),
        "candidate_strings": CANDIDATES,
        "tie_rule": "select A",
        "shuffle_seed": seed,
        "candidate_tokenization_audit": (
            tokenization_audit
        ),
        "started_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "completed": False,
        "output_file": str(output_path),
    }

    write_json(metadata_path, metadata)

    print(f"Model device: {model_device}")
    print(f"Prompts to run: {len(shuffled_rows)}")
    print(f"Batch size: {args.batch_size}")
    print(f"Shuffle seed: {seed}")
    print(f"Output file: {output_path}")

    total_started = time.time()
    completed_count = 0
    tie_count = 0

    for start_index in range(
        0,
        len(shuffled_rows),
        args.batch_size,
    ):
        batch_rows = shuffled_rows[
            start_index:
            start_index + args.batch_size
        ]

        prompts = [
            row["prompt"]
            for row in batch_rows
        ]

        call_started = time.time()

        try:
            scores_a, token_ids_a = score_candidate_batch(
                model=model,
                tokenizer=tokenizer,
                prompts=prompts,
                candidate=CANDIDATES["A"],
                model_device=model_device,
            )

            scores_b, token_ids_b = score_candidate_batch(
                model=model,
                tokenizer=tokenizer,
                prompts=prompts,
                candidate=CANDIDATES["B"],
                model_device=model_device,
            )

            batch_seconds = (
                time.time() - call_started
            )

            for index, row in enumerate(batch_rows):
                logprob_a = scores_a[index]
                logprob_b = scores_b[index]

                tied = math.isclose(
                    logprob_a,
                    logprob_b,
                    rel_tol=0.0,
                    abs_tol=0.0,
                )

                if tied:
                    selected_label = "A"
                    tie_count += 1
                elif logprob_a > logprob_b:
                    selected_label = "A"
                else:
                    selected_label = "B"

                result = {
                    "custom_id": row["prompt_id"],
                    "status": "success",
                    "requested_model": args.model_id,
                    "returned_model": args.model_id,
                    "raw_response": selected_label,
                    "selected_label": selected_label,
                    "candidate_a": CANDIDATES["A"],
                    "candidate_b": CANDIDATES["B"],
                    "logprob_a": logprob_a,
                    "logprob_b": logprob_b,
                    "logprob_a_minus_b": (
                        logprob_a - logprob_b
                    ),
                    "candidate_a_token_ids": (
                        token_ids_a[index]
                    ),
                    "candidate_b_token_ids": (
                        token_ids_b[index]
                    ),
                    "exact_tie": tied,
                    "input_tokens": len(
                        tokenizer.encode(
                            row["prompt"],
                            add_special_tokens=False,
                        )
                    ),
                    "output_tokens": None,
                    "reasoning_tokens": None,
                    "stop_reason": (
                        "constrained_label_probability"
                    ),
                    "incomplete_reason": None,
                    "batch_seconds": batch_seconds,
                }

                append_jsonl(output_path, result)
                completed_count += 1

        except Exception as exc:
            batch_seconds = (
                time.time() - call_started
            )

            for row in batch_rows:
                result = {
                    "custom_id": row["prompt_id"],
                    "status": "error",
                    "requested_model": args.model_id,
                    "returned_model": None,
                    "raw_response": None,
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                    "input_tokens": None,
                    "output_tokens": None,
                    "reasoning_tokens": None,
                    "stop_reason": None,
                    "incomplete_reason": None,
                    "batch_seconds": batch_seconds,
                }

                append_jsonl(output_path, result)
                completed_count += 1

        print(
            f"\rCompleted {completed_count}/"
            f"{len(shuffled_rows)}",
            end="",
            flush=True,
        )

    elapsed_seconds = time.time() - total_started

    metadata.update(
        {
            "completed": True,
            "completed_at_utc": datetime.now(
                timezone.utc
            ).isoformat(),
            "elapsed_seconds": elapsed_seconds,
            "completed_prompt_count": completed_count,
            "exact_tie_count": tie_count,
        }
    )

    write_json(metadata_path, metadata)

    print()
    print(
        f"Finished {completed_count} prompts in "
        f"{elapsed_seconds:.2f} seconds."
    )
    print(f"Exact ties: {tie_count}")
    print(f"Results saved to: {output_path}")
    print(f"Metadata saved to: {metadata_path}")


if __name__ == "__main__":
    main()
