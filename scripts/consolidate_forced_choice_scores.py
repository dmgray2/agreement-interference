#!/usr/bin/env python3

"""
Build the canonical CE-Agree forced-choice master score table.

Inputs:
    results/final/*_paired_v1.jsonl

Output:
    results/final/master_forced_choice_scores_v1.csv

The master contains one row per model × base item.

Expected:
    18 models
    2,000 paired items per model
    36,000 rows total

Nested number and lexical metadata are flattened into ordinary
CSV columns. No scores are changed.
"""

import csv
import json
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]

FINAL_DIR = (
    REPO_ROOT
    / "results"
    / "final"
)

DEFAULT_OUTPUT = (
    FINAL_DIR
    / "master_forced_choice_scores_v1.csv"
)


EXPECTED_FILES = {
    # Open-weight models
    "gpt2_forced_choice_paired_v1.jsonl": {
        "model": "gpt2",
        "response_interface": (
            "constrained_next_token_logprob"
        ),
    },
    "pythia-410m_forced_choice_paired_v1.jsonl": {
        "model": "pythia-410m",
        "response_interface": (
            "constrained_next_token_logprob"
        ),
    },
    "llama-3.2-1b_forced_choice_paired_v1.jsonl": {
        "model": "llama-3.2-1b",
        "response_interface": (
            "constrained_next_token_logprob"
        ),
    },
    "llama-3.2-3b_forced_choice_paired_v1.jsonl": {
        "model": "llama-3.2-3b",
        "response_interface": (
            "constrained_next_token_logprob"
        ),
    },
    "llama-3.1-70b_forced_choice_paired_v1.jsonl": {
        "model": "llama-3.1-70b",
        "response_interface": (
            "constrained_next_token_logprob"
        ),
    },
    "qwen2.5-1.5b_forced_choice_paired_v1.jsonl": {
        "model": "qwen2.5-1.5b",
        "response_interface": (
            "constrained_next_token_logprob"
        ),
    },
    "qwen2.5-3b_forced_choice_paired_v1.jsonl": {
        "model": "qwen2.5-3b",
        "response_interface": (
            "constrained_next_token_logprob"
        ),
    },
    "qwen2.5-7b_forced_choice_paired_v1.jsonl": {
        "model": "qwen2.5-7b",
        "response_interface": (
            "constrained_next_token_logprob"
        ),
    },
    "qwen2.5-14b_forced_choice_paired_v1.jsonl": {
        "model": "qwen2.5-14b",
        "response_interface": (
            "constrained_next_token_logprob"
        ),
    },
    "qwen2.5-32b_forced_choice_paired_v1.jsonl": {
        "model": "qwen2.5-32b",
        "response_interface": (
            "constrained_next_token_logprob"
        ),
    },
    "qwen2.5-72b_forced_choice_paired_v1.jsonl": {
        "model": "qwen2.5-72b",
        "response_interface": (
            "constrained_next_token_logprob"
        ),
    },

    # OpenAI
    "openai_gpt-3.5_paired_v1.jsonl": {
        "model": "gpt-3.5",
        "response_interface": "free_text",
    },
    "openai_gpt-4.1_paired_v1.jsonl": {
        "model": "gpt-4.1",
        "response_interface": (
            "forced_record_answer_tool"
        ),
    },
    "openai_gpt-5_paired_v1.jsonl": {
        "model": "gpt-5",
        "response_interface": (
            "forced_record_answer_tool"
        ),
    },
    "openai_gpt-5.6-sol_paired_v1.jsonl": {
        "model": "gpt-5.6-sol",
        "response_interface": (
            "forced_record_answer_tool"
        ),
    },

    # Anthropic
    "anthropic_opus-4.1_paired_v1.jsonl": {
        "model": "opus-4.1",
        "response_interface": "forced_tool_call",
    },
    "anthropic_opus-4.5_paired_v1.jsonl": {
        "model": "opus-4.5",
        "response_interface": "forced_tool_call",
    },
    "anthropic_opus-5_paired_v1.jsonl": {
        "model": "opus-5",
        "response_interface": "forced_tool_call",
    },
}


FIELDNAMES = [
    "model",
    "provider",
    "requested_model",
    "response_interface",
    "source_file",
    "base_item_id",
    "source_family_id",
    "number_pattern",
    "cue_regime",
    "global_congruence",
    "nearest_attractor_congruence",
    "np1_num",
    "np2_num",
    "np3_num",
    "np1_lexeme",
    "np2_lexeme",
    "np3_lexeme",
    "v2_lexeme",
    "v3_lexeme",
    "correct_order_A",
    "correct_order_B",
    "valid_order_A",
    "valid_order_B",
    "raw_response_order_A",
    "raw_response_order_B",
    "technical_missing",
    "counterbalanced_accuracy",
    "strict_paired_correct",
    "order_inconsistent",
]


def load_jsonl(
    path: Path,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    with path.open(
        "r",
        encoding="utf-8",
    ) as source:
        for line_number, line in enumerate(
            source,
            start=1,
        ):
            if not line.strip():
                continue

            try:
                rows.append(
                    json.loads(line)
                )
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON on line "
                    f"{line_number} of {path}"
                ) from exc

    return rows


def flatten_row(
    row: dict[str, Any],
    model: str,
    response_interface: str,
    source_file: str,
) -> dict[str, Any]:
    nums = row.get("nums") or {}
    lex = row.get("lex") or {}

    return {
        "model": model,
        "provider": row.get("provider"),
        "requested_model": row.get(
            "requested_model"
        ),
        "response_interface": (
            response_interface
        ),
        "source_file": source_file,
        "base_item_id": row.get(
            "base_item_id"
        ),
        "source_family_id": row.get(
            "source_family_id"
        ),
        "number_pattern": row.get(
            "number_pattern"
        ),
        "cue_regime": row.get(
            "cue_regime"
        ),
        "global_congruence": row.get(
            "global_congruence"
        ),
        "nearest_attractor_congruence": (
            row.get(
                "nearest_attractor_congruence"
            )
        ),
        "np1_num": nums.get("NP1"),
        "np2_num": nums.get("NP2"),
        "np3_num": nums.get("NP3"),
        "np1_lexeme": lex.get("NP1"),
        "np2_lexeme": lex.get("NP2"),
        "np3_lexeme": lex.get("NP3"),
        "v2_lexeme": lex.get("V2"),
        "v3_lexeme": lex.get("V3"),
        "correct_order_A": row.get(
            "correct_order_A"
        ),
        "correct_order_B": row.get(
            "correct_order_B"
        ),
        "valid_order_A": row.get(
            "valid_order_A"
        ),
        "valid_order_B": row.get(
            "valid_order_B"
        ),
        "raw_response_order_A": row.get(
            "raw_response_order_A"
        ),
        "raw_response_order_B": row.get(
            "raw_response_order_B"
        ),
        "technical_missing": row.get(
            "technical_missing"
        ),
        "counterbalanced_accuracy": row.get(
            "counterbalanced_accuracy"
        ),
        "strict_paired_correct": row.get(
            "strict_paired_correct"
        ),
        "order_inconsistent": row.get(
            "order_inconsistent"
        ),
    }


def validate_model_rows(
    filename: str,
    rows: list[dict[str, Any]],
) -> None:
    if len(rows) != 2000:
        raise ValueError(
            f"{filename}: expected 2,000 rows, "
            f"found {len(rows)}"
        )

    item_ids = [
        row.get("base_item_id")
        for row in rows
    ]

    if None in item_ids:
        raise ValueError(
            f"{filename}: missing base_item_id"
        )

    if len(item_ids) != len(set(item_ids)):
        raise ValueError(
            f"{filename}: duplicate "
            "base_item_id values"
        )

    for row in rows:
        score = row.get(
            "counterbalanced_accuracy"
        )

        if score not in (
            0,
            0.0,
            0.5,
            1,
            1.0,
            None,
        ):
            raise ValueError(
                f"{filename}: unexpected "
                f"counterbalanced score {score!r}"
            )


def write_csv(
    path: Path,
    rows: list[dict[str, Any]],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as output:
        writer = csv.DictWriter(
            output,
            fieldnames=FIELDNAMES,
        )

        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    master_rows: list[dict[str, Any]] = []
    reference_item_ids: set[str] | None = None

    print(
        "Building forced-choice master score table."
    )
    print()

    for filename, metadata in (
        EXPECTED_FILES.items()
    ):
        path = FINAL_DIR / filename

        if not path.exists():
            raise FileNotFoundError(
                f"Missing expected paired file: "
                f"{path}"
            )

        rows = load_jsonl(path)

        validate_model_rows(
            filename,
            rows,
        )

        item_ids = {
            row["base_item_id"]
            for row in rows
        }

        if reference_item_ids is None:
            reference_item_ids = item_ids
        elif item_ids != reference_item_ids:
            raise ValueError(
                f"{filename}: item IDs do not "
                "match the reference model"
            )

        flattened = [
            flatten_row(
                row=row,
                model=metadata["model"],
                response_interface=metadata[
                    "response_interface"
                ],
                source_file=filename,
            )
            for row in rows
        ]

        master_rows.extend(flattened)

        valid_pairs = sum(
            row.get("valid_order_A") is True
            and row.get("valid_order_B") is True
            for row in rows
        )

        print(
            f"Loaded {len(rows):>4} rows  "
            f"{metadata['model']:<16} "
            f"valid pairs={valid_pairs}"
        )

    expected_rows = (
        len(EXPECTED_FILES) * 2000
    )

    if len(master_rows) != expected_rows:
        raise ValueError(
            f"Expected {expected_rows} master "
            f"rows, found {len(master_rows)}"
        )

    write_csv(
        DEFAULT_OUTPUT,
        master_rows,
    )

    print()
    print("Master checks:")
    print(
        f"Models: {len(EXPECTED_FILES)}"
    )
    print(
        f"Unique items: "
        f"{len(reference_item_ids or set())}"
    )
    print(
        f"Rows: {len(master_rows)}"
    )

    technical_missing = sum(
        bool(row["technical_missing"])
        for row in master_rows
    )

    invalid_pairs = sum(
        row["valid_order_A"] is not True
        or row["valid_order_B"] is not True
        for row in master_rows
    )

    print(
        f"Technical-missing rows: "
        f"{technical_missing}"
    )
    print(
        f"Rows with at least one invalid "
        f"response: {invalid_pairs}"
    )

    print()
    print(
        f"Wrote {DEFAULT_OUTPUT}"
    )


if __name__ == "__main__":
    main()
