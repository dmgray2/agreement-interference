#!/usr/bin/env python3

import json
from collections import Counter
from pathlib import Path


INPUT = Path(
    "data/stimuli/minpairs.ce2.v2.forced_choice.counterbalanced.v1.jsonl"
)

OUTPUT_A = Path(
    "data/stimuli/minpairs.ce2.v2.forced_choice.orderA.v1.jsonl"
)

OUTPUT_B = Path(
    "data/stimuli/minpairs.ce2.v2.forced_choice.orderB.v1.jsonl"
)


def load_jsonl(path: Path) -> list[dict]:
    records = []

    with path.open("r", encoding="utf-8") as source_file:
        for line_number, line in enumerate(source_file, start=1):
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON on line {line_number} of {path}"
                ) from exc

    return records


def write_jsonl(path: Path, records: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as output_file:
        for record in records:
            output_file.write(
                json.dumps(record, ensure_ascii=False) + "\n"
            )


def validate_batch(
    records: list[dict],
    expected_label: str,
    expected_order: int,
) -> None:
    if len(records) != 2000:
        raise ValueError(
            f"Expected 2000 records for {expected_label}, found {len(records)}"
        )

    labels = Counter(row["grammatical_label"] for row in records)
    if labels != Counter({expected_label: 2000}):
        raise ValueError(
            f"Unexpected grammatical-label counts: {dict(labels)}"
        )

    orders = Counter(row["order_version"] for row in records)
    if orders != Counter({expected_order: 2000}):
        raise ValueError(
            f"Unexpected order-version counts: {dict(orders)}"
        )

    base_ids = [row["base_item_id"] for row in records]
    if len(base_ids) != len(set(base_ids)):
        raise ValueError(
            f"Duplicate base_item_id values found in {expected_label} batch"
        )


def main() -> None:
    records = load_jsonl(INPUT)

    order_a = [
        row for row in records
        if row["grammatical_label"] == "A"
    ]

    order_b = [
        row for row in records
        if row["grammatical_label"] == "B"
    ]

    validate_batch(
        records=order_a,
        expected_label="A",
        expected_order=1,
    )

    validate_batch(
        records=order_b,
        expected_label="B",
        expected_order=2,
    )

    write_jsonl(OUTPUT_A, order_a)
    write_jsonl(OUTPUT_B, order_b)

    print(f"Wrote {len(order_a)} records to {OUTPUT_A}")
    print(f"Wrote {len(order_b)} records to {OUTPUT_B}")


if __name__ == "__main__":
    main()
