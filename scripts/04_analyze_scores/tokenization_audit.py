#!/usr/bin/env python3
"""
tokenization_audit.py

Audit how each model tokenizer segments the nouns, verbs, and copulas used
in the CE2 agreement benchmark.

This script loads tokenizers only. It does not load model weights.

Outputs:
    data/scores/summaries/tokenization_audit_forms.csv
    data/scores/summaries/tokenization_audit_pairs.csv

Run from repository root:

    python3 scripts/04_analyze_scores/tokenization_audit.py
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import pandas as pd
from transformers import AutoTokenizer


REPO_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_NOUNS = REPO_ROOT / "data" / "lexicons" / "lexicon_nouns.csv"
DEFAULT_VERBS = REPO_ROOT / "data" / "lexicons" / "lexicon_verbs_transitive.csv"
DEFAULT_OUT_FORMS = REPO_ROOT / "data" / "scores" / "summaries" / "tokenization_audit_forms.csv"
DEFAULT_OUT_PAIRS = REPO_ROOT / "data" / "scores" / "summaries" / "tokenization_audit_pairs.csv"


# Match the model set used in the paper.
# These are tokenizer/model identifiers only; model weights are not loaded.
MODELS = {
    "gpt2": "openai-community/gpt2",
    "pythia410": "EleutherAI/pythia-410m",
    "llama32_1b": "meta-llama/Llama-3.2-1B",
    "llama32_3b": "meta-llama/Llama-3.2-3B",
    "llama31_70b": "meta-llama/Llama-3.1-70B",
    "qwen25_1.5b": "Qwen/Qwen2.5-1.5B",
    "qwen25_3b": "Qwen/Qwen2.5-3B",
    "qwen25_7b": "Qwen/Qwen2.5-7B",
    "qwen25_14b": "Qwen/Qwen2.5-14B",
    "qwen25_32b": "Qwen/Qwen2.5-32B",
    "qwen25_72b": "Qwen/Qwen2.5-72B",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def load_tokenizer(model_id: str):
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        use_fast=True,
        trust_remote_code=True,
    )
    return tokenizer


def tokenize_form(tokenizer, text: str) -> tuple[list[int], list[str]]:
    ids = tokenizer.encode(
        text,
        add_special_tokens=False,
    )
    tokens = tokenizer.convert_ids_to_tokens(ids)
    return ids, tokens


def make_form_rows(model_name: str, model_id: str, tokenizer, nouns, verbs) -> list[dict]:
    rows = []

    def add_form(kind: str, item_id: str, role: str, number: str, form: str):
        # Leading-space form is the scoring-relevant form for continuations.
        text = " " + form
        ids, tokens = tokenize_form(tokenizer, text)

        rows.append(
            {
                "model": model_name,
                "model_id": model_id,
                "kind": kind,
                "item_id": item_id,
                "role": role,
                "number": number,
                "form": form,
                "text_tokenized": text,
                "n_tokens": len(ids),
                "token_ids_json": json.dumps(ids, ensure_ascii=False),
                "tokens_json": json.dumps(tokens, ensure_ascii=False),
            }
        )

    for row in nouns:
        noun_id = row.get("noun_id", "")
        add_form("noun", noun_id, "noun_sg", "sg", row["sg"])
        add_form("noun", noun_id, "noun_pl", "pl", row["pl"])

    for row in verbs:
        verb_id = row.get("verb_id", "")
        add_form("verb", verb_id, "verb_sg", "sg", row["pres_sg"])
        add_form("verb", verb_id, "verb_pl", "pl", row["pres_pl"])

    add_form("copula", "copula", "copula_sg", "sg", "is")
    add_form("copula", "copula", "copula_pl", "pl", "are")

    return rows


def make_pair_rows(forms_df: pd.DataFrame) -> pd.DataFrame:
    pair_rows = []

    for (model, model_id, kind, item_id), sub in forms_df.groupby(
        ["model", "model_id", "kind", "item_id"],
        dropna=False,
    ):
        nums = set(sub["number"])

        if not {"sg", "pl"}.issubset(nums):
            continue

        sg = sub[sub["number"] == "sg"].iloc[0]
        pl = sub[sub["number"] == "pl"].iloc[0]

        pair_rows.append(
            {
                "model": model,
                "model_id": model_id,
                "kind": kind,
                "item_id": item_id,
                "sg_form": sg["form"],
                "pl_form": pl["form"],
                "sg_n_tokens": int(sg["n_tokens"]),
                "pl_n_tokens": int(pl["n_tokens"]),
                "same_token_length": int(sg["n_tokens"] == pl["n_tokens"]),
                "token_length_difference_pl_minus_sg": int(pl["n_tokens"]) - int(sg["n_tokens"]),
                "sg_tokens_json": sg["tokens_json"],
                "pl_tokens_json": pl["tokens_json"],
            }
        )

    return pd.DataFrame(pair_rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit tokenizer segmentation for CE2 benchmark word forms."
    )

    parser.add_argument("--nouns", type=Path, default=DEFAULT_NOUNS)
    parser.add_argument("--verbs", type=Path, default=DEFAULT_VERBS)
    parser.add_argument("--out-forms", type=Path, default=DEFAULT_OUT_FORMS)
    parser.add_argument("--out-pairs", type=Path, default=DEFAULT_OUT_PAIRS)

    args = parser.parse_args()

    nouns = read_csv(args.nouns)
    verbs = read_csv(args.verbs)

    all_form_rows = []

    for model_name, model_id in MODELS.items():
        print(f"Loading tokenizer: {model_name}  ({model_id})")
        tokenizer = load_tokenizer(model_id)

        rows = make_form_rows(
            model_name=model_name,
            model_id=model_id,
            tokenizer=tokenizer,
            nouns=nouns,
            verbs=verbs,
        )

        all_form_rows.extend(rows)

    forms_df = pd.DataFrame(all_form_rows)
    pairs_df = make_pair_rows(forms_df)

    args.out_forms.parent.mkdir(parents=True, exist_ok=True)

    forms_df.to_csv(args.out_forms, index=False)
    pairs_df.to_csv(args.out_pairs, index=False)

    print()
    print(f"Wrote {args.out_forms} ({len(forms_df)} rows)")
    print(f"Wrote {args.out_pairs} ({len(pairs_df)} rows)")

    print()
    print("Unequal singular/plural token-length pairs by model and kind:")
    summary = (
        pairs_df
        .assign(unequal=lambda d: 1 - d["same_token_length"])
        .groupby(["model", "kind"], as_index=False)
        .agg(
            n_pairs=("item_id", "count"),
            n_unequal=("unequal", "sum"),
            share_unequal=("unequal", "mean"),
        )
        .sort_values(["model", "kind"])
    )

    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
