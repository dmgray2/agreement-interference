#!/usr/bin/env python3

from pathlib import Path
import pandas as pd


INPUT = Path("results/final/master_forced_choice_scores_v1.csv")
OUT_DIR = Path("paper/tables")

OUT_CSV = OUT_DIR / "table_forced_choice_model_comparison.csv"
OUT_TEX = OUT_DIR / "table_forced_choice_model_comparison.tex"


MODEL_ORDER = [
    # Open weight
    "gpt2",
    "pythia-410m",
    "llama-3.2-1b",
    "llama-3.2-3b",
    "qwen2.5-1.5b",
    "qwen2.5-3b",
    "qwen2.5-7b",
    "qwen2.5-14b",
    "qwen2.5-32b",
    "llama-3.1-70b",
    "qwen2.5-72b",

    # OpenAI
    "gpt-4.1",
    "gpt-5",
    "gpt-5.6-sol",

    # Anthropic
    "opus-4.1",
    "opus-4.5",
    "opus-5",
]


DISPLAY_NAMES = {
    "gpt2": "GPT-2",
    "pythia-410m": "Pythia-410M",
    "llama-3.2-1b": "Llama-3.2-1B",
    "llama-3.2-3b": "Llama-3.2-3B",
    "llama-3.1-70b": "Llama-3.1-70B",
    "qwen2.5-1.5b": "Qwen2.5-1.5B",
    "qwen2.5-3b": "Qwen2.5-3B",
    "qwen2.5-7b": "Qwen2.5-7B",
    "qwen2.5-14b": "Qwen2.5-14B",
    "qwen2.5-32b": "Qwen2.5-32B",
    "qwen2.5-72b": "Qwen2.5-72B",
    "gpt-4.1": "GPT-4.1",
    "gpt-5": "GPT-5",
    "gpt-5.6-sol": "GPT-5.6 Sol",
    "opus-4.1": "Claude Opus 4.1",
    "opus-4.5": "Claude Opus 4.5",
    "opus-5": "Claude Opus 5",
}


def family(model):
    if model.startswith(("gpt2", "pythia", "llama", "qwen")):
        return "Open weight"
    if model.startswith("gpt-"):
        return "OpenAI"
    if model.startswith("opus-"):
        return "Anthropic"
    return "Other"


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(INPUT, low_memory=False)

    # GPT-3.5 is audit-only because the response format failed.
    df = df[df["model"] != "gpt-3.5"].copy()

    summary = (
        df.groupby("model", as_index=False)
        .agg(
            n_items=("base_item_id", "size"),
            paired_accuracy=("counterbalanced_accuracy", "mean"),
            strict_both_correct=("strict_paired_correct", "mean"),
            order_inconsistency=("order_inconsistent", "mean"),
        )
    )

    found = set(summary["model"])
    expected = set(MODEL_ORDER)

    if found != expected:
        raise ValueError(
            f"Model mismatch.\nExpected: {sorted(expected)}\n"
            f"Found: {sorted(found)}"
        )

    summary["model_order"] = summary["model"].map(
        {m: i for i, m in enumerate(MODEL_ORDER)}
    )
    summary = summary.sort_values("model_order").reset_index(drop=True)

    summary["family"] = summary["model"].map(family)
    summary["display_model"] = summary["model"].map(DISPLAY_NAMES)

    out = summary[
        [
            "family",
            "model",
            "display_model",
            "n_items",
            "paired_accuracy",
            "strict_both_correct",
            "order_inconsistency",
        ]
    ]

    out.to_csv(OUT_CSV, index=False)

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"Model & Paired acc. & Both correct & Order incons. \\",
        r"\midrule",
    ]

    previous_family = None

    for _, row in out.iterrows():
        current_family = row["family"]

        if previous_family is not None and current_family != previous_family:
            lines.append(r"\midrule")

        lines.append(
            f"{row['display_model']} & "
            f"{row['paired_accuracy']:.3f} & "
            f"{row['strict_both_correct']:.3f} & "
            f"{row['order_inconsistency']:.3f} \\\\"
        )

        previous_family = current_family

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\caption{Forced-choice performance across models. "
        r"Paired accuracy averages correctness across the two "
        r"counterbalanced A/B presentations. Both correct is the "
        r"proportion of items answered correctly in both orders. "
        r"Order inconsistency is the proportion for which correctness "
        r"changes under A/B reversal. GPT-3.5 is excluded because its "
        r"responses were malformed under the preregistered validity rule.}",
        r"\label{tab:forced-choice-model-comparison}",
        r"\end{table}",
    ])

    OUT_TEX.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("Validation PASS")
    print(f"Models: {len(out)}")
    print(f"Wrote: {OUT_CSV}")
    print(f"Wrote: {OUT_TEX}")
    print()
    print(
        out[
            [
                "display_model",
                "paired_accuracy",
                "strict_both_correct",
                "order_inconsistency",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
