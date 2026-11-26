#!/usr/bin/env python3

from pathlib import Path
import pandas as pd


INPUT = Path("results/final/master_forced_choice_scores_v1.csv")
OUT_DIR = Path("paper/tables")

REGIME_CSV = OUT_DIR / "table_forced_choice_regimes.csv"
REGIME_TEX = OUT_DIR / "table_forced_choice_regimes.tex"

PATTERN_CSV = OUT_DIR / "table_forced_choice_number_patterns.csv"
PATTERN_TEX = OUT_DIR / "table_forced_choice_number_patterns.tex"


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


REGIME_ORDER = [
    "both_match",
    "N3_mismatch_only",
    "N1_mismatch_only",
    "both_mismatch",
]

PATTERN_ORDER = [
    "SSS",
    "PPP",
    "SSP",
    "PPS",
    "PSS",
    "SPP",
    "SPS",
    "PSP",
]


def family(model):
    if model.startswith(("gpt2", "pythia", "llama", "qwen")):
        return "Open weight"
    if model.startswith("gpt-"):
        return "OpenAI"
    if model.startswith("opus-"):
        return "Anthropic"
    return "Other"


def validate_models(df):
    found = set(df["model"].unique())
    expected = set(MODEL_ORDER)

    if found != expected:
        raise ValueError(
            f"Model mismatch.\nExpected: {sorted(expected)}\n"
            f"Found: {sorted(found)}"
        )


def add_model_metadata(df):
    df = df.copy()

    df["model_order"] = df["model"].map(
        {m: i for i, m in enumerate(MODEL_ORDER)}
    )
    df["family"] = df["model"].map(family)
    df["display_model"] = df["model"].map(DISPLAY_NAMES)

    return df.sort_values("model_order").reset_index(drop=True)


def make_regime_table(df):
    grouped = (
        df.groupby(["model", "cue_regime"])["counterbalanced_accuracy"]
        .mean()
        .unstack("cue_regime")
    )

    missing = [
        regime for regime in REGIME_ORDER
        if regime not in grouped.columns
    ]

    if missing:
        raise ValueError(f"Missing cue regimes: {missing}")

    grouped = grouped[REGIME_ORDER].reset_index()

    grouped["N3_effect"] = (
        grouped["both_match"] -
        grouped["N3_mismatch_only"]
    )

    grouped["N1_effect"] = (
        grouped["both_match"] -
        grouped["N1_mismatch_only"]
    )

    # Positive means the N1 mismatch is associated with a larger
    # reduction in accuracy than the N3 mismatch.
    grouped["N1_minus_N3_asymmetry"] = (
        grouped["N3_mismatch_only"] -
        grouped["N1_mismatch_only"]
    )

    grouped = add_model_metadata(grouped)

    out = grouped[
        [
            "family",
            "model",
            "display_model",
            "both_match",
            "N3_mismatch_only",
            "N1_mismatch_only",
            "both_mismatch",
            "N3_effect",
            "N1_effect",
            "N1_minus_N3_asymmetry",
        ]
    ]

    out.to_csv(REGIME_CSV, index=False)

    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{lrrrrrrr}",
        r"\toprule",
        r"Model & Both match & N3 mis. & N1 mis. & Both mis. & "
        r"$I_{N3}$ & $I_{N1}$ & N1--N3 \\",
        r"\midrule",
    ]

    previous_family = None

    for _, row in out.iterrows():
        current_family = row["family"]

        if previous_family is not None and current_family != previous_family:
            lines.append(r"\midrule")

        lines.append(
            f"{row['display_model']} & "
            f"{row['both_match']:.3f} & "
            f"{row['N3_mismatch_only']:.3f} & "
            f"{row['N1_mismatch_only']:.3f} & "
            f"{row['both_mismatch']:.3f} & "
            f"{row['N3_effect']:.3f} & "
            f"{row['N1_effect']:.3f} & "
            f"{row['N1_minus_N3_asymmetry']:.3f} \\\\"
        )

        previous_family = current_family

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\caption{Forced-choice accuracy by cue regime. "
        r"$I_{N3}$ is both-match accuracy minus N3-mismatch-only accuracy; "
        r"$I_{N1}$ is both-match accuracy minus N1-mismatch-only accuracy. "
        r"N1--N3 is N3-mismatch-only accuracy minus N1-mismatch-only "
        r"accuracy, so positive values indicate a larger accuracy reduction "
        r"for N1 mismatch than for N3 mismatch.}",
        r"\label{tab:forced-choice-regimes}",
        r"\end{table*}",
    ])

    REGIME_TEX.write_text("\n".join(lines) + "\n", encoding="utf-8")

    return out


def make_pattern_table(df):
    grouped = (
        df.groupby(["model", "number_pattern"])["counterbalanced_accuracy"]
        .mean()
        .unstack("number_pattern")
    )

    missing = [
        pattern for pattern in PATTERN_ORDER
        if pattern not in grouped.columns
    ]

    if missing:
        raise ValueError(f"Missing number patterns: {missing}")

    grouped = grouped[PATTERN_ORDER].reset_index()
    grouped = add_model_metadata(grouped)

    out = grouped[
        [
            "family",
            "model",
            "display_model",
            *PATTERN_ORDER,
        ]
    ]

    out.to_csv(PATTERN_CSV, index=False)

    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{3.5pt}",
        r"\begin{tabular}{lrrrrrrrr}",
        r"\toprule",
        r"Model & SSS & PPP & SSP & PPS & PSS & SPP & SPS & PSP \\",
        r"\midrule",
    ]

    previous_family = None

    for _, row in out.iterrows():
        current_family = row["family"]

        if previous_family is not None and current_family != previous_family:
            lines.append(r"\midrule")

        vals = " & ".join(
            f"{row[p]:.3f}"
            for p in PATTERN_ORDER
        )

        lines.append(
            f"{row['display_model']} & {vals} \\\\"
        )

        previous_family = current_family

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\caption{Forced-choice paired accuracy for each N1--N2--N3 "
        r"number pattern. S and P indicate singular and plural noun number, "
        r"respectively.}",
        r"\label{tab:forced-choice-number-patterns}",
        r"\end{table*}",
    ])

    PATTERN_TEX.write_text("\n".join(lines) + "\n", encoding="utf-8")

    return out


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(INPUT, low_memory=False)

    # GPT-3.5 remains audit-only because all responses failed the
    # preregistered validity rule.
    df = df[df["model"] != "gpt-3.5"].copy()

    validate_models(df)

    counts = df.groupby("model").size()

    if not (counts == 2000).all():
        raise ValueError(
            "Expected 2000 items per model:\n"
            f"{counts}"
        )

    regime = make_regime_table(df)
    pattern = make_pattern_table(df)

    print("Validation PASS")
    print(f"Models: {df['model'].nunique()}")
    print("Items/model: 2000")
    print()
    print(f"Wrote: {REGIME_CSV}")
    print(f"Wrote: {REGIME_TEX}")
    print(f"Wrote: {PATTERN_CSV}")
    print(f"Wrote: {PATTERN_TEX}")
    print()

    print("REGIME TABLE")
    print(
        regime[
            [
                "display_model",
                "both_match",
                "N3_mismatch_only",
                "N1_mismatch_only",
                "both_mismatch",
                "N3_effect",
                "N1_effect",
                "N1_minus_N3_asymmetry",
            ]
        ].to_string(index=False)
    )

    print()
    print("NUMBER-PATTERN TABLE")
    print(
        pattern[
            [
                "display_model",
                *PATTERN_ORDER,
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
