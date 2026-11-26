#!/usr/bin/env python3

import argparse
from pathlib import Path
import pandas as pd


OUT_DIR = Path("paper/tables")
OUT_CSV = OUT_DIR / "table_frontier_inference_settings.csv"
OUT_TEX = OUT_DIR / "table_frontier_inference_settings.tex"


ROWS = [
    {
        "provider": "OpenAI",
        "model": "GPT-3.5 Turbo",
        "api_id": "gpt-3.5-turbo-0125",
        "reasoning_thinking": "No reasoning parameter",
        "temperature": "omitted",
        "top_p": "omitted",
        "max_output_tokens": 250,
        "response_interface": "Free text, strict A/B",
    },
    {
        "provider": "OpenAI",
        "model": "GPT-4.1",
        "api_id": "gpt-4.1-2025-04-14",
        "reasoning_thinking": "No reasoning parameter",
        "temperature": "omitted",
        "top_p": "omitted",
        "max_output_tokens": 250,
        "response_interface": "Forced A/B tool call",
    },
    {
        "provider": "OpenAI",
        "model": "GPT-5",
        "api_id": "gpt-5-2025-08-07",
        "reasoning_thinking": "reasoning.effort = minimal",
        "temperature": "omitted",
        "top_p": "omitted",
        "max_output_tokens": 250,
        "response_interface": "Forced A/B tool call",
    },
    {
        "provider": "OpenAI",
        "model": "GPT-5.6 Sol",
        "api_id": "gpt-5.6-sol",
        "reasoning_thinking": "reasoning.effort = none",
        "temperature": "omitted",
        "top_p": "omitted",
        "max_output_tokens": 250,
        "response_interface": "Forced A/B tool call",
    },
    {
        "provider": "Anthropic",
        "model": "Claude Opus 4.1",
        "api_id": "claude-opus-4-1-20250805",
        "reasoning_thinking": "thinking parameter omitted",
        "temperature": "omitted",
        "top_p": "omitted",
        "max_output_tokens": 100,
        "response_interface": "Forced A/B tool call",
    },
    {
        "provider": "Anthropic",
        "model": "Claude Opus 4.5",
        "api_id": "claude-opus-4-5-20251101",
        "reasoning_thinking": "thinking parameter omitted",
        "temperature": "omitted",
        "top_p": "omitted",
        "max_output_tokens": 100,
        "response_interface": "Forced A/B tool call",
    },
    {
        "provider": "Anthropic",
        "model": "Claude Opus 5",
        "api_id": "claude-opus-5",
        "reasoning_thinking": "thinking parameter omitted",
        "temperature": "omitted",
        "top_p": "omitted",
        "max_output_tokens": 100,
        "response_interface": "Forced A/B tool call",
    },
]


def latex_escape(text):
    return (
        str(text)
        .replace("\\", r"\textbackslash{}")
        .replace("_", r"\_")
        .replace("%", r"\%")
        .replace("&", r"\&")
        .replace("#", r"\#")
    )


def main():
    parser = argparse.ArgumentParser(description="Write proprietary-model inference settings.")
    parser.add_argument("--outdir", type=Path, default=OUT_DIR)
    outdir = parser.parse_args().outdir
    outdir.mkdir(parents=True, exist_ok=True)
    out_csv = outdir / OUT_CSV.name
    out_tex = outdir / OUT_TEX.name

    df = pd.DataFrame(ROWS)

    df.to_csv(out_csv, index=False)

    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{3.5pt}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{llllrl}",
        r"\toprule",
        r"Provider & Model & API identifier & Reasoning/thinking & "
        r"Max tokens & Response interface \\",
        r"\midrule",
    ]

    last_provider = None

    for _, row in df.iterrows():
        if (
            last_provider is not None
            and row["provider"] != last_provider
        ):
            lines.append(r"\midrule")

        lines.append(
            f"{latex_escape(row['provider'])} & "
            f"{latex_escape(row['model'])} & "
            f"\\texttt{{{latex_escape(row['api_id'])}}} & "
            f"{latex_escape(row['reasoning_thinking'])} & "
            f"{row['max_output_tokens']} & "
            f"{latex_escape(row['response_interface'])} \\\\"
        )

        last_provider = row["provider"]

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}%",
        r"}",
        r"\caption{Proprietary-model inference settings for the selected V2 "
        r"forced-choice evaluation. Temperature and top-$p$ were omitted. "
        r"GPT-3.5 Turbo used free text, with all 4,000 responses failing strict A/B validation. "
        r"The other six systems used forced A/B tool calls, a departure from the registered "
        r"ordinary-generation protocol. Claude requests omitted the thinking parameter. "
        r"Omission does not establish that internal reasoning was disabled.}",
        r"\label{tab:frontier-inference-settings}",
        r"\end{table*}",
    ])

    out_tex.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    print("Validation PASS")
    print(f"Rows: {len(df)}")
    print()
    print(df.to_string(index=False))
    print()
    print(f"Wrote: {out_csv}")
    print(f"Wrote: {out_tex}")


if __name__ == "__main__":
    main()
