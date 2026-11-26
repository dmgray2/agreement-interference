import csv
from transformers import AutoTokenizer

MODELS = [
    "openai-community/gpt2",
    "EleutherAI/pythia-410m",
    "meta-llama/Llama-3.2-1B",
    "meta-llama/Llama-3.2-3B",
    "meta-llama/Llama-3.1-70B",
    "Qwen/Qwen2.5-1.5B",
    "Qwen/Qwen2.5-3B",
    "Qwen/Qwen2.5-7B",
    "Qwen/Qwen2.5-14B",
    "Qwen/Qwen2.5-32B",
    "Qwen/Qwen2.5-72B",
]

NOUN_LEXICON = "lexicon_nouns.csv"
VERB_LEXICON = "lexicon_verbs_transitive.csv"
DETAIL_CSV = "lexicon_tokenization_detail.csv"
SUMMARY_CSV = "lexicon_tokenization_summary.csv"

def n_tokens(tokenizer, word):
    # Leading space matches mid-sentence occurrence
    return len(tokenizer.tokenize(" " + word))

def main():
    print("Loading tokenizers...")
    tokenizers = {}
    for model_name in MODELS:
        print(f"  {model_name}")
        tokenizers[model_name] = AutoTokenizer.from_pretrained(model_name)

    items = []
    with open(NOUN_LEXICON, newline="") as f:
        for row in csv.DictReader(f):
            items.append({
                "source_type": "noun",
                "source_id": row["noun_id"],
                "form_type": "sg",
                "word": row["sg"],
            })
            items.append({
                "source_type": "noun",
                "source_id": row["noun_id"],
                "form_type": "pl",
                "word": row["pl"],
            })

    with open(VERB_LEXICON, newline="") as f:
        for row in csv.DictReader(f):
            items.append({
                "source_type": "verb",
                "source_id": row["verb_id"],
                "form_type": "pres_sg",
                "word": row["pres_sg"],
            })
            items.append({
                "source_type": "verb",
                "source_id": row["verb_id"],
                "form_type": "pres_pl",
                "word": row["pres_pl"],
            })

    detail_rows = []
    summary_rows = []

    for item in items:
        word = item["word"]
        counts = {}
        for model_name, tok in tokenizers.items():
            c = n_tokens(tok, word)
            counts[model_name] = c
            detail_rows.append({
                "source_type": item["source_type"],
                "source_id": item["source_id"],
                "form_type": item["form_type"],
                "word": word,
                "model": model_name,
                "n_tokens": c,
            })

        summary_rows.append({
            "source_type": item["source_type"],
            "source_id": item["source_id"],
            "form_type": item["form_type"],
            "word": word,
            "min_tokens": min(counts.values()),
            "max_tokens": max(counts.values()),
            "mean_tokens": round(sum(counts.values()) / len(counts), 2),
            "all_single_token": all(v == 1 for v in counts.values()),
        })

    with open(DETAIL_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(detail_rows[0].keys()))
        writer.writeheader()
        writer.writerows(detail_rows)

    with open(SUMMARY_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    n_single = sum(1 for r in summary_rows if r["all_single_token"])
    print(f"Wrote {len(detail_rows)} detail rows to {DETAIL_CSV}")
    print(f"Wrote {len(summary_rows)} summary rows to {SUMMARY_CSV}")
    print(f"{n_single} / {len(summary_rows)} real-word forms are single-token across all 11 models")
    print(f"Mean tokens across all forms/models: {round(sum(r['mean_tokens'] for r in summary_rows) / len(summary_rows), 2)}")

if __name__ == "__main__":
    main()
