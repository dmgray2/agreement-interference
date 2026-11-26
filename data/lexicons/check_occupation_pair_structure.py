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

INPUT_CSV = "nonce_occupations.csv"
DETAIL_CSV = "occupation_pair_structure_detail.csv"
SUMMARY_CSV = "occupation_pair_structure_summary.csv"
MAX_TOKEN_COLS = 6
GOOD_QUALITIES = {"identical_prefix_same_count", "identical_prefix_extra_token"}

def tokenize(tokenizer, word):
    return tokenizer.tokenize(" " + word)

def classify(tokens_a, tokens_b):
    n_a, n_b = len(tokens_a), len(tokens_b)
    shared = 0
    for ta, tb in zip(tokens_a, tokens_b):
        if ta == tb:
            shared += 1
        else:
            break
    if n_a == n_b:
        if shared == n_a - 1:
            quality = "identical_prefix_same_count"
        elif shared == n_a:
            quality = "identical_tokens"
        else:
            quality = "partial_or_no_prefix_same_count"
    else:
        shorter = min(n_a, n_b)
        if shared == shorter:
            quality = "identical_prefix_extra_token"
        else:
            quality = "mismatched_count_and_prefix"
    return shared, quality

def token_cols(tokens, prefix):
    cols = {}
    overflow = len(tokens) > MAX_TOKEN_COLS
    for i in range(MAX_TOKEN_COLS):
        cols[f"{prefix}_token_{i+1}"] = tokens[i] if i < len(tokens) else ""
    if overflow:
        cols[f"{prefix}_token_{MAX_TOKEN_COLS}"] += " [+more]"
    return cols

def main():
    print("Loading tokenizers...")
    tokenizers = {}
    for model_name in MODELS:
        print(f"  {model_name}")
        tokenizers[model_name] = AutoTokenizer.from_pretrained(model_name)

    with open(INPUT_CSV, newline="") as f:
        candidates = list(csv.DictReader(f))

    detail_rows = []
    per_candidate_quality = {}

    for idx, row in enumerate(candidates):
        sg = row["nonce_singular"]
        pl = row["nonce_plural"]
        per_candidate_quality[idx] = []

        for model_name, tok in tokenizers.items():
            sg_tokens = tokenize(tok, sg)
            pl_tokens = tokenize(tok, pl)
            shared, quality = classify(sg_tokens, pl_tokens)

            detail_row = {
                "source_word": row["source_word"],
                "head_word": row["head_word"],
                "nonce_singular": sg,
                "nonce_plural": pl,
                "model": model_name,
                "singular_n_tokens": len(sg_tokens),
                "plural_n_tokens": len(pl_tokens),
                "shared_prefix_len": shared,
                "match_quality": quality,
            }
            detail_row.update(token_cols(sg_tokens, "singular"))
            detail_row.update(token_cols(pl_tokens, "plural"))
            detail_rows.append(detail_row)
            per_candidate_quality[idx].append(quality in GOOD_QUALITIES)

    fieldnames = list(detail_rows[0].keys())
    with open(DETAIL_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(detail_rows)

    summary_rows = []
    for idx, row in enumerate(candidates):
        good_flags = per_candidate_quality[idx]
        summary_rows.append({
            "source_word": row["source_word"],
            "head_word": row["head_word"],
            "nonce_singular": row["nonce_singular"],
            "nonce_plural": row["nonce_plural"],
            "n_models_good": sum(good_flags),
            "n_models_total": len(good_flags),
            "all_models_good": all(good_flags),
        })

    with open(SUMMARY_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    n_perfect = sum(1 for r in summary_rows if r["all_models_good"])
    print(f"Wrote {len(detail_rows)} detail rows to {DETAIL_CSV}")
    print(f"Wrote {len(summary_rows)} summary rows to {SUMMARY_CSV}")
    print(f"{n_perfect} / {len(summary_rows)} candidates are clean (good structure) across all 11 models")

if __name__ == "__main__":
    main()
