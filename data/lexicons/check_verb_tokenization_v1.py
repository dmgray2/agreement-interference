#!/usr/bin/env python3
"""
Check which intransitive verbs can be used in a CE0 frame such as
    "The vigaph arrives."  vs  "The vigaph arrive."
without breaking critical-region scoring.

Why this matters
  score_minpairs.py finds the critical region by comparing the good and bad
  token sequences position by position. If the singular has one more token
  than the plural (" admire" + "s" vs " admire"), every later position is
  out of line, and the "critical region" runs to the end of the sentence.
  The tokenization audit found this for admire, criticize and consult in all
  models, and for accuse, blame, praise and interrupt in Pythia.
  A verb is usable only if its singular and plural forms are each ONE token,
  so that the critical region is exactly the verb.

What the script does
  1. Reads the verbs in lexicon_verbs_intransitive.csv, plus a built-in list
     of common intransitive verbs (so there are enough to choose from).
  2. For each of the four core tokenizer families (GPT-2, Pythia/NeoX,
     Llama 3, Qwen2), builds the good and bad sentences after two prefixes,
     one with a real noun and one with a nonce noun, and computes the
     critical span with the same function as score_minpairs.py.
  3. Reports, for screening by hand:
       span_<family>        critical-span length (1 is what we want)
       pass_all             1 if the span is 1 in every family and prefix
       noun_senses          WordNet noun senses of the bare form. A verb that
                            is also a common count noun ("smile", "protest")
                            lets "The vigaph smiles" be read as a plural
                            compound noun, which favors the -s form for a
                            reason that has nothing to do with agreement.
       zipf_sg, zipf_pl     how frequent the two forms are

Output (in the folder you run it in)
  verb_tokenization_check_v1.csv

Usage
  python3 check_verb_tokenization_v1.py
  python3 check_verb_tokenization_v1.py --lexicon lexicon_verbs_intransitive.csv
"""

import argparse
import csv
import os

from transformers import AutoTokenizer
from wordfreq import zipf_frequency

FAMILIES = {
    "gpt2": "openai-community/gpt2",
    "pythia_neox": "EleutherAI/pythia-410m",
    "llama3": "meta-llama/Llama-3.1-70B",
    "qwen2": "Qwen/Qwen2.5-7B",
}

# Prefixes used to build the test sentences: one real noun, one nonce noun.
PREFIXES = ["The reporter", "The vigaph"]

# Common verbs that can be used intransitively, with a regular present tense.
CANDIDATES = [
    "arrive", "smile", "retire", "resign", "panic", "apologize", "disappear",
    "collapse", "hesitate", "protest", "laugh", "cry", "sleep", "wait",
    "sing", "dance", "swim", "run", "walk", "talk", "sit", "stand", "fall",
    "rise", "agree", "complain", "cough", "sneeze", "frown", "blush", "wink",
    "yawn", "nod", "listen", "vote", "travel", "work", "rest", "relax",
    "shout", "scream", "faint", "sigh", "kneel", "pray", "sweat", "shiver",
    "tremble", "recover", "survive", "succeed", "fail", "quit", "struggle",
    "worry", "leave", "return", "die", "speak", "eat", "drink", "think",
    "react", "respond", "remain", "exist", "appear", "vanish", "wake",
    "hide", "jump", "climb", "shrug", "grin", "weep", "mourn", "pause",
    "linger", "wander", "argue", "lie", "sneer", "gasp", "stare", "blink",
]


def third_singular(v):
    """Regular 3rd person singular: wait -> waits, relax -> relaxes, cry -> cries."""
    if v.endswith(("s", "x", "z", "ch", "sh")):
        return v + "es"
    if v.endswith("y") and v[-2] not in "aeiou":
        return v[:-1] + "ies"
    return v + "s"


def first_differing_span(g_ids, b_ids, prefix_len):
    """Copied from score_minpairs.py so the result is identical."""
    g_len = len(g_ids)
    b_len = len(b_ids)
    i = prefix_len
    while i < g_len and i < b_len and g_ids[i] == b_ids[i]:
        i += 1
    if i >= g_len or i >= b_len:
        if i < max(g_len, b_len):
            return (i, min(g_len, b_len))
        return (prefix_len, prefix_len)
    start = i
    j = i
    while j < g_len and j < b_len and g_ids[j] != b_ids[j]:
        j += 1
    return (start, j)


def encode(tok, text):
    return tok(text, add_special_tokens=False)["input_ids"]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lexicon", default="lexicon_verbs_intransitive.csv",
                    help="intransitive verb lexicon (columns lemma, pres_sg, pres_pl)")
    args = ap.parse_args()

    # Verbs to test: the lexicon first, then the built-in candidates.
    verbs = []
    seen = set()
    if os.path.exists(args.lexicon):
        with open(args.lexicon, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                verbs.append((row["lemma"], row["pres_sg"], row["pres_pl"], "lexicon"))
                seen.add(row["lemma"])
    else:
        print(f"(lexicon {args.lexicon} not found: testing built-in candidates only)")
    for v in CANDIDATES:
        if v not in seen:
            verbs.append((v, third_singular(v), v, "candidate"))
            seen.add(v)

    try:
        from nltk.corpus import wordnet as wn
        wn.synsets("dog")
    except (ImportError, LookupError):
        wn = None
        print("(WordNet not available: noun_senses left blank)")

    toks = {}
    for key, name in FAMILIES.items():
        print(f"loading {name}")
        toks[key] = AutoTokenizer.from_pretrained(name)

    rows = []
    for lemma, sg, pl, source in verbs:
        row = {"lemma": lemma, "pres_sg": sg, "pres_pl": pl, "source": source}
        ok_all = True
        for key, tok in toks.items():
            spans = []
            for prefix in PREFIXES:
                p_len = len(encode(tok, prefix))
                # The span is the same whichever form counts as "good",
                # because the comparison is position by position.
                g = encode(tok, prefix + " " + sg + ".")
                b = encode(tok, prefix + " " + pl + ".")
                start, end = first_differing_span(g, b, p_len)
                spans.append(end - start)
            span = max(spans)
            row[f"span_{key}"] = span
            row[f"tokens_{key}"] = (
                "|".join(tok.decode([i]) for i in encode(tok, " " + sg)) + "  /  "
                + "|".join(tok.decode([i]) for i in encode(tok, " " + pl)))
            ok_all = ok_all and span == 1
        row["pass_all"] = int(ok_all)
        row["noun_senses"] = len(wn.synsets(lemma, pos=wn.NOUN)) if wn else ""
        row["zipf_sg"] = round(zipf_frequency(sg, "en"), 2)
        row["zipf_pl"] = round(zipf_frequency(pl, "en"), 2)
        rows.append(row)

    # Passing verbs first; among them, fewest noun senses first.
    rows.sort(key=lambda r: (-r["pass_all"],
                             r["noun_senses"] if r["noun_senses"] != "" else 0,
                             r["lemma"]))
    fields = (["lemma", "pres_sg", "pres_pl", "source", "pass_all", "noun_senses",
               "zipf_sg", "zipf_pl"]
              + [f"span_{k}" for k in FAMILIES] + [f"tokens_{k}" for k in FAMILIES])
    with open("verb_tokenization_check_v1.csv", "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=fields)
        wr.writeheader()
        wr.writerows(rows)

    n_pass = sum(r["pass_all"] for r in rows)
    print(f"\n{n_pass} of {len(rows)} verbs have a one-token critical span in all four families")
    print("Passing verbs with no noun senses:",
          ", ".join(r["lemma"] for r in rows if r["pass_all"] and r["noun_senses"] == 0) or "none")
    print("wrote verb_tokenization_check_v1.csv")


if __name__ == "__main__":
    main()
