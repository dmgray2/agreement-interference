#!/usr/bin/env python3
"""
gen_minpairs_ce2_family_v0.7.py

CE2-only "family scoring" dataset:
- For each family, generate 3 minimal pairs (checkpoints):
  * CE2_V3      : vary V3 form to match/mismatch NP3
  * CE2_V2      : vary V2 form to match/mismatch NP2
  * CE2_COPULA  : vary copula is/are to match/mismatch NP1

Bare CE2 surface form (no "that"):
  "The N1 the N2 the N3 V3 V2" + " is/are late."

Noun-number sampling in v0.7:
- The singular and plural NP1 counts differ by at most one.
- NP2 and NP3 are sampled independently. The eight patterns need not
  have equal counts.
- The legacy trap_status flag is true when NP3 differs from NP1.

Output JSONL items include:
pair_id, condition, phenomenon, depth, prefix, good, bad, meta

pair_id encodes family + checkpoint so we can aggregate without needing meta in the scores CSV:
  ce2fam_00001_V3
  ce2fam_00001_V2
  ce2fam_00001_COPULA
"""

import argparse
import csv
import json
import os
import random
from typing import Dict, List

# ---------- Lexicon loaders ----------

def load_csv(path: str) -> List[Dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append(row)
    return rows

def noun_form(noun_row: Dict, num: str) -> str:
    return noun_row["sg"] if num == "sg" else noun_row["pl"]

def verb_form(v_row: Dict, num: str) -> str:
    return v_row["pres_sg"] if num == "sg" else v_row["pres_pl"]

def copula(num: str) -> str:
    return "is" if num == "sg" else "are"

def opposite(num: str) -> str:
    return "pl" if num == "sg" else "sg"
    
NUMS = ["sg", "pl"]

def choose_distinct(items: List[Dict], k: int, rng: random.Random) -> List[Dict]:
    if len(items) < k:
        raise ValueError(f"Need {k} distinct items but only have {len(items)}")
    return rng.sample(items, k)

def ensure_dir(path: str):
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)

# ---------- CE2 prefix builder (no "that") ----------

def build_ce2_prefix(n1, n2, n3, v3_form: str, v2_form: str) -> str:
    # "The N1 the N2 the N3 V3 V2"
    return f"The {n1} the {n2} the {n3} {v3_form} {v2_form}"

# ---------- Generation ----------

def gen_ce2_families(
    n_families: int,
    nouns: List[Dict],
    vt: List[Dict],
    rng: random.Random,
) -> List[Dict]:
    rows: List[Dict] = []

    # balance NP1 number
    half = n_families // 2
    np1_nums = (["sg"] * half) + (["pl"] * (n_families - half))
    rng.shuffle(np1_nums)

    # Keep the legacy shuffle to preserve the seeded sequence of draws.
    traps = ([0] * half) + ([1] * (n_families - half))
    rng.shuffle(traps)

    for i in range(n_families):
        fam_idx = i + 1
        family_id = f"ce2fam_{fam_idx:05d}"

        num1 = np1_nums[i]          # NP1 controller
        trap = traps[i]             # legacy assignment overwritten below
        # Sample NP2 and NP3 independently.
        num2 = rng.choice(NUMS)
        num3 = rng.choice(NUMS)
        # attractor incongruence flags (NP3 is the local attractor near V1 and V2 in this template)
        v1_attractor_incongruent = (num3 != num1)  # V1/COPULA controller is NP1
        v2_attractor_incongruent = (num3 != num2)  # V2 controller is NP2
        # legacy alias for backward compatibility with older code/plots
        trap = 1 if v1_attractor_incongruent else 0
        # sample lexicon
        n1r, n2r, n3r = choose_distinct(nouns, 3, rng)
        v2r, v3r = choose_distinct(vt, 2, rng)

        n1 = noun_form(n1r, num1)
        n2 = noun_form(n2r, num2)
        n3 = noun_form(n3r, num3)

        # "correct" forms for non-target verbs/cops we hold fixed in each checkpoint item
        v2_correct = verb_form(v2r, num2)
        v3_correct = verb_form(v3r, num3)
        cop_correct = copula(num1)

        # -------- Checkpoint 1: V3 agreement with NP3 --------
        # Prefix holds V2 + copula correct; vary V3 only
        prefix_v3 = build_ce2_prefix(n1, n2, n3, v3_form="", v2_form=v2_correct).strip()
        # We'll include V3 in good/bad continuation so ONLY V3 differs.
        good_v3 = f" {verb_form(v3r, num3)}{('' if prefix_v3.endswith(n3) else '')} {v2_correct} {cop_correct} late."
        bad_v3  = f" {verb_form(v3r, opposite(num3))} {v2_correct} {cop_correct} late."

        rows.append({
            "pair_id": f"{family_id}_V3",
            "condition": "CE2_V3",
            "phenomenon": "AGREEMENT",
            "depth": 2,
            "prefix": f"The {n1} the {n2} the {n3}",  # stop right before V3
            "good": f" {verb_form(v3r, num3)} {v2_correct} {cop_correct} late.",
            "bad":  f" {verb_form(v3r, opposite(num3))} {v2_correct} {cop_correct} late.",
            "meta": {
                "family_id": family_id,
                "checkpoint": "V3",
                "trap_status": bool(trap),  # legacy = v1_attractor_incongruent
                "v1_attractor_incongruent": bool(v1_attractor_incongruent),
                "v2_attractor_incongruent": bool(v2_attractor_incongruent),
                "nums": {"NP1": num1, "NP2": num2, "NP3": num3},
                "lex": {
                    "NP1": n1r["noun_id"], "NP2": n2r["noun_id"], "NP3": n3r["noun_id"],
                    "V2": v2r["verb_id"], "V3": v3r["verb_id"],
                }
            }
        })

        # -------- Checkpoint 2: V2 agreement with NP2 --------
        # Hold V3 + copula correct; vary V2 only
        rows.append({
            "pair_id": f"{family_id}_V2",
            "condition": "CE2_V2",
            "phenomenon": "AGREEMENT",
            "depth": 2,
            "prefix": f"The {n1} the {n2} the {n3} {v3_correct}",  # stop right before V2
            "good": f" {verb_form(v2r, num2)} {cop_correct} late.",
            "bad":  f" {verb_form(v2r, opposite(num2))} {cop_correct} late.",
            "meta": {
                "family_id": family_id,
                "checkpoint": "V2",
                "trap_status": bool(trap),  # legacy = v1_attractor_incongruent
                "v1_attractor_incongruent": bool(v1_attractor_incongruent),
                "v2_attractor_incongruent": bool(v2_attractor_incongruent),
                "nums": {"NP1": num1, "NP2": num2, "NP3": num3},
                "lex": {
                    "NP1": n1r["noun_id"], "NP2": n2r["noun_id"], "NP3": n3r["noun_id"],
                    "V2": v2r["verb_id"], "V3": v3r["verb_id"],
                }
            }
        })

        # -------- Checkpoint 3: COPULA agreement with NP1 --------
        # Hold V3 + V2 correct; vary copula only
        rows.append({
            "pair_id": f"{family_id}_COPULA",
            "condition": "CE2_COPULA",
            "phenomenon": "AGREEMENT",
            "depth": 2,
            "prefix": f"The {n1} the {n2} the {n3} {v3_correct} {v2_correct}",
            "good": f" {copula(num1)} late.",
            "bad":  f" {copula(opposite(num1))} late.",
            "meta": {
                "family_id": family_id,
                "checkpoint": "COPULA",
                "trap_status": bool(trap),  # legacy = v1_attractor_incongruent
                "v1_attractor_incongruent": bool(v1_attractor_incongruent),
                "v2_attractor_incongruent": bool(v2_attractor_incongruent),
                "nums": {"NP1": num1, "NP2": num2, "NP3": num3},
                "lex": {
                    "NP1": n1r["noun_id"], "NP2": n2r["noun_id"], "NP3": n3r["noun_id"],
                    "V2": v2r["verb_id"], "V3": v3r["verb_id"],
                }
            }
        })

    return rows

def gen_sanity() -> List[Dict]:
    return [
        {
            "pair_id": "sanity_00001",
            "condition": "SANITY",
            "phenomenon": "AGREEMENT",
            "depth": 0,
            "prefix": "The dogs",
            "good": " are late.",
            "bad":  " is late.",
            "meta": {"family_id": "sanity_00001", "checkpoint": "COPULA", "trap_status": False},
        },
        {
            "pair_id": "sanity_00002",
            "condition": "SANITY",
            "phenomenon": "AGREEMENT",
            "depth": 0,
            "prefix": "The dog",
            "good": " is late.",
            "bad":  " are late.",
            "meta": {"family_id": "sanity_00002", "checkpoint": "COPULA", "trap_status": False},
        },
    ]

def write_jsonl(rows: List[Dict], out_path: str):
    ensure_dir(out_path)
    with open(out_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def write_audit(rows: List[Dict], out_csv: str):
    ensure_dir(out_csv)
    from collections import Counter
    c = Counter()
    for r in rows:
        c[r["condition"]] += 1
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["condition", "count"])
        for k, v in sorted(c.items()):
            w.writerow([k, v])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--n_families", type=int, default=500)

    ap.add_argument("--lex_nouns", default="data/lexicon_nouns.csv")
    ap.add_argument("--lex_vt", default="data/lexicon_verbs_transitive.csv")

    ap.add_argument("--out_jsonl", default="data/minpairs.ce2fam.generated.v0.7.jsonl")
    ap.add_argument("--out_audit", default="data/minpairs.ce2fam.audit.v0.7.csv")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    nouns = load_csv(args.lex_nouns)
    vt = load_csv(args.lex_vt)

    rows = gen_ce2_families(args.n_families, nouns, vt, rng)
    rows += gen_sanity()
    rng.shuffle(rows)

    write_jsonl(rows, args.out_jsonl)
    write_audit(rows, args.out_audit)

    print(f"Wrote JSONL: {args.out_jsonl}")
    print(f"Wrote AUDIT: {args.out_audit}")
    print(f"Total rows: {len(rows)}  (families={args.n_families}, checkpoints=3)")

if __name__ == "__main__":
    main()
