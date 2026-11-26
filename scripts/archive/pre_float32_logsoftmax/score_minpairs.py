#!/usr/bin/env python3
"""
score_minpairs.py

Compute minimal-pair preferences using continuation log probabilities.

For each JSONL row with fields:
  - pair_id: string
  - prefix: string
  - good: string continuation
  - bad:  string continuation
Optional meta fields (we read them if present):
  - meta.family_id
  - meta.checkpoint
  - meta.trap_status
  - meta.v1_attractor_incongruent
  - meta.v2_attractor_incongruent

We compute (full):
  delta = logP(good | prefix) - logP(bad | prefix)
  correct = 1 if delta > 0 else 0

Optionally (critical region):
  delta_crit = logP_crit(good | prefix) - logP_crit(bad | prefix)
where "crit" sums logprobs only over the first contiguous token span
after the prefix where good/bad tokenizations differ.

Outputs:
  - per-item CSV (always)
  - optional family CSV and JSON summary
"""

import argparse
import csv
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class PairItem:
    pair_id: str
    condition: str
    phenomenon: str
    depth: int
    prefix: str
    good: str
    bad: str
    family_id: str
    checkpoint: str
    trap_status: int  # 0/1
    v1_attractor_incongruent: Optional[int]  # 0/1/None
    v2_attractor_incongruent: Optional[int]  # 0/1/None


def _get_meta(obj: Dict[str, Any]) -> Dict[str, Any]:
    m = obj.get("meta", {})
    return m if isinstance(m, dict) else {}


def _get_trap_status(meta: Dict[str, Any]) -> int:
    if "trap_status" in meta:
        return 1 if bool(meta.get("trap_status")) else 0
    trap = meta.get("trap", {})
    if isinstance(trap, dict) and "trap_status" in trap:
        return 1 if bool(trap.get("trap_status")) else 0
    return 0


def _bool_to_int_or_none(x: Any) -> Optional[int]:
    if x is None:
        return None
    return 1 if bool(x) else 0


def read_jsonl(path: str) -> List[PairItem]:
    items: List[PairItem] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            meta = _get_meta(obj)

            pair_id = str(obj["pair_id"])

            fam = str(meta.get("family_id", ""))
            if not fam:
                if "_" in pair_id:
                    fam = "_".join(pair_id.split("_")[:-1])
                else:
                    fam = pair_id

            checkpoint = str(meta.get("checkpoint", ""))
            trap_status = _get_trap_status(meta)

            v1 = _bool_to_int_or_none(meta.get("v1_attractor_incongruent", None))
            v2 = _bool_to_int_or_none(meta.get("v2_attractor_incongruent", None))

            try:
                items.append(
                    PairItem(
                        pair_id=pair_id,
                        condition=str(obj.get("condition", "")),
                        phenomenon=str(obj.get("phenomenon", "")),
                        depth=int(obj.get("depth", 0)),
                        prefix=str(obj["prefix"]),
                        good=str(obj["good"]),
                        bad=str(obj["bad"]),
                        family_id=fam,
                        checkpoint=checkpoint,
                        trap_status=int(trap_status),
                        v1_attractor_incongruent=v1,
                        v2_attractor_incongruent=v2,
                    )
                )
            except Exception as e:
                raise ValueError(f"Bad row at line {line_no}: {e}\nRow: {obj}") from e

    return items


def load_model(model_name: str, dtype: str, device_map: str):
    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    torch_dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[dtype]

    try:
        model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch_dtype, device_map=device_map, trust_remote_code=True)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch_dtype, device_map=device_map, trust_remote_code=True)

    model.eval()
    return model, tok


def _first_differing_span(g_ids: List[int], b_ids: List[int], prefix_len: int) -> Tuple[int, int]:
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
    end = j
    return (start, end)


def _logprob_batch(
    model,
    tokenizer,
    texts: List[str],
    prefix_lens: List[int],
    device: torch.device,
    spans: Optional[List[Optional[Tuple[int, int]]]] = None,
) -> List[float]:
    enc = tokenizer(
        texts,
        return_tensors="pt",
        add_special_tokens=False,
        padding=True,
    )
    input_ids = enc.input_ids.to(device)
    attention_mask = enc.attention_mask.to(device)

    with torch.no_grad():
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits  # [B, L, V]
    log_probs = torch.log_softmax(logits, dim=-1)

    B, _L = input_ids.shape
    totals: List[float] = []

    for i in range(B):
        seq_len = int(attention_mask[i].sum().item())
        p_len = int(prefix_lens[i])
        if seq_len <= p_len:
            raise ValueError("Continuation produced no additional tokens; add a leading space to continuations.")

        if spans is None or spans[i] is None:
            start_j, end_j = p_len, seq_len
        else:
            start_j, end_j = spans[i]
            start_j = max(start_j, p_len)
            end_j = min(end_j, seq_len)
            if end_j < start_j:
                end_j = start_j

        total = 0.0
        for j in range(start_j, end_j):
            tok_id = int(input_ids[i, j].item())
            pred_pos = j - 1
            total += float(log_probs[i, pred_pos, tok_id].item())
        totals.append(total)

    return totals


def summarize_and_write(
    out_summary: Optional[str],
    out_family: Optional[str],
    scored_rows: List[Dict[str, Any]],
    include_critical: bool,
):
    if not out_summary and not out_family:
        return

    n = len(scored_rows)
    correct_n = sum(int(r["correct"]) for r in scored_rows)
    mean_delta = sum(float(r["delta"]) for r in scored_rows) / max(n, 1)

    if include_critical:
        correct_crit_n = sum(int(r["correct_crit"]) for r in scored_rows)
        mean_delta_crit = sum(float(r["delta_crit"]) for r in scored_rows) / max(n, 1)
    else:
        correct_crit_n = 0
        mean_delta_crit = 0.0

    from collections import defaultdict

    fam_correct = defaultdict(int)
    fam_total = defaultdict(int)
    fam_correct_crit = defaultdict(int)

    for r in scored_rows:
        fam = r["family_id"]
        fam_total[fam] += 1
        fam_correct[fam] += int(r["correct"])
        if include_critical:
            fam_correct_crit[fam] += int(r["correct_crit"])

    fam_ids = list(fam_total.keys())

    fam_all = []
    fam_all_multi = []
    fam_all_crit = []
    fam_all_crit_multi = []

    for fam in fam_ids:
        n_items = fam_total[fam]
        all_correct = 1 if fam_correct[fam] == n_items else 0
        fam_all.append(all_correct)
        if n_items > 1:
            fam_all_multi.append(all_correct)

        if include_critical:
            all_correct_crit = 1 if fam_correct_crit[fam] == n_items else 0
            fam_all_crit.append(all_correct_crit)
            if n_items > 1:
                fam_all_crit_multi.append(all_correct_crit)

    fam_all_acc_all = sum(fam_all) / max(len(fam_all), 1)
    fam_all_acc_multi = sum(fam_all_multi) / max(len(fam_all_multi), 1)

    if include_critical:
        fam_all_acc_all_crit = sum(fam_all_crit) / max(len(fam_all_crit), 1)
        fam_all_acc_multi_crit = sum(fam_all_crit_multi) / max(len(fam_all_crit_multi), 1)
    else:
        fam_all_acc_all_crit = 0.0
        fam_all_acc_multi_crit = 0.0

    if out_family:
        os.makedirs(os.path.dirname(out_family) or ".", exist_ok=True)
        with open(out_family, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if include_critical:
                w.writerow(["family_id", "n_items", "n_correct", "all_correct", "n_correct_crit", "all_correct_crit"])
            else:
                w.writerow(["family_id", "n_items", "n_correct", "all_correct"])

            for fam in sorted(fam_ids):
                n_items = fam_total[fam]
                n_cor = fam_correct[fam]
                all_cor = 1 if n_cor == n_items else 0
                if include_critical:
                    n_cor_c = fam_correct_crit[fam]
                    all_cor_c = 1 if n_cor_c == n_items else 0
                    w.writerow([fam, n_items, n_cor, all_cor, n_cor_c, all_cor_c])
                else:
                    w.writerow([fam, n_items, n_cor, all_cor])

    if out_summary:
        os.makedirs(os.path.dirname(out_summary) or ".", exist_ok=True)
        summary = {
            "n_items": n,
            "item_accuracy": correct_n / max(n, 1),
            "item_mean_delta": mean_delta,
            "n_families": len(fam_ids),
            "family_all_correct_accuracy_all_families": fam_all_acc_all,
            "family_all_correct_accuracy_multi_item_families": fam_all_acc_multi,
        }
        if include_critical:
            summary.update(
                {
                    "item_accuracy_crit": correct_crit_n / max(n, 1),
                    "item_mean_delta_crit": mean_delta_crit,
                    "family_all_correct_accuracy_all_families_crit": fam_all_acc_all_crit,
                    "family_all_correct_accuracy_multi_item_families_crit": fam_all_acc_multi_crit,
                }
            )

        with open(out_summary, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
            f.write("\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="HF model id or local path")
    ap.add_argument("--data", required=True, help="Path to JSONL minimal pairs")
    ap.add_argument("--out", default="results/scores.csv", help="Output CSV path")

    ap.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="bfloat16")
    ap.add_argument("--device-map", default="auto", help='e.g., "auto" or "cpu"')

    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-items", type=int, default=0, help="If >0, only score first N items")
    ap.add_argument("--progress-every", type=int, default=50)
    ap.add_argument("--flush-every", type=int, default=200)

    ap.add_argument("--out-family", default="", help="Optional: family-level CSV")
    ap.add_argument("--out-summary", default="", help="Optional: JSON summary")

    ap.add_argument("--fewshot", default="", help="Optional: path to a text file to prepend to every prefix")

    ap.add_argument(
        "--score-critical",
        action="store_true",
        help="Also score only the first differing token span after the prefix (critical region)",
    )

    args = ap.parse_args()

    items = read_jsonl(args.data)

    # Optional few-shot prefixing: prepend a fixed block to every item's prefix.
    # This is a simple prompt-level manipulation and does not change scoring logic.
    fewshot_text = ""
    if getattr(args, "fewshot", ""):
        with open(args.fewshot, "r", encoding="utf-8") as ff:
            fewshot_text = ff.read().strip()
        if fewshot_text:
            # Ensure a clean boundary between few-shot block and the item prefix.
            fewshot_text = fewshot_text + "\n\n"

    if fewshot_text:
        for it in items:
            it.prefix = fewshot_text + it.prefix

    if args.max_items and args.max_items > 0:
        items = items[: args.max_items]

    model, tok = load_model(args.model, dtype=args.dtype, device_map=args.device_map)

    if args.device_map == "cpu":
        device = torch.device("cpu")
    else:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    scored_rows: List[Dict[str, Any]] = []

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)

        header = [
            "pair_id",
            "family_id",
            "checkpoint",
            "trap_status",
            "v1_attractor_incongruent",
            "v2_attractor_incongruent",
            "condition",
            "phenomenon",
            "depth",
            "prefix",
            "good",
            "bad",
            "logp_good",
            "logp_bad",
            "delta",
            "correct",
        ]
        if args.score_critical:
            header += [
                "logp_good_crit",
                "logp_bad_crit",
                "delta_crit",
                "correct_crit",
                "crit_start",
                "crit_end",
            ]

        w.writerow(header)
        f.flush()

        total_items = len(items)
        correct_n = 0
        correct_crit_n = 0
        written = 0
        bs = max(1, int(args.batch_size))

        for start in range(0, total_items, bs):
            batch = items[start : start + bs]

            prefixes = [it.prefix for it in batch]
            prefix_enc = tok(prefixes, add_special_tokens=False)
            prefix_lens = [len(ids) for ids in prefix_enc["input_ids"]]

            texts_good = [it.prefix + it.good for it in batch]
            texts_bad = [it.prefix + it.bad for it in batch]
            texts = texts_good + texts_bad
            lens2 = prefix_lens + prefix_lens

            totals = _logprob_batch(model, tok, texts, lens2, device)
            B = len(batch)
            logp_good = totals[:B]
            logp_bad = totals[B:]

            if args.score_critical:
                spans_good: List[Optional[Tuple[int, int]]] = []
                spans_bad: List[Optional[Tuple[int, int]]] = []
                crit_starts: List[int] = []
                crit_ends: List[int] = []

                for k, it in enumerate(batch):
                    g_ids = tok(it.prefix + it.good, add_special_tokens=False)["input_ids"]
                    b_ids = tok(it.prefix + it.bad, add_special_tokens=False)["input_ids"]
                    p_len = prefix_lens[k]
                    s, e = _first_differing_span(g_ids, b_ids, p_len)
                    spans_good.append((s, e))
                    spans_bad.append((s, e))
                    crit_starts.append(s)
                    crit_ends.append(e)

                totals_crit = _logprob_batch(model, tok, texts, lens2, device, spans=spans_good + spans_bad)
                logp_good_crit = totals_crit[:B]
                logp_bad_crit = totals_crit[B:]
            else:
                logp_good_crit = []
                logp_bad_crit = []
                crit_starts = []
                crit_ends = []

            for k, it in enumerate(batch):
                lg = float(logp_good[k])
                lb = float(logp_bad[k])
                delta = lg - lb
                correct = 1 if delta > 0 else 0
                correct_n += correct

                row = {
                    "pair_id": it.pair_id,
                    "family_id": it.family_id,
                    "checkpoint": it.checkpoint,
                    "trap_status": it.trap_status,
                    "v1_attractor_incongruent": it.v1_attractor_incongruent,
                    "v2_attractor_incongruent": it.v2_attractor_incongruent,
                    "condition": it.condition,
                    "phenomenon": it.phenomenon,
                    "depth": it.depth,
                    "prefix": it.prefix,
                    "good": it.good,
                    "bad": it.bad,
                    "logp_good": lg,
                    "logp_bad": lb,
                    "delta": delta,
                    "correct": correct,
                }

                out_row = [
                    it.pair_id,
                    it.family_id,
                    it.checkpoint,
                    it.trap_status,
                    it.v1_attractor_incongruent,
                    it.v2_attractor_incongruent,
                    it.condition,
                    it.phenomenon,
                    it.depth,
                    it.prefix,
                    it.good,
                    it.bad,
                    lg,
                    lb,
                    delta,
                    correct,
                ]

                if args.score_critical:
                    lgc = float(logp_good_crit[k])
                    lbc = float(logp_bad_crit[k])
                    delta_c = lgc - lbc
                    correct_c = 1 if delta_c > 0 else 0
                    correct_crit_n += correct_c

                    row.update(
                        {
                            "logp_good_crit": lgc,
                            "logp_bad_crit": lbc,
                            "delta_crit": delta_c,
                            "correct_crit": correct_c,
                            "crit_start": int(crit_starts[k]),
                            "crit_end": int(crit_ends[k]),
                        }
                    )

                    out_row += [
                        lgc,
                        lbc,
                        delta_c,
                        correct_c,
                        int(crit_starts[k]),
                        int(crit_ends[k]),
                    ]

                scored_rows.append(row)
                w.writerow(out_row)

                written += 1

            if args.progress_every and written % int(args.progress_every) == 0:
                acc = correct_n / max(written, 1)
                if args.score_critical:
                    acc_c = correct_crit_n / max(written, 1)
                    print(f"Scored {written}/{total_items} (acc={acc:.3f}, acc_crit={acc_c:.3f})", flush=True)
                else:
                    print(f"Scored {written}/{total_items} (acc={acc:.3f})", flush=True)

            if args.flush_every and written % int(args.flush_every) == 0:
                f.flush()

    acc = correct_n / max(len(items), 1)
    if args.score_critical:
        acc_c = correct_crit_n / max(len(items), 1)
        print(f"Scored {len(items)} items. Accuracy={acc:.3f}  CritAccuracy={acc_c:.3f}. Wrote {args.out}", flush=True)
    else:
        print(f"Scored {len(items)} items. Accuracy={acc:.3f}. Wrote {args.out}", flush=True)

    summarize_and_write(
        args.out_summary or None,
        args.out_family or None,
        scored_rows,
        include_critical=bool(args.score_critical),
    )


if __name__ == "__main__":
    main()
