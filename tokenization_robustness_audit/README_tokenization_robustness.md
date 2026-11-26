## AI Assistance Disclosure

The authors used ChatGPT for coding assistance, debugging, and checklist-style review of materials. AI tools were also used to assist with formatting the README, pipeline, and manifest documents. The authors made all experimental design decisions, conducted and checked the statistical analyses, verified the sources, reviewed the code, and approved the final manuscript. The authors take full responsibility for the paper, code, data, and results.

# CE2 tokenization-robustness audit

## What this audit is

This directory documents the post hoc tokenization robustness checks on the
V2 results in `master_scores_full.csv`. 

The most important fact is:

**No model was rerun, and no new language-model probabilities were computed.**

The analysis reads the existing score columns produced during the original
model runs. The original scorer had already used each model's own tokenizer
and recorded `crit_start`, `crit_end`, `delta_crit`, and `correct_crit`.
This audit filters and reanalyzes those stored results.

Calling this procedure "rescoring" would be inaccurate. It is a
**restricted-subset reanalysis of previously scored items**.

## Source of the critical-span information

The original `score_minpairs.py`:

1. tokenized the grammatical and ungrammatical full strings with the
   tokenizer belonging to the model being scored;
2. found the first contiguous token span at which those two token sequences
   differed;
3. stored the boundaries as `crit_start` and `crit_end`;
4. summed log probabilities over that span and stored the resulting
   `correct_crit`.

Therefore, critical-span length is model-specific:

```text
critical_span_length = crit_end - crit_start
```

The same sentence can have a one-token critical span for one model and a
multi-token span for another.

## Analysis sets

### 1. Full V2 analysis

- 2,000 V2 sentence families.
- All 11 models.
- No family excluded for tokenization.
- This reconstructs an unfiltered comparison using the existing V2 scores.

### 2. Common all-model single-token analysis

A family is included only when:

```text
crit_end - crit_start == 1
```

for **every one of the 11 models**.

- Included: 1,094 families.
- Excluded from this robustness subset: 906 families.
- Models excluded: none.

The 906 families remain in the original master file and original analysis.
They are excluded only from this stricter derived robustness subset.

### 3. Common non-Pythia single-token analysis

Pythia-410M has a substantially smaller one-token set. A second robustness
analysis therefore defines the common set over the other ten models.

- Included: 1,636 families.
- Excluded from this robustness subset: 364 families.
- Pythia-410M is omitted from this particular model comparison.
- Pythia remains in the original 2,000-item analysis and in the 1,094-item
  all-model analysis.

This is not a proposal to remove Pythia from the paper's main experiment.
It is a sensitivity analysis showing what happens when the common-item
criterion is not determined by Pythia's tokenizer.

## What was excluded—and what was not

Excluded from a strict common-item robustness subset:

- families whose critical span was not exactly one token for at least one
  tokenizer in the relevant model set;
- Pythia itself, but only in the explicitly labeled non-Pythia analysis.

Not excluded:

- no item based on whether a model answered correctly;
- no item based on effect size;
- no item based on model size;
- no NP-number pattern;
- no congruent or incongruent condition as a class;
- no model from the main 2,000-item analysis;
- no model from the 1,094-item all-model analysis;
- no V2 row was deleted or altered in `master_scores_full.csv`;
- no log probabilities, deltas, or correctness values were changed.

V3 and COPULA are not part of this check because the reviewer concern and the
main attractor result concern V2. They were not deleted from the source data.

## Statistical analysis

For every model in a given subset:

```text
V2 drop =
    mean(correct_crit | NP3 matches NP2)
    -
    mean(correct_crit | NP3 mismatches NP2)
```

The script reports:

- congruent and incongruent accuracy;
- the V2 drop;
- a 95% percentile family bootstrap confidence interval;
- an exact one-sided conditional label-permutation p-value;
- a 10,000-draw Monte Carlo label-permutation p-value matching the logic of `stats_master.py`;
- Holm-adjusted versions of both p-values across models within that subset.

The default bootstrap uses 20,000 family resamples. The random seed is fixed
and recorded.

The exact p-value holds fixed the total number of correct responses and the
sizes of the congruent and incongruent groups. Under permutation of the
congruence labels, the number of correct responses assigned to the congruent
group follows a hypergeometric distribution.

This exact test is why very small p-values can be reported; it is not limited
by a Monte Carlo minimum such as 1/10,001. The output also includes the
10,000-draw Monte Carlo p-value so the robustness analysis can be compared
directly with the main analysis method.

## Files

- `analyze_common_token_items.py`
  Reproducible analysis script.

- `run_summary.csv`
  Input checksum, model set, subset sizes, seed, and analysis settings.

- `model_filter_counts.csv`
  Per-model V2 total, one-token count, excluded count, and retention rate.

- `family_filter_status.csv`
  Complete 2,000-family audit table. It gives every model's span length and
  single-token status for every family, plus membership in each subset.

- `included_families_common_all_models.csv`
  Exact 1,094 family IDs used in the all-model robustness analysis.

- `excluded_from_common_all_models.csv`
  Exact 906 family IDs omitted from that subset and the tokenizer(s) causing
  the omission.

- `included_families_common_non_pythia.csv`
  Exact 1,636 family IDs used in the non-Pythia robustness analysis.

- `excluded_from_common_non_pythia.csv`
  Exact 364 family IDs omitted from that subset.

- `subset_composition_counts.csv`
  Counts by NP-number pattern and V2 attractor congruence for all three sets.

- `results_full_v2_2000.csv`
  Recomputed statistics on all V2 families using the audit script's methods.

- `results_common_all_models.csv`
  Results for the 1,094-family all-model common set.

- `results_common_non_pythia.csv`
  Results for the 1,636-family common set across the ten non-Pythia models.

- `MANIFEST_SHA256.txt`
  SHA-256 checksums for the input and the original audit outputs. The baseline manifest predates these camera-ready documentation and derived-figure additions.

## Reproduction command

From a directory containing the master score file:

```bash
python3 analyze_common_token_items.py \
  --master /path/to/master_scores_full.csv \
  --out-dir /path/to/tokenization_robustness \
  --excluded-model pythia410 \
  --bootstrap 20000 \
  --permutations 10000 \
  --seed 20260712
```

The script requires Python, NumPy, and SciPy.

## Relation to the earlier ad hoc calculation

The initial figures were calculated interactively from the uploaded
`master_scores_full.csv`. No standalone script was saved at that point.
That was inadequate as a permanent research record.

This directory converts the same logic into a checked, rerunnable script and
preserves the exact included and excluded family IDs. The audit script also
corrects a presentational ambiguity in the earlier explanation by explicitly
distinguishing:

- model-specific one-token eligibility;
- the intersection common to all 11 models;
- the intersection common to the ten non-Pythia models.
## Camera-ready appendix generation

The audit input SHA-256 in `run_summary.csv` matches the repository's `data/scores/master_score_lists/master_scores_full.csv`. To generate the appendix plot and table from the archived CSV results, run from the repository root:

```bash
python3 scripts/05_make_figures_tables/plot_master.py \
  --camera-ready-only --outdir "EACL paper/figures"
```

The outputs are `EACL paper/figures/tokenization_robustness_v2.pdf`, its PNG preview, and `EACL paper/tables/table_tokenization_robustness.tex`. The figure and table use the archived point estimates and 95% intervals from 20,000 resamples.

This appendix uses NP3–NP2 congruence. The registered forced-choice H1–H3 analyses use the global NP1/NP3–NP2 congruence contrast. The two definitions should be labeled explicitly in the manuscript.
