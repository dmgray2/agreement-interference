# Data README

This folder contains the lexicons, generated stimuli, model score files, and summary tables used in the analysis.

## Folder layout

```text
data/lexicons/
data/stimuli/
data/scores/model_individual_level/
data/scores/model_family_level/
data/scores/master_score_lists/
data/scores/summaries/
```

## Lexicons

```text
data/lexicons/lexicon_nouns.csv
data/lexicons/lexicon_verbs_transitive.csv
```

These files provide the vocabulary used by the stimulus generator.

## Stimuli

Main stimulus file:

```text
data/stimuli/minpairs.ce2fam.generated.v0.7_nf2000.jsonl
```

Audit files:

```text
data/stimuli/minpairs.ce2fam.audit.v0.7_nf2000.csv
data/stimuli/minpairs.ce2fam.audit.v0.7_nf2000.per_family.csv
```

The released benchmark should contain:

```text
2000 sentence families
6000 benchmark items
2000 COPULA items
2000 V2 items
2000 V3 items
```

The released benchmark should not contain `SANITY` or `TEST` rows.

Files with `sanity` or `backup` in the name are not needed for the main analysis. They should be excluded from the anonymous review package unless there is a specific reason to keep them.

## Model score files

Item-level model score files are in:

```text
data/scores/model_individual_level/
```

Family-level model score files are in:

```text
data/scores/model_family_level/
```

Each scored model has one item-level CSV and one family-level CSV.

The important item-level columns include:

```text
model
pair_id
family_id
checkpoint
correct
correct_crit
np1_num
np2_num
np3_num
np_pattern
v1_attractor_incongruent
v2_attractor_incongruent
np2_np1_incongruent
regime_v1v2
```

`correct` is full-continuation accuracy. `correct_crit` is critical-region accuracy.

## Master score files

```text
data/scores/master_score_lists/master_scores_full.csv
data/scores/master_score_lists/master_scores_family.csv
```

Expected row counts:

```text
master_scores_full.csv:   66000
master_scores_family.csv: 22000
```

`master_scores_full.csv` is the main input for most analysis scripts.

## Summary files

```text
data/scores/summaries/pattern_breakdown.csv
data/scores/summaries/regime_breakdown.csv
data/scores/summaries/stats_summary.csv
data/scores/summaries/regime_contrasts.csv
data/scores/summaries/pattern_model_pvalues.csv
```

Optional mixed-effects outputs:

```text
data/scores/summaries/r_checkpoint_means.csv
data/scores/summaries/r_checkpoint_per_model.csv
data/scores/summaries/r_checkpoint_interaction.txt
```

## Pattern and regime notation

`np_pattern` records the number of NP1, NP2, and NP3:

```text
S = singular
P = plural
```

For example:

```text
SSP = NP1 singular, NP2 singular, NP3 plural
PSS = NP1 plural, NP2 singular, NP3 singular
```

The regime columns are:

```text
v1_attractor_incongruent = 1 if NP3 differs from NP1
v2_attractor_incongruent = 1 if NP3 differs from NP2
np2_np1_incongruent      = 1 if NP2 differs from NP1
regime_v1v2              = (v1_attractor_incongruent, v2_attractor_incongruent)
```

For V2, `np2_np1_incongruent` is important because NP2 is the V2 controller and NP1 can also interfere.
## Forced-choice V2 data

The prompt file is `data/stimuli/minpairs.ce2.v2.forced_choice.counterbalanced.v1.jsonl`. The consolidated paired file is `results/final/master_forced_choice_scores_v1.csv`, with 2,000 rows for each of 18 models. All 36,000 selected pairs have both orders and no technical missingness.

Read `response_interface`, both order-validity fields, `technical_missing`, and `counterbalanced_accuracy` together. Under the registered rule, substantive invalid answers receive zero. GPT-3.5 has 4,000 such responses. The six later proprietary systems used forced tool calls.

Critical-region and forced-choice scores represent different response procedures. The critical-region master contains the original 11 models. The standalone 405B run, nonce studies, CE0 studies, and CE3 studies are additional research material and need their own explicit input selection for any analysis.

## Tokenization subset data

The root `tokenization_robustness_audit/` directory contains the original script, exact retained/excluded family lists, results, and input checksum. Full V2 uses 2,000 families. The all-model single-token subset uses 1,094 families; the ten-model subset excluding Pythia uses 1,636. Eligibility concerns the recorded critical span. The forced-choice A/B token audit in `scripts/audit_open_weight_answer_tokens.py` checks answer labels separately.


## Camera-ready data scope

The release keeps the original 11-model CE2 master, the selected paired V2
forced-choice master, and the open-weight prompt-level bfloat16 master.
The prompt-level master retains the ties needed for Appendix F. Keep
individual selected paired inputs for reconstructing the paired master.
The nonce, CE0, CE3 and standalone 405B outputs are archived separately.
See the root release list for the exact public data files.
