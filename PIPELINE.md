# Pipeline

This document explains how the repository files fit together and how to reproduce the analysis. I take these to be low novelty texts and thus we used GPT to generate the pipeline and all readme's in the repository.  They have been verified by the author. All design ideas, experiments, and analysis are the author's own.

The project tests language models on agreement dependencies in doubly center-embedded sentences. The released benchmark contains:

```text
2000 sentence families
3 checkpoints per family: V3, V2, COPULA
6000 benchmark items total
11 scored models
```

Each family has one item for each checkpoint. The main reported measure is **critical-region accuracy**. Critical-region scoring compares the model's probability for the grammatical and ungrammatical agreement form only at the first agreement-critical region.

## Repository layout

```text
data/lexicons/                         input lexicons for stimulus generation
data/stimuli/                          generated benchmark stimuli and audit files
data/scores/model_individual_level/    item-level model scores
data/scores/model_family_level/        family-level model scores
data/scores/master_score_lists/        consolidated score files
data/scores/summaries/                 analysis summaries
scripts/01_generate_dataset/           stimulus-generation script
scripts/02_score_models/               model-scoring script and Slurm template
scripts/03_consolidate_scores/         score consolidation
scripts/04_analyze_scores/             statistical analyses
scripts/05_make_figures_tables/        figure and table generation
paper/figures/                         generated figures
paper/tables/                          generated LaTeX tables
environment/01_scoring_environment/    GPU scoring environment records
environment/02_analysis_environment/   local analysis environment records
```

The numbered steps below cover the original critical-region analysis. The `paper/` output paths are created when the original generators run. Camera-ready commands and forced-choice result paths appear at the end of this guide.

## 1. Generate the stimulus dataset

Main inputs:

```text
data/lexicons/lexicon_nouns.csv
data/lexicons/lexicon_verbs_transitive.csv
```

The generator loads the noun and transitive-verb lexicons. The COPULA checkpoint varies the fixed forms `is` and `are` in the ending `late`.

Main script:

```text
scripts/01_generate_dataset/gen_minpairs_ce2_family_v0.7.py
```

For replication, use the archived 6,000-item JSONL listed below. It contains
the exact stimuli scored in the paper. The original invocation that produced
this archive has not been verified. Running the generator creates a sample
from the same design using the supplied lexicons and seed.

Archived benchmark files:

```text
data/stimuli/minpairs.ce2fam.generated.v0.7_nf2000.jsonl
data/stimuli/minpairs.ce2fam.audit.v0.7_nf2000.csv
data/stimuli/minpairs.ce2fam.audit.v0.7_nf2000.per_family.csv
```

Expected benchmark counts:

```text
COPULA: 2000
V2:     2000
V3:     2000
Total:  6000
```

There should be no `SANITY` or `TEST` rows in the released benchmark file.

## 2. Score each model

Main script:

```text
scripts/02_score_models/score_minpairs.py
```

Main input:

```text
data/stimuli/minpairs.ce2fam.generated.v0.7_nf2000.jsonl
```

Main output folders:

```text
data/scores/model_individual_level/
data/scores/model_family_level/
```

The important item-level columns include:

```text
model
family_id
pair_id
checkpoint
condition
correct
correct_crit
```

`correct` is full-continuation accuracy. `correct_crit` is critical-region accuracy. The main paper uses `correct_crit`.

The generic DeltaAI template is:

```text
scripts/02_score_models/slurm_templates/run_full_ce2fam_v0.7_nf2000_crit.deltaai.template.sbatch
```

The template uses placeholders for accounts, model paths, and repository paths.

## 3. Consolidate model scores and add row metadata

Main script:

```text
scripts/03_consolidate_scores/consolidate_scores.py
```

Main inputs:

```text
data/scores/model_individual_level/
data/scores/model_family_level/
data/stimuli/minpairs.ce2fam.generated.v0.7_nf2000.jsonl
```

Main outputs:

```text
data/scores/master_score_lists/master_scores_full.csv
data/scores/master_score_lists/master_scores_family.csv
```

The script removes non-benchmark rows if present and adds metadata used by later analyses:

```text
np1_num
np2_num
np3_num
np_pattern
v1_attractor_incongruent
v2_attractor_incongruent
np2_np1_incongruent
regime_v1v2
```

Definitions:

```text
np_pattern = number pattern of NP1, NP2, NP3
S = singular
P = plural

v1_attractor_incongruent = 1 if NP3 differs from NP1
v2_attractor_incongruent = 1 if NP3 differs from NP2
np2_np1_incongruent      = 1 if NP2 differs from NP1
regime_v1v2              = (v1_attractor_incongruent, v2_attractor_incongruent)
```

Expected output counts:

```text
master_scores_full.csv:   66000 rows
master_scores_family.csv: 22000 rows
```

Expected checkpoint counts in the item-level master file:

```text
COPULA: 22000 rows
V2:     22000 rows
V3:     22000 rows
```

## 4. Summarize NP patterns and regimes

Main script:

```text
scripts/04_analyze_scores/summarize_patterns_regimes.py
```

Main input:

```text
data/scores/master_score_lists/master_scores_full.csv
```

Main outputs:

```text
data/scores/summaries/pattern_breakdown.csv
data/scores/summaries/regime_breakdown.csv
```

Expected row counts:

```text
pattern_breakdown.csv: 264 rows
11 models x 3 checkpoints x 8 NP patterns

regime_breakdown.csv: 132 rows
11 models x 3 checkpoints x 4 regimes
```

Expected family counts by NP pattern, using one model as the reference:

```text
SSS: 243
PPP: 254
SSP: 260
PPS: 232
PSS: 257
SPP: 249
SPS: 248
PSP: 257
```

Expected V2 regime counts, using one model as the reference:

```text
(0,0): 497
(0,1): 505
(1,0): 506
(1,1): 492
```

## 5. Compute the main V2 attractor statistics

This is the main inferential analysis for the paper. It estimates the V2 critical-region accuracy drop between congruent and incongruent NP3 attractor conditions.

Main script:

```text
scripts/04_analyze_scores/stats_master.py
```

Main input:

```text
data/scores/master_score_lists/master_scores_full.csv
```

Main output:

```text
data/scores/summaries/stats_summary.csv
```

Run command:

```bash
python3 scripts/04_analyze_scores/stats_master.py \
  --reproduce-archived \
  --bootstrap 10000 \
  --perm 10000 \
  --workers 12 \
  --seed 0
```

Use the full released master with the command above for the paper's
archived results. The default run uses model-specific random seeds in a
single process and writes stats_summary_worker_independent.csv. Its
resampling intervals and permutation p-values can differ slightly.

For each model, the script computes:

```text
overall full-continuation accuracy
overall critical-region accuracy
V2 critical-region accuracy when NP3 matches NP2
V2 critical-region accuracy when NP3 mismatches NP2
V2 drop: congruent accuracy - incongruent accuracy
95% bootstrap confidence interval for the V2 drop
one-sided permutation p-value for the V2 drop
```

Within each model, the V2 analysis has one checkpoint item per sentence family. Resampling those family records is therefore equivalent to resampling V2 items. The one-sided permutation test shuffles the V2 congruence labels across
families within each model, keeping the total count of each label fixed.
It does not stratify by NP1 alignment or by noun-number pattern. The three
checkpoints are modeled together only in the supplementary mixed-effects
analysis.

Expected row count:

```text
11 rows
```

## 6. Compute regime contrasts

This step tests whether accuracy differs between diagonal and off-diagonal regimes.

Main script:

```text
scripts/04_analyze_scores/compute_regime_contrasts.py
```

Main input:

```text
data/scores/master_score_lists/master_scores_full.csv
```

Main output:

```text
data/scores/summaries/regime_contrasts.csv
```

Run command:

```bash
python3 scripts/04_analyze_scores/compute_regime_contrasts.py \
  --bootstrap 10000 \
  --workers 12 \
  --seed 0
```

The diagonal contrast is:

```text
((acc(0,0) + acc(1,1)) / 2) - ((acc(0,1) + acc(1,0)) / 2)
```

For V2, the diagonal regimes are cases where NP1 and NP2 match. The off-diagonal regimes are cases where NP1 and NP2 mismatch. This makes the V2 diagonal contrast a compact test of NP1-NP2 alignment at the V2 checkpoint.

Expected row count:

```text
33 rows
11 models x 3 checkpoints
```

## 7. Compute NP-pattern p-values

This step prepares the p-values used in the NP-pattern tables.

Main script:

```text
scripts/04_analyze_scores/compute_pattern_model_pvalues.py
```

Main input:

```text
data/scores/master_score_lists/master_scores_full.csv
```

Main output:

```text
data/scores/summaries/pattern_model_pvalues.csv
```

Run command:

```bash
python3 scripts/04_analyze_scores/compute_pattern_model_pvalues.py
```

For each model, checkpoint, and NP pattern, the script reports:

```text
critical-region accuracy
number of items
number of families
number of correct families
two-sided exact binomial p-value against chance accuracy = 0.5
direction relative to chance
significance label
```

Expected row count:

```text
176 rows
2 checkpoints x 8 NP patterns x 11 models
```

The two checkpoints summarized here are:

```text
V2
COPULA
```

The current summary file uses the column name:

```text
p_two_sided
```

## 8. Make paper and appendix tables

Main scripts:

```text
scripts/05_make_figures_tables/make_main_results_tables.py
scripts/05_make_figures_tables/make_pattern_model_tables.py
```

Main inputs:

```text
data/scores/summaries/stats_summary.csv
data/scores/summaries/regime_contrasts.csv
data/scores/summaries/pattern_model_pvalues.csv
```

Main outputs used by the paper:

```text
paper/tables/table_v2_main_results.tex
paper/tables/table_v2_regime_contrasts.tex
paper/tables/table_pattern_v2_wide.tex
paper/tables/table_pattern_v2_stacked.tex
paper/tables/table_pattern_copula_stacked.tex
paper/tables/table_mixed_effects_per_model.tex
```

Run commands:

```bash
python3 scripts/05_make_figures_tables/make_main_results_tables.py

python3 scripts/05_make_figures_tables/make_pattern_model_tables.py
```

The main-text V2 pattern table can be wide. Appendix tables should stay compatible with the two-column appendix format.

## 9. Make figures

Main script:

```text
scripts/05_make_figures_tables/plot_master.py
```

Main inputs:

```text
data/scores/master_score_lists/master_scores_full.csv
data/scores/summaries/stats_summary.csv
```

Main output folder:

```text
paper/figures/
```

Typical outputs:

```text
checkpoint_accuracy.png
v2_accuracy_by_attractor.png
copula_accuracy_by_attractor.png
scaling_v2_scatter.png
scaling_copula_scatter.png
appendix_np_pattern_v2.png
appendix_np_pattern_copula.png
appendix_np_pattern_by_model_v2.png
appendix_np_pattern_by_model_copula.png
```

Run command:

```bash
python3 scripts/05_make_figures_tables/plot_master.py
```

## 10. Optional mixed-effects check

The mixed-effects analysis is supplementary. It checks the checkpoint-level pattern with sentence-family random intercepts.

Main script:

```text
scripts/04_analyze_scores/mixed_effects.R
```

Main input:

```text
data/scores/master_score_lists/master_scores_full.csv
```

Main outputs:

```text
data/scores/summaries/r_checkpoint_means.csv
data/scores/summaries/r_checkpoint_per_model.csv
data/scores/summaries/r_checkpoint_interaction.txt
paper/tables/table_mixed_effects_per_model.tex
```

The model uses all three checkpoints so that each sentence family contributes repeated observations. The per-model model is:

```text
correct_crit ~ checkpoint + (1 | family_id)
```

The combined model is:

```text
correct_crit ~ checkpoint * model + (1 | family_id)
```

The checkpoint reference level is V3.

Run command:

```bash
Rscript scripts/04_analyze_scores/mixed_effects.R
```

## 11. Environment documentation

Use the [reproduction YAML](environment/02_analysis_environment/environment-reproduce-analysis.yml)
for the Python analyses on the released score files. The [environment notes](environment/environment_README.md)
describe the exact Linux build list, the optional Windows R check, and the
historical DeltaAI scoring and home analysis environments.

## 12. Suggested run order

From the repository root:

```bash
python3 scripts/03_consolidate_scores/consolidate_scores.py

python3 scripts/04_analyze_scores/summarize_patterns_regimes.py

python3 scripts/04_analyze_scores/stats_master.py \
  --reproduce-archived \
  --bootstrap 10000 \
  --perm 10000 \
  --workers 12 \
  --seed 0

python3 scripts/04_analyze_scores/compute_regime_contrasts.py \
  --bootstrap 10000 \
  --workers 12 \
  --seed 0

python3 scripts/04_analyze_scores/compute_pattern_model_pvalues.py

python3 scripts/05_make_figures_tables/make_main_results_tables.py

python3 scripts/05_make_figures_tables/make_pattern_model_tables.py

python3 scripts/05_make_figures_tables/plot_master.py
```

The mixed-effects script is optional:

```bash
Rscript scripts/04_analyze_scores/mixed_effects.R
```

If the repository already includes the consolidated master score files, users can begin at:

```bash
python3 scripts/04_analyze_scores/summarize_patterns_regimes.py
```

## 13. Quick validation checks

Stimulus file:

```text
6000 benchmark items
2000 COPULA
2000 V2
2000 V3
0 SANITY rows
0 TEST rows
```

Master item file:

```text
66000 rows
11 models
6000 rows per model
22000 COPULA rows
22000 V2 rows
22000 V3 rows
0 SANITY rows
0 TEST rows
```

Master family file:

```text
22000 rows
11 models
2000 families per model
```

Summary files:

```text
pattern_breakdown.csv:       264 rows
regime_breakdown.csv:        132 rows
stats_summary.csv:            11 rows
regime_contrasts.csv:         33 rows
pattern_model_pvalues.csv:   176 rows
```

NP-pattern family counts, using one model as the reference:

```text
SSS: 243
PPP: 254
SSP: 260
PPS: 232
PSS: 257
SPP: 249
SPS: 248
PSP: 257
```

V2 regime counts, using one model as the reference:

```text
(0,0): 497
(0,1): 505
(1,0): 506
(1,1): 492
```

## 14. Release packaging

`release_files.txt` lists every public input and script. The ZIP builder
uses that inventory and generates the two SHA-256 manifests. The manuscript
workspace, temporary outputs, local metadata, and provider archives remain
in the working checkout.

## 15. AI assistance disclosure

Generative AI assistance was used for code debugging, file organization, formatting help, and editing support. The author checked the analyses, scientific claims, final code, generated tables, generated figures, and final manuscript text.

## Counterbalanced V2 forced-choice extension

The registered extension presents each of 2,000 V2 items in both A/B orders to 18 models. The selected evaluation contains 72,000 prompt-level observations, consolidated into 36,000 paired rows in `results/final/master_forced_choice_scores_v1.csv`.

The 11 open-weight base models use constrained next-token A/B log probabilities. GPT-3.5 Turbo used free text with strict A/B validation. All 4,000 GPT-3.5 responses were classified as invalid and received zero under the registered scoring rule. The other six proprietary systems used forced A/B tool calls, including the two H2/H3 endpoints. The manuscript must disclose this departure from the ordinary-generation protocol in the [OSF registration](https://doi.org/10.17605/OSF.IO/P3A7M).

The inference-settings table records the selected request settings. Generate it using the command below. Its API identifiers, budgets, reasoning/thinking options, temperature/top-p omissions, and tool interfaces were checked against all 28,000 selected proprietary requests.

### Primary result files

- H1 calibration — `results/final/preregistered_h1_v1.calibration.json` and its model-contrast CSV.
- H2 fully congruent endpoint comparisons — `results/final/preregistered_h2_v1.frontier_comparisons.json`.
- H3 global congruence gaps — the GPT-5.6 Sol and Opus 5 `*_analysis_v1.global_contrast.json` files.
- Historical releases, regimes, number patterns, order consistency, and strict paired correctness — secondary outputs under `results/final/` and `results/analysis/`.

Global congruence means that both NP1 and NP3 match the V2 controller NP2. The original critical-region attractor analysis and its tokenization appendix use the narrower NP3–NP2 relation. Consult the per-analysis manifests before interpreting or regenerating estimates.

## Camera-ready figures and appendix tables

The commands below write to `results/reproduced/`. They run from the benchmark release using the archived scores and audit inputs.

From the repository root, using the analysis environment:

```bash
python3 scripts/05_make_figures_tables/plot_master.py \
  --camera-ready-only --outdir results/reproduced/figures

python3 scripts/05_make_figures_tables/make_frontier_inference_settings_table.py \
  --outdir results/reproduced/tables
```

The first command reads the original master and archived tokenization results and writes:

- `results/reproduced/figures/checkpoint_accuracy.png` and `checkpoint_accuracy_by_family.pdf` — the family-faceted replacement for Figure 1. Shared accuracy axes and checkpoint encoding are used throughout. Model sizes are categorical, equally spaced positions. Llama labels identify the versions, because the series combines Llama 3.2 and 3.1.
- `results/reproduced/figures/tokenization_robustness_v2.png` and `.pdf` — full versus shared single-token subsets with archived 95% intervals.
- `results/reproduced/tables/table_tokenization_robustness.tex` — the corresponding gap estimates and intervals.

The second command writes `table_frontier_inference_settings.csv` and `.tex` in the requested table directory, including GPT-3.5.

The tokenization figure/table reuse stored scores and archived statistics. The [audit README](tokenization_robustness_audit/README_tokenization_robustness.md) documents the 2,000-, 1,094-, and 1,636-family analysis sets, the eligibility rule, and reproduction details. The 1,636-family comparison covers ten models and excludes Pythia. The audit input SHA-256 matches the current critical-region master.

## Exploratory V2 noun-number appendix

After the selected critical-region and forced-choice masters are available:

~~~bash
python3 scripts/04_analyze_scores/analyze_v2_noun_pull.py \
  --bootstrap 10000 --seed 20261118 \
  --figdir results/reproduced/figures \
  --tex-output results/reproduced/tables/appendix_v2_noun_pull.tex
~~~

This reads saved scores only. It generates the two-panel coefficient chart,
Appendix F LaTeX input, and CSVs/manifest under results/analysis/v2_noun_pull_v1/.
The source includes the appendix and a Results cross-reference. The analysis
is exploratory; it does not alter registered scores or replace the primary
accuracy contrasts. The root README and analysis README specify tie policies,
the pointwise pattern-stratified bootstrap, exclusions, and additive-model
limitations. No model inference or external API calls are required.


## Reproduce the preregistered forced-choice results

Run these commands from the repository root in the analysis environment.
H1 uses the archived critical-region and forced-choice masters and writes
its results back to `results/final/`. H2 and H3 below write to
`results/reproduced/forced-choice/`.

~~~bash
python3 scripts/analyze_forced_choice_h1.py

inputs=results/final
out=results/reproduced/forced-choice
mkdir -p "$out"

models=(
  gpt2:gpt2 pythia410:pythia-410m
  llama1b:llama-3.2-1b llama3b:llama-3.2-3b llama70b:llama-3.1-70b
  qwen1.5b:qwen2.5-1.5b qwen3b:qwen2.5-3b qwen7b:qwen2.5-7b
  qwen14b:qwen2.5-14b qwen32b:qwen2.5-32b qwen72b:qwen2.5-72b
)
comparators=()
for model in "${models[@]}"; do
  alias="${model%%:*}"
  stem="${model#*:}"
  comparators+=(--comparator-model "$alias"
                --model-file "$alias=$inputs/${stem}_forced_choice_paired_v1.jsonl")
done

for model in gpt5.6:openai_gpt-5.6-sol:20260729 opus5:anthropic_opus-5:20260730; do
  IFS=: read -r alias stem seed <<< "$model"
  paired="$inputs/${stem}_paired_v1.jsonl"
  python3 scripts/compare_forced_choice_models.py \
    --frontier-model "$alias" --model-file "$alias=$paired" \
    "${comparators[@]}" --bootstrap-replicates 10000 --bootstrap-seed "$seed" \
    --output-prefix "$out/h2-$alias"
  python3 scripts/analyze_forced_choice_results.py \
    --item-results "$paired" --bootstrap-replicates 10000 --bootstrap-seed 20260728 \
    --output-prefix "$out/h3-$alias"
done
~~~

H2 uses the equal-weight comparator of eleven open-weight models.
The endpoint-specific seeds reproduce the released H2 intervals.
The selected paired inputs and recorded H1–H3 results are under
`results/final/`.

## Rebuild the camera-ready artefacts

Use the archived critical-region, paired forced-choice, and open-weight
prompt score masters. The commands in README rebuild the Figure 1 facets
and tokenization panel/table, the API settings table, and the Appendix F
noun-choice figure. Preserve the selected scoring rules and bootstrap
settings. The endpoint and all-model appendix tables use the stored
preregistered results and selected paired master.

Use `release_files.txt` and the packaging command in README for the ZIP.
The selected paired inputs remain available to reconstruct the forced-choice
master. Raw provider archives are kept locally for provenance.
