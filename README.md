## AI Assistance Disclosure

The authors used ChatGPT for coding assistance, debugging, and checklist-style review of materials. AI tools were also used to assist with formatting the README, pipeline, and manifest documents. The authors made all experimental design decisions, conducted and checked the statistical analyses, verified the sources, reviewed the code, and approved the final manuscript. The authors take full responsibility for the paper, code, data, and results.

# Agreement Interference

Replication materials for an EACL 2027 study of subject-verb agreement
in doubly center-embedded sentences. The materials include the scoring
and analysis code, released model scores, and instructions for
reproducing the figures and tables.

The [CE-Agree v1.0.0 benchmark](https://github.com/dmgray2/ce-agree/tree/v1.0.0)
contains 6,000 minimal pairs from 2,000 sentence families at V3, V2, and
COPULA. This replication package includes the exact stimulus file and
lexicons at the paths the scripts use. The item-level score master
contains 66,000 rows from eleven models.

## Set up the analysis environment

~~~bash
conda env create --file environment/02_analysis_environment/environment-reproduce-analysis.yml
conda activate agreement-interference-analysis
~~~

Start with [the original critical-region analysis](#reproducing-the-original-critical-region-analysis)
or the [camera-ready figures and appendix tables](#camera-ready-figures-and-appendix-tables).
These commands run from the repository root on the released scores.
See [software environments](environment/environment_README.md) for the
Linux build list, the optional R analysis, and original scoring records.

## Repository layout

| Path | Contents |
| --- | --- |
| `data/` | Benchmark inputs, item and family scores, and analysis summaries |
| `scripts/` | Stimulus generation, scoring, analysis, and figure and table scripts |
| `results/` | Released forced-choice and supplementary analyses |
| `tokenization_robustness_audit/` | Matched tokenization subsets and results |
| `environment/` | Reproduction YAML and historical environment records |
| `PIPELINE.md` | Full pipeline, including model scoring |

## Main files

The consolidated score files are:

```text
data/scores/master_score_lists/master_scores_full.csv
data/scores/master_score_lists/master_scores_family.csv
```

The main summary files are:

```text
data/scores/summaries/stats_summary.csv
data/scores/summaries/regime_contrasts.csv
data/scores/summaries/pattern_breakdown.csv
data/scores/summaries/regime_breakdown.csv
data/scores/summaries/pattern_model_pvalues.csv
```

The original table generators write to:

```text
paper/tables/
```

The original figure generator writes to:

```text
paper/figures/
```
## Benchmark name

This repository contains **CE-Agree**, a benchmark for agreement under
attractor interference in center-embedded sentences.

The current release evaluates doubly center-embedded structures. The broader
name is intended to permit future releases at other embedding depths while
retaining agreement as the benchmark's target phenomenon.

## Counterbalanced V2 forced-choice extension

The registered extension presents each of 2,000 V2 items in both A/B orders to 18 models. The selected evaluation contains 72,000 prompt-level observations, consolidated into 36,000 paired rows in `results/final/master_forced_choice_scores_v1.csv`.

The 11 open-weight base models use constrained next-token A/B log probabilities. GPT-3.5 Turbo used free text with strict A/B validation. All 4,000 GPT-3.5 responses were classified as invalid and received zero under the registered scoring rule. The other six proprietary systems used forced A/B tool calls, including the two H2/H3 endpoints. The manuscript must disclose this departure from the ordinary-generation protocol in the [OSF registration](https://doi.org/10.17605/OSF.IO/P3A7M).

The inference-settings table records the selected request settings. Generate it using the command below. Its API identifiers, budgets, reasoning/thinking options, temperature/top-p omissions, and tool interfaces were checked against all 28,000 selected proprietary requests.

### Primary result files

- H1 calibration — `results/final/preregistered_h1_v1.calibration.json` and its model-contrast CSV.
- H2 fully congruent endpoint comparisons — `results/final/preregistered_h2_v1.frontier_comparisons.json`.
- H3 global congruence gaps — [GPT-5.6 Sol](results/final/openai_gpt-5.6-sol_analysis_v1.global_contrast.json) and [Opus 5](results/final/anthropic_opus-5_analysis_v1.global_contrast.json).
- Historical releases, regimes, number patterns, order consistency, and strict paired correctness — secondary outputs under `results/final/` and `results/analysis/`.

Global congruence means that both NP1 and NP3 match the V2 controller NP2. The original critical-region attractor analysis and its tokenization appendix use the narrower NP3–NP2 relation. The [pipeline](PIPELINE.md#reproduce-the-preregistered-forced-choice-results) gives the commands and seeds for reproducing H1–H3.

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

## Numerical precision and exact ties

The reported original critical-region scores were computed with model weights
loaded in bfloat16 and without an explicit float32 log-softmax output dtype.
Exact zero critical-region margins were counted as incorrect. The following
audit concerns the original 11-model CE2 master, not the forced-choice master.

Across all 6,000 CE2 items per model, ties account for 1.0-1.6% in the ten
models other than GPT-2, and 7.9% (474/6,000) in GPT-2. At V2, GPT-2 has
140/2,000 ties (7.0%). Counting saved ties as half correct raises GPT-2's
all-checkpoint accuracy by .0395 and V2 accuracy by .035.

Sensitivity checks retain the saved scores and either assign exact ties .5
accuracy or exclude them before recomputing the condition means. Across the
11 models, either alternative changes:

- the original V2 NP3-match minus NP3-mismatch accuracy contrast by at most
  .006944 (Qwen-3B, exclusion);
- the equal-regime NP1 diagonal accuracy contrast by at most .024324
  (GPT-2, exclusion). This is the mean of regimes (0,0) and (1,1) minus the
  mean of (0,1) and (1,0), as in compute_regime_contrasts.py.

These bounds apply to those accuracy contrasts. They do not bound absolute
accuracy changes, every statistic, or the noun-choice regression coefficients.
For example, assigning tied plural-choice outcomes .5 changes GPT-2's NP2
OLS coefficient from about .027 to .097. Such a regression is exploratory;
it needs separate tie sensitivity and uncertainty estimates. There is one V2
item per family per model, so resampling V2 families is item-level resampling.
A family bootstrap does not automatically widen the intervals or account for
the reuse of vocabulary. Counterbalancing does not inherently shrink all
coefficients, although answer-position bias can mask sensitivity to noun number.

The corrected scripts compute log-softmax with dtype=torch.float32 before
extracting token log probabilities, which are accumulated as Python floats.
Model weights and inference still follow the requested --dtype. This avoids
additional low-precision log-softmax output rounding; it cannot recover
precision already lost in logits and does not guarantee that all ties disappear.
Float32 probability tensors can require more GPU memory; reduce batch size if needed.

The corrected entry points are:

- scripts/02_score_models/score_minpairs.py
- scripts/run_open_weight_forced_choice_logprob.py

Unmodified pre-fix copies and their SHA-256 hashes are preserved in
[scripts/archive/pre_float32_logsoftmax/](scripts/archive/pre_float32_logsoftmax/).
Use those copies to reproduce the historical computation. Existing master
scores, analyses, and figures have not been replaced or rescored. New float32
runs must use separate output paths. The corrected forced-choice scorer uses
the distinct filename suffix logprob_fp32_v2 and records precision in its
manifest. The critical-region scorer records precision in --out-summary JSON
and prints it at startup; request a summary for each new run.

Forced choice has a separate registered rule: exact A/B score ties select A,
so they are not automatically incorrect. Malformed proprietary responses follow
the invalid-response rule and cannot be reconstructed as opposite verb choices.
The selected prompt master verifies the forced-choice maximum at 583/4,000
(14.575%, rounding to 14.6%) for Llama-3.2-1B. All 1,418 tied prompt scores
across the eleven selected open-weight systems use the registered select-A
rule. The exploratory noun-choice analysis separately reports coefficient
sensitivity. The .007/.025 bounds above must not be applied to forced-choice
accuracy contrasts or presented as a float32 rescoring validation.

The active LaTeX source contains the critical-region disclosure plus marked
CAMERA-READY REWRITE NOTES covering these distinctions.

## Exploratory noun-number associations at V2

The Python entry point is scripts/04_analyze_scores/analyze_v2_noun_pull.py.
It uses saved scores only; it does not load models or send API requests.
Run from the repository root in the analysis environment:

~~~bash
python3 scripts/04_analyze_scores/analyze_v2_noun_pull.py \
  --bootstrap 10000 --seed 20261118 \
  --figdir results/reproduced/figures \
  --tex-output results/reproduced/tables/appendix_v2_noun_pull.tex
~~~

It reads the original critical-region master, the selected paired
forced-choice master, and the open-weight prompt-level bfloat16 master.
The script verifies identical base-item IDs, number patterns, and lexical
prefixes, and reconciles prompt correctness with the paired master.

The appendix chart includes all eleven original models under critical-region
scoring and seventeen valid forced-choice systems (the same eleven plus six
proprietary systems). GPT-3.5 is excluded from recovered-choice regressions
because 0/4,000 responses passed strict A/B validation. Its registered score
and format-failure accounting remain in the primary study. The 405B, CE0,
CE3, and nonce runs are outside this analysis.

Each model-method regression predicts plural-choice rate from an intercept
and the three noun numbers (singular=0, plural=1). Exact critical-score ties
are assigned .5. Open-weight A/B ties are assigned .5 before the two answer
orders are averaged. These are exploratory outcomes, separate from the
historical accuracy rules. Coefficients use observed item weights across
all eight number patterns. Intervals are pointwise central 95% percentile
bootstrap intervals, with items resampled within patterns using shared
draws across models and methods; both answer orders remain together.
There is one V2 item per family. Intervals condition on the sampled
vocabulary, template, and selected systems.

Outputs:

- `results/reproduced/figures/v2_noun_pull.pdf` and `.png` — the coefficient plot.
- `results/reproduced/tables/appendix_v2_noun_pull.tex` — generated Appendix F methods,
  figure, caption, labels, and limitations.
- results/analysis/v2_noun_pull_v1/ -- coefficient estimates and intervals,
  tie sensitivity, eight-pattern means, conditional contrasts, additive-fit
  diagnostics, tie counts, and an input-hash/software-version manifest.
- [Detailed analysis notes](results/analysis/v2_noun_pull_v1/README.md).


The coefficients are descriptive behavioral associations, not internal
causal weights. Associations vary with number configuration; a small
average coefficient does not establish no controller sensitivity everywhere.
Some additive fitted means exceed the probability range, so the regression
is a summary rather than a complete probability model. Consult the
conditional contrasts and pattern residuals before interpreting a coefficient.

## Reproducing the original critical-region analysis

The paper's V2 nearest-noun statistics come from the full released master
and the settings below, including `stats_master.py`'s `--reproduce-archived`
mode. The default mode uses a stable seed per model. With the same master,
seed, and draw counts, the point estimates matched exactly, and the
largest difference in a confidence-interval endpoint was 0.20 percentage
points. Permutation p-values can also vary between modes.

With the released master score files, run:

```bash
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

For the full process, including dataset generation and model scoring, see `PIPELINE.md`.

## Expected counts

The main files should have these row counts:

```text
master_scores_full.csv:       66000
master_scores_family.csv:     22000
pattern_breakdown.csv:          264
regime_breakdown.csv:           132
stats_summary.csv:               11
regime_contrasts.csv:            33
pattern_model_pvalues.csv:      176
```

The item-level master file should contain:

```text
COPULA: 22000 rows
V2:     22000 rows
V3:     22000 rows
```


## Licenses

The software in this repository—including the stimulus-generation,
model-scoring, score-consolidation, statistical-analysis, and figure- and
table-generation scripts—is released under the **MIT License**.

The CE-Agree benchmark materials—including the generated stimuli, lexicons,
audit files, and released score data—are licensed under the **Creative Commons
Attribution 4.0 International License (CC BY 4.0)**.

See [license scope](licenses/LICENSE_README.md), [MIT](LICENSE),
and [CC BY 4.0](licenses/CC-BY-4.0.txt).

Third-party materials, if any, remain subject to their original licenses.

## Camera-ready release

CE-Agree covers the eleven original CE2 base models and the selected
18-system V2 forced-choice extension. The release includes the three score
masters, paired inputs for consolidating forced-choice scores, selected
results, the tokenization audit, and analysis scripts.
The paper figures and tables can be rebuilt from the archived scores
without model inference or API calls.

The full pre-cleanup checkout is preserved outside this working tree.
The archive retains the nonce, CE0, CE3, and exploratory
runs. Selected per-model paired inputs remain here because the consolidation
script uses them. Provider request and response archives stay local for
provenance. The public ZIP contains the curated analysis inputs.

Use the file list when packaging. Review any file added to it before release,
then run from the repository root:

    bash scripts/00_setup_repository/make_release_zip.sh --output /tmp/agreement-interference-release.zip

The ZIP includes size and SHA-256 manifests derived from its contents.
It excludes the entire manuscript workspace, including its LaTeX sources,
bibliography, PDFs, figures, and tables. For a new public repository, start
from the extracted release in an empty directory with fresh Git history.

