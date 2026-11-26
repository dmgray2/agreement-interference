# Scripts README

This folder contains the scripts used to generate the dataset, score models, analyze scores, and make tables and figures.

Run scripts from the repository root unless a script says otherwise.

## Folder layout

```text
scripts/00_setup_repository/
scripts/01_generate_dataset/
scripts/02_score_models/
scripts/03_consolidate_scores/
scripts/04_analyze_scores/
scripts/05_make_figures_tables/
```

## 00_setup_repository

```text
scripts/00_setup_repository/make_repo_manifest.sh
```

This script writes a file manifest for checking the repository before release.

The environment-documentation script may also be stored here during private preparation. Do not include scripts that reveal local paths, usernames, accounts, or machine names in the anonymous release.

## 01_generate_dataset

```text
scripts/01_generate_dataset/gen_minpairs_ce2_family_v0.7.py
```

This script generates the CE2-family minimal-pair dataset from the lexicon files in `data/lexicons/`.

Main outputs:

```text
data/stimuli/minpairs.ce2fam.generated.v0.7_nf2000.jsonl
data/stimuli/minpairs.ce2fam.audit.v0.7_nf2000.csv
data/stimuli/minpairs.ce2fam.audit.v0.7_nf2000.per_family.csv
```

## 02_score_models

```text
scripts/02_score_models/score_minpairs.py
```

This script scores a causal language model on the generated minimal pairs. It writes item-level and family-level score files.

Important output columns:

```text
correct
correct_crit
delta
delta_crit
```

The paper mainly uses `correct_crit`.

Slurm templates are stored in:

```text
scripts/02_score_models/slurm_templates/
```

These templates should use placeholders for accounts, partitions, model paths, and repository paths.

## Scoring precision: corrected entry points and historical reproduction

Both active scorers now compute log-softmax explicitly in float32:

    torch.log_softmax(logits, dim=-1, dtype=torch.float32)

This is independent of --dtype, which continues to control model precision.
Token log probabilities are accumulated as Python floats. Casting after
log-softmax would retain its previous rounding; promoting the logits before
this operation does not recover precision lost during model inference.
Float32 probability tensors can use more GPU memory.

- scripts/02_score_models/score_minpairs.py retains strict-positive-margin
  correctness (exact ties remain incorrect). Its optional --out-summary JSON
  records scoring_version=float32_logsoftmax_v2 and logprob_dtype=float32.
- scripts/run_open_weight_forced_choice_logprob.py retains the registered
  exact-tie rule, select A. New results and manifests use logprob_fp32_v2
  rather than logprob_v1, and metadata records logprob precision and resolved
  model dtype. The existing refusal to overwrite output files remains.
- Unmodified copies of both scripts are in
  [archive/pre_float32_logsoftmax/](archive/pre_float32_logsoftmax/), with a
  README and SHA256SUMS. Run these from the repository root for the historical
  calculation. Existing scores have not been regenerated.

For a new critical-region run, use separate output CSV, family CSV, and
summary JSON paths; do not overwrite the selected paper inputs. For example,
choose outputs under results/precision_checks/ and include --out-summary.
Existing Slurm templates that invoke the active scorer will now use float32
log-softmax. To reproduce the old calculation, point a copied job template
at the archived script.

### What the tie sensitivity establishes

See [the root README](../README.md#numerical-precision-and-exact-ties) for the
full scope and interpretation. In the original CE2 master, critical-region
tie rates over all 6,000 items/model are 1.0-1.6% except GPT-2 (7.9%).
Changing saved ties to half credit or excluding them changes the V2 NP3
accuracy gap by at most .006944 and the equal-regime NP1 diagonal accuracy
contrast by at most .024324. Absolute accuracies can change more: GPT-2
increases by .0395 overall and .035 at V2 with half credit.

These checks do not rescore logits. They do not establish bounds for the
noun-"pull" regression or for the forced-choice analysis. Assigning exact
verb-score ties .5 in the recovered plural-choice outcome changes GPT-2's
controller coefficient by about .070. Invalid responses are not verb choices.
The selected prompt master verifies 583/4,000 ties (14.575%, rounding to
14.6%) in Llama-3.2-1B, the maximum across the eleven open-weight systems.
All 1,418 tied prompt scores select A. The exploratory coefficient analysis
has its own sensitivity outputs; the .007/.025 bounds still do not cover
forced-choice accuracy contrasts.

## 03_consolidate_scores

```text
scripts/03_consolidate_scores/consolidate_scores.py
```

This script combines the per-model score files and adds the NP-pattern and regime metadata.

Main outputs:

```text
data/scores/master_score_lists/master_scores_full.csv
data/scores/master_score_lists/master_scores_family.csv
```

## 04_analyze_scores

```text
scripts/04_analyze_scores/summarize_patterns_regimes.py
scripts/04_analyze_scores/stats_master.py
scripts/04_analyze_scores/compute_regime_contrasts.py
scripts/04_analyze_scores/compute_pattern_model_pvalues.py
scripts/04_analyze_scores/mixed_effects.R
```

`summarize_patterns_regimes.py` writes pattern and regime breakdowns.

`stats_master.py` computes the main V2 attractor effect, including a bootstrap confidence interval and a one-sided permutation p-value. Each model contributes one V2 item per sentence family.

`compute_regime_contrasts.py` computes the diagonal regime contrast.

`compute_pattern_model_pvalues.py` computes exact binomial p-values for pattern-level tables.

`mixed_effects.R` is optional. It fits all-checkpoint repeated-measures logistic models with a sentence-family random intercept.

## 05_make_figures_tables

```text
scripts/05_make_figures_tables/make_main_results_tables.py
scripts/05_make_figures_tables/make_pattern_model_tables.py
scripts/05_make_figures_tables/plot_master.py
```

These scripts create LaTeX tables and figures for the paper.

Main table outputs:

```text
paper/tables/table_v2_main_results.tex
paper/tables/table_v2_regime_contrasts.tex
paper/tables/table_pattern_v2_wide.tex
paper/tables/table_pattern_v2_stacked.tex
paper/tables/table_pattern_copula_stacked.tex
```

Main figure outputs are written to:

```text
paper/figures/
```

## Suggested analysis run

If the master score files already exist:

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

Run the optional R model separately:

```bash
Rscript scripts/04_analyze_scores/mixed_effects.R
```

## Files to exclude before release

Do not include these in the anonymous release:

```text
__pycache__/
*.pyc
.RData
.Rhistory
*.before_*
*:Zone.Identifier
*:com.dropbox.attrs
```
## Camera-ready generation

`plot_master.py --camera-ready-only --outdir results/reproduced/figures` builds the family-faceted checkpoint plot and tokenization appendix plot/table from saved data. The table is written under `results/reproduced/tables/`. The normal plot command continues to generate the wider set of original figures.

`make_frontier_inference_settings_table.py --outdir results/reproduced/tables` writes the seven proprietary model settings, including GPT-3.5. The table includes the actual response interface and the recorded departure from the preregistration. Its rows describe the selected archived requests.

Forced-choice scripts at the root of `scripts/` support prompt construction, provider requests, parsing, paired-score consolidation, and inference. Main analysis entry points include `analyze_forced_choice_h1.py`, `compare_forced_choice_models.py`, and `analyze_forced_choice_results.py`. Use each script's help or recorded invocation to select inputs and outputs. The stored outputs and hypotheses are mapped in the root README.

## Exploratory V2 noun-choice regressions

~~~bash
python3 scripts/04_analyze_scores/analyze_v2_noun_pull.py \
  --bootstrap 10000 --seed 20261118 \
  --figdir results/reproduced/figures \
  --tex-output results/reproduced/tables/appendix_v2_noun_pull.tex
~~~

This analysis uses saved data, with no model loading or API calls. It fits
an additive OLS regression separately for each of 28 model-method combinations:
eleven original critical-region models and seventeen valid forced-choice
systems. All models use the same 2,000 V2 items. GPT-3.5's invalid responses
are excluded from recovered choices, while its registered scores remain intact.

The primary exploratory outcome assigns .5 to exact numeric ties before
averaging A/B orders. Coefficient sensitivity outputs also fit the historical
rules and exclude any base item with a tie. Bootstrap intervals are computed
for the primary exploratory outcome only, using item resamples within all
eight number patterns, paired across methods/models with answer orders together.

The script writes results/analysis/v2_noun_pull_v1/, the PDF/PNG coefficient
plot in results/reproduced/figures/, and the appendix table in
results/reproduced/tables/. The analysis manifest records source hashes,
script hash, bootstrap
parameters, cohort selection, exclusions, and package versions.

Inputs are validated against one another. Direct OLS is checked against the
equivalent cell-weighted computation. Offline validation covered known
controller-based, first-noun-copying, constant-plural, answer-position-biased,
and nonadditive response patterns. Pattern means, conditional contrasts, and
additive residuals accompany the averaged coefficients. In the saved analysis,
effects vary across configurations and some fitted means lie outside [0,1].
Do not interpret these coefficients as complete choice rules or internal
causal weights. See the analysis README for output definitions and limitations.


## Camera-ready script scope

The active analysis covers critical-region CE2 and the selected V2
forced-choice extension. Use the archived score masters to reproduce
paper outputs without inference. Use the historical scorer copies for
the original low-precision calculations. See the root README for exact
commands and the release file list for included scripts.
