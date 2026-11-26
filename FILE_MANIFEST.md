# File Manifest

This file gives a short map of the files that matter for reproducing the analysis.

For the full command sequence, see `PIPELINE.md`.

## Top-level files

```text
README.md              project overview and quick start
PIPELINE.md            full reproduction guide
data/data_README.md    data-folder notes
scripts/scripts_README.md  script-folder notes
environment/environment_README.md  environment notes
FILE_MANIFEST.md       this file
```

## Data

### Lexicons

```text
data/lexicons/lexicon_nouns.csv
data/lexicons/lexicon_verbs_transitive.csv
```

Used by the dataset generator.

### Stimuli

```text
data/stimuli/minpairs.ce2fam.generated.v0.7_nf2000.jsonl
data/stimuli/minpairs.ce2fam.audit.v0.7_nf2000.csv
data/stimuli/minpairs.ce2fam.audit.v0.7_nf2000.per_family.csv
```

Main benchmark and audit files.

### Model scores

```text
data/scores/model_individual_level/
data/scores/model_family_level/
```

Per-model score files.

### Consolidated scores

```text
data/scores/master_score_lists/master_scores_full.csv
data/scores/master_score_lists/master_scores_family.csv
```

Main consolidated score files.

### Summaries

```text
data/scores/summaries/pattern_breakdown.csv
data/scores/summaries/regime_breakdown.csv
data/scores/summaries/stats_summary.csv
data/scores/summaries/regime_contrasts.csv
data/scores/summaries/pattern_model_pvalues.csv
```

Optional mixed-effects summaries:

```text
data/scores/summaries/r_checkpoint_means.csv
data/scores/summaries/r_checkpoint_per_model.csv
data/scores/summaries/r_checkpoint_interaction.txt
```

## Scripts

### Dataset generation

```text
scripts/01_generate_dataset/gen_minpairs_ce2_family_v0.7.py
```

### Model scoring

```text
scripts/02_score_models/score_minpairs.py
scripts/02_score_models/slurm_templates/
```

### Score consolidation

```text
scripts/03_consolidate_scores/consolidate_scores.py
```

### Analysis

```text
scripts/04_analyze_scores/summarize_patterns_regimes.py
scripts/04_analyze_scores/stats_master.py
scripts/04_analyze_scores/compute_regime_contrasts.py
scripts/04_analyze_scores/compute_pattern_model_pvalues.py
scripts/04_analyze_scores/mixed_effects.R
```

### Tables and figures

```text
scripts/05_make_figures_tables/make_main_results_tables.py
scripts/05_make_figures_tables/make_pattern_model_tables.py
scripts/05_make_figures_tables/plot_master.py
```

## Paper support files

```text
paper/acl.sty
paper/acl_natbib.bst
paper/references.bib
paper/figures/
paper/tables/
```

The generated tables in `paper/tables/` are produced by the table scripts. The generated figures in `paper/figures/` are produced by `plot_master.py`.

## Environment files

```text
environment/01_scoring_environment/
environment/02_analysis_environment/
```

The scoring environment folder documents the GPU scoring environment. The analysis environment folder documents the local conda/R environment used for statistics, tables, and figures.

## Files that should not be in the anonymous release

Remove these before making the final zip:

```text
Dataset Development History.docx
docs_README.md
.RData
.Rhistory
__pycache__/
*.pyc
*.before_*
*:Zone.Identifier
*:com.dropbox.attrs
repo manifests that contain local absolute paths
```

Also remove old backup stimulus files unless the paper or repository documentation explicitly uses them.
## Camera-ready additions

- Paired forced-choice results — `results/final/master_forced_choice_scores_v1.csv`.
- H1/H2 registered-analysis outputs — `results/final/preregistered_h1_v1.*` and `preregistered_h2_v1.*`.
- Tokenization script, subset membership, and archived statistics — `tokenization_robustness_audit/`.
- Camera-ready plot command — `scripts/05_make_figures_tables/plot_master.py --camera-ready-only`, with the output directory supplied as documented in the README.
- Inference-settings generator — `scripts/05_make_figures_tables/make_frontier_inference_settings_table.py`.

- Exploratory noun-choice analysis script -- scripts/04_analyze_scores/analyze_v2_noun_pull.py.
- Coefficients, bootstrap intervals, tie sensitivity, pattern/conditional contrasts,
  fit checks, tie audit, and provenance -- results/analysis/v2_noun_pull_v1/.
- Appendix F outputs — the noun-choice command in the README regenerates these.


## Camera-ready package inventory

`release_files.txt` is the explicit public inventory. The archive includes
CE2 and forced-choice analysis inputs, environment records, and licenses.
The packaging command generates both manifests from the archive's contents.
The manuscript workspace, provider archives, and private experiments remain local.

## AI Assistance Disclosure

The authors used ChatGPT for coding assistance, debugging, and checklist-style review of materials. AI tools were also used to assist with formatting the README, pipeline, and manifest documents. The authors made all experimental design decisions, conducted and checked the statistical analyses, verified the sources, reviewed the code, and approved the final manuscript. The authors take full responsibility for the paper, code, data, and results.