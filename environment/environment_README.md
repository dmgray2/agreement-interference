# Software environments

## Reproduce the analyses from the released scores

Create the small conda environment from the repository root:

~~~bash
conda env create --file environment/02_analysis_environment/environment-reproduce-analysis.yml
conda activate agreement-interference-analysis
~~~

The YAML pins Python 3.12.13, NumPy 2.4.3, pandas 3.0.2, SciPy 1.17.1, and
Matplotlib 3.10.9. These are the observed versions used for the
camera-ready Python reproduction check. No GPU drivers, model weights, or
API credentials are needed for analysis of saved scores.

For the exact conda build set resolved during the Linux x86-64 check
(including transitive dependencies and SHA-256 hashes), use:

~~~bash
conda create --name agreement-interference-analysis --file environment/02_analysis_environment/conda-explicit-reproduce-analysis-linux-64.txt
~~~

Use the YAML to let conda resolve builds for another platform. The explicit
file is specific to Linux x86-64.

## Optional mixed-effects models in R

The R script uses `lme4` and `broom.mixed`. See the [R session record](02_analysis_environment/r-session-reproduce-windows.txt)
for versions used in the Windows check.

~~~bash
Rscript scripts/04_analyze_scores/mixed_effects.R --out-dir results/reproduced/r
~~~

The Windows rerun matched all 33 checkpoint means and the significance
labels for all 22 per-model effects. The largest difference in per-model
odds ratios was 0.000043. In the combined model, the largest difference
in printed log-odds coefficients was 0.00022. The rerun emitted an optimizer
budget warning. The original mixed-effects outputs are in
`data/scores/summaries/`.

## Historical environment records

The `01_scoring_environment/` folder records the original DeltaAI GPU
scoring environment, with conda exports and package versions. These files
are provenance records. Recreating model scores from scratch also
requires a compatible GPU runtime and the correct model checkpoints.

The files with `home-ler-data` in `02_analysis_environment/` record the
historical home analysis environment. The small YAML and the explicit
build list above document the separate reproduction check. Raw pip
freezes with machine-specific build paths are kept in the working
checkout. The conda records do not describe provider-side inference
runtimes. See the main README for the generator of the API settings
table and `model_revisions.csv` for archived checkout revisions.
