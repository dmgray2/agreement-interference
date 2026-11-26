# Exploratory V2 noun-number associations

This is an exploratory analysis of saved scores. It was developed after
inspection of the data and is not a new preregistered confirmatory test.

## Reproduce from the repository root

~~~bash
python3 scripts/04_analyze_scores/analyze_v2_noun_pull.py \
  --bootstrap 10000 --seed 20261118
~~~

Dependencies are NumPy, pandas, and Matplotlib in the existing analysis
environment. CPU execution is sufficient. No models are loaded and no API
requests are made. Use --help for alternate input/output paths. The manifest
records the seed, chunk size, package versions, and SHA-256 hashes of the
script and all three input files.

## Data and exclusions

There are 2,000 shared V2 base items, one per sentence family. All eleven
original open-weight models are included under critical-region scoring.
Forced choice includes those eleven plus GPT-4.1, GPT-5, GPT-5.6 Sol, and
Opus 4.1/4.5/5: seventeen valid systems. GPT-3.5 has 0/4,000 valid A/B
responses and is excluded from recovered-choice regressions; its registered
accuracy and format-failure reporting remain intact. The analysis does
not include 405B, CE0, CE3, or nonce runs.

The prompt-level open-weight master is reconciled to both orders of the
selected forced-choice master. Item IDs, number patterns, lexical prefixes,
label selections, correctness, validity, and pair-score arithmetic are checked.

## Outcome and regression

With NP2 plural, plural-choice outcome equals correctness. With NP2 singular,
it equals 1 minus correctness. For critical-region ties the exploratory
plural-choice value is .5. For open-weight A/B ties, the order-level
correctness is assigned .5 before averaging both orders and reconstructing
the plural-choice rate. Resulting paired outcomes may be 0, .25, .5, .75, or 1.
The six proprietary systems retain their recorded valid choices.

Separate ordinary least-squares regressions include an intercept plus NP1,
NP2 and NP3 numbers, singular=0 and plural=1. Coefficients use observed
item counts across all eight patterns. The intercept is the fitted
all-singular baseline, not the overall plural-choice rate. A positive noun
coefficient summarizes a higher plural-choice rate with that noun plural,
adjusting for the other noun numbers under the additive approximation.

Perfect controller-based behavior yields intercept=0, NP1=0, NP2=1, NP3=0.
Always-plural behavior yields intercept=1 and zero noun coefficients.
Always selecting one answer position in counterbalanced forced choice
yields outcome=.5, intercept=.5, and zero noun coefficients. These cases
illustrate why zero coefficients do not establish robust agreement.

## Uncertainty and tie sensitivity

The chart reports pointwise central 95% percentile intervals from 10,000
bootstrap resamples of base items within each of the eight number patterns.
Shared item draws preserve pairing across all models and both methods.
Both A/B orders stay together. There is one V2 observation per family:
family resampling here is item resampling. These intervals condition on this
vocabulary, template, and model selection; they do not represent sampling
new vocabularies or new model families. There is no multiplicity correction.

The primary exploratory coefficients use half-valued exact ties.
Sensitivity estimates separately use historical rules (critical ties wrong,
A/B ties select A), or exclude a base item if its critical score ties or
either A/B order ties. Intervals are computed only for the primary
half-tie estimates. Exclusion can change the represented mixture of items.

The selected open-weight prompt master contains 1,418 exact A/B score ties.
The highest prompt-level rate is 583/4,000=14.575% in Llama-3.2-1B. All tied
prompts select A under the registered rule. Numeric tie counts do not apply
to the six proprietary systems, whose output is a valid tool response rather
than exposed candidate log probabilities.

These are sensitivity checks on saved scores, not float32 rescoring.
No existing registered scores or reported accuracy tables were changed.

## Additive approximation and interpretation

These coefficients summarize behavior; they are not identified internal
causal weights. Conditional contrasts compare observed pattern means across
different lexical items with the other two noun numbers fixed. They are
not direct interventions on a model's internal representations.

Associations vary across number configurations. A small average coefficient
can coexist with larger conditional associations, including opposite signs.
For example, in Qwen-72B critical-region scoring, the NP2 conditional
contrasts range approximately from -.012 to .149, while the additive NP2
coefficient is .059. The largest absolute pattern-mean residual in that
model is approximately .148. Some models' additive fitted means exceed 1.
The coefficients therefore are not a complete probability model or a
uniform choice rule. See conditional_contrasts.csv and fit_diagnostics.csv.

Tasks and interfaces differ across panels. Answer-position bias can mask
noun-number sensitivity in forced choice. Model-size comparisons are
descriptive and do not isolate parameter count from training or model-family
differences.

## Outputs

| File | Contents |
| --- | --- |
| coefficients.csv | Half-tie OLS estimates, pointwise bootstrap intervals, and intercepts |
| tie_sensitivity.csv | Estimates under half, historical, and exclusion policies |
| pattern_means.csv | Eight observed plural-choice means, fitted means, and residuals per model/method |
| conditional_contrasts.csv | Four conditional contrasts per noun, with pointwise bootstrap intervals |
| fit_diagnostics.csv | Maximum/RMS pattern residuals and fitted-value ranges |
| tie_audit.csv | Critical-region and open-weight A/B tie counts and denominators |
| analysis_manifest.json | Cohort, exclusions, input/script checksums, methods, and software versions |

The script also writes EACL paper/figures/v2_noun_pull.pdf and .png, and
EACL paper/tables/appendix_v2_noun_pull.tex. The main source includes that
generated appendix and a brief Results cross-reference. The source captions
and technical methods remain for the author's review.

Offline validation checked ideal controller behavior, first-noun copying,
constant plural choices, answer-position bias, and a known nonadditive rule.
Direct item-level OLS is also checked against the equivalent cell-weighted
calculation. No published figure or table was removed.
