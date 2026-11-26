# Scorers before the float32 log-softmax correction

Preserved on November 18, 2026 before modifying the active entry points:

| Archived file | Original path |
| --- | --- |
| score_minpairs.py | scripts/02_score_models/score_minpairs.py |
| run_open_weight_forced_choice_logprob.py | scripts/run_open_weight_forced_choice_logprob.py |

These files are byte-for-byte snapshots of the scripts present before the
precision correction. SHA256SUMS records their hashes. They establish what was
preserved locally; exact provenance for any particular historical run also
requires that run's archived invocation and environment.

Run from the repository root with the original arguments. These scripts do
not explicitly request float32 log-softmax. Their output dtype follows the
logits supplied by the model. The critical-region script counts exact ties
as incorrect; the forced-choice script selects A on an exact tie.

The active entry points now compute log-softmax in float32. Their model
weights and inference precision remain controlled by --dtype. No existing
scores, masters, or analyses were changed when making this correction.

Use separate output paths for experiments. Historical scorers can overwrite
outputs according to their original behavior; do not point them at the selected
paper inputs. See the root and scripts READMEs for sensitivity-check limits.
