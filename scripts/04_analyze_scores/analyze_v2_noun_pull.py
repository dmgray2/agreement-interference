#!/usr/bin/env python3
"""Exploratory V2 noun-number OLS associations from stored scores.

No model inference or API requests. Primary exploratory outcomes assign .5
to exact verb-score ties and to exact A/B label-score ties before averaging
answer orders. Historical scoring rules are retained as a sensitivity check.
The selected preregistered scores are never modified.
"""
import argparse
import hashlib
import json
import platform
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

VERSION = "v2_noun_pull_exploratory_v1"
# Explicit original cohort: no 405B, CE0, CE3, or nonce data.
MODEL_MAP = {
    "gpt2": "gpt2",
    "pythia410": "pythia-410m",
    "llama32_1b": "llama-3.2-1b",
    "llama32_3b": "llama-3.2-3b",
    "llama31_70b": "llama-3.1-70b",
    "qwen25_1.5b": "qwen2.5-1.5b",
    "qwen25_3b": "qwen2.5-3b",
    "qwen25_7b": "qwen2.5-7b",
    "qwen25_14b": "qwen2.5-14b",
    "qwen25_32b": "qwen2.5-32b",
    "qwen25_72b": "qwen2.5-72b",
}
BASE_MODELS = list(MODEL_MAP.values())
API_MODELS = ["gpt-4.1", "gpt-5", "gpt-5.6-sol",
              "opus-4.1", "opus-4.5", "opus-5"]
LABELS = dict(zip(BASE_MODELS, [
    "GPT-2 124M", "Pythia 410M", "Llama 3.2 1B", "Llama 3.2 3B",
    "Llama 3.1 70B", "Qwen2.5 1.5B", "Qwen2.5 3B", "Qwen2.5 7B",
    "Qwen2.5 14B", "Qwen2.5 32B", "Qwen2.5 72B"]))
LABELS.update(dict(zip(API_MODELS, [
    "GPT-4.1", "GPT-5", "GPT-5.6 Sol", "Opus 4.1", "Opus 4.5", "Opus 5"])))
TERMS = ["intercept", "NP1", "NP2", "NP3"]
COLORS = ["#D55E00", "#0072B2", "#009E73"]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def booleans(series):
    values = series.astype(str).str.lower().map(
        {"true": True, "false": False, "1": True, "0": False})
    require(values.notna().all(), "Invalid boolean values: " + series.name)
    return values.to_numpy(dtype=bool)


def numbers(frame, suffix):
    columns = ["np1_" + suffix, "np2_" + suffix, "np3_" + suffix]
    result = []
    for column in columns:
        values = frame[column].map({"S": 0, "P": 1, "sg": 0, "pl": 1})
        require(values.notna().all(), "Unknown noun number: " + column)
        result.append(values.to_numpy(dtype=float))
    return np.column_stack(result)


def align(frame, item_ids, id_field):
    require(not frame[id_field].duplicated().any(), "Duplicate " + id_field)
    require(set(frame[id_field]) == set(item_ids), "Item ID mismatch")
    return frame.set_index(id_field).loc[item_ids].reset_index()


def plural_choice(correctness, controller):
    return np.where(controller == 1, correctness, 1.0 - correctness)


def assemble(args):
    critical = pd.read_csv(args.critical_master, low_memory=False)
    forced = pd.read_csv(args.forced_master, low_memory=False)
    prompts = pd.read_csv(args.prompt_master, low_memory=False)
    reference = critical[
        (critical["model"] == "gpt2") & (critical["checkpoint"] == "V2")
    ].sort_values("pair_id")
    require(len(reference) == 2000, "Expected exactly 2,000 original V2 items")
    require(reference.family_id.nunique() == 2000, "Repeated V2 families")
    item_ids = reference.pair_id.tolist()
    x = numbers(reference, "num")
    patterns = np.array(["".join("P" if v else "S" for v in row) for row in x])
    require((patterns == reference.np_pattern.to_numpy()).all(),
            "Critical-region pattern metadata disagree")
    spec, primary, legacy, exclude, tie_rows = [], [], [], [], []

    for old_name, canonical in MODEL_MAP.items():
        frame = align(critical[
            (critical.model == old_name) & (critical.checkpoint == "V2")
        ].copy(), item_ids, "pair_id")
        require(np.array_equal(numbers(frame, "num"), x),
                "Number metadata differ across critical-region models")
        require(frame.family_id.nunique() == 2000, "Repeated families")
        margins = frame.delta_crit.to_numpy(dtype=float)
        correct = frame.correct_crit.to_numpy(dtype=float)
        require(np.isfinite(margins).all(), "Nonfinite critical-region scores")
        require(np.array_equal(correct, (margins > 0).astype(float)),
                "Critical-region correctness is not strict-positive-margin")
        ties = margins == 0
        old_y = plural_choice(correct, x[:, 1])
        new_y = np.where(ties, .5, old_y)
        spec.append(("critical_region", canonical))
        primary.append(new_y)
        legacy.append(old_y)
        exclude.append(~ties)
        tie_rows.append(dict(method="critical_region", model=canonical,
                             n_items=2000, numeric_tie_observations=int(ties.sum()),
                             n_numeric_comparisons=2000,
                             numeric_tie_rate=float(ties.mean()),
                             items_with_any_tie=int(ties.sum()),
                             historical_rule="ties incorrect"))

    require(set(BASE_MODELS + API_MODELS + ["gpt-3.5"]).issubset(set(forced.model)),
            "Missing selected forced-choice models")
    for canonical in BASE_MODELS + API_MODELS:
        frame = align(forced[forced.model == canonical].copy(),
                      item_ids, "base_item_id")
        require(np.array_equal(numbers(frame, "num"), x),
                "Number metadata differ across methods")
        require(not booleans(frame.technical_missing).any(),
                "Technical missingness in selected model")
        require(booleans(frame.valid_order_A).all()
                and booleans(frame.valid_order_B).all(),
                "Invalid response in an included noun-choice model")
        a = frame.correct_order_A.to_numpy(dtype=float)
        b = frame.correct_order_B.to_numpy(dtype=float)
        require(np.isin(a, [0, 1]).all() and np.isin(b, [0, 1]).all(),
                "Nonbinary answer-order correctness")
        require(np.array_equal((a+b)/2,
                               frame.counterbalanced_accuracy.to_numpy(float)),
                "Paired-score arithmetic mismatch")
        old_y = plural_choice((a+b)/2, x[:, 1])
        any_tie = np.zeros(2000, dtype=bool)
        new_a, new_b = a.copy(), b.copy()
        numeric_ties, comparisons, tie_rate = None, None, None
        if canonical in BASE_MODELS:
            raw = prompts[prompts.model == canonical].copy()
            require(len(raw) == 4000, "Expected 4,000 open-weight prompt rows")
            ties_by_order = []
            for label, stored_correct in [("A", a), ("B", b)]:
                order = align(raw[raw.grammatical_label == label].copy(),
                              item_ids, "base_item_id")
                require(np.array_equal(numbers(order, "number"), x),
                        "Prompt number metadata mismatch")
                require(np.array_equal(order.correct.to_numpy(float),
                                       stored_correct),
                        "Prompt scores do not match selected paired master")
                require(np.array_equal(
                    order.prefix.astype(str).str.strip().to_numpy(),
                    reference.prefix.astype(str).str.strip().to_numpy()),
                    "Prompt lexical prefixes differ from original items")
                pa = order.logprob_a.to_numpy(float)
                pb = order.logprob_b.to_numpy(float)
                require(np.isfinite(pa).all() and np.isfinite(pb).all(),
                        "Nonfinite A/B log probabilities")
                selected = np.where(pa >= pb, "A", "B")
                require(np.array_equal(selected, order.answer.to_numpy()),
                        "Stored A/B selections disagree with tie-selects-A rule")
                require(np.array_equal((selected == label).astype(float),
                                       stored_correct),
                        "Selected label and correctness disagree")
                ties = pa == pb
                ties_by_order.append(ties)
                if label == "A":
                    new_a[ties] = .5
                else:
                    new_b[ties] = .5
            any_tie = ties_by_order[0] | ties_by_order[1]
            numeric_ties = int(sum(t.sum() for t in ties_by_order))
            comparisons = 4000
            tie_rate = numeric_ties / comparisons
        spec.append(("forced_choice", canonical))
        primary.append(plural_choice((new_a+new_b)/2, x[:, 1]))
        legacy.append(old_y)
        exclude.append(~any_tie)
        tie_rows.append(dict(
            method="forced_choice", model=canonical, n_items=2000,
            numeric_tie_observations=numeric_ties,
            n_numeric_comparisons=comparisons, numeric_tie_rate=tie_rate,
            items_with_any_tie=(int(any_tie.sum())
                                if canonical in BASE_MODELS else None),
            historical_rule=("select A on tie" if canonical in BASE_MODELS
                             else "valid forced A/B tool response")))

    invalid = forced[forced.model == "gpt-3.5"]
    require(len(invalid) == 2000, "Expected GPT-3.5 accounting rows")
    require(not booleans(invalid.valid_order_A).any()
            and not booleans(invalid.valid_order_B).any(),
            "GPT-3.5 exclusion needs review: valid responses found")
    return (item_ids, x, patterns, spec, np.column_stack(primary),
            np.column_stack(legacy), np.column_stack(exclude), tie_rows)


def fit_and_bootstrap(x, patterns, outcomes, n_boot, seed, chunk_size):
    design = np.column_stack([np.ones(len(x)), x])
    levels = sorted(set(patterns))
    groups = [np.flatnonzero(patterns == p) for p in levels]
    cell_design = np.array([design[g[0]] for g in groups])
    counts = np.array([len(g) for g in groups])
    means = np.array([outcomes[g].mean(axis=0) for g in groups])
    # Exact OLS sufficient statistics; each cell contains an identical X row.
    projection = np.linalg.solve(
        design.T @ design, cell_design.T * counts)
    estimates = projection @ means
    require(np.allclose(estimates, np.linalg.lstsq(
        design, outcomes, rcond=None)[0], atol=1e-12, rtol=0),
        "Cell-weighted calculation does not match direct OLS")
    rng = np.random.default_rng(seed)
    cell_draws = np.empty((n_boot, len(levels), outcomes.shape[1]))
    # One shared item draw within each pattern for all models and both methods.
    for start in range(0, n_boot, chunk_size):
        stop = min(n_boot, start + chunk_size)
        for k, g in enumerate(groups):
            indices = rng.integers(0, len(g), size=(stop-start, len(g)))
            cell_draws[start:stop, k] = outcomes[g[indices]].mean(axis=1)
    draws = np.einsum("kt,btm->bkm", projection, cell_draws)
    return design, levels, counts, means, cell_design, estimates, draws, cell_draws


def make_plot(coefficients, path):
    plt.rcParams.update({"font.size": 8, "pdf.fonttype": 42,
                         "ps.fonttype": 42, "axes.spines.top": False,
                         "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(7.15, 5.65))
    panels = [("critical_region", BASE_MODELS, "Critical-region preference"),
              ("forced_choice", BASE_MODELS + API_MODELS,
               "Counterbalanced forced choice")]
    for ax, (method, models, title) in zip(axes, panels):
        for k, (term, color) in enumerate(zip(TERMS[1:], COLORS)):
            sub = coefficients[
                (coefficients.method == method) & (coefficients.term == term)
            ].set_index("model").loc[models]
            y = np.arange(len(models)) + (k-1)*.19
            estimate = sub.estimate.to_numpy()
            ax.errorbar(estimate, y,
                        xerr=np.array([estimate-sub.ci_low.to_numpy(),
                                       sub.ci_high.to_numpy()-estimate]),
                        fmt=["o", "s", "^"][k], ms=3.0, color=color,
                        elinewidth=.8, capsize=1.6, lw=0)
        ax.set_yticks(np.arange(len(models)), [LABELS[m] for m in models],
                      fontsize=7.0)
        ax.invert_yaxis()
        ax.set_ylim(len(models)-.35, -.65)
        ax.axvline(0, color=".45", lw=.75, zorder=0)
        ax.axvline(1, color=".7", ls=":", lw=.7, zorder=0)
        ax.set_xlim(-.16, 1.07)
        ax.set_xticks([0, .25, .5, .75, 1.0])
        ax.grid(axis="x", alpha=.15)
        ax.set_title(title, fontsize=8.5, pad=9)
        ax.tick_params(axis="y", length=0)
        if method == "forced_choice":
            ax.axhline(10.5, color=".8", lw=.7)
    handles = [Line2D([], [], marker=mark, color=color, ls="",
                      markersize=4, label=label)
               for mark, color, label in zip(
                   ["o", "s", "^"], COLORS,
                   ["NP1: non-controller", "NP2: controller",
                    "NP3: non-controller"])]
    fig.legend(handles=handles, loc="upper center", ncol=3,
               frameon=False, fontsize=7.5, bbox_to_anchor=(.52, 1.0))
    fig.supxlabel("Change in plural-choice rate (OLS coefficient)",
                  fontsize=8, y=.015)
    fig.subplots_adjust(left=.15, right=.99, bottom=.09, top=.895, wspace=.68)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    fig.savefig(path.with_suffix(".png"), dpi=250)
    plt.close(fig)


def appendix_tex(n_boot):
    return r"""% Generated by scripts/04_analyze_scores/analyze_v2_noun_pull.py.
% Technical methods/caption: review during the author's manuscript rewrite.
\begin{figure*}[p]
\section{Exploratory Noun-Number Associations at V2}
\label{app:v2-noun-pull}
\noindent
For each model and scoring method, an exploratory ordinary least-squares
regression predicts the plural-verb choice rate from the three noun numbers:
\[
\widehat{y}=\alpha+w_1N_1+w_2N_2+w_3N_3,
\qquad N_j=0\ \text{(singular)},\quad N_j=1\ \text{(plural)}.
\]
For a plural controller, the response equals correctness; for a singular
controller, it equals one minus correctness. Exact critical-region ties
are assigned .5. For open-weight forced choice, exact A/B score ties are
assigned .5 before averaging the two answer orders. Other responses retain
their recorded valid choices. These exploratory outcomes leave the
registered accuracy scores unchanged. The intercept is the fitted
all-singular baseline. All eight number patterns and all 2,000 items enter
each regression. The model is additive and does not include interactions.
Associations vary across number configurations, so a small average
coefficient need not imply absent sensitivity in every configuration.
Fitted cell means can fall outside $[0,1]$; the regression is a descriptive
summary, not a complete probability model. Pattern means, conditional
contrasts, and fit residuals are released alongside the coefficients.

\centering
\includegraphics[width=\textwidth]{figures/v2_noun_pull.pdf}
\caption{Exploratory associations between noun number and V2 verb choice.
A coefficient of .2 represents a 20-percentage-point increase in the fitted
plural-choice rate when that noun changes from singular to plural,
adjusting for the other two nouns under the additive model.
Perfect controller-based behavior would yield $w_2=1$ and $w_1=w_3=0$.
Intervals are pointwise 95\% percentile bootstrap intervals from
BOOTCOUNT item resamples within the eight number patterns, paired across
models and methods. Each forced-choice item retains its two answer orders.
All 11 original models are included; the right panel also includes six
proprietary systems with valid forced A/B tool responses. GPT-3.5 is
excluded because none of its responses passed strict A/B validation.
The two scoring methods use different tasks; answer-position bias can
mask noun-number sensitivity in forced choice. Coefficients summarize
behavioral associations, not internal causal weights.}
\label{fig:v2-noun-pull}
\end{figure*}
""".replace("BOOTCOUNT", f"{n_boot:,}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--critical-master", type=Path, default=Path(
        "data/scores/master_score_lists/master_scores_full.csv"))
    parser.add_argument("--forced-master", type=Path, default=Path(
        "results/final/master_forced_choice_scores_v1.csv"))
    parser.add_argument("--prompt-master", type=Path, default=Path(
        "data/scores/master_score_lists/"
        "master_forced_choice_open_weight_bfloat16.v1.csv"))
    parser.add_argument("--outdir", type=Path,
                        default=Path("results/analysis/v2_noun_pull_v1"))
    parser.add_argument("--figdir", type=Path,
                        default=Path("EACL paper/figures"))
    parser.add_argument("--tex-output", type=Path, default=Path(
        "EACL paper/tables/appendix_v2_noun_pull.tex"))
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20261118)
    parser.add_argument("--chunk-size", type=int, default=200)
    args = parser.parse_args()
    require(args.bootstrap >= 1000, "Use at least 1,000 bootstrap resamples")
    require(args.chunk_size > 0, "Chunk size must be positive")
    inputs = [args.critical_master, args.forced_master, args.prompt_master]
    input_hashes = {str(p): sha256(p) for p in inputs}
    (item_ids, x, patterns, spec, y, legacy,
     keep, tie_rows) = assemble(args)
    (design, levels, counts, means, cell_design, beta,
     draws, cell_draws) = fit_and_bootstrap(
         x, patterns, y, args.bootstrap, args.seed, args.chunk_size)
    ci = np.quantile(draws, [.025, .975], axis=0)
    coef_rows, sensitivity_rows, pattern_rows = [], [], []
    diagnostics, conditional_rows = [], []
    for j, (method, model) in enumerate(spec):
        base = dict(method=method, model=model, display_name=LABELS[model])
        for k, term in enumerate(TERMS):
            coef_rows.append(dict(
                **base, term=term, estimate=float(beta[k, j]),
                ci_low=float(ci[0, k, j]), ci_high=float(ci[1, k, j]),
                n_items=len(item_ids), tie_policy="half"))
        for policy, outcome, mask in [
                ("half", y[:, j], np.ones(len(y), dtype=bool)),
                ("historical", legacy[:, j], np.ones(len(y), dtype=bool)),
                ("exclude_any_tied_item", legacy[:, j], keep[:, j])]:
            require(np.linalg.matrix_rank(design[mask]) == 4,
                    "Rank-deficient sensitivity regression")
            b = np.linalg.lstsq(design[mask], outcome[mask], rcond=None)[0]
            for k, term in enumerate(TERMS):
                sensitivity_rows.append(dict(
                    **base, term=term, tie_policy=policy,
                    estimate=float(b[k]), n_items=int(mask.sum()),
                    difference_from_half=float(b[k]-beta[k, j])))
        fitted = cell_design @ beta[:, j]
        residuals = means[:, j] - fitted
        diagnostics.append(dict(
            **base, max_abs_pattern_residual=float(abs(residuals).max()),
            weighted_rms_pattern_residual=float(np.sqrt(
                np.average(residuals**2, weights=counts))),
            min_fitted_pattern=float(fitted.min()),
            max_fitted_pattern=float(fitted.max())))
        for k, pattern in enumerate(levels):
            pattern_rows.append(dict(
                **base, number_pattern=pattern, n_items=int(counts[k]),
                plural_choice_mean=float(means[k, j]),
                additive_fitted_mean=float(fitted[k]),
                additive_residual=float(residuals[k])))
        for term_index, term in enumerate(TERMS[1:]):
            for k, low_pattern in enumerate(levels):
                if low_pattern[term_index] != "S":
                    continue
                high_pattern = (low_pattern[:term_index] + "P"
                                + low_pattern[term_index+1:])
                h = levels.index(high_pattern)
                difference = cell_draws[:, h, j] - cell_draws[:, k, j]
                lo, hi = np.quantile(difference, [.025, .975])
                conditional_rows.append(dict(
                    **base, term=term, singular_pattern=low_pattern,
                    plural_pattern=high_pattern,
                    conditional_contrast=float(means[h, j]-means[k, j]),
                    ci_low=float(lo), ci_high=float(hi)))
    args.outdir.mkdir(parents=True, exist_ok=True)
    tables = {
        "coefficients.csv": pd.DataFrame(coef_rows),
        "tie_sensitivity.csv": pd.DataFrame(sensitivity_rows),
        "pattern_means.csv": pd.DataFrame(pattern_rows),
        "conditional_contrasts.csv": pd.DataFrame(conditional_rows),
        "fit_diagnostics.csv": pd.DataFrame(diagnostics),
        "tie_audit.csv": pd.DataFrame(tie_rows),
    }
    protected = {p.resolve() for p in inputs}
    for name, frame in tables.items():
        path = args.outdir/name
        require(path.resolve() not in protected, "Output would overwrite input")
        frame.to_csv(path, index=False, float_format="%.12g")
    figpath = args.figdir/"v2_noun_pull.pdf"
    make_plot(tables["coefficients.csv"], figpath)
    args.tex_output.parent.mkdir(parents=True, exist_ok=True)
    args.tex_output.write_text(appendix_tex(args.bootstrap), encoding="utf-8")
    manifest = {
        "analysis": VERSION,
        "status": "exploratory; not a preregistered confirmatory analysis",
        "inputs_sha256": input_hashes,
        "script_sha256": sha256(__file__),
        "model_selection": [{"method": a, "model": b} for a, b in spec],
        "excluded": {
            "gpt-3.5": "0/4,000 valid A/B responses; no recoverable verb choices",
            "405B_CE0_CE3_nonce": "outside original paper analysis cohort"},
        "response": "plural-verb choice rate; exact numeric ties assigned .5",
        "forced_choice": "mean of two answer-order outcomes, kept together",
        "predictors": "intercept plus NP1/NP2/NP3; singular=0, plural=1",
        "intercept": "fitted all-singular baseline; not a global plural bias",
        "model": "additive OLS; all 8 patterns weighted by observed item counts",
        "bootstrap": {
            "n_resamples": args.bootstrap, "seed": args.seed,
            "unit": "base item (one V2 item per family)",
            "stratification": dict(zip(levels, counts.tolist())),
            "paired_across_models_and_methods": True,
            "two_orders_resampled_together": True,
            "ci": "pointwise central 95% percentile; no multiplicity correction",
            "chunk_size": args.chunk_size},
        "tie_sensitivity": {
            "half": "primary exploratory outcome; each numeric tie contributes .5",
            "historical": "critical ties wrong; A/B ties select A",
            "exclude_any_tied_item":
                "drop a base item if its critical score ties or either A/B order ties",
            "intervals": "computed for primary half-tie coefficients only"},
        "interpretation": [
            "Behavioral associations, not internal causal weights.",
            "No precision rescoring; bfloat16 master inputs retained.",
            "Conditional contrasts and pattern residuals check additive summaries.",
            "Intervals condition on this vocabulary, template and selected models.",
            "Answer-position bias can mask noun-number sensitivity."],
        "versions": {
            "python": platform.python_version(), "numpy": np.__version__,
            "pandas": pd.__version__, "matplotlib": matplotlib.__version__},
        "outputs": [str(args.outdir/name) for name in tables]
                   + [str(figpath), str(figpath.with_suffix(".png")),
                      str(args.tex_output)],
    }
    for p, digest in input_hashes.items():
        require(sha256(p) == digest, "Input changed during analysis: " + p)
    (args.outdir/"analysis_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Analyzed {len(item_ids)} items for {len(spec)} model-method combinations.")
    print(f"Saved {args.outdir}, {figpath}, and {args.tex_output}")


if __name__ == "__main__":
    main()
