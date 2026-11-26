#!/usr/bin/env python3
"""
stats_forced_choice.py

Analyze counterbalanced forced-choice agreement results.

Inputs:
    Paired CSV files with one row per base_item_id and one model per file.

Primary outcome:
    paired_score = 1.0 if both A/B orders are correct,
                   0.5 if one order is correct,
                   0.0 if neither order is correct.

Unit of analysis and resampling:
    One complete counterbalanced item pair. Bootstrap resampling is done
    within each of the eight NP number-pattern strata. The same sampled
    items are used for every model, preserving paired model comparisons.

Outputs:
    forced_choice_summary.csv
    forced_choice_within_model_contrasts.csv
    forced_choice_between_model_contrasts.csv
    forced_choice_bootstrap_draws.csv
    forced_choice_analysis_manifest.json

Pandas and SciPy are used for data handling and descriptive statistics.
The resampling wrapper is explicit because no single standard function
implements both the required eight-stratum bootstrap and shared paired
resamples across models.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
from scipy import stats


REPO_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_INPUT_DIR = REPO_ROOT / "results" / "final"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "results" / "analysis" / "forced_choice_v1"
DEFAULT_INPUT_PATTERN = "*_paired_v1.csv"

PATTERN_ORDER = ["SSS", "PPP", "SSP", "PPS", "PSS", "SPP", "SPS", "PSP"]
CONGRUENCE_ORDER = ["congruent", "incongruent"]
REGIME_ORDER = [
    "both_match",
    "N3_mismatch_only",
    "N1_mismatch_only",
    "both_mismatch",
]

REQUIRED_COLUMNS = [
    "model",
    "base_item_id",
    "source_family_id",
    "condition",
    "checkpoint",
    "number_pattern",
    "cue_regime",
    "global_congruence",
    "order_a_answer",
    "order_a_correct",
    "order_b_answer",
    "order_b_correct",
    "paired_score",
    "response_pattern",
    "response_interface",
]

METADATA_COLUMNS = [
    "base_item_id",
    "source_family_id",
    "condition",
    "checkpoint",
    "number_pattern",
    "cue_regime",
    "global_congruence",
    "nearest_attractor_congruence",
    "np1_number",
    "np2_number",
    "np3_number",
    "np1_lexeme",
    "np2_lexeme",
    "np3_lexeme",
    "v2_lexeme",
    "v3_lexeme",
    "source_good",
    "source_bad",
]

METRICS = [
    "paired_score",
    "strict_both_correct",
    "order_inconsistency",
]

OUTPUT_NAMES = [
    "forced_choice_summary.csv",
    "forced_choice_within_model_contrasts.csv",
    "forced_choice_between_model_contrasts.csv",
    "forced_choice_bootstrap_draws.csv",
    "forced_choice_analysis_manifest.json",
]


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of a file."""
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)

    return digest.hexdigest()


def check_allowed(series: pd.Series, allowed: set[object], label: str) -> None:
    """Stop if a column contains values outside an allowed set."""
    unexpected = set(series.dropna().unique()) - allowed

    if unexpected:
        raise ValueError(f"Unexpected values in {label}: {sorted(unexpected)}")


def read_paired_csv(path: Path, expected_items: int) -> pd.DataFrame:
    """Read and validate one paired forced-choice CSV."""
    df = pd.read_csv(path)
    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]

    if missing:
        raise ValueError(f"{path.name}: missing required columns: {missing}")

    if expected_items > 0 and len(df) != expected_items:
        raise ValueError(
            f"{path.name}: expected {expected_items} rows, found {len(df)}"
        )

    if df["base_item_id"].isna().any():
        raise ValueError(f"{path.name}: base_item_id contains missing values")

    if df["base_item_id"].duplicated().any():
        raise ValueError(f"{path.name}: duplicate base_item_id values")

    model_labels = df["model"].dropna().astype(str).unique()

    if len(model_labels) != 1:
        raise ValueError(
            f"{path.name}: expected one model label, found {model_labels.tolist()}"
        )

    for column in ["order_a_correct", "order_b_correct"]:
        df[column] = pd.to_numeric(df[column], errors="raise").astype(int)
        check_allowed(df[column], {0, 1}, f"{path.name}:{column}")

    df["paired_score"] = pd.to_numeric(df["paired_score"], errors="raise")
    check_allowed(
        df["paired_score"],
        {0.0, 0.5, 1.0},
        f"{path.name}:paired_score",
    )

    for column in ["order_a_answer", "order_b_answer"]:
        if df[column].isna().any():
            raise ValueError(f"{path.name}: {column} contains missing values")

        df[column] = df[column].astype(str).str.upper()
        check_allowed(df[column], {"A", "B"}, f"{path.name}:{column}")

    expected_score = (df["order_a_correct"] + df["order_b_correct"]) / 2.0

    if not np.allclose(df["paired_score"], expected_score, atol=1e-12, rtol=0.0):
        raise ValueError(
            f"{path.name}: paired_score does not equal the mean of the two orders"
        )

    expected_pattern = df["order_a_answer"] + df["order_b_answer"]
    actual_pattern = df["response_pattern"].astype(str).str.upper()

    if not actual_pattern.equals(expected_pattern):
        raise ValueError(f"{path.name}: response_pattern does not match the answers")

    check_allowed(
        df["number_pattern"],
        set(PATTERN_ORDER),
        f"{path.name}:number_pattern",
    )
    check_allowed(
        df["global_congruence"],
        set(CONGRUENCE_ORDER),
        f"{path.name}:global_congruence",
    )
    check_allowed(
        df["cue_regime"],
        set(REGIME_ORDER),
        f"{path.name}:cue_regime",
    )

    if set(df["checkpoint"].dropna().astype(str).unique()) != {"V2"}:
        raise ValueError(f"{path.name}: expected only V2 rows")

    df["strict_both_correct"] = (
        (df["order_a_correct"] == 1) & (df["order_b_correct"] == 1)
    ).astype(int)
    df["order_inconsistency"] = (
        df["order_a_correct"] != df["order_b_correct"]
    ).astype(int)
    df["a_response_rate"] = (
        (df["order_a_answer"] == "A").astype(float)
        + (df["order_b_answer"] == "A").astype(float)
    ) / 2.0

    return df.sort_values("base_item_id").reset_index(drop=True)


def discover_inputs(
    explicit_inputs: list[Path],
    input_dir: Path,
    input_pattern: str,
) -> list[Path]:
    """Resolve explicit inputs or discover files using a glob pattern."""
    if explicit_inputs:
        paths = [path.resolve() for path in explicit_inputs]
    else:
        if not input_dir.exists():
            raise FileNotFoundError(f"Could not find input directory: {input_dir}")

        paths = sorted(input_dir.glob(input_pattern))

    if not paths:
        raise FileNotFoundError(
            f"No files matched {input_pattern!r} in {input_dir}"
        )

    missing = [path for path in paths if not path.exists()]

    if missing:
        raise FileNotFoundError(f"Could not find input files: {missing}")

    return paths


def validate_alignment(
    frames: list[pd.DataFrame],
) -> tuple[list[str], pd.DataFrame]:
    """Check that all models contain identical items and item metadata."""
    model_names = [str(df["model"].iloc[0]) for df in frames]

    if len(model_names) != len(set(model_names)):
        raise ValueError(f"Duplicate model labels across files: {model_names}")

    reference = frames[0]
    columns = [column for column in METADATA_COLUMNS if column in reference.columns]
    reference_metadata = reference[columns].reset_index(drop=True)

    for df, model_name in zip(frames[1:], model_names[1:]):
        if df["base_item_id"].tolist() != reference["base_item_id"].tolist():
            raise ValueError(f"{model_name}: base-item set differs from the reference")

        if any(column not in df.columns for column in columns):
            raise ValueError(f"{model_name}: required alignment metadata are missing")

        try:
            pd.testing.assert_frame_equal(
                df[columns].reset_index(drop=True),
                reference_metadata,
                check_dtype=False,
                check_exact=True,
            )
        except AssertionError as error:
            raise ValueError(
                f"{model_name}: item metadata differ from the reference model"
            ) from error

    return model_names, reference_metadata


def make_scopes(metadata: pd.DataFrame) -> dict[tuple[str, str], list[str]]:
    """Map each reporting scope to its member number-pattern strata."""
    pattern_map = (
        metadata[["number_pattern", "global_congruence", "cue_regime"]]
        .drop_duplicates()
        .set_index("number_pattern")
    )

    if set(pattern_map.index) != set(PATTERN_ORDER):
        missing = set(PATTERN_ORDER) - set(pattern_map.index)
        raise ValueError(f"Missing number-pattern strata: {sorted(missing)}")

    if pattern_map.index.duplicated().any():
        raise ValueError(
            "Each number pattern must map to one congruence label and one cue regime"
        )

    scopes = {("overall", "all"): PATTERN_ORDER.copy()}

    for label in CONGRUENCE_ORDER:
        scopes[("global_congruence", label)] = [
            pattern
            for pattern in PATTERN_ORDER
            if pattern_map.loc[pattern, "global_congruence"] == label
        ]

    for label in REGIME_ORDER:
        scopes[("cue_regime", label)] = [
            pattern
            for pattern in PATTERN_ORDER
            if pattern_map.loc[pattern, "cue_regime"] == label
        ]

    for pattern in PATTERN_ORDER:
        scopes[("number_pattern", pattern)] = [pattern]

    if any(len(patterns) == 0 for patterns in scopes.values()):
        raise ValueError("At least one reporting scope is empty")

    return scopes


def make_metric_matrices(
    frames: list[pd.DataFrame],
) -> dict[str, np.ndarray]:
    """Make aligned item-by-model matrices for bootstrap metrics."""
    matrices = {}

    for metric in METRICS:
        matrices[metric] = np.column_stack(
            [df[metric].to_numpy(dtype=float) for df in frames]
        )

    return matrices


def descriptive_se(values: np.ndarray) -> float:
    """Return SciPy's standard error, handling constant samples cleanly."""
    if len(values) < 2:
        return np.nan

    if np.all(values == values[0]):
        return 0.0

    return float(stats.sem(values))


def percentile_ci(
    values: np.ndarray,
    confidence: float,
) -> tuple[float, float]:
    """Return a two-sided percentile interval from bootstrap draws."""
    alpha = 1.0 - confidence
    low, high = np.quantile(values, [alpha / 2.0, 1.0 - (alpha / 2.0)])

    return float(low), float(high)


def bootstrap_scope_means(
    matrices: dict[str, np.ndarray],
    metadata: pd.DataFrame,
    scopes: dict[tuple[str, str], list[str]],
    n_bootstrap: int,
    seed: int,
    chunk_size: int,
) -> dict[tuple[str, str], np.ndarray]:
    """
    Generate shared stratified bootstrap means for all metrics and models.

    Output columns are metric-major: all models for paired_score, then all
    models for strict_both_correct, then all models for order_inconsistency.
    """
    if n_bootstrap <= 0:
        raise ValueError("--bootstrap must be greater than zero")

    if chunk_size <= 0:
        raise ValueError("--chunk-size must be greater than zero")

    rng = np.random.default_rng(seed)
    pattern_values = metadata["number_pattern"].astype(str).to_numpy()
    n_models = next(iter(matrices.values())).shape[1]
    n_columns = len(METRICS) * n_models

    stratum_draws = {}
    stratum_sizes = {}

    for pattern in PATTERN_ORDER:
        rows = np.flatnonzero(pattern_values == pattern)
        n_rows = len(rows)
        stratum_sizes[pattern] = n_rows

        values = np.concatenate(
            [matrices[metric][rows, :] for metric in METRICS],
            axis=1,
        )

        if values.shape != (n_rows, n_columns):
            raise ValueError(f"Unexpected matrix shape for stratum {pattern}")

        draws = np.empty((n_bootstrap, n_columns), dtype=float)

        for start in range(0, n_bootstrap, chunk_size):
            stop = min(start + chunk_size, n_bootstrap)
            sample_positions = rng.integers(
                0,
                n_rows,
                size=(stop - start, n_rows),
            )
            draws[start:stop] = values[sample_positions].mean(axis=1)

        stratum_draws[pattern] = draws

    scope_draws = {}

    for key, patterns in scopes.items():
        total = sum(stratum_sizes[pattern] for pattern in patterns)
        combined = np.zeros((n_bootstrap, n_columns), dtype=float)

        for pattern in patterns:
            weight = stratum_sizes[pattern] / total
            combined += weight * stratum_draws[pattern]

        scope_draws[key] = combined

    return scope_draws


def draw_column(metric: str, model_index: int, n_models: int) -> int:
    """Return the bootstrap column for a metric and model."""
    return (METRICS.index(metric) * n_models) + model_index


def make_summary(
    frames: list[pd.DataFrame],
    model_names: list[str],
    metadata: pd.DataFrame,
    scopes: dict[tuple[str, str], list[str]],
    scope_draws: dict[tuple[str, str], np.ndarray],
    confidence: float,
) -> pd.DataFrame:
    """Compute observed statistics and bootstrap intervals for every scope."""
    rows = []
    n_models = len(model_names)
    pattern_values = metadata["number_pattern"].astype(str)

    for model_index, (model_name, df) in enumerate(zip(model_names, frames)):
        for (scope_type, scope_value), patterns in scopes.items():
            mask = pattern_values.isin(patterns).to_numpy()
            sub = df.loc[mask]
            draws = scope_draws[(scope_type, scope_value)]

            row = {
                "model": model_name,
                "scope_type": scope_type,
                "scope_value": scope_value,
                "n_items": int(len(sub)),
                "paired_accuracy": float(sub["paired_score"].mean()),
                "paired_se": descriptive_se(sub["paired_score"].to_numpy(dtype=float)),
                "strict_both_correct": float(sub["strict_both_correct"].mean()),
                "strict_se": descriptive_se(
                    sub["strict_both_correct"].to_numpy(dtype=float)
                ),
                "order_inconsistency": float(sub["order_inconsistency"].mean()),
                "order_inconsistency_se": descriptive_se(
                    sub["order_inconsistency"].to_numpy(dtype=float)
                ),
                "order_a_accuracy": float(sub["order_a_correct"].mean()),
                "order_b_accuracy": float(sub["order_b_correct"].mean()),
                "a_response_rate": float(sub["a_response_rate"].mean()),
            }

            for metric, prefix in [
                ("paired_score", "paired"),
                ("strict_both_correct", "strict"),
                ("order_inconsistency", "order_inconsistency"),
            ]:
                column = draw_column(metric, model_index, n_models)
                low, high = percentile_ci(draws[:, column], confidence)
                row[f"{prefix}_ci95_lo"] = low
                row[f"{prefix}_ci95_hi"] = high

            rows.append(row)

    return pd.DataFrame(rows)


def get_estimate(
    summary: pd.DataFrame,
    model: str,
    scope_type: str,
    scope_value: str,
    metric: str,
) -> float:
    """Read one unique observed estimate from the summary table."""
    match = summary[
        (summary["model"] == model)
        & (summary["scope_type"] == scope_type)
        & (summary["scope_value"] == scope_value)
    ]

    if len(match) != 1:
        raise ValueError(
            f"Expected one row for {model}, {scope_type}={scope_value}"
        )

    return float(match[metric].iloc[0])


def make_within_model_contrasts(
    summary: pd.DataFrame,
    scope_draws: dict[tuple[str, str], np.ndarray],
    model_names: list[str],
    confidence: float,
) -> pd.DataFrame:
    """Compute the global congruent-minus-incongruent contrast for each model."""
    rows = []
    n_models = len(model_names)

    for model_index, model_name in enumerate(model_names):
        column = draw_column("paired_score", model_index, n_models)

        comparisons = [
            (
                "global_congruence",
                "congruent",
                "incongruent",
            )
        ]
        for family, level_1, level_2 in comparisons:
            estimate = get_estimate(
                summary,
                model_name,
                family,
                level_1,
                "paired_accuracy",
            ) - get_estimate(
                summary,
                model_name,
                family,
                level_2,
                "paired_accuracy",
            )
            draws = (
                scope_draws[(family, level_1)][:, column]
                - scope_draws[(family, level_2)][:, column]
            )
            low, high = percentile_ci(draws, confidence)

            rows.append(
                {
                    "model": model_name,
                    "contrast_family": family,
                    "level_1": level_1,
                    "level_2": level_2,
                    "estimate_level_1_minus_level_2": estimate,
                    "ci95_lo": low,
                    "ci95_hi": high,
                }
            )

    return pd.DataFrame(rows)


def make_between_model_contrasts(
    summary: pd.DataFrame,
    scope_draws: dict[tuple[str, str], np.ndarray],
    model_names: list[str],
    confidence: float,
) -> pd.DataFrame:
    """Compute paired model differences using the shared bootstrap draws."""
    rows = []
    n_models = len(model_names)
    model_index = {model: index for index, model in enumerate(model_names)}

    for model_1, model_2 in itertools.combinations(model_names, 2):
        column_1 = draw_column("paired_score", model_index[model_1], n_models)
        column_2 = draw_column("paired_score", model_index[model_2], n_models)

        for scope_type, scope_value in [
            ("overall", "all"),
            ("global_congruence", "congruent"),
            ("global_congruence", "incongruent"),
        ]:
            estimate = get_estimate(
                summary,
                model_1,
                scope_type,
                scope_value,
                "paired_accuracy",
            ) - get_estimate(
                summary,
                model_2,
                scope_type,
                scope_value,
                "paired_accuracy",
            )
            draws = (
                scope_draws[(scope_type, scope_value)][:, column_1]
                - scope_draws[(scope_type, scope_value)][:, column_2]
            )
            low, high = percentile_ci(draws, confidence)

            rows.append(
                {
                    "model_1": model_1,
                    "model_2": model_2,
                    "comparison": scope_type,
                    "scope_value": scope_value,
                    "estimate_model_1_minus_model_2": estimate,
                    "ci95_lo": low,
                    "ci95_hi": high,
                }
            )

        drop_1 = get_estimate(
            summary,
            model_1,
            "global_congruence",
            "congruent",
            "paired_accuracy",
        ) - get_estimate(
            summary,
            model_1,
            "global_congruence",
            "incongruent",
            "paired_accuracy",
        )
        drop_2 = get_estimate(
            summary,
            model_2,
            "global_congruence",
            "congruent",
            "paired_accuracy",
        ) - get_estimate(
            summary,
            model_2,
            "global_congruence",
            "incongruent",
            "paired_accuracy",
        )
        model_1_drop_draws = (
            scope_draws[("global_congruence", "congruent")][:, column_1]
            - scope_draws[("global_congruence", "incongruent")][:, column_1]
        )
        model_2_drop_draws = (
            scope_draws[("global_congruence", "congruent")][:, column_2]
            - scope_draws[("global_congruence", "incongruent")][:, column_2]
        )
        low, high = percentile_ci(
            model_1_drop_draws - model_2_drop_draws,
            confidence,
        )

        rows.append(
            {
                "model_1": model_1,
                "model_2": model_2,
                "comparison": "congruence_drop",
                "scope_value": "congruent_minus_incongruent",
                "estimate_model_1_minus_model_2": drop_1 - drop_2,
                "ci95_lo": low,
                "ci95_hi": high,
            }
        )

    return pd.DataFrame(rows)


def make_draw_table(
    scope_draws: dict[tuple[str, str], np.ndarray],
    model_names: list[str],
) -> pd.DataFrame:
    """Save compact primary bootstrap draws for audit and later contrasts."""
    n_models = len(model_names)
    n_bootstrap = len(scope_draws[("overall", "all")])
    tables = []

    for model_index, model_name in enumerate(model_names):
        paired = draw_column("paired_score", model_index, n_models)
        strict = draw_column("strict_both_correct", model_index, n_models)
        inconsistent = draw_column("order_inconsistency", model_index, n_models)

        table = pd.DataFrame(
            {
                "bootstrap_id": np.arange(1, n_bootstrap + 1),
                "model": model_name,
                "paired_overall": scope_draws[("overall", "all")][:, paired],
                "paired_congruent": scope_draws[
                    ("global_congruence", "congruent")
                ][:, paired],
                "paired_incongruent": scope_draws[
                    ("global_congruence", "incongruent")
                ][:, paired],
                "strict_overall": scope_draws[("overall", "all")][:, strict],
                "order_inconsistency_overall": scope_draws[("overall", "all")][
                    :, inconsistent
                ],
            }
        )
        table["paired_congruence_drop"] = (
            table["paired_congruent"] - table["paired_incongruent"]
        )

        for regime in REGIME_ORDER:
            table[f"paired_regime_{regime}"] = scope_draws[
                ("cue_regime", regime)
            ][:, paired]

        tables.append(table)

    return pd.concat(tables, ignore_index=True)


def check_outputs(output_dir: Path, overwrite: bool) -> dict[str, Path]:
    """Refuse silent overwriting and return output paths."""
    paths = {name: output_dir / name for name in OUTPUT_NAMES}
    existing = [path for path in paths.values() if path.exists()]

    if existing and not overwrite:
        listing = "\n".join(f"  {path}" for path in existing)
        raise FileExistsError(
            "Output files already exist. Use --overwrite to replace them:\n"
            f"{listing}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    return paths


def round_table(df: pd.DataFrame) -> pd.DataFrame:
    """Round floating output columns while retaining integer counts."""
    df = df.copy()
    float_columns = df.select_dtypes(include=["floating"]).columns
    df[float_columns] = df[float_columns].round(6)

    return df


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze counterbalanced forced-choice agreement results."
    )
    parser.add_argument(
        "--inputs",
        nargs="*",
        type=Path,
        default=[],
        help="Explicit paired CSV files; overrides input directory discovery.",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help="Directory searched when --inputs is not supplied.",
    )
    parser.add_argument(
        "--input-pattern",
        default=DEFAULT_INPUT_PATTERN,
        help="Glob pattern used within --input-dir.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory.",
    )
    parser.add_argument(
        "--bootstrap",
        type=int,
        default=10000,
        help="Number of stratified paired bootstrap samples.",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.95,
        help="Confidence level for percentile intervals.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed.",
    )
    parser.add_argument(
        "--expected-items",
        type=int,
        default=2000,
        help="Expected paired rows per model; use 0 to disable.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=250,
        help="Bootstrap samples processed at once.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacement of existing output files.",
    )
    args = parser.parse_args()

    if not 0.0 < args.confidence < 1.0:
        raise ValueError("--confidence must be between zero and one")

    paths = discover_inputs(args.inputs, args.input_dir, args.input_pattern)
    print(f"Found {len(paths)} paired input files.")

    frames = []

    for path in paths:
        print(f"Reading {path}")
        frame = read_paired_csv(path, args.expected_items)
        frames.append(frame)
        print(
            f"  {frame['model'].iloc[0]}: {len(frame)} rows, "
            f"paired mean={frame['paired_score'].mean():.4f}"
        )

    model_names, metadata = validate_alignment(frames)
    print(f"Validated {len(metadata)} matching items across {len(model_names)} models.")

    scopes = make_scopes(metadata)
    matrices = make_metric_matrices(frames)

    print(
        f"Running {args.bootstrap} shared stratified paired bootstrap samples "
        f"with seed {args.seed}."
    )
    scope_draws = bootstrap_scope_means(
        matrices=matrices,
        metadata=metadata,
        scopes=scopes,
        n_bootstrap=args.bootstrap,
        seed=args.seed,
        chunk_size=args.chunk_size,
    )

    summary = make_summary(
        frames=frames,
        model_names=model_names,
        metadata=metadata,
        scopes=scopes,
        scope_draws=scope_draws,
        confidence=args.confidence,
    )
    within_model = make_within_model_contrasts(
        summary,
        scope_draws,
        model_names,
        args.confidence,
    )
    between_model = make_between_model_contrasts(
        summary,
        scope_draws,
        model_names,
        args.confidence,
    )
    bootstrap_draws = make_draw_table(scope_draws, model_names)

    summary = round_table(summary)
    within_model = round_table(within_model)
    between_model = round_table(between_model)

    output_paths = check_outputs(args.output_dir, args.overwrite)

    summary.to_csv(output_paths["forced_choice_summary.csv"], index=False)
    within_model.to_csv(
        output_paths["forced_choice_within_model_contrasts.csv"],
        index=False,
    )
    between_model.to_csv(
        output_paths["forced_choice_between_model_contrasts.csv"],
        index=False,
    )
    bootstrap_draws.to_csv(
        output_paths["forced_choice_bootstrap_draws.csv"],
        index=False,
    )

    manifest = {
        "analysis_name": "counterbalanced_forced_choice_v1",
        "unit_of_analysis": "counterbalanced base item",
        "unit_of_resampling": "complete counterbalanced item pair",
        "stratification": PATTERN_ORDER,
        "bootstrap_samples": int(args.bootstrap),
        "confidence_level": float(args.confidence),
        "confidence_interval_method": "percentile",
        "random_seed": int(args.seed),
        "models": model_names,
        "n_aligned_items": int(len(metadata)),
        "stratum_sizes": {
            pattern: int((metadata["number_pattern"] == pattern).sum())
            for pattern in PATTERN_ORDER
        },
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
        },
        "inputs": [
            {
                "file_name": path.name,
                "sha256": sha256_file(path),
                "model": model_name,
                "n_rows": int(len(frame)),
                "response_interfaces": sorted(
                    frame["response_interface"].dropna().astype(str).unique()
                ),
            }
            for path, model_name, frame in zip(paths, model_names, frames)
        ],
    }

    with output_paths["forced_choice_analysis_manifest.json"].open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(manifest, handle, indent=2)
        handle.write("\n")

    for name in OUTPUT_NAMES:
        path = output_paths[name]
        print(f"Wrote {path}")

    primary = summary[
        (summary["scope_type"].isin(["overall", "global_congruence"]))
    ][
        [
            "model",
            "scope_type",
            "scope_value",
            "n_items",
            "paired_accuracy",
            "paired_ci95_lo",
            "paired_ci95_hi",
        ]
    ]
    drops = within_model[
        within_model["contrast_family"] == "global_congruence"
    ][
        [
            "model",
            "estimate_level_1_minus_level_2",
            "ci95_lo",
            "ci95_hi",
        ]
    ]

    print()
    print("Primary paired-accuracy results:")
    print(primary.to_string(index=False))
    print()
    print("Congruent minus incongruent contrasts:")
    print(drops.to_string(index=False))
    print()
    print("PASS: forced-choice analysis completed successfully.")


if __name__ == "__main__":
    main()
