#!/usr/bin/env Rscript
# mixed_effects.R
#
# Optional mixed-effects robustness checks for checkpoint-level accuracy.
#
# This script uses all three checkpoints in the benchmark: V3, V2, and COPULA.
# It is intended as a supplementary analysis, not as the source of the main
# paper tables. The main paper tables are produced by the Python scripts.
#
# Why use all three checkpoints here?
# Each sentence family contributes one item at each checkpoint. That gives the
# model repeated observations from the same family, so a family-level random
# intercept can do useful work. This is different from fitting a separate model
# at only one checkpoint, where each family contributes only one observation.
#
# Per-model model:
#   correct_crit ~ checkpoint + (1 | family_id)
#
# Combined interaction model:
#   correct_crit ~ checkpoint * model + (1 | family_id)
#
# The checkpoint reference level is V3. Coefficients for V2 and COPULA therefore
# estimate how the odds of a correct critical-region decision differ from V3.
#
# Inputs:
#   data/scores/master_score_lists/master_scores_full.csv
#
# Outputs:
#   data/scores/summaries/r_checkpoint_means.csv
#   data/scores/summaries/r_checkpoint_per_model.csv
#   data/scores/summaries/r_checkpoint_interaction.txt
#
# Run from the repository root:
#   Rscript scripts/04_analyze_scores/mixed_effects.R
#
# Optional path overrides:
#   Rscript scripts/04_analyze_scores/mixed_effects.R \
#     --master data/scores/master_score_lists/master_scores_full.csv \
#     --out-dir data/scores/summaries


# ---- command-line arguments --------------------------------------------------

get_arg <- function(flag, default) {
  args <- commandArgs(trailingOnly = TRUE)
  hit <- which(args == flag)

  if (length(hit) == 0) {
    return(default)
  }

  if (hit[1] == length(args)) {
    stop(paste("Missing value after", flag), call. = FALSE)
  }

  args[hit[1] + 1]
}

master_path <- get_arg(
  "--master",
  file.path("data", "scores", "master_score_lists", "master_scores_full.csv")
)

out_dir <- get_arg(
  "--out-dir",
  file.path("data", "scores", "summaries")
)


# ---- setup ------------------------------------------------------------------

dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

suppressPackageStartupMessages({
  library(lme4)
  library(broom.mixed)
})

required_columns <- c(
  "model",
  "family_id",
  "checkpoint",
  "correct_crit"
)

sig_label <- function(p_value) {
  if (is.na(p_value)) return(NA_character_)
  if (p_value < 0.001) return("***")
  if (p_value < 0.01) return("**")
  if (p_value < 0.05) return("*")
  "ns"
}

check_required_columns <- function(d, required_columns) {
  missing <- setdiff(required_columns, names(d))

  if (length(missing) > 0) {
    stop(
      paste("Missing required columns:", paste(missing, collapse = ", ")),
      call. = FALSE
    )
  }
}

cat("Reading", master_path, "\n")

if (!file.exists(master_path)) {
  stop(paste("Could not find input file:", master_path), call. = FALSE)
}

d <- read.csv(master_path)
check_required_columns(d, required_columns)

d <- subset(d, checkpoint %in% c("V3", "V2", "COPULA"))
d <- d[!is.na(d$correct_crit), ]

d$correct_crit <- as.integer(d$correct_crit)
d$family_id <- factor(d$family_id)
d$model <- factor(d$model)
d$checkpoint <- factor(d$checkpoint, levels = c("V3", "V2", "COPULA"))

cat("Rows:", nrow(d), "\n")
cat("Families:", length(unique(d$family_id)), "\n")
cat("Models:", length(unique(d$model)), "\n")
cat("Checkpoints:\n")
print(table(d$checkpoint))


# ---- descriptive checkpoint means ------------------------------------------

checkpoint_means <- aggregate(
  correct_crit ~ model + checkpoint,
  data = d,
  FUN = mean
)

names(checkpoint_means)[names(checkpoint_means) == "correct_crit"] <- "acc_crit"

checkpoint_counts <- aggregate(
  family_id ~ model + checkpoint,
  data = d,
  FUN = function(x) length(unique(x))
)

names(checkpoint_counts)[names(checkpoint_counts) == "family_id"] <- "n_families"

checkpoint_means <- merge(
  checkpoint_means,
  checkpoint_counts,
  by = c("model", "checkpoint")
)

checkpoint_means <- checkpoint_means[order(checkpoint_means$model, checkpoint_means$checkpoint), ]
checkpoint_means$acc_crit <- round(checkpoint_means$acc_crit, 4)

means_path <- file.path(out_dir, "r_checkpoint_means.csv")
write.csv(checkpoint_means, means_path, row.names = FALSE)
cat("Wrote", means_path, "\n")


# ---- per-model repeated-measures mixed models ------------------------------

fit_one_model <- function(model_name, d_model) {
  cat("Fitting checkpoint model for", model_name, "...\n")

  fit <- glmer(
    correct_crit ~ checkpoint + (1 | family_id),
    data = d_model,
    family = binomial,
    control = glmerControl(optimizer = "bobyqa")
  )

  tab <- broom.mixed::tidy(
    fit,
    effects = "fixed",
    conf.int = TRUE,
    conf.method = "Wald",
    exponentiate = TRUE
  )

  wanted <- tab[tab$term %in% c("checkpointV2", "checkpointCOPULA"), ]

  out <- data.frame(
    model = model_name,
    term = wanted$term,
    odds_ratio = wanted$estimate,
    ci_low = wanted$conf.low,
    ci_high = wanted$conf.high,
    p_value = wanted$p.value,
    sig_label = vapply(wanted$p.value, sig_label, character(1)),
    n_obs = nrow(d_model),
    n_families = length(unique(d_model$family_id))
  )

  out
}

per_model_rows <- list()

for (model_name in levels(d$model)) {
  d_model <- subset(d, model == model_name)
  per_model_rows[[model_name]] <- fit_one_model(model_name, d_model)
}

per_model_df <- do.call(rbind, per_model_rows)
rownames(per_model_df) <- NULL

numeric_columns <- c("odds_ratio", "ci_low", "ci_high", "p_value")
for (column in numeric_columns) {
  per_model_df[[column]] <- round(per_model_df[[column]], 6)
}

per_model_path <- file.path(out_dir, "r_checkpoint_per_model.csv")
write.csv(per_model_df, per_model_path, row.names = FALSE)
cat("Wrote", per_model_path, "\n")
print(per_model_df)


# ---- combined checkpoint-by-model interaction model -------------------------

cat("Fitting combined checkpoint-by-model interaction model...\n")

combined_fit <- glmer(
  correct_crit ~ checkpoint * model + (1 | family_id),
  data = d,
  family = binomial,
  control = glmerControl(optimizer = "bobyqa")
)

interaction_path <- file.path(out_dir, "r_checkpoint_interaction.txt")

sink(interaction_path)
cat("Mixed-effects logistic regression over all three checkpoints\n")
cat("Formula: correct_crit ~ checkpoint * model + (1 | family_id)\n")
cat("Checkpoint reference level: V3\n\n")

print(summary(combined_fit))

cat("\n\nOdds ratios with 95% Wald CIs:\n")
print(
  broom.mixed::tidy(
    combined_fit,
    effects = "fixed",
    conf.int = TRUE,
    conf.method = "Wald",
    exponentiate = TRUE
  )
)
sink()

cat("Wrote", interaction_path, "\n")
cat("Done.\n")
