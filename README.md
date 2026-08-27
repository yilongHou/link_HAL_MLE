# Experiment Reproduction for HAL-MLE Log-Splines Density Estimation (Part I: Univariate)

> **Origin clean snapshot (source of truth).** This private Origin repository is a slim working copy for regenerating paper figures from kept Monte Carlo JSON — not a full mirror of the public GitHub history. There is **no Git LFS**. Per-seed density PNG dumps, license files, and other bulky non-figure artifacts were omitted. TFPP `seed_*.json` files had unused `HAL_results.Hinv` matrices stripped (figure scripts such as `bias_variance_mse_analysis.py` read `estimated_density` / reconstruction fields, not `Hinv`). Paper: [arXiv 2602.16259](https://arxiv.org/abs/2602.16259).

This repository accompanies the final paper [HAL-MLE Log-Splines Density Estimation (Part I: Univariate)](https://arxiv.org/pdf/2602.16259). This README maps each experiment **by name** (and figure/caption identity) to the scripts in this repo and gives bash commands.

Section and figure numbers differ between the arXiv preprint and the Bernoulli manuscript. Use the experiment names below; do not look experiments up by preprint or journal numbering.

## Scope

The paper reports three experiment groups:

1. Optimization / knot-selection for HAL-MLE solvers.
2. Monte Carlo simulations for HAL-MLE theory and method comparisons (uniform convergence, pointwise normality, efficiency, EIC coverage, and the `n=800` method comparison).
3. Galaxy-velocity case study.

The six simulation DGPs are:

- Truncated Normal (`TruncatedNormal`)
- Truncated GMM, symmetric three modes (`TruncatedGMMSymmetricThree`)
- Truncated GMM, asymmetric three modes (`TruncatedGMMAsymmetricThree`)
- Truncated GMM, five spikes (`TruncatedGMMFiveSpikes`)
- Step function (`StepFunction`)
- Sinusoidal (`Sinusoidal`)

## What "minimum reproducible version" means here

The paper's main simulation settings use:

- `1000` Monte Carlo replicates for the HAL-MLE simulations
- `50` Optuna trials for cross-validation
- `5` CV folds

The commands below keep the same experiment structure, DGPs, estimands, and post-processing scripts, but reduce compute cost to:

- `NUM_SEEDS=20`
- `N_TRIALS=10`
- `CV_FOLDS=5`

To move closer to the paper's final figures, do increase `NUM_SEEDS` to `1000` and `N_TRIALS` to `50` and run on a big cluster.

## Environment

Run everything from the repository root.

```bash
uv sync

export NUM_SEEDS=20
export N_TRIALS=10
export CV_FOLDS=5
export N_WORKERS=4
export SETUP_DIR=experiments/uniform_convergence/setups_minimum
```

Notes:

- All commands below use `uv run`.
- The HAL-MLE experiments in Section 7 use the CVXPY-based estimator. MOSEK is the default solver, and when MOSEK fails the workflow falls back to ECOS and then SCS, as formalized in the HALDensity package.
- **LogSplines dependency:** The LogSplines comparison method wraps the external R `logspline` package via `rpy2`. Our current local environment uses `logspline` version `2.1.22`, but this R dependency is not pinned or lockfile-managed in this repository, so `LogSplinesEstimator` support remains commented out by default in `experiments/run_experiment.py` and `experiments/run_bulk_experiment.py`. The remaining methods (HAL-MLE, TF, TFPP, KDE) work without R.
- The case study notebook uses the provided `case_study/bootstrap_results.json` so that the bootstrap-based figure can be reproduced without rerunning the bootstrap from scratch.

## Experiment-to-Script Map

Look up an experiment by **name**. The second column is the figure/caption identity in the paper, not a preprint or journal number.

| Experiment | Figure / caption identity | Raw experiment scripts | Analysis / plotting scripts |
| --- | --- | --- | --- |
| Optimization / knot-selection (Truncated Normal, 2nd-order basis) | Optimization-algorithm comparison (loss and knot-selection vs iteration and vs FLOP) | `experiments/create_single_experiment.py`, `experiments/run_experiment.py` | `experiments/compare_knot_selection/visualize_loss_per_iter.py`, `experiments/compare_knot_selection/visualize_knot_selection_per_iter.py`, `experiments/compare_knot_selection/visualize_loss_per_flop.py`, `experiments/compare_knot_selection/visualize_knot_selection_per_flop.py` |
| Uniform convergence (sup-norm) | HAL-MLE sup-norm error decay and scaling-law table | `experiments/create_bulk_experiment.py`, `experiments/run_bulk_experiment.py` | `experiments/uniform_convergence/uniform_convergence_results.py` |
| Pointwise asymptotic normality / delta-method density CI | Pointwise CI width and coverage for the HAL-MLE density | `experiments/create_bulk_experiment.py`, `experiments/run_bulk_experiment.py` | `experiments/uniform_convergence/asymptotic_normality_results_parallel.py` |
| Asymptotic efficiency of plug-in HAL-MLE vs HAL-TMLE (mean, median, survival at 0.5, second moment) | Efficiency panels for the four estimands; six-DGP appendix panels | `experiments/create_bulk_experiment.py`, `experiments/run_bulk_experiment.py`, `experiments/uniform_convergence/asymptotic_efficiency_run_targeting_step.py` | `experiments/uniform_convergence/asymptotic_efficiency_results.py`, `experiments/uniform_convergence/asymptotic_efficiency_results_with_targeting.py`, `experiments/uniform_convergence/asymptotic_efficiency_compairson.py` |
| EIC-based coverage for those four estimands | EIC standard-error coverage tables for mean, median, survival at 0.5, and second moment | uses targeted results from the efficiency experiment | `experiments/uniform_convergence/targeting_estimand_variance_result.py` |
| n=800 method comparison (HAL-MLE, TF, TFPP, LogSplines, KDE) | Bias, variance, and MSE comparison across DGPs at `n=800` | `experiments/create_bulk_experiment.py`, `experiments/run_bulk_experiment.py` | `experiments/uniform_convergence/bias_variance_mse_analysis.py` |
| Galaxy-velocity case study | Galaxy-velocity density and estimand figures | `test_case_study.ipynb` | `test_case_study.ipynb` |

## 1. Optimization / knot-selection (Truncated Normal, 2nd-order basis)

The optimization / knot-selection experiment is the Truncated Normal DGP with 2nd-order basis functions. The repo setup generator creates all DGPs and all basis orders, but this figure only needs the `TruncatedNormal` order-2 runs.

```bash
uv run python experiments/create_single_experiment.py \
  --output_base_dir experiments/compare_knot_selection

for setup in experiments/compare_knot_selection/single_TruncatedNormal/setups/*Order2.json; do
  uv run python experiments/run_experiment.py "$setup"
done

uv run python experiments/compare_knot_selection/visualize_loss_per_iter.py \
  --dgp TruncatedNormal \
  --figure-dir paper/resources/optimization_algorithms/per_iter \
  --save-legend

uv run python experiments/compare_knot_selection/visualize_knot_selection_per_iter.py \
  --dgp TruncatedNormal \
  --figure-dir paper/resources/optimization_algorithms/per_iter \
  --save-legend

uv run python experiments/compare_knot_selection/visualize_loss_per_flop.py \
  --dgp TruncatedNormal \
  --figure-dir paper/resources/optimization_algorithms/per_flop \
  --save-legend

uv run python experiments/compare_knot_selection/visualize_knot_selection_per_flop.py \
  --dgp TruncatedNormal \
  --figure-dir paper/resources/optimization_algorithms/per_flop
```

Outputs for the paper figure are written under `paper/resources/optimization_algorithms/`.

If you also want the remaining five DGP panels, rerun the visualization scripts with `--dgp all` after generating the corresponding single-experiment results.

## Shared HAL-MLE Monte Carlo sweep

Uniform convergence, pointwise asymptotic normality, asymptotic efficiency, and EIC-based coverage all start from the same HAL-MLE Monte Carlo sweep: six DGPs, sample sizes `25, 50, 100, 200, 400, 800, 1600, 3200`, and the HAL-MLE estimator (`CVXPYEstimator`).

```bash
uv run python experiments/create_bulk_experiment.py \
  --output_dir "$SETUP_DIR" \
  --num_seeds "$NUM_SEEDS" \
  --seed_start 42

for setup in ${SETUP_DIR}/*_CVXPYEstimator_N*.json; do
  uv run python experiments/run_bulk_experiment.py "$setup" \
    --n-workers "$N_WORKERS" \
    --n-trials "$N_TRIALS" \
    --cv-folds "$CV_FOLDS"
done
```

These commands write per-seed JSON results to `experiments/uniform_convergence/results/`.

## 2. Uniform convergence (sup-norm)

This experiment checks the sup-norm error decay of the HAL-MLE.

```bash
uv run python experiments/uniform_convergence/uniform_convergence_results.py
```

Outputs:

- Plots in `paper/resources/density_uniform_convergence/`
- Summary CSV in `experiments/uniform_convergence/uniform_convergence_summary.csv`
- Scaling-law table in `paper/resources/density_uniform_convergence/uniform_convergence_scaling_table.tex`

## 3. Pointwise asymptotic normality / delta-method density CI

This experiment evaluates pointwise CI width and coverage for the HAL-MLE density using the delta-method variance estimator.

```bash
uv run python experiments/uniform_convergence/asymptotic_normality_results_parallel.py \
  --n-jobs "$N_WORKERS" \
  --max-files "$NUM_SEEDS"
```

Outputs:

- Plots in `paper/resources/density_asymptotic_normality_and_var_est/`
- Coverage table in `paper/resources/density_asymptotic_normality_and_var_est/coverage_analysis_table.tex`
- Caches in `experiments/uniform_convergence/cache/`

## 4. Asymptotic efficiency of plug-in HAL-MLE vs HAL-TMLE

The paper compares the plug-in HAL-MLE and HAL-TMLE against the asymptotically efficient estimators for:

- mean
- median
- survival at `0.5`
- second moment

Run the plug-in analysis first, then generate targeted results, then combine them into the paper figure panels.

```bash
uv run python experiments/uniform_convergence/asymptotic_efficiency_results.py

uv run python experiments/uniform_convergence/asymptotic_efficiency_run_targeting_step.py \
  --results_dir experiments/uniform_convergence/results \
  --output_dir experiments/uniform_convergence/targeted_results

mkdir -p experiments/uniform_convergence/targeted_plots

for targeter in mean second_moment survival_0_5 median; do
  uv run python experiments/uniform_convergence/asymptotic_efficiency_results_with_targeting.py \
    --targeter "$targeter" \
    --results_dir experiments/uniform_convergence/targeted_results \
    --plot_dir experiments/uniform_convergence/targeted_plots
done

uv run python experiments/uniform_convergence/asymptotic_efficiency_compairson.py \
  --original_results experiments/uniform_convergence/efficiency_analysis_results.json \
  --targeting_dir experiments/uniform_convergence/targeted_plots \
  --save_dir paper/resources/density_asymptotic_efficiency
```

Outputs:

- Plug-in efficiency summaries in `experiments/uniform_convergence/efficiency_analysis_results.json`
- Targeted per-estimand summaries in `experiments/uniform_convergence/targeted_plots/`
- Final comparison panels in `paper/resources/density_asymptotic_efficiency/`

The main-text efficiency panel is `TruncatedGMMAsymmetricThree`. The same script also writes the full six-DGP panels.

## 5. EIC-based coverage for those four estimands

This step uses the targeted results from the efficiency experiment and evaluates coverage of the EIC-based standard errors for the same four estimands.

```bash
uv run python experiments/uniform_convergence/targeting_estimand_variance_result.py \
  --targeted_dir experiments/uniform_convergence/targeted_results \
  --n-jobs "$N_WORKERS" \
  --latex-table
```

Outputs:

- Plots in `paper/resources/target_estimand_variance/`
- Coverage tables in:
  - `paper/resources/target_estimand_variance/coverage_targeting_mean.tex`
  - `paper/resources/target_estimand_variance/coverage_targeting_second_moment.tex`
  - `paper/resources/target_estimand_variance/coverage_targeting_survival_0_5.tex`
  - `paper/resources/target_estimand_variance/coverage_targeting_median.tex`
- Caches in `experiments/uniform_convergence/cache_targeting_variance/`

## 6. n=800 method comparison (HAL-MLE, TF, TFPP, LogSplines, KDE)

The paper compares HAL-MLE against:

- TF
- TFPP
- LogSplines
- KDE

at sample size `n=800`.

### Raw runs

HAL-MLE at `n=800` is already included in the shared HAL-MLE Monte Carlo sweep above. Run the additional TF, TFPP, LogSplines, and KDE jobs as follows:

```bash
for setup in \
  ${SETUP_DIR}/*_TrendFilteringADMMEstimator_N800.json \
  ${SETUP_DIR}/*_TrendFilteringCVXPYPP_N800.json \
  ${SETUP_DIR}/*_LogSplinesEstimator_N800.json \
  ${SETUP_DIR}/*_KDEEstimator_N800.json; do
  uv run python experiments/run_bulk_experiment.py "$setup" \
    --n-workers "$N_WORKERS" \
    --n-trials "$N_TRIALS" \
    --cv-folds "$CV_FOLDS"
done
```

### Generate comparison figures

Run this only after the HAL-MLE, TF, TFPP, KDE, and LogSplines `n=800` results all exist:

```bash
uv run python experiments/uniform_convergence/bias_variance_mse_analysis.py \
  --sample-size 800 \
  --results-dir experiments/uniform_convergence/results \
  --save-dir paper/resources/density_bias_variance_mse_analysis \
  --with-tfpp
```

Outputs:

- `paper/resources/density_bias_variance_mse_analysis/methods_compare_bias_across_dgps_N800.png`
- `paper/resources/density_bias_variance_mse_analysis/methods_compare_variance_across_dgps_N800.png`
- `paper/resources/density_bias_variance_mse_analysis/methods_compare_mse_across_dgps_N800.png`

## 7. Galaxy-velocity case study

The case study is implemented in the notebook `test_case_study.ipynb`. It uses:

- `case_study/galaxies.csv`
- `case_study/bootstrap_results.json`

To execute the notebook non-interactively:

```bash
uv run jupyter nbconvert --to notebook --execute test_case_study.ipynb --inplace
```

Outputs are written to `paper/resources/case_study/`:

- `bootstrap_vs_delta_method_variance_estimation.png`
- `hal_mle_vs_hal_tmle_for_mean.png`
- `hal_mle_vs_hal_tmle_for_median.png`
- `hal_mle_vs_hal_tmle_for_survival.png`

If you prefer to inspect the notebook interactively, open it with:

```bash
uv run jupyter lab test_case_study.ipynb
```
