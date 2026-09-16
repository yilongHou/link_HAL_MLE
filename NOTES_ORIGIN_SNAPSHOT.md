# Origin snapshot notes

This tree was imported as a **fresh snapshot** of the GitHub default-branch working tree (no LFS history), then slimmed for figure reproduction.

## Intentionally omitted

- Git LFS / `.gitattributes` LFS rules
- Per-seed and other plot PNG dumps (regenerate via analysis scripts)
- `mosek.lic` (do not commit license files; obtain MOSEK separately if needed)
- `2602.16259v1.pdf` (use arXiv)
- `.DS_Store`, `texput.log`, and the old `recover.sh` skip-worktree helper
- Non-figure result dirs (e.g. KDE/TF at N≠800, `*_recon`, `*A2Layered*`, `TF_TFPP_HAL/` plot dumps)

## Kept for figure regeneration

- Source under `methods/`, `utils/`, `targeting/`, `cross_validation/`, `density_variance/`, and experiment scripts
- `experiments/compare_knot_selection/` — result JSONs, setups, combinations, and knot-count CSVs for all six DGPs, plus the per-iteration optimizer logs (`single_<DGP>/logs/*.log`) that the four `visualize_*.py` scripts plot from (re-included via a `.gitignore` exception to the global `*.log` rule)
- `experiments/uniform_convergence/results/` — CVXPY all N; TF / TFPP / KDE / LogSplines at N=800
- `experiments/uniform_convergence/targeted_results/`
- Summaries: `uniform_convergence_summary.csv`, `efficiency_analysis_results.json`
- Case study: `case_study/galaxies.csv`, `case_study/estimation_results/estimation_results.json`, `case_study/bootstrap_results.json`

## Removed in the September cleanup (exploratory work not reported in the paper)

The archive of the full development history (including everything below and the non-figure result dirs) is kept privately; this repository holds only what the paper's figures and tables need.

- Trend filtering with data-adaptive knot placement ("Algorithm 2", a dead end): `methods/non_HAL_method/TF_CVXPY_PP_A2/`, `experiments/uniform_convergence/create_algo2_setups.py`, the `*_TrendFilteringCVXPYPPA2Layered_N800.json` setups, and the corresponding estimator-registry entries and HAL-overlay plotting code in `experiments/run_bulk_experiment.py`, `cross_validation/optuna_hyperparam_selector.py`, and `experiments/uniform_convergence/bias_variance_mse_analysis.py`
- HAL-MLE optimizers that are not part of the reported knot-selection comparison: `methods/deep_learning_method/` (AutoDiff), `methods/first_order_method/{proximal_gradient_descent,projected_gradient_descent}/`, `methods/second_order_method/{proximal_newton_sdcd,proximal_newton_lbfgs_full}/` (the paper compares CVXPY, FISTA, proximal AdaGrad, proximal Newton, and proximal Newton L-BFGS)
- Abandoned `N=6400` runs (10 seeds each): the 12 `*_N6400` result/targeted-result dirs and 24 `*_N6400.json` setups
- Development notebooks and scratch files: `demo_tf_cvxpy_pp.ipynb`, `test_selector.ipynb`, `test_solver.ipynb`, `test_targeting.ipynb`, `tf_cv_landscape_norm_constraint.ipynb`, `tmp.sh`, `tmp_density_variance.py`
- Unreported analyses and one-off cluster helpers: `experiments/run_bootstrap_experiment.py` (simulation bootstrap; the case-study bootstrap lives in `test_case_study.ipynb`), `asymptotic_normality_results_parallel_compare{,_v2,_v3}.py`, `plot_oracle_for_5spike_n400.py`, `identify_failed_json.py` + `indentified_failed_json.{json,sh}`, `asymptotic_normality_results.sh`, `run_bulk_test_rerun.sh`, `run_bulk_test_reverse.sh`, `run_experiment.sh`

Kept although not used for the reported numbers: `methods/non_HAL_method/TF_ADMM/` (the specialized ADMM algorithm derived in the paper's trend-filtering appendix; the reported TF results use the CVXPY implementation, as that appendix states) and `experiments/uniform_convergence/scripts/` (the post-processing that was applied to the saved TF/TFPP outputs).

See `README.md` for the experiment-to-script map and `uv run` commands.
