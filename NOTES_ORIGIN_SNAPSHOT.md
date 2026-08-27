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
- `experiments/compare_knot_selection/` — result JSONs, setups, combinations, and knot-count CSVs for all six DGPs
- `experiments/uniform_convergence/results/` — CVXPY all N; TF / TFPP / KDE / LogSplines at N=800
- `experiments/uniform_convergence/targeted_results/`
- Summaries: `uniform_convergence_summary.csv`, `efficiency_analysis_results.json`
- Case study: `case_study/galaxies.csv`, `case_study/estimation_results/estimation_results.json`, `case_study/bootstrap_results.json`

See `README.md` for the experiment-to-script map and `uv run` commands.
