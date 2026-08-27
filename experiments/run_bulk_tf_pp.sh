for f in experiments/uniform_convergence/setups/*_TrendFilteringCVXPYPP_N800.json; do
  uv run python experiments/run_bulk_experiment.py "$f" --n-workers 30
done