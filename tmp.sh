for json_file in $(ls -r experiments/uniform_convergence/setups/*TruncatedNormal_CVXPYEstimator_N3200*.json); do
    uv run experiments/run_bulk_experiment.py \
        "$json_file" \
        --n-workers 31
done