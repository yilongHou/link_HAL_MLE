for json_file in $(ls -r experiments/uniform_convergence/setups/*CVXPYEstimator_N*.json); do
        if [[ "$json_file" == *"N6400"* ]]; then
                continue
        else
                uv run experiments/run_bulk_experiment.py \
                        "$json_file" \
                        --n-workers 8
        fi
done

