
# for json_file in experiments/uniform_convergence/setups_test/*.json; do
#     uv run experiments/run_bulk_experiment.py \
#         "$json_file" \
#         --n-workers 5
# done

# for json_file in experiments/uniform_convergence/setups/*.json; do
#     uv run experiments/run_bulk_experiment.py \
#         "$json_file" \
#         --n-workers 20
# done

# uv run experiments/run_bulk_experiment.py \
#     experiments/uniform_convergence/setups_test/Sinusoidal_KDEEstimator_N200.json \
#     --n-workers 5

# uv run experiments/run_bulk_experiment.py \
#     experiments/uniform_convergence/setups/Sinusoidal_CVXPYEstimator_N200.json \
#     --n-workers 2

# Example: Run only files containing CVXPYEstimator in the filename

# for json_file in experiments/uniform_convergence/setups/*CVXPYEstimator_N100.json; do
# for json_file in experiments/uniform_convergence/setups/Sinusoidal_CVXPYEstimator_N1600.json; do

# for json_file in experiments/uniform_convergence/setups/*CVXPYEstimator_N6400*.json; do
#     uv run experiments/run_bulk_experiment.py \
#         "$json_file" \
#         --n-workers 12
# done

for json_file in $(ls -r experiments/uniform_convergence/setups/*CVXPYEstimator_N400*.json); do
    uv run experiments/run_bulk_experiment.py \
        "$json_file" \
        --n-workers 80
done


# uv run experiments/run_bulk_experiment.py \
#     experiments/uniform_convergence/setups_test/TruncatedNormal_CVXPYEstimator_N400.json \
#     --n-workers 6


for json_file in $(ls -r experiments/uniform_convergence/setups/*TrendFilteringADMMEstimator_N800.json); do
    uv run experiments/run_bulk_experiment.py \
        "$json_file" \
        --n-workers 30
done


for json_file in $(ls -r experiments/uniform_convergence/setups/*TrendFilteringADMMEstimator_N800.json); do
    echo "$json_file"
done
