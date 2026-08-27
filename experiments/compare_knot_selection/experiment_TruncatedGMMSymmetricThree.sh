FPATHS=(
    # "experiments/compare_knot_selection/single_TruncatedGMMSymmetricThree/setups/TruncatedGMMSymmetricThree_CVXPYEstimator_Order0.json"
    # "experiments/compare_knot_selection/single_TruncatedGMMSymmetricThree/setups/TruncatedGMMSymmetricThree_CVXPYEstimator_Order1.json"
    # "experiments/compare_knot_selection/single_TruncatedGMMSymmetricThree/setups/TruncatedGMMSymmetricThree_CVXPYEstimator_Order2.json"
    
    # "experiments/compare_knot_selection/single_TruncatedGMMSymmetricThree/setups/TruncatedGMMSymmetricThree_FISTAEstimator_Order0.json"
    # "experiments/compare_knot_selection/single_TruncatedGMMSymmetricThree/setups/TruncatedGMMSymmetricThree_FISTAEstimator_Order1.json"
    # "experiments/compare_knot_selection/single_TruncatedGMMSymmetricThree/setups/TruncatedGMMSymmetricThree_FISTAEstimator_Order2.json"

    # "experiments/compare_knot_selection/single_TruncatedGMMSymmetricThree/setups/TruncatedGMMSymmetricThree_ProximalAdaGradEstimator_Order0.json"
    # "experiments/compare_knot_selection/single_TruncatedGMMSymmetricThree/setups/TruncatedGMMSymmetricThree_ProximalAdaGradEstimator_Order1.json"
    # "experiments/compare_knot_selection/single_TruncatedGMMSymmetricThree/setups/TruncatedGMMSymmetricThree_ProximalAdaGradEstimator_Order2.json"

    # "experiments/compare_knot_selection/single_TruncatedGMMSymmetricThree/setups/TruncatedGMMSymmetricThree_ProximalNewtonEstimator_Order0.json"
    # "experiments/compare_knot_selection/single_TruncatedGMMSymmetricThree/setups/TruncatedGMMSymmetricThree_ProximalNewtonEstimator_Order1.json"
    # "experiments/compare_knot_selection/single_TruncatedGMMSymmetricThree/setups/TruncatedGMMSymmetricThree_ProximalNewtonEstimator_Order2.json"

    # "experiments/compare_knot_selection/single_TruncatedGMMSymmetricThree/setups/TruncatedGMMSymmetricThree_ProximalNewtonLBFGSEstimator_Order0.json"
    # "experiments/compare_knot_selection/single_TruncatedGMMSymmetricThree/setups/TruncatedGMMSymmetricThree_ProximalNewtonLBFGSEstimator_Order1.json"
    # "experiments/compare_knot_selection/single_TruncatedGMMSymmetricThree/setups/TruncatedGMMSymmetricThree_ProximalNewtonLBFGSEstimator_Order2.json"
)

source .venv/bin/activate

for FPATH in "${FPATHS[@]}"; do
    echo "Running experiment with setup file: $FPATH"
    python -m experiments.run_experiment "$FPATH"
    python -m experiments.visualize_experiment "$FPATH"
done
