#!/bin/bash
#
# This script runs the asymptotic efficiency analysis for all DGPs.
# It compares the HAL-MLE estimator against asymptotically efficient estimators
# for key population parameters like mean, median, and survival probability.

echo "Running Asymptotic Efficiency Analysis for all DGPs..."

# List of DGPs to analyze
DGPs=(
    "TruncatedNormal"
    "TruncatedGMMSymmetricThree"
    "TruncatedGMMAsymmetricThree"
    "TruncatedGMMFiveSpikes"
    "StepFunction"
    "Sinusoidal"
)

# Loop through each DGP and run the analysis script
for dgp in "${DGPs[@]}"; do
    echo "--------------------------------------------------"
    echo "Analyzing DGP: $dgp"
    echo "--------------------------------------------------"
    uv run experiments/uniform_convergence/asymptotic_efficiency_results.py --dgp "$dgp" --recompute_hal_density
    if [ $? -ne 0 ]; then
        echo "Error analyzing $dgp. Aborting."
        exit 1
    fi
done

echo "--------------------------------------------------"
echo "Asymptotic efficiency analysis completed for all DGPs."
echo "--------------------------------------------------"