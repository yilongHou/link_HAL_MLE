#!/bin/bash
#
# This script runs the targeting step and asymptotic efficiency analysis for all DGPs.
# It applies targeting steps to HAL-MLE results and then analyzes efficiency
# for key population parameters like mean, median, and survival probability.

# Step 1: Run the targeting step for all DGPs
echo "Running Targeting M-Step for all DGPs..."
uv run experiments/uniform_convergence/asymptotic_efficiency_run_targeting_step.py
if [ $? -ne 0 ]; then
    echo "Error during targeting step. Aborting."
    exit 1
fi
echo "Targeting step completed."

# Step 2: Run analysis on the targeted results for each targeter
TARGETERS=(
    "mean"
    "second_moment"
    "survival_0_5"
    "median"
)

echo "Running Asymptotic Efficiency Analysis on Targeted Results..."

for targeter in "${TARGETERS[@]}"; do
    echo "--------------------------------------------------"
    echo "Analyzing for Targeter: $targeter"
    echo "--------------------------------------------------"
    uv run experiments/uniform_convergence/asymptotic_efficiency_results_with_targeting.py --targeter "$targeter"
    if [ $? -ne 0 ]; then
        echo "Error analyzing targeted results for $targeter. Aborting."
        exit 1
    fi
done

echo "--------------------------------------------------"
echo "Targeted analysis completed for all targeters."
echo "--------------------------------------------------"