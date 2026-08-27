#!/bin/bash

# Create single experiment configurations for all distributions and estimators
# This will generate directory structures and setup files for knot selection experiments

echo "Creating single experiment configurations..."

uv run experiments/create_single_experiment.py \
    --output_base_dir experiments/compare_knot_selection

echo "Single experiment configurations created successfully!"
echo ""
echo "Generated directories:"
echo "  - single_TruncatedNormal/"
echo "  - single_TruncatedGMMSymmetricThree/"
echo "  - single_TruncatedGMMAsymmetricThree/"
echo "  - single_TruncatedGMMFiveSpikes/"
echo "  - single_StepFunction/"
echo "  - single_Sinusoidal/"
echo ""
echo "Each directory contains:"
echo "  - logs/ (for experiment logs)"
echo "  - plots/ (for generated plots)"
echo "  - results/ (for experiment results)"
echo "  - setups/ (with 15 JSON configuration files: 5 estimators × 3 basis orders)"
