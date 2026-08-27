
FPATHS=(

)

# Use the venv python
PYTHON_CMD="/Users/zzp/Desktop/Github/link_HAL_MLE/.venv/bin/python"

for FPATH in "${FPATHS[@]}"; do
    echo "Running experiment with setup file: $FPATH"
    $PYTHON_CMD -m experiments.run_experiment "$FPATH"
    $PYTHON_CMD -m experiments.visualize_experiment "$FPATH"
done