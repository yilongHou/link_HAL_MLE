"""
Generate setup files for TrendFilteringCVXPYPPA2Layered (Algorithm 2) experiments.
Copies the random seeds from existing TrendFilteringCVXPYPP setups.
"""
import json
import os

# IMPORTANT:
# Do NOT hand-specify sampler parameters here.
# The canonical sampler parameters live in the existing TrendFilteringCVXPYPP setups.
# We copy the full sampler_setup (sampler + sampler_params + n_samples) from those files
# to ensure compatibility with the sampler class constructors used by run_bulk_experiment.
DGPS = [
    "TruncatedNormal",
    "TruncatedGMMSymmetricThree",
    "TruncatedGMMAsymmetricThree",
    "TruncatedGMMFiveSpikes",
    "StepFunction",
    "Sinusoidal",
]

SAMPLE_SIZES = [800]  # Focus on N=800 for now

def main():
    setups_dir = os.path.dirname(os.path.abspath(__file__)) + "/setups"
    
    for dgp_name in DGPS:
        for n_samples in SAMPLE_SIZES:
            # Load existing TFPP setup to get the random seeds
            existing_setup_file = f"{setups_dir}/{dgp_name}_TrendFilteringCVXPYPP_N{n_samples}.json"
            
            if not os.path.exists(existing_setup_file):
                print(f"Skipping {dgp_name} N={n_samples}: no existing TFPP setup found")
                continue
            
            with open(existing_setup_file, 'r') as f:
                existing_setup = json.load(f)
            
            if "sampler_setup" not in existing_setup:
                raise ValueError(f"Missing 'sampler_setup' in {existing_setup_file}")
            if "random_seeds" not in existing_setup:
                raise ValueError(f"Missing 'random_seeds' in {existing_setup_file}")

            # Create new setup for Algorithm 2
            new_setup = {
                "sampler_setup": existing_setup["sampler_setup"],
                "estimator_setup": {
                    "estimator": "TrendFilteringCVXPYPPA2Layered"
                },
                "random_seeds": existing_setup["random_seeds"]
            }

            # Ensure n_samples matches filename (defensive)
            new_setup["sampler_setup"]["n_samples"] = n_samples
            
            # Save new setup
            new_setup_file = f"{setups_dir}/{dgp_name}_TrendFilteringCVXPYPPA2Layered_N{n_samples}.json"
            with open(new_setup_file, 'w') as f:
                json.dump(new_setup, f, indent=2)
            
            print(f"Created: {new_setup_file}")

if __name__ == "__main__":
    main()

