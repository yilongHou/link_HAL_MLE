import json
import numpy as np
import os
import glob



results_dir = "experiments/uniform_convergence/bootstrap_variance_estimation/results/TruncatedGMMFiveSpikes_CVXPYEstimator_N800"
include_term = "boot"

# Find all JSON files in subdirectories that contain the include_term
all_jsons = []

for root, dirs, files in os.walk(results_dir):
    # Check if the current directory path contains the include_term
    if include_term in root:
        if "6400" in root:
            continue
        for file in files:
            if file.endswith('.json'):
                json_path = os.path.join(root, file)
                all_jsons.append(json_path)


print(f"Found {len(all_jsons)} JSON files")

# config_info_jsons = []
failed_jsons = []

for json_path in all_jsons:
    dgp_name = json_path.split("/")[3].split("_")[0]
    n_sample = json_path.split("/")[3].split("_")[-1][1:]
    
    try:
        with open(json_path, 'r') as f:
            result = json.load(f)
        
        try:
            density = result["HAL_results"]["estimated_density"]
            # Check for None values
            if density is None or (isinstance(density, (list, np.ndarray)) and None in density):
                print(f"None values found in {json_path}")
                raise Exception("Contains None values")
            
            # Check for NaN values
            if isinstance(density, (list, np.ndarray)):
                if np.any(np.isnan(density)):
                    print(f"NaN values found in {json_path}")
                    raise Exception("Contains NaN values")
            elif np.isnan(density):
                print(f"NaN value found in {json_path}")
                raise Exception("Contains NaN values")

            # Check for extreme values
            if np.max(density) >= 500:
                print("Extreme value detect, solver failure expected")
                raise Exception("Contains extreme values")

        except KeyError:
            print(f"KeyError in {json_path}")
            failed_jsons.append(json_path)
            density = np.nan
            
    except json.JSONDecodeError as e:
        print(f"JSONDecodeError in {json_path}: {e}")
        failed_jsons.append(json_path)
        density = np.nan
    except Exception as e:
        print(f"Unexpected error in {json_path}: {e}")
        failed_jsons.append(json_path)
        density = np.nan
    
    # config_info_jsons.append({
    #     "dgp_name": dgp_name,
    #     "n_sample": int(n_sample),
    #     "density": density
    # })

print(len(failed_jsons))

# Save failed JSONs to a file for debugging
with open("experiments/uniform_convergence/indentified_failed_json.json", "w") as f:
    json.dump(failed_jsons, f, indent=4)

