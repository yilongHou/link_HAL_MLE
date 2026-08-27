#!/usr/bin/env python3
"""
Targeting Step Runner for HAL-MLE Experiments

This script applies targeting steps to existing HAL-MLE experiment results.
It performs one-step M-step updates with targeting basis functions for:
- Mean (first moment)  
- Second moment
- Survival at 0.5
- Median

The script reads from experiments/uniform_convergence/results and saves
targeted results to experiments/uniform_convergence/targeted_results
with the same folder structure and __target=<name>.json suffixes.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path
from tqdm import tqdm
import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings('ignore')

# Add the project root to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

# Import targeting learners
from targeting.moments.learner import MomentsTargetLearner
from targeting.survival.learner import SurvivalTargetLearner  
from targeting.median.learner import MedianTargetLearner


def get_targeter_config(targeter_name):
    """
    Get configuration for each targeter type.
    
    Args:
        targeter_name: Name of the targeter ('mean', 'second_moment', 'survival_0_5', 'median')
        
    Returns:
        dict: Configuration with 'learner_class' and 'kwargs'
    """
    configs = {
        'mean': {
            'learner_class': MomentsTargetLearner,
            'kwargs': {'x_moment': 1}
        },
        'second_moment': {
            'learner_class': MomentsTargetLearner,
            'kwargs': {'x_moment': 2}
        },
        'survival_0_5': {
            'learner_class': SurvivalTargetLearner,
            'kwargs': {'targeting_points': [0.5]}
        },
        'median': {
            'learner_class': MedianTargetLearner,
            'kwargs': {}  # MedianTargetLearner computes median internally
        }
    }
    
    if targeter_name not in configs:
        raise ValueError(f"Unknown targeter: {targeter_name}. Available: {list(configs.keys())}")
    
    return configs[targeter_name]


def apply_targeting_step(result_data, targeter_name, norm_constraint_multiplier=5, basis_order=None, verbose=False):
    """
    Apply targeting step to a single experiment result.
    
    Args:
        result_data: Loaded JSON data from experiment
        targeter_name: Name of the targeter to apply
        norm_constraint_multiplier: Multiplier for the cross-validated norm constraint (default: 5)
        basis_order: Order of basis functions (if None, extract from hyperparams)
        verbose: Whether to show verbose error output
        
    Returns:
        dict: Updated result data with targeted density
    """
    try:
        # Extract data from JSON
        hal_results = result_data['HAL_results']
        hyperparams = result_data.get('hyperparams', {})
        
        # Extract adaptive norm constraint from cross-validation results
        cv_norm_constraint = hyperparams.get('norm_constraint', 20)  # Fallback to 20
        adaptive_norm_constraint = cv_norm_constraint * norm_constraint_multiplier
        
        # Extract basis order if not provided
        if basis_order is None:
            basis_order = hyperparams.get('basis_order', 0)  # Fallback to 0
        
        # print(f"  Using adaptive norm constraint: {cv_norm_constraint} * {norm_constraint_multiplier} = {adaptive_norm_constraint}")
        
        # Prepare inputs for M-step
        uncensored_augmented = pd.DataFrame({'W1': hal_results['data_points']})
        # IMPORTANT: Use ALL data points as knots, not just the selected ones
        # The original HAL fit used all data points as potential knots
        grid_points_hal_selected = np.array(hal_results['data_points'])
        old_theta = np.array(hal_results['theta_hat'])
        
        # Validate dimensions (no debug output)
        expected_basis_cols = basis_order + len(grid_points_hal_selected)
        expected_theta_length = 1 + expected_basis_cols
        
        # Get targeter configuration
        config = get_targeter_config(targeter_name)
        learner_class = config['learner_class']
        learner_kwargs = config['kwargs'].copy()
        
        # For median targeting, we need to pass old_theta and grid_points_hal_selected
        if targeter_name == 'median':
            learner_kwargs['_old_theta'] = old_theta
            learner_kwargs['_grid_points_hal_selected'] = grid_points_hal_selected
        
        # Instantiate learner
        learner = learner_class(norm_constraint=adaptive_norm_constraint, basis_order=basis_order)
        
        # Run M-step
        m_step_start = time.time()
        
        m_step_results = learner.run_m_step(
            uncensored_augmented=uncensored_augmented,
            grid_points_hal_selected=grid_points_hal_selected,
            old_theta=old_theta,
            **learner_kwargs
        )
        
        m_step_time = time.time() - m_step_start
        
        # Create new result data with completely replaced HAL_results
        import copy
        new_result_data = copy.deepcopy(result_data)
        
        # Completely replace HAL_results with targeted results
        # Keep ONLY original data_points for efficient estimates, everything else is new
        original_data_points = hal_results['data_points']  # Keep for efficient estimates
        
        new_result_data['HAL_results'] = {
            # Keep original data for efficient estimates in downstream analysis
            'data_points': original_data_points,
            
            # Everything else is new targeted results
            'estimated_density': m_step_results['estimated_density'].tolist(),
            'grid_points': m_step_results['grid_midpoints'].tolist()
        }
        
        # Add targeting metadata (keeping this separate for debugging/traceability)
        new_result_data['targeting_info'] = {
            'targeter': targeter_name,
            'method': 'one_step_m_step',
            'cv_norm_constraint': cv_norm_constraint,
            'norm_constraint_multiplier': norm_constraint_multiplier,
            'adaptive_norm_constraint': adaptive_norm_constraint,
            'basis_order': basis_order,
            'm_step_time_seconds': m_step_time,
            'm_step_output_keys': list(m_step_results.keys()),
            'success': True,
            'note': 'Original HAL_results completely replaced with targeted results'
        }
        
        return new_result_data
        
    except Exception as e:
        # Return error metadata instead of failing completely
        import copy
        error_result_data = copy.deepcopy(result_data)
        
        # Clear HAL_results to avoid confusion in error cases
        if 'HAL_results' in error_result_data:
            original_data_points = error_result_data['HAL_results']['data_points']
            error_result_data['HAL_results'] = {
                'data_points': original_data_points,
                'targeting_failed': True
            }
        
        error_result_data['targeting_info'] = {
            'targeter': targeter_name,
            'method': 'one_step_m_step',
            'norm_constraint_multiplier': norm_constraint_multiplier,
            'basis_order': basis_order,
            'error': str(e),
            'success': False
        }
        return error_result_data


def process_single_file(input_file_path, output_dir, targeters, overwrite=False, 
                       norm_constraint_multiplier=5, basis_order=None, pbar=None, verbose=False):
    """
    Process a single JSON file and apply targeting for all specified targeters.
    
    Args:
        input_file_path: Path to input JSON file
        output_dir: Base output directory
        targeters: List of targeter names to apply
        overwrite: Whether to overwrite existing files
        norm_constraint_multiplier: Multiplier for the cross-validated norm constraint
        basis_order: Basis function order (if None, extract from hyperparams)
        pbar: Optional progress bar to update
        verbose: Whether to show verbose output
    """
    try:
        # Load input data
        with open(input_file_path, 'r') as f:
            result_data = json.load(f)
        
        # Determine relative path structure for mirroring
        input_path = Path(input_file_path)
        # Find the position of 'results' in the path
        results_idx = None
        for i, part in enumerate(input_path.parts):
            if part == 'results':
                results_idx = i
                break
        
        if results_idx is None:
            return  # Skip files not in results directory structure
        
        # Extract the path after 'results/'
        relative_path = Path(*input_path.parts[results_idx+1:])
        
        # Create output directory structure
        output_file_dir = Path(output_dir) / relative_path.parent
        output_file_dir.mkdir(parents=True, exist_ok=True)
        
        # Get base filename without extension
        base_filename = input_path.stem
        
        # Process each targeter
        for targeter in targeters:
            if pbar:
                pbar.set_description(f"Processing {Path(input_file_path).name} - {targeter}")
            output_filename = f"{base_filename}__target={targeter}.json"
            output_file_path = output_file_dir / output_filename
            
            # Skip if file exists and not overwriting
            if output_file_path.exists() and not overwrite:
                continue
            
            # Apply targeting
            targeted_result = apply_targeting_step(
                result_data, 
                targeter, 
                norm_constraint_multiplier=norm_constraint_multiplier,
                basis_order=basis_order,
                verbose=verbose
            )
            
            # Save result
            with open(output_file_path, 'w') as f:
                json.dump(targeted_result, f, indent=2)
                
            if pbar:
                pbar.update(0)  # Trigger refresh for description update
            
    except Exception as e:
        # Handle file processing errors
        if verbose and pbar:
            pbar.write(f"Error processing {Path(input_file_path).name}: {str(e)}")
        elif pbar:
            pbar.set_description(f"Error in {Path(input_file_path).name}")
        elif verbose:
            print(f"Error processing {Path(input_file_path).name}: {str(e)}")
        pass


def find_experiment_files(results_dir, dgp_filter=None):
    """
    Find all experiment JSON files in the results directory.
    
    Args:
        results_dir: Path to results directory
        dgp_filter: Optional DGP name to filter by
        
    Returns:
        list: List of JSON file paths
    """
    json_files = []
    
    for root, dirs, files in os.walk(results_dir):
        # Filter by DGP if specified
        if dgp_filter:
            if dgp_filter not in root:
                continue
        
        # Only process CVXPYEstimator directories
        if 'CVXPYEstimator' not in root:
            continue
            
        # Find JSON files
        for file in files:
            if file.endswith('.json'):
                json_files.append(os.path.join(root, file))
    
    return sorted(json_files)


def main():
    """Main function to run the targeting step."""
    parser = argparse.ArgumentParser(description="Apply targeting steps to HAL-MLE experiment results")
    parser.add_argument("--dgp", type=str, help="Filter by specific DGP name")
    parser.add_argument("--results_dir", type=str, 
                       default="experiments/uniform_convergence/results",
                       help="Input results directory")
    parser.add_argument("--output_dir", type=str,
                       default="experiments/uniform_convergence/targeted_results", 
                       help="Output directory for targeted results")
    parser.add_argument("--targets", type=str,
                       default="mean,second_moment,survival_0_5,median",
                       help="Comma-separated list of targeters to apply")
    parser.add_argument("--overwrite", action="store_true",
                       help="Overwrite existing targeted files")
    parser.add_argument("--norm_constraint_multiplier", type=float, default=5,
                       help="Multiplier for the cross-validated norm constraint (default: 5)")
    parser.add_argument("--basis_order", type=int, default=None,
                       help="Order of basis functions (if not specified, will be extracted from hyperparams)")
    parser.add_argument("--verbose", action="store_true",
                       help="Show verbose output including error details")
    
    args = parser.parse_args()
    
    # Parse target list
    targeters = [t.strip() for t in args.targets.split(',')]
    valid_targeters = ['mean', 'second_moment', 'survival_0_5', 'median']
    
    for targeter in targeters:
        if targeter not in valid_targeters:
            print(f"Error: Invalid targeter '{targeter}'. Valid options: {valid_targeters}")
            sys.exit(1)
    
    print(f"Starting targeting step runner...")
    print(f"Input directory: {args.results_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Targeters: {targeters}")
    print(f"DGP filter: {args.dgp or 'None'}")
    print(f"Overwrite: {args.overwrite}")
    print("-" * 50)
    
    # Find all experiment files
    experiment_files = find_experiment_files(args.results_dir, args.dgp)
    
    if not experiment_files:
        print(f"No experiment files found in {args.results_dir}")
        if args.dgp:
            print(f"With DGP filter: {args.dgp}")
        print("Make sure the results directory exists and contains CVXPYEstimator subdirectories with .json files")
        return
    
    print(f"Found {len(experiment_files)} experiment files to process")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Calculate total operations for better progress tracking
    total_operations = len(experiment_files) * len(targeters)
    
    # Initialize counters
    files_processed = 0
    files_with_errors = 0
    
    # Process each file with detailed progress tracking
    with tqdm(total=total_operations, desc="Applying targeting steps") as pbar:
        for input_file in experiment_files:
            try:
                process_single_file(
                    input_file_path=input_file,
                    output_dir=args.output_dir,
                    targeters=targeters,
                    overwrite=args.overwrite,
                    norm_constraint_multiplier=args.norm_constraint_multiplier,
                    basis_order=args.basis_order,
                    pbar=pbar,
                    verbose=args.verbose
                )
                files_processed += 1
                # Update progress for each targeter processed
                pbar.update(len(targeters))
            except Exception as e:
                files_with_errors += 1
                # Still update progress even for failed files
                pbar.update(len(targeters))
                error_msg = str(e)[:50] + "..." if len(str(e)) > 50 else str(e)
                pbar.set_description(f"Error in {Path(input_file).name}: {error_msg}")
                if args.verbose:
                    pbar.write(f"Full error for {Path(input_file).name}: {str(e)}")
                
        # Final status update
        pbar.set_description(f"Completed: {files_processed} files processed, {files_with_errors} errors")
    
    print(f"\n{'-'*50}")
    print(f"Targeting step completed!")
    print(f"Files processed: {files_processed}")
    if files_with_errors > 0:
        print(f"Files with errors: {files_with_errors}")
    print(f"Targeters applied: {', '.join(targeters)}")
    print(f"Results saved to: {args.output_dir}")
    if files_with_errors > 0 and not args.verbose:
        print("Use --verbose flag to see detailed error information")


if __name__ == "__main__":
    main()