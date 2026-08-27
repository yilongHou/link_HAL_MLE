#!/bin/bash

# Script to remove all failed JSON files listed in experiments/uniform_convergence/indentified_failed_json.json
# This script will delete corrupted or malformed JSON files from the experiment results

echo "Starting removal of failed JSON files..."

# Check if indentified_failed_json.json exists
if [ ! -f "experiments/uniform_convergence/indentified_failed_json.json" ]; then
    echo "Error: experiments/uniform_convergence/indentified_failed_json.json not found!"
    exit 1
fi

# Count total files to be removed
total_files=$(jq length experiments/uniform_convergence/indentified_failed_json.json)
echo "Found $total_files files to remove"

# Counter for removed files
removed_count=0
failed_count=0

# Read each file path from the JSON array and remove it
while IFS= read -r file_path; do
    if [ -f "$file_path" ]; then
        echo "Removing: $file_path"
        rm "$file_path"
        if [ $? -eq 0 ]; then
            ((removed_count++))
        else
            echo "Failed to remove: $file_path"
            ((failed_count++))
        fi
    else
        echo "File not found (already removed?): $file_path"
        ((failed_count++))
    fi
done < <(jq -r '.[]' experiments/uniform_convergence/indentified_failed_json.json)

echo ""
echo "Removal completed!"
echo "Successfully removed: $removed_count files"
echo "Failed or not found: $failed_count files"
echo "Total processed: $((removed_count + failed_count)) files"

# Optional: Remove empty directories
echo ""
echo "Cleaning up empty directories..."
find experiments/uniform_convergence/results -type d -empty -delete 2>/dev/null
echo "Empty directory cleanup completed."
