"""
Aggregate evaluation results from multiple GPU workers.
"""

import argparse
import json
import os
import sys
from pathlib import Path


def aggregate_results(output_dir: str):
    """Aggregate inference results from all GPU workers."""
    output_path = Path(output_dir)

    if not output_path.exists():
        print(f"Error: Output directory {output_dir} does not exist")
        sys.exit(1)

    # Find all GPU result files
    gpu_result_files = sorted(output_path.glob("inference_results_gpu*.json"))

    if not gpu_result_files:
        print(f"Error: No GPU result files found in {output_dir}")
        print("Expected files like: inference_results_gpu0.json, inference_results_gpu1.json, etc.")
        sys.exit(1)

    print(f"Found {len(gpu_result_files)} GPU result file(s):")
    for f in gpu_result_files:
        print(f"  - {f.name}")

    # Aggregate all results
    all_results = []
    total_samples = 0

    for gpu_file in gpu_result_files:
        print(f"\nReading {gpu_file.name}...")
        with open(gpu_file, 'r') as f:
            gpu_results = json.load(f)

        samples_count = len(gpu_results)
        total_samples += samples_count
        all_results.extend(gpu_results)
        print(f"  Loaded {samples_count} samples")

    # Save merged results
    merged_file = output_path / "inference_results_merged.json"
    with open(merged_file, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'='*80}")
    print(f"Aggregation completed!")
    print(f"{'='*80}")
    print(f"Total samples: {total_samples}")
    print(f"Merged results saved to: {merged_file}")
    print(f"{'='*80}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Aggregate inference results from multiple GPUs")
    parser.add_argument("output_dir", type=str, help="Directory containing GPU result files")
    args = parser.parse_args()

    aggregate_results(args.output_dir)
