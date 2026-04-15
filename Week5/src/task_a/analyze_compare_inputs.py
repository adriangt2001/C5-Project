#!/usr/bin/env python3
"""
Quick analysis script to compare text-only vs text+image generation results
"""
import json
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pathlib import Path
import numpy as np

def analyze_compare_inputs_results(output_dir: str = "outputs/task_a/compare_inputs"):
    """Analyze the results of the input comparison experiment"""

    output_path = Path(output_dir)
    summary_file = output_path / "summary.json"

    if not summary_file.exists():
        print(f"Summary file not found: {summary_file}")
        return

    with open(summary_file, 'r') as f:
        summary = json.load(f)

    print("=== Input Comparison Analysis ===")
    print(f"Model: {summary['model']}")
    print(f"Device: {summary['device']}")
    print()

    for result in summary['results']:
        mode = result['mode']
        time = result['generation_time_seconds']
        num_images = len(result['image_paths'])

        print(f"Mode: {mode}")
        print(f"  Generation time: {time:.2f} seconds")
        print(f"  Images generated: {num_images}")
        print(f"  Average time per image: {time/num_images:.2f} seconds")
        print()

    # Display grid images if they exist
    caption_only_grid = output_path / "caption_only" / "prompt_00_grid.png"
    caption_and_image_grid = output_path / "caption_and_image" / "prompt_00_grid.png"

    if caption_only_grid.exists() and caption_and_image_grid.exists():
        print("Grid images generated successfully!")
        print(f"Text-only grid: {caption_only_grid}")
        print(f"Text+image grid: {caption_and_image_grid}")
    else:
        print("Warning: Grid images not found")

if __name__ == "__main__":
    analyze_compare_inputs_results()