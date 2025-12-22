#!/usr/bin/env python
"""
Cellpose segmentation script for VTEA.

This script takes an input image and runs Cellpose segmentation,
saving the label image to the specified output path.

Usage:
    python cellpose_segment.py <input_path> <output_path> <diameter> <model> <flow_threshold> <cellprob_threshold>

Arguments:
    input_path: Path to input image file
    output_path: Path to save output label image
    diameter: Estimated cell diameter in pixels (or 0 for automatic)
    model: Cellpose model to use (cyto, nuclei, cyto2, etc.)
    flow_threshold: Flow threshold parameter (default 0.4)
    cellprob_threshold: Cell probability threshold (default 0.0)
"""

import sys
import numpy as np
from skimage import io
from cellpose import models

def run_cellpose(input_path, output_path, diameter, model_type, flow_threshold, cellprob_threshold):
    """
    Run Cellpose segmentation on the input image.

    Args:
        input_path: Path to input image
        output_path: Path to save label image
        diameter: Cell diameter in pixels
        model_type: Cellpose model name
        flow_threshold: Flow threshold
        cellprob_threshold: Cell probability threshold
    """
    try:
        print(f"Loading image from {input_path}")
        img = io.imread(input_path)

        # Convert to grayscale if needed
        if len(img.shape) == 3:
            if img.shape[2] == 3 or img.shape[2] == 4:
                # Convert RGB to grayscale
                img = np.mean(img[:, :, :3], axis=2)

        print(f"Image shape: {img.shape}, dtype: {img.dtype}")
        print(f"Running Cellpose with model={model_type}, diameter={diameter}")

        # Initialize model
        model = models.Cellpose(gpu=False, model_type=model_type)

        # Run segmentation
        masks, flows, styles, diams = model.eval(
            img,
            diameter=diameter if diameter > 0 else None,
            flow_threshold=flow_threshold,
            cellprob_threshold=cellprob_threshold,
            channels=[0, 0]  # grayscale
        )

        print(f"Segmentation complete. Found {masks.max()} objects.")

        # Save label image
        io.imsave(output_path, masks.astype(np.uint16), check_contrast=False)
        print(f"Saved label image to {output_path}")

        return 0

    except Exception as e:
        print(f"ERROR: {str(e)}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    if len(sys.argv) != 7:
        print("Usage: python cellpose_segment.py <input> <output> <diameter> <model> <flow_threshold> <cellprob_threshold>")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2]
    diameter = float(sys.argv[3])
    model_type = sys.argv[4]
    flow_threshold = float(sys.argv[5])
    cellprob_threshold = float(sys.argv[6])

    exit_code = run_cellpose(input_path, output_path, diameter, model_type, flow_threshold, cellprob_threshold)
    sys.exit(exit_code)
