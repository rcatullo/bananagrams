"""
Visual validation script to verify that masks are correctly aligned with input images.

Usage:
    python validate.py <shard_directory> [--output overlay_dir] [--max-samples N]
"""

import argparse
import json
import os
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def create_overlay(input_img: np.ndarray, mask_img: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """
    Create an overlay visualization showing the mask on the input image.
    
    Args:
        input_img: RGB input image
        mask_img: Grayscale mask
        alpha: Transparency of mask overlay (0-1)
    
    Returns:
        Overlay image with mask highlighted in red
    """
    if len(input_img.shape) == 2:
        input_img = cv2.cvtColor(input_img, cv2.COLOR_GRAY2RGB)
    
    mask_colored = np.zeros_like(input_img)
    mask_colored[:, :, 2] = mask_img 
    
    overlay = cv2.addWeighted(input_img, 1.0, mask_colored, alpha, 0)
    
    mask_binary = (mask_img > 128).astype(np.uint8)
    contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (0, 255, 0), 2)
    
    return overlay


def create_side_by_side(input_img: np.ndarray, mask_img: np.ndarray, overlay: np.ndarray) -> np.ndarray:
    """Create a side-by-side comparison image."""
    h = input_img.shape[0]
    w = input_img.shape[1]
    
    mask_rgb = cv2.cvtColor(mask_img, cv2.COLOR_GRAY2RGB)
    
    if mask_rgb.shape[:2] != (h, w):
        mask_rgb = cv2.resize(mask_rgb, (w, h))
    if overlay.shape[:2] != (h, w):
        overlay = cv2.resize(overlay, (w, h))
    
    combined = np.hstack([input_img, mask_rgb, overlay])
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(combined, "Input", (10, 30), font, 1, (255, 255, 255), 2)
    cv2.putText(combined, "Mask", (w + 10, 30), font, 1, (255, 255, 255), 2)
    cv2.putText(combined, "Overlay", (2*w + 10, 30), font, 1, (255, 255, 255), 2)
    
    return combined


def analyze_mask_quality(mask_img: np.ndarray, input_size: tuple, border_width: int = 10) -> dict:
    """
    Analyze mask quality metrics.
    
    Args:
        mask_img: Binary mask image
        input_size: Tuple of (height, width)
        border_width: Width of excluded border (used to adjust edge detection)
    
    Returns:
        Dictionary with quality metrics
    """
    h, w = input_size
    total_pixels = h * w
    mask_pixels = np.sum(mask_img > 128)
    
    coords = np.where(mask_img > 128)
    if len(coords[0]) == 0:
        return {
            'mask_coverage': 0.0,
            'bbox': None,
            'near_edges': False,
            'num_components': 0
        }
    
    y_min, y_max = coords[0].min(), coords[0].max()
    x_min, x_max = coords[1].min(), coords[1].max()
    
    # Check if mask is VERY close to edges (within border exclusion zone)
    # We expect masks won't be in the outermost border_width pixels
    # So only flag if mask is suspiciously close to the inner boundary
    edge_margin = border_width + 5  # Add 5px tolerance beyond excluded border
    near_edges = (
        x_min < edge_margin or y_min < edge_margin or
        x_max > w - edge_margin or y_max > h - edge_margin
    )
    
    # Count connected components
    mask_binary = (mask_img > 128).astype(np.uint8)
    num_components = cv2.connectedComponents(mask_binary)[0] - 1  # Subtract background
    
    return {
        'mask_coverage': (mask_pixels / total_pixels) * 100,
        'bbox': (x_min, y_min, x_max, y_max),
        'bbox_size': (x_max - x_min, y_max - y_min),
        'near_edges': near_edges,
        'num_components': num_components
    }


def validate_shard(shard_dir: str, output_dir: str = None, max_samples: int = None):
    """
    Validate all samples in a shard directory.
    
    Args:
        shard_dir: Path to shard directory containing extracted files
        output_dir: Optional output directory for overlay images
        max_samples: Maximum number of samples to process
    """
    shard_path = Path(shard_dir)
    if not shard_path.exists():
        print(f"Error: Shard directory '{shard_dir}' does not exist")
        return
    
    input_files = sorted(shard_path.glob("*.input.jpg"))
    if not input_files:
        print(f"No input files found in {shard_dir}")
        return
    
    print(f"Found {len(input_files)} samples in {shard_dir}")
    
    if max_samples:
        input_files = input_files[:max_samples]
        print(f"Processing first {max_samples} samples")
    
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        print(f"Saving overlays to {output_dir}")
    
    print(f"\n{'='*80}")
    print(f"{'Sample ID':<40} {'Coverage':<10} {'Components':<12} {'Edge?'}")
    print(f"{'='*80}")
    
    warnings = []
    
    for input_file in input_files:
        base = input_file.stem.replace(".input", "")
        mask_file = shard_path / f"{base}.mask.png"
        meta_file = shard_path / f"{base}.meta.json"
        text_file = shard_path / f"{base}.text.txt"
        
        if not mask_file.exists():
            print(f"Warning: Missing mask for {base}")
            continue
        
        input_img = np.array(Image.open(input_file))
        mask_img = np.array(Image.open(mask_file))
        
        if input_img.shape[:2] != mask_img.shape[:2]:
            warnings.append(f"Warning {base}: Dimension mismatch! Input {input_img.shape[:2]} != Mask {mask_img.shape[:2]}")
            continue
        
        metrics = analyze_mask_quality(mask_img, input_img.shape[:2])
        
        edit_type = "Unknown"
        if meta_file.exists():
            with open(meta_file) as f:
                meta = json.load(f)
                edit_type = meta.get('edit_type', 'Unknown')
        
        coverage_str = f"{metrics['mask_coverage']:.2f}%"
        components_str = str(metrics['num_components'])
        edge_str = "YES" if metrics['near_edges'] else "no"
        
        print(f"{base:<40} {coverage_str:<10} {components_str:<12} {edge_str}")
        
        if metrics['near_edges']:
            warnings.append(f"Warning {base}: Mask very close to edges (within excluded border zone)")
        if metrics['mask_coverage'] > 50:
            warnings.append(f"Warning {base}: Large mask coverage ({metrics['mask_coverage']:.1f}%) - verify alignment")
        if metrics['num_components'] > 20:
            warnings.append(f"Warning {base}: Many components ({metrics['num_components']}) - check if top-k filtering is working")
        
        if output_dir:
            overlay = create_overlay(input_img, mask_img)
            combined = create_side_by_side(input_img, mask_img, overlay)
            
            info_lines = [
                f"Edit: {edit_type}",
                f"Coverage: {metrics['mask_coverage']:.2f}%",
                f"Components: {metrics['num_components']}",
            ]
            
            if text_file.exists():
                with open(text_file) as f:
                    text = f.read().strip()
                    if len(text) > 100:
                        text = text[:97] + "..."
                    info_lines.append(f"Text: {text}")
            
            y_offset = 60
            for line in info_lines:
                cv2.putText(combined, line, (10, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                y_offset += 30
            
            output_file = output_path / f"{base}_validation.jpg"
            Image.fromarray(combined).save(output_file)
    
    print(f"{'='*80}\n")
    
    if warnings:
        print(f"\nWarnings ({len(warnings)}):")
        for warning in warnings:
            print(f"  {warning}")
    else:
        print("No issues detected! All masks appear properly aligned.")
    
    print(f"\nValidation complete!")


def main():
    parser = argparse.ArgumentParser(
        description="Validate mask alignment in dataset shard",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Validate and print metrics only
    python validate.py dev_extracted/shard-000000

    # Validate and create overlay images
    python validate.py dev_extracted/shard-000000 --output overlays/

    # Process only first 5 samples
    python validate.py dev_extracted/shard-000000 --output overlays/ --max-samples 5
        """
    )
    
    parser.add_argument(
        "shard_dir",
        help="Path to shard directory containing extracted .input.jpg and .mask.png files"
    )
    parser.add_argument(
        "--output", "-o",
        help="Output directory for overlay images (optional)"
    )
    parser.add_argument(
        "--max-samples", "-n",
        type=int,
        help="Maximum number of samples to process"
    )
    
    args = parser.parse_args()
    
    validate_shard(args.shard_dir, args.output, args.max_samples)


if __name__ == "__main__":
    main()

