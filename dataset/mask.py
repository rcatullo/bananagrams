"""
Mask generation module for computing alignment and difference masks.

This module handles:
- Feature-based image alignment (SIFT)
- Mask computation from aligned images
- Edge artifact detection and removal
- Noise reduction via component filtering
"""

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
from PIL import Image
from config import load_config

CONFIG = load_config()

BLUR_KERNEL_SIZE = CONFIG["blur_kernel_size"]
MORPHOLOGICAL_KERNEL_SIZE = CONFIG["morphological_kernel_size"]
MASK_THRESHOLD = CONFIG["mask_threshold"]
TOP_K_COMPONENTS = CONFIG["top_k_components"]

@dataclass
class AlignmentResult:
    """Result of image alignment with quality metrics."""
    aligned_image: Optional[np.ndarray]
    success: bool
    num_matches: int = 0
    inlier_ratio: float = 0.0
    failure_reason: str = ""


@dataclass
class MaskResult:
    """Result of mask computation."""
    mask: Optional[Image.Image]
    success: bool
    alignment_result: Optional[AlignmentResult] = None
    failure_reason: str = ""


def _resize_for_sift(arr: np.ndarray, max_side: int = 512):
    """Return (small_arr, scale) where scale = small/full."""
    h, w = arr.shape[:2]
    long_side = max(h, w)
    if long_side <= max_side:
        return arr, 1.0  # no change
    scale = max_side / long_side
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    small = cv2.resize(arr, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return small, scale

def align_images_sift(
    input_arr: np.ndarray,
    output_arr: np.ndarray,
    min_inlier_ratio: float = 0.3,
    max_side: int = 512,
) -> AlignmentResult:
    """
    Align output image to input using SIFT features.
    
    Args:
        input_arr: Reference image (RGB)
        output_arr: Image to align (RGB)
        min_inlier_ratio: Minimum inlier ratio for valid alignment
        
    Returns:
        AlignmentResult with aligned image and metrics
    """
    # Convert to grayscale
    input_small, s_in = _resize_for_sift(input_arr, max_side)
    output_small, s_out = _resize_for_sift(output_arr, max_side)

    input_gray = cv2.cvtColor(input_small, cv2.COLOR_RGB2GRAY)
    output_gray = cv2.cvtColor(output_small, cv2.COLOR_RGB2GRAY)
    
    # Detect SIFT features
    #Changing from 2000 to 800 features
    detector = cv2.SIFT_create(nfeatures=800)
    kp1, des1 = detector.detectAndCompute(input_gray, None)
    kp2, des2 = detector.detectAndCompute(output_gray, None)
    
    if des1 is None or des2 is None or len(kp1) < 4 or len(kp2) < 4:
        return AlignmentResult(None, False, 0, 0.0, "insufficient_keypoints")
    
    # Match features
    matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    matches = matcher.knnMatch(des2, des1, k=2)
    
    # Lowe's ratio test
    good_matches = []
    for match_pair in matches:
        if len(match_pair) == 2:
            m, n = match_pair
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)
    
    if len(good_matches) < 10:
        return AlignmentResult(None, False, len(good_matches), 0.0, "too_few_matches")
    
    # Compute homography
    src_pts = np.float32([kp2[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp1[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    
    H_small, mask_inliers = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    
    if H_small is None:
        return AlignmentResult(None, False, len(good_matches), 0.0, "homography_failed")
    
    # Check alignment quality
    inlier_ratio = np.sum(mask_inliers) / len(mask_inliers) if len(mask_inliers) > 0 else 0.0
    
    if inlier_ratio < min_inlier_ratio:
        return AlignmentResult(
            None, False, len(good_matches), inlier_ratio,
            f"low_inlier_ratio_{inlier_ratio:.2f}"
        )

    # lift homography back to FULL resolution
    # S_in maps full -> small input
    # S_out maps full -> small output
    S_in = np.array([[s_in, 0,    0],
                     [0,    s_in, 0],
                     [0,    0,    1]], dtype=np.float32)
    S_out = np.array([[s_out, 0,     0],
                      [0,     s_out, 0],
                      [0,     0,     1]], dtype=np.float32)

    # H_full = S_in^{-1} * H_small * S_out
    S_in_inv = np.array([[1.0/s_in, 0,         0],
                         [0,        1.0/s_in,  0],
                         [0,        0,         1]], dtype=np.float32)
    H_full = S_in_inv @ H_small @ S_out


    # Warp image
    h_full, w_full = input_arr.shape[:2]
    aligned_full = cv2.warpPerspective(output_arr, H_full, (w_full, h_full), flags=cv2.INTER_LINEAR)

    return AlignmentResult(
        aligned_full,
        True,
        len(good_matches),
        inlier_ratio,
        ""
    )


def detect_edge_artifacts(image: np.ndarray, threshold: int = 15, min_border: int = 5) -> dict:
    """
    Detect edge artifacts from alignment warping.
    
    Scans from each edge inward to find where artifacts end.
    Artifacts are typically black/dark borders from warping or distorted regions.
    
    Args:
        image: RGB image array
        threshold: Pixel intensity threshold for detecting dark edges
        min_border: Minimum border width to exclude even if no artifacts detected
        
    Returns:
        Dict with 'top', 'bottom', 'left', 'right' border widths
    """
    h, w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Detect top border - scan multiple rows and find where image stabilizes
    top = min_border
    for y in range(min(100, h // 3)):
        if y < 2:
            continue
        # Check current row and a few rows ahead
        current_mean = np.mean(gray[y, :])
        ahead_mean = np.mean(gray[min(y+5, h-1), :])
        
        # If both are bright and similar, we've found good region
        if current_mean > threshold and ahead_mean > threshold and abs(current_mean - ahead_mean) < 10:
            top = max(min_border, y)
            break
    
    # Detect bottom border
    bottom = min_border
    for y in range(h - 1, max(h - 100, 2 * h // 3), -1):
        if y > h - 3:
            continue
        current_mean = np.mean(gray[y, :])
        ahead_mean = np.mean(gray[max(y-5, 0), :])
        
        if current_mean > threshold and ahead_mean > threshold and abs(current_mean - ahead_mean) < 10:
            bottom = max(min_border, h - y - 1)
            break
    
    # Detect left border
    left = min_border
    for x in range(min(100, w // 3)):
        if x < 2:
            continue
        current_mean = np.mean(gray[:, x])
        ahead_mean = np.mean(gray[:, min(x+5, w-1)])
        
        if current_mean > threshold and ahead_mean > threshold and abs(current_mean - ahead_mean) < 10:
            left = max(min_border, x)
            break
    
    # Detect right border
    right = min_border
    for x in range(w - 1, max(w - 100, 2 * w // 3), -1):
        if x > w - 3:
            continue
        current_mean = np.mean(gray[:, x])
        ahead_mean = np.mean(gray[:, max(x-5, 0)])
        
        if current_mean > threshold and ahead_mean > threshold and abs(current_mean - ahead_mean) < 10:
            right = max(min_border, w - x - 1)
            break
    
    return {'top': top, 'bottom': bottom, 'left': left, 'right': right}


def compute_mask(
    input_img: Image.Image,
    output_img: Image.Image,
    min_inlier_ratio: float = 0.3,
) -> MaskResult:
    """
    Compute edit mask from input and output images.
    
    Pipeline:
    1. Align output to input using SIFT
    2. Compute pixel differences
    3. Apply Gaussian blur
    4. Threshold to binary mask
    5. Morphological operations
    6. Keep top-5 largest components
    7. Remove detected edge artifacts
    
    Args:
        input_img: Original input image
        output_img: Edited output image
        min_inlier_ratio: Alignment quality threshold
        
    Returns:
        MaskResult with mask and alignment info
    """
    # Convert to RGB
    input_img = input_img.convert("RGB")
    output_img = output_img.convert("RGB")
    
    # Resize if needed
    if input_img.size != output_img.size:
        output_img = output_img.resize(input_img.size, resample=Image.BILINEAR)
    
    # Convert to arrays
    in_arr = np.asarray(input_img, dtype=np.uint8)
    out_arr = np.asarray(output_img, dtype=np.uint8)
    
    # Align images
    alignment_result = align_images_sift(in_arr, out_arr, min_inlier_ratio)
    
    if not alignment_result.success:
        return MaskResult(
            None, False, alignment_result,
            f"alignment_failed_{alignment_result.failure_reason}"
        )
    
    out_arr = alignment_result.aligned_image
    
    # Compute difference
    diff = np.abs(out_arr.astype(np.int16) - in_arr.astype(np.int16))
    diff_mag = diff.max(axis=2)
    
    # Gaussian blur
    diff_mag = cv2.GaussianBlur(diff_mag.astype(np.float32), (BLUR_KERNEL_SIZE, BLUR_KERNEL_SIZE), 0)
    
    # Threshold
    mask = (diff_mag > MASK_THRESHOLD).astype(np.uint8) * 255
    
    # Morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (MORPHOLOGICAL_KERNEL_SIZE, MORPHOLOGICAL_KERNEL_SIZE))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    # Detect and remove edge artifacts FIRST
    borders = detect_edge_artifacts(out_arr)
    h, w = mask.shape
    
    if borders['top'] > 0:
        mask[:borders['top'], :] = 0
    if borders['bottom'] > 0:
        mask[-borders['bottom']:, :] = 0
    if borders['left'] > 0:
        mask[:, :borders['left']] = 0
    if borders['right'] > 0:
        mask[:, -borders['right']:] = 0
    
    # Keep top-k largest connected components AFTER edge removal
    # This ensures we don't count edge artifacts as components
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    
    if num_labels > 1:
        # Get component sizes (skip background at 0)
        areas = [(i, stats[i, cv2.CC_STAT_AREA]) for i in range(1, num_labels)]
        areas.sort(key=lambda x: x[1], reverse=True)
        
        # Keep top-k largest connected components
        keep_labels = set([idx for idx, _ in areas[:TOP_K_COMPONENTS]])
        
        cleaned_mask = np.zeros_like(mask)
        for label_idx in keep_labels:
            cleaned_mask[labels == label_idx] = 255
        
        mask = cleaned_mask
    
    mask_img = Image.fromarray(mask, mode="L")
    return MaskResult(mask_img, True, alignment_result, "")

