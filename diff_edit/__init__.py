"""
Diffusion-based image editing with mask-based inpainting.

This module integrates the trained mask prediction model with Stable Diffusion
inpainting to generate high-quality image edits.
"""

from .mask_generator import MaskGenerator

__all__ = [
    'MaskGenerator',
]
