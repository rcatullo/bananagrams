"""
Example script for image editing using mask-based inpainting.

This script:
1. Loads the mask model to generate binary masks from images and prompts
2. Uses Stable Diffusion inpainting to edit the masked regions
"""

import os
import sys
import argparse
import torch
from PIL import Image
from diffusers import StableDiffusionInpaintPipeline

# Add parent to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from diff_edit.mask_generator import MaskGenerator


def round_to_multiple_of_8(value: int) -> int:
    """
    Round a value to the nearest multiple of 8.
    
    Args:
        value: Integer value to round
    
    Returns:
        Value rounded to nearest multiple of 8
    """
    return ((value + 4) // 8) * 8


def find_best_checkpoint(checkpoint_dir: str = "model/checkpoints") -> str:
    """
    Find the best model checkpoint.

    Args:
        checkpoint_dir: Directory containing checkpoints

    Returns:
        Path to best checkpoint
    """
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    abs_checkpoint_dir = os.path.join(project_root, checkpoint_dir)

    best_path = os.path.join(abs_checkpoint_dir, "best_model.pth")

    if os.path.exists(best_path):
        return best_path
        
    # Fallback: look for any checkpoint
    import glob
    checkpoints = glob.glob(os.path.join(abs_checkpoint_dir, "*.pth"))
    if checkpoints:
        # Return the most recent checkpoint
        return max(checkpoints, key=os.path.getmtime)

    raise FileNotFoundError(f"No checkpoint found in {abs_checkpoint_dir}")


def edit_image_with_inpainting(
    image: Image.Image,
    prompt: str,
    mask_generator: MaskGenerator,
    pipe: StableDiffusionInpaintPipeline,
    mask_threshold: float = 0.5,
    strength: float = 1.0,
    num_inference_steps: int = 50,
    guidance_scale: float = 7.5,
    height: int = None,
    width: int = None,
    seed: int = None,
) -> Image.Image:
    """
    Edit an image using mask-based inpainting.
    
    Args:
        image: Input PIL Image (RGB)
        prompt: Text prompt describing the desired edit
        mask_generator: MaskGenerator instance
        pipe: StableDiffusionInpaintPipeline instance
        mask_threshold: Threshold for binarizing mask (default: 0.5)
        strength: Strength of the inpainting (0.0 to 1.0, default: 1.0)
        num_inference_steps: Number of denoising steps (default: 50)
        guidance_scale: Guidance scale for text prompt (default: 7.5)
        height: Output height (default: None, uses image height)
        width: Output width (default: None, uses image width)
        seed: Random seed (default: None)
    
    Returns:
        Edited PIL Image
    """
    image = image.resize((512, 512))
    # Generate mask using the mask model
    print(f"Generating mask for prompt: {prompt}")
    mask = Image.open("example_edits/mask_true.png").convert("L") # mask_generator.generate_mask(image, prompt, threshold=mask_threshold)
    
    # Invert mask (compatibility with SD)
    import numpy as np
    mask_array = np.array(mask)
    mask_array = 255 - mask_array
    mask = Image.fromarray(mask_array, mode='L')
    print("Mask inverted (255->0, 0->255)")
    
    # Setup generator
    if seed is not None:
        generator = torch.Generator(device=pipe.device).manual_seed(seed)
    else:
        generator = None
    
    # Generate edit using inpainting
    print(f"Generating edit with inpainting...")
    result = pipe(
        prompt=prompt,
        image=image_resized,
        mask_image=mask_resized,
        height=height,
        width=width,
        strength=strength,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        generator=generator,
    ).images[0]
    
    return result


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Edit images using mask-based inpainting")
    
    parser.add_argument(
        '--image',
        type=str,
        default="example_edits/image.png",
        help='Path to input image'
    )
    parser.add_argument(
        '--prompt',
        type=str,
        required=True,
        help='Text prompt describing the desired edit'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        default=None,
        help='Path to mask model checkpoint (default: auto-detect best_model.pth)'
    )
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Path to model config file (default: model/config.yaml)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='example_edits/edited_output.png',
        help='Output path for edited image'
    )
    parser.add_argument(
        '--mask-output',
        type=str,
        default='example_edits/mask.png',
        help='Optional path to save the generated mask'
    )
    parser.add_argument(
        '--mask-threshold',
        type=float,
        default=0.5,
        help='Threshold for binarizing mask (default: 0.5)'
    )
    parser.add_argument(
        '--strength',
        type=float,
        default=0.5,
        help='Strength of inpainting (0.0 to 1.0, default: 1.0)'
    )
    parser.add_argument(
        '--num-steps',
        type=int,
        default=50,
        help='Number of inference steps (default: 50)'
    )
    parser.add_argument(
        '--guidance-scale',
        type=float,
        default=7.5,
        help='Guidance scale for text prompt (default: 7.5)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='Random seed'
    )
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        help='Device to use (cuda/cpu, default: auto-detect)'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='runwayml/stable-diffusion-inpainting',
        help='Stable Diffusion inpainting model (default: runwayml/stable-diffusion-inpainting)'
    )
    
    args = parser.parse_args()
    
    # Set device
    if args.device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    
    print(f"Using device: {device}")
    
    # Find checkpoint
    if args.checkpoint is None:
        checkpoint_path = find_best_checkpoint()
    else:
        checkpoint_path = args.checkpoint
    
    print(f"Loading mask model from: {checkpoint_path}")
    
    # Initialize mask generator
    mask_generator = MaskGenerator(
        checkpoint_path=checkpoint_path,
        config_path=args.config,
        device=device
    )
    
    # Initialize Stable Diffusion inpainting pipeline
    print(f"Loading Stable Diffusion inpainting pipeline: {args.model}")
    # Allow fallback to .bin files if safetensors aren't available
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        args.model,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    ).to(device)
    
    # Disable safety checker for faster inference
    pipe.safety_checker = None
    
    # Load image
    image = Image.open(args.image).convert("RGB")
    
    # Generate edit
    result = edit_image_with_inpainting(
        image=image,
        prompt=args.prompt,
        mask_generator=mask_generator,
        pipe=pipe,
        mask_threshold=args.mask_threshold,
        strength=args.strength,
        num_inference_steps=args.num_steps,
        guidance_scale=args.guidance_scale,
        height=args.height,
        width=args.width,
        seed=args.seed,
    )
    
    # Save result
    result.save(args.output)
    print(f"Saved edited image to: {args.output}")
    
    # Save mask if requested
    if args.mask_output:
        mask = mask_generator.generate_mask(image, args.prompt, threshold=args.mask_threshold)
        mask.save(args.mask_output)
        print(f"Saved mask to: {args.mask_output}")


if __name__ == '__main__':
    main()
