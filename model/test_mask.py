"""
Test script to visualize example inputs and outputs from the dataset.

Loads a small number of samples and displays:
- Input image
- Text prompt
- Binary mask
"""

import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF

from dataset import HuggingFaceMaskDataset, MaskDatasetTransform
from utils import load_config


def denormalize_image(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """Denormalize image tensor for display."""
    mean = torch.tensor(mean).view(3, 1, 1)
    std = torch.tensor(std).view(3, 1, 1)
    return tensor * std + mean


def tensor_to_image(tensor):
    """Convert tensor [C, H, W] to PIL Image."""
    if tensor.dim() == 4:
        tensor = tensor[0]
    if tensor.dim() == 3:
        # Denormalize if needed (check if values are in [0, 1] range)
        if tensor.max() <= 1.0:
            tensor = tensor.clamp(0, 1)
        img_array = (tensor.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        return Image.fromarray(img_array)
    elif tensor.dim() == 2:
        # Grayscale mask
        img_array = (tensor.cpu().numpy() * 255).astype(np.uint8)
        return Image.fromarray(img_array, mode='L')
    else:
        raise ValueError(f"Unexpected tensor shape: {tensor.shape}")


def load_samples_from_dataset(config, num_samples=5):
    """
    Load a small number of samples from the training dataset.
    
    Args:
        config: Configuration dictionary
        num_samples: Number of samples to load
    
    Returns:
        List of sample dictionaries
    """
    data_config = config['data']
    
    # Create transform (no augmentation for testing)
    transform = MaskDatasetTransform(
        image_size=data_config['image_size'],
        normalize_mean=data_config['normalize_mean'],
        normalize_std=data_config['normalize_std'],
        augmentation_config=None,
        is_training=False
    )
    
    # Create dataset
    dataset_name = data_config.get('dataset_name', 'BryanW/HumanEdit')
    train_split = data_config.get('train_split', 'train')
    
    # Load only a small subset efficiently
    print(f"Loading first {num_samples} samples from {dataset_name} split '{train_split}'...")
    
    # Load dataset and use select() to only get the samples we need
    from datasets import load_dataset
    full_dataset = load_dataset(dataset_name, split=train_split)
    
    # Use select() to only load the indices we need (more efficient)
    num_available = len(full_dataset)
    num_to_load = min(num_samples, num_available)
    indices = list(range(num_to_load))
    dataset_subset = full_dataset.select(indices)
    
    # Get only the samples we need
    samples = []
    for i in range(num_to_load):
        # Get raw sample
        raw_sample = dataset_subset[i]
        
        # Extract raw image and mask for display
        input_img = raw_sample['INPUT_IMG']
        mask_img = raw_sample['MASK_IMG']
        text = raw_sample.get('EDITING_INSTRUCTION', '')
        
        # Convert to PIL if needed (same logic as in dataset)
        if not isinstance(input_img, Image.Image):
            if isinstance(input_img, dict) and 'bytes' in input_img:
                import io
                input_img = Image.open(io.BytesIO(input_img['bytes'])).convert('RGB')
            elif isinstance(input_img, np.ndarray):
                if input_img.dtype != np.uint8:
                    input_img = (input_img * 255).astype(np.uint8) if input_img.max() <= 1.0 else input_img.astype(np.uint8)
                input_img = Image.fromarray(input_img).convert('RGB')
            else:
                import io
                input_img = Image.open(io.BytesIO(input_img) if isinstance(input_img, bytes) else input_img).convert('RGB')
        
        if not isinstance(mask_img, Image.Image):
            if isinstance(mask_img, dict) and 'bytes' in mask_img:
                import io
                mask_img = Image.open(io.BytesIO(mask_img['bytes'])).convert('L')
            elif isinstance(mask_img, np.ndarray):
                if mask_img.dtype != np.uint8:
                    mask_img = (mask_img * 255).astype(np.uint8) if mask_img.max() <= 1.0 else mask_img.astype(np.uint8)
                mask_img = Image.fromarray(mask_img).convert('L')
            else:
                import io
                mask_img = Image.open(io.BytesIO(mask_img) if isinstance(mask_img, bytes) else mask_img).convert('L')
        
        # Ensure mask is binary
        mask_array = np.array(mask_img)
        
        # Handle multi-dimensional arrays (e.g., RGB, RGBA masks)
        if mask_array.ndim == 3:
            # If RGB or RGBA, convert to grayscale
            if mask_array.shape[2] == 3:
                # RGB: take mean across channels
                mask_array = mask_array.mean(axis=2).astype(np.float32)
            elif mask_array.shape[2] == 4:
                # RGBA: use alpha channel or convert to grayscale
                # Use alpha channel if it exists, otherwise convert RGB to grayscale
                mask_array = mask_array[:, :, 3].astype(np.float32)  # Use alpha channel
            else:
                mask_array = mask_array.squeeze()
        elif mask_array.ndim > 2:
            mask_array = mask_array.squeeze()
        
        # Ensure 2D array
        if mask_array.ndim != 2:
            raise ValueError(f"Mask array should be 2D, got shape {mask_array.shape}")
        
        # Normalize to 0-1 range if needed
        if mask_array.max() > 1:
            mask_array = mask_array.astype(np.float32) / 255.0
        mask_array = (mask_array > 0.5).astype(np.float32)
        mask_img = Image.fromarray((mask_array * 255).astype(np.uint8), mode='L')
        
        # Resize to match model input size for display
        image_size = data_config['image_size']
        input_img = input_img.resize((image_size, image_size), Image.BILINEAR)
        mask_img = mask_img.resize((image_size, image_size), Image.NEAREST)
        
        # Ensure text is a string
        if not isinstance(text, str):
            text = str(text) if text is not None else ""
        text = text.strip()
        
        samples.append({
            'image': input_img,
            'mask': mask_img,
            'text': text
        })
    
    return samples


def display_samples(samples):
    """
    Display samples with input image, text prompt, and mask.
    
    Args:
        samples: List of sample dictionaries
    """
    num_samples = len(samples)
    fig, axes = plt.subplots(num_samples, 3, figsize=(15, 5 * num_samples))
    
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    for i, sample in enumerate(samples):
        # Input image
        axes[i, 0].imshow(sample['image'])
        axes[i, 0].set_title(f'Sample {i+1}: Input Image')
        axes[i, 0].axis('off')
        
        # Text prompt
        text = sample['text']
        # Truncate long text
        if len(text) > 100:
            text = text[:97] + '...'
        axes[i, 1].text(0.1, 0.5, f'Text Prompt:\n\n{text}', 
                       fontsize=10, verticalalignment='center',
                       wrap=True, transform=axes[i, 1].transAxes)
        axes[i, 1].set_title(f'Sample {i+1}: Text Prompt')
        axes[i, 1].axis('off')
        
        # Mask
        mask = sample['mask']
        axes[i, 2].imshow(mask, cmap='gray', vmin=0, vmax=255)
        axes[i, 2].set_title(f'Sample {i+1}: Binary Mask')
        axes[i, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig('mask_test_examples.png', dpi=150, bbox_inches='tight')
    print(f"Saved visualization to mask_test_examples.png")
    plt.show()


def main():
    """Main function to test dataset samples."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Test dataset samples')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to config file')
    parser.add_argument('--num_samples', type=int, default=5,
                       help='Number of samples to display')
    args = parser.parse_args()
    
    # Load config
    print(f"Loading config from {args.config}...")
    config = load_config(args.config)
    
    # Load samples
    print(f"Loading {args.num_samples} samples from training dataset...")
    samples = load_samples_from_dataset(config, num_samples=args.num_samples)
    
    print(f"Loaded {len(samples)} samples")
    print("Displaying results...")
    display_samples(samples)


if __name__ == '__main__':
    main()
