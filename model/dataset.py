"""
WebDataset data loader for image editing mask prediction.

Loads tar shards containing:
- *.input.jpg: Input images
- *.mask.png: Binary masks
- *.text.txt: Edit instructions
"""

import os
import io
import re
import math
from typing import Dict, Optional, Callable

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from PIL import Image
import webdataset as wds
import clip


def decode_sample(sample: Dict) -> Dict:
    """
    Decode a single sample from WebDataset.
    
    Args:
        sample: Dictionary with keys ending in .input.jpg, .mask.png, .text.txt
    
    Returns:
        Dictionary with 'image', 'mask', 'text' keys
    """
    # Find keys (they have prefixes like "images_positive-edit_1.png")
    input_key = None
    mask_key = None
    text_key = None
    
    for key in sample.keys():
        if key.endswith('.input.jpg'):
            input_key = key
        elif key.endswith('.mask.png'):
            mask_key = key
        elif key.endswith('.text.txt'):
            text_key = key
    
    if not all([input_key, mask_key, text_key]):
        raise ValueError(f"Missing required keys in sample. Found keys: {sample.keys()}")
    
    # Decode image
    image = Image.open(io.BytesIO(sample[input_key])).convert('RGB')
    
    # Decode mask
    mask = Image.open(io.BytesIO(sample[mask_key])).convert('L')
    
    # Decode text (already decoded by WebDataset's .decode())
    text = sample[text_key]
    if isinstance(text, bytes):
        text = text.decode('utf-8')
    text = text.strip()
    
    return {
        'image': image,
        'mask': mask,
        'text': text
    }


class MaskDatasetTransform:
    """Transforms for training and validation."""
    
    def __init__(
        self,
        image_size: int = 512,
        normalize_mean: list = [0.485, 0.456, 0.406],
        normalize_std: list = [0.229, 0.224, 0.225],
        augmentation_config: Optional[Dict] = None,
        is_training: bool = True
    ):
        """
        Args:
            image_size: Target image size
            normalize_mean: Mean for normalization
            normalize_std: Std for normalization
            augmentation_config: Data augmentation configuration
            is_training: Whether this is for training (enables augmentation)
        """
        self.image_size = image_size
        self.is_training = is_training
        self.augmentation_config = augmentation_config or {}
        
        # Base transforms
        self.resize = transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.BILINEAR)
        self.normalize = transforms.Normalize(mean=normalize_mean, std=normalize_std)
        self.to_tensor = transforms.ToTensor()
        
        # Augmentation transforms (only for training)
        self.augmentations = []
        if is_training and self.augmentation_config.get('enabled', True):
            # Horizontal flip
            if self.augmentation_config.get('random_horizontal_flip', 0.0) > 0:
                self.hflip_prob = self.augmentation_config['random_horizontal_flip']
            else:
                self.hflip_prob = 0.0
            
            # Vertical flip
            if self.augmentation_config.get('random_vertical_flip', 0.0) > 0:
                self.vflip_prob = self.augmentation_config['random_vertical_flip']
            else:
                self.vflip_prob = 0.0
            
            # Color jitter
            color_jitter_config = self.augmentation_config.get('color_jitter', {})
            if color_jitter_config:
                self.color_jitter = transforms.ColorJitter(
                    brightness=color_jitter_config.get('brightness', 0.0),
                    contrast=color_jitter_config.get('contrast', 0.0),
                    saturation=color_jitter_config.get('saturation', 0.0),
                    hue=color_jitter_config.get('hue', 0.0)
                )
            else:
                self.color_jitter = None
        else:
            self.hflip_prob = 0.0
            self.vflip_prob = 0.0
            self.color_jitter = None
        
        # CLIP tokenizer
        self.clip_tokenize = clip.tokenize
    
    def __call__(self, sample: Dict) -> Dict:
        """
        Apply transforms to sample.
        
        Args:
            sample: Dict with 'image', 'mask', 'text'
        
        Returns:
            Dict with transformed tensors
        """
        image = sample['image']
        mask = sample['mask']
        text = sample['text']
        
        # Resize
        image = self.resize(image)
        mask = self.resize(mask)
        
        # Apply augmentations (same for image and mask)
        if self.is_training:
            # Horizontal flip
            if torch.rand(1).item() < self.hflip_prob:
                image = transforms.functional.hflip(image)
                mask = transforms.functional.hflip(mask)
            
            # Vertical flip
            if torch.rand(1).item() < self.vflip_prob:
                image = transforms.functional.vflip(image)
                mask = transforms.functional.vflip(mask)
            
            # Color jitter (only for image)
            if self.color_jitter is not None:
                image = self.color_jitter(image)
        
        # Convert to tensor
        image_tensor = self.to_tensor(image)
        mask_tensor = self.to_tensor(mask)
        
        # Normalize image
        image_tensor = self.normalize(image_tensor)
        
        # Tokenize text (truncate to 77 tokens for CLIP)
        try:
            text_tokens = self.clip_tokenize(text, truncate=True)[0]
        except Exception as e:
            print(f"Warning: Failed to tokenize text: {text[:50]}... Error: {e}")
            text_tokens = self.clip_tokenize("", truncate=True)[0]
        
        return {
            'image': image_tensor,
            'mask': mask_tensor,
            'text': text,
            'text_tokens': text_tokens
        }


def create_webdataset(
    data_root: str,
    shard_pattern: str,
    transform: Callable,
    shuffle_buffer: int = 1000,
    is_training: bool = True,
    num_workers: int = 1
):
    """
    Create WebDataset from tar shards (supports local paths and S3).
    Implements virtual padding to ensure shard count is divisible by num_workers.
    
    Args:
        data_root: Root directory/S3 bucket containing shards
        shard_pattern: Pattern for shard files (e.g., "train/shard-{000000..000009}.tar")
        transform: Transform function to apply
        shuffle_buffer: Size of shuffle buffer
        is_training: Whether this is training data
        num_workers: Number of data loader workers (for padding calculation)
    
    Returns:
        WebDataset instance
    """
    # Parse brace expansion pattern to build explicit shard list
    # Pattern format: "train/shard-{000000..000009}.tar"
    brace_match = re.search(r'\{(\d+)\.\.(\d+)\}', shard_pattern)
    
    if brace_match:
        # Extract start and end indices
        start_idx = int(brace_match.group(1))
        end_idx = int(brace_match.group(2))
        num_shards = end_idx - start_idx + 1
        
        # Get prefix and suffix
        prefix = shard_pattern[:brace_match.start()]
        suffix = shard_pattern[brace_match.end():]
        
        # Build list of shard paths
        shard_list = []
        for i in range(start_idx, end_idx + 1):
            # Preserve zero-padding in format (e.g., 000000)
            num_digits = len(brace_match.group(1))
            shard_name = f"{prefix}{i:0{num_digits}d}{suffix}"
            shard_list.append(shard_name)
        
        # Apply virtual padding if needed
        if num_workers > 1 and num_shards % num_workers != 0:
            # Calculate target count (round up to next multiple of num_workers)
            target_count = math.ceil(num_shards / num_workers) * num_workers
            padding_needed = target_count - num_shards
            
            # Cycle first shards to pad
            for i in range(padding_needed):
                shard_list.append(shard_list[i % num_shards])
        
        # Build full paths
        if data_root.startswith('s3://'):
            # S3 paths - use pipe notation for each shard
            full_paths = [f"pipe:aws s3 cp {data_root}/{shard} -" for shard in shard_list]
        else:
            # Local paths
            full_paths = [os.path.join(data_root, shard) for shard in shard_list]
        
        # Create dataset with explicit shard list
        dataset = (
            wds.WebDataset(full_paths, shardshuffle=1000 if is_training else False)
            .decode()
            .map(decode_sample)
            .map(transform)
        )
    else:
        # No brace expansion - single shard or already explicit list
        if data_root.startswith('s3://'):
            shard_path = f"pipe:aws s3 cp {data_root}/{shard_pattern} -"
        else:
            shard_path = os.path.join(data_root, shard_pattern)
        
        dataset = (
            wds.WebDataset(shard_path, shardshuffle=1000 if is_training else False)
            .decode()
            .map(decode_sample)
            .map(transform)
        )
    
    # Shuffle for training
    if is_training and shuffle_buffer > 0:
        dataset = dataset.shuffle(shuffle_buffer)
    
    return dataset


def collate_fn(batch):
    """
    Custom collate function for batching.
    
    Args:
        batch: List of samples
    
    Returns:
        Batched dictionary
    """
    images = torch.stack([item['image'] for item in batch])
    masks = torch.stack([item['mask'] for item in batch])
    text_tokens = torch.stack([item['text_tokens'] for item in batch])
    texts = [item['text'] for item in batch]
    
    return {
        'image': images,
        'mask': masks,
        'text_tokens': text_tokens,
        'text': texts
    }


def create_dataloaders(config: Dict):
    """
    Create train and validation dataloaders.
    
    Args:
        config: Configuration dictionary
    
    Returns:
        Tuple of (train_loader, val_loader)
    """
    data_config = config['data']
    
    # Create transforms
    train_transform = MaskDatasetTransform(
        image_size=data_config['image_size'],
        normalize_mean=data_config['normalize_mean'],
        normalize_std=data_config['normalize_std'],
        augmentation_config=data_config['augmentation'],
        is_training=True
    )
    
    val_transform = MaskDatasetTransform(
        image_size=data_config['image_size'],
        normalize_mean=data_config['normalize_mean'],
        normalize_std=data_config['normalize_std'],
        augmentation_config=None,
        is_training=False
    )
    
    # Create datasets
    train_dataset = create_webdataset(
        data_root=data_config['data_root'],
        shard_pattern=data_config['train_shards'],
        transform=train_transform,
        shuffle_buffer=1000,
        is_training=True,
        num_workers=data_config['num_workers']
    )
    
    val_dataset = create_webdataset(
        data_root=data_config['data_root'],
        shard_pattern=data_config['val_shards'],
        transform=val_transform,
        shuffle_buffer=0,
        is_training=False,
        num_workers=data_config['num_workers']
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        num_workers=data_config['num_workers'],
        pin_memory=data_config['pin_memory'],
        collate_fn=collate_fn,
        prefetch_factor=data_config['prefetch_factor'] if data_config['num_workers'] > 0 else None,
        persistent_workers=True if data_config['num_workers'] > 0 else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['evaluation']['batch_size'],
        num_workers=data_config['num_workers'],
        pin_memory=data_config['pin_memory'],
        collate_fn=collate_fn,
        prefetch_factor=data_config['prefetch_factor'] if data_config['num_workers'] > 0 else None,
        persistent_workers=True if data_config['num_workers'] > 0 else False
    )
    
    return train_loader, val_loader


def create_test_dataloader(config: Dict):
    """
    Create test dataloader.
    
    Args:
        config: Configuration dictionary
    
    Returns:
        Test dataloader
    """
    data_config = config['data']
    
    # Create transform (no augmentation)
    test_transform = MaskDatasetTransform(
        image_size=data_config['image_size'],
        normalize_mean=data_config['normalize_mean'],
        normalize_std=data_config['normalize_std'],
        augmentation_config=None,
        is_training=False
    )
    
    # Create dataset
    test_dataset = create_webdataset(
        data_root=data_config['data_root'],
        shard_pattern=data_config['test_shards'],
        transform=test_transform,
        shuffle_buffer=0,
        is_training=False,
        num_workers=data_config['num_workers']
    )
    
    # Create dataloader
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['evaluation']['batch_size'],
        num_workers=data_config['num_workers'],
        pin_memory=data_config['pin_memory'],
        collate_fn=collate_fn,
        prefetch_factor=data_config['prefetch_factor'] if data_config['num_workers'] > 0 else None,
        persistent_workers=True if data_config['num_workers'] > 0 else False
    )
    
    return test_loader

