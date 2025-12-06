"""
HuggingFace dataset loader for image editing mask prediction.

Loads from HuggingFace dataset "BryanW/HumanEdit" containing:
- INPUT_IMG: Input images
- MASK_IMG: Binary masks
- EDITING_INSTRUCTION: Edit instructions
"""

import os
import io
from typing import Dict, Optional, Callable

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from PIL import Image, ImageFilter
import numpy as np
from datasets import load_dataset
import clip


class HuggingFaceMaskDataset(Dataset):
    """
    Dataset wrapper for HuggingFace HumanEdit dataset.
    """
    
    def __init__(
        self,
        dataset_name: str = "BryanW/HumanEdit",
        split: str = "train",
        transform: Optional[Callable] = None
    ):
        """
        Args:
            dataset_name: HuggingFace dataset identifier
            split: Dataset split ('train', 'validation', 'test')
            transform: Transform function to apply
        """
        self.transform = transform
        
        # Load dataset from HuggingFace
        print(f"Loading dataset {dataset_name} split '{split}'...")
        try:
            self.dataset = load_dataset(dataset_name, split=split)
        except ValueError as e:
            # If split doesn't exist, create it from train split
            if split in ["validation", "val", "test"]:
                print(f"Split '{split}' not found. Creating '{split}' from 'train' split...")
                try:
                    # Load full train dataset (will be cached)
                    train_dataset = load_dataset(dataset_name, split="train")
                    total_size = len(train_dataset)
                    
                    # Deterministic split: 80% train, 10% val, 10% test
                    train_size = int(total_size * 0.8)
                    val_size = int(total_size * 0.1)
                    
                    if split in ["validation", "val"]:
                        # Validation: indices [train_size : train_size + val_size]
                        val_start = train_size
                        val_end = min(train_size + val_size, total_size)
                        self.dataset = train_dataset.select(range(val_start, val_end))
                        print(f"Created validation split with {len(self.dataset)} samples (10% of train)")
                    elif split == "test":
                        # Test: indices [train_size + val_size : ]
                        test_start = train_size + val_size
                        self.dataset = train_dataset.select(range(test_start, total_size))
                        print(f"Created test split with {len(self.dataset)} samples (10% of train)")
                except Exception as e2:
                    raise ValueError(f"Could not load dataset {dataset_name} or create split '{split}': {e2}")
            elif split == "train":
                # For train, if val/test don't exist, limit to 80% to leave room for them
                # Check if val/test splits exist by trying to load them
                has_val = False
                try:
                    _ = load_dataset(dataset_name, split="validation")
                    has_val = True
                except:
                    try:
                        _ = load_dataset(dataset_name, split="val")
                        has_val = True
                    except:
                        pass
                
                if not has_val:
                    # No validation split exists, limit train to 80% for future splits
                    self.dataset = load_dataset(dataset_name, split="train")
                    total_size = len(self.dataset)
                    train_size = int(total_size * 0.8)
                    self.dataset = self.dataset.select(range(0, train_size))
                    print(f"Limited train split to {len(self.dataset)} samples (80% of total) for train/val/test split")
                else:
                    # Validation split exists, use full train
                    self.dataset = load_dataset(dataset_name, split="train")
            else:
                raise e
        
        print(f"Loaded {len(self.dataset)} samples")
    
    def __len__(self):
        return len(self.dataset)
    
    def _normalize_array_to_uint8(self, arr: np.ndarray) -> np.ndarray:
        """Normalize array to uint8 format (0-255 range)."""
        if arr.dtype != np.uint8:
            if arr.max() <= 1.0:
                arr = (arr * 255).astype(np.uint8)
            else:
                arr = arr.astype(np.uint8)
        return arr
    
    def _convert_to_pil_image(self, img_data, target_mode: str = 'RGB') -> Image.Image:
        """
        Convert various image formats to PIL Image.
        
        Args:
            img_data: Image data (PIL Image, dict with bytes, numpy array, str, or bytes)
            target_mode: Target PIL mode ('RGB' or 'L')
        
        Returns:
            PIL Image in target mode
        """
        # Already a PIL Image
        if isinstance(img_data, Image.Image):
            return img_data.convert(target_mode) if img_data.mode != target_mode else img_data
        
        # Dict with bytes (HuggingFace format)
        if isinstance(img_data, dict) and 'bytes' in img_data:
            return Image.open(io.BytesIO(img_data['bytes'])).convert(target_mode)
        
        # Numpy array
        if isinstance(img_data, np.ndarray):
            arr = self._normalize_array_to_uint8(img_data)
            
            if target_mode == 'RGB':
                if arr.ndim == 2:
                    # Grayscale: convert to RGB
                    return Image.fromarray(arr, mode='L').convert('RGB')
                elif arr.ndim == 3:
                    return Image.fromarray(arr).convert('RGB')
                else:
                    raise ValueError(f"Unexpected array shape for RGB: {arr.shape}")
            else:  # target_mode == 'L'
                if arr.ndim == 2:
                    return Image.fromarray(arr, mode='L')
                elif arr.ndim == 3:
                    # Multi-channel: convert to grayscale
                    return Image.fromarray(arr).convert('L')
                else:
                    raise ValueError(f"Unexpected array shape for grayscale: {arr.shape}")
        
        # String path or bytes
        if isinstance(img_data, (str, bytes)):
            source = io.BytesIO(img_data) if isinstance(img_data, bytes) else img_data
            return Image.open(source).convert(target_mode)
        
        raise ValueError(f"Unexpected image type: {type(img_data)}")
    
    def _process_mask_to_binary(self, mask_img: Image.Image) -> Image.Image:
        """
        Process mask image to ensure it's binary (0 or 255).
        
        Args:
            mask_img: PIL Image mask in 'L' mode
        
        Returns:
            Binary PIL Image in 'L' mode
        """
        # Convert to numpy array (already 2D since input is 'L' mode)
        mask_array = np.array(mask_img, dtype=np.float32)
        
        # Normalize to 0-1 range and threshold to binary
        if mask_array.max() > 1:
            mask_array = mask_array / 255.0
        mask_array = (mask_array > 0.5).astype(np.float32)
        
        # Convert back to PIL Image
        return Image.fromarray((mask_array * 255).astype(np.uint8), mode='L')
    
    def __getitem__(self, idx):
        """
        Get a single sample from the dataset.
        
        Args:
            idx: Sample index
        
        Returns:
            Dictionary with 'image', 'mask', 'text' keys (or transformed version)
        """
        sample = self.dataset[idx]
        
        input_img = self._convert_to_pil_image(sample['INPUT_IMG'], target_mode='RGB')
        mask_img = self._convert_to_pil_image(sample['MASK_IMG'], target_mode='L')
        mask_img = self._process_mask_to_binary(mask_img)
        
        # Process text
        text = sample.get('EDITING_INSTRUCTION', '')
        if not isinstance(text, str):
            text = str(text) if text is not None else ""
        text = text.strip()
        
        # Create sample dictionary
        sample_dict = {
            'image': input_img,
            'mask': mask_img,
            'text': text
        }
        
        # Apply transform if provided
        if self.transform is not None:
            sample_dict = self.transform(sample_dict)
        
        return sample_dict


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
            
            # Rotation
            self.rotation_degrees = self.augmentation_config.get('random_rotation', 0)
            
            # Scale/zoom
            self.scale_range = self.augmentation_config.get('random_scale', None)
            
            # Gaussian blur
            self.blur_prob = self.augmentation_config.get('random_blur', 0.0)
            self.blur_sigma_range = self.augmentation_config.get('blur_sigma_range', [0.1, 2.0])
            
            # Random erasing
            self.erasing_prob = self.augmentation_config.get('random_erasing', 0.0)
            self.erasing_scale = self.augmentation_config.get('erasing_scale', (0.02, 0.33))
            self.erasing_ratio = self.augmentation_config.get('erasing_ratio', (0.3, 3.3))
            
        else:
            self.hflip_prob = 0.0
            self.vflip_prob = 0.0
            self.color_jitter = None
            self.rotation_degrees = 0
            self.scale_range = None
            self.blur_prob = 0.0
            self.erasing_prob = 0.0
        
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
        
        # Resize first (faster to do augmentations on smaller images)
        image = self.resize(image)
        mask = self.resize(mask)
        
        # Apply augmentations after resize (faster on smaller images)
        if self.is_training:
            # Random rotation (after resize for speed)
            if self.rotation_degrees > 0:
                angle = np.random.uniform(-self.rotation_degrees, self.rotation_degrees)
                image = TF.rotate(image, angle, interpolation=TF.InterpolationMode.BILINEAR, fill=0)
                mask = TF.rotate(mask, angle, interpolation=TF.InterpolationMode.NEAREST, fill=0)
            
            # Random scale/zoom (simplified - just disable for now to speed up)
            # Scale augmentation is expensive, disabled by default
            # if self.scale_range is not None:
            #     pass  # Disabled for performance
        
        # Apply augmentations after resize
        if self.is_training:
            # Horizontal flip
            if torch.rand(1).item() < self.hflip_prob:
                image = TF.hflip(image)
                mask = TF.hflip(mask)
            
            # Vertical flip
            if torch.rand(1).item() < self.vflip_prob:
                image = TF.vflip(image)
                mask = TF.vflip(mask)
            
            # Gaussian blur (only for image)
            if torch.rand(1).item() < self.blur_prob:
                sigma = np.random.uniform(self.blur_sigma_range[0], self.blur_sigma_range[1])
                image = image.filter(ImageFilter.GaussianBlur(radius=sigma))
            
            # Color jitter (only for image)
            if self.color_jitter is not None:
                image = self.color_jitter(image)
        
        # Convert to tensor
        image_tensor = self.to_tensor(image)
        mask_tensor = self.to_tensor(mask)
        
        # Random erasing (on tensor, only for image)
        if self.is_training and self.erasing_prob > 0 and torch.rand(1).item() < self.erasing_prob:
            # Apply random erasing
            if len(image_tensor.shape) == 3:
                C, H, W = image_tensor.shape
                area = H * W
                
                for _ in range(1):  # Try once
                    erase_area = np.random.uniform(self.erasing_scale[0], self.erasing_scale[1]) * area
                    aspect_ratio = np.random.uniform(self.erasing_ratio[0], self.erasing_ratio[1])
                    
                    h = int(round(np.sqrt(erase_area * aspect_ratio)))
                    w = int(round(np.sqrt(erase_area / aspect_ratio)))
                    
                    if h < H and w < W:
                        top = np.random.randint(0, H - h)
                        left = np.random.randint(0, W - w)
                        image_tensor[:, top:top+h, left:left+w] = torch.randn(C, h, w) * 0.5 + 0.5
                        break
        
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
    dataset_name = data_config.get('dataset_name', 'BryanW/HumanEdit')
    
    train_dataset = HuggingFaceMaskDataset(
        dataset_name=dataset_name,
        split=data_config.get('train_split', 'train'),
        transform=train_transform
    )
    
    val_dataset = HuggingFaceMaskDataset(
        dataset_name=dataset_name,
        split=data_config.get('val_split', 'validation'),
        transform=val_transform
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=data_config['num_workers'],
        pin_memory=data_config['pin_memory'],
        collate_fn=collate_fn,
        prefetch_factor=data_config['prefetch_factor'] if data_config['num_workers'] > 0 else None,
        persistent_workers=True if data_config['num_workers'] > 0 else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['evaluation']['batch_size'],
        shuffle=False,
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
    dataset_name = data_config.get('dataset_name', 'BryanW/HumanEdit')
    
    test_dataset = HuggingFaceMaskDataset(
        dataset_name=dataset_name,
        split=data_config.get('test_split', 'test'),
        transform=test_transform
    )
    
    # Create dataloader
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['evaluation']['batch_size'],
        shuffle=False,
        num_workers=data_config['num_workers'],
        pin_memory=data_config['pin_memory'],
        collate_fn=collate_fn,
        prefetch_factor=data_config['prefetch_factor'] if data_config['num_workers'] > 0 else None,
        persistent_workers=True if data_config['num_workers'] > 0 else False
    )
    
    return test_loader
