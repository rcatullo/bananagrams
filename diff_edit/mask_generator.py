"""
Mask generation utility using the trained mask prediction model.

This module loads the mask model checkpoint and generates masks from
input images and text instructions.
"""

import os
import sys
import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms
import clip

# Add parent directory to path to import model modules
# The model directory should be importable as 'model' when parent is in path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Add model directory to path so we can import directly like train.py does
model_dir = os.path.join(parent_dir, 'model')
if model_dir not in sys.path:
    sys.path.insert(0, model_dir)

# Import from model directory (same as train.py)
from model import build_model
from utils import load_config, get_device


class MaskGenerator:
    """
    Wrapper for generating masks using the trained mask prediction model.
    """
    
    def __init__(self, checkpoint_path: str, config_path: str = None, device: str = None):
        """
        Initialize the mask generator.
        
        Args:
            checkpoint_path: Path to the model checkpoint (e.g., 'model/checkpoints/best_model.pth')
            config_path: Path to model config file (defaults to 'model/config.yaml')
            device: Device to use ('cuda', 'cpu', or None for auto-detect)
        """
        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # Load config
        if config_path is None:
            config_path = os.path.join(os.path.dirname(__file__), '..', 'model', 'config.yaml')
        
        self.config = load_config(config_path)
        
        # Build model
        self.model = build_model(self.config)
        self.model.to(self.device)
        self.model.eval()
        
        # Load checkpoint
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        print(f"Loaded mask model from {checkpoint_path}")
        print(f"Using device: {self.device}")
        
        # Setup transforms
        self.image_size = self.config['data']['image_size']
        self.normalize_mean = self.config['data']['normalize_mean']
        self.normalize_std = self.config['data']['normalize_std']
        
        self.transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.normalize_mean, std=self.normalize_std)
        ])
        
        # CLIP tokenizer
        self.clip_tokenize = clip.tokenize
    
    def generate_mask(self, image: Image.Image, text: str, threshold: float = 0.5) -> Image.Image:
        """
        Generate a mask from an image and text instruction.
        
        Args:
            image: PIL Image (RGB)
            text: Text instruction
            threshold: Threshold for binarizing mask (default: 0.5)
        
        Returns:
            PIL Image mask (L mode, 0-255)
        """
        # Preprocess image
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Tokenize text
        try:
            text_tokens = self.clip_tokenize(text, truncate=True).to(self.device)
        except Exception as e:
            print(f"Warning: Failed to tokenize text: {text[:50]}... Error: {e}")
            text_tokens = self.clip_tokenize("", truncate=True).to(self.device)
        
        # Generate mask
        with torch.no_grad():
            mask_logits = self.model(image_tensor, text_tokens)
            mask_probs = torch.sigmoid(mask_logits)
        
        # Convert to PIL Image
        mask_np = mask_probs[0, 0].cpu().numpy()
        mask_binary = (mask_np > threshold).astype(float)
        
        # Resize to original image size if needed
        if image.size != (self.image_size, self.image_size):
            mask_pil = Image.fromarray((mask_binary * 255).astype('uint8'), mode='L')
            mask_pil = mask_pil.resize(image.size, Image.Resampling.BILINEAR)
        else:
            mask_pil = Image.fromarray((mask_binary * 255).astype('uint8'), mode='L')
        
        return mask_pil
    
    def generate_mask_batch(self, images, texts, threshold: float = 0.5):
        """
        Generate masks for a batch of images and texts.
        
        Args:
            images: List of PIL Images or tensor of shape (B, 3, H, W)
            texts: List of text instructions
            threshold: Threshold for binarizing mask
        
        Returns:
            List of PIL Image masks
        """
        if isinstance(images, list):
            # Convert list of PIL Images to tensor
            image_tensors = torch.stack([self.transform(img) for img in images]).to(self.device)
        else:
            image_tensors = images.to(self.device)
        
        # Tokenize texts
        text_tokens = torch.stack([
            self.clip_tokenize(text, truncate=True) for text in texts
        ]).to(self.device)
        
        # Generate masks
        with torch.no_grad():
            mask_logits = self.model(image_tensors, text_tokens)
            mask_probs = torch.sigmoid(mask_logits)
        
        # Convert to PIL Images
        masks = []
        for i in range(mask_probs.shape[0]):
            mask_np = mask_probs[i, 0].cpu().numpy()
            mask_binary = (mask_np > threshold).astype(float)
            mask_pil = Image.fromarray((mask_binary * 255).astype('uint8'), mode='L')
            masks.append(mask_pil)
        
        return masks
