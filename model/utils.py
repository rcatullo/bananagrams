"""
Utility functions for training, logging, and visualization.
"""

import os
import yaml
import json
import random
import logging
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import torchvision.transforms.functional as TF


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def save_config(config: Dict[str, Any], save_path: str) -> None:
    """Save configuration to YAML file."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)


def setup_logging(log_dir: str, experiment_name: str) -> logging.Logger:
    """Setup logging configuration."""
    os.makedirs(log_dir, exist_ok=True)
    
    log_file = os.path.join(log_dir, f"{experiment_name}.log")
    
    logger = logging.getLogger(experiment_name)
    logger.setLevel(logging.INFO)
    
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger


def set_seed(seed: int, deterministic: bool = False) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device(config: Dict[str, Any]) -> torch.device:
    """Get compute device based on configuration."""
    device_str = config['system']['device']
    
    if device_str == 'cuda':
        if torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            print("CUDA not available, falling back to CPU")
            device = torch.device('cpu')
    elif device_str.startswith('cuda:'):
        device = torch.device(device_str)
    else:
        device = torch.device('cpu')
    
    return device


def count_parameters(model: torch.nn.Module) -> Dict[str, int]:
    """Count model parameters."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'total': total_params,
        'trainable': trainable_params,
        'frozen': total_params - trainable_params
    }


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[Any],
    epoch: int,
    metrics: Dict[str, float],
    config: Dict[str, Any],
    save_path: str
) -> None:
    """Save model checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics,
        'config': config
    }
    
    if scheduler is not None:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(checkpoint, save_path)


def load_checkpoint(
    checkpoint_path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    device: Optional[torch.device] = None
) -> Dict[str, Any]:
    """Load model checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    return checkpoint


def manage_checkpoints(checkpoint_dir: str, keep_last_n: int = 3) -> None:
    """Keep only the last N checkpoints to save space."""
    checkpoint_files = sorted(
        Path(checkpoint_dir).glob('checkpoint_epoch_*.pth'),
        key=lambda x: int(x.stem.split('_')[-1])
    )
    
    # Remove old checkpoints, but keep best model
    if len(checkpoint_files) > keep_last_n:
        for checkpoint in checkpoint_files[:-keep_last_n]:
            if 'best' not in checkpoint.name:
                checkpoint.unlink()


def denormalize_image(
    image: torch.Tensor,
    mean: list = [0.485, 0.456, 0.406],
    std: list = [0.229, 0.224, 0.225]
) -> torch.Tensor:
    """Denormalize image tensor for visualization."""
    mean = torch.tensor(mean).view(3, 1, 1).to(image.device)
    std = torch.tensor(std).view(3, 1, 1).to(image.device)
    return image * std + mean


def visualize_predictions(
    images: torch.Tensor,
    text_list: list,
    pred_masks: torch.Tensor,
    gt_masks: torch.Tensor,
    num_samples: int = 4,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Visualize predictions with input image, ground truth mask, and predicted mask.
    
    Args:
        images: (B, 3, H, W) input images (normalized)
        text_list: List of text instructions
        pred_masks: (B, 1, H, W) predicted mask logits
        gt_masks: (B, 1, H, W) ground truth masks
        num_samples: Number of samples to visualize
        save_path: Optional path to save figure
    
    Returns:
        Matplotlib figure
    """
    num_samples = min(num_samples, images.size(0))
    
    fig, axes = plt.subplots(num_samples, 4, figsize=(16, 4 * num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    images_denorm = denormalize_image(images[:num_samples])
    
    pred_binary = torch.sigmoid(pred_masks[:num_samples]) > 0.5
    
    for idx in range(num_samples):
        img = images_denorm[idx].cpu().permute(1, 2, 0).numpy()
        img = np.clip(img, 0, 1)
        
        pred_mask = pred_binary[idx, 0].cpu().numpy()
        gt_mask = gt_masks[idx, 0].cpu().numpy()
        text = text_list[idx][:100]  # Truncate long text
        
        # Plot input image
        axes[idx, 0].imshow(img)
        axes[idx, 0].set_title(f"Input\n{text}", fontsize=8)
        axes[idx, 0].axis('off')
        
        # Plot ground truth mask
        axes[idx, 1].imshow(gt_mask, cmap='gray')
        axes[idx, 1].set_title("Ground Truth Mask")
        axes[idx, 1].axis('off')
        
        # Plot predicted mask
        axes[idx, 2].imshow(pred_mask, cmap='gray')
        axes[idx, 2].set_title("Predicted Mask")
        axes[idx, 2].axis('off')
        
        # Plot overlay (prediction on image)
        axes[idx, 3].imshow(img)
        axes[idx, 3].imshow(pred_mask, cmap='Reds', alpha=0.5)
        axes[idx, 3].set_title("Overlay")
        axes[idx, 3].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def save_metrics(metrics: Dict[str, Any], save_path: str) -> None:
    """Save metrics to JSON file."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w') as f:
        json.dump(metrics, f, indent=2)


def load_metrics(load_path: str) -> Dict[str, Any]:
    """Load metrics from JSON file."""
    with open(load_path, 'r') as f:
        metrics = json.load(f)
    return metrics


class AverageMeter:
    """Computes and stores the average and current value."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count > 0 else 0


class MetricsTracker:
    """Track multiple metrics over time."""
    
    def __init__(self):
        self.metrics = {}
    
    def update(self, metrics_dict: Dict[str, float]):
        """Update metrics with new values."""
        for key, value in metrics_dict.items():
            if key not in self.metrics:
                self.metrics[key] = AverageMeter()
            self.metrics[key].update(value)
    
    def get_averages(self) -> Dict[str, float]:
        """Get average values for all metrics."""
        return {key: meter.avg for key, meter in self.metrics.items()}
    
    def reset(self):
        """Reset all metrics."""
        for meter in self.metrics.values():
            meter.reset()


def format_time(seconds: float) -> str:
    """Format seconds into human-readable time string."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"


def print_config(config: Dict[str, Any], logger: Optional[logging.Logger] = None):
    """Pretty print configuration."""
    config_str = yaml.dump(config, default_flow_style=False, sort_keys=False)
    
    message = f"\n{'='*60}\nConfiguration:\n{'='*60}\n{config_str}{'='*60}"
    
    if logger:
        logger.info(message)
    else:
        print(message)

