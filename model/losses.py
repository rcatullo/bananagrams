"""
Loss functions for mask prediction.

Implements Binary Cross-Entropy and Dice loss for segmentation tasks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BCELoss(nn.Module):
    """Binary Cross-Entropy loss for mask prediction."""
    
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
    
    def forward(self, pred_logits, target_mask):
        """
        Args:
            pred_logits: (B, 1, H, W) raw logits from model
            target_mask: (B, 1, H, W) binary ground truth mask [0, 1]
        
        Returns:
            BCE loss value
        """
        return self.bce(pred_logits, target_mask)


class DiceLoss(nn.Module):
    """
    Dice loss for segmentation.
    
    Dice coefficient: 2 * |X ∩ Y| / (|X| + |Y|)
    Dice loss: 1 - Dice coefficient
    """
    
    def __init__(self, smooth=1.0):
        """
        Args:
            smooth: Smoothing constant to avoid division by zero
        """
        super().__init__()
        self.smooth = smooth
    
    def forward(self, pred_logits, target_mask):
        """
        Args:
            pred_logits: (B, 1, H, W) raw logits from model
            target_mask: (B, 1, H, W) binary ground truth mask [0, 1]
        
        Returns:
            Dice loss value
        """
        pred_probs = torch.sigmoid(pred_logits)
        
        # Flatten spatial dimensions
        pred_flat = pred_probs.view(pred_probs.size(0), -1)
        target_flat = target_mask.view(target_mask.size(0), -1)
        
        # Compute intersection and union
        intersection = (pred_flat * target_flat).sum(dim=1)
        pred_sum = pred_flat.sum(dim=1)
        target_sum = target_flat.sum(dim=1)
        
        dice = (2.0 * intersection + self.smooth) / (pred_sum + target_sum + self.smooth)
        
        return 1.0 - dice.mean()


class CombinedLoss(nn.Module):
    """Combined BCE + Dice loss for mask prediction."""
    
    def __init__(self, bce_weight=1.0, dice_weight=1.0):
        """
        Args:
            bce_weight: Weight for BCE loss
            dice_weight: Weight for Dice loss
        """
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.bce_loss = BCELoss()
        self.dice_loss = DiceLoss()
    
    def forward(self, pred_logits, target_mask):
        """
        Args:
            pred_logits: (B, 1, H, W) raw logits from model
            target_mask: (B, 1, H, W) binary ground truth mask [0, 1]
        
        Returns:
            Dictionary with total loss and individual loss components
        """
        bce = self.bce_loss(pred_logits, target_mask)
        dice = self.dice_loss(pred_logits, target_mask)
        
        total_loss = self.bce_weight * bce + self.dice_weight * dice
        
        return {
            'loss': total_loss,
            'bce_loss': bce.item(),
            'dice_loss': dice.item()
        }


def compute_metrics(pred_logits, target_mask, threshold=0.5):
    """
    Compute segmentation metrics.
    
    Args:
        pred_logits: (B, 1, H, W) raw logits from model
        target_mask: (B, 1, H, W) binary ground truth mask [0, 1]
        threshold: Threshold for binary prediction
    
    Returns:
        Dictionary with IoU, F1, precision, recall, accuracy
    """
    with torch.no_grad():
        pred_probs = torch.sigmoid(pred_logits)
        pred_binary = (pred_probs > threshold).float()
        
        # Flatten
        pred_flat = pred_binary.view(pred_binary.size(0), -1)
        target_flat = target_mask.view(target_mask.size(0), -1)
        
        intersection = (pred_flat * target_flat).sum(dim=1)
        union = pred_flat.sum(dim=1) + target_flat.sum(dim=1) - intersection
        
        iou = (intersection + 1e-6) / (union + 1e-6)
        
        tp = intersection
        fp = pred_flat.sum(dim=1) - intersection
        fn = target_flat.sum(dim=1) - intersection
        tn = (1 - pred_flat).sum(dim=1) - fn
        
        precision = (tp + 1e-6) / (tp + fp + 1e-6)
        recall = (tp + 1e-6) / (tp + fn + 1e-6)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-6)
        
        accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-6)
        
        return {
            'iou': iou.mean().item(),
            'f1': f1.mean().item(),
            'precision': precision.mean().item(),
            'recall': recall.mean().item(),
            'accuracy': accuracy.mean().item()
        }

