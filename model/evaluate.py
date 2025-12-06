"""
Evaluation script for image editing mask prediction model.

Evaluates a trained model on the test dataset and saves metrics and visualizations.

Usage:
    python evaluate.py --checkpoint checkpoints/best_model.pth --config config.yaml
    python evaluate.py --checkpoint checkpoints/best_model.pth --config config.yaml --output_dir test_results
"""

import argparse
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np

# Import model components
from model import build_model
from dataset import create_dataloaders, create_test_dataloader
from losses import CombinedLoss, compute_metrics
from utils import (
    load_config,
    get_device,
    load_checkpoint,
    visualize_predictions,
    MetricsTracker,
    save_metrics,
    format_time,
    setup_logging
)


@torch.no_grad()
def evaluate(
    model,
    test_loader,
    criterion,
    device,
    config,
    logger,
    output_dir,
    num_visualizations=20
):
    """
    Evaluate the model on test dataset.
    
    Args:
        model: Trained model
        test_loader: Test dataloader
        criterion: Loss function
        device: Compute device
        config: Configuration dictionary
        logger: Logger instance
        output_dir: Directory to save results
        num_visualizations: Number of samples to visualize
    
    Returns:
        Dictionary with average metrics
    """
    model.eval()
    metrics_tracker = MetricsTracker()
    
    # Create output directories
    vis_dir = os.path.join(output_dir, 'visualizations')
    masks_dir = os.path.join(output_dir, 'masks')
    os.makedirs(vis_dir, exist_ok=True)
    os.makedirs(masks_dir, exist_ok=True)
    
    threshold = config['evaluation']['threshold']
    
    logger.info(f"Starting evaluation on {len(test_loader.dataset)} test samples...")
    logger.info(f"Using threshold: {threshold}")
    
    all_predictions = []
    all_masks = []
    all_images = []
    all_texts = []
    
    pbar = tqdm(test_loader, desc="Evaluating")
    
    for batch_idx, batch in enumerate(pbar):
        # Move to device
        images = batch['image'].to(device)
        masks = batch['mask'].to(device)
        text_tokens = batch['text_tokens'].to(device)
        
        # Forward pass
        pred_logits = model(images, text_tokens)
        loss_dict = criterion(pred_logits, masks)
        
        # Compute metrics
        batch_metrics = compute_metrics(pred_logits, masks, threshold=threshold)
        
        # Update tracking
        metrics_tracker.update({
            'loss': loss_dict['loss'].item(),
            'bce_loss': loss_dict['bce_loss'],
            'dice_loss': loss_dict['dice_loss'],
            **batch_metrics
        })
        
        # Store samples for visualization
        batch_size = images.size(0)
        num_to_store = min(batch_size, num_visualizations - len(all_images))
        
        if len(all_images) < num_visualizations:
            all_images.append(images[:num_to_store].cpu())
            all_masks.append(masks[:num_to_store].cpu())
            all_predictions.append(pred_logits[:num_to_store].cpu())
            all_texts.extend(batch['text'][:num_to_store])
        
        # Update progress bar
        current_metrics = metrics_tracker.get_averages()
        pbar.set_postfix({
            'loss': f"{current_metrics['loss']:.4f}",
            'iou': f"{current_metrics['iou']:.4f}",
            'f1': f"{current_metrics['f1']:.4f}"
        })
    
    # Get average metrics
    avg_metrics = metrics_tracker.get_averages()
    
    # Log metrics
    log_msg = (
        f"\n{'='*60}\n"
        f"Test Evaluation Results\n"
        f"{'='*60}\n"
        f"Loss: {avg_metrics['loss']:.4f}\n"
        f"  - BCE Loss: {avg_metrics['bce_loss']:.4f}\n"
        f"  - Dice Loss: {avg_metrics['dice_loss']:.4f}\n"
        f"\nSegmentation Metrics:\n"
        f"  - IoU: {avg_metrics['iou']:.4f}\n"
        f"  - F1 Score: {avg_metrics['f1']:.4f}\n"
        f"  - Precision: {avg_metrics['precision']:.4f}\n"
        f"  - Recall: {avg_metrics['recall']:.4f}\n"
        f"  - Accuracy: {avg_metrics['accuracy']:.4f}\n"
        f"{'='*60}"
    )
    logger.info(log_msg)
    
    # Concatenate stored samples for visualization
    if all_images:
        all_images_tensor = torch.cat(all_images, dim=0)
        all_masks_tensor = torch.cat(all_masks, dim=0)
        all_predictions_tensor = torch.cat(all_predictions, dim=0)
        
        # Create visualizations
        logger.info(f"Creating visualizations for {len(all_texts)} samples...")
        
        # Visualize in batches
        batch_size = 4
        num_batches = (len(all_texts) + batch_size - 1) // batch_size
        
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(all_texts))
            
            batch_images = all_images_tensor[start_idx:end_idx]
            batch_masks = all_masks_tensor[start_idx:end_idx]
            batch_preds = all_predictions_tensor[start_idx:end_idx]
            batch_texts = all_texts[start_idx:end_idx]
            
            # Create visualization
            fig = visualize_predictions(
                batch_images,
                batch_texts,
                batch_preds,
                batch_masks,
                num_samples=end_idx - start_idx,
                save_path=os.path.join(vis_dir, f'sample_{start_idx:05d}.png')
            )
            
            # Save individual predicted masks
            pred_probs = torch.sigmoid(batch_preds)
            pred_binary = (pred_probs > threshold).float()
            
            for i in range(end_idx - start_idx):
                mask_idx = start_idx + i
                pred_mask = pred_binary[i, 0].cpu().numpy()
                
                # Save as PNG
                from PIL import Image
                mask_img = Image.fromarray((pred_mask * 255).astype(np.uint8), mode='L')
                mask_img.save(os.path.join(masks_dir, f'mask_{mask_idx:05d}.png'))
        
        logger.info(f"Saved visualizations to {vis_dir}")
        logger.info(f"Saved predicted masks to {masks_dir}")
    
    # Save metrics to JSON
    metrics_path = os.path.join(output_dir, 'test_metrics.json')
    save_metrics(avg_metrics, metrics_path)
    logger.info(f"Saved metrics to {metrics_path}")
    
    return avg_metrics


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Evaluate mask prediction model on test set")
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to model checkpoint file'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='test_results',
        help='Directory to save evaluation results'
    )
    parser.add_argument(
        '--num_visualizations',
        type=int,
        default=20,
        help='Number of samples to visualize'
    )
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Setup logging
    logger = setup_logging(
        args.output_dir,
        'test_eval'
    )
    
    logger.info(f"Loading checkpoint from {args.checkpoint}...")
    logger.info(f"Using config from {args.config}")
    logger.info(f"Output directory: {args.output_dir}")
    
    # Get device
    device = get_device(config)
    logger.info(f"Using device: {device}")
    
    # Build model
    logger.info("Building model...")
    model = build_model(config)
    model = model.to(device)
    
    # Load checkpoint
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    
    logger.info(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = load_checkpoint(args.checkpoint, model, device=device)
    
    # Log checkpoint info
    if 'epoch' in checkpoint:
        logger.info(f"Checkpoint epoch: {checkpoint['epoch']}")
    if 'metrics' in checkpoint:
        logger.info(f"Checkpoint metrics: {checkpoint['metrics']}")
    
    # Create test dataloader
    logger.info("Creating test dataloader...")
    train_loader, val_loader = create_dataloaders(config)
    test_loader = val_loader
    logger.info(f"Test dataset size: {len(test_loader.dataset)}")
    
    # Create loss function
    criterion = CombinedLoss(
        bce_weight=config['loss']['bce_weight'],
        dice_weight=config['loss']['dice_weight']
    )
    
    # Run evaluation
    start_time = time.time()
    metrics = evaluate(
        model,
        test_loader,
        criterion,
        device,
        config,
        logger,
        args.output_dir,
        num_visualizations=args.num_visualizations
    )
    elapsed_time = time.time() - start_time
    
    logger.info(f"Evaluation completed in {format_time(elapsed_time)}")
    logger.info(f"Results saved to {args.output_dir}")


if __name__ == '__main__':
    main()
