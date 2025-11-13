"""
Training script for image editing mask prediction model.

Usage:
    python train.py --config config.yaml
    python train.py --config config.yaml --resume checkpoints/checkpoint_epoch_10.pth
"""

import argparse
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# Import model components
from model import build_model
from dataset import create_dataloaders
from losses import CombinedLoss, compute_metrics
from utils import (
    load_config,
    save_config,
    setup_logging,
    set_seed,
    get_device,
    count_parameters,
    save_checkpoint,
    load_checkpoint,
    manage_checkpoints,
    visualize_predictions,
    MetricsTracker,
    format_time,
    print_config
)


def create_optimizer(model, config):
    """Create optimizer based on configuration."""
    optimizer_name = config['training']['optimizer'].lower()
    lr = config['training']['learning_rate']
    weight_decay = config['training']['weight_decay']
    
    if optimizer_name == 'adamw':
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
    elif optimizer_name == 'adam':
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
    elif optimizer_name == 'sgd':
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=0.9,
            weight_decay=weight_decay
        )
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")
    
    return optimizer


def create_scheduler(optimizer, config, steps_per_epoch):
    """Create learning rate scheduler."""
    scheduler_name = config['training']['lr_scheduler'].lower()
    
    if scheduler_name == 'none':
        return None
    
    num_epochs = config['training']['num_epochs']
    warmup_epochs = config['training'].get('warmup_epochs', 0)
    
    if scheduler_name == 'cosine':
        total_steps = num_epochs * steps_per_epoch
        warmup_steps = warmup_epochs * steps_per_epoch
        
        def lr_lambda(step):
            if step < warmup_steps:
                return step / warmup_steps
            else:
                progress = (step - warmup_steps) / (total_steps - warmup_steps)
                min_lr_ratio = config['training']['min_lr'] / config['training']['learning_rate']
                return min_lr_ratio + (1 - min_lr_ratio) * 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159)))
        
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        return scheduler
    
    elif scheduler_name == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=10 * steps_per_epoch,
            gamma=0.1
        )
        return scheduler
    
    elif scheduler_name == 'plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='max',
            factor=0.5,
            patience=5,
            verbose=True
        )
        return scheduler
    
    else:
        raise ValueError(f"Unsupported scheduler: {scheduler_name}")


def train_one_epoch(
    model,
    train_loader,
    criterion,
    optimizer,
    scheduler,
    device,
    epoch,
    config,
    logger,
    writer=None,
    steps_per_epoch=1000
):
    """Train for one epoch."""
    model.train()
    metrics_tracker = MetricsTracker()
    
    use_amp = config['training']['use_amp']
    scaler = GradScaler() if use_amp else None
    max_grad_norm = config['training']['max_grad_norm']
    
    log_every = config['logging']['log_every_n_steps']
    log_images_every = config['logging']['log_images_every_n_steps']
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
    global_step = epoch * steps_per_epoch
    
    for batch_idx, batch in enumerate(pbar):
        # Move to device
        images = batch['image'].to(device)
        masks = batch['mask'].to(device)
        text_tokens = batch['text_tokens'].to(device)
        
        # Forward pass
        optimizer.zero_grad()
        
        if use_amp:
            with autocast():
                pred_logits = model(images, text_tokens)
                loss_dict = criterion(pred_logits, masks)
                loss = loss_dict['loss']
        else:
            pred_logits = model(images, text_tokens)
            loss_dict = criterion(pred_logits, masks)
            loss = loss_dict['loss']
        
        # Backward pass
        if use_amp:
            scaler.scale(loss).backward()
            if max_grad_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
        
        # Update scheduler (if step-based)
        if scheduler is not None and config['training']['lr_scheduler'] in ['cosine', 'step']:
            scheduler.step()
        
        # Compute metrics
        with torch.no_grad():
            batch_metrics = compute_metrics(pred_logits, masks)
        
        # Update tracking
        metrics_tracker.update({
            'loss': loss.item(),
            'bce_loss': loss_dict['bce_loss'],
            'dice_loss': loss_dict['dice_loss'],
            **batch_metrics
        })
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f"{metrics_tracker.metrics['loss'].avg:.4f}",
            'iou': f"{metrics_tracker.metrics['iou'].avg:.4f}"
        })
        
        # Logging
        if (batch_idx + 1) % log_every == 0:
            current_lr = optimizer.param_groups[0]['lr']
            avg_metrics = metrics_tracker.get_averages()
            
            log_msg = (
                f"Epoch {epoch} [{batch_idx+1}/{len(train_loader)}] "
                f"Loss: {avg_metrics['loss']:.4f} | "
                f"IoU: {avg_metrics['iou']:.4f} | "
                f"F1: {avg_metrics['f1']:.4f} | "
                f"LR: {current_lr:.6f}"
            )
            logger.info(log_msg)
            
            # TensorBoard logging
            if writer is not None:
                step = global_step + batch_idx
                writer.add_scalar('train/loss', avg_metrics['loss'], step)
                writer.add_scalar('train/bce_loss', avg_metrics['bce_loss'], step)
                writer.add_scalar('train/dice_loss', avg_metrics['dice_loss'], step)
                writer.add_scalar('train/iou', avg_metrics['iou'], step)
                writer.add_scalar('train/f1', avg_metrics['f1'], step)
                writer.add_scalar('train/lr', current_lr, step)
        
        # Log images
        if (batch_idx + 1) % log_images_every == 0:
            num_images = config['logging']['num_images_to_log']
            fig = visualize_predictions(
                images, batch['text'], pred_logits, masks,
                num_samples=num_images
            )
            
            if writer is not None:
                writer.add_figure('train/predictions', fig, global_step + batch_idx)
    
    return metrics_tracker.get_averages()


@torch.no_grad()
def validate(model, val_loader, criterion, device, epoch, config, logger, writer=None):
    """Validate the model."""
    model.eval()
    metrics_tracker = MetricsTracker()
    
    pbar = tqdm(val_loader, desc=f"Validation Epoch {epoch}")
    
    all_predictions = []
    all_masks = []
    all_images = []
    all_texts = []
    
    for batch_idx, batch in enumerate(pbar):
        # Move to device
        images = batch['image'].to(device)
        masks = batch['mask'].to(device)
        text_tokens = batch['text_tokens'].to(device)
        
        # Forward pass
        pred_logits = model(images, text_tokens)
        loss_dict = criterion(pred_logits, masks)
        
        # Compute metrics
        batch_metrics = compute_metrics(pred_logits, masks)
        
        # Update tracking
        metrics_tracker.update({
            'loss': loss_dict['loss'].item(),
            'bce_loss': loss_dict['bce_loss'],
            'dice_loss': loss_dict['dice_loss'],
            **batch_metrics
        })
        
        # Store for visualization
        if batch_idx == 0:
            all_images = images[:4].cpu()
            all_masks = masks[:4].cpu()
            all_predictions = pred_logits[:4].cpu()
            all_texts = batch['text'][:4]
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f"{metrics_tracker.metrics['loss'].avg:.4f}",
            'iou': f"{metrics_tracker.metrics['iou'].avg:.4f}"
        })
    
    # Get average metrics
    avg_metrics = metrics_tracker.get_averages()
    
    # Log metrics
    log_msg = (
        f"Validation Epoch {epoch} | "
        f"Loss: {avg_metrics['loss']:.4f} | "
        f"IoU: {avg_metrics['iou']:.4f} | "
        f"F1: {avg_metrics['f1']:.4f} | "
        f"Precision: {avg_metrics['precision']:.4f} | "
        f"Recall: {avg_metrics['recall']:.4f}"
    )
    logger.info(log_msg)
    
    # TensorBoard logging
    if writer is not None:
        for key, value in avg_metrics.items():
            writer.add_scalar(f'val/{key}', value, epoch)
        
        # Log validation images
        fig = visualize_predictions(
            all_images, all_texts, all_predictions, all_masks,
            num_samples=4
        )
        writer.add_figure('val/predictions', fig, epoch)
    
    return avg_metrics


def train(config, resume_checkpoint=None):
    """Main training function."""

    os.makedirs(config['training']['checkpoint_dir'], exist_ok=True)
    os.makedirs(config['logging']['log_dir'], exist_ok=True)
    
    logger = setup_logging(
        config['logging']['log_dir'],
        config['logging']['experiment_name']
    )
    
    print_config(config, logger)
    
    set_seed(config['system']['seed'], config['system']['deterministic'])
    
    device = get_device(config)
    logger.info(f"Using device: {device}")
    
    logger.info("Creating dataloaders...")
    train_loader, val_loader = create_dataloaders(config)
    
    # WebDataset doesn't have len(), so we can't print batch counts
    logger.info("Dataloaders created successfully")
    
    # Build model
    logger.info("Building model...")
    model = build_model(config)
    model = model.to(device)
    
    param_counts = count_parameters(model)
    logger.info(f"Model parameters: {param_counts['total']:,} total, {param_counts['trainable']:,} trainable")
    
    num_train_samples = config['data'].get('num_train_samples')
    batch_size = config['training']['batch_size']
    steps_per_epoch = (num_train_samples + batch_size - 1) // batch_size
    logger.info(f"Calculated steps per epoch: {steps_per_epoch} ({num_train_samples} samples / batch size {batch_size})")
    # Create optimizer and scheduler
    optimizer = create_optimizer(model, config)
    scheduler = create_scheduler(optimizer, config, steps_per_epoch=steps_per_epoch)
    
    criterion = CombinedLoss(
        bce_weight=config['loss']['bce_weight'],
        dice_weight=config['loss']['dice_weight']
    )
    
    # Setup logging
    writer = None
    if config['logging']['use_tensorboard']:
        tensorboard_dir = os.path.join(
            config['logging']['tensorboard_dir'],
            config['logging']['experiment_name']
        )
        writer = SummaryWriter(tensorboard_dir)
        logger.info(f"TensorBoard logging to: {tensorboard_dir}")

    # Resume from checkpoint
    start_epoch = 0
    best_iou = 0.0
    
    if resume_checkpoint is not None and os.path.exists(resume_checkpoint):
        logger.info(f"Resuming from checkpoint: {resume_checkpoint}")
        checkpoint = load_checkpoint(resume_checkpoint, model, optimizer, scheduler, device)
        start_epoch = checkpoint['epoch'] + 1
        best_iou = checkpoint['metrics'].get('iou', 0.0)
        logger.info(f"Resumed from epoch {start_epoch}, best IoU: {best_iou:.4f}")
    
    config_save_path = os.path.join(config['training']['checkpoint_dir'], 'config.yaml')
    save_config(config, config_save_path)
    
    # Training loop
    logger.info("Starting training...")
    num_epochs = config['training']['num_epochs']
    
    for epoch in range(start_epoch, num_epochs):
        epoch_start_time = time.time()
        
        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, scheduler,
            device, epoch, config, logger, writer,
            steps_per_epoch=steps_per_epoch
        )
        
        # Validate
        if (epoch + 1) % config['training']['val_every_n_epochs'] == 0:
            val_metrics = validate(
                model, val_loader, criterion, device, epoch,
                config, logger, writer
            )
            
            current_iou = val_metrics['iou']
            
            # Update scheduler (if metric-based)
            if scheduler is not None and config['training']['lr_scheduler'] == 'plateau':
                scheduler.step(current_iou)
            
            # Save checkpoint
            is_best = current_iou > best_iou
            if is_best:
                best_iou = current_iou
                logger.info(f"New best IoU: {best_iou:.4f}")
            
            # Save regular checkpoint
            if (epoch + 1) % config['training']['save_every_n_epochs'] == 0:
                checkpoint_path = os.path.join(
                    config['training']['checkpoint_dir'],
                    f"checkpoint_epoch_{epoch}.pth"
                )
                save_checkpoint(
                    model, optimizer, scheduler, epoch, val_metrics, config, checkpoint_path
                )
                logger.info(f"Saved checkpoint: {checkpoint_path}")
            
            # Save best checkpoint
            if is_best or not config['training']['save_best_only']:
                best_path = os.path.join(
                    config['training']['checkpoint_dir'],
                    "best_model.pth"
                )
                save_checkpoint(
                    model, optimizer, scheduler, epoch, val_metrics, config, best_path
                )
                if is_best:
                    logger.info(f"Saved best model: {best_path}")
        
            manage_checkpoints(
                config['training']['checkpoint_dir'],
                config['training']['keep_last_n_checkpoints']
            )
        
        epoch_time = time.time() - epoch_start_time
        logger.info(f"Epoch {epoch} completed in {format_time(epoch_time)}")
    
    if writer is not None:
        writer.close()
    
    logger.info("Training complete!")


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Train mask prediction model")
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='Path to checkpoint to resume from'
    )
    
    args = parser.parse_args()
    
    config = load_config(args.config)
    
    train(config, resume_checkpoint=args.resume)


if __name__ == '__main__':
    main()

