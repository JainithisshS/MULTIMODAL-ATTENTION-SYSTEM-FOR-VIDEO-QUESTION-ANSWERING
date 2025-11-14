"""
Training Pipeline for Multi-Modal Video Question Answering

This module implements the complete training pipeline including:
- Data loading from preprocessed features
- Loss functions and optimizers
- Training and validation loops
- Model checkpointing and logging
- Learning rate scheduling
- Mixed precision training for CUDA optimization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import json
import os
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
from tqdm import tqdm
import logging

# Import our modules
import sys
sys.path.append(str(Path(__file__).parent.parent))

from configs.config import Config
from src.data_loader import TVQADataset
from src.model.multimodal_qa_model import create_model
from src.training.collate_fn import create_training_collate_fn
from transformers import BertTokenizer


class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance in multiple choice QA"""
    
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = 'mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class LabelSmoothingCrossEntropy(nn.Module):
    """Cross-entropy loss with label smoothing"""
    
    def __init__(self, smoothing: float = 0.1, reduction: str = 'mean'):
        super().__init__()
        self.smoothing = smoothing
        self.reduction = reduction
        
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_prob = F.log_softmax(inputs, dim=-1)
        weight = inputs.new_ones(inputs.size()) * self.smoothing / (inputs.size(-1) - 1.)
        weight.scatter_(-1, targets.unsqueeze(-1), (1. - self.smoothing))
        loss = (-weight * log_prob).sum(dim=-1)
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class MultiModalQATrainer:
    """Complete training pipeline for multimodal video QA"""
    
    def __init__(self, config: Config):
        self.config = config
        self.device = config.DEVICE
        
        # Setup logging
        self.setup_logging()
        
        # Initialize tokenizer
        self.tokenizer = BertTokenizer.from_pretrained(config.TEXT_MODEL)
        
        # Create data loaders
        self.train_loader, self.val_loader, self.test_loader = self.create_data_loaders()
        
        # Create model
        self.model = create_model(config.MODEL_CONFIG).to(self.device)
        self.logger.info(f"Model created with {sum(p.numel() for p in self.model.parameters()):,} parameters")
        
        # Setup loss functions
        self.setup_loss_functions()
        
        # Setup optimizer and scheduler
        self.setup_optimizer()
        
        # Mixed precision training
        self.scaler = GradScaler() if config.USE_MIXED_PRECISION else None
        
        # Tracking variables
        self.global_step = 0
        self.best_val_accuracy = 0.0
        self.patience_counter = 0
        
        # Setup tensorboard
        self.writer = SummaryWriter(log_dir=config.LOGS_DIR / "tensorboard")
        
        # Create checkpoint directory
        self.checkpoint_dir = config.MODELS_DIR / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
    def setup_logging(self):
        """Setup logging configuration"""
        log_dir = self.config.LOGS_DIR
        log_dir.mkdir(parents=True, exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_dir / 'training.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def create_data_loaders(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Create data loaders for train, validation, and test sets"""
        from torch.utils.data import DataLoader
        
        # Create datasets for each split
        train_dataset = TVQADataset(
            features_dir=self.config.PROCESSED_FEATURES_DIR,
            split='train',
            max_qa_per_clip=5,
            random_qa_sampling=True
        )
        
        val_dataset = TVQADataset(
            features_dir=self.config.PROCESSED_FEATURES_DIR,
            split='val',
            max_qa_per_clip=5,
            random_qa_sampling=False  # Deterministic for validation
        )
        
        test_dataset = TVQADataset(
            features_dir=self.config.PROCESSED_FEATURES_DIR,
            split='test',
            max_qa_per_clip=5,
            random_qa_sampling=False  # Deterministic for testing
        )
        
        # Create collate function with tokenizer
        collate_fn = create_training_collate_fn(
            self.tokenizer,
            max_question_length=self.config.MAX_QUESTION_LENGTH,
            max_answer_length=self.config.MAX_ANSWER_LENGTH
        )
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.BATCH_SIZE,
            shuffle=True,
            num_workers=self.config.NUM_WORKERS,
            pin_memory=self.config.PIN_MEMORY,
            collate_fn=collate_fn
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.BATCH_SIZE,
            shuffle=False,
            num_workers=self.config.NUM_WORKERS,
            pin_memory=self.config.PIN_MEMORY,
            collate_fn=collate_fn
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.config.BATCH_SIZE,
            shuffle=False,
            num_workers=self.config.NUM_WORKERS,
            pin_memory=self.config.PIN_MEMORY,
            collate_fn=collate_fn
        )
        
        self.logger.info(f"Data loaders created:")
        self.logger.info(f"  Train samples: {len(train_dataset)}, batches: {len(train_loader)}")
        self.logger.info(f"  Val samples: {len(val_dataset)}, batches: {len(val_loader)}")
        self.logger.info(f"  Test samples: {len(test_dataset)}, batches: {len(test_loader)}")
        
        return train_loader, val_loader, test_loader
        
    def setup_loss_functions(self):
        """Setup loss functions"""
        if hasattr(self.config, 'FOCAL_LOSS_ALPHA'):
            self.criterion = FocalLoss(
                alpha=self.config.FOCAL_LOSS_ALPHA,
                gamma=self.config.FOCAL_LOSS_GAMMA
            )
            self.logger.info("Using Focal Loss")
        elif hasattr(self.config, 'LABEL_SMOOTHING'):
            self.criterion = LabelSmoothingCrossEntropy(
                smoothing=self.config.LABEL_SMOOTHING
            )
            self.logger.info(f"Using Label Smoothing Cross-Entropy (smoothing={self.config.LABEL_SMOOTHING})")
        else:
            self.criterion = nn.CrossEntropyLoss()
            self.logger.info("Using standard Cross-Entropy Loss")
            
    def setup_optimizer(self):
        """Setup optimizer and learning rate scheduler"""
        # Separate parameters for different learning rates
        bert_params = []
        other_params = []
        
        for name, param in self.model.named_parameters():
            if 'bert' in name.lower():
                bert_params.append(param)
            else:
                other_params.append(param)
                
        # Different learning rates for BERT and other components
        param_groups = [
            {'params': bert_params, 'lr': self.config.LEARNING_RATE * 0.1},  # Lower LR for pre-trained BERT
            {'params': other_params, 'lr': self.config.LEARNING_RATE}
        ]
        
        self.optimizer = optim.AdamW(
            param_groups,
            weight_decay=self.config.WEIGHT_DECAY
        )
        
        # Learning rate scheduler with warmup
        total_steps = len(self.train_loader) * self.config.NUM_EPOCHS
        warmup_steps = getattr(self.config, 'WARMUP_STEPS', 1000)
        
        def lr_lambda(step):
            if step < warmup_steps:
                return step / warmup_steps
            else:
                return 0.5 * (1 + np.cos(np.pi * (step - warmup_steps) / (total_steps - warmup_steps)))
                
        self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
        
        self.logger.info(f"Optimizer setup with {len(param_groups)} parameter groups")
        self.logger.info(f"Total training steps: {total_steps}, Warmup steps: {warmup_steps}")
        
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.config.NUM_EPOCHS}")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move batch to device
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass with mixed precision
            if self.config.USE_MIXED_PRECISION:
                with autocast():
                    outputs = self.model(batch)
                    
                    # Only calculate loss for samples with ground truth answers
                    has_answers = batch.get('has_answers', torch.ones(batch['answer_idx'].size(0), dtype=torch.bool))
                    if has_answers.any():
                        # Filter outputs and targets to only include samples with answers
                        valid_outputs = outputs[has_answers]
                        valid_targets = batch['answer_idx'][has_answers]
                        loss = self.criterion(valid_outputs, valid_targets)
                    else:
                        # Skip this batch if no samples have ground truth
                        continue
                
                # Backward pass
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(batch)
                
                # Only calculate loss for samples with ground truth answers
                has_answers = batch.get('has_answers', torch.ones(batch['answer_idx'].size(0), dtype=torch.bool))
                if has_answers.any():
                    # Filter outputs and targets to only include samples with answers
                    valid_outputs = outputs[has_answers]
                    valid_targets = batch['answer_idx'][has_answers]
                    loss = self.criterion(valid_outputs, valid_targets)
                else:
                    # Skip this batch if no samples have ground truth
                    continue
                
                # Backward pass
                loss.backward()
                self.optimizer.step()
            
            # Update learning rate
            self.scheduler.step()
            
            # Calculate accuracy (only for samples with ground truth)
            predictions = torch.argmax(outputs, dim=1)
            has_answers = batch.get('has_answers', torch.ones(batch['answer_idx'].size(0), dtype=torch.bool))
            
            if has_answers.any():
                valid_predictions = predictions[has_answers]
                valid_targets = batch['answer_idx'][has_answers]
                correct = (valid_predictions == valid_targets).sum().item()
                valid_samples = has_answers.sum().item()
            else:
                correct = 0
                valid_samples = 0
            
            # Update metrics
            total_loss += loss.item()
            total_correct += correct
            total_samples += valid_samples
            
            # Update progress bar (avoid division by zero)
            current_accuracy = correct / valid_samples if valid_samples > 0 else 0.0
            progress_bar.set_postfix({
                'Loss': f"{loss.item():.4f}",
                'Acc': f"{current_accuracy:.4f}",
                'LR': f"{self.scheduler.get_last_lr()[0]:.2e}",
                'Valid': f"{valid_samples}/{batch['answer_idx'].size(0)}"
            })
            
            # Log to tensorboard
            if self.global_step % 100 == 0:
                self.writer.add_scalar('Train/Loss', loss.item(), self.global_step)
                self.writer.add_scalar('Train/Accuracy', current_accuracy, self.global_step)
                self.writer.add_scalar('Train/LearningRate', self.scheduler.get_last_lr()[0], self.global_step)
                
            self.global_step += 1
            
        avg_loss = total_loss / len(self.train_loader)
        avg_accuracy = total_correct / total_samples
        
        return {
            'loss': avg_loss,
            'accuracy': avg_accuracy
        }
        
    def validate_epoch(self, epoch: int) -> Dict[str, float]:
        """Validate for one epoch"""
        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        with torch.no_grad():
            progress_bar = tqdm(self.val_loader, desc="Validation")
            
            for batch in progress_bar:
                # Move batch to device
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                
                # Forward pass
                if self.config.USE_MIXED_PRECISION:
                    with autocast():
                        outputs = self.model(batch)
                        
                        # Only calculate loss for samples with ground truth answers
                        has_answers = batch.get('has_answers', torch.ones(batch['answer_idx'].size(0), dtype=torch.bool))
                        if has_answers.any():
                            valid_outputs = outputs[has_answers]
                            valid_targets = batch['answer_idx'][has_answers]
                            loss = self.criterion(valid_outputs, valid_targets)
                        else:
                            continue
                else:
                    outputs = self.model(batch)
                    
                    # Only calculate loss for samples with ground truth answers
                    has_answers = batch.get('has_answers', torch.ones(batch['answer_idx'].size(0), dtype=torch.bool))
                    if has_answers.any():
                        valid_outputs = outputs[has_answers]
                        valid_targets = batch['answer_idx'][has_answers]
                        loss = self.criterion(valid_outputs, valid_targets)
                    else:
                        continue
                
                # Calculate accuracy (only for samples with ground truth)
                predictions = torch.argmax(outputs, dim=1)
                
                if has_answers.any():
                    valid_predictions = predictions[has_answers]
                    valid_targets = batch['answer_idx'][has_answers]
                    correct = (valid_predictions == valid_targets).sum().item()
                    valid_samples = has_answers.sum().item()
                else:
                    correct = 0
                    valid_samples = 0
                
                # Update metrics
                total_loss += loss.item()
                total_correct += correct
                total_samples += valid_samples
                
                # Update progress bar
                current_accuracy = correct / valid_samples if valid_samples > 0 else 0.0
                progress_bar.set_postfix({
                    'Loss': f"{loss.item():.4f}",
                    'Acc': f"{current_accuracy:.4f}",
                    'Valid': f"{valid_samples}/{batch['answer_idx'].size(0)}"
                })
                
        avg_loss = total_loss / len(self.val_loader)
        avg_accuracy = total_correct / total_samples
        
        # Log to tensorboard
        self.writer.add_scalar('Val/Loss', avg_loss, epoch)
        self.writer.add_scalar('Val/Accuracy', avg_accuracy, epoch)
        
        return {
            'loss': avg_loss,
            'accuracy': avg_accuracy
        }
        
    def save_checkpoint(self, epoch: int, metrics: Dict[str, float], is_best: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'metrics': metrics,
            'config': self.config.MODEL_CONFIG,
            'global_step': self.global_step
        }
        if self.scaler:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
            
        # Save latest checkpoint
        checkpoint_path = self.checkpoint_dir / 'latest_checkpoint.pth'
        torch.save(checkpoint, checkpoint_path)
        
        # Save best checkpoint
        if is_best:
            best_checkpoint_path = self.checkpoint_dir / 'best_checkpoint.pth'
            torch.save(checkpoint, best_checkpoint_path)
            self.logger.info(f"New best model saved with validation accuracy: {metrics['val_accuracy']:.4f}")
            
    def load_checkpoint(self, checkpoint_path: str) -> int:
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.global_step = checkpoint['global_step']
        if self.scaler and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        self.logger.info(f"Checkpoint loaded from {checkpoint_path}")
        return checkpoint['epoch']
        
    def train(self, resume_from_checkpoint: Optional[str] = None):
        """Main training loop"""
        start_epoch = 0
        
        if resume_from_checkpoint:
            start_epoch = self.load_checkpoint(resume_from_checkpoint) + 1
            
        self.logger.info("Starting training...")
        self.logger.info(f"Device: {self.device}")
        self.logger.info(f"Mixed precision: {self.config.USE_MIXED_PRECISION}")
        self.logger.info(f"Batch size: {self.config.BATCH_SIZE}")
        
        for epoch in range(start_epoch, self.config.NUM_EPOCHS):
            epoch_start_time = time.time()
            
            # Training
            train_metrics = self.train_epoch(epoch)
            
            # Validation
            val_metrics = self.validate_epoch(epoch)
            
            epoch_time = time.time() - epoch_start_time
            
            # Combine metrics
            metrics = {
                'train_loss': train_metrics['loss'],
                'train_accuracy': train_metrics['accuracy'],
                'val_loss': val_metrics['loss'],
                'val_accuracy': val_metrics['accuracy'],
                'epoch_time': epoch_time
            }
            
            # Log epoch results
            self.logger.info(
                f"Epoch {epoch+1}/{self.config.NUM_EPOCHS} - "
                f"Train Loss: {train_metrics['loss']:.4f}, Train Acc: {train_metrics['accuracy']:.4f}, "
                f"Val Loss: {val_metrics['loss']:.4f}, Val Acc: {val_metrics['accuracy']:.4f}, "
                f"Time: {epoch_time:.1f}s"
            )
            
            # Check for best model
            is_best = val_metrics['accuracy'] > self.best_val_accuracy
            if is_best:
                self.best_val_accuracy = val_metrics['accuracy']
                self.patience_counter = 0
            else:
                self.patience_counter += 1
                
            # Save checkpoint
            self.save_checkpoint(epoch, metrics, is_best)
            
            # Early stopping
            if self.patience_counter >= self.config.EARLY_STOPPING_PATIENCE:
                self.logger.info(f"Early stopping triggered after {epoch+1} epochs")
                break
                
        self.logger.info("Training completed!")
        self.logger.info(f"Best validation accuracy: {self.best_val_accuracy:.4f}")
        
        # Close tensorboard writer
        self.writer.close()
        
    def evaluate_test_set(self, checkpoint_path: str = None):
        """Evaluate on test set"""
        if checkpoint_path:
            self.load_checkpoint(checkpoint_path)
        
        self.model.eval()
        total_correct = 0
        total_samples = 0
        
        with torch.no_grad():
            progress_bar = tqdm(self.test_loader, desc="Testing")
            
            for batch in progress_bar:
                # Move batch to device
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                
                # Forward pass
                outputs = self.model(batch)
                predictions = torch.argmax(outputs, dim=1)
                correct = (predictions == batch['answer_idx']).sum().item()
                
                total_correct += correct
                total_samples += batch['answer_idx'].size(0)
                
                # Update progress bar
                current_accuracy = correct / batch['answer_idx'].size(0)
                progress_bar.set_postfix({'Acc': f"{current_accuracy:.4f}"})
                
        test_accuracy = total_correct / total_samples
        self.logger.info(f"Test Accuracy: {test_accuracy:.4f}")
        
        return test_accuracy


def main():
    """Main training function"""
    config = Config()
    
    # Create trainer
    trainer = MultiModalQATrainer(config)
    
    # Start training
    trainer.train()
    
    # Evaluate on test set with best checkpoint
    best_checkpoint = trainer.checkpoint_dir / 'best_checkpoint.pth'
    if best_checkpoint.exists():
        trainer.evaluate_test_set(str(best_checkpoint))


if __name__ == "__main__":
    main()
