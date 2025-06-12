# Module 4, Lesson 4: Training Loop Implementation

## 🎯 Learning Objectives
By the end of this lesson, you will:
- Implement complete DINO training loop with all optimizations
- Master gradient clipping and learning rate scheduling strategies
- Build robust checkpoint management and resumable training
- Create comprehensive monitoring and logging systems

## 📚 Theoretical Background

### DINO Training Loop Components

The **complete DINO training loop** integrates:

1. **Data Loading**: Multi-crop augmented batches
2. **Forward Pass**: Student-teacher network execution
3. **Loss Computation**: Complete DINO loss with all components
4. **Gradient Management**: Clipping and optimization
5. **Teacher Updates**: EMA weight synchronization
6. **Monitoring**: Loss tracking and validation
7. **Checkpointing**: State preservation and recovery

### Optimization Strategy

**Learning Rate Schedule**:
- **Warmup**: Linear increase for first few epochs
- **Cosine Decay**: Smooth reduction over training
- **Base LR**: Typically 0.0005 * batch_size / 256

**Gradient Clipping**:
- **Global Norm Clipping**: Clip gradients to maximum norm (usually 3.0)
- **Prevents Instability**: Avoids gradient explosion
- **Maintains Training**: Allows aggressive learning rates

**Weight Decay**:
- **L2 Regularization**: Typically 0.04-0.1
- **Backbone vs Head**: Different decay rates for different components

## 🛠️ Implementation

### Step 1: Complete Training Loop Implementation

```python
# training_loop.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import time
import math
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import wandb
import numpy as np
from collections import defaultdict
import json

class DINOTrainingLoop:
    """
    Complete DINO training loop with all optimizations
    """
    
    def __init__(
        self,
        model,  # Student-teacher model
        loss_fn,  # Complete DINO loss
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
        device: str = 'cuda',
        
        # Optimization parameters
        base_lr: float = 0.0005,
        weight_decay: float = 0.04,
        momentum: float = 0.9,
        batch_size: int = 64,
        
        # Training schedule
        epochs: int = 100,
        warmup_epochs: int = 10,
        lr_schedule: str = 'cosine',
        
        # Gradient management
        clip_grad_norm: float = 3.0,
        freeze_last_layer: int = 1,
        
        # Monitoring and saving
        log_interval: int = 50,
        val_interval: int = 1000,
        save_interval: int = 5000,
        save_dir: str = './checkpoints',
        
        # Experiment tracking
        use_wandb: bool = False,
        experiment_name: str = 'dino_training',
        
        # Advanced options
        mixed_precision: bool = True,
        compile_model: bool = False,
        gradient_accumulation_steps: int = 1
    ):
        self.model = model
        self.loss_fn = loss_fn
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.device = device
        
        # Training configuration
        self.base_lr = base_lr
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.batch_size = batch_size
        self.epochs = epochs
        self.warmup_epochs = warmup_epochs
        self.lr_schedule = lr_schedule
        self.clip_grad_norm = clip_grad_norm
        self.freeze_last_layer = freeze_last_layer
        
        # Monitoring configuration
        self.log_interval = log_interval
        self.val_interval = val_interval
        self.save_interval = save_interval
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Experiment tracking
        self.use_wandb = use_wandb
        self.experiment_name = experiment_name
        
        # Advanced options
        self.mixed_precision = mixed_precision
        self.compile_model = compile_model
        self.gradient_accumulation_steps = gradient_accumulation_steps
        
        # Initialize components
        self._setup_optimizer_and_scheduler()
        self._setup_logging()
        self._setup_monitoring()
        self._setup_mixed_precision()
        
        # Training state
        self.global_step = 0
        self.current_epoch = 0
        self.best_loss = float('inf')
        self.training_start_time = None
        
        # Move model to device
        self.model.to(device)
        self.loss_fn.to(device)
        
        # Model compilation (PyTorch 2.0+)
        if self.compile_model:
            try:
                self.model = torch.compile(self.model)
                self.logger.info("Model compiled successfully")
            except Exception as e:
                self.logger.warning(f"Model compilation failed: {e}")
    
    def _setup_optimizer_and_scheduler(self):
        """Setup optimizer and learning rate scheduler"""
        
        # Create parameter groups with different weight decay
        params_with_decay = []
        params_without_decay = []
        
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                # No weight decay for bias and normalization layers
                if 'bias' in name or 'norm' in name or 'bn' in name:
                    params_without_decay.append(param)
                else:
                    params_with_decay.append(param)
        
        param_groups = [
            {'params': params_with_decay, 'weight_decay': self.weight_decay},
            {'params': params_without_decay, 'weight_decay': 0.0}
        ]
        
        # Create optimizer
        self.optimizer = optim.AdamW(
            param_groups,
            lr=self.base_lr,
            betas=(0.9, 0.95),
            eps=1e-8
        )
        
        # Create learning rate scheduler
        total_steps = len(self.train_dataloader) * self.epochs
        warmup_steps = len(self.train_dataloader) * self.warmup_epochs
        
        if self.lr_schedule == 'cosine':
            self.scheduler = optim.lr_scheduler.LambdaLR(
                self.optimizer,
                lr_lambda=lambda step: self._cosine_schedule_with_warmup(
                    step, warmup_steps, total_steps
                )
            )
        elif self.lr_schedule == 'linear':
            self.scheduler = optim.lr_scheduler.LambdaLR(
                self.optimizer,
                lr_lambda=lambda step: self._linear_schedule_with_warmup(
                    step, warmup_steps, total_steps
                )
            )
        else:
            self.scheduler = None
    
    def _cosine_schedule_with_warmup(self, step: int, warmup_steps: int, total_steps: int) -> float:
        """Cosine learning rate schedule with warmup"""
        if step < warmup_steps:
            return step / warmup_steps
        else:
            progress = (step - warmup_steps) / (total_steps - warmup_steps)
            return 0.5 * (1 + math.cos(math.pi * progress))
    
    def _linear_schedule_with_warmup(self, step: int, warmup_steps: int, total_steps: int) -> float:
        """Linear learning rate schedule with warmup"""
        if step < warmup_steps:
            return step / warmup_steps
        else:
            return max(0.0, (total_steps - step) / (total_steps - warmup_steps))
    
    def _setup_logging(self):
        """Setup logging system"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.save_dir / 'training.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger('DINOTraining')
    
    def _setup_monitoring(self):
        """Setup experiment monitoring"""
        if self.use_wandb:
            wandb.init(
                project='dino-training',
                name=self.experiment_name,
                config={
                    'base_lr': self.base_lr,
                    'weight_decay': self.weight_decay,
                    'batch_size': self.batch_size,
                    'epochs': self.epochs,
                    'warmup_epochs': self.warmup_epochs,
                    'lr_schedule': self.lr_schedule,
                    'clip_grad_norm': self.clip_grad_norm,
                }
            )
        
        # Metrics tracking
        self.metrics_history = defaultdict(list)
        self.epoch_metrics = defaultdict(list)
    
    def _setup_mixed_precision(self):
        """Setup mixed precision training"""
        if self.mixed_precision:
            self.scaler = torch.cuda.amp.GradScaler()
        else:
            self.scaler = None
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Single training step
        
        Args:
            batch: Dictionary with 'global_crops', 'local_crops', 'batch_size'
            
        Returns:
            Dictionary with step metrics
        """
        
        # Extract batch data
        global_crops = batch['global_crops'].to(self.device, non_blocking=True)
        local_crops = batch['local_crops'].to(self.device, non_blocking=True)
        batch_size = batch['batch_size']
        
        # Forward pass with mixed precision
        if self.mixed_precision:
            with torch.cuda.amp.autocast():
                step_metrics = self._forward_and_loss(global_crops, local_crops, batch_size)
                loss = step_metrics['total_loss_tensor']
        else:
            step_metrics = self._forward_and_loss(global_crops, local_crops, batch_size)
            loss = step_metrics['total_loss_tensor']
        
        # Backward pass with gradient accumulation
        if self.gradient_accumulation_steps > 1:
            loss = loss / self.gradient_accumulation_steps
        
        if self.mixed_precision:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()
        
        # Optimizer step (if gradient accumulation is complete)
        if (self.global_step + 1) % self.gradient_accumulation_steps == 0:
            self._optimizer_step(step_metrics)
            
            # Update teacher weights
            self.model.update_teacher(self.current_epoch)
        
        # Clean up tensor to avoid memory issues
        step_metrics.pop('total_loss_tensor', None)
        
        return step_metrics
    
    def _forward_and_loss(
        self, 
        global_crops: torch.Tensor, 
        local_crops: torch.Tensor, 
        batch_size: int
    ) -> Dict[str, Any]:
        """Forward pass and loss computation"""
        
        # Student processes all crops
        student_global_out = self.model.forward_student(global_crops)
        student_local_out = self.model.forward_student(local_crops)
        
        # Teacher processes only global crops  
        teacher_global_out = self.model.forward_teacher(global_crops)
        
        # Compute DINO loss
        loss, loss_dict = self.loss_fn(
            student_global_crops=student_global_out,
            student_local_crops=student_local_out,
            teacher_global_crops=teacher_global_out,
            epoch=self.current_epoch,
            update_center=True
        )
        
        # Add tensor version for backward pass
        loss_dict['total_loss_tensor'] = loss
        
        # Add training metadata
        loss_dict.update({
            'global_step': self.global_step,
            'epoch': self.current_epoch,
            'learning_rate': self.optimizer.param_groups[0]['lr']
        })
        
        return loss_dict
    
    def _optimizer_step(self, step_metrics: Dict[str, Any]):
        """Perform optimizer step with gradient clipping"""
        
        # Gradient clipping
        if self.clip_grad_norm > 0:
            if self.mixed_precision:
                # Unscale gradients before clipping
                self.scaler.unscale_(self.optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.get_student_parameters(), self.clip_grad_norm
                )
                step_metrics['grad_norm'] = grad_norm.item()
                
                # Optimizer step with scaling
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.get_student_parameters(), self.clip_grad_norm
                )
                step_metrics['grad_norm'] = grad_norm.item()
                self.optimizer.step()
        else:
            if self.mixed_precision:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()
        
        # Clear gradients
        self.optimizer.zero_grad()
        
        # Update learning rate
        if self.scheduler is not None:
            self.scheduler.step()
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch"""
        
        self.current_epoch = epoch
        self.model.train()
        
        epoch_start_time = time.time()
        step_times = []
        
        # Freeze last layer for first few epochs
        if epoch < self.freeze_last_layer:
            self._freeze_last_layer()
        else:
            self._unfreeze_last_layer()
        
        for batch_idx, batch in enumerate(self.train_dataloader):
            step_start_time = time.time()
            
            # Training step
            step_metrics = self.train_step(batch)
            
            # Track step time
            step_time = time.time() - step_start_time
            step_times.append(step_time)
            step_metrics['step_time'] = step_time
            
            # Update metrics tracking
            for key, value in step_metrics.items():
                if isinstance(value, (int, float)):
                    self.metrics_history[key].append(value)
                    self.epoch_metrics[key].append(value)
            
            # Logging
            if self.global_step % self.log_interval == 0:
                self._log_training_step(step_metrics, step_times)
            
            # Validation
            if self.val_dataloader and self.global_step % self.val_interval == 0:
                val_metrics = self.validate()
                if self.use_wandb:
                    wandb.log({f'val_{k}': v for k, v in val_metrics.items()}, 
                             step=self.global_step)
            
            # Checkpointing
            if self.global_step % self.save_interval == 0:
                self._save_checkpoint(f'checkpoint_step_{self.global_step}.pt')
            
            # WandB logging
            if self.use_wandb:
                wandb.log(step_metrics, step=self.global_step)
            
            self.global_step += 1
        
        # Epoch summary
        epoch_time = time.time() - epoch_start_time
        epoch_summary = self._compute_epoch_summary(epoch_time, step_times)
        
        # Clear epoch metrics
        self.epoch_metrics.clear()
        
        return epoch_summary
    
    def _freeze_last_layer(self):
        """Freeze last layer parameters"""
        if hasattr(self.model, 'student') and hasattr(self.model.student, 'projection_head'):
            last_layer = self.model.student.projection_head.final_layer
            for param in last_layer.parameters():
                param.requires_grad = False
    
    def _unfreeze_last_layer(self):
        """Unfreeze last layer parameters"""
        if hasattr(self.model, 'student') and hasattr(self.model.student, 'projection_head'):
            last_layer = self.model.student.projection_head.final_layer
            for param in last_layer.parameters():
                param.requires_grad = True
    
    def validate(self) -> Dict[str, float]:
        """Run validation"""
        if not self.val_dataloader:
            return {}
        
        self.model.eval()
        val_metrics = defaultdict(list)
        
        with torch.no_grad():
            for batch in self.val_dataloader:
                global_crops = batch['global_crops'].to(self.device, non_blocking=True)
                local_crops = batch['local_crops'].to(self.device, non_blocking=True)
                batch_size = batch['batch_size']
                
                # Forward pass
                if self.mixed_precision:
                    with torch.cuda.amp.autocast():
                        step_metrics = self._forward_and_loss(global_crops, local_crops, batch_size)
                else:
                    step_metrics = self._forward_and_loss(global_crops, local_crops, batch_size)
                
                # Accumulate metrics
                for key, value in step_metrics.items():
                    if isinstance(value, (int, float)) and key != 'total_loss_tensor':
                        val_metrics[key].append(value)
        
        # Compute averages
        val_summary = {key: np.mean(values) for key, values in val_metrics.items()}
        
        self.model.train()
        return val_summary
    
    def _log_training_step(self, step_metrics: Dict[str, float], step_times: List[float]):
        """Log training step information"""
        avg_step_time = np.mean(step_times[-10:])  # Average of last 10 steps
        
        self.logger.info(
            f"Step {self.global_step:6d} | "
            f"Epoch {self.current_epoch:3d} | "
            f"Loss: {step_metrics.get('total_loss', 0):.4f} | "
            f"Global: {step_metrics.get('global_loss', 0):.4f} | "
            f"Local: {step_metrics.get('local_loss', 0):.4f} | "
            f"LR: {step_metrics.get('learning_rate', 0):.6f} | "
            f"Time: {avg_step_time:.3f}s"
        )
    
    def _compute_epoch_summary(self, epoch_time: float, step_times: List[float]) -> Dict[str, float]:
        """Compute epoch summary statistics"""
        
        summary = {
            'epoch': self.current_epoch,
            'epoch_time': epoch_time,
            'avg_step_time': np.mean(step_times),
            'steps_per_second': len(step_times) / epoch_time,
        }
        
        # Add averages of epoch metrics
        for key, values in self.epoch_metrics.items():
            if isinstance(values[0], (int, float)):
                summary[f'avg_{key}'] = np.mean(values)
        
        self.logger.info(
            f"Epoch {self.current_epoch} completed in {epoch_time:.2f}s | "
            f"Avg Loss: {summary.get('avg_total_loss', 0):.4f} | "
            f"Steps/sec: {summary['steps_per_second']:.2f}"
        )
        
        return summary
    
    def _save_checkpoint(self, filename: str):
        """Save training checkpoint"""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss_state_dict': self.loss_fn.state_dict(),
            'global_step': self.global_step,
            'current_epoch': self.current_epoch,
            'best_loss': self.best_loss,
            'metrics_history': dict(self.metrics_history),
        }
        
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        if self.mixed_precision:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        checkpoint_path = self.save_dir / filename
        torch.save(checkpoint, checkpoint_path)
        self.logger.info(f"Checkpoint saved: {checkpoint_path}")
        
        # Save best model
        current_loss = self.metrics_history['total_loss'][-1] if self.metrics_history['total_loss'] else float('inf')
        if current_loss < self.best_loss:
            self.best_loss = current_loss
            best_path = self.save_dir / 'best_model.pt'
            torch.save(checkpoint, best_path)
            self.logger.info(f"New best model saved: {best_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load training checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.loss_fn.load_state_dict(checkpoint['loss_state_dict'])
        self.global_step = checkpoint['global_step']
        self.current_epoch = checkpoint['current_epoch']
        self.best_loss = checkpoint.get('best_loss', float('inf'))
        
        if 'metrics_history' in checkpoint:
            self.metrics_history = defaultdict(list, checkpoint['metrics_history'])
        
        if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        if self.mixed_precision and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        self.logger.info(f"Checkpoint loaded from {checkpoint_path}")
    
    def train(self, resume_from: Optional[str] = None) -> Dict[str, Any]:
        """
        Main training loop
        
        Args:
            resume_from: Path to checkpoint to resume from
            
        Returns:
            Training summary and metrics
        """
        
        # Resume from checkpoint if specified
        if resume_from:
            self.load_checkpoint(resume_from)
        
        self.training_start_time = time.time()
        
        try:
            self.logger.info(f"Starting DINO training for {self.epochs} epochs")
            self.logger.info(f"Total steps: {len(self.train_dataloader) * self.epochs}")
            
            for epoch in range(self.current_epoch, self.epochs):
                epoch_summary = self.train_epoch(epoch)
                
                # Log epoch summary
                if self.use_wandb:
                    wandb.log({f'epoch_{k}': v for k, v in epoch_summary.items()}, 
                             step=self.global_step)
                
                # Save end-of-epoch checkpoint
                if epoch % 5 == 0 or epoch == self.epochs - 1:
                    self._save_checkpoint(f'epoch_{epoch}.pt')
        
        except KeyboardInterrupt:
            self.logger.info("Training interrupted by user")
            self._save_checkpoint('interrupted.pt')
        
        except Exception as e:
            self.logger.error(f"Training failed with error: {e}")
            self._save_checkpoint('error.pt')
            raise
        
        finally:
            # Training completed
            total_training_time = time.time() - self.training_start_time
            self.logger.info(f"Training completed in {total_training_time:.2f}s")
            
            if self.use_wandb:
                wandb.finish()
        
        return self._generate_training_summary(total_training_time)
    
    def _generate_training_summary(self, total_time: float) -> Dict[str, Any]:
        """Generate comprehensive training summary"""
        
        summary = {
            'total_training_time': total_time,
            'total_steps': self.global_step,
            'total_epochs': self.current_epoch + 1,
            'best_loss': self.best_loss,
            'final_loss': self.metrics_history['total_loss'][-1] if self.metrics_history['total_loss'] else None,
            'avg_step_time': np.mean(self.metrics_history.get('step_time', [0])),
        }
        
        # Add final metrics
        for key, values in self.metrics_history.items():
            if values and isinstance(values[-1], (int, float)):
                summary[f'final_{key}'] = values[-1]
                summary[f'avg_{key}'] = np.mean(values)
                summary[f'min_{key}'] = np.min(values)
                summary[f'max_{key}'] = np.max(values)
        
        # Save summary to file
        summary_path = self.save_dir / 'training_summary.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        return summary
```

### Step 2: Advanced Training Utilities

```python
# training_utilities.py
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

class EarlyStopping:
    """Early stopping utility for DINO training"""
    
    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.001,
        mode: str = 'min',
        restore_best_weights: bool = True
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.restore_best_weights = restore_best_weights
        
        self.best_score = None
        self.counter = 0
        self.best_weights = None
        
    def __call__(self, score: float, model: nn.Module) -> bool:
        """
        Check if training should stop
        
        Returns:
            True if training should stop, False otherwise
        """
        
        if self.best_score is None:
            self.best_score = score
            if self.restore_best_weights:
                self.best_weights = model.state_dict().copy()
        elif self._is_better(score, self.best_score):
            self.best_score = score
            self.counter = 0
            if self.restore_best_weights:
                self.best_weights = model.state_dict().copy()
        else:
            self.counter += 1
        
        if self.counter >= self.patience:
            if self.restore_best_weights and self.best_weights:
                model.load_state_dict(self.best_weights)
            return True
        
        return False
    
    def _is_better(self, current: float, best: float) -> bool:
        """Check if current score is better than best"""
        if self.mode == 'min':
            return current < best - self.min_delta
        else:
            return current > best + self.min_delta


class GradientMonitor:
    """Monitor gradient statistics during training"""
    
    def __init__(self, model: nn.Module):
        self.model = model
        self.gradient_history = []
    
    def log_gradients(self):
        """Log current gradient statistics"""
        total_norm = 0.0
        param_count = 0
        
        gradients = {}
        
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
                param_count += param.numel()
                
                gradients[name] = {
                    'norm': param_norm.item(),
                    'mean': param.grad.data.mean().item(),
                    'std': param.grad.data.std().item(),
                    'max': param.grad.data.max().item(),
                    'min': param.grad.data.min().item()
                }
        
        total_norm = total_norm ** (1. / 2)
        
        gradient_stats = {
            'total_norm': total_norm,
            'param_count': param_count,
            'per_param': gradients
        }
        
        self.gradient_history.append(gradient_stats)
        return gradient_stats
    
    def plot_gradient_norms(self, save_path: Optional[str] = None):
        """Plot gradient norm evolution"""
        if not self.gradient_history:
            return
        
        total_norms = [stats['total_norm'] for stats in self.gradient_history]
        
        plt.figure(figsize=(10, 6))
        plt.plot(total_norms, alpha=0.7)
        plt.title('Gradient Norm Evolution')
        plt.xlabel('Training Step')
        plt.ylabel('Gradient Norm')
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path)
        plt.show()


class LearningRateScheduler:
    """Advanced learning rate scheduling for DINO"""
    
    @staticmethod
    def cosine_with_restarts(
        step: int,
        total_steps: int,
        warmup_steps: int,
        num_restarts: int = 0,
        restart_decay: float = 1.0
    ) -> float:
        """Cosine annealing with warm restarts"""
        
        if step < warmup_steps:
            return step / warmup_steps
        
        step = step - warmup_steps
        total_steps = total_steps - warmup_steps
        
        if num_restarts > 0:
            # Calculate restart intervals
            restart_interval = total_steps // (num_restarts + 1)
            restart_number = step // restart_interval
            step_in_restart = step % restart_interval
            
            # Apply decay to learning rate for each restart
            decay_factor = restart_decay ** restart_number
            
            # Cosine schedule within restart
            cosine_factor = 0.5 * (1 + math.cos(math.pi * step_in_restart / restart_interval))
            
            return decay_factor * cosine_factor
        else:
            # Standard cosine decay
            return 0.5 * (1 + math.cos(math.pi * step / total_steps))
    
    @staticmethod
    def polynomial_decay(
        step: int,
        total_steps: int,
        warmup_steps: int,
        power: float = 1.0,
        end_lr_ratio: float = 0.0
    ) -> float:
        """Polynomial learning rate decay"""
        
        if step < warmup_steps:
            return step / warmup_steps
        
        step = step - warmup_steps
        total_steps = total_steps - warmup_steps
        
        decay_factor = (1 - step / total_steps) ** power
        return end_lr_ratio + (1 - end_lr_ratio) * decay_factor


class MemoryProfiler:
    """Profile memory usage during training"""
    
    def __init__(self):
        self.memory_history = []
    
    def log_memory(self):
        """Log current memory usage"""
        if torch.cuda.is_available():
            memory_stats = {
                'allocated': torch.cuda.memory_allocated() / 1024**3,  # GB
                'cached': torch.cuda.memory_reserved() / 1024**3,  # GB
                'max_allocated': torch.cuda.max_memory_allocated() / 1024**3,  # GB
            }
        else:
            memory_stats = {'allocated': 0, 'cached': 0, 'max_allocated': 0}
        
        self.memory_history.append(memory_stats)
        return memory_stats
    
    def plot_memory_usage(self, save_path: Optional[str] = None):
        """Plot memory usage evolution"""
        if not self.memory_history:
            return
        
        allocated = [stats['allocated'] for stats in self.memory_history]
        cached = [stats['cached'] for stats in self.memory_history]
        
        plt.figure(figsize=(10, 6))
        plt.plot(allocated, label='Allocated', alpha=0.7)
        plt.plot(cached, label='Cached', alpha=0.7)
        plt.title('GPU Memory Usage')
        plt.xlabel('Training Step')
        plt.ylabel('Memory (GB)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path)
        plt.show()


def create_training_pipeline(
    model,
    loss_fn,
    train_dataloader,
    val_dataloader=None,
    config: Optional[Dict] = None
) -> DINOTrainingLoop:
    """
    Factory function to create complete training pipeline
    
    Args:
        model: Student-teacher DINO model
        loss_fn: Complete DINO loss function
        train_dataloader: Training data loader
        val_dataloader: Validation data loader (optional)
        config: Training configuration dictionary
    
    Returns:
        Configured DINOTrainingLoop instance
    """
    
    # Default configuration
    default_config = {
        'base_lr': 0.0005,
        'weight_decay': 0.04,
        'batch_size': 64,
        'epochs': 100,
        'warmup_epochs': 10,
        'lr_schedule': 'cosine',
        'clip_grad_norm': 3.0,
        'mixed_precision': True,
        'use_wandb': False,
        'experiment_name': 'dino_training',
        'save_dir': './checkpoints'
    }
    
    # Merge configurations
    if config:
        default_config.update(config)
    
    # Create training loop
    trainer = DINOTrainingLoop(
        model=model,
        loss_fn=loss_fn,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        **default_config
    )
    
    return trainer


def test_training_loop():
    """Test the training loop with synthetic data"""
    
    # Create synthetic model and data
    from torch.utils.data import TensorDataset, DataLoader
    
    # Synthetic dataset
    num_samples = 1000
    global_crops = torch.randn(num_samples, 2, 3, 224, 224)  # 2 global crops per sample
    local_crops = torch.randn(num_samples, 8, 3, 96, 96)    # 8 local crops per sample
    
    # Custom collate function for testing
    def test_collate_fn(batch):
        global_batch = torch.stack([item[0] for item in batch]).view(-1, 3, 224, 224)
        local_batch = torch.stack([item[1] for item in batch]).view(-1, 3, 96, 96)
        return {
            'global_crops': global_batch,
            'local_crops': local_batch,
            'batch_size': len(batch)
        }
    
    # Create datasets
    train_dataset = TensorDataset(global_crops, local_crops)
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=4, 
        shuffle=True, 
        collate_fn=test_collate_fn
    )
    
    # Create dummy model and loss (would use real implementations)
    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = nn.Linear(3*224*224, 1000)
            self.projection = nn.Linear(1000, 65536)
        
        def forward_student(self, x):
            x = x.view(x.size(0), -1)
            return self.projection(self.backbone(x))
        
        def forward_teacher(self, x):
            with torch.no_grad():
                x = x.view(x.size(0), -1)
                return self.projection(self.backbone(x))
        
        def update_teacher(self, epoch):
            pass
        
        def get_student_parameters(self):
            return self.parameters()
        
        def to(self, device):
            return super().to(device)
    
    class DummyLoss(nn.Module):
        def forward(self, student_global_crops, student_local_crops, teacher_global_crops, epoch, update_center=True):
            loss = torch.mean(student_global_crops) + torch.mean(student_local_crops) + torch.mean(teacher_global_crops)
            loss_dict = {
                'total_loss': loss.item(),
                'global_loss': loss.item() * 0.7,
                'local_loss': loss.item() * 0.3,
                'center_norm': 1.0,
                'teacher_temperature': 0.04
            }
            return loss, loss_dict
    
    # Create training components
    model = DummyModel()
    loss_fn = DummyLoss()
    
    # Test training loop
    config = {
        'epochs': 2,
        'log_interval': 10,
        'save_interval': 50,
        'use_wandb': False,
        'mixed_precision': False  # Disable for testing
    }
    
    trainer = create_training_pipeline(
        model=model,
        loss_fn=loss_fn,
        train_dataloader=train_dataloader,
        config=config
    )
    
    # Run training
    summary = trainer.train()
    
    print("Training completed!")
    print(f"Summary: {summary}")

if __name__ == "__main__":
    test_training_loop()
```

## 🧪 Hands-on Exercise: Build Your Training Loop

### Exercise 1: Custom Learning Rate Schedule

Implement a custom learning rate schedule:

```python
# exercise1.py
import math

def custom_lr_schedule(step, total_steps, warmup_steps, **kwargs):
    """
    Implement a custom learning rate schedule that:
    1. Has linear warmup for first 10% of training
    2. Uses cosine decay for next 70% of training  
    3. Has constant low learning rate for final 20%
    """
    # TODO: Implement the three-phase schedule
    pass

def test_lr_schedule():
    """Test your learning rate schedule"""
    total_steps = 10000
    warmup_steps = 1000
    
    steps = range(total_steps)
    lr_values = [custom_lr_schedule(step, total_steps, warmup_steps) for step in steps]
    
    # TODO: Plot the learning rate schedule
    import matplotlib.pyplot as plt
    plt.plot(steps, lr_values)
    plt.title('Custom Learning Rate Schedule')
    plt.xlabel('Step')
    plt.ylabel('Learning Rate Multiplier')
    plt.show()

test_lr_schedule()
```

### Exercise 2: Training Diagnostics

Build diagnostic tools for training monitoring:

```python
# exercise2.py
class TrainingDiagnostics:
    def __init__(self):
        self.loss_history = []
        self.gradient_history = []
        
    def diagnose_training_health(self, recent_losses, recent_gradients):
        """
        Diagnose potential training issues:
        1. Loss explosion/plateau
        2. Gradient explosion/vanishing
        3. Oscillating training
        4. Mode collapse indicators
        """
        diagnostics = {}
        
        # TODO: Implement diagnostic checks
        # Check for loss explosion
        # Check for gradient issues
        # Check for training instability
        # Return structured diagnostic report
        
        return diagnostics
    
    def suggest_hyperparameter_adjustments(self, diagnostics):
        """Suggest hyperparameter changes based on diagnostics"""
        suggestions = []
        
        # TODO: Implement suggestion logic based on common issues
        
        return suggestions

# Test your diagnostics
diagnostics = TrainingDiagnostics()
```

### Exercise 3: Resumable Training

Implement robust checkpoint and resume functionality:

```python
# exercise3.py
class RobustCheckpointing:
    def __init__(self, save_dir):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        
    def save_checkpoint(self, trainer, extra_data=None):
        """
        Save comprehensive checkpoint including:
        1. Model state
        2. Optimizer state  
        3. Training progress
        4. Random number generator states
        5. Custom training data
        """
        # TODO: Implement comprehensive checkpointing
        pass
    
    def find_latest_checkpoint(self):
        """Find the most recent checkpoint"""
        # TODO: Implement logic to find latest checkpoint
        pass
    
    def verify_checkpoint(self, checkpoint_path):
        """Verify checkpoint integrity"""
        # TODO: Implement checkpoint verification
        pass
    
    def resume_training(self, trainer, checkpoint_path=None):
        """Resume training from checkpoint"""
        # TODO: Implement resuming logic
        pass

# Test your checkpointing system
```

## 🔍 Key Insights

### Training Loop Best Practices
1. **Gradient Clipping**: Essential for DINO stability, typically clip to norm 3.0
2. **Mixed Precision**: Reduces memory usage and improves speed
3. **Learning Rate Warmup**: Critical for stable training start
4. **Teacher Freezing**: Freeze last layer for first few epochs

### Optimization Strategies
1. **AdamW Optimizer**: Better than Adam for DINO due to weight decay handling
2. **Cosine LR Schedule**: Smooth decay works well for self-supervised learning
3. **Gradient Accumulation**: Enables larger effective batch sizes
4. **Parameter Groups**: Different weight decay for different parameter types

### Monitoring and Debugging
1. **Loss Components**: Track global and local losses separately
2. **Temperature Evolution**: Monitor teacher temperature scheduling
3. **Center Norm**: Watch for center explosion indicating issues
4. **Agreement Metrics**: Track student-teacher agreement

## 📝 Summary

In this lesson, you learned:

✅ **Complete Training Loop**: Professional implementation with all optimizations

✅ **Advanced Optimization**: Gradient clipping, learning rate scheduling, mixed precision

✅ **Robust Checkpointing**: Comprehensive state saving and resumable training

✅ **Training Monitoring**: Extensive logging, validation, and diagnostic tools

✅ **Production Features**: Early stopping, memory profiling, experiment tracking

### Module 4 Complete!
You now have a complete DINO implementation with:
- Centering mechanism for collapse prevention
- Temperature sharpening for controlled learning
- Complete loss function with all components
- Production-ready training loop with all optimizations

## 🔗 Additional Resources

- [PyTorch Training Loop Best Practices](https://pytorch.org/tutorials/beginner/pytorch_with_examples.html)
- [Mixed Precision Training](https://pytorch.org/docs/stable/amp.html)
- [Weights & Biases for Experiment Tracking](https://wandb.ai/)

---

**Next**: [Module 5, Lesson 1: Complete Training Implementation](module5_lesson1_complete_training.md)
