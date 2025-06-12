# Module 5, Lesson 1: Complete Training Implementation

## 🎯 Learning Objectives
By the end of this lesson, you will:
- Integrate all DINO components into a complete training system
- Implement end-to-end training on CIFAR-10 and ImageNet
- Master the full student-teacher update cycle
- Build production-ready training scripts with proper configuration management

## 📚 Theoretical Background

### Complete DINO Training Pipeline

The **complete DINO training system** integrates all components we've built:

1. **Data Pipeline**: Multi-crop augmentation → Batch loading
2. **Forward Pass**: Student-teacher networks → Feature extraction  
3. **Loss Computation**: DINO loss with centering and temperature
4. **Optimization**: Gradient clipping → Parameter updates
5. **Teacher Update**: EMA weight synchronization
6. **Monitoring**: Loss tracking and validation metrics

### Training Dynamics

**Student-Teacher Update Cycle**:
```
For each batch:
1. Generate multi-crop views: x_global, x_local
2. Student forward: f_s(x_global), f_s(x_local)  
3. Teacher forward: f_t(x_global), f_t(x_local)
4. Compute DINO loss: L(P_s, P_t)
5. Backward pass: ∇L w.r.t. student parameters
6. Update student: θ_s ← θ_s - lr * ∇L
7. Update teacher: θ_t ← τ * θ_t + (1-τ) * θ_s
```

**Key Training Considerations**:
- **Batch Size**: Larger batches (256-1024) improve stability
- **Learning Rate**: Scale with batch size: `lr = base_lr * batch_size / 256`
- **Warmup**: Linear warmup prevents early training instability
- **EMA Momentum**: Start at 0.996, increase to 0.999 over training

## 🛠️ Implementation

### Step 1: Complete Training Configuration

```python
# config/dino_config.yaml
dino_training:
  # Dataset Configuration
  dataset:
    name: "cifar10"  # or "imagenet", "custom"
    data_path: "./data"
    num_classes: 10
    
  # Model Configuration  
  model:
    backbone: "resnet50"  # or "vit_small", "vit_base"
    projection_dim: 65536
    projection_hidden_dim: 2048
    projection_layers: 3
    
  # Augmentation Configuration
  augmentation:
    global_crop_size: 224
    local_crop_size: 96
    global_crop_scale: [0.4, 1.0]
    local_crop_scale: [0.05, 0.4]
    n_local_crops: 8
    
  # Training Configuration
  training:
    epochs: 100
    batch_size: 64
    num_workers: 4
    
    # Optimization
    optimizer: "adamw"
    base_lr: 0.0005
    weight_decay: 0.04
    gradient_clip: 3.0
    
    # Scheduling
    warmup_epochs: 10
    lr_schedule: "cosine"
    
    # DINO Specific
    student_temp: 0.1
    teacher_temp_start: 0.04
    teacher_temp_end: 0.07
    teacher_temp_warmup_epochs: 30
    ema_momentum_start: 0.996
    ema_momentum_end: 0.999
    center_momentum: 0.9
    
  # Checkpointing
  checkpoint:
    save_freq: 10
    checkpoint_dir: "./checkpoints"
    resume_from: null
    
  # Logging
  logging:
    use_wandb: true
    project_name: "dino-training"
    log_freq: 100
    save_visualizations: true
```

### Step 2: Complete Training Script

```python
# train_dino.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import yaml
import argparse
import os
import time
from pathlib import Path

# Import our implemented components
from models.student_teacher import StudentTeacherWrapper
from models.backbones import create_backbone
from data.multicrop_dataset import MultiCropDataset, MultiCropCollator
from loss.dino_loss import CompleteDINOLoss
from training.training_loop import DINOTrainingLoop
from utils.config import load_config
from utils.logger import setup_logger
from utils.metrics import DINOMetrics

class CompleteDINOTrainer:
    """Complete DINO training system integrating all components."""
    
    def __init__(self, config_path: str):
        """Initialize trainer with configuration."""
        self.config = load_config(config_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger = setup_logger("DINO Training")
        
        # Initialize components
        self._setup_data()
        self._setup_model()
        self._setup_loss()
        self._setup_optimizer()
        self._setup_scheduler()
        self._setup_metrics()
        
        self.logger.info(f"DINO trainer initialized with config: {config_path}")
        self.logger.info(f"Training on device: {self.device}")
        
    def _setup_data(self):
        """Setup data loading pipeline."""
        dataset_config = self.config['dataset']
        aug_config = self.config['augmentation']
        train_config = self.config['training']
        
        # Create dataset with multi-crop augmentation
        self.train_dataset = MultiCropDataset(
            data_path=dataset_config['data_path'],
            dataset_name=dataset_config['name'],
            global_crop_size=aug_config['global_crop_size'],
            local_crop_size=aug_config['local_crop_size'],
            global_crop_scale=aug_config['global_crop_scale'],
            local_crop_scale=aug_config['local_crop_scale'],
            n_local_crops=aug_config['n_local_crops'],
            split='train'
        )
        
        # Create dataloader with custom collator
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=train_config['batch_size'],
            shuffle=True,
            num_workers=train_config['num_workers'],
            pin_memory=True,
            drop_last=True,
            collate_fn=MultiCropCollator()
        )
        
        # Validation dataset (for monitoring)
        self.val_dataset = MultiCropDataset(
            data_path=dataset_config['data_path'],
            dataset_name=dataset_config['name'],
            global_crop_size=aug_config['global_crop_size'],
            local_crop_size=aug_config['local_crop_size'],
            n_local_crops=2,  # Fewer crops for validation
            split='val'
        )
        
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=train_config['batch_size'],
            shuffle=False,
            num_workers=train_config['num_workers'],
            pin_memory=True,
            collate_fn=MultiCropCollator()
        )
        
        self.logger.info(f"Dataset: {dataset_config['name']}")
        self.logger.info(f"Training samples: {len(self.train_dataset)}")
        self.logger.info(f"Validation samples: {len(self.val_dataset)}")
        
    def _setup_model(self):
        """Setup student-teacher model."""
        model_config = self.config['model']
        
        # Create backbone
        backbone = create_backbone(
            arch=model_config['backbone'],
            pretrained=False  # Train from scratch
        )
        
        # Create student-teacher wrapper
        self.model = StudentTeacherWrapper(
            backbone=backbone,
            projection_dim=model_config['projection_dim'],
            projection_hidden_dim=model_config['projection_hidden_dim'],
            projection_layers=model_config['projection_layers'],
            ema_momentum_start=self.config['training']['ema_momentum_start'],
            ema_momentum_end=self.config['training']['ema_momentum_end']
        ).to(self.device)
        
        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        self.logger.info(f"Model: {model_config['backbone']}")
        self.logger.info(f"Total parameters: {total_params:,}")
        self.logger.info(f"Trainable parameters: {trainable_params:,}")
        
    def _setup_loss(self):
        """Setup DINO loss function."""
        train_config = self.config['training']
        
        self.loss_fn = CompleteDINOLoss(
            student_temp=train_config['student_temp'],
            teacher_temp_start=train_config['teacher_temp_start'],
            teacher_temp_end=train_config['teacher_temp_end'],
            teacher_temp_warmup_epochs=train_config['teacher_temp_warmup_epochs'],
            center_momentum=train_config['center_momentum'],
            output_dim=self.config['model']['projection_dim']
        ).to(self.device)
        
        self.logger.info("DINO loss function initialized")
        
    def _setup_optimizer(self):
        """Setup optimizer."""
        train_config = self.config['training']
        
        # Scale learning rate with batch size
        scaled_lr = train_config['base_lr'] * train_config['batch_size'] / 256
        
        if train_config['optimizer'] == 'adamw':
            self.optimizer = optim.AdamW(
                self.model.student.parameters(),  # Only optimize student
                lr=scaled_lr,
                weight_decay=train_config['weight_decay']
            )
        elif train_config['optimizer'] == 'sgd':
            self.optimizer = optim.SGD(
                self.model.student.parameters(),
                lr=scaled_lr,
                momentum=0.9,
                weight_decay=train_config['weight_decay']
            )
        else:
            raise ValueError(f"Unknown optimizer: {train_config['optimizer']}")
            
        self.logger.info(f"Optimizer: {train_config['optimizer']}")
        self.logger.info(f"Learning rate: {scaled_lr:.6f}")
        
    def _setup_scheduler(self):
        """Setup learning rate scheduler."""
        train_config = self.config['training']
        
        if train_config['lr_schedule'] == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=train_config['epochs'],
                eta_min=0
            )
        elif train_config['lr_schedule'] == 'step':
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=30,
                gamma=0.1
            )
        else:
            self.scheduler = None
            
        self.logger.info(f"LR Schedule: {train_config['lr_schedule']}")
        
    def _setup_metrics(self):
        """Setup metrics tracking."""
        self.metrics = DINOMetrics()
        
    def train_epoch(self, epoch: int) -> dict:
        """Train one epoch."""
        self.model.train()
        self.loss_fn.train()
        
        epoch_metrics = {
            'loss': 0.0,
            'lr': 0.0,
            'teacher_temp': 0.0,
            'ema_momentum': 0.0
        }
        
        num_batches = len(self.train_loader)
        
        for batch_idx, batch in enumerate(self.train_loader):
            # Move data to device
            crops = [crop.to(self.device) for crop in batch['crops']]
            
            # Update learning rate for warmup
            if epoch < self.config['training']['warmup_epochs']:
                warmup_lr = self._get_warmup_lr(epoch, batch_idx, num_batches)
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = warmup_lr
                    
            # Update teacher temperature
            self.loss_fn.update_teacher_temp(epoch)
            
            # Update EMA momentum
            self.model.update_momentum_schedule(epoch, self.config['training']['epochs'])
            
            # Forward pass
            student_outputs = []
            teacher_outputs = []
            
            for crop in crops:
                student_out = self.model.student(crop)
                with torch.no_grad():
                    teacher_out = self.model.teacher(crop)
                student_outputs.append(student_out)
                teacher_outputs.append(teacher_out)
                
            # Compute loss
            loss = self.loss_fn(student_outputs, teacher_outputs)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.student.parameters(),
                self.config['training']['gradient_clip']
            )
            
            # Update student
            self.optimizer.step()
            
            # Update teacher (EMA)
            self.model.update_teacher()
            
            # Track metrics
            epoch_metrics['loss'] += loss.item()
            epoch_metrics['lr'] = self.optimizer.param_groups[0]['lr']
            epoch_metrics['teacher_temp'] = self.loss_fn.teacher_temp
            epoch_metrics['ema_momentum'] = self.model.momentum
            
            # Log batch metrics
            if batch_idx % self.config['logging']['log_freq'] == 0:
                self.logger.info(
                    f"Epoch {epoch:3d} [{batch_idx:4d}/{num_batches:4d}] "
                    f"Loss: {loss.item():.4f} "
                    f"LR: {epoch_metrics['lr']:.6f} "
                    f"Teacher Temp: {epoch_metrics['teacher_temp']:.4f}"
                )
                
        # Average metrics over epoch
        for key in epoch_metrics:
            if key != 'lr':  # lr is already current value
                epoch_metrics[key] /= num_batches
                
        return epoch_metrics
        
    def validate_epoch(self, epoch: int) -> dict:
        """Validate one epoch."""
        self.model.eval()
        self.loss_fn.eval()
        
        val_metrics = {'val_loss': 0.0}
        num_batches = len(self.val_loader)
        
        with torch.no_grad():
            for batch in self.val_loader:
                crops = [crop.to(self.device) for crop in batch['crops']]
                
                # Forward pass (fewer crops for efficiency)
                student_outputs = []
                teacher_outputs = []
                
                for crop in crops[:4]:  # Use only first 4 crops
                    student_out = self.model.student(crop)
                    teacher_out = self.model.teacher(crop)
                    student_outputs.append(student_out)
                    teacher_outputs.append(teacher_out)
                    
                # Compute loss
                loss = self.loss_fn(student_outputs, teacher_outputs)
                val_metrics['val_loss'] += loss.item()
                
        val_metrics['val_loss'] /= num_batches
        return val_metrics
        
    def _get_warmup_lr(self, epoch: int, batch_idx: int, num_batches: int) -> float:
        """Compute learning rate for warmup."""
        warmup_epochs = self.config['training']['warmup_epochs']
        base_lr = self.optimizer.param_groups[0]['lr']
        
        # Linear warmup
        warmup_progress = (epoch + batch_idx / num_batches) / warmup_epochs
        return base_lr * warmup_progress
        
    def save_checkpoint(self, epoch: int, metrics: dict, checkpoint_dir: str):
        """Save training checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'loss_state_dict': self.loss_fn.state_dict(),
            'metrics': metrics,
            'config': self.config
        }
        
        checkpoint_path = os.path.join(checkpoint_dir, f"dino_epoch_{epoch:03d}.pth")
        torch.save(checkpoint, checkpoint_path)
        
        # Save latest checkpoint
        latest_path = os.path.join(checkpoint_dir, "dino_latest.pth")
        torch.save(checkpoint, latest_path)
        
        self.logger.info(f"Checkpoint saved: {checkpoint_path}")
        
    def load_checkpoint(self, checkpoint_path: str) -> int:
        """Load training checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if self.scheduler and checkpoint['scheduler_state_dict']:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.loss_fn.load_state_dict(checkpoint['loss_state_dict'])
        
        start_epoch = checkpoint['epoch'] + 1
        self.logger.info(f"Checkpoint loaded: {checkpoint_path}")
        self.logger.info(f"Resuming from epoch {start_epoch}")
        
        return start_epoch
        
    def train(self):
        """Complete training loop."""
        train_config = self.config['training']
        checkpoint_config = self.config['checkpoint']
        
        # Setup checkpoint directory
        checkpoint_dir = checkpoint_config['checkpoint_dir']
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Resume from checkpoint if specified
        start_epoch = 0
        if checkpoint_config['resume_from']:
            start_epoch = self.load_checkpoint(checkpoint_config['resume_from'])
            
        # Training loop
        self.logger.info("Starting DINO training...")
        start_time = time.time()
        
        for epoch in range(start_epoch, train_config['epochs']):
            epoch_start = time.time()
            
            # Train one epoch
            train_metrics = self.train_epoch(epoch)
            
            # Validate
            val_metrics = self.validate_epoch(epoch)
            
            # Combine metrics
            all_metrics = {**train_metrics, **val_metrics}
            
            # Learning rate step (after warmup)
            if epoch >= train_config['warmup_epochs'] and self.scheduler:
                self.scheduler.step()
                
            # Log epoch results
            epoch_time = time.time() - epoch_start
            self.logger.info(
                f"Epoch {epoch:3d} completed in {epoch_time:.1f}s - "
                f"Train Loss: {train_metrics['loss']:.4f} "
                f"Val Loss: {val_metrics['val_loss']:.4f}"
            )
            
            # Save checkpoint
            if (epoch + 1) % checkpoint_config['save_freq'] == 0:
                self.save_checkpoint(epoch, all_metrics, checkpoint_dir)
                
        # Training complete
        total_time = time.time() - start_time
        self.logger.info(f"Training completed in {total_time/3600:.1f} hours")
        
        # Save final checkpoint
        self.save_checkpoint(train_config['epochs'] - 1, all_metrics, checkpoint_dir)


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train DINO model')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to config file')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (cuda/cpu/auto)')
    
    args = parser.parse_args()
    
    # Override device if specified
    if args.device != 'auto':
        import os
        os.environ['CUDA_VISIBLE_DEVICES'] = args.device
        
    # Create and run trainer
    trainer = CompleteDINOTrainer(args.config)
    trainer.train()


if __name__ == "__main__":
    main()
```

### Step 3: Supporting Utilities

```python
# utils/config.py
import yaml
from typing import Dict, Any

def load_config(config_path: str) -> Dict[str, Any]:
    """Load YAML configuration file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config['dino_training']

def save_config(config: Dict[str, Any], save_path: str):
    """Save configuration to YAML file."""
    with open(save_path, 'w') as f:
        yaml.dump({'dino_training': config}, f, default_flow_style=False)
```

```python
# utils/logger.py
import logging
import sys
from pathlib import Path

def setup_logger(name: str, log_file: str = None) -> logging.Logger:
    """Setup logger with console and file output."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    
    # File handler (optional)
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        logger.addHandler(file_handler)
    
    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_handler.setFormatter(formatter)
    if log_file:
        file_handler.setFormatter(formatter)
    
    logger.addHandler(console_handler)
    return logger
```

```python
# utils/metrics.py
import torch
import numpy as np
from typing import Dict, List

class DINOMetrics:
    """Metrics tracking for DINO training."""
    
    def __init__(self):
        self.reset()
        
    def reset(self):
        """Reset all metrics."""
        self.metrics = {
            'loss': [],
            'lr': [],
            'teacher_temp': [],
            'ema_momentum': []
        }
        
    def update(self, **kwargs):
        """Update metrics with new values."""
        for key, value in kwargs.items():
            if key in self.metrics:
                self.metrics[key].append(value)
                
    def get_averages(self, last_n: int = None) -> Dict[str, float]:
        """Get average values over last n steps."""
        averages = {}
        for key, values in self.metrics.items():
            if values:
                recent_values = values[-last_n:] if last_n else values
                averages[f'avg_{key}'] = np.mean(recent_values)
        return averages
        
    def get_latest(self) -> Dict[str, float]:
        """Get latest metric values."""
        latest = {}
        for key, values in self.metrics.items():
            if values:
                latest[key] = values[-1]
        return latest
```

## 🧪 Practical Exercises

### Exercise 1: CIFAR-10 Training Setup
```python
# Create configuration for CIFAR-10 training
config = {
    'dataset': {'name': 'cifar10', 'data_path': './data'},
    'model': {'backbone': 'resnet18', 'projection_dim': 1024},
    'training': {'epochs': 50, 'batch_size': 32}
}

# Run short training experiment
trainer = CompleteDINOTrainer('config/cifar10_config.yaml')
trainer.train()
```

### Exercise 2: Training Analysis
```python
# Analyze training dynamics
def analyze_training_run(checkpoint_dir: str):
    """Analyze metrics from training run."""
    # Load checkpoints and plot metrics
    epochs = []
    losses = []
    
    for checkpoint_file in sorted(Path(checkpoint_dir).glob("dino_epoch_*.pth")):
        checkpoint = torch.load(checkpoint_file)
        epochs.append(checkpoint['epoch'])
        losses.append(checkpoint['metrics']['loss'])
        
    # Plot training curve
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, losses)
    plt.title('DINO Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()

# Usage
analyze_training_run('./checkpoints')
```

### Exercise 3: Multi-Dataset Training
```python
# Train on multiple datasets
datasets = ['cifar10', 'cifar100', 'stl10']

for dataset in datasets:
    print(f"Training DINO on {dataset}")
    config_path = f'config/{dataset}_config.yaml'
    trainer = CompleteDINOTrainer(config_path)
    trainer.train()
```

## 🚀 Advanced Concepts

### Mixed Precision Training
```python
# Enable automatic mixed precision
from torch.cuda.amp import GradScaler, autocast

class DINOTrainerAMP(CompleteDINOTrainer):
    """DINO trainer with automatic mixed precision."""
    
    def __init__(self, config_path: str):
        super().__init__(config_path)
        self.scaler = GradScaler()
        
    def train_epoch(self, epoch: int) -> dict:
        """Train epoch with mixed precision."""
        # ... existing code ...
        
        for batch_idx, batch in enumerate(self.train_loader):
            crops = [crop.to(self.device) for crop in batch['crops']]
            
            # Forward pass with autocast
            with autocast():
                student_outputs = []
                teacher_outputs = []
                
                for crop in crops:
                    student_out = self.model.student(crop)
                    with torch.no_grad():
                        teacher_out = self.model.teacher(crop)
                    student_outputs.append(student_out)
                    teacher_outputs.append(teacher_out)
                    
                loss = self.loss_fn(student_outputs, teacher_outputs)
                
            # Backward pass with scaling
            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.student.parameters(),
                self.config['training']['gradient_clip']
            )
            
            # Update with scaling
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            # Update teacher
            self.model.update_teacher()
```

### Distributed Training
```python
# Multi-GPU training setup
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

class DistributedDINOTrainer(CompleteDINOTrainer):
    """Distributed DINO trainer for multi-GPU training."""
    
    def __init__(self, config_path: str, rank: int, world_size: int):
        self.rank = rank
        self.world_size = world_size
        
        # Initialize distributed training
        dist.init_process_group(
            backend='nccl',
            rank=rank,
            world_size=world_size
        )
        
        super().__init__(config_path)
        
        # Wrap model in DDP
        self.model = DDP(self.model, device_ids=[rank])
        
    def _setup_data(self):
        """Setup distributed data loading."""
        from torch.utils.data.distributed import DistributedSampler
        
        # ... existing dataset creation ...
        
        # Add distributed sampler
        train_sampler = DistributedSampler(
            self.train_dataset,
            num_replicas=self.world_size,
            rank=self.rank
        )
        
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config['training']['batch_size'],
            sampler=train_sampler,
            num_workers=self.config['training']['num_workers'],
            pin_memory=True,
            collate_fn=MultiCropCollator()
        )
```

## 🎯 Key Takeaways

1. **Complete Integration**: Successfully combine all DINO components into working system
2. **Production Ready**: Robust configuration, logging, and checkpoint management
3. **Scalable Design**: Support for different datasets, models, and training strategies
4. **Performance Optimized**: Mixed precision, gradient clipping, and efficient data loading
5. **Research Friendly**: Configurable hyperparameters and comprehensive monitoring

## 🔍 What's Next?

In **Lesson 5.2**, we'll implement comprehensive monitoring and logging systems to track training progress, visualize learned features, and debug training issues in real-time.

The complete training implementation you've built is production-ready and can handle training DINO from scratch on any dataset. This is the same training pipeline used to achieve state-of-the-art results in the original paper!
