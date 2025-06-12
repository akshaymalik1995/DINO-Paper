# Module 5, Lesson 3: Checkpoint Management

## 🎯 Learning Objectives
By the end of this lesson, you will:
- Implement robust checkpoint management for long DINO training runs
- Build automatic recovery systems for interrupted training
- Create experiment versioning and reproducibility systems
- Design efficient storage and backup strategies for large models

## 📚 Theoretical Background

### Why Checkpoint Management Matters

**Training Challenges**:
- **Long Training Times**: DINO requires 100+ epochs (days/weeks of training)
- **Hardware Failures**: GPU crashes, power outages, network issues
- **Experiment Tracking**: Multiple runs with different hyperparameters  
- **Model Versioning**: Track different model versions and configurations
- **Storage Costs**: Large models (ViT-Large can be >1GB per checkpoint)

**Checkpoint Components**:
```python
checkpoint = {
    'epoch': current_epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'scheduler_state_dict': scheduler.state_dict(),
    'loss_state_dict': loss_fn.state_dict(),
    'metrics': training_metrics,
    'config': training_config,
    'random_states': random_number_states,
    'timestamp': creation_time,
    'git_commit': code_version
}
```

### Checkpoint Management Strategies

**1. Periodic Checkpointing**:
- Save every N epochs (e.g., every 10 epochs)
- Save based on improvement (new best validation loss)
- Save at specific milestones (25%, 50%, 75%, 100% of training)

**2. Checkpoint Rotation**:
- Keep only last K checkpoints to save storage
- Always preserve best checkpoint
- Keep milestone checkpoints permanently

**3. Incremental Checkpointing**:
- Only save changed parameters (for very large models)
- Delta compression for storage efficiency

**4. Distributed Checkpointing**:
- Handle multi-GPU training states
- Coordinate checkpoint saving across ranks

## 🛠️ Implementation

### Step 1: Advanced Checkpoint Manager

```python
# checkpoint/checkpoint_manager.py
import torch
import os
import json
import shutil
import time
import hashlib
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime
import logging

@dataclass
class CheckpointMetadata:
    """Metadata for a training checkpoint."""
    epoch: int
    step: int
    timestamp: str
    loss: float
    metrics: Dict[str, float]
    config_hash: str
    git_commit: Optional[str]
    file_size: int
    is_best: bool = False
    is_milestone: bool = False

class DINOCheckpointManager:
    """Comprehensive checkpoint management for DINO training."""
    
    def __init__(self, 
                 checkpoint_dir: str,
                 max_keep: int = 5,
                 save_best: bool = True,
                 save_milestones: bool = True,
                 milestone_epochs: List[int] = None,
                 compression: bool = True):
        """
        Args:
            checkpoint_dir: Directory to save checkpoints
            max_keep: Maximum number of regular checkpoints to keep
            save_best: Whether to keep best checkpoint separately
            save_milestones: Whether to save milestone checkpoints
            milestone_epochs: Specific epochs to save as milestones
            compression: Whether to compress checkpoints
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.max_keep = max_keep
        self.save_best = save_best
        self.save_milestones = save_milestones
        self.milestone_epochs = milestone_epochs or [25, 50, 75, 100]
        self.compression = compression
        
        # Track checkpoints
        self.checkpoints: List[CheckpointMetadata] = []
        self.best_checkpoint: Optional[CheckpointMetadata] = None
        self.best_loss = float('inf')
        
        # Setup logging
        self.logger = logging.getLogger('CheckpointManager')
        
        # Load existing checkpoint metadata
        self._load_checkpoint_registry()
        
        self.logger.info(f"Checkpoint manager initialized: {checkpoint_dir}")
        
    def _load_checkpoint_registry(self):
        """Load existing checkpoint metadata from disk."""
        registry_file = self.checkpoint_dir / "checkpoint_registry.json"
        
        if registry_file.exists():
            with open(registry_file, 'r') as f:
                registry_data = json.load(f)
                
            self.checkpoints = [
                CheckpointMetadata(**cp_data) 
                for cp_data in registry_data.get('checkpoints', [])
            ]
            
            best_data = registry_data.get('best_checkpoint')
            if best_data:
                self.best_checkpoint = CheckpointMetadata(**best_data)
                self.best_loss = self.best_checkpoint.loss
                
        self.logger.info(f"Loaded {len(self.checkpoints)} existing checkpoints")
        
    def _save_checkpoint_registry(self):
        """Save checkpoint metadata to disk."""
        registry_data = {
            'checkpoints': [cp.__dict__ for cp in self.checkpoints],
            'best_checkpoint': self.best_checkpoint.__dict__ if self.best_checkpoint else None,
            'last_updated': datetime.now().isoformat()
        }
        
        registry_file = self.checkpoint_dir / "checkpoint_registry.json"
        with open(registry_file, 'w') as f:
            json.dump(registry_data, f, indent=2)
            
    def _get_git_commit(self) -> Optional[str]:
        """Get current git commit hash."""
        try:
            result = subprocess.run(
                ['git', 'rev-parse', 'HEAD'],
                capture_output=True,
                text=True,
                timeout=5
            )
            return result.stdout.strip() if result.returncode == 0 else None
        except:
            return None
            
    def _compute_config_hash(self, config: Dict) -> str:
        """Compute hash of configuration for reproducibility."""
        config_str = json.dumps(config, sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()[:8]
        
    def _is_milestone_epoch(self, epoch: int, total_epochs: int) -> bool:
        """Check if epoch is a milestone."""
        if not self.save_milestones:
            return False
            
        # Check absolute milestones
        if epoch in self.milestone_epochs:
            return True
            
        # Check percentage milestones
        if total_epochs > 0:
            percentage = (epoch / total_epochs) * 100
            milestone_percentages = [25, 50, 75, 100]
            for pct in milestone_percentages:
                if abs(percentage - pct) < 1.0:  # Within 1% tolerance
                    return True
                    
        return False
        
    def save_checkpoint(self,
                       epoch: int,
                       step: int,
                       model: torch.nn.Module,
                       optimizer: torch.optim.Optimizer,
                       scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
                       loss_fn: torch.nn.Module,
                       metrics: Dict[str, float],
                       config: Dict[str, Any],
                       total_epochs: int = None,
                       force_save: bool = False) -> str:
        """Save a training checkpoint."""
        
        # Determine checkpoint type
        is_milestone = self._is_milestone_epoch(epoch, total_epochs)
        is_best = metrics.get('loss', float('inf')) < self.best_loss
        
        # Create checkpoint data
        checkpoint_data = {
            'epoch': epoch,
            'step': step,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'loss_state_dict': loss_fn.state_dict(),
            'metrics': metrics,
            'config': config,
            'timestamp': datetime.now().isoformat(),
            'git_commit': self._get_git_commit(),
            'torch_version': torch.__version__,
            'python_version': f"{os.sys.version_info.major}.{os.sys.version_info.minor}",
        }
        
        # Add random states for full reproducibility
        checkpoint_data['random_states'] = {
            'python': os.getstate() if hasattr(os, 'getstate') else None,
            'numpy': None,  # Will add if numpy available
            'torch': torch.get_rng_state(),
            'torch_cuda': torch.cuda.get_rng_state() if torch.cuda.is_available() else None
        }
        
        try:
            import numpy as np
            checkpoint_data['random_states']['numpy'] = np.random.get_state()
        except ImportError:
            pass
            
        # Determine filename
        if is_best and self.save_best:
            filename = "best_checkpoint.pth"
        elif is_milestone:
            filename = f"milestone_epoch_{epoch:03d}.pth"
        else:
            filename = f"checkpoint_epoch_{epoch:03d}.pth"
            
        filepath = self.checkpoint_dir / filename
        
        # Save checkpoint
        start_time = time.time()
        torch.save(checkpoint_data, filepath)
        save_time = time.time() - start_time
        
        # Get file size
        file_size = filepath.stat().st_size
        
        # Create metadata
        metadata = CheckpointMetadata(
            epoch=epoch,
            step=step,
            timestamp=checkpoint_data['timestamp'],
            loss=metrics.get('loss', float('inf')),
            metrics=metrics,
            config_hash=self._compute_config_hash(config),
            git_commit=checkpoint_data['git_commit'],
            file_size=file_size,
            is_best=is_best,
            is_milestone=is_milestone
        )
        
        # Update tracking
        if is_best and self.save_best:
            self.best_checkpoint = metadata
            self.best_loss = metadata.loss
            self.logger.info(f"New best checkpoint saved: {filepath} (loss: {metadata.loss:.4f})")
            
        if not is_best or not self.save_best:
            self.checkpoints.append(metadata)
            
        # Cleanup old checkpoints
        if not is_milestone and not is_best:
            self._cleanup_old_checkpoints()
            
        # Save registry
        self._save_checkpoint_registry()
        
        # Log save information
        self.logger.info(
            f"Checkpoint saved: {filename} "
            f"(epoch: {epoch}, loss: {metadata.loss:.4f}, "
            f"size: {file_size / 1024**2:.1f}MB, time: {save_time:.1f}s)"
        )
        
        return str(filepath)
        
    def _cleanup_old_checkpoints(self):
        """Remove old checkpoints beyond max_keep."""
        # Sort checkpoints by epoch (keep most recent)
        regular_checkpoints = [
            cp for cp in self.checkpoints 
            if not cp.is_milestone and not cp.is_best
        ]
        
        if len(regular_checkpoints) > self.max_keep:
            # Remove oldest checkpoints
            to_remove = sorted(regular_checkpoints, key=lambda x: x.epoch)[:-self.max_keep]
            
            for cp in to_remove:
                filepath = self.checkpoint_dir / f"checkpoint_epoch_{cp.epoch:03d}.pth"
                if filepath.exists():
                    filepath.unlink()
                    self.logger.info(f"Removed old checkpoint: {filepath}")
                    
                self.checkpoints.remove(cp)
                
    def load_checkpoint(self, 
                       model: torch.nn.Module,
                       optimizer: torch.optim.Optimizer,
                       scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
                       loss_fn: torch.nn.Module,
                       checkpoint_path: Optional[str] = None,
                       load_best: bool = False,
                       load_latest: bool = False,
                       device: str = 'cpu') -> Tuple[int, int, Dict[str, float]]:
        """
        Load a checkpoint and restore training state.
        
        Returns:
            Tuple of (epoch, step, metrics)
        """
        
        # Determine which checkpoint to load
        if load_best and self.best_checkpoint:
            filepath = self.checkpoint_dir / "best_checkpoint.pth"
        elif load_latest and self.checkpoints:
            latest_cp = max(self.checkpoints, key=lambda x: x.epoch)
            filepath = self.checkpoint_dir / f"checkpoint_epoch_{latest_cp.epoch:03d}.pth"
        elif checkpoint_path:
            filepath = Path(checkpoint_path)
        else:
            raise ValueError("Must specify checkpoint_path, load_best=True, or load_latest=True")
            
        if not filepath.exists():
            raise FileNotFoundError(f"Checkpoint not found: {filepath}")
            
        self.logger.info(f"Loading checkpoint: {filepath}")
        
        # Load checkpoint
        start_time = time.time()
        checkpoint = torch.load(filepath, map_location=device)
        load_time = time.time() - start_time
        
        # Restore model state
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Restore optimizer state
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Restore scheduler state
        if scheduler and checkpoint['scheduler_state_dict']:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
        # Restore loss function state
        if 'loss_state_dict' in checkpoint:
            loss_fn.load_state_dict(checkpoint['loss_state_dict'])
            
        # Restore random states for reproducibility
        if 'random_states' in checkpoint:
            random_states = checkpoint['random_states']
            
            if random_states.get('torch'):
                torch.set_rng_state(random_states['torch'])
                
            if random_states.get('torch_cuda') and torch.cuda.is_available():
                torch.cuda.set_rng_state(random_states['torch_cuda'])
                
            if random_states.get('numpy'):
                try:
                    import numpy as np
                    np.random.set_state(random_states['numpy'])
                except ImportError:
                    pass
                    
        epoch = checkpoint['epoch']
        step = checkpoint['step']
        metrics = checkpoint['metrics']
        
        self.logger.info(
            f"Checkpoint loaded successfully "
            f"(epoch: {epoch}, step: {step}, time: {load_time:.1f}s)"
        )
        
        return epoch, step, metrics
        
    def get_latest_checkpoint(self) -> Optional[str]:
        """Get path to latest checkpoint."""
        if not self.checkpoints:
            return None
            
        latest_cp = max(self.checkpoints, key=lambda x: x.epoch)
        return str(self.checkpoint_dir / f"checkpoint_epoch_{latest_cp.epoch:03d}.pth")
        
    def get_best_checkpoint(self) -> Optional[str]:
        """Get path to best checkpoint."""
        if not self.best_checkpoint:
            return None
            
        return str(self.checkpoint_dir / "best_checkpoint.pth")
        
    def list_checkpoints(self) -> List[CheckpointMetadata]:
        """List all available checkpoints."""
        all_checkpoints = self.checkpoints.copy()
        if self.best_checkpoint:
            all_checkpoints.append(self.best_checkpoint)
            
        return sorted(all_checkpoints, key=lambda x: x.epoch)
        
    def get_checkpoint_info(self) -> Dict[str, Any]:
        """Get summary information about checkpoints."""
        total_size = sum(cp.file_size for cp in self.checkpoints)
        if self.best_checkpoint:
            total_size += self.best_checkpoint.file_size
            
        return {
            'total_checkpoints': len(self.checkpoints),
            'has_best': self.best_checkpoint is not None,
            'best_loss': self.best_loss if self.best_checkpoint else None,
            'total_size_mb': total_size / 1024**2,
            'checkpoint_dir': str(self.checkpoint_dir),
            'latest_epoch': max((cp.epoch for cp in self.checkpoints), default=0)
        }
        
    def backup_checkpoints(self, backup_dir: str):
        """Create backup of all checkpoints."""
        backup_path = Path(backup_dir)
        backup_path.mkdir(parents=True, exist_ok=True)
        
        # Copy all checkpoint files
        for filepath in self.checkpoint_dir.glob("*.pth"):
            shutil.copy2(filepath, backup_path)
            
        # Copy registry
        registry_file = self.checkpoint_dir / "checkpoint_registry.json"
        if registry_file.exists():
            shutil.copy2(registry_file, backup_path)
            
        self.logger.info(f"Checkpoints backed up to: {backup_path}")
        
    def cleanup_all(self):
        """Remove all checkpoints (use with caution!)."""
        for filepath in self.checkpoint_dir.glob("*.pth"):
            filepath.unlink()
            
        registry_file = self.checkpoint_dir / "checkpoint_registry.json"
        if registry_file.exists():
            registry_file.unlink()
            
        self.checkpoints.clear()
        self.best_checkpoint = None
        self.best_loss = float('inf')
        
        self.logger.info("All checkpoints removed")


class AutoCheckpointManager:
    """Automatic checkpoint management with failure recovery."""
    
    def __init__(self, checkpoint_manager: DINOCheckpointManager):
        self.checkpoint_manager = checkpoint_manager
        self.logger = logging.getLogger('AutoCheckpointManager')
        
    def auto_save_checkpoint(self,
                           epoch: int,
                           step: int,
                           model: torch.nn.Module,
                           optimizer: torch.optim.Optimizer,
                           scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
                           loss_fn: torch.nn.Module,
                           metrics: Dict[str, float],
                           config: Dict[str, Any],
                           save_freq: int = 10,
                           total_epochs: int = None) -> Optional[str]:
        """Automatically save checkpoint based on conditions."""
        
        should_save = False
        reason = ""
        
        # Regular interval
        if epoch % save_freq == 0:
            should_save = True
            reason = f"regular interval ({save_freq} epochs)"
            
        # Best model
        if metrics.get('loss', float('inf')) < self.checkpoint_manager.best_loss:
            should_save = True
            reason = "new best model"
            
        # Milestone
        if self.checkpoint_manager._is_milestone_epoch(epoch, total_epochs):
            should_save = True
            reason = "milestone epoch"
            
        # Last epoch
        if total_epochs and epoch >= total_epochs - 1:
            should_save = True
            reason = "final epoch"
            
        if should_save:
            try:
                filepath = self.checkpoint_manager.save_checkpoint(
                    epoch=epoch,
                    step=step,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    loss_fn=loss_fn,
                    metrics=metrics,
                    config=config,
                    total_epochs=total_epochs
                )
                self.logger.info(f"Auto-saved checkpoint: {reason}")
                return filepath
            except Exception as e:
                self.logger.error(f"Failed to save checkpoint: {e}")
                return None
                
        return None
        
    def auto_resume_training(self,
                           model: torch.nn.Module,
                           optimizer: torch.optim.Optimizer,
                           scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
                           loss_fn: torch.nn.Module,
                           device: str = 'cpu') -> Tuple[int, int, Dict[str, float]]:
        """Automatically resume from latest checkpoint if available."""
        
        latest_checkpoint = self.checkpoint_manager.get_latest_checkpoint()
        
        if latest_checkpoint:
            self.logger.info("Found existing checkpoint, resuming training...")
            return self.checkpoint_manager.load_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                loss_fn=loss_fn,
                checkpoint_path=latest_checkpoint,
                device=device
            )
        else:
            self.logger.info("No existing checkpoint found, starting from scratch")
            return 0, 0, {}
```

### Step 2: Integration with Training Loop

```python
# training/checkpointed_trainer.py
import torch
import time
from pathlib import Path

from .monitored_trainer import MonitoredDINOTrainer
from ..checkpoint.checkpoint_manager import DINOCheckpointManager, AutoCheckpointManager

class CheckpointedDINOTrainer(MonitoredDINOTrainer):
    """DINO trainer with advanced checkpoint management."""
    
    def __init__(self, config_path: str, enable_wandb: bool = True, 
                 enable_live_dashboard: bool = False, auto_resume: bool = True):
        super().__init__(config_path, enable_wandb, enable_live_dashboard)
        
        # Initialize checkpoint management
        self.checkpoint_manager = DINOCheckpointManager(
            checkpoint_dir=self.config['checkpoint']['checkpoint_dir'],
            max_keep=self.config['checkpoint'].get('max_keep', 5),
            save_best=self.config['checkpoint'].get('save_best', True),
            save_milestones=self.config['checkpoint'].get('save_milestones', True),
            milestone_epochs=self.config['checkpoint'].get('milestone_epochs', [25, 50, 75, 100])
        )
        
        self.auto_checkpoint_manager = AutoCheckpointManager(self.checkpoint_manager)
        
        # Auto-resume if enabled
        self.start_epoch = 0
        self.start_step = 0
        if auto_resume:
            self.start_epoch, self.start_step, resume_metrics = self.auto_checkpoint_manager.auto_resume_training(
                model=self.model,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                loss_fn=self.loss_fn,
                device=str(self.device)
            )
            
            if self.start_epoch > 0:
                self.global_step = self.start_step
                self.logger.info(f"Resumed training from epoch {self.start_epoch}, step {self.start_step}")
                
        # Manual resume from specific checkpoint
        elif self.config['checkpoint'].get('resume_from'):
            self.start_epoch, self.start_step, resume_metrics = self.checkpoint_manager.load_checkpoint(
                model=self.model,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                loss_fn=self.loss_fn,
                checkpoint_path=self.config['checkpoint']['resume_from'],
                device=str(self.device)
            )
            self.global_step = self.start_step
            
    def train(self):
        """Training loop with checkpoint management."""
        train_config = self.config['training']
        checkpoint_config = self.config['checkpoint']
        
        # Setup checkpoint directory
        checkpoint_dir = checkpoint_config['checkpoint_dir']
        Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
        
        # Training loop
        self.logger.info("Starting DINO training with checkpoint management...")
        start_time = time.time()
        
        try:
            for epoch in range(self.start_epoch, train_config['epochs']):
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
                    
                # Auto-save checkpoint
                saved_checkpoint = self.auto_checkpoint_manager.auto_save_checkpoint(
                    epoch=epoch,
                    step=self.global_step,
                    model=self.model,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                    loss_fn=self.loss_fn,
                    metrics=all_metrics,
                    config=self.config,
                    save_freq=checkpoint_config.get('save_freq', 10),
                    total_epochs=train_config['epochs']
                )
                
                # Log epoch results
                epoch_time = time.time() - epoch_start
                self.logger.info(
                    f"Epoch {epoch:3d} completed in {epoch_time:.1f}s - "
                    f"Train Loss: {train_metrics['loss']:.4f} "
                    f"Val Loss: {val_metrics['val_loss']:.4f}"
                    + (f" - Checkpoint saved" if saved_checkpoint else "")
                )
                
        except KeyboardInterrupt:
            self.logger.info("Training interrupted by user")
            # Save emergency checkpoint
            emergency_checkpoint = self.checkpoint_manager.save_checkpoint(
                epoch=epoch,
                step=self.global_step,
                model=self.model,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                loss_fn=self.loss_fn,
                metrics=all_metrics,
                config=self.config,
                force_save=True
            )
            self.logger.info(f"Emergency checkpoint saved: {emergency_checkpoint}")
            
        except Exception as e:
            self.logger.error(f"Training failed with error: {e}")
            # Save error checkpoint for debugging
            error_checkpoint = self.checkpoint_manager.save_checkpoint(
                epoch=epoch,
                step=self.global_step,
                model=self.model,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                loss_fn=self.loss_fn,
                metrics=all_metrics,
                config=self.config,
                force_save=True
            )
            self.logger.info(f"Error checkpoint saved: {error_checkpoint}")
            raise
            
        # Training complete
        total_time = time.time() - start_time
        self.logger.info(f"Training completed in {total_time/3600:.1f} hours")
        
        # Print checkpoint summary
        checkpoint_info = self.checkpoint_manager.get_checkpoint_info()
        self.logger.info(f"Training completed with {checkpoint_info['total_checkpoints']} checkpoints")
        self.logger.info(f"Best checkpoint loss: {checkpoint_info['best_loss']:.4f}")
        self.logger.info(f"Total checkpoint storage: {checkpoint_info['total_size_mb']:.1f} MB")
        
    def create_training_backup(self, backup_dir: str):
        """Create complete backup of training state."""
        self.checkpoint_manager.backup_checkpoints(backup_dir)
        
        # Also backup config and logs
        backup_path = Path(backup_dir)
        backup_path.mkdir(parents=True, exist_ok=True)
        
        # Copy config
        config_backup = backup_path / "training_config.yaml"
        import yaml
        with open(config_backup, 'w') as f:
            yaml.dump({'dino_training': self.config}, f)
            
        self.logger.info(f"Complete training backup created: {backup_path}")
```

### Step 3: Checkpoint Analysis Tools

```python
# checkpoint/checkpoint_analyzer.py
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Optional
import json

class CheckpointAnalyzer:
    """Analyze checkpoint files and training progression."""
    
    def __init__(self, checkpoint_manager: DINOCheckpointManager):
        self.checkpoint_manager = checkpoint_manager
        
    def analyze_model_evolution(self) -> Dict[str, List[float]]:
        """Analyze how model parameters evolve during training."""
        checkpoints = self.checkpoint_manager.list_checkpoints()
        
        if len(checkpoints) < 2:
            return {}
            
        evolution_data = {
            'epochs': [],
            'param_norm': [],
            'param_change_norm': [],
            'grad_norm': []
        }
        
        prev_params = None
        
        for cp in sorted(checkpoints, key=lambda x: x.epoch):
            if cp.is_best:
                filepath = self.checkpoint_manager.checkpoint_dir / "best_checkpoint.pth"
            elif cp.is_milestone:
                filepath = self.checkpoint_manager.checkpoint_dir / f"milestone_epoch_{cp.epoch:03d}.pth"
            else:
                filepath = self.checkpoint_manager.checkpoint_dir / f"checkpoint_epoch_{cp.epoch:03d}.pth"
                
            if not filepath.exists():
                continue
                
            checkpoint = torch.load(filepath, map_location='cpu')
            
            # Calculate parameter norms
            param_norm = 0.0
            param_count = 0
            current_params = []
            
            for name, param in checkpoint['model_state_dict'].items():
                if 'weight' in name:  # Focus on weight parameters
                    param_norm += param.norm().item() ** 2
                    param_count += 1
                    current_params.append(param.flatten())
                    
            param_norm = (param_norm ** 0.5) / param_count if param_count > 0 else 0
            
            evolution_data['epochs'].append(cp.epoch)
            evolution_data['param_norm'].append(param_norm)
            
            # Calculate parameter change
            if prev_params is not None:
                param_change_norm = 0.0
                for curr, prev in zip(current_params, prev_params):
                    param_change_norm += (curr - prev).norm().item() ** 2
                param_change_norm = param_change_norm ** 0.5
                evolution_data['param_change_norm'].append(param_change_norm)
            else:
                evolution_data['param_change_norm'].append(0.0)
                
            prev_params = current_params
            
        return evolution_data
        
    def compare_checkpoints(self, epoch1: int, epoch2: int) -> Dict[str, float]:
        """Compare two checkpoints and analyze differences."""
        cp1_path = self.checkpoint_manager.checkpoint_dir / f"checkpoint_epoch_{epoch1:03d}.pth"
        cp2_path = self.checkpoint_manager.checkpoint_dir / f"checkpoint_epoch_{epoch2:03d}.pth"
        
        if not (cp1_path.exists() and cp2_path.exists()):
            return {}
            
        cp1 = torch.load(cp1_path, map_location='cpu')
        cp2 = torch.load(cp2_path, map_location='cpu')
        
        comparison = {
            'epoch_diff': epoch2 - epoch1,
            'loss_diff': cp2['metrics']['loss'] - cp1['metrics']['loss'],
            'param_differences': {}
        }
        
        # Compare parameters layer by layer
        for name in cp1['model_state_dict']:
            if name in cp2['model_state_dict']:
                param1 = cp1['model_state_dict'][name]
                param2 = cp2['model_state_dict'][name]
                
                diff_norm = (param2 - param1).norm().item()
                relative_change = diff_norm / (param1.norm().item() + 1e-8)
                
                comparison['param_differences'][name] = {
                    'absolute_change': diff_norm,
                    'relative_change': relative_change
                }
                
        return comparison
        
    def plot_checkpoint_timeline(self) -> plt.Figure:
        """Create visual timeline of checkpoints."""
        checkpoints = self.checkpoint_manager.list_checkpoints()
        
        if not checkpoints:
            return None
            
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Checkpoint timeline
        epochs = [cp.epoch for cp in checkpoints]
        losses = [cp.loss for cp in checkpoints]
        types = ['Best' if cp.is_best else 'Milestone' if cp.is_milestone else 'Regular' 
                for cp in checkpoints]
        
        # Color code by type
        colors = {'Best': 'red', 'Milestone': 'orange', 'Regular': 'blue'}
        
        for checkpoint_type in ['Regular', 'Milestone', 'Best']:
            type_epochs = [e for e, t in zip(epochs, types) if t == checkpoint_type]
            type_losses = [l for l, t in zip(losses, types) if t == checkpoint_type]
            
            if type_epochs:
                ax1.scatter(type_epochs, type_losses, 
                          c=colors[checkpoint_type], label=checkpoint_type, alpha=0.7)
                
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Checkpoint Timeline')
        ax1.legend()
        ax1.grid(True)
        
        # File sizes
        sizes_mb = [cp.file_size / 1024**2 for cp in checkpoints]
        ax2.bar(range(len(epochs)), sizes_mb, alpha=0.7)
        ax2.set_xlabel('Checkpoint Index')
        ax2.set_ylabel('File Size (MB)')
        ax2.set_title('Checkpoint File Sizes')
        ax2.grid(True)
        
        plt.tight_layout()
        return fig
        
    def generate_checkpoint_report(self) -> str:
        """Generate comprehensive checkpoint report."""
        info = self.checkpoint_manager.get_checkpoint_info()
        checkpoints = self.checkpoint_manager.list_checkpoints()
        
        report = f"""
# DINO Training Checkpoint Report

## Summary
- **Total Checkpoints**: {info['total_checkpoints']}
- **Has Best Checkpoint**: {info['has_best']}
- **Best Loss**: {info['best_loss']:.4f if info['best_loss'] else 'N/A'}
- **Total Storage**: {info['total_size_mb']:.1f} MB
- **Latest Epoch**: {info['latest_epoch']}

## Checkpoint Details
"""
        
        for cp in sorted(checkpoints, key=lambda x: x.epoch):
            report += f"""
### Epoch {cp.epoch}
- **Type**: {'Best' if cp.is_best else 'Milestone' if cp.is_milestone else 'Regular'}
- **Loss**: {cp.loss:.4f}
- **File Size**: {cp.file_size / 1024**2:.1f} MB
- **Timestamp**: {cp.timestamp}
- **Git Commit**: {cp.git_commit or 'Unknown'}
"""
        
        return report
```

## 🧪 Practical Exercises

### Exercise 1: Basic Checkpoint Management
```python
# Setup checkpointed training
trainer = CheckpointedDINOTrainer(
    config_path='config/cifar10_config.yaml',
    auto_resume=True  # Automatically resume if checkpoints exist
)

# Train with automatic checkpointing
trainer.train()

# Analyze checkpoints
analyzer = CheckpointAnalyzer(trainer.checkpoint_manager)
report = analyzer.generate_checkpoint_report()
print(report)
```

### Exercise 2: Checkpoint Recovery Simulation
```python
# Simulate training interruption and recovery
def simulate_training_interruption():
    """Simulate interrupted training scenario."""
    
    # Start training
    trainer = CheckpointedDINOTrainer('config/test_config.yaml')
    
    # Train for a few epochs
    for epoch in range(5):
        trainer.train_epoch(epoch)
        
        # Simulate interruption at epoch 3
        if epoch == 3:
            print("🚨 Simulating training interruption...")
            break
            
    # Create new trainer to simulate restart
    print("🔄 Restarting training...")
    new_trainer = CheckpointedDINOTrainer(
        'config/test_config.yaml',
        auto_resume=True
    )
    
    # Should automatically resume from epoch 3
    print(f"Resumed from epoch: {new_trainer.start_epoch}")
```

### Exercise 3: Checkpoint Space Optimization
```python
# Implement checkpoint compression
class CompressedCheckpointManager(DINOCheckpointManager):
    def save_checkpoint(self, **kwargs):
        """Save checkpoint with compression."""
        import gzip
        
        # Save normally first
        filepath = super().save_checkpoint(**kwargs)
        
        # Compress the file
        with open(filepath, 'rb') as f_in:
            with gzip.open(f"{filepath}.gz", 'wb') as f_out:
                f_out.writelines(f_in)
                
        # Remove uncompressed version
        Path(filepath).unlink()
        
        return f"{filepath}.gz"
```

## 🎯 Key Takeaways

1. **Robust Recovery**: Automatic checkpoint management prevents training loss
2. **Efficient Storage**: Smart cleanup and compression save disk space
3. **Reproducibility**: Complete state preservation enables exact training reproduction
4. **Monitoring Integration**: Checkpoint metadata tracks training progression
5. **Error Handling**: Emergency checkpoints save progress during failures

## 🔍 What's Next?

In **Lesson 5.4**, we'll focus on training optimization techniques including memory efficiency, performance profiling, and scaling strategies for larger models and datasets.

The checkpoint management system you've built provides enterprise-grade reliability for long DINO training runs, ensuring your training investment is always protected!
