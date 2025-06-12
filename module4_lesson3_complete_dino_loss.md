# Module 4, Lesson 3: Complete DINO Loss Function

## 🎯 Learning Objectives
By the end of this lesson, you will:
- Integrate centering and temperature scaling into the complete DINO loss
- Implement asymmetric loss formulation (student learns from teacher)
- Build loss aggregation across multiple crops
- Create comprehensive loss monitoring and debugging tools

## 📚 Theoretical Background

### Complete DINO Loss Formulation

The **complete DINO loss** combines all components:

1. **Multi-crop generation**: Global and local crops from same image
2. **Student-teacher forward**: Process crops through both networks
3. **Centering**: Subtract running mean from teacher outputs
4. **Temperature scaling**: Apply different temperatures to student/teacher
5. **Cross-entropy loss**: Student learns to match teacher distributions
6. **Asymmetric updates**: Only student receives gradients

### Mathematical Framework

**Complete Loss Equation**:
```
L_DINO = Σ_{s∈S} Σ_{t∈T} H(P_t, P_s)
```

Where:
- `S` = all student crops (global + local)
- `T` = teacher crops (global only)
- `P_t = softmax((f_t - c) / τ_t)` (teacher probabilities)
- `P_s = softmax(f_s / τ_s)` (student probabilities)
- `c` = running center
- `τ_t, τ_s` = teacher and student temperatures

**Key Properties**:
- **Asymmetric**: Only student backpropagates
- **Multi-scale**: Global and local crops provide different views
- **Self-supervised**: No external labels needed
- **Collapse-resistant**: Centering prevents trivial solutions

## 🛠️ Implementation

### Step 1: Complete DINO Loss Implementation

```python
# complete_dino_loss.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional, List
import warnings

class CompleteDINOLoss(nn.Module):
    """
    Complete DINO loss function with all components:
    - Multi-crop strategy
    - Centering mechanism
    - Temperature scaling
    - Asymmetric loss computation
    """
    
    def __init__(
        self,
        output_dim: int,
        teacher_temperature: float = 0.04,
        student_temperature: float = 0.1,
        center_momentum: float = 0.9,
        lambda_local: float = 1.0,
        warmup_teacher_temp_epochs: int = 30,
        teacher_temp_schedule: str = 'warmup_cosine',
        clip_grad: Optional[float] = 3.0,
        freeze_last_layer: int = 1
    ):
        super().__init__()
        
        self.output_dim = output_dim
        self.teacher_temperature = teacher_temperature
        self.student_temperature = student_temperature
        self.center_momentum = center_momentum
        self.lambda_local = lambda_local
        self.warmup_teacher_temp_epochs = warmup_teacher_temp_epochs
        self.teacher_temp_schedule = teacher_temp_schedule
        self.clip_grad = clip_grad
        self.freeze_last_layer = freeze_last_layer
        
        # Initialize center for teacher outputs
        self.register_buffer("center", torch.zeros(1, output_dim))
        
        # Track training statistics
        self.register_buffer("num_updates", torch.tensor(0, dtype=torch.long))
        
        # Loss component tracking
        self.loss_components = {
            'total': [],
            'global_to_global': [],
            'local_to_global': [],
            'center_norm': [],
            'teacher_temp': [],
            'entropy_teacher': [],
            'entropy_student': []
        }
    
    def forward(
        self,
        student_global_crops: torch.Tensor,
        student_local_crops: torch.Tensor,
        teacher_global_crops: torch.Tensor,
        epoch: int,
        update_center: bool = True
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute complete DINO loss
        
        Args:
            student_global_crops: [batch_size * 2, output_dim]
            student_local_crops: [batch_size * num_local, output_dim] 
            teacher_global_crops: [batch_size * 2, output_dim]
            epoch: Current training epoch
            update_center: Whether to update the center
            
        Returns:
            loss: Total DINO loss
            loss_dict: Detailed loss breakdown
        """
        
        batch_size = teacher_global_crops.shape[0] // 2  # 2 global crops per image
        num_local_crops = student_local_crops.shape[0] // batch_size
        
        # Get current teacher temperature
        teacher_temp = self._get_teacher_temperature(epoch)
        
        # Update center with teacher outputs
        if update_center and self.training:
            self._update_center(teacher_global_crops)
        
        # Apply centering to teacher outputs
        teacher_centered = teacher_global_crops - self.center
        
        # Compute probability distributions
        teacher_probs = F.softmax(teacher_centered / teacher_temp, dim=1)
        student_global_probs = F.softmax(student_global_crops / self.student_temperature, dim=1)
        student_local_probs = F.softmax(student_local_crops / self.student_temperature, dim=1)
        
        # Compute global-to-global loss
        global_loss = self._compute_global_to_global_loss(
            student_global_probs, teacher_probs, batch_size
        )
        
        # Compute local-to-global loss
        local_loss = self._compute_local_to_global_loss(
            student_local_probs, teacher_probs, batch_size, num_local_crops
        )
        
        # Total loss
        total_loss = global_loss + self.lambda_local * local_loss
        
        # Gather detailed statistics
        loss_dict = self._compute_loss_statistics(
            total_loss, global_loss, local_loss,
            student_global_probs, student_local_probs, teacher_probs,
            teacher_temp, epoch
        )
        
        # Update tracking
        self._update_loss_tracking(loss_dict)
        
        return total_loss, loss_dict
    
    def _get_teacher_temperature(self, epoch: int) -> float:
        """Get teacher temperature with scheduling"""
        if self.teacher_temp_schedule == 'constant':
            return self.teacher_temperature
        elif self.teacher_temp_schedule == 'warmup_cosine':
            if epoch < self.warmup_teacher_temp_epochs:
                # Warmup phase: linearly increase from 0.04 to final temperature
                return self.teacher_temperature + (0.04 - self.teacher_temperature) * (
                    (self.warmup_teacher_temp_epochs - epoch) / self.warmup_teacher_temp_epochs
                )
            else:
                # Post-warmup: use final temperature
                return self.teacher_temperature
        else:
            return self.teacher_temperature
    
    def _update_center(self, teacher_outputs: torch.Tensor):
        """Update center with exponential moving average"""
        with torch.no_grad():
            batch_center = torch.mean(teacher_outputs, dim=0, keepdim=True)
            
            if self.num_updates == 0:
                self.center.copy_(batch_center)
            else:
                self.center.mul_(self.center_momentum).add_(
                    batch_center, alpha=1.0 - self.center_momentum
                )
            
            self.num_updates += 1
    
    def _compute_global_to_global_loss(
        self,
        student_probs: torch.Tensor,
        teacher_probs: torch.Tensor,
        batch_size: int
    ) -> torch.Tensor:
        """
        Compute loss between global crops
        Student global crops learn from teacher global crops
        """
        total_loss = 0.0
        n_loss_terms = 0
        
        for iq in range(2):  # 2 global crops for teacher
            for iv in range(2):  # 2 global crops for student
                if iq == iv:
                    # Skip same crop
                    continue
                
                # Get corresponding batches
                teacher_batch = teacher_probs[iq * batch_size:(iq + 1) * batch_size]
                student_batch = student_probs[iv * batch_size:(iv + 1) * batch_size]
                
                # Cross-entropy loss: student learns from teacher
                loss = torch.sum(-teacher_batch * torch.log(student_batch), dim=1)
                total_loss += loss.mean()
                n_loss_terms += 1
        
        return total_loss / n_loss_terms
    
    def _compute_local_to_global_loss(
        self,
        student_local_probs: torch.Tensor,
        teacher_global_probs: torch.Tensor,
        batch_size: int,
        num_local_crops: int
    ) -> torch.Tensor:
        """
        Compute loss between local crops (student) and global crops (teacher)
        """
        total_loss = 0.0
        n_loss_terms = 0
        
        for iq in range(2):  # 2 global crops for teacher
            for iv in range(num_local_crops):  # Local crops for student
                # Get corresponding batches
                teacher_batch = teacher_global_probs[iq * batch_size:(iq + 1) * batch_size]
                student_batch = student_local_probs[iv * batch_size:(iv + 1) * batch_size]
                
                # Cross-entropy loss
                loss = torch.sum(-teacher_batch * torch.log(student_batch), dim=1)
                total_loss += loss.mean()
                n_loss_terms += 1
        
        return total_loss / n_loss_terms if n_loss_terms > 0 else torch.tensor(0.0)
    
    def _compute_loss_statistics(
        self,
        total_loss: torch.Tensor,
        global_loss: torch.Tensor,
        local_loss: torch.Tensor,
        student_global_probs: torch.Tensor,
        student_local_probs: torch.Tensor,
        teacher_probs: torch.Tensor,
        teacher_temp: float,
        epoch: int
    ) -> Dict[str, float]:
        """Compute detailed loss statistics"""
        
        with torch.no_grad():
            # Basic loss components
            loss_dict = {
                'total_loss': total_loss.item(),
                'global_loss': global_loss.item(),
                'local_loss': local_loss.item(),
                'center_norm': torch.norm(self.center).item(),
                'teacher_temperature': teacher_temp,
                'student_temperature': self.student_temperature,
                'epoch': epoch
            }
            
            # Entropy analysis
            def compute_entropy(probs):
                return torch.mean(-torch.sum(probs * torch.log(probs + 1e-8), dim=1))
            
            loss_dict.update({
                'entropy_teacher': compute_entropy(teacher_probs).item(),
                'entropy_student_global': compute_entropy(student_global_probs).item(),
                'entropy_student_local': compute_entropy(student_local_probs).item(),
            })
            
            # Confidence analysis (max probability)
            loss_dict.update({
                'confidence_teacher': torch.mean(torch.max(teacher_probs, dim=1)[0]).item(),
                'confidence_student_global': torch.mean(torch.max(student_global_probs, dim=1)[0]).item(),
                'confidence_student_local': torch.mean(torch.max(student_local_probs, dim=1)[0]).item(),
            })
            
            # Agreement analysis
            teacher_pred = torch.argmax(teacher_probs, dim=1)
            student_global_pred = torch.argmax(student_global_probs, dim=1)
            
            # Agreement between teacher global crops
            batch_size = teacher_probs.shape[0] // 2
            teacher_agreement = (
                teacher_pred[:batch_size] == teacher_pred[batch_size:]
            ).float().mean().item()
            
            # Agreement between student and teacher (first global crop)
            student_teacher_agreement = (
                student_global_pred[:batch_size] == teacher_pred[:batch_size]
            ).float().mean().item()
            
            loss_dict.update({
                'teacher_self_agreement': teacher_agreement,
                'student_teacher_agreement': student_teacher_agreement,
            })
            
            # Gradient norm (if available)
            if hasattr(self, '_last_grad_norm'):
                loss_dict['grad_norm'] = self._last_grad_norm
        
        return loss_dict
    
    def _update_loss_tracking(self, loss_dict: Dict[str, float]):
        """Update loss component tracking for analysis"""
        self.loss_components['total'].append(loss_dict['total_loss'])
        self.loss_components['global_to_global'].append(loss_dict['global_loss'])
        self.loss_components['local_to_global'].append(loss_dict['local_loss'])
        self.loss_components['center_norm'].append(loss_dict['center_norm'])
        self.loss_components['teacher_temp'].append(loss_dict['teacher_temperature'])
        self.loss_components['entropy_teacher'].append(loss_dict['entropy_teacher'])
        self.loss_components['entropy_student'].append(loss_dict['entropy_student_global'])
        
        # Keep only recent history (last 1000 steps)
        for key in self.loss_components:
            if len(self.loss_components[key]) > 1000:
                self.loss_components[key] = self.loss_components[key][-1000:]
    
    def get_loss_statistics(self) -> Dict[str, np.ndarray]:
        """Get historical loss statistics"""
        return {k: np.array(v) for k, v in self.loss_components.items()}
    
    def reset_center(self):
        """Reset the center (useful for debugging)"""
        self.center.zero_()
        self.num_updates.zero_()


class DINOLossWithValidation(CompleteDINOLoss):
    """
    DINO loss with additional validation and debugging features
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Validation thresholds
        self.validation_config = {
            'min_entropy': 0.1,
            'max_confidence': 0.99,
            'max_center_norm': 10.0,
            'min_agreement': 0.01,
            'max_loss': 100.0
        }
        
        # Issue tracking
        self.validation_issues = []
    
    def forward(self, *args, **kwargs) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Forward with validation"""
        
        # Run normal forward pass
        loss, loss_dict = super().forward(*args, **kwargs)
        
        # Validate results
        issues = self._validate_training_state(loss_dict)
        if issues:
            self.validation_issues.extend(issues)
            # Keep only recent issues
            self.validation_issues = self.validation_issues[-100:]
        
        # Add validation info to loss dict
        loss_dict['num_validation_issues'] = len(issues)
        loss_dict['total_validation_issues'] = len(self.validation_issues)
        
        return loss, loss_dict
    
    def _validate_training_state(self, loss_dict: Dict[str, float]) -> List[str]:
        """Validate training state and return list of issues"""
        issues = []
        
        # Check entropy
        if loss_dict['entropy_teacher'] < self.validation_config['min_entropy']:
            issues.append(f"Low teacher entropy: {loss_dict['entropy_teacher']:.4f}")
        
        if loss_dict['entropy_student_global'] < self.validation_config['min_entropy']:
            issues.append(f"Low student entropy: {loss_dict['entropy_student_global']:.4f}")
        
        # Check confidence
        if loss_dict['confidence_teacher'] > self.validation_config['max_confidence']:
            issues.append(f"High teacher confidence: {loss_dict['confidence_teacher']:.4f}")
        
        # Check center norm
        if loss_dict['center_norm'] > self.validation_config['max_center_norm']:
            issues.append(f"High center norm: {loss_dict['center_norm']:.4f}")
        
        # Check agreement
        if loss_dict['teacher_self_agreement'] < self.validation_config['min_agreement']:
            issues.append(f"Low teacher self-agreement: {loss_dict['teacher_self_agreement']:.4f}")
        
        # Check loss magnitude
        if loss_dict['total_loss'] > self.validation_config['max_loss']:
            issues.append(f"High total loss: {loss_dict['total_loss']:.4f}")
        
        if torch.isnan(torch.tensor(loss_dict['total_loss'])):
            issues.append("NaN loss detected!")
        
        return issues
    
    def get_validation_report(self) -> Dict:
        """Get comprehensive validation report"""
        if not self.validation_issues:
            return {"status": "healthy", "issues": []}
        
        # Count issue types
        issue_counts = {}
        for issue in self.validation_issues:
            issue_type = issue.split(':')[0]
            issue_counts[issue_type] = issue_counts.get(issue_type, 0) + 1
        
        return {
            "status": "issues_detected",
            "total_issues": len(self.validation_issues),
            "issue_types": issue_counts,
            "recent_issues": self.validation_issues[-10:],
            "recommendations": self._get_recommendations(issue_counts)
        }
    
    def _get_recommendations(self, issue_counts: Dict[str, int]) -> List[str]:
        """Get recommendations based on validation issues"""
        recommendations = []
        
        if "Low teacher entropy" in issue_counts:
            recommendations.append("Consider increasing teacher temperature or checking for mode collapse")
        
        if "Low student entropy" in issue_counts:
            recommendations.append("Consider increasing student temperature")
        
        if "High center norm" in issue_counts:
            recommendations.append("Check center momentum or consider resetting center")
        
        if "High total loss" in issue_counts:
            recommendations.append("Check learning rate, gradient clipping, or model initialization")
        
        if "NaN loss detected!" in issue_counts:
            recommendations.append("URGENT: Check for numerical instability, reduce learning rate")
        
        return recommendations
```

### Step 2: Training Integration

```python
# dino_trainer.py
import torch
import torch.nn as nn
from typing import Dict, Any, Optional
import logging
from pathlib import Path

class DINOTrainer:
    """
    Complete DINO trainer with loss computation and optimization
    """
    
    def __init__(
        self,
        model,  # Student-teacher model from Module 3
        loss_fn: CompleteDINOLoss,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        device: str = 'cuda',
        clip_grad_norm: float = 3.0,
        log_interval: int = 100,
        save_interval: int = 1000
    ):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.clip_grad_norm = clip_grad_norm
        self.log_interval = log_interval
        self.save_interval = save_interval
        
        # Move to device
        self.model.to(device)
        self.loss_fn.to(device)
        
        # Training state
        self.global_step = 0
        self.current_epoch = 0
        
        # Logging
        self.setup_logging()
        
    def setup_logging(self):
        """Setup logging for training"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def train_step(
        self,
        global_crops: torch.Tensor,
        local_crops: torch.Tensor,
        batch_size: int
    ) -> Dict[str, float]:
        """
        Single training step
        
        Args:
            global_crops: [batch_size * 2, 3, H, W]
            local_crops: [batch_size * num_local, 3, H_local, W_local]
            batch_size: Original batch size
            
        Returns:
            Dictionary with loss components and metrics
        """
        
        self.model.train()
        
        # Move to device
        global_crops = global_crops.to(self.device)
        local_crops = local_crops.to(self.device)
        
        # Forward pass through student and teacher
        with torch.cuda.amp.autocast():  # Mixed precision
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
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        if self.clip_grad_norm > 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.get_student_parameters(),
                self.clip_grad_norm
            )
            loss_dict['grad_norm'] = grad_norm.item()
            
            # Store for loss validation
            if hasattr(self.loss_fn, '_last_grad_norm'):
                self.loss_fn._last_grad_norm = grad_norm.item()
        
        # Optimizer step
        self.optimizer.step()
        
        # Update teacher weights
        self.model.update_teacher(self.current_epoch)
        
        # Update learning rate
        if self.scheduler is not None:
            self.scheduler.step()
            loss_dict['learning_rate'] = self.scheduler.get_last_lr()[0]
        
        # Update step counter
        self.global_step += 1
        
        # Add training metadata
        loss_dict.update({
            'global_step': self.global_step,
            'epoch': self.current_epoch,
            'batch_size': batch_size
        })
        
        return loss_dict
    
    def train_epoch(self, dataloader, epoch: int) -> Dict[str, float]:
        """Train for one epoch"""
        
        self.current_epoch = epoch
        epoch_metrics = []
        
        for batch_idx, batch in enumerate(dataloader):
            # Extract batch data (assuming MultiCropCollator format)
            global_crops = batch['global_crops']
            local_crops = batch['local_crops']
            batch_size = batch['batch_size']
            
            # Training step
            step_metrics = self.train_step(global_crops, local_crops, batch_size)
            epoch_metrics.append(step_metrics)
            
            # Logging
            if self.global_step % self.log_interval == 0:
                self._log_training_step(step_metrics)
            
            # Validation check (if using DINOLossWithValidation)
            if hasattr(self.loss_fn, 'get_validation_report'):
                validation_report = self.loss_fn.get_validation_report()
                if validation_report['status'] != 'healthy':
                    self.logger.warning(f"Validation issues detected: {validation_report}")
        
        # Compute epoch averages
        epoch_avg = self._compute_epoch_averages(epoch_metrics)
        
        self.logger.info(
            f"Epoch {epoch} completed - "
            f"Loss: {epoch_avg['total_loss']:.4f}, "
            f"Global: {epoch_avg['global_loss']:.4f}, "
            f"Local: {epoch_avg['local_loss']:.4f}"
        )
        
        return epoch_avg
    
    def _log_training_step(self, metrics: Dict[str, float]):
        """Log training step metrics"""
        self.logger.info(
            f"Step {self.global_step:6d} - "
            f"Loss: {metrics['total_loss']:.4f} "
            f"(G: {metrics['global_loss']:.4f}, L: {metrics['local_loss']:.4f}) - "
            f"T_temp: {metrics['teacher_temperature']:.3f} - "
            f"Center: {metrics['center_norm']:.3f}"
        )
    
    def _compute_epoch_averages(self, epoch_metrics: List[Dict]) -> Dict[str, float]:
        """Compute average metrics for epoch"""
        if not epoch_metrics:
            return {}
        
        # Get all metric keys
        all_keys = set()
        for metrics in epoch_metrics:
            all_keys.update(metrics.keys())
        
        # Compute averages
        averages = {}
        for key in all_keys:
            values = [m.get(key, 0) for m in epoch_metrics if key in m]
            if values:
                averages[key] = sum(values) / len(values)
        
        return averages
    
    def save_checkpoint(self, path: Path, extra_data: Optional[Dict] = None):
        """Save training checkpoint"""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss_state_dict': self.loss_fn.state_dict(),
            'global_step': self.global_step,
            'current_epoch': self.current_epoch,
        }
        
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        if extra_data:
            checkpoint.update(extra_data)
        
        torch.save(checkpoint, path)
        self.logger.info(f"Checkpoint saved to {path}")
    
    def load_checkpoint(self, path: Path):
        """Load training checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.loss_fn.load_state_dict(checkpoint['loss_state_dict'])
        self.global_step = checkpoint['global_step']
        self.current_epoch = checkpoint['current_epoch']
        
        if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.logger.info(f"Checkpoint loaded from {path}")
```

### Step 3: Loss Analysis and Visualization

```python
# loss_analysis.py
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from typing import Dict, List
import pandas as pd

class DINOLossAnalyzer:
    """
    Comprehensive analysis tools for DINO loss components
    """
    
    def __init__(self):
        self.metrics_history = []
        
    def log_metrics(self, metrics: Dict[str, float]):
        """Log metrics from training step"""
        self.metrics_history.append(metrics.copy())
    
    def plot_loss_components(self, save_path: str = None):
        """Plot evolution of different loss components"""
        if not self.metrics_history:
            print("No metrics to plot")
            return
        
        df = pd.DataFrame(self.metrics_history)
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Total loss
        if 'total_loss' in df.columns:
            axes[0, 0].plot(df['total_loss'], alpha=0.7)
            axes[0, 0].set_title('Total Loss')
            axes[0, 0].set_xlabel('Step')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].grid(True, alpha=0.3)
        
        # Loss components
        if 'global_loss' in df.columns and 'local_loss' in df.columns:
            axes[0, 1].plot(df['global_loss'], label='Global', alpha=0.7)
            axes[0, 1].plot(df['local_loss'], label='Local', alpha=0.7)
            axes[0, 1].set_title('Loss Components')
            axes[0, 1].set_xlabel('Step')
            axes[0, 1].set_ylabel('Loss')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
        
        # Temperature evolution
        if 'teacher_temperature' in df.columns:
            axes[0, 2].plot(df['teacher_temperature'], alpha=0.7, color='red')
            axes[0, 2].set_title('Teacher Temperature')
            axes[0, 2].set_xlabel('Step')
            axes[0, 2].set_ylabel('Temperature')
            axes[0, 2].grid(True, alpha=0.3)
        
        # Center norm
        if 'center_norm' in df.columns:
            axes[1, 0].plot(df['center_norm'], alpha=0.7, color='green')
            axes[1, 0].set_title('Center Norm')
            axes[1, 0].set_xlabel('Step')
            axes[1, 0].set_ylabel('L2 Norm')
            axes[1, 0].grid(True, alpha=0.3)
        
        # Entropy
        entropy_cols = [col for col in df.columns if 'entropy' in col]
        if entropy_cols:
            for col in entropy_cols:
                axes[1, 1].plot(df[col], label=col.replace('entropy_', ''), alpha=0.7)
            axes[1, 1].set_title('Entropy Evolution')
            axes[1, 1].set_xlabel('Step')
            axes[1, 1].set_ylabel('Entropy')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        
        # Confidence
        confidence_cols = [col for col in df.columns if 'confidence' in col]
        if confidence_cols:
            for col in confidence_cols:
                axes[1, 2].plot(df[col], label=col.replace('confidence_', ''), alpha=0.7)
            axes[1, 2].set_title('Confidence Evolution')
            axes[1, 2].set_xlabel('Step')
            axes[1, 2].set_ylabel('Max Probability')
            axes[1, 2].legend()
            axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        plt.show()
    
    def analyze_correlations(self) -> Dict[str, float]:
        """Analyze correlations between different metrics"""
        if not self.metrics_history:
            return {}
        
        df = pd.DataFrame(self.metrics_history)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        correlations = {}
        
        # Key correlations to analyze
        correlation_pairs = [
            ('total_loss', 'teacher_temperature'),
            ('total_loss', 'center_norm'),
            ('entropy_teacher', 'confidence_teacher'),
            ('entropy_student_global', 'confidence_student_global'),
            ('global_loss', 'local_loss'),
            ('teacher_self_agreement', 'student_teacher_agreement')
        ]
        
        for col1, col2 in correlation_pairs:
            if col1 in numeric_cols and col2 in numeric_cols:
                corr = df[col1].corr(df[col2])
                correlations[f'{col1}_vs_{col2}'] = corr
        
        return correlations
    
    def detect_anomalies(self, window_size: int = 100) -> List[Dict]:
        """Detect anomalies in training metrics"""
        if len(self.metrics_history) < window_size:
            return []
        
        df = pd.DataFrame(self.metrics_history)
        anomalies = []
        
        # Check for sudden spikes in loss
        if 'total_loss' in df.columns:
            loss_values = df['total_loss'].values
            for i in range(window_size, len(loss_values)):
                window_mean = np.mean(loss_values[i-window_size:i])
                window_std = np.std(loss_values[i-window_size:i])
                
                if loss_values[i] > window_mean + 3 * window_std:
                    anomalies.append({
                        'type': 'loss_spike',
                        'step': i,
                        'value': loss_values[i],
                        'expected': window_mean,
                        'severity': (loss_values[i] - window_mean) / window_std
                    })
        
        # Check for center norm explosion
        if 'center_norm' in df.columns:
            center_values = df['center_norm'].values
            for i in range(len(center_values)):
                if center_values[i] > 10.0:  # Threshold
                    anomalies.append({
                        'type': 'center_explosion',
                        'step': i,
                        'value': center_values[i],
                        'threshold': 10.0
                    })
        
        # Check for entropy collapse
        if 'entropy_teacher' in df.columns:
            entropy_values = df['entropy_teacher'].values
            for i in range(len(entropy_values)):
                if entropy_values[i] < 0.1:  # Very low entropy
                    anomalies.append({
                        'type': 'entropy_collapse',
                        'step': i,
                        'value': entropy_values[i],
                        'threshold': 0.1
                    })
        
        return anomalies
    
    def generate_training_report(self) -> Dict:
        """Generate comprehensive training report"""
        if not self.metrics_history:
            return {"status": "no_data"}
        
        df = pd.DataFrame(self.metrics_history)
        
        # Basic statistics
        report = {
            'total_steps': len(self.metrics_history),
            'final_metrics': self.metrics_history[-1] if self.metrics_history else {},
            'correlations': self.analyze_correlations(),
            'anomalies': self.detect_anomalies()
        }
        
        # Loss statistics
        if 'total_loss' in df.columns:
            report['loss_stats'] = {
                'final_loss': df['total_loss'].iloc[-1],
                'min_loss': df['total_loss'].min(),
                'max_loss': df['total_loss'].max(),
                'mean_loss': df['total_loss'].mean(),
                'loss_trend': np.polyfit(range(len(df)), df['total_loss'], 1)[0]
            }
        
        # Training health assessment
        health_score = self._compute_health_score(df)
        report['health_score'] = health_score
        report['recommendations'] = self._get_health_recommendations(health_score, df)
        
        return report
    
    def _compute_health_score(self, df: pd.DataFrame) -> float:
        """Compute overall training health score (0-1)"""
        score = 1.0
        
        # Penalize high loss values
        if 'total_loss' in df.columns:
            recent_loss = df['total_loss'].tail(100).mean()
            if recent_loss > 10:
                score -= 0.3
            elif recent_loss > 5:
                score -= 0.1
        
        # Penalize low entropy (mode collapse)
        if 'entropy_teacher' in df.columns:
            recent_entropy = df['entropy_teacher'].tail(100).mean()
            if recent_entropy < 0.5:
                score -= 0.3
            elif recent_entropy < 1.0:
                score -= 0.1
        
        # Penalize high center norm
        if 'center_norm' in df.columns:
            recent_center = df['center_norm'].tail(100).mean()
            if recent_center > 5:
                score -= 0.2
            elif recent_center > 2:
                score -= 0.1
        
        # Reward stable training (low variance)
        if 'total_loss' in df.columns and len(df) > 100:
            recent_loss_std = df['total_loss'].tail(100).std()
            if recent_loss_std < 0.1:
                score += 0.1
        
        return max(0.0, min(1.0, score))
    
    def _get_health_recommendations(self, health_score: float, df: pd.DataFrame) -> List[str]:
        """Get recommendations based on health score and metrics"""
        recommendations = []
        
        if health_score < 0.5:
            recommendations.append("URGENT: Training appears unstable")
        
        if 'total_loss' in df.columns:
            recent_loss = df['total_loss'].tail(100).mean()
            if recent_loss > 10:
                recommendations.append("Consider reducing learning rate - loss too high")
        
        if 'entropy_teacher' in df.columns:
            recent_entropy = df['entropy_teacher'].tail(100).mean()
            if recent_entropy < 0.5:
                recommendations.append("Possible mode collapse - check centering and temperature")
        
        if 'center_norm' in df.columns:
            recent_center = df['center_norm'].tail(100).mean()
            if recent_center > 5:
                recommendations.append("High center norm - consider resetting center")
        
        if len(recommendations) == 0:
            recommendations.append("Training appears healthy")
        
        return recommendations


def test_complete_dino_loss():
    """Test the complete DINO loss implementation"""
    
    # Create synthetic data
    batch_size = 4
    output_dim = 1000
    num_local_crops = 8
    
    # Student outputs
    student_global = torch.randn(batch_size * 2, output_dim)
    student_local = torch.randn(batch_size * num_local_crops, output_dim)
    
    # Teacher outputs  
    teacher_global = torch.randn(batch_size * 2, output_dim)
    
    # Create loss function
    loss_fn = CompleteDINOLoss(
        output_dim=output_dim,
        teacher_temperature=0.04,
        student_temperature=0.1,
        center_momentum=0.9
    )
    
    # Test forward pass
    loss, loss_dict = loss_fn(
        student_global_crops=student_global,
        student_local_crops=student_local,
        teacher_global_crops=teacher_global,
        epoch=0
    )
    
    print(f"Loss: {loss.item():.4f}")
    print("Loss components:")
    for key, value in loss_dict.items():
        print(f"  {key}: {value}")
    
    # Test multiple steps
    analyzer = DINOLossAnalyzer()
    
    for step in range(100):
        # Simulate changing distributions
        student_global = torch.randn(batch_size * 2, output_dim) * (1 + 0.01 * step)
        student_local = torch.randn(batch_size * num_local_crops, output_dim) * (1 + 0.01 * step)
        teacher_global = torch.randn(batch_size * 2, output_dim) * (1 + 0.005 * step)
        
        loss, loss_dict = loss_fn(
            student_global_crops=student_global,
            student_local_crops=student_local,
            teacher_global_crops=teacher_global,
            epoch=step // 10
        )
        
        loss_dict['step'] = step
        analyzer.log_metrics(loss_dict)
    
    # Generate analysis
    print("\nTraining Analysis:")
    report = analyzer.generate_training_report()
    print(f"Health Score: {report['health_score']:.3f}")
    print("Recommendations:")
    for rec in report['recommendations']:
        print(f"  - {rec}")
    
    # Plot results
    analyzer.plot_loss_components()

if __name__ == "__main__":
    test_complete_dino_loss()
```

## 🧪 Hands-on Exercise: Build Your Complete DINO Loss

### Exercise 1: Loss Component Integration

Implement the complete loss step by step:

```python
# exercise1.py
import torch
import torch.nn.functional as F

def compute_dino_loss_step_by_step(
    student_global, student_local, teacher_global,
    center, teacher_temp, student_temp, batch_size
):
    """
    Implement DINO loss computation step by step
    
    Returns:
        Dictionary with all intermediate computations
    """
    
    results = {}
    
    # Step 1: Apply centering to teacher
    # TODO: Subtract center from teacher outputs
    teacher_centered = None
    results['teacher_centered'] = teacher_centered
    
    # Step 2: Apply temperature scaling
    # TODO: Compute probability distributions
    teacher_probs = None
    student_global_probs = None  
    student_local_probs = None
    results.update({
        'teacher_probs': teacher_probs,
        'student_global_probs': student_global_probs,
        'student_local_probs': student_local_probs
    })
    
    # Step 3: Compute global-to-global loss
    # TODO: Cross-entropy between student global and teacher global
    global_loss = None
    results['global_loss'] = global_loss
    
    # Step 4: Compute local-to-global loss  
    # TODO: Cross-entropy between student local and teacher global
    local_loss = None
    results['local_loss'] = local_loss
    
    # Step 5: Combine losses
    total_loss = global_loss + local_loss
    results['total_loss'] = total_loss
    
    return results

# Test your implementation
```

### Exercise 2: Loss Debugging

Build tools to debug loss computation:

```python
# exercise2.py
def debug_loss_computation(loss_fn, *args, **kwargs):
    """
    Debug DINO loss computation by analyzing each component
    """
    
    # TODO: Implement debugging that checks:
    # 1. Input tensor shapes and ranges
    # 2. Probability distribution properties (sum to 1, non-negative)
    # 3. Loss component magnitudes
    # 4. Gradient flow (if tensors require gradients)
    # 5. Numerical stability (NaN/inf detection)
    
    pass

def visualize_probability_distributions(student_probs, teacher_probs):
    """Visualize student and teacher probability distributions"""
    
    # TODO: Create visualizations showing:
    # 1. Entropy distributions
    # 2. Max probability distributions  
    # 3. Probability mass concentration
    # 4. Agreement between student and teacher
    
    pass
```

### Exercise 3: Loss Scheduling

Implement adaptive loss weighting:

```python
# exercise3.py
class AdaptiveLossWeighting:
    def __init__(self):
        # TODO: Initialize adaptive weighting system
        pass
    
    def update_weights(self, global_loss, local_loss, epoch):
        """Update loss weights based on training dynamics"""
        # TODO: Implement adaptive weighting that:
        # 1. Balances global and local losses
        # 2. Adapts based on loss magnitudes
        # 3. Considers training phase (early vs late)
        pass
    
    def get_current_weights(self):
        """Get current loss weights"""
        # TODO: Return current lambda values
        pass
```

## 🔍 Key Insights

### DINO Loss Design Principles
1. **Asymmetric Learning**: Only student receives gradients, teacher provides stable targets
2. **Multi-Scale Consistency**: Global and local crops enforce scale-invariant representations
3. **Collapse Prevention**: Centering mechanism prevents trivial solutions
4. **Controlled Sharpness**: Temperature scaling controls learning dynamics

### Critical Implementation Details
1. **Loss Aggregation**: Proper averaging across multiple crop combinations
2. **Gradient Isolation**: Teacher must never receive gradients
3. **Numerical Stability**: Careful handling of log probabilities and division
4. **Memory Efficiency**: Efficient computation for large numbers of crops

### Training Dynamics
1. **Early Training**: High loss, rapid center adaptation, unstable dynamics
2. **Mid Training**: Stabilizing loss, slower center changes, emerging structure
3. **Late Training**: Low loss, stable center, fine-tuned representations
4. **Convergence**: Asymptotic behavior, minimal center updates

## 📝 Summary

In this lesson, you learned:

✅ **Complete DINO Loss Integration**: All components working together in one coherent loss function

✅ **Multi-Crop Loss Computation**: Proper aggregation across global and local crop combinations

✅ **Asymmetric Training**: Student-only gradient flow with teacher target generation

✅ **Loss Analysis Tools**: Comprehensive debugging and monitoring capabilities

✅ **Training Integration**: Complete trainer class with optimization and checkpointing

### Next Steps
In the next lesson, we'll implement the complete training loop with gradient clipping, learning rate scheduling, and optimization strategies.

## 🔗 Additional Resources

- [DINO Paper - Complete Method](https://arxiv.org/abs/2104.14294)
- [Cross-Entropy Loss in Deep Learning](https://machinelearningmastery.com/cross-entropy-for-machine-learning/)
- [Self-Supervised Learning Survey](https://arxiv.org/abs/1902.06162)

---

**Next**: [Module 4, Lesson 4: Training Loop Implementation](module4_lesson4_training_loop.md)
