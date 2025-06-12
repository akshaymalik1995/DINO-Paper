# Module 4, Lesson 1: Centering Mechanism Implementation

## 🎯 Learning Objectives
By the end of this lesson, you will:
- Understand why centering prevents mode collapse in DINO
- Implement the running mean computation with momentum
- Build center computation across batch dimensions
- Integrate centering into the complete DINO loss function

## 📚 Theoretical Background

### The Mode Collapse Problem

In self-supervised learning, **mode collapse** occurs when:
- All samples map to the same representation
- The model learns trivial solutions (constant outputs)
- Training becomes ineffective due to lack of diversity

**Example of Collapse:**
```
Input: [cat, dog, car, tree] → Output: [0.5, 0.5, 0.5, 0.5] (all same)
```

### How Centering Prevents Collapse

DINO's **centering mechanism** works by:
1. **Computing running average** of teacher outputs across batches
2. **Subtracting this center** from teacher predictions before softmax
3. **Preventing uniform distributions** by biasing away from the mean
4. **Maintaining diversity** in learned representations

### Mathematical Foundation

**Center Update Rule:**
```
c ← m * c + (1 - m) * E[teacher_outputs]
```

**Teacher Output Centering:**
```
teacher_centered = teacher_outputs - c
```

**Final Teacher Probabilities:**
```
P_teacher = softmax((teacher_outputs - c) / τ_teacher)
```

Where:
- `c` is the running center
- `m` is the momentum coefficient (typically 0.9)
- `τ_teacher` is the teacher temperature

## 🛠️ Implementation

### Step 1: Basic Centering Mechanism

```python
# centering.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple
import matplotlib.pyplot as plt

class CenteringMechanism(nn.Module):
    """
    Centering mechanism to prevent mode collapse in DINO
    """
    def __init__(
        self,
        output_dim: int,
        center_momentum: float = 0.9,
        eps: float = 1e-6
    ):
        super().__init__()
        
        self.output_dim = output_dim
        self.center_momentum = center_momentum
        self.eps = eps
        
        # Initialize center as zeros
        self.register_buffer('center', torch.zeros(1, output_dim))
        self.register_buffer('num_updates', torch.tensor(0, dtype=torch.long))
        
    def update_center(self, teacher_outputs: torch.Tensor):
        """
        Update the center with exponential moving average
        
        Args:
            teacher_outputs: [batch_size, output_dim] teacher predictions
        """
        with torch.no_grad():
            # Compute batch center
            batch_center = torch.mean(teacher_outputs, dim=0, keepdim=True)
            
            # Update running center with momentum
            if self.num_updates == 0:
                # First update: initialize with batch center
                self.center.copy_(batch_center)
            else:
                # EMA update
                self.center.mul_(self.center_momentum).add_(
                    batch_center, alpha=1 - self.center_momentum
                )
            
            self.num_updates += 1
    
    def apply_centering(self, teacher_outputs: torch.Tensor) -> torch.Tensor:
        """
        Apply centering to teacher outputs
        
        Args:
            teacher_outputs: [batch_size, output_dim] teacher predictions
            
        Returns:
            Centered teacher outputs
        """
        return teacher_outputs - self.center
    
    def forward(self, teacher_outputs: torch.Tensor, update_center: bool = True) -> torch.Tensor:
        """
        Forward pass: update center and apply centering
        
        Args:
            teacher_outputs: Teacher network outputs
            update_center: Whether to update the running center
            
        Returns:
            Centered teacher outputs
        """
        if update_center and self.training:
            self.update_center(teacher_outputs.detach())
        
        return self.apply_centering(teacher_outputs)
    
    def get_center_stats(self) -> dict:
        """Get statistics about the current center"""
        with torch.no_grad():
            center_norm = torch.norm(self.center).item()
            center_mean = torch.mean(self.center).item()
            center_std = torch.std(self.center).item()
            
            return {
                'center_norm': center_norm,
                'center_mean': center_mean,
                'center_std': center_std,
                'num_updates': self.num_updates.item()
            }


class AdaptiveCentering(nn.Module):
    """
    Adaptive centering with learnable momentum and per-dimension centers
    """
    def __init__(
        self,
        output_dim: int,
        initial_momentum: float = 0.9,
        learnable_momentum: bool = True,
        per_dimension_centers: bool = False,
        warmup_steps: int = 1000
    ):
        super().__init__()
        
        self.output_dim = output_dim
        self.per_dimension_centers = per_dimension_centers
        self.warmup_steps = warmup_steps
        
        # Center storage
        if per_dimension_centers:
            self.register_buffer('center', torch.zeros(1, output_dim))
        else:
            self.register_buffer('center', torch.zeros(1, 1))
        
        # Momentum parameter
        if learnable_momentum:
            self.momentum = nn.Parameter(torch.tensor(initial_momentum))
        else:
            self.register_buffer('momentum', torch.tensor(initial_momentum))
        
        self.register_buffer('step_count', torch.tensor(0, dtype=torch.long))
        
    def _get_effective_momentum(self) -> float:
        """Get effective momentum with warmup"""
        if self.step_count < self.warmup_steps:
            # Linear warmup
            warmup_factor = self.step_count.float() / self.warmup_steps
            return 0.1 + (self.momentum - 0.1) * warmup_factor
        else:
            return self.momentum
    
    def update_center(self, teacher_outputs: torch.Tensor):
        """Update center with adaptive momentum"""
        with torch.no_grad():
            if self.per_dimension_centers:
                batch_center = torch.mean(teacher_outputs, dim=0, keepdim=True)
            else:
                batch_center = torch.mean(teacher_outputs).unsqueeze(0).unsqueeze(0)
            
            effective_momentum = self._get_effective_momentum()
            
            if self.step_count == 0:
                self.center.copy_(batch_center)
            else:
                self.center.mul_(effective_momentum).add_(
                    batch_center, alpha=1 - effective_momentum
                )
            
            self.step_count += 1
    
    def forward(self, teacher_outputs: torch.Tensor, update_center: bool = True) -> torch.Tensor:
        if update_center and self.training:
            self.update_center(teacher_outputs.detach())
        
        if self.per_dimension_centers:
            return teacher_outputs - self.center
        else:
            return teacher_outputs - self.center.expand_as(teacher_outputs)


class DynamicCentering(nn.Module):
    """
    Dynamic centering that adapts based on training phase
    """
    def __init__(
        self,
        output_dim: int,
        momentum_schedule: dict = None,
        adaptive_threshold: float = 0.1
    ):
        super().__init__()
        
        self.output_dim = output_dim
        self.adaptive_threshold = adaptive_threshold
        
        # Default momentum schedule
        if momentum_schedule is None:
            momentum_schedule = {
                0: 0.5,      # Start aggressive
                1000: 0.7,   # Gradually increase
                5000: 0.9,   # Standard momentum
                10000: 0.95  # Very stable
            }
        
        self.momentum_schedule = momentum_schedule
        self.register_buffer('center', torch.zeros(1, output_dim))
        self.register_buffer('step_count', torch.tensor(0, dtype=torch.long))
        self.register_buffer('center_history', torch.zeros(100, output_dim))
        self.history_idx = 0
        
    def _get_scheduled_momentum(self) -> float:
        """Get momentum based on training step"""
        step = self.step_count.item()
        
        # Find appropriate momentum value
        momentum_keys = sorted(self.momentum_schedule.keys())
        
        for i, key in enumerate(momentum_keys):
            if step <= key:
                if i == 0:
                    return self.momentum_schedule[key]
                else:
                    # Linear interpolation
                    prev_key = momentum_keys[i-1]
                    prev_momentum = self.momentum_schedule[prev_key]
                    curr_momentum = self.momentum_schedule[key]
                    
                    alpha = (step - prev_key) / (key - prev_key)
                    return prev_momentum + alpha * (curr_momentum - prev_momentum)
        
        # Use last value if step exceeds all keys
        return self.momentum_schedule[momentum_keys[-1]]
    
    def _detect_instability(self) -> bool:
        """Detect if center is changing too rapidly"""
        if self.step_count < 10:
            return False
        
        # Compute variance of recent center changes
        recent_centers = self.center_history[:min(self.history_idx, 100)]
        if len(recent_centers) < 5:
            return False
        
        center_changes = torch.diff(recent_centers, dim=0)
        change_magnitude = torch.norm(center_changes, dim=1).mean()
        
        return change_magnitude > self.adaptive_threshold
    
    def update_center(self, teacher_outputs: torch.Tensor):
        """Update center with dynamic momentum"""
        with torch.no_grad():
            batch_center = torch.mean(teacher_outputs, dim=0, keepdim=True)
            
            # Get base momentum from schedule
            base_momentum = self._get_scheduled_momentum()
            
            # Adjust for instability
            if self._detect_instability():
                effective_momentum = min(base_momentum + 0.1, 0.99)
            else:
                effective_momentum = base_momentum
            
            # Update center
            if self.step_count == 0:
                self.center.copy_(batch_center)
            else:
                self.center.mul_(effective_momentum).add_(
                    batch_center, alpha=1 - effective_momentum
                )
            
            # Store in history
            hist_idx = self.step_count % 100
            self.center_history[hist_idx] = self.center.squeeze()
            self.history_idx = max(self.history_idx, hist_idx + 1)
            
            self.step_count += 1
    
    def forward(self, teacher_outputs: torch.Tensor, update_center: bool = True) -> torch.Tensor:
        if update_center and self.training:
            self.update_center(teacher_outputs.detach())
        
        return teacher_outputs - self.center
```

### Step 2: Integration with DINO Loss

```python
# dino_loss_with_centering.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

class DINOLossWithCentering(nn.Module):
    """
    DINO loss function with integrated centering mechanism
    """
    def __init__(
        self,
        output_dim: int,
        teacher_temperature: float = 0.04,
        student_temperature: float = 0.1,
        center_momentum: float = 0.9,
        centering_type: str = 'basic'
    ):
        super().__init__()
        
        self.teacher_temperature = teacher_temperature
        self.student_temperature = student_temperature
        
        # Initialize centering mechanism
        if centering_type == 'basic':
            self.centering = CenteringMechanism(output_dim, center_momentum)
        elif centering_type == 'adaptive':
            self.centering = AdaptiveCentering(output_dim)
        elif centering_type == 'dynamic':
            self.centering = DynamicCentering(output_dim)
        else:
            raise ValueError(f"Unknown centering type: {centering_type}")
    
    def forward(
        self,
        student_outputs: torch.Tensor,
        teacher_outputs: torch.Tensor,
        update_center: bool = True
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute DINO loss with centering
        
        Args:
            student_outputs: [N, output_dim] student predictions
            teacher_outputs: [M, output_dim] teacher predictions  
            update_center: Whether to update the center
            
        Returns:
            loss: Scalar loss value
            info: Dictionary with loss components and statistics
        """
        
        # Apply centering to teacher outputs
        teacher_centered = self.centering(teacher_outputs, update_center=update_center)
        
        # Apply temperature scaling and softmax
        student_probs = F.softmax(student_outputs / self.student_temperature, dim=1)
        teacher_probs = F.softmax(teacher_centered / self.teacher_temperature, dim=1)
        
        # Compute cross-entropy loss
        # Student learns from teacher: -sum(teacher_probs * log(student_probs))
        loss = -torch.sum(teacher_probs * torch.log(student_probs + 1e-8), dim=1).mean()
        
        # Gather statistics
        info = {
            'loss': loss.item(),
            'teacher_entropy': self._compute_entropy(teacher_probs),
            'student_entropy': self._compute_entropy(student_probs),
            'teacher_max_prob': torch.max(teacher_probs, dim=1)[0].mean().item(),
            'student_max_prob': torch.max(student_probs, dim=1)[0].mean().item(),
            **self.centering.get_center_stats()
        }
        
        return loss, info
    
    def _compute_entropy(self, probs: torch.Tensor) -> float:
        """Compute average entropy of probability distributions"""
        entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)
        return entropy.mean().item()


class MultiCropDINOLoss(nn.Module):
    """
    DINO loss for multi-crop training with centering
    """
    def __init__(
        self,
        output_dim: int,
        teacher_temperature: float = 0.04,
        student_temperature: float = 0.1,
        center_momentum: float = 0.9,
        lambda_local: float = 1.0
    ):
        super().__init__()
        
        self.lambda_local = lambda_local
        self.base_loss = DINOLossWithCentering(
            output_dim=output_dim,
            teacher_temperature=teacher_temperature,
            student_temperature=student_temperature,
            center_momentum=center_momentum
        )
    
    def forward(
        self,
        student_global: torch.Tensor,
        student_local: torch.Tensor,
        teacher_global: torch.Tensor,
        batch_size: int,
        update_center: bool = True
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute multi-crop DINO loss
        
        Args:
            student_global: [batch_size * 2, output_dim] global crop predictions
            student_local: [batch_size * num_local, output_dim] local crop predictions
            teacher_global: [batch_size * 2, output_dim] teacher global predictions
            batch_size: Original batch size
            update_center: Whether to update center
            
        Returns:
            total_loss: Combined loss across all crop combinations
            info: Detailed loss breakdown and statistics
        """
        
        total_loss = 0
        loss_count = 0
        all_info = {}
        
        # Global crop cross-predictions (student global learns from teacher global)
        for i in range(2):  # 2 global crops
            for j in range(2):  # 2 teacher crops
                if i != j:  # Don't learn from same crop
                    student_batch = student_global[i*batch_size:(i+1)*batch_size]
                    teacher_batch = teacher_global[j*batch_size:(j+1)*batch_size]
                    
                    loss, info = self.base_loss(
                        student_batch, teacher_batch, 
                        update_center=(update_center and loss_count == 0)  # Update center only once
                    )
                    
                    total_loss += loss
                    loss_count += 1
                    
                    if loss_count == 1:  # Store info from first loss computation
                        all_info.update({f'global_{k}': v for k, v in info.items()})
        
        # Local crop predictions (student local learns from teacher global)
        num_local_crops = student_local.shape[0] // batch_size
        local_loss_sum = 0
        local_loss_count = 0
        
        for i in range(num_local_crops):
            for j in range(2):  # 2 teacher global crops
                student_batch = student_local[i*batch_size:(i+1)*batch_size]
                teacher_batch = teacher_global[j*batch_size:(j+1)*batch_size]
                
                loss, info = self.base_loss(
                    student_batch, teacher_batch, 
                    update_center=False  # Don't update center for local crops
                )
                
                local_loss_sum += loss
                local_loss_count += 1
        
        # Add weighted local loss
        if local_loss_count > 0:
            local_loss_avg = local_loss_sum / local_loss_count
            total_loss += self.lambda_local * local_loss_avg
            all_info['local_loss'] = local_loss_avg.item()
        
        # Average global loss
        global_loss_avg = total_loss / loss_count if loss_count > 0 else 0
        all_info['global_loss'] = global_loss_avg.item() if hasattr(global_loss_avg, 'item') else global_loss_avg
        all_info['total_loss'] = total_loss.item()
        all_info['num_loss_terms'] = loss_count + (local_loss_count if local_loss_count > 0 else 0)
        
        return total_loss, all_info
```

### Step 3: Visualization and Analysis Tools

```python
# centering_analysis.py
import matplotlib.pyplot as plt
import numpy as np
import torch
from typing import List, Dict
import seaborn as sns

class CenteringAnalyzer:
    """
    Tools for analyzing centering behavior during training
    """
    
    def __init__(self):
        self.center_history = []
        self.entropy_history = []
        self.loss_history = []
        
    def log_step(self, centering_mechanism, loss_info):
        """Log information from a training step"""
        center_stats = centering_mechanism.get_center_stats()
        
        self.center_history.append({
            'step': len(self.center_history),
            'center_norm': center_stats['center_norm'],
            'center_mean': center_stats['center_mean'],
            'center_std': center_stats['center_std']
        })
        
        self.entropy_history.append({
            'step': len(self.entropy_history),
            'teacher_entropy': loss_info.get('teacher_entropy', 0),
            'student_entropy': loss_info.get('student_entropy', 0)
        })
        
        self.loss_history.append({
            'step': len(self.loss_history),
            'loss': loss_info.get('loss', 0),
            'teacher_max_prob': loss_info.get('teacher_max_prob', 0),
            'student_max_prob': loss_info.get('student_max_prob', 0)
        })
    
    def plot_center_evolution(self, save_path: str = None):
        """Plot how the center evolves during training"""
        if not self.center_history:
            print("No center history to plot")
            return
        
        steps = [item['step'] for item in self.center_history]
        center_norms = [item['center_norm'] for item in self.center_history]
        center_means = [item['center_mean'] for item in self.center_history]
        center_stds = [item['center_std'] for item in self.center_history]
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Center norm
        axes[0, 0].plot(steps, center_norms)
        axes[0, 0].set_title('Center L2 Norm Over Time')
        axes[0, 0].set_xlabel('Training Step')
        axes[0, 0].set_ylabel('L2 Norm')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Center mean
        axes[0, 1].plot(steps, center_means)
        axes[0, 1].set_title('Center Mean Over Time')
        axes[0, 1].set_xlabel('Training Step')
        axes[0, 1].set_ylabel('Mean Value')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Center std
        axes[1, 0].plot(steps, center_stds)
        axes[1, 0].set_title('Center Std Over Time')
        axes[1, 0].set_xlabel('Training Step')
        axes[1, 0].set_ylabel('Standard Deviation')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Combined view
        axes[1, 1].plot(steps, center_norms, label='Norm', alpha=0.7)
        axes[1, 1].plot(steps, np.abs(center_means), label='|Mean|', alpha=0.7)
        axes[1, 1].plot(steps, center_stds, label='Std', alpha=0.7)
        axes[1, 1].set_title('Center Statistics Combined')
        axes[1, 1].set_xlabel('Training Step')
        axes[1, 1].set_ylabel('Value')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        plt.show()
    
    def plot_entropy_evolution(self, save_path: str = None):
        """Plot entropy evolution during training"""
        if not self.entropy_history:
            print("No entropy history to plot")
            return
        
        steps = [item['step'] for item in self.entropy_history]
        teacher_entropy = [item['teacher_entropy'] for item in self.entropy_history]
        student_entropy = [item['student_entropy'] for item in self.entropy_history]
        
        plt.figure(figsize=(10, 6))
        plt.plot(steps, teacher_entropy, label='Teacher Entropy', alpha=0.8)
        plt.plot(steps, student_entropy, label='Student Entropy', alpha=0.8)
        plt.title('Entropy Evolution During Training')
        plt.xlabel('Training Step')
        plt.ylabel('Entropy (nats)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        plt.show()
    
    def analyze_mode_collapse(self, threshold: float = 0.1) -> dict:
        """Analyze potential mode collapse indicators"""
        if not self.entropy_history or not self.loss_history:
            return {}
        
        # Get recent values
        recent_teacher_entropy = [item['teacher_entropy'] for item in self.entropy_history[-100:]]
        recent_teacher_max_prob = [item['teacher_max_prob'] for item in self.loss_history[-100:]]
        recent_student_max_prob = [item['student_max_prob'] for item in self.loss_history[-100:]]
        
        analysis = {
            'avg_teacher_entropy': np.mean(recent_teacher_entropy),
            'avg_teacher_max_prob': np.mean(recent_teacher_max_prob),
            'avg_student_max_prob': np.mean(recent_student_max_prob),
            'entropy_trend': np.polyfit(range(len(recent_teacher_entropy)), recent_teacher_entropy, 1)[0],
            'potential_collapse': False
        }
        
        # Check for collapse indicators
        if (analysis['avg_teacher_entropy'] < threshold or 
            analysis['avg_teacher_max_prob'] > 0.9 or
            analysis['entropy_trend'] < -0.01):
            analysis['potential_collapse'] = True
        
        return analysis


def test_centering_mechanisms():
    """Test different centering mechanisms"""
    
    output_dim = 1000
    batch_size = 32
    num_steps = 1000
    
    # Create different centering mechanisms
    basic_centering = CenteringMechanism(output_dim, center_momentum=0.9)
    adaptive_centering = AdaptiveCentering(output_dim, learnable_momentum=True)
    dynamic_centering = DynamicCentering(output_dim)
    
    # Simulate training
    centers_basic = []
    centers_adaptive = []
    centers_dynamic = []
    
    for step in range(num_steps):
        # Simulate teacher outputs with drift
        base_output = torch.randn(batch_size, output_dim)
        drift = 0.01 * step * torch.ones(1, output_dim)
        teacher_outputs = base_output + drift
        
        # Apply different centering mechanisms
        basic_centered = basic_centering(teacher_outputs)
        adaptive_centered = adaptive_centering(teacher_outputs)
        dynamic_centered = dynamic_centering(teacher_outputs)
        
        # Store center norms
        centers_basic.append(torch.norm(basic_centering.center).item())
        centers_adaptive.append(torch.norm(adaptive_centering.center).item())
        centers_dynamic.append(torch.norm(dynamic_centering.center).item())
    
    # Plot comparison
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    plt.plot(centers_basic, label='Basic', alpha=0.8)
    plt.plot(centers_adaptive, label='Adaptive', alpha=0.8)
    plt.plot(centers_dynamic, label='Dynamic', alpha=0.8)
    plt.title('Center Norm Evolution')
    plt.xlabel('Training Step')
    plt.ylabel('Center L2 Norm')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Test collapse resistance
    plt.subplot(2, 2, 2)
    collapse_outputs = torch.ones(batch_size, output_dim) * 0.5  # Collapsed outputs
    
    basic_response = basic_centering(collapse_outputs)
    adaptive_response = adaptive_centering(collapse_outputs)
    dynamic_response = dynamic_centering(collapse_outputs)
    
    responses = [
        torch.norm(basic_response).item(),
        torch.norm(adaptive_response).item(), 
        torch.norm(dynamic_response).item()
    ]
    
    plt.bar(['Basic', 'Adaptive', 'Dynamic'], responses)
    plt.title('Response to Collapsed Outputs')
    plt.ylabel('Output Norm After Centering')
    
    # Test adaptation speed
    plt.subplot(2, 2, 3)
    sudden_shift = torch.randn(batch_size, output_dim) + 5.0  # Sudden distribution shift
    
    basic_adaptation = []
    adaptive_adaptation = []
    dynamic_adaptation = []
    
    for i in range(50):
        basic_out = basic_centering(sudden_shift)
        adaptive_out = adaptive_centering(sudden_shift)
        dynamic_out = dynamic_centering(sudden_shift)
        
        basic_adaptation.append(torch.norm(basic_out).item())
        adaptive_adaptation.append(torch.norm(adaptive_out).item())
        dynamic_adaptation.append(torch.norm(dynamic_out).item())
    
    plt.plot(basic_adaptation, label='Basic', alpha=0.8)
    plt.plot(adaptive_adaptation, label='Adaptive', alpha=0.8)
    plt.plot(dynamic_adaptation, label='Dynamic', alpha=0.8)
    plt.title('Adaptation to Distribution Shift')
    plt.xlabel('Steps After Shift')
    plt.ylabel('Centered Output Norm')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    print("Testing centering mechanisms...")
    test_centering_mechanisms()
```

## 🧪 Hands-on Exercise: Implement Your Centering Mechanism

### Exercise 1: Basic Centering Implementation

Implement a simple centering mechanism from scratch:

```python
# exercise1.py
import torch
import torch.nn as nn

class SimpleCentering(nn.Module):
    def __init__(self, output_dim, momentum=0.9):
        super().__init__()
        # TODO: Initialize center buffer and momentum
        pass
    
    def update_center(self, teacher_outputs):
        # TODO: Implement EMA update for center
        pass
    
    def apply_centering(self, teacher_outputs):
        # TODO: Subtract center from outputs
        pass
    
    def forward(self, teacher_outputs):
        # TODO: Combine update and centering
        pass

# Test your implementation
centering = SimpleCentering(1000)
outputs = torch.randn(32, 1000)
centered = centering(outputs)
print(f"Original mean: {outputs.mean().item():.4f}")
print(f"Centered mean: {centered.mean().item():.4f}")
```

### Exercise 2: Collapse Detection

Build a system to detect mode collapse:

```python
# exercise2.py
def detect_mode_collapse(teacher_outputs, threshold=0.1):
    """
    Detect if outputs show signs of mode collapse
    
    Args:
        teacher_outputs: [batch_size, output_dim] tensor
        threshold: Entropy threshold for collapse detection
    
    Returns:
        bool: True if collapse detected
    """
    # TODO: Implement collapse detection based on:
    # 1. Entropy of probability distributions
    # 2. Maximum probability values
    # 3. Standard deviation of outputs
    pass

# Test with different scenarios
normal_outputs = torch.randn(32, 1000)
collapsed_outputs = torch.ones(32, 1000) * 0.5

print(f"Normal outputs collapse: {detect_mode_collapse(normal_outputs)}")
print(f"Collapsed outputs collapse: {detect_mode_collapse(collapsed_outputs)}")
```

### Exercise 3: Momentum Sensitivity Analysis

Analyze how different momentum values affect centering:

```python
# exercise3.py
def analyze_momentum_sensitivity():
    """Analyze how momentum affects centering behavior"""
    
    momentums = [0.1, 0.5, 0.9, 0.99, 0.999]
    output_dim = 100
    num_steps = 500
    
    results = {}
    
    for momentum in momentums:
        # TODO: 
        # 1. Create centering mechanism with this momentum
        # 2. Simulate training steps with changing distributions
        # 3. Track center evolution and stability
        # 4. Store results
        pass
    
    # TODO: Plot comparison of different momentum values
    pass

# Run analysis
analyze_momentum_sensitivity()
```

## 🔍 Key Insights

### Why Centering Works
1. **Prevents Trivial Solutions**: Stops model from outputting constant values
2. **Maintains Diversity**: Keeps different samples from collapsing to same point
3. **Stabilizes Training**: Reduces variance in teacher targets
4. **Enables Self-Supervision**: Creates meaningful learning signal without labels

### Implementation Considerations
1. **Momentum Choice**: Too high = slow adaptation, too low = unstable
2. **Initialization**: Start with zero center, let it adapt naturally
3. **Update Frequency**: Update center every batch during training
4. **Gradient Isolation**: Never backpropagate through center updates

### Common Pitfalls
1. **Forgetting Updates**: Not updating center leads to poor performance
2. **Wrong Momentum**: Inappropriate momentum causes instability
3. **Gradient Leakage**: Accidentally backpropagating through center
4. **Inconsistent Application**: Not applying centering consistently

## 📝 Summary

In this lesson, you learned:

✅ **Mode Collapse Problem**: Understanding why self-supervised learning needs collapse prevention

✅ **Centering Mechanism**: How running mean subtraction prevents trivial solutions

✅ **Implementation Variants**: Basic, adaptive, and dynamic centering strategies

✅ **Integration**: How centering fits into the complete DINO loss function

✅ **Analysis Tools**: Methods to monitor and debug centering behavior

### Next Steps
In the next lesson, we'll implement temperature sharpening, which works alongside centering to create effective teacher-student dynamics.

## 🔗 Additional Resources

- [DINO Paper - Centering Analysis](https://arxiv.org/abs/2104.14294)
- [Mode Collapse in Self-Supervised Learning](https://arxiv.org/abs/2006.07733)
- [Understanding Knowledge Distillation](https://arxiv.org/abs/1503.02531)

---

**Next**: [Module 4, Lesson 2: Temperature Sharpening Implementation](module4_lesson2_temperature_sharpening.md)
