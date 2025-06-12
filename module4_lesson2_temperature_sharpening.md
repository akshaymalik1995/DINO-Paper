# Module 4, Lesson 2: Temperature Sharpening Implementation

## 🎯 Learning Objectives
By the end of this lesson, you will:
- Understand the role of temperature in probability sharpening
- Implement asymmetric temperature scaling for student and teacher
- Build temperature scheduling strategies
- Analyze the impact of temperature on gradient flow and training dynamics

## 📚 Theoretical Background

### Temperature in Probability Distributions

**Temperature scaling** controls the "sharpness" of probability distributions:

```python
P(x) = softmax(logits / τ)
```

Where `τ` (tau) is the temperature parameter:
- **τ → 0**: Very sharp distributions (one-hot)
- **τ = 1**: Standard softmax
- **τ → ∞**: Very smooth/uniform distributions

### DINO's Asymmetric Temperature Strategy

DINO uses **different temperatures** for student and teacher:

**Teacher Temperature** (`τ_t = 0.04-0.07`):
- **Low temperature** → Sharp, confident predictions
- **Stable targets** for student to learn from
- **Reduces noise** in teacher signals

**Student Temperature** (`τ_s = 0.1`):
- **Higher temperature** → Smoother predictions
- **Easier optimization** with less sharp gradients
- **Prevents overconfident predictions**

### Mathematical Analysis

**Gradient Flow Impact**:
```
∂L/∂logits ∝ (P_teacher - P_student) / τ_student
```

**Key Insights**:
1. **Lower student temperature** → Larger gradients
2. **Lower teacher temperature** → More confident targets
3. **Temperature ratio** controls learning dynamics

## 🛠️ Implementation

### Step 1: Temperature Scaling Functions

```python
# temperature_scaling.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from typing import Union, Optional, Tuple, Dict

class TemperatureScaling(nn.Module):
    """
    Temperature scaling for probability distributions
    """
    def __init__(
        self,
        initial_temperature: float = 1.0,
        learnable: bool = False,
        min_temperature: float = 0.01,
        max_temperature: float = 10.0
    ):
        super().__init__()
        
        self.min_temperature = min_temperature
        self.max_temperature = max_temperature
        
        if learnable:
            # Learnable temperature parameter
            self.log_temperature = nn.Parameter(torch.log(torch.tensor(initial_temperature)))
        else:
            # Fixed temperature
            self.register_buffer('log_temperature', torch.log(torch.tensor(initial_temperature)))
    
    @property
    def temperature(self) -> torch.Tensor:
        """Get current temperature with bounds"""
        temp = torch.exp(self.log_temperature)
        return torch.clamp(temp, self.min_temperature, self.max_temperature)
    
    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """Apply temperature scaling to logits"""
        return logits / self.temperature
    
    def get_probabilities(self, logits: torch.Tensor) -> torch.Tensor:
        """Get temperature-scaled probabilities"""
        scaled_logits = self.forward(logits)
        return F.softmax(scaled_logits, dim=-1)


class AsymmetricTemperatureScheduler:
    """
    Scheduler for asymmetric student-teacher temperatures
    """
    def __init__(
        self,
        student_temperature: float = 0.1,
        teacher_temp_min: float = 0.04,
        teacher_temp_max: float = 0.07,
        warmup_epochs: int = 30,
        total_epochs: int = 300,
        schedule_type: str = 'cosine'
    ):
        self.student_temperature = student_temperature
        self.teacher_temp_min = teacher_temp_min
        self.teacher_temp_max = teacher_temp_max
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.schedule_type = schedule_type
    
    def get_temperatures(self, epoch: int) -> Tuple[float, float]:
        """
        Get student and teacher temperatures for current epoch
        
        Returns:
            (student_temp, teacher_temp)
        """
        student_temp = self.student_temperature
        teacher_temp = self._get_teacher_temperature(epoch)
        
        return student_temp, teacher_temp
    
    def _get_teacher_temperature(self, epoch: int) -> float:
        """Get teacher temperature with scheduling"""
        if epoch < self.warmup_epochs:
            # Linear warmup from max to min
            progress = epoch / self.warmup_epochs
            teacher_temp = self.teacher_temp_max - progress * (
                self.teacher_temp_max - self.teacher_temp_min
            )
        else:
            # Post-warmup scheduling
            if self.schedule_type == 'constant':
                teacher_temp = self.teacher_temp_min
            elif self.schedule_type == 'cosine':
                # Cosine annealing
                progress = (epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
                teacher_temp = self.teacher_temp_min + 0.5 * (
                    self.teacher_temp_max - self.teacher_temp_min
                ) * (1 + np.cos(np.pi * progress))
            elif self.schedule_type == 'linear':
                # Linear decay
                progress = (epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
                teacher_temp = self.teacher_temp_max - progress * (
                    self.teacher_temp_max - self.teacher_temp_min
                )
            else:
                raise ValueError(f"Unknown schedule type: {self.schedule_type}")
        
        return max(teacher_temp, self.teacher_temp_min)


class AdaptiveTemperatureController:
    """
    Adaptive temperature controller based on training dynamics
    """
    def __init__(
        self,
        base_student_temp: float = 0.1,
        base_teacher_temp: float = 0.04,
        adaptation_rate: float = 0.01,
        target_entropy: float = 3.0,
        entropy_window: int = 100
    ):
        self.base_student_temp = base_student_temp
        self.base_teacher_temp = base_teacher_temp
        self.adaptation_rate = adaptation_rate
        self.target_entropy = target_entropy
        self.entropy_window = entropy_window
        
        # Track entropy history
        self.entropy_history = []
        self.current_student_temp = base_student_temp
        self.current_teacher_temp = base_teacher_temp
    
    def update_temperatures(
        self,
        student_probs: torch.Tensor,
        teacher_probs: torch.Tensor
    ) -> Tuple[float, float]:
        """
        Update temperatures based on current probability distributions
        
        Args:
            student_probs: Student probability distributions
            teacher_probs: Teacher probability distributions
            
        Returns:
            (updated_student_temp, updated_teacher_temp)
        """
        
        # Compute current entropy
        student_entropy = self._compute_entropy(student_probs)
        teacher_entropy = self._compute_entropy(teacher_probs)
        
        # Store in history
        self.entropy_history.append({
            'student_entropy': student_entropy,
            'teacher_entropy': teacher_entropy
        })
        
        # Keep only recent history
        if len(self.entropy_history) > self.entropy_window:
            self.entropy_history.pop(0)
        
        # Adapt student temperature based on entropy
        if len(self.entropy_history) >= 10:
            recent_student_entropy = np.mean([
                h['student_entropy'] for h in self.entropy_history[-10:]
            ])
            
            # If entropy too low (overconfident), increase temperature
            if recent_student_entropy < self.target_entropy * 0.8:
                self.current_student_temp *= (1 + self.adaptation_rate)
            # If entropy too high (underconfident), decrease temperature
            elif recent_student_entropy > self.target_entropy * 1.2:
                self.current_student_temp *= (1 - self.adaptation_rate)
            
            # Clamp temperature
            self.current_student_temp = np.clip(
                self.current_student_temp, 0.05, 0.5
            )
        
        # Adapt teacher temperature more conservatively
        if len(self.entropy_history) >= 50:
            recent_teacher_entropy = np.mean([
                h['teacher_entropy'] for h in self.entropy_history[-50:]
            ])
            
            # Adjust teacher temperature based on consistency
            entropy_std = np.std([
                h['teacher_entropy'] for h in self.entropy_history[-50:]
            ])
            
            if entropy_std > 0.5:  # High variance, increase temperature for stability
                self.current_teacher_temp *= (1 + self.adaptation_rate * 0.5)
            elif entropy_std < 0.1:  # Low variance, can decrease temperature
                self.current_teacher_temp *= (1 - self.adaptation_rate * 0.5)
            
            # Clamp teacher temperature
            self.current_teacher_temp = np.clip(
                self.current_teacher_temp, 0.01, 0.1
            )
        
        return self.current_student_temp, self.current_teacher_temp
    
    def _compute_entropy(self, probs: torch.Tensor) -> float:
        """Compute average entropy of probability distributions"""
        entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1)
        return entropy.mean().item()


class DINOTemperatureManager(nn.Module):
    """
    Complete temperature management for DINO training
    """
    def __init__(
        self,
        student_temperature: float = 0.1,
        teacher_temperature: float = 0.04,
        temperature_scheduling: bool = True,
        adaptive_temperature: bool = False,
        **scheduler_kwargs
    ):
        super().__init__()
        
        # Base temperatures
        self.base_student_temp = student_temperature
        self.base_teacher_temp = teacher_temperature
        
        # Initialize scheduler
        if temperature_scheduling:
            self.scheduler = AsymmetricTemperatureScheduler(
                student_temperature=student_temperature,
                teacher_temp_min=teacher_temperature,
                **scheduler_kwargs
            )
        else:
            self.scheduler = None
        
        # Initialize adaptive controller
        if adaptive_temperature:
            self.adaptive_controller = AdaptiveTemperatureController(
                base_student_temp=student_temperature,
                base_teacher_temp=teacher_temperature
            )
        else:
            self.adaptive_controller = None
        
        # Current temperatures
        self.current_student_temp = student_temperature
        self.current_teacher_temp = teacher_temperature
    
    def update_temperatures(
        self,
        epoch: int,
        student_probs: Optional[torch.Tensor] = None,
        teacher_probs: Optional[torch.Tensor] = None
    ):
        """Update temperatures based on epoch and/or training dynamics"""
        
        # Scheduled temperature update
        if self.scheduler is not None:
            student_temp, teacher_temp = self.scheduler.get_temperatures(epoch)
            self.current_student_temp = student_temp
            self.current_teacher_temp = teacher_temp
        
        # Adaptive temperature update
        if (self.adaptive_controller is not None and 
            student_probs is not None and teacher_probs is not None):
            
            adaptive_student, adaptive_teacher = self.adaptive_controller.update_temperatures(
                student_probs, teacher_probs
            )
            
            # Combine scheduled and adaptive temperatures
            if self.scheduler is not None:
                # Use adaptive as modulation of scheduled
                self.current_student_temp = student_temp * (adaptive_student / self.base_student_temp)
                self.current_teacher_temp = teacher_temp * (adaptive_teacher / self.base_teacher_temp)
            else:
                # Use adaptive directly
                self.current_student_temp = adaptive_student
                self.current_teacher_temp = adaptive_teacher
    
    def apply_temperature_scaling(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply temperature scaling to student and teacher logits
        
        Returns:
            (student_probs, teacher_probs)
        """
        
        # Apply temperature scaling
        student_scaled = student_logits / self.current_student_temp
        teacher_scaled = teacher_logits / self.current_teacher_temp
        
        # Convert to probabilities
        student_probs = F.softmax(student_scaled, dim=-1)
        teacher_probs = F.softmax(teacher_scaled, dim=-1)
        
        return student_probs, teacher_probs
    
    def get_temperature_info(self) -> Dict[str, float]:
        """Get current temperature information"""
        return {
            'student_temperature': self.current_student_temp,
            'teacher_temperature': self.current_teacher_temp,
            'temperature_ratio': self.current_student_temp / self.current_teacher_temp
        }
```

### Step 2: Integration with DINO Loss

```python
# dino_loss_with_temperature.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Optional

class DINOLossWithTemperature(nn.Module):
    """
    Complete DINO loss with centering and temperature scaling
    """
    def __init__(
        self,
        output_dim: int,
        student_temperature: float = 0.1,
        teacher_temperature: float = 0.04,
        center_momentum: float = 0.9,
        temperature_scheduling: bool = True,
        adaptive_temperature: bool = False,
        **temperature_kwargs
    ):
        super().__init__()
        
        # Initialize centering (from previous lesson)
        from .centering import CenteringMechanism
        self.centering = CenteringMechanism(output_dim, center_momentum)
        
        # Initialize temperature manager
        self.temperature_manager = DINOTemperatureManager(
            student_temperature=student_temperature,
            teacher_temperature=teacher_temperature,
            temperature_scheduling=temperature_scheduling,
            adaptive_temperature=adaptive_temperature,
            **temperature_kwargs
        )
        
        # Track loss components
        self.loss_components = {}
    
    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        epoch: int,
        update_center: bool = True,
        return_probs: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict]]:
        """
        Compute DINO loss with temperature scaling and centering
        
        Args:
            student_logits: Student network outputs
            teacher_logits: Teacher network outputs
            epoch: Current training epoch
            update_center: Whether to update center
            return_probs: Whether to return probability distributions
            
        Returns:
            loss or (loss, info_dict)
        """
        
        # Apply centering to teacher logits
        teacher_centered = self.centering(teacher_logits, update_center=update_center)
        
        # Apply temperature scaling
        student_probs, teacher_probs = self.temperature_manager.apply_temperature_scaling(
            student_logits, teacher_centered
        )
        
        # Update temperatures based on current distributions (if adaptive)
        self.temperature_manager.update_temperatures(
            epoch=epoch,
            student_probs=student_probs,
            teacher_probs=teacher_probs
        )
        
        # Compute cross-entropy loss
        loss = -torch.sum(teacher_probs * torch.log(student_probs + 1e-8), dim=-1).mean()
        
        # Gather detailed information
        info = self._compute_detailed_info(student_probs, teacher_probs, loss)
        
        if return_probs:
            return loss, info, student_probs, teacher_probs
        else:
            return loss, info
    
    def _compute_detailed_info(
        self,
        student_probs: torch.Tensor,
        teacher_probs: torch.Tensor,
        loss: torch.Tensor
    ) -> Dict:
        """Compute detailed information about the loss and distributions"""
        
        with torch.no_grad():
            # Basic loss info
            info = {
                'loss': loss.item(),
                **self.temperature_manager.get_temperature_info(),
                **self.centering.get_center_stats()
            }
            
            # Entropy analysis
            student_entropy = -torch.sum(student_probs * torch.log(student_probs + 1e-8), dim=-1)
            teacher_entropy = -torch.sum(teacher_probs * torch.log(teacher_probs + 1e-8), dim=-1)
            
            info.update({
                'student_entropy_mean': student_entropy.mean().item(),
                'student_entropy_std': student_entropy.std().item(),
                'teacher_entropy_mean': teacher_entropy.mean().item(),
                'teacher_entropy_std': teacher_entropy.std().item(),
            })
            
            # Confidence analysis
            student_max_probs = torch.max(student_probs, dim=-1)[0]
            teacher_max_probs = torch.max(teacher_probs, dim=-1)[0]
            
            info.update({
                'student_confidence_mean': student_max_probs.mean().item(),
                'student_confidence_std': student_max_probs.std().item(),
                'teacher_confidence_mean': teacher_max_probs.mean().item(),
                'teacher_confidence_std': teacher_max_probs.std().item(),
            })
            
            # Distribution similarity
            kl_div = F.kl_div(
                torch.log(student_probs + 1e-8), teacher_probs, reduction='none'
            ).sum(dim=-1)
            
            js_div = self._compute_js_divergence(student_probs, teacher_probs)
            
            info.update({
                'kl_divergence_mean': kl_div.mean().item(),
                'kl_divergence_std': kl_div.std().item(),
                'js_divergence_mean': js_div.mean().item(),
                'js_divergence_std': js_div.std().item(),
            })
        
        return info
    
    def _compute_js_divergence(self, p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        """Compute Jensen-Shannon divergence between two distributions"""
        m = 0.5 * (p + q)
        js_div = 0.5 * F.kl_div(torch.log(p + 1e-8), m, reduction='none').sum(dim=-1) + \
                 0.5 * F.kl_div(torch.log(q + 1e-8), m, reduction='none').sum(dim=-1)
        return js_div


class MultiCropDINOLossWithTemperature(nn.Module):
    """
    Multi-crop DINO loss with temperature management
    """
    def __init__(
        self,
        output_dim: int,
        student_temperature: float = 0.1,
        teacher_temperature: float = 0.04,
        lambda_local: float = 1.0,
        **loss_kwargs
    ):
        super().__init__()
        
        self.lambda_local = lambda_local
        self.base_loss = DINOLossWithTemperature(
            output_dim=output_dim,
            student_temperature=student_temperature,
            teacher_temperature=teacher_temperature,
            **loss_kwargs
        )
    
    def forward(
        self,
        student_global: torch.Tensor,
        student_local: torch.Tensor,
        teacher_global: torch.Tensor,
        batch_size: int,
        epoch: int,
        update_center: bool = True
    ) -> Tuple[torch.Tensor, Dict]:
        """Compute multi-crop DINO loss with temperature scaling"""
        
        total_loss = 0
        loss_count = 0
        combined_info = {}
        
        # Global-to-global losses
        for i in range(2):
            for j in range(2):
                if i != j:
                    student_batch = student_global[i*batch_size:(i+1)*batch_size]
                    teacher_batch = teacher_global[j*batch_size:(j+1)*batch_size]
                    
                    loss, info = self.base_loss(
                        student_batch, teacher_batch, epoch,
                        update_center=(update_center and loss_count == 0)
                    )
                    
                    total_loss += loss
                    loss_count += 1
                    
                    if loss_count == 1:
                        combined_info.update({f'global_{k}': v for k, v in info.items()})
        
        # Local-to-global losses
        num_local_crops = student_local.shape[0] // batch_size
        local_loss_sum = 0
        local_loss_count = 0
        
        for i in range(num_local_crops):
            for j in range(2):
                student_batch = student_local[i*batch_size:(i+1)*batch_size]
                teacher_batch = teacher_global[j*batch_size:(j+1)*batch_size]
                
                loss, info = self.base_loss(
                    student_batch, teacher_batch, epoch,
                    update_center=False
                )
                
                local_loss_sum += loss
                local_loss_count += 1
        
        # Combine losses
        if local_loss_count > 0:
            local_loss_avg = local_loss_sum / local_loss_count
            total_loss += self.lambda_local * local_loss_avg
            combined_info['local_loss'] = local_loss_avg.item()
        
        combined_info['total_loss'] = total_loss.item()
        combined_info['global_loss'] = (total_loss.item() - combined_info.get('local_loss', 0))
        
        return total_loss, combined_info
```

### Step 3: Temperature Analysis and Visualization

```python
# temperature_analysis.py
import matplotlib.pyplot as plt
import numpy as np
import torch
import seaborn as sns
from typing import List, Dict

class TemperatureAnalyzer:
    """
    Tools for analyzing temperature effects during training
    """
    
    def __init__(self):
        self.temperature_history = []
        self.entropy_history = []
        self.confidence_history = []
        
    def log_step(self, temp_info: Dict, entropy_info: Dict, confidence_info: Dict):
        """Log temperature and distribution information"""
        self.temperature_history.append({
            'step': len(self.temperature_history),
            **temp_info
        })
        
        self.entropy_history.append({
            'step': len(self.entropy_history),
            **entropy_info
        })
        
        self.confidence_history.append({
            'step': len(self.confidence_history),
            **confidence_info
        })
    
    def plot_temperature_evolution(self, save_path: str = None):
        """Plot temperature evolution over training"""
        if not self.temperature_history:
            return
        
        steps = [h['step'] for h in self.temperature_history]
        student_temps = [h['student_temperature'] for h in self.temperature_history]
        teacher_temps = [h['teacher_temperature'] for h in self.temperature_history]
        temp_ratios = [h['temperature_ratio'] for h in self.temperature_history]
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Temperature evolution
        axes[0, 0].plot(steps, student_temps, label='Student', alpha=0.8)
        axes[0, 0].plot(steps, teacher_temps, label='Teacher', alpha=0.8)
        axes[0, 0].set_title('Temperature Evolution')
        axes[0, 0].set_xlabel('Training Step')
        axes[0, 0].set_ylabel('Temperature')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Temperature ratio
        axes[0, 1].plot(steps, temp_ratios, color='purple', alpha=0.8)
        axes[0, 1].set_title('Student/Teacher Temperature Ratio')
        axes[0, 1].set_xlabel('Training Step')
        axes[0, 1].set_ylabel('Ratio')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Entropy correlation
        if self.entropy_history:
            student_entropy = [h.get('student_entropy_mean', 0) for h in self.entropy_history]
            teacher_entropy = [h.get('teacher_entropy_mean', 0) for h in self.entropy_history]
            
            axes[1, 0].plot(steps[:len(student_entropy)], student_entropy, 
                           label='Student Entropy', alpha=0.8)
            axes[1, 0].plot(steps[:len(teacher_entropy)], teacher_entropy, 
                           label='Teacher Entropy', alpha=0.8)
            axes[1, 0].set_title('Entropy vs Temperature')
            axes[1, 0].set_xlabel('Training Step')
            axes[1, 0].set_ylabel('Entropy')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        
        # Confidence correlation
        if self.confidence_history:
            student_conf = [h.get('student_confidence_mean', 0) for h in self.confidence_history]
            teacher_conf = [h.get('teacher_confidence_mean', 0) for h in self.confidence_history]
            
            axes[1, 1].plot(steps[:len(student_conf)], student_conf, 
                           label='Student Confidence', alpha=0.8)
            axes[1, 1].plot(steps[:len(teacher_conf)], teacher_conf, 
                           label='Teacher Confidence', alpha=0.8)
            axes[1, 1].set_title('Confidence vs Temperature')
            axes[1, 1].set_xlabel('Training Step')
            axes[1, 1].set_ylabel('Max Probability')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        plt.show()
    
    def analyze_temperature_sensitivity(
        self,
        logits: torch.Tensor,
        temperature_range: List[float] = None
    ) -> Dict:
        """Analyze sensitivity to different temperature values"""
        
        if temperature_range is None:
            temperature_range = [0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0]
        
        results = {}
        
        for temp in temperature_range:
            # Apply temperature scaling
            scaled_logits = logits / temp
            probs = F.softmax(scaled_logits, dim=-1)
            
            # Compute metrics
            entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1)
            max_probs = torch.max(probs, dim=-1)[0]
            
            results[temp] = {
                'entropy_mean': entropy.mean().item(),
                'entropy_std': entropy.std().item(),
                'confidence_mean': max_probs.mean().item(),
                'confidence_std': max_probs.std().item(),
                'sparsity': (probs > 0.01).float().mean().item()
            }
        
        return results
    
    def plot_temperature_sensitivity(
        self,
        sensitivity_results: Dict,
        save_path: str = None
    ):
        """Plot temperature sensitivity analysis"""
        
        temperatures = list(sensitivity_results.keys())
        entropies = [sensitivity_results[t]['entropy_mean'] for t in temperatures]
        confidences = [sensitivity_results[t]['confidence_mean'] for t in temperatures]
        sparsities = [sensitivity_results[t]['sparsity'] for t in temperatures]
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Entropy vs Temperature
        axes[0].semilogx(temperatures, entropies, 'o-', alpha=0.8)
        axes[0].set_title('Entropy vs Temperature')
        axes[0].set_xlabel('Temperature')
        axes[0].set_ylabel('Average Entropy')
        axes[0].grid(True, alpha=0.3)
        
        # Confidence vs Temperature
        axes[1].semilogx(temperatures, confidences, 'o-', color='orange', alpha=0.8)
        axes[1].set_title('Confidence vs Temperature')
        axes[1].set_xlabel('Temperature')
        axes[1].set_ylabel('Average Max Probability')
        axes[1].grid(True, alpha=0.3)
        
        # Sparsity vs Temperature
        axes[2].semilogx(temperatures, sparsities, 'o-', color='green', alpha=0.8)
        axes[2].set_title('Sparsity vs Temperature')
        axes[2].set_xlabel('Temperature')
        axes[2].set_ylabel('Fraction of Non-zero Probabilities')
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        plt.show()


def test_temperature_effects():
    """Test the effects of different temperature settings"""
    
    # Create synthetic logits
    batch_size = 1000
    num_classes = 10000
    
    # Different types of logit distributions
    uniform_logits = torch.randn(batch_size, num_classes) * 0.1
    peaked_logits = torch.zeros(batch_size, num_classes)
    peaked_logits[:, 0] = 5.0  # Strong peak at first class
    
    mixed_logits = torch.randn(batch_size, num_classes)
    mixed_logits[:batch_size//2, :10] += 3.0  # Some samples have clear preferences
    
    logit_types = {
        'uniform': uniform_logits,
        'peaked': peaked_logits,
        'mixed': mixed_logits
    }
    
    analyzer = TemperatureAnalyzer()
    
    # Test different temperatures
    for name, logits in logit_types.items():
        print(f"\nAnalyzing {name} logits...")
        
        sensitivity = analyzer.analyze_temperature_sensitivity(logits)
        analyzer.plot_temperature_sensitivity(sensitivity)
        
        # Print key insights
        low_temp_entropy = sensitivity[0.05]['entropy_mean']
        high_temp_entropy = sensitivity[1.0]['entropy_mean']
        
        print(f"Entropy at T=0.05: {low_temp_entropy:.3f}")
        print(f"Entropy at T=1.0: {high_temp_entropy:.3f}")
        print(f"Entropy ratio: {high_temp_entropy/low_temp_entropy:.2f}")


def demonstrate_asymmetric_temperatures():
    """Demonstrate the effect of asymmetric student-teacher temperatures"""
    
    # Create teacher logits (confident)
    teacher_logits = torch.zeros(100, 1000)
    teacher_logits[:, :10] = torch.randn(100, 10) * 2 + 3  # Confident about first 10 classes
    
    # Create student logits (less confident)
    student_logits = torch.randn(100, 1000) * 1.5
    
    # Test different temperature combinations
    teacher_temps = [0.04, 0.07, 0.1]
    student_temps = [0.1, 0.2, 0.3]
    
    fig, axes = plt.subplots(len(teacher_temps), len(student_temps), 
                            figsize=(15, 12))
    
    for i, t_temp in enumerate(teacher_temps):
        for j, s_temp in enumerate(student_temps):
            # Apply temperature scaling
            teacher_probs = F.softmax(teacher_logits / t_temp, dim=-1)
            student_probs = F.softmax(student_logits / s_temp, dim=-1)
            
            # Compute loss
            loss = -torch.sum(teacher_probs * torch.log(student_probs + 1e-8), dim=-1)
            
            # Plot loss distribution
            axes[i, j].hist(loss.detach().numpy(), bins=20, alpha=0.7)
            axes[i, j].set_title(f'T_teacher={t_temp}, T_student={s_temp}')
            axes[i, j].set_xlabel('Loss')
            axes[i, j].set_ylabel('Frequency')
            
            # Add statistics
            axes[i, j].axvline(loss.mean(), color='red', linestyle='--', 
                              label=f'Mean: {loss.mean():.3f}')
            axes[i, j].legend()
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    print("Testing temperature effects...")
    test_temperature_effects()
    
    print("\nDemonstrating asymmetric temperatures...")
    demonstrate_asymmetric_temperatures()
```

## 🧪 Hands-on Exercise: Implement Temperature Sharpening

### Exercise 1: Basic Temperature Scaling

Implement temperature scaling from scratch:

```python
# exercise1.py
import torch
import torch.nn.functional as F

def apply_temperature_scaling(logits, temperature):
    """
    Apply temperature scaling to logits
    
    Args:
        logits: [batch_size, num_classes] logits
        temperature: scalar temperature value
    
    Returns:
        probabilities: [batch_size, num_classes] softmax probabilities
    """
    # TODO: Implement temperature scaling
    pass

def compare_temperatures(logits, temperatures):
    """Compare the effect of different temperatures"""
    # TODO: Apply different temperatures and compare:
    # 1. Entropy of resulting distributions
    # 2. Maximum probabilities
    # 3. Effective number of classes (entropy exponential)
    pass

# Test with sample data
logits = torch.randn(10, 100) * 2
temperatures = [0.1, 0.5, 1.0, 2.0, 5.0]
compare_temperatures(logits, temperatures)
```

### Exercise 2: Temperature Scheduling

Implement different temperature scheduling strategies:

```python
# exercise2.py
def linear_temperature_schedule(epoch, max_epochs, min_temp, max_temp):
    """Linear temperature schedule"""
    # TODO: Implement linear interpolation between min and max temperature
    pass

def cosine_temperature_schedule(epoch, max_epochs, min_temp, max_temp):
    """Cosine temperature schedule"""
    # TODO: Implement cosine annealing schedule
    pass

def exponential_temperature_schedule(epoch, max_epochs, min_temp, max_temp, decay_rate):
    """Exponential temperature schedule"""
    # TODO: Implement exponential decay schedule
    pass

# Plot and compare different schedules
epochs = range(100)
schedules = {}
for epoch in epochs:
    schedules[epoch] = {
        'linear': linear_temperature_schedule(epoch, 100, 0.04, 0.1),
        'cosine': cosine_temperature_schedule(epoch, 100, 0.04, 0.1),
        'exponential': exponential_temperature_schedule(epoch, 100, 0.04, 0.1, 0.95)
    }

# TODO: Plot comparison
```

### Exercise 3: Gradient Analysis

Analyze how temperature affects gradient flow:

```python
# exercise3.py
def analyze_temperature_gradients(student_logits, teacher_logits, temperatures):
    """
    Analyze how temperature affects gradients
    
    Args:
        student_logits: Student network outputs
        teacher_logits: Teacher network outputs (detached)
        temperatures: List of temperature values to test
    
    Returns:
        gradient_analysis: Dict with gradient statistics for each temperature
    """
    results = {}
    
    for temp in temperatures:
        # TODO: 
        # 1. Apply temperature scaling
        # 2. Compute DINO loss
        # 3. Compute gradients w.r.t. student_logits
        # 4. Analyze gradient magnitude and distribution
        pass
    
    return results

# Test gradient analysis
student_logits = torch.randn(32, 1000, requires_grad=True)
teacher_logits = torch.randn(32, 1000).detach()
temperatures = [0.05, 0.1, 0.2, 0.5, 1.0]

gradient_analysis = analyze_temperature_gradients(student_logits, teacher_logits, temperatures)
```

## 🔍 Key Insights

### Temperature Effects on Learning
1. **Lower Teacher Temperature**: More confident, peaked teacher distributions
2. **Higher Student Temperature**: Smoother student distributions, easier optimization
3. **Temperature Ratio**: Controls the "hardness" of the knowledge transfer
4. **Gradient Scaling**: Temperature directly affects gradient magnitudes

### Optimal Temperature Selection
1. **Teacher Temperature**: 0.04-0.07 provides good balance of confidence and stability
2. **Student Temperature**: 0.1 allows smooth optimization while maintaining discrimination
3. **Scheduling**: Gradual reduction can improve training stability
4. **Adaptation**: Dynamic adjustment based on training dynamics can help

### Common Issues
1. **Too Low Temperature**: Can cause gradient explosion or numerical instability
2. **Too High Temperature**: Makes targets too uniform, reduces learning signal
3. **Wrong Ratio**: Imbalanced student/teacher temperatures hurt performance
4. **Fixed Temperatures**: May not adapt to changing training dynamics

## 📝 Summary

In this lesson, you learned:

✅ **Temperature Scaling Theory**: How temperature controls probability distribution sharpness

✅ **Asymmetric Temperature Strategy**: Different temperatures for student and teacher networks

✅ **Temperature Scheduling**: Dynamic adjustment of temperatures during training

✅ **Gradient Flow Analysis**: How temperature affects optimization dynamics

✅ **Integration**: Combining temperature scaling with centering in complete DINO loss

### Next Steps
In the next lesson, we'll combine centering and temperature scaling into the complete DINO loss function with all components working together.

## 🔗 Additional Resources

- [Temperature Scaling in Neural Networks](https://arxiv.org/abs/1706.04599)
- [Knowledge Distillation with Temperature](https://arxiv.org/abs/1503.02531)
- [DINO Paper - Temperature Analysis](https://arxiv.org/abs/2104.14294)

---

**Next**: [Module 4, Lesson 3: Complete DINO Loss Function](module4_lesson3_complete_dino_loss.md)
