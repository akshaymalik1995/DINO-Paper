# Module 3, Lesson 1: Implementing Student and Teacher Networks

## 🎯 Learning Objectives
By the end of this lesson, you will:
- Understand the student-teacher architecture in DINO
- Implement the Exponential Moving Average (EMA) update mechanism
- Build a complete student-teacher network pair
- Understand weight synchronization between networks

## 📚 Theoretical Background

### Student-Teacher Architecture in DINO

DINO employs a **student-teacher paradigm** where:
- **Student network** learns from multiple augmented views
- **Teacher network** provides stable targets
- Both networks share identical architecture but different weights
- Teacher weights are updated via Exponential Moving Average (EMA) of student weights

### Why This Works

1. **Stability**: Teacher provides stable targets while student adapts
2. **Self-supervision**: No external labels needed
3. **Momentum**: EMA prevents teacher from changing too rapidly
4. **Asymmetry**: Different augmentations for student/teacher create learning signal

### Mathematical Foundation

**EMA Update Rule:**
```
θ_teacher ← τ * θ_teacher + (1 - τ) * θ_student
```

Where:
- `τ` (tau) is the momentum coefficient (typically 0.996 → 0.999)
- `θ` represents network parameters

## 🛠️ Implementation

### Step 1: Base Network Architecture

```python
# student_teacher.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any
import copy
import math

class DINONetwork(nn.Module):
    """
    Base DINO network combining backbone + projection head
    """
    def __init__(
        self,
        backbone: nn.Module,
        projection_dim: int = 65536,
        hidden_dim: int = 2048,
        bottleneck_dim: int = 256,
        use_bn: bool = False,
        norm_last_layer: bool = True
    ):
        super().__init__()
        self.backbone = backbone
        self.projection_dim = projection_dim
        
        # Get backbone output dimension
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224)
            backbone_output = self.backbone(dummy_input)
            if isinstance(backbone_output, tuple):
                backbone_output = backbone_output[0]
            backbone_dim = backbone_output.shape[-1]
        
        # Build projection head
        self.projection_head = self._build_projection_head(
            backbone_dim, hidden_dim, bottleneck_dim, projection_dim, use_bn, norm_last_layer
        )
        
    def _build_projection_head(
        self, 
        backbone_dim: int, 
        hidden_dim: int, 
        bottleneck_dim: int, 
        projection_dim: int,
        use_bn: bool,
        norm_last_layer: bool
    ) -> nn.Module:
        """Build the projection head (MLP)"""
        layers = []
        
        # First layer: backbone_dim -> hidden_dim
        layers.append(nn.Linear(backbone_dim, hidden_dim))
        if use_bn:
            layers.append(nn.BatchNorm1d(hidden_dim))
        layers.append(nn.GELU())
        
        # Second layer: hidden_dim -> hidden_dim
        layers.append(nn.Linear(hidden_dim, hidden_dim))
        if use_bn:
            layers.append(nn.BatchNorm1d(hidden_dim))
        layers.append(nn.GELU())
        
        # Bottleneck layer: hidden_dim -> bottleneck_dim
        layers.append(nn.Linear(hidden_dim, bottleneck_dim))
        
        # Final projection: bottleneck_dim -> projection_dim
        last_layer = nn.Linear(bottleneck_dim, projection_dim, bias=False)
        
        # Initialize last layer with small weights
        if norm_last_layer:
            last_layer.weight.data.normal_(0, 0.01)
            last_layer.weight.data = F.normalize(last_layer.weight.data, dim=1)
        
        layers.append(last_layer)
        
        return nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through backbone + projection head"""
        # Extract features from backbone
        features = self.backbone(x)
        if isinstance(features, tuple):
            features = features[0]  # Take first output if tuple
        
        # Apply projection head
        projected = self.projection_head(features)
        
        # L2 normalize
        projected = F.normalize(projected, dim=1, p=2)
        
        return projected


class StudentTeacherWrapper(nn.Module):
    """
    Wrapper for student-teacher architecture with EMA updates
    """
    def __init__(
        self,
        backbone: nn.Module,
        projection_dim: int = 65536,
        hidden_dim: int = 2048,
        bottleneck_dim: int = 256,
        use_bn: bool = False,
        norm_last_layer: bool = True,
        momentum: float = 0.996,
        warmup_teacher_epochs: int = 30,
        total_epochs: int = 100
    ):
        super().__init__()
        
        # Create student network
        self.student = DINONetwork(
            backbone=backbone,
            projection_dim=projection_dim,
            hidden_dim=hidden_dim,
            bottleneck_dim=bottleneck_dim,
            use_bn=use_bn,
            norm_last_layer=norm_last_layer
        )
        
        # Create teacher network (copy of student)
        self.teacher = DINONetwork(
            backbone=copy.deepcopy(backbone),
            projection_dim=projection_dim,
            hidden_dim=hidden_dim,
            bottleneck_dim=bottleneck_dim,
            use_bn=use_bn,
            norm_last_layer=False  # Teacher doesn't need normalized last layer
        )
        
        # Disable gradients for teacher
        for param in self.teacher.parameters():
            param.requires_grad = False
        
        # EMA parameters
        self.momentum = momentum
        self.warmup_teacher_epochs = warmup_teacher_epochs
        self.total_epochs = total_epochs
        self.current_epoch = 0
        
        # Initialize teacher with student weights
        self._copy_student_to_teacher()
    
    def _copy_student_to_teacher(self):
        """Copy student weights to teacher (used for initialization)"""
        with torch.no_grad():
            for student_param, teacher_param in zip(
                self.student.parameters(), self.teacher.parameters()
            ):
                teacher_param.data.copy_(student_param.data)
    
    def _get_momentum_schedule(self, epoch: int) -> float:
        """
        Momentum schedule: starts from 0.996 and increases to 0.999
        """
        if epoch < self.warmup_teacher_epochs:
            # Linear warmup
            base_momentum = 0.996
            target_momentum = 0.999
            progress = epoch / self.warmup_teacher_epochs
            return base_momentum + (target_momentum - base_momentum) * progress
        else:
            return 0.999
    
    def update_teacher(self, epoch: int):
        """
        Update teacher weights using EMA of student weights
        """
        self.current_epoch = epoch
        momentum = self._get_momentum_schedule(epoch)
        
        with torch.no_grad():
            for student_param, teacher_param in zip(
                self.student.parameters(), self.teacher.parameters()
            ):
                teacher_param.data.mul_(momentum).add_(
                    student_param.data, alpha=1 - momentum
                )
    
    def forward_student(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through student network"""
        return self.student(x)
    
    def forward_teacher(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through teacher network"""
        with torch.no_grad():
            return self.teacher(x)
    
    def get_student_parameters(self):
        """Get student parameters for optimizer"""
        return self.student.parameters()
```

### Step 2: Weight Synchronization Utilities

```python
# sync_utils.py
import torch
import torch.distributed as dist
from typing import List

def sync_batch_norm(model: torch.nn.Module):
    """
    Convert BatchNorm layers to SyncBatchNorm for distributed training
    """
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    return model

def all_reduce_mean(tensor: torch.Tensor) -> torch.Tensor:
    """
    All-reduce tensor across all processes (for distributed training)
    """
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        tensor = tensor.clone()
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        tensor /= dist.get_world_size()
    return tensor

class EMAUpdater:
    """
    Utility class for EMA updates with momentum scheduling
    """
    def __init__(
        self,
        base_momentum: float = 0.996,
        final_momentum: float = 0.999,
        warmup_epochs: int = 30
    ):
        self.base_momentum = base_momentum
        self.final_momentum = final_momentum
        self.warmup_epochs = warmup_epochs
    
    def get_momentum(self, epoch: int) -> float:
        """Get momentum value for current epoch"""
        if epoch < self.warmup_epochs:
            # Cosine schedule during warmup
            progress = epoch / self.warmup_epochs
            momentum = self.base_momentum + (self.final_momentum - self.base_momentum) * (
                1 + math.cos(math.pi * progress)
            ) / 2
        else:
            momentum = self.final_momentum
        
        return momentum
    
    def update_parameters(
        self,
        student_params: List[torch.Tensor],
        teacher_params: List[torch.Tensor],
        momentum: float
    ):
        """Update teacher parameters using EMA"""
        with torch.no_grad():
            for student_param, teacher_param in zip(student_params, teacher_params):
                teacher_param.data.mul_(momentum).add_(
                    student_param.data, alpha=1 - momentum
                )
```

### Step 3: Testing the Implementation

```python
# test_student_teacher.py
import torch
import torch.nn as nn
from torchvision.models import resnet50
import matplotlib.pyplot as plt
import numpy as np

def test_student_teacher_architecture():
    """Test the student-teacher implementation"""
    
    # Create a simple backbone
    backbone = nn.Sequential(
        nn.Conv2d(3, 64, 7, 2, 3),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(64, 512)
    )
    
    # Create student-teacher wrapper
    model = StudentTeacherWrapper(
        backbone=backbone,
        projection_dim=1024,
        hidden_dim=512,
        bottleneck_dim=128
    )
    
    # Test forward passes
    batch_size = 4
    x = torch.randn(batch_size, 3, 224, 224)
    
    # Student forward
    student_out = model.forward_student(x)
    print(f"Student output shape: {student_out.shape}")
    print(f"Student output norm: {torch.norm(student_out, dim=1)}")
    
    # Teacher forward
    teacher_out = model.forward_teacher(x)
    print(f"Teacher output shape: {teacher_out.shape}")
    print(f"Teacher output norm: {torch.norm(teacher_out, dim=1)}")
    
    # Test EMA update
    print("\nTesting EMA updates...")
    initial_teacher_param = list(model.teacher.parameters())[0].clone()
    
    # Simulate training step
    student_out = model.forward_student(x)
    loss = torch.mean(student_out)  # Dummy loss
    loss.backward()
    
    # Update teacher
    model.update_teacher(epoch=0)
    
    updated_teacher_param = list(model.teacher.parameters())[0]
    
    print(f"Teacher parameter changed: {not torch.equal(initial_teacher_param, updated_teacher_param)}")
    
    return model

def visualize_momentum_schedule():
    """Visualize the momentum schedule"""
    updater = EMAUpdater(warmup_epochs=30)
    epochs = range(100)
    momentums = [updater.get_momentum(epoch) for epoch in epochs]
    
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, momentums)
    plt.title("Teacher Momentum Schedule")
    plt.xlabel("Epoch")
    plt.ylabel("Momentum")
    plt.grid(True)
    plt.axvline(x=30, color='r', linestyle='--', label='Warmup End')
    plt.legend()
    plt.show()

def test_parameter_differences():
    """Test how teacher parameters evolve"""
    backbone = nn.Linear(512, 256)
    model = StudentTeacherWrapper(
        backbone=backbone,
        projection_dim=128,
        hidden_dim=64,
        bottleneck_dim=32
    )
    
    # Track parameter differences
    differences = []
    
    for epoch in range(50):
        # Get initial teacher param
        teacher_param = list(model.teacher.parameters())[0].clone()
        
        # Simulate student update
        x = torch.randn(4, 512)
        student_out = model.forward_student(x)
        loss = torch.mean(student_out)
        loss.backward()
        
        # Update teacher
        model.update_teacher(epoch)
        
        # Calculate difference
        new_teacher_param = list(model.teacher.parameters())[0]
        diff = torch.norm(new_teacher_param - teacher_param).item()
        differences.append(diff)
        
        # Clear gradients
        model.student.zero_grad()
    
    plt.figure(figsize=(10, 6))
    plt.plot(differences)
    plt.title("Teacher Parameter Changes Over Time")
    plt.xlabel("Epoch")
    plt.ylabel("Parameter Change Magnitude")
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    print("Testing Student-Teacher Architecture...")
    model = test_student_teacher_architecture()
    
    print("\nVisualizing momentum schedule...")
    visualize_momentum_schedule()
    
    print("\nTesting parameter evolution...")
    test_parameter_differences()
```

### Step 4: Integration with Existing Backbone

```python
# integration_example.py
import torch
from torchvision.models import resnet50
import sys
sys.path.append('.')  # Add current directory to path

from module2.backbones import ResNetBackbone, ViTBackbone
from student_teacher import StudentTeacherWrapper

def create_dino_model(
    backbone_type: str = "resnet50",
    pretrained: bool = True,
    **kwargs
) -> StudentTeacherWrapper:
    """
    Create a complete DINO model with student-teacher architecture
    """
    
    if backbone_type.startswith("resnet"):
        backbone = ResNetBackbone(
            arch=backbone_type,
            pretrained=pretrained,
            remove_last_layer=True
        )
    elif backbone_type.startswith("vit"):
        backbone = ViTBackbone(
            arch=backbone_type,
            pretrained=pretrained,
            remove_head=True
        )
    else:
        raise ValueError(f"Unsupported backbone type: {backbone_type}")
    
    # Create student-teacher model
    model = StudentTeacherWrapper(
        backbone=backbone,
        **kwargs
    )
    
    return model

def demonstrate_training_cycle():
    """Demonstrate a typical training cycle"""
    
    # Create model
    model = create_dino_model(
        backbone_type="resnet50",
        projection_dim=65536,
        hidden_dim=2048,
        bottleneck_dim=256
    )
    
    # Create optimizer (only for student)
    optimizer = torch.optim.AdamW(
        model.get_student_parameters(),
        lr=0.0005,
        weight_decay=0.04
    )
    
    # Simulate training loop
    model.train()
    
    for epoch in range(5):
        print(f"\nEpoch {epoch}")
        
        # Simulate batch
        global_crops = torch.randn(2, 3, 224, 224)  # 2 global crops
        local_crops = torch.randn(6, 3, 96, 96)    # 6 local crops
        
        # Student forward on all crops
        student_global = model.forward_student(global_crops)
        student_local = model.forward_student(
            torch.nn.functional.interpolate(local_crops, size=224)
        )
        
        # Teacher forward on global crops only
        teacher_global = model.forward_teacher(global_crops)
        
        # Simulate loss computation
        loss = torch.mean(student_global) + torch.mean(student_local) + torch.mean(teacher_global)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Update teacher
        model.update_teacher(epoch)
        
        print(f"Loss: {loss.item():.4f}")
        
        # Print momentum
        momentum = model._get_momentum_schedule(epoch)
        print(f"Teacher momentum: {momentum:.6f}")

if __name__ == "__main__":
    print("Creating DINO model...")
    model = create_dino_model()
    print(f"Model created successfully!")
    
    print("\nDemonstrating training cycle...")
    demonstrate_training_cycle()
```

## 🧪 Hands-on Exercise: Build Your Student-Teacher Network

### Exercise 1: Basic Implementation

Implement a simple student-teacher network from scratch:

```python
# exercise1.py
import torch
import torch.nn as nn

class SimpleStudentTeacher(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        # TODO: Implement student and teacher networks
        # Both should have the same architecture but separate parameters
        pass
    
    def forward_student(self, x):
        # TODO: Implement student forward pass
        pass
    
    def forward_teacher(self, x):
        # TODO: Implement teacher forward pass (no gradients)
        pass
    
    def update_teacher(self, momentum: float = 0.99):
        # TODO: Implement EMA update for teacher
        pass

# Test your implementation
model = SimpleStudentTeacher(784, 256, 128)
x = torch.randn(32, 784)

student_out = model.forward_student(x)
teacher_out = model.forward_teacher(x)

print(f"Student output shape: {student_out.shape}")
print(f"Teacher output shape: {teacher_out.shape}")
```

### Exercise 2: Momentum Schedule Analysis

Analyze different momentum schedules:

```python
# exercise2.py
import matplotlib.pyplot as plt
import numpy as np

def cosine_momentum_schedule(epoch, total_epochs, base_momentum=0.996, final_momentum=0.999):
    """Implement cosine momentum schedule"""
    # TODO: Implement cosine annealing for momentum
    pass

def linear_momentum_schedule(epoch, warmup_epochs, base_momentum=0.996, final_momentum=0.999):
    """Implement linear momentum schedule"""
    # TODO: Implement linear warmup for momentum
    pass

# Compare different schedules
epochs = np.arange(100)
cosine_momentum = [cosine_momentum_schedule(e, 100) for e in epochs]
linear_momentum = [linear_momentum_schedule(e, 30) for e in epochs]

# TODO: Plot and compare the schedules
```

### Exercise 3: Parameter Synchronization

Track how well teacher and student parameters stay synchronized:

```python
# exercise3.py
def track_parameter_divergence(model, num_steps=100):
    """Track how teacher and student parameters diverge over time"""
    
    divergences = []
    
    for step in range(num_steps):
        # TODO: 
        # 1. Compute current parameter divergence
        # 2. Simulate a training step
        # 3. Update teacher
        # 4. Record new divergence
        pass
    
    return divergences

# TODO: Visualize parameter divergence over time
```

## 🔍 Key Insights

### Why EMA Works
1. **Stability**: Teacher provides stable targets while student learns
2. **Momentum**: Prevents teacher from changing too rapidly
3. **Asymmetry**: Creates learning signal without external labels

### Critical Implementation Details
1. **Gradient Isolation**: Teacher must not receive gradients
2. **Momentum Scheduling**: Start conservative, increase over time
3. **Weight Initialization**: Teacher initialized as copy of student
4. **Normalization**: Consistent L2 normalization of outputs

### Common Pitfalls
1. **Gradient Leakage**: Ensuring teacher parameters don't accumulate gradients
2. **Momentum Too High**: Teacher changes too slowly
3. **Momentum Too Low**: Teacher becomes unstable
4. **Memory Issues**: Teacher doubles memory requirements

## 📝 Summary

In this lesson, you learned:

✅ **Student-Teacher Architecture**: How DINO uses two identical networks with different update rules

✅ **EMA Updates**: Mathematical foundation and implementation of exponential moving averages

✅ **Momentum Scheduling**: How to gradually increase teacher stability over training

✅ **Weight Synchronization**: Proper initialization and gradient isolation

✅ **Integration**: How to combine with existing backbone architectures

### Next Steps
In the next lesson, we'll implement the multi-crop strategy that creates the asymmetric learning signal between student and teacher networks.

## 🔗 Additional Resources

- [Original DINO Paper](https://arxiv.org/abs/2104.14294)
- [Exponential Moving Averages in Deep Learning](https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage)
- [Knowledge Distillation Survey](https://arxiv.org/abs/2006.05525)

---

**Next**: [Module 3, Lesson 2: Multi-Crop Strategy Implementation](module3_lesson2_multicrop_strategy.md)
