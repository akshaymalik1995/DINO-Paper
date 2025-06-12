# Module 3, Lesson 3: Projection Heads and Feature Normalization

## 🎯 Learning Objectives
By the end of this lesson, you will:
- Understand the role of projection heads in DINO
- Implement MLP projection heads with proper architecture
- Master L2 normalization and its importance
- Build dimension reduction strategies for efficient training

## 📚 Theoretical Background

### Projection Heads in Self-Supervised Learning

**Projection heads** are crucial components that:
1. **Transform representations** from backbone features to a space suitable for comparison
2. **Reduce dimensionality** while preserving semantic information
3. **Enable contrastive/distillation learning** through proper feature alignment
4. **Prevent representation collapse** through architectural design

### DINO's Projection Head Design

DINO uses a specific MLP architecture:
- **3-layer MLP** with hidden dimensions
- **GELU activations** for smooth gradients
- **Batch normalization** (optional) for training stability
- **No bias in final layer** to encourage centering
- **L2 normalization** of final outputs

### Mathematical Framework

**Projection Head Structure**:
```
backbone_features → Linear(d_backbone, d_hidden) → GELU → 
Linear(d_hidden, d_hidden) → GELU → 
Linear(d_hidden, d_bottleneck) → 
Linear(d_bottleneck, d_out, bias=False) → L2_normalize
```

**L2 Normalization**:
```
z_normalized = z / ||z||_2
```

Where `||z||_2` is the L2 norm of vector z.

## 🛠️ Implementation

### Step 1: Advanced Projection Head Implementation

```python
# projection_heads.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, List
import numpy as np

class DINOProjectionHead(nn.Module):
    """
    DINO-style projection head with configurable architecture
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 2048,
        bottleneck_dim: int = 256,
        output_dim: int = 65536,
        num_layers: int = 3,
        use_bn: bool = False,
        use_bias_in_head: bool = False,
        norm_last_layer: bool = True,
        activation: str = 'gelu',
        dropout: float = 0.0
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.norm_last_layer = norm_last_layer
        
        # Build layers
        layers = []
        current_dim = input_dim
        
        # Hidden layers
        for i in range(num_layers - 1):
            if i == 0:
                next_dim = hidden_dim
            elif i == num_layers - 2:
                next_dim = bottleneck_dim
            else:
                next_dim = hidden_dim
            
            # Linear layer
            layers.append(nn.Linear(current_dim, next_dim))
            
            # Batch normalization
            if use_bn and i < num_layers - 2:  # No BN before last layer
                layers.append(nn.BatchNorm1d(next_dim))
            
            # Activation
            if i < num_layers - 2:  # No activation after bottleneck
                layers.append(self._get_activation(activation))
            
            # Dropout
            if dropout > 0 and i < num_layers - 2:
                layers.append(nn.Dropout(dropout))
            
            current_dim = next_dim
        
        # Final projection layer
        self.projection_layers = nn.Sequential(*layers)
        self.final_layer = nn.Linear(bottleneck_dim, output_dim, bias=use_bias_in_head)
        
        # Initialize weights
        self._init_weights()
        
        # Special initialization for last layer
        if norm_last_layer:
            self.final_layer.weight.data.normal_(0, 0.01)
            self.final_layer.weight.data = F.normalize(self.final_layer.weight.data, dim=1)
    
    def _get_activation(self, activation: str) -> nn.Module:
        """Get activation function"""
        activations = {
            'relu': nn.ReLU(),
            'gelu': nn.GELU(),
            'elu': nn.ELU(),
            'leaky_relu': nn.LeakyReLU(0.01),
            'swish': nn.SiLU()
        }
        
        if activation not in activations:
            raise ValueError(f"Unsupported activation: {activation}")
        
        return activations[activation]
    
    def _init_weights(self):
        """Initialize network weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Xavier normal initialization
                nn.init.xavier_normal_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through projection head
        
        Args:
            x: Input features [batch_size, input_dim]
            
        Returns:
            Normalized output features [batch_size, output_dim]
        """
        # Pass through projection layers
        features = self.projection_layers(x)
        
        # Final projection
        output = self.final_layer(features)
        
        # L2 normalization
        output = F.normalize(output, dim=1, p=2)
        
        return output


class MultiHeadProjection(nn.Module):
    """
    Multi-head projection for different downstream tasks
    """
    def __init__(
        self,
        input_dim: int,
        head_configs: List[dict],
        shared_layers: int = 2,
        shared_hidden_dim: int = 2048
    ):
        super().__init__()
        
        # Shared backbone layers
        shared_layers_list = []
        current_dim = input_dim
        
        for i in range(shared_layers):
            shared_layers_list.append(nn.Linear(current_dim, shared_hidden_dim))
            shared_layers_list.append(nn.GELU())
            current_dim = shared_hidden_dim
        
        self.shared_layers = nn.Sequential(*shared_layers_list)
        
        # Individual projection heads
        self.heads = nn.ModuleDict()
        for head_config in head_configs:
            name = head_config['name']
            self.heads[name] = DINOProjectionHead(
                input_dim=current_dim,
                **{k: v for k, v in head_config.items() if k != 'name'}
            )
    
    def forward(self, x: torch.Tensor, head_name: str = 'default') -> torch.Tensor:
        """Forward pass through specific head"""
        shared_features = self.shared_layers(x)
        return self.heads[head_name](shared_features)


class AdaptiveProjectionHead(nn.Module):
    """
    Projection head that adapts its architecture based on input dimension
    """
    def __init__(
        self,
        input_dim: int,
        output_dim: int = 65536,
        compression_ratio: float = 0.5,
        min_hidden_dim: int = 512,
        max_hidden_dim: int = 4096
    ):
        super().__init__()
        
        # Calculate optimal hidden dimension
        hidden_dim = int(input_dim * compression_ratio)
        hidden_dim = max(min_hidden_dim, min(hidden_dim, max_hidden_dim))
        
        # Calculate bottleneck dimension
        bottleneck_dim = max(output_dim // 256, 256)
        
        self.projection_head = DINOProjectionHead(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            bottleneck_dim=bottleneck_dim,
            output_dim=output_dim
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.projection_head(x)
```

### Step 2: Feature Normalization Strategies

```python
# normalization.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

class L2Normalization(nn.Module):
    """
    L2 normalization layer with optional learnable temperature
    """
    def __init__(
        self,
        dim: int = 1,
        eps: float = 1e-8,
        learnable_temp: bool = False,
        initial_temp: float = 1.0
    ):
        super().__init__()
        self.dim = dim
        self.eps = eps
        
        if learnable_temp:
            self.temperature = nn.Parameter(torch.tensor(initial_temp))
        else:
            self.register_buffer('temperature', torch.tensor(initial_temp))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """L2 normalize with optional temperature scaling"""
        normalized = F.normalize(x, dim=self.dim, eps=self.eps)
        return normalized / self.temperature


class LayerNormalization(nn.Module):
    """
    Custom layer normalization for features
    """
    def __init__(
        self,
        normalized_shape: int,
        eps: float = 1e-5,
        elementwise_affine: bool = True
    ):
        super().__init__()
        self.layer_norm = nn.LayerNorm(
            normalized_shape,
            eps=eps,
            elementwise_affine=elementwise_affine
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer_norm(x)


class SpectralNormalization(nn.Module):
    """
    Spectral normalization for stabilizing training
    """
    def __init__(self, power_iterations: int = 1):
        super().__init__()
        self.power_iterations = power_iterations
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply spectral normalization"""
        # Compute SVD
        U, S, V = torch.svd(x)
        
        # Normalize by largest singular value
        return x / S[0]


class AdaptiveNormalization(nn.Module):
    """
    Adaptive normalization that chooses between different strategies
    """
    def __init__(
        self,
        feature_dim: int,
        norm_type: str = 'l2',
        adaptive: bool = True
    ):
        super().__init__()
        self.adaptive = adaptive
        
        if norm_type == 'l2':
            self.norm = L2Normalization()
        elif norm_type == 'layer':
            self.norm = LayerNormalization(feature_dim)
        elif norm_type == 'batch':
            self.norm = nn.BatchNorm1d(feature_dim)
        else:
            raise ValueError(f"Unknown normalization type: {norm_type}")
        
        if adaptive:
            # Learnable mixing weights
            self.mix_weights = nn.Parameter(torch.ones(3) / 3)
            self.l2_norm = L2Normalization()
            self.layer_norm = LayerNormalization(feature_dim)
            self.batch_norm = nn.BatchNorm1d(feature_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.adaptive:
            return self.norm(x)
        
        # Compute different normalizations
        l2_out = self.l2_norm(x)
        layer_out = self.layer_norm(x)
        batch_out = self.batch_norm(x)
        
        # Weighted combination
        weights = F.softmax(self.mix_weights, dim=0)
        output = (weights[0] * l2_out + 
                 weights[1] * layer_out + 
                 weights[2] * batch_out)
        
        return output


def analyze_feature_statistics(features: torch.Tensor) -> dict:
    """
    Analyze feature statistics for debugging normalization
    """
    stats = {}
    
    # Basic statistics
    stats['mean'] = torch.mean(features).item()
    stats['std'] = torch.std(features).item()
    stats['min'] = torch.min(features).item()
    stats['max'] = torch.max(features).item()
    
    # L2 norms
    l2_norms = torch.norm(features, dim=1)
    stats['l2_norm_mean'] = torch.mean(l2_norms).item()
    stats['l2_norm_std'] = torch.std(l2_norms).item()
    
    # Cosine similarities (pairwise)
    normalized_features = F.normalize(features, dim=1)
    cosine_sim = torch.mm(normalized_features, normalized_features.t())
    # Remove diagonal (self-similarities)
    mask = ~torch.eye(cosine_sim.shape[0], dtype=bool)
    cosine_sim_off_diag = cosine_sim[mask]
    stats['cosine_sim_mean'] = torch.mean(cosine_sim_off_diag).item()
    stats['cosine_sim_std'] = torch.std(cosine_sim_off_diag).item()
    
    return stats
```

### Step 3: Integration with Complete DINO Architecture

```python
# complete_dino.py
import torch
import torch.nn as nn
from typing import Dict, Optional, Any
import copy

class CompleteDINOModel(nn.Module):
    """
    Complete DINO model with backbone + projection head + student-teacher
    """
    def __init__(
        self,
        backbone: nn.Module,
        projection_config: dict,
        teacher_momentum: float = 0.996,
        teacher_warmup_epochs: int = 30,
        center_momentum: float = 0.9
    ):
        super().__init__()
        
        # Get backbone output dimension
        self.backbone_dim = self._get_backbone_dim(backbone)
        
        # Create student network
        self.student_backbone = backbone
        self.student_projection = DINOProjectionHead(
            input_dim=self.backbone_dim,
            **projection_config
        )
        
        # Create teacher network (copy of student)
        self.teacher_backbone = copy.deepcopy(backbone)
        self.teacher_projection = DINOProjectionHead(
            input_dim=self.backbone_dim,
            **projection_config
        )
        
        # Disable gradients for teacher
        for param in self.teacher_backbone.parameters():
            param.requires_grad = False
        for param in self.teacher_projection.parameters():
            param.requires_grad = False
        
        # Teacher update parameters
        self.teacher_momentum = teacher_momentum
        self.teacher_warmup_epochs = teacher_warmup_epochs
        self.center_momentum = center_momentum
        
        # Initialize center for teacher outputs
        self.register_buffer('center', torch.zeros(1, projection_config['output_dim']))
        
        # Copy student weights to teacher
        self._copy_student_to_teacher()
    
    def _get_backbone_dim(self, backbone: nn.Module) -> int:
        """Get the output dimension of backbone"""
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224)
            output = backbone(dummy_input)
            if isinstance(output, tuple):
                output = output[0]
            return output.shape[-1]
    
    def _copy_student_to_teacher(self):
        """Copy student weights to teacher"""
        with torch.no_grad():
            # Copy backbone
            for student_param, teacher_param in zip(
                self.student_backbone.parameters(),
                self.teacher_backbone.parameters()
            ):
                teacher_param.data.copy_(student_param.data)
            
            # Copy projection head
            for student_param, teacher_param in zip(
                self.student_projection.parameters(),
                self.teacher_projection.parameters()
            ):
                teacher_param.data.copy_(student_param.data)
    
    def forward_student(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through student network"""
        features = self.student_backbone(x)
        if isinstance(features, tuple):
            features = features[0]
        
        projections = self.student_projection(features)
        return projections
    
    def forward_teacher(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through teacher network"""
        with torch.no_grad():
            features = self.teacher_backbone(x)
            if isinstance(features, tuple):
                features = features[0]
            
            projections = self.teacher_projection(features)
            return projections
    
    def update_teacher(self, epoch: int):
        """Update teacher weights using EMA"""
        # Calculate momentum
        momentum = self._get_momentum_schedule(epoch)
        
        with torch.no_grad():
            # Update backbone
            for student_param, teacher_param in zip(
                self.student_backbone.parameters(),
                self.teacher_backbone.parameters()
            ):
                teacher_param.data.mul_(momentum).add_(
                    student_param.data, alpha=1 - momentum
                )
            
            # Update projection head
            for student_param, teacher_param in zip(
                self.student_projection.parameters(),
                self.teacher_projection.parameters()
            ):
                teacher_param.data.mul_(momentum).add_(
                    student_param.data, alpha=1 - momentum
                )
    
    def _get_momentum_schedule(self, epoch: int) -> float:
        """Get momentum value for current epoch"""
        if epoch < self.teacher_warmup_epochs:
            return self.teacher_momentum
        else:
            return min(self.teacher_momentum + (1 - self.teacher_momentum) * epoch / 100, 0.999)
    
    def update_center(self, teacher_outputs: torch.Tensor):
        """Update center for teacher outputs"""
        batch_center = torch.mean(teacher_outputs, dim=0, keepdim=True)
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)
    
    def get_student_parameters(self):
        """Get student parameters for optimizer"""
        return list(self.student_backbone.parameters()) + list(self.student_projection.parameters())
    
    def extract_features(self, x: torch.Tensor, use_teacher: bool = True) -> torch.Tensor:
        """Extract features for downstream tasks"""
        if use_teacher:
            features = self.teacher_backbone(x)
        else:
            features = self.student_backbone(x)
        
        if isinstance(features, tuple):
            features = features[0]
        
        return features


def create_dino_model(
    backbone_name: str = 'resnet50',
    output_dim: int = 65536,
    hidden_dim: int = 2048,
    bottleneck_dim: int = 256,
    **kwargs
) -> CompleteDINOModel:
    """
    Factory function to create DINO model
    """
    
    # Import backbone
    if backbone_name.startswith('resnet'):
        from torchvision.models import resnet50, resnet18
        if backbone_name == 'resnet50':
            backbone = resnet50(pretrained=False)
            backbone.fc = nn.Identity()  # Remove classification head
        elif backbone_name == 'resnet18':
            backbone = resnet18(pretrained=False)
            backbone.fc = nn.Identity()
    else:
        raise ValueError(f"Unsupported backbone: {backbone_name}")
    
    # Projection configuration
    projection_config = {
        'hidden_dim': hidden_dim,
        'bottleneck_dim': bottleneck_dim,
        'output_dim': output_dim,
        'use_bn': False,
        'norm_last_layer': True
    }
    
    # Create model
    model = CompleteDINOModel(
        backbone=backbone,
        projection_config=projection_config,
        **kwargs
    )
    
    return model
```

### Step 4: Testing and Validation

```python
# test_projection_heads.py
import torch
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

def test_projection_head():
    """Test projection head implementation"""
    
    # Create test data
    batch_size = 32
    input_dim = 2048
    x = torch.randn(batch_size, input_dim)
    
    # Create projection head
    projection_head = DINOProjectionHead(
        input_dim=input_dim,
        hidden_dim=4096,
        bottleneck_dim=256,
        output_dim=65536
    )
    
    # Forward pass
    output = projection_head(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    
    # Check normalization
    norms = torch.norm(output, dim=1)
    print(f"L2 norms - mean: {norms.mean().item():.6f}, std: {norms.std().item():.6f}")
    
    # Check for collapse
    cosine_sim = torch.mm(output, output.t())
    off_diagonal = cosine_sim[~torch.eye(batch_size, dtype=bool)]
    print(f"Cosine similarity - mean: {off_diagonal.mean().item():.6f}")
    
    return projection_head

def test_normalization_strategies():
    """Test different normalization strategies"""
    
    batch_size = 64
    feature_dim = 1024
    x = torch.randn(batch_size, feature_dim) * 10  # Large variance
    
    # Test different normalizations
    l2_norm = L2Normalization()
    layer_norm = LayerNormalization(feature_dim)
    adaptive_norm = AdaptiveNormalization(feature_dim, adaptive=True)
    
    # Apply normalizations
    l2_out = l2_norm(x)
    layer_out = layer_norm(x)
    adaptive_out = adaptive_norm(x)
    
    # Analyze results
    print("Original features:")
    print(analyze_feature_statistics(x))
    
    print("\nL2 normalized:")
    print(analyze_feature_statistics(l2_out))
    
    print("\nLayer normalized:")
    print(analyze_feature_statistics(layer_out))
    
    print("\nAdaptive normalized:")
    print(analyze_feature_statistics(adaptive_out))

def visualize_feature_space():
    """Visualize features in different spaces"""
    
    # Create synthetic data with clusters
    np.random.seed(42)
    n_samples = 300
    n_clusters = 3
    
    data = []
    labels = []
    
    for i in range(n_clusters):
        cluster_data = np.random.multivariate_normal(
            mean=[i*5, i*3],
            cov=[[1, 0.5], [0.5, 1]],
            size=n_samples//n_clusters
        )
        data.append(cluster_data)
        labels.extend([i] * (n_samples//n_clusters))
    
    data = np.vstack(data)
    labels = np.array(labels)
    
    # Convert to high-dimensional space
    high_dim_data = np.random.randn(n_samples, 512) @ np.random.randn(512, 2) @ data.T
    high_dim_data = high_dim_data.T
    
    # Test projection head
    x = torch.tensor(high_dim_data, dtype=torch.float32)
    projection_head = DINOProjectionHead(
        input_dim=512,
        output_dim=128,
        hidden_dim=256,
        bottleneck_dim=64
    )
    
    projected = projection_head(x).detach().numpy()
    
    # Reduce dimensionality for visualization
    pca = PCA(n_components=2)
    tsne = TSNE(n_components=2, random_state=42)
    
    original_2d = pca.fit_transform(high_dim_data)
    projected_2d = pca.fit_transform(projected)
    
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Original features
    for i in range(n_clusters):
        mask = labels == i
        axes[0].scatter(original_2d[mask, 0], original_2d[mask, 1], 
                       label=f'Cluster {i}', alpha=0.6)
    axes[0].set_title('Original High-Dim Features (PCA)')
    axes[0].legend()
    
    # Projected features
    for i in range(n_clusters):
        mask = labels == i
        axes[1].scatter(projected_2d[mask, 0], projected_2d[mask, 1], 
                       label=f'Cluster {i}', alpha=0.6)
    axes[1].set_title('Projected Features (PCA)')
    axes[1].legend()
    
    plt.tight_layout()
    plt.show()

def test_complete_model():
    """Test complete DINO model"""
    
    # Create model
    model = create_dino_model(
        backbone_name='resnet18',
        output_dim=1024,
        hidden_dim=512,
        bottleneck_dim=128
    )
    
    # Test forward passes
    batch_size = 8
    x = torch.randn(batch_size, 3, 224, 224)
    
    # Student forward
    student_out = model.forward_student(x)
    print(f"Student output shape: {student_out.shape}")
    
    # Teacher forward
    teacher_out = model.forward_teacher(x)
    print(f"Teacher output shape: {teacher_out.shape}")
    
    # Check normalization
    student_norms = torch.norm(student_out, dim=1)
    teacher_norms = torch.norm(teacher_out, dim=1)
    
    print(f"Student L2 norms: {student_norms.mean().item():.6f} ± {student_norms.std().item():.6f}")
    print(f"Teacher L2 norms: {teacher_norms.mean().item():.6f} ± {teacher_norms.std().item():.6f}")
    
    # Test teacher update
    print("\nTesting teacher update...")
    initial_teacher_param = list(model.teacher_projection.parameters())[0].clone()
    
    # Simulate gradient step
    loss = student_out.mean()
    loss.backward()
    
    # Update teacher
    model.update_teacher(epoch=0)
    
    updated_teacher_param = list(model.teacher_projection.parameters())[0]
    param_change = torch.norm(updated_teacher_param - initial_teacher_param)
    
    print(f"Teacher parameter change: {param_change.item():.8f}")

if __name__ == "__main__":
    print("Testing projection head...")
    test_projection_head()
    
    print("\nTesting normalization strategies...")
    test_normalization_strategies()
    
    print("\nVisualizing feature space...")
    visualize_feature_space()
    
    print("\nTesting complete model...")
    test_complete_model()
```

## 🧪 Hands-on Exercise: Build Your Projection Head

### Exercise 1: Custom Projection Architecture

Design your own projection head architecture:

```python
# exercise1.py
import torch
import torch.nn as nn

class CustomProjectionHead(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        # TODO: Design your own architecture
        # Consider: skip connections, different activations, etc.
        pass
    
    def forward(self, x):
        # TODO: Implement forward pass
        pass

# Test different architectural choices
```

### Exercise 2: Normalization Ablation Study

Compare different normalization strategies:

```python
# exercise2.py
def compare_normalizations(features):
    """Compare different normalization methods"""
    
    # TODO: Implement and compare:
    # 1. L2 normalization
    # 2. Layer normalization  
    # 3. Batch normalization
    # 4. No normalization
    
    # Analyze:
    # - Feature distribution
    # - Cosine similarities
    # - Training stability
    pass

# Run comparison with synthetic data
```

### Exercise 3: Feature Quality Analysis

Analyze the quality of learned features:

```python
# exercise3.py
def analyze_feature_quality(model, dataloader):
    """Analyze the quality of learned features"""
    
    features = []
    labels = []
    
    with torch.no_grad():
        for batch in dataloader:
            # TODO: Extract features and analyze:
            # 1. Clustering quality
            # 2. Linear separability
            # 3. Representation collapse
            pass
    
    # TODO: Compute metrics and visualizations
    return metrics

# Test with your trained model
```

## 🔍 Key Insights

### Projection Head Design Principles
1. **Progressive Dimensionality**: Start wide, narrow to bottleneck, expand to output
2. **Smooth Activations**: GELU works better than ReLU for gradients
3. **Normalization Strategy**: L2 normalization crucial for stability
4. **No Bias in Final Layer**: Encourages centering behavior

### Feature Normalization Benefits
1. **Gradient Stability**: Prevents gradient explosion/vanishing
2. **Training Speed**: Faster convergence with proper normalization
3. **Representation Quality**: Better feature clustering and separation
4. **Collapse Prevention**: Helps prevent mode collapse

### Common Pitfalls
1. **Over-normalization**: Too much normalization can hurt expressiveness
2. **Wrong Dimensions**: Incorrect normalization dimensions break training
3. **Gradient Flow**: Poor initialization can block gradient flow
4. **Memory Issues**: Large projection dimensions increase memory usage

## 📝 Summary

In this lesson, you learned:

✅ **Projection Head Architecture**: How to design effective MLP projection heads for DINO

✅ **Feature Normalization**: L2 normalization and its critical role in training stability

✅ **Advanced Techniques**: Multi-head projections and adaptive normalization strategies

✅ **Integration**: How projection heads fit into the complete DINO architecture

✅ **Quality Analysis**: Methods to analyze and debug feature quality

### Module 3 Complete!
You've now implemented the complete student-teacher architecture with multi-crop augmentation and projection heads. Next, we'll dive into the DINO loss function and training mechanisms.

## 🔗 Additional Resources

- [DINO Paper - Projection Head Analysis](https://arxiv.org/abs/2104.14294)
- [Feature Normalization in Deep Learning](https://arxiv.org/abs/1502.03167)
- [Understanding Self-Supervised Learning Dynamics](https://arxiv.org/abs/2106.15132)

---

**Next**: [Module 4, Lesson 1: Centering Mechanism Implementation](module4_lesson1_centering_mechanism.md)
