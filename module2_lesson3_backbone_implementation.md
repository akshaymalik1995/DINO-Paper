# 🔹 Module 2, Lesson 2.3: Backbone Architecture Implementation

## 📚 Learning Objectives
By the end of this lesson, you will:
- Implement ResNet backbone with proper modifications for DINO
- Build Vision Transformer (ViT) backbone from scratch
- Create MLP projection heads with DINO-specific design choices
- Understand feature dimension and normalization strategies
- Build complete backbone + projection head combinations
- Compare different backbone architectures for self-supervised learning

---

## 🏗️ DINO Backbone Architecture Overview

### Key Design Principles

1. **Feature Extraction**: Strong backbone for meaningful representations
2. **Projection Head**: Maps features to embedding space for comparison
3. **Normalization**: Critical for stable self-supervised training
4. **Modularity**: Easy to swap backbones (ResNet ↔ ViT)

### Architecture Components

```
Input Image → Backbone Network → Feature Vector → Projection Head → Output Embedding
   (3×H×W)      (ResNet/ViT)       (d_backbone)      (3-layer MLP)     (d_output)
```

---

## 🧱 ResNet Backbone Implementation

### ResNet Architecture for DINO (`models/backbones/resnet.py`)

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List


class BasicBlock(nn.Module):
    """Basic ResNet block for ResNet-18/34"""
    expansion = 1
    
    def __init__(self, inplanes: int, planes: int, stride: int = 1, 
                 downsample: Optional[nn.Module] = None):
        super().__init__()
        
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride,
                              padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,
                              padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        
        self.downsample = downsample
        self.stride = stride
    
    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        out = self.relu(out)
        
        return out


class Bottleneck(nn.Module):
    """Bottleneck block for ResNet-50/101/152"""
    expansion = 4
    
    def __init__(self, inplanes: int, planes: int, stride: int = 1,
                 downsample: Optional[nn.Module] = None):
        super().__init__()
        
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                              padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
    
    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        
        out = self.conv3(out)
        out = self.bn3(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        out = self.relu(out)
        
        return out


class ResNet(nn.Module):
    """
    ResNet backbone for DINO
    
    Modified from standard ResNet:
    - Removes final classification layer
    - Returns feature vectors instead of logits
    - Supports different input sizes (CIFAR-10 vs ImageNet)
    """
    
    def __init__(self, block, layers: List[int], num_classes: int = 1000,
                 zero_init_residual: bool = False, input_size: int = 224):
        super().__init__()
        
        self.inplanes = 64
        self.input_size = input_size
        
        # Adjust initial conv for different input sizes
        if input_size <= 64:  # CIFAR-10 style
            self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            self.maxpool = nn.Identity()  # No maxpool for small images
        else:  # ImageNet style
            self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        
        # ResNet layers
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        
        # Global average pooling
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Feature dimension
        self.feature_dim = 512 * block.expansion
        
        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
        # Zero-initialize the last BN in each residual branch
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)
    
    def _make_layer(self, block, planes: int, blocks: int, stride: int = 1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                         kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        
        return nn.Sequential(*layers)
    
    def forward(self, x):
        # Initial conv and pooling
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        # ResNet layers
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        # Global pooling
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        
        return x


def resnet18(input_size: int = 224, **kwargs):
    """ResNet-18 for DINO"""
    return ResNet(BasicBlock, [2, 2, 2, 2], input_size=input_size, **kwargs)


def resnet34(input_size: int = 224, **kwargs):
    """ResNet-34 for DINO"""
    return ResNet(BasicBlock, [3, 4, 6, 3], input_size=input_size, **kwargs)


def resnet50(input_size: int = 224, **kwargs):
    """ResNet-50 for DINO"""
    return ResNet(Bottleneck, [3, 4, 6, 3], input_size=input_size, **kwargs)


def resnet101(input_size: int = 224, **kwargs):
    """ResNet-101 for DINO"""
    return ResNet(Bottleneck, [3, 4, 23, 3], input_size=input_size, **kwargs)
```

---

## 🔍 Vision Transformer Implementation

### ViT Backbone for DINO (`models/backbones/vit.py`)

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class PatchEmbedding(nn.Module):
    """
    Image to Patch Embedding for Vision Transformer
    """
    def __init__(self, img_size: int = 224, patch_size: int = 16, 
                 in_channels: int = 3, embed_dim: int = 768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        self.embed_dim = embed_dim
        
        # Patch embedding via convolution
        self.projection = nn.Conv2d(
            in_channels, embed_dim, 
            kernel_size=patch_size, stride=patch_size
        )
    
    def forward(self, x):
        # x: (B, C, H, W)
        B, C, H, W = x.shape
        
        # Apply patch embedding
        x = self.projection(x)  # (B, embed_dim, H//P, W//P)
        x = x.flatten(2)        # (B, embed_dim, N)
        x = x.transpose(1, 2)   # (B, N, embed_dim)
        
        return x


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Self-Attention for Vision Transformer
    """
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0,
                 qkv_bias: bool = True):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        
        # QKV projection
        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=qkv_bias)
        
        # Output projection
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_dropout = nn.Dropout(dropout)
        
        # Attention dropout
        self.attn_dropout = nn.Dropout(dropout)
    
    def forward(self, x, return_attention: bool = False):
        B, N, C = x.shape
        
        # Generate Q, K, V
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, heads, N, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Attention computation
        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, heads, N, N)
        attn = attn.softmax(dim=-1)
        attn = self.attn_dropout(attn)
        
        # Apply attention to values
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)  # (B, N, C)
        
        # Output projection
        x = self.proj(x)
        x = self.proj_dropout(x)
        
        if return_attention:
            return x, attn
        return x


class TransformerBlock(nn.Module):
    """
    Transformer block (Multi-Head Attention + MLP)
    """
    def __init__(self, embed_dim: int, num_heads: int, mlp_ratio: float = 4.0,
                 dropout: float = 0.0, drop_path: float = 0.0):
        super().__init__()
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        # Multi-head attention
        self.attn = MultiHeadAttention(embed_dim, num_heads, dropout)
        
        # MLP
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )
        
        # Drop path (stochastic depth)
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()
    
    def forward(self, x, return_attention: bool = False):
        if return_attention:
            attn_out, attn_weights = self.attn(self.norm1(x), return_attention=True)
            x = x + self.drop_path(attn_out)
            x = x + self.drop_path(self.mlp(self.norm2(x)))
            return x, attn_weights
        else:
            x = x + self.drop_path(self.attn(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
            return x


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample"""
    def __init__(self, drop_prob: float = 0.):
        super().__init__()
        self.drop_prob = drop_prob
    
    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        output = x.div(keep_prob) * random_tensor
        return output


class VisionTransformer(nn.Module):
    """
    Vision Transformer backbone for DINO
    """
    def __init__(self, img_size: int = 224, patch_size: int = 16, 
                 in_channels: int = 3, embed_dim: int = 768, depth: int = 12,
                 num_heads: int = 12, mlp_ratio: float = 4.0, 
                 dropout: float = 0.0, drop_path_rate: float = 0.0):
        super().__init__()
        
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.depth = depth
        
        # Patch embedding
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        n_patches = self.patch_embed.n_patches
        
        # CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # Positional embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches + 1, embed_dim))
        
        # Dropout
        self.pos_dropout = nn.Dropout(dropout)
        
        # Transformer blocks
        drop_path_rates = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            TransformerBlock(
                embed_dim, num_heads, mlp_ratio, dropout, drop_path_rates[i]
            ) for i in range(depth)
        ])
        
        # Final layer norm
        self.norm = nn.LayerNorm(embed_dim)
        
        # Feature dimension (for projection head)
        self.feature_dim = embed_dim
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        # Initialize positional embedding
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        
        # Initialize other weights
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
    
    def forward(self, x, return_attention: bool = False):
        B = x.shape[0]
        
        # Patch embedding
        x = self.patch_embed(x)  # (B, N, embed_dim)
        
        # Add CLS token
        cls_token = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_token, x), dim=1)  # (B, N+1, embed_dim)
        
        # Add positional embedding
        x = x + self.pos_embed
        x = self.pos_dropout(x)
        
        # Transformer blocks
        attentions = []
        for i, block in enumerate(self.blocks):
            if return_attention and i == len(self.blocks) - 1:
                x, attn = block(x, return_attention=True)
                attentions.append(attn)
            else:
                x = block(x)
        
        # Final layer norm
        x = self.norm(x)
        
        # Return CLS token
        cls_output = x[:, 0]
        
        if return_attention:
            return cls_output, attentions
        return cls_output


def vit_tiny(patch_size: int = 16, **kwargs):
    """ViT-Tiny"""
    return VisionTransformer(
        patch_size=patch_size, embed_dim=192, depth=12, num_heads=3, **kwargs
    )


def vit_small(patch_size: int = 16, **kwargs):
    """ViT-Small"""
    return VisionTransformer(
        patch_size=patch_size, embed_dim=384, depth=12, num_heads=6, **kwargs
    )


def vit_base(patch_size: int = 16, **kwargs):
    """ViT-Base"""
    return VisionTransformer(
        patch_size=patch_size, embed_dim=768, depth=12, num_heads=12, **kwargs
    )


def vit_large(patch_size: int = 16, **kwargs):
    """ViT-Large"""
    return VisionTransformer(
        patch_size=patch_size, embed_dim=1024, depth=24, num_heads=16, **kwargs
    )
```

---

## 🎯 DINO Projection Head Implementation

### MLP Projection Head (`models/heads.py`)

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class DINOProjectionHead(nn.Module):
    """
    DINO projection head implementation
    
    3-layer MLP with specific design choices:
    - GELU activation
    - Optional batch normalization  
    - L2 normalization before final layer
    - Weight normalization on final layer
    """
    
    def __init__(self, 
                 in_dim: int,
                 out_dim: int = 65536,
                 hidden_dim: int = 2048,
                 bottleneck_dim: int = 256,
                 use_bn: bool = False,
                 norm_last_layer: bool = True,
                 nlayers: int = 3):
        super().__init__()
        
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hidden_dim = hidden_dim
        self.bottleneck_dim = bottleneck_dim
        self.nlayers = nlayers
        self.norm_last_layer = norm_last_layer
        
        # Build MLP layers
        layers = []
        
        if nlayers == 1:
            # Single layer case
            layers.append(nn.Linear(in_dim, bottleneck_dim))
        else:
            # Multi-layer case
            # First layer
            layers.append(nn.Linear(in_dim, hidden_dim))
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.GELU())
            
            # Hidden layers
            for _ in range(nlayers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                if use_bn:
                    layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(nn.GELU())
            
            # Last hidden layer
            layers.append(nn.Linear(hidden_dim, bottleneck_dim))
        
        self.mlp = nn.Sequential(*layers)
        
        # Final projection layer with weight normalization
        self.last_layer = nn.utils.weight_norm(
            nn.Linear(bottleneck_dim, out_dim, bias=False)
        )
        
        # Initialize last layer
        self.last_layer.weight_g.data.fill_(1)
        if norm_last_layer:
            self.last_layer.weight_g.requires_grad = False
    
    def forward(self, x):
        # Apply MLP
        x = self.mlp(x)
        
        # L2 normalize before final layer
        x = F.normalize(x, dim=-1, p=2)
        
        # Final projection
        x = self.last_layer(x)
        
        return x


class SimpleProjectionHead(nn.Module):
    """
    Simplified projection head for experimentation
    """
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int = 2048):
        super().__init__()
        
        self.projection = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim)
        )
    
    def forward(self, x):
        return F.normalize(self.projection(x), dim=-1, p=2)


def create_projection_head(head_type: str, in_dim: int, config) -> nn.Module:
    """
    Factory function to create projection heads
    
    Args:
        head_type: Type of projection head ('dino', 'simple')
        in_dim: Input feature dimension
        config: Configuration object
        
    Returns:
        Projection head module
    """
    if head_type.lower() == 'dino':
        return DINOProjectionHead(
            in_dim=in_dim,
            out_dim=config.model.projection_dim,
            hidden_dim=config.model.hidden_dim,
            bottleneck_dim=config.model.bottleneck_dim,
            use_bn=config.model.use_bn_in_head,
            norm_last_layer=config.model.norm_last_layer,
        )
    
    elif head_type.lower() == 'simple':
        return SimpleProjectionHead(
            in_dim=in_dim,
            out_dim=config.model.projection_dim,
            hidden_dim=config.model.hidden_dim,
        )
    
    else:
        raise ValueError(f"Unknown projection head type: {head_type}")
```

---

## 🔧 Complete DINO Model (`models/dino_model.py`)

```python
import torch
import torch.nn as nn
from typing import Dict, Any, Optional
from .backbones.resnet import resnet18, resnet34, resnet50, resnet101
from .backbones.vit import vit_tiny, vit_small, vit_base, vit_large
from .heads import create_projection_head


class DINOModel(nn.Module):
    """
    Complete DINO model combining backbone and projection head
    """
    
    def __init__(self, config):
        super().__init__()
        
        self.config = config
        self.backbone_name = config.model.backbone
        
        # Create backbone
        self.backbone = self._create_backbone()
        
        # Create projection head
        self.projection_head = create_projection_head(
            head_type='dino',
            in_dim=self.backbone.feature_dim,
            config=config
        )
        
        # Store feature dimension
        self.feature_dim = self.backbone.feature_dim
        self.output_dim = config.model.projection_dim
    
    def _create_backbone(self):
        """Create backbone network based on configuration"""
        backbone_name = self.backbone_name.lower()
        input_size = self.config.data.global_crop_size
        
        # ResNet backbones
        if backbone_name == 'resnet18':
            return resnet18(input_size=input_size)
        elif backbone_name == 'resnet34':
            return resnet34(input_size=input_size)
        elif backbone_name == 'resnet50':
            return resnet50(input_size=input_size)
        elif backbone_name == 'resnet101':
            return resnet101(input_size=input_size)
        
        # Vision Transformer backbones
        elif backbone_name == 'vit_tiny' or backbone_name == 'vit-tiny':
            patch_size = getattr(self.config.model, 'patch_size', 16)
            return vit_tiny(img_size=input_size, patch_size=patch_size)
        elif backbone_name == 'vit_small' or backbone_name == 'vit-small':
            patch_size = getattr(self.config.model, 'patch_size', 16)
            return vit_small(img_size=input_size, patch_size=patch_size)
        elif backbone_name == 'vit_base' or backbone_name == 'vit-base':
            patch_size = getattr(self.config.model, 'patch_size', 16)
            return vit_base(img_size=input_size, patch_size=patch_size)
        elif backbone_name == 'vit_large' or backbone_name == 'vit-large':
            patch_size = getattr(self.config.model, 'patch_size', 16)
            return vit_large(img_size=input_size, patch_size=patch_size)
        
        else:
            raise ValueError(f"Unknown backbone: {backbone_name}")
    
    def forward(self, x, return_features: bool = False):
        """
        Forward pass through backbone and projection head
        
        Args:
            x: Input tensor
            return_features: Whether to return intermediate features
            
        Returns:
            Output embeddings, optionally with features
        """
        # Extract features
        features = self.backbone(x)
        
        # Project to output space
        output = self.projection_head(features)
        
        if return_features:
            return output, features
        return output
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'backbone': self.backbone_name,
            'feature_dim': self.feature_dim,
            'output_dim': self.output_dim,
            'total_params': total_params,
            'trainable_params': trainable_params,
            'backbone_params': sum(p.numel() for p in self.backbone.parameters()),
            'head_params': sum(p.numel() for p in self.projection_head.parameters()),
        }


def create_dino_model(config) -> DINOModel:
    """
    Factory function to create DINO model
    
    Args:
        config: Configuration object
        
    Returns:
        DINO model instance
    """
    return DINOModel(config)


def load_dino_model(checkpoint_path: str, config) -> DINOModel:
    """
    Load DINO model from checkpoint
    
    Args:
        checkpoint_path: Path to model checkpoint
        config: Configuration object
        
    Returns:
        Loaded DINO model
    """
    model = create_dino_model(config)
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    return model
```

---

## 🧪 **Hands-on Exercise**: Build and Test Complete Backbone

### Test Script (`scripts/test_backbones.py`)

```python
import sys
import os
sys.path.append('.')

import torch
import torch.nn as nn
from utils.config import load_config
from models.dino_model import create_dino_model
from models.backbones.resnet import resnet18, resnet50
from models.backbones.vit import vit_small, vit_base
from models.heads import DINOProjectionHead


def test_backbone_architectures():
    """Test different backbone architectures"""
    
    print("🏗️ Testing DINO Backbone Architectures")
    print("=" * 50)
    
    # Test configurations
    configs = [
        ('cifar10_resnet18', 32),
        ('cifar10_vit_small', 32),
        ('imagenet_resnet50', 224),
        ('imagenet_vit_base', 224)
    ]
    
    for config_name, input_size in configs:
        print(f"\n🔧 Testing {config_name} (input size: {input_size}x{input_size})")
        
        # Create test configuration
        if 'cifar10' in config_name:
            config = load_config('cifar10_config')
            if 'vit' in config_name:
                config.model.backbone = 'vit_small'
                config.model.patch_size = 8  # Smaller patches for CIFAR-10
        else:
            config = load_config('base_config')
            if 'resnet50' in config_name:
                config.model.backbone = 'resnet50'
            elif 'vit_base' in config_name:
                config.model.backbone = 'vit_base'
        
        try:
            # Create model
            model = create_dino_model(config)
            model.eval()
            
            # Test input
            batch_size = 4
            channels = 3
            x = torch.randn(batch_size, channels, input_size, input_size)
            
            # Forward pass
            with torch.no_grad():
                output = model(x)
                features = model.backbone(x)
            
            # Print results
            model_info = model.get_model_info()
            print(f"   ✓ Backbone: {model_info['backbone']}")
            print(f"   ✓ Input shape: {x.shape}")
            print(f"   ✓ Feature shape: {features.shape}")
            print(f"   ✓ Output shape: {output.shape}")
            print(f"   ✓ Feature dim: {model_info['feature_dim']}")
            print(f"   ✓ Output dim: {model_info['output_dim']}")
            print(f"   ✓ Total params: {model_info['total_params']:,}")
            print(f"   ✓ Backbone params: {model_info['backbone_params']:,}")
            print(f"   ✓ Head params: {model_info['head_params']:,}")
            
            # Test gradient flow
            loss = output.sum()
            loss.backward()
            
            has_gradients = any(p.grad is not None for p in model.parameters() if p.requires_grad)
            print(f"   ✓ Gradient flow: {'✓' if has_gradients else '❌'}")
            
        except Exception as e:
            print(f"   ❌ Error: {e}")


def benchmark_model_performance():
    """Benchmark different backbone performance"""
    import time
    
    print("\n⚡ Benchmarking Model Performance")
    print("=" * 50)
    
    configs = [
        ('ResNet-18', 'resnet18', 32),
        ('ResNet-50', 'resnet50', 224),
        ('ViT-Small', 'vit_small', 224),
        ('ViT-Base', 'vit_base', 224)
    ]
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    for name, backbone_name, input_size in configs:
        print(f"\n🔍 Benchmarking {name}")
        
        # Create configuration
        if input_size == 32:
            config = load_config('cifar10_config')
        else:
            config = load_config('base_config')
        
        config.model.backbone = backbone_name
        if 'vit' in backbone_name and input_size == 32:
            config.model.patch_size = 8
        
        try:
            # Create model
            model = create_dino_model(config).to(device)
            model.eval()
            
            # Test data
            batch_size = 16
            x = torch.randn(batch_size, 3, input_size, input_size).to(device)
            
            # Warmup
            with torch.no_grad():
                for _ in range(5):
                    _ = model(x)
            
            # Benchmark
            torch.cuda.synchronize() if device.type == 'cuda' else None
            start_time = time.time()
            
            n_iterations = 20
            with torch.no_grad():
                for _ in range(n_iterations):
                    _ = model(x)
            
            torch.cuda.synchronize() if device.type == 'cuda' else None
            end_time = time.time()
            
            # Calculate metrics
            total_time = end_time - start_time
            time_per_batch = total_time / n_iterations
            samples_per_second = batch_size / time_per_batch
            
            print(f"   Time per batch: {time_per_batch*1000:.2f}ms")
            print(f"   Samples/second: {samples_per_second:.1f}")
            
            # Memory usage (if CUDA)
            if device.type == 'cuda':
                memory_used = torch.cuda.max_memory_allocated() / 1024**3  # GB
                print(f"   GPU memory: {memory_used:.2f}GB")
                torch.cuda.reset_peak_memory_stats()
            
        except Exception as e:
            print(f"   ❌ Error: {e}")


def test_projection_heads():
    """Test different projection head configurations"""
    
    print("\n🎯 Testing Projection Heads")
    print("=" * 50)
    
    # Test different head configurations
    configs = [
        {'in_dim': 512, 'out_dim': 8192, 'hidden_dim': 2048, 'bottleneck_dim': 256, 'nlayers': 3},
        {'in_dim': 768, 'out_dim': 65536, 'hidden_dim': 2048, 'bottleneck_dim': 256, 'nlayers': 3},
        {'in_dim': 2048, 'out_dim': 65536, 'hidden_dim': 4096, 'bottleneck_dim': 512, 'nlayers': 2},
    ]
    
    for i, head_config in enumerate(configs):
        print(f"\n🔧 Testing head configuration {i+1}")
        
        # Create projection head
        head = DINOProjectionHead(**head_config)
        
        # Test input
        batch_size = 8
        x = torch.randn(batch_size, head_config['in_dim'])
        
        # Forward pass
        output = head(x)
        
        # Check output
        print(f"   Input shape: {x.shape}")
        print(f"   Output shape: {output.shape}")
        print(f"   Output norm: {torch.norm(output, dim=1).mean():.3f} (should be ~1.0)")
        
        # Check parameters
        n_params = sum(p.numel() for p in head.parameters())
        print(f"   Parameters: {n_params:,}")
        
        # Test gradient flow
        loss = output.sum()
        loss.backward()
        
        has_gradients = any(p.grad is not None for p in head.parameters())
        print(f"   Gradient flow: {'✓' if has_gradients else '❌'}")


if __name__ == "__main__":
    # Test backbone architectures
    test_backbone_architectures()
    
    # Benchmark performance
    benchmark_model_performance()
    
    # Test projection heads
    test_projection_heads()
    
    print("\n🎉 All backbone tests completed!")
```

### Run Tests

```powershell
# Run the backbone test script
python scripts/test_backbones.py
```

---

## 📊 Model Comparison Analysis

### Expected Performance Characteristics

| Backbone | Params | CIFAR-10 Speed | ImageNet Speed | Memory (GB) |
|----------|--------|----------------|----------------|-------------|
| ResNet-18 | 11M | ~500 samples/s | ~300 samples/s | 1.2 |
| ResNet-50 | 25M | ~300 samples/s | ~150 samples/s | 2.1 |
| ViT-Small | 22M | ~200 samples/s | ~100 samples/s | 1.8 |
| ViT-Base | 86M | ~100 samples/s | ~50 samples/s | 3.2 |

### Architecture Trade-offs

#### ResNet Advantages:
- ✅ Fast inference
- ✅ Lower memory usage
- ✅ Works well with small images
- ✅ Mature, well-understood

#### Vision Transformer Advantages:
- ✅ Better attention maps for DINO
- ✅ Superior transfer learning
- ✅ Global receptive field
- ✅ Scales better with data

---

## 📝 Configuration Updates

### Enhanced Model Configuration (`configs/base_config.yaml`)

```yaml
# Enhanced model configuration
model:
  backbone: "resnet50"  # resnet18, resnet50, vit_small, vit_base
  
  # ViT-specific parameters
  patch_size: 16  # 8 for small images, 16 for ImageNet
  
  # Projection head parameters
  projection_dim: 65536  # Output embedding dimension
  hidden_dim: 2048       # Hidden layer dimension
  bottleneck_dim: 256    # Bottleneck before final layer
  use_bn_in_head: false  # Batch norm in projection head
  norm_last_layer: true  # Weight norm on last layer
  
  # Head architecture
  head_nlayers: 3        # Number of layers in projection head
```

---

## ✅ Lesson 2.3 Checklist

### Backbone Implementation
- [ ] Implemented ResNet backbone with DINO modifications
- [ ] Built Vision Transformer from scratch with attention
- [ ] Created flexible backbone factory functions
- [ ] Added support for different input sizes (CIFAR-10 vs ImageNet)

### Projection Head Implementation  
- [ ] Implemented DINO-specific projection head design
- [ ] Added weight normalization and L2 normalization
- [ ] Created configurable MLP architecture
- [ ] Built projection head factory function

### Complete Model Integration
- [ ] Combined backbone and projection head into DINOModel
- [ ] Added model factory and loading functions
- [ ] Implemented comprehensive testing suite
- [ ] Benchmarked different architecture combinations

### Performance Analysis
- [ ] Compared ResNet vs ViT performance
- [ ] Analyzed memory usage and speed trade-offs
- [ ] Validated gradient flow and training readiness
- [ ] Documented architecture characteristics

---

## 🎯 Key Takeaways

1. **Backbone Choice Matters**: ResNet for speed, ViT for quality
2. **Projection Head Design**: DINO-specific design choices are crucial
3. **Modularity**: Clean separation enables easy experimentation
4. **Performance Trade-offs**: Memory vs speed vs quality considerations
5. **Configuration Flexibility**: Easy to adapt for different datasets

**Next**: 🏗️ Module 3 - Student-Teacher Architecture

You now have complete, tested backbone implementations ready for DINO training! In Module 3, we'll combine these backbones into the student-teacher architecture that makes DINO work.
