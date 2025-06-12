# 🔹 Module 1, Lesson 1.2: Vision Transformers (ViT) Primer

## 📚 Learning Objectives
By the end of this lesson, you will:
- Understand why transformers work for computer vision
- Master the ViT (Vision Transformer) architecture components
- Implement image tokenization, patching, and positional encoding
- Build a complete patch embedding layer from scratch
- Understand how ViT relates to DINO

---

## 🤔 Why Transformers for Vision?

### Traditional CNN Limitations
- **Local receptive fields**: CNNs process local neighborhoods
- **Inductive biases**: Strong assumptions about spatial locality
- **Limited long-range dependencies**: Harder to capture global relationships
- **Fixed hierarchical structure**: Predefined feature hierarchy

### Transformer Advantages
- **Global attention**: Every patch can attend to every other patch
- **Flexible architecture**: No fixed spatial hierarchies
- **Scalability**: Performance improves with more data and compute
- **Transfer learning**: Great representations for downstream tasks

### 🎯 Key Insight
> **"An image is worth 16x16 words"** - Vision Transformer treats image patches as tokens, just like words in NLP!

---

## 🏗️ Vision Transformer Architecture Overview

### High-Level Architecture

```
Input Image (224x224x3)
         ↓
   Patch Embedding (16x16 patches → 196 tokens)
         ↓
   Add Position Embeddings
         ↓
   Add [CLS] Token
         ↓
   Transformer Encoder (12 layers)
         ↓
   Classification Head (MLP)
         ↓
   Output Predictions
```

### 🧩 Core Components

1. **Patch Embedding**: Convert image patches to token embeddings
2. **Position Encoding**: Add spatial information to patches
3. **Transformer Encoder**: Self-attention and feed-forward layers
4. **Classification Head**: Final prediction layer

---

## 🔍 Component 1: Image Tokenization and Patching

### The Patching Process

**Goal**: Convert a 2D image into a sequence of 1D tokens

```python
# Image: (H, W, C) → Patches: (N, P²·C) where N = HW/P²
# Example: (224, 224, 3) → (196, 768) with P=16
```

### Mathematical Foundation

For an image of size `H × W × C` with patch size `P`:
- **Number of patches**: `N = (H × W) / P²`
- **Patch dimension**: `P² × C`
- **Sequence length**: `N + 1` (including CLS token)

### 🛠️ Implementation: Patch Embedding Layer

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

class PatchEmbedding(nn.Module):
    """
    Convert image into sequence of patch embeddings
    
    Args:
        img_size: Input image size (assumed square)
        patch_size: Size of each patch
        in_channels: Number of input channels (3 for RGB)
        embed_dim: Embedding dimension
    """
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        self.embed_dim = embed_dim
        
        # Learnable linear projection (can be implemented as conv2d)
        self.projection = nn.Conv2d(
            in_channels, 
            embed_dim, 
            kernel_size=patch_size, 
            stride=patch_size
        )
        
    def forward(self, x):
        """
        Args:
            x: (batch_size, channels, height, width)
        Returns:
            patch_embeddings: (batch_size, n_patches, embed_dim)
        """
        batch_size = x.shape[0]
        
        # Project patches to embedding dimension
        # (B, C, H, W) → (B, embed_dim, H/P, W/P)
        x = self.projection(x)
        
        # Flatten spatial dimensions
        # (B, embed_dim, H/P, W/P) → (B, embed_dim, N)
        x = x.flatten(2)
        
        # Transpose to get sequence format
        # (B, embed_dim, N) → (B, N, embed_dim)
        x = x.transpose(1, 2)
        
        return x

# Test the patch embedding
def test_patch_embedding():
    # Create sample image batch
    batch_size = 2
    img_size = 224
    patch_size = 16
    embed_dim = 768
    
    # Random image batch
    images = torch.randn(batch_size, 3, img_size, img_size)
    
    # Create patch embedding layer
    patch_embed = PatchEmbedding(img_size, patch_size, 3, embed_dim)
    
    # Forward pass
    patch_embeddings = patch_embed(images)
    
    print(f"Input shape: {images.shape}")
    print(f"Output shape: {patch_embeddings.shape}")
    print(f"Expected patches: {(img_size // patch_size) ** 2}")
    
    # Visualize patches
    visualize_patches(images[0], patch_size)

def visualize_patches(image, patch_size):
    """Visualize how an image is divided into patches"""
    # Convert to numpy and rearrange for plotting
    img_np = image.permute(1, 2, 0).numpy()
    img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())
    
    height, width = img_np.shape[:2]
    n_patches_h = height // patch_size
    n_patches_w = width // patch_size
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    
    # Original image
    axes[0, 0].imshow(img_np)
    axes[0, 0].set_title("Original Image")
    axes[0, 0].axis('off')
    
    # Image with patch grid
    axes[0, 1].imshow(img_np)
    for i in range(0, height, patch_size):
        axes[0, 1].axhline(y=i, color='red', linewidth=1)
    for j in range(0, width, patch_size):
        axes[0, 1].axvline(x=j, color='red', linewidth=1)
    axes[0, 1].set_title(f"Patch Grid ({patch_size}x{patch_size})")
    axes[0, 1].axis('off')
    
    # Show some individual patches
    patch_examples = []
    for i in range(min(4, n_patches_h)):
        for j in range(min(4, n_patches_w)):
            if len(patch_examples) < 16:
                start_h = i * patch_size
                end_h = start_h + patch_size
                start_w = j * patch_size
                end_w = start_w + patch_size
                patch = img_np[start_h:end_h, start_w:end_w]
                patch_examples.append(patch)
    
    # Display patch examples
    for idx, (i, j) in enumerate([(1, 0), (1, 1)]):
        if idx < len(patch_examples):
            # Show grid of patches
            grid_size = 4
            patch_grid = np.zeros((grid_size * patch_size, grid_size * patch_size, 3))
            for row in range(grid_size):
                for col in range(grid_size):
                    patch_idx = row * grid_size + col
                    if patch_idx < len(patch_examples):
                        start_h = row * patch_size
                        end_h = start_h + patch_size
                        start_w = col * patch_size
                        end_w = start_w + patch_size
                        patch_grid[start_h:end_h, start_w:end_w] = patch_examples[patch_idx]
            
            axes[i, j].imshow(patch_grid)
            axes[i, j].set_title(f"First {grid_size}x{grid_size} Patches")
            axes[i, j].axis('off')
    
    plt.tight_layout()
    plt.show()

# Run the test
test_patch_embedding()
```

---

## 🧭 Component 2: Positional Encoding

### Why Positional Encoding?

**Problem**: Transformers are permutation-invariant
- Self-attention doesn't care about token order
- Image patches have important spatial relationships
- Need to inject positional information

**Solution**: Add learnable position embeddings

### Types of Position Encoding

1. **Learnable Embeddings** (ViT default)
2. **Fixed Sinusoidal** (Original Transformer)
3. **Relative Position** (Recent variants)

### 🛠️ Implementation: Position Embeddings

```python
class PositionalEmbedding(nn.Module):
    """
    Learnable positional embeddings for Vision Transformer
    """
    def __init__(self, n_patches, embed_dim, dropout=0.1):
        super().__init__()
        self.n_patches = n_patches
        self.embed_dim = embed_dim
        
        # Learnable positional embeddings
        # +1 for CLS token
        self.position_embeddings = nn.Parameter(
            torch.zeros(1, n_patches + 1, embed_dim)
        )
        
        # Initialize with small random values
        nn.init.normal_(self.position_embeddings, std=0.02)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        """
        Args:
            x: (batch_size, n_patches + 1, embed_dim)
        Returns:
            x: (batch_size, n_patches + 1, embed_dim)
        """
        # Add positional embeddings
        x = x + self.position_embeddings
        
        # Apply dropout
        x = self.dropout(x)
        
        return x

def visualize_position_embeddings(pos_embed, patch_size=16, img_size=224):
    """Visualize learned positional embeddings"""
    n_patches_per_side = img_size // patch_size
    n_patches = n_patches_per_side ** 2
    
    # Extract patch positions (exclude CLS token)
    patch_pos_embed = pos_embed[0, 1:, :].detach()  # (n_patches, embed_dim)
    
    # Compute cosine similarity between positions
    patch_pos_embed_norm = F.normalize(patch_pos_embed, dim=1)
    similarity_matrix = torch.mm(patch_pos_embed_norm, patch_pos_embed_norm.t())
    
    # Reshape to spatial grid
    similarity_spatial = similarity_matrix.view(n_patches_per_side, n_patches_per_side, 
                                              n_patches_per_side, n_patches_per_side)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Show similarity for different reference patches
    reference_positions = [(0, 0), (7, 7), (13, 13)]  # corners and center
    
    for idx, (ref_i, ref_j) in enumerate(reference_positions):
        similarity_map = similarity_spatial[ref_i, ref_j, :, :]
        
        axes[0, idx].imshow(similarity_map.numpy(), cmap='viridis')
        axes[0, idx].set_title(f'Position Similarity\nRef: ({ref_i}, {ref_j})')
        axes[0, idx].axis('off')
    
    # Show first few principal components of position embeddings
    U, S, V = torch.svd(patch_pos_embed)
    
    for i in range(3):
        component = U[:, i].view(n_patches_per_side, n_patches_per_side)
        axes[1, i].imshow(component.numpy(), cmap='RdBu')
        axes[1, i].set_title(f'PC {i+1} (var: {S[i]/S.sum():.2%})')
        axes[1, i].axis('off')
    
    plt.tight_layout()
    plt.show()

# Test positional embeddings
n_patches = (224 // 16) ** 2  # 196 patches
embed_dim = 768
pos_embed = PositionalEmbedding(n_patches, embed_dim)

# Generate some dummy embeddings to visualize
dummy_tokens = torch.randn(1, n_patches + 1, embed_dim)
embedded_tokens = pos_embed(dummy_tokens)

print(f"Position embedding shape: {pos_embed.position_embeddings.shape}")
visualize_position_embeddings(pos_embed.position_embeddings)
```

---

## 🔗 Component 3: CLS Token and Complete Embedding

### The CLS Token

**Purpose**: Global representation for classification
- **Origin**: Borrowed from BERT
- **Function**: Aggregates information from all patches
- **Usage**: Fed to classification head

### 🛠️ Complete Vision Transformer Embedding

```python
class VisionTransformerEmbedding(nn.Module):
    """
    Complete ViT embedding: patches + position + CLS token
    """
    def __init__(self, img_size=224, patch_size=16, in_channels=3, 
                 embed_dim=768, dropout=0.1):
        super().__init__()
        
        # Patch embedding
        self.patch_embedding = PatchEmbedding(
            img_size, patch_size, in_channels, embed_dim
        )
        
        # CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.normal_(self.cls_token, std=0.02)
        
        # Positional embedding
        n_patches = self.patch_embedding.n_patches
        self.pos_embedding = PositionalEmbedding(n_patches, embed_dim, dropout)
        
    def forward(self, x):
        """
        Args:
            x: (batch_size, channels, height, width)
        Returns:
            embeddings: (batch_size, n_patches + 1, embed_dim)
        """
        batch_size = x.shape[0]
        
        # Get patch embeddings
        patch_embeddings = self.patch_embedding(x)  # (B, N, D)
        
        # Expand CLS token for batch
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # (B, 1, D)
        
        # Concatenate CLS token with patch embeddings
        embeddings = torch.cat([cls_tokens, patch_embeddings], dim=1)  # (B, N+1, D)
        
        # Add positional embeddings
        embeddings = self.pos_embedding(embeddings)
        
        return embeddings

def comprehensive_embedding_test():
    """Test the complete embedding pipeline"""
    
    # Parameters
    batch_size = 4
    img_size = 224
    patch_size = 16
    embed_dim = 768
    
    # Create random images
    images = torch.randn(batch_size, 3, img_size, img_size)
    
    # Create embedding module
    vit_embedding = VisionTransformerEmbedding(
        img_size, patch_size, 3, embed_dim
    )
    
    # Forward pass
    embeddings = vit_embedding(images)
    
    print("=== ViT Embedding Test ===")
    print(f"Input images shape: {images.shape}")
    print(f"Output embeddings shape: {embeddings.shape}")
    print(f"Expected sequence length: {(img_size//patch_size)**2 + 1}")
    print(f"CLS token shape: {vit_embedding.cls_token.shape}")
    print(f"Positional embeddings shape: {vit_embedding.pos_embedding.position_embeddings.shape}")
    
    # Verify CLS token is at position 0
    cls_output = embeddings[:, 0, :]  # Should be CLS token
    print(f"CLS token output shape: {cls_output.shape}")
    
    # Check that different images produce different CLS tokens
    cos_sim = F.cosine_similarity(cls_output[0], cls_output[1], dim=0)
    print(f"CLS token similarity between samples: {cos_sim:.4f}")

# Run comprehensive test
comprehensive_embedding_test()
```

---

## 🎯 **Hands-on Exercise**: Complete Patch Embedding Implementation

### Task
Implement a full patch embedding layer with the following requirements:

1. **Flexible patch sizes**: Support 8x8, 16x16, and 32x32 patches
2. **Multiple image sizes**: Support 224x224 and 384x384 images
3. **Visualization tools**: Create patch and embedding visualizations
4. **Performance analysis**: Compare computational costs

### 🛠️ Implementation Challenge

```python
class FlexiblePatchEmbedding(nn.Module):
    """
    Flexible patch embedding supporting multiple configurations
    """
    def __init__(self, img_size=224, patch_size=16, in_channels=3, 
                 embed_dim=768, flatten_patches=True):
        super().__init__()
        
        # Validate inputs
        assert img_size % patch_size == 0, f"Image size {img_size} must be divisible by patch size {patch_size}"
        
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.flatten_patches = flatten_patches
        
        # Calculate number of patches
        self.n_patches = (img_size // patch_size) ** 2
        self.patch_dim = in_channels * patch_size * patch_size
        
        # Two implementation options:
        if flatten_patches:
            # Option 1: Manual patching + linear layer
            self.projection = nn.Linear(self.patch_dim, embed_dim)
        else:
            # Option 2: Convolutional layer
            self.projection = nn.Conv2d(
                in_channels, embed_dim, 
                kernel_size=patch_size, stride=patch_size
            )
    
    def manual_patching(self, x):
        """Manual implementation of patch extraction"""
        batch_size, channels, height, width = x.shape
        
        # Reshape to patches
        # (B, C, H, W) → (B, C, H//P, P, W//P, P)
        x = x.view(
            batch_size, channels,
            height // self.patch_size, self.patch_size,
            width // self.patch_size, self.patch_size
        )
        
        # Rearrange to sequence of patches
        # (B, C, H//P, P, W//P, P) → (B, H//P, W//P, C, P, P)
        x = x.permute(0, 2, 4, 1, 3, 5)
        
        # Flatten patches
        # (B, H//P, W//P, C, P, P) → (B, N, C*P*P)
        x = x.contiguous().view(batch_size, self.n_patches, self.patch_dim)
        
        return x
    
    def forward(self, x):
        """
        Args:
            x: (batch_size, channels, height, width)
        Returns:
            patch_embeddings: (batch_size, n_patches, embed_dim)
        """
        if self.flatten_patches:
            # Manual patching approach
            patches = self.manual_patching(x)
            embeddings = self.projection(patches)
        else:
            # Convolutional approach
            embeddings = self.projection(x)  # (B, embed_dim, H//P, W//P)
            embeddings = embeddings.flatten(2).transpose(1, 2)  # (B, N, embed_dim)
        
        return embeddings

def compare_patch_embeddings():
    """Compare different patch embedding configurations"""
    
    configs = [
        (224, 8),   # 28x28 = 784 patches
        (224, 16),  # 14x14 = 196 patches  
        (224, 32),  # 7x7 = 49 patches
        (384, 16),  # 24x24 = 576 patches
    ]
    
    embed_dim = 768
    batch_size = 1
    
    results = []
    
    for img_size, patch_size in configs:
        # Create input
        x = torch.randn(batch_size, 3, img_size, img_size)
        
        # Test both implementations
        for flatten in [True, False]:
            model = FlexiblePatchEmbedding(
                img_size, patch_size, 3, embed_dim, flatten
            )
            
            # Time the forward pass
            import time
            start_time = time.time()
            with torch.no_grad():
                output = model(x)
            end_time = time.time()
            
            # Calculate memory usage
            n_params = sum(p.numel() for p in model.parameters())
            
            results.append({
                'img_size': img_size,
                'patch_size': patch_size,
                'method': 'manual' if flatten else 'conv',
                'n_patches': model.n_patches,
                'output_shape': output.shape,
                'time_ms': (end_time - start_time) * 1000,
                'n_params': n_params
            })
    
    # Display results
    print("=== Patch Embedding Comparison ===")
    print(f"{'Config':<12} {'Method':<8} {'Patches':<8} {'Time(ms)':<10} {'Params':<8}")
    print("-" * 60)
    
    for result in results:
        config = f"{result['img_size']}/{result['patch_size']}"
        print(f"{config:<12} {result['method']:<8} {result['n_patches']:<8} "
              f"{result['time_ms']:<10.2f} {result['n_params']:<8}")

# Run comparison
compare_patch_embeddings()
```

### 📊 Analysis Questions

1. **Computational Trade-offs**:
   - How does patch size affect the number of tokens?
   - Which approach (manual vs conv) is more efficient?

2. **Representation Quality**:
   - What information is lost with larger patches?
   - How does image resolution affect patch granularity?

3. **Memory Considerations**:
   - How does sequence length impact transformer computation?
   - What are the memory trade-offs of different configurations?

---

## 🔗 Connection to DINO

### Why ViT Matters for DINO

1. **Global Attention**: ViT's self-attention enables DINO to learn global image representations
2. **Patch-based Processing**: Aligns with DINO's multi-crop strategy
3. **Scalability**: ViT architectures scale well with DINO's training approach
4. **Representation Quality**: Strong patch-level features benefit self-distillation

### DINO-Specific Considerations

- **Multi-scale Patches**: DINO uses global (224×224) and local (96×96) crops
- **Teacher-Student**: Both networks use identical ViT architectures
- **Attention Maps**: DINO produces high-quality attention visualizations
- **Feature Quality**: ViT features work well for self-supervised objectives

---

## 🎯 Key Takeaways

1. **Vision Transformers** treat images as sequences of patches
2. **Patch embedding** converts spatial data to token representations
3. **Position encoding** preserves spatial relationships
4. **CLS token** provides global image representation
5. **Flexible implementations** allow for different patch sizes and image resolutions
6. **DINO leverages ViT** for high-quality self-supervised learning

---

## 📚 Further Reading

1. **Original Papers**:
   - [An Image is Worth 16x16 Words](https://arxiv.org/abs/2010.11929): Vision Transformer
   - [DeiT](https://arxiv.org/abs/2012.12877): Data-efficient Image Transformers
   - [DINO](https://arxiv.org/abs/2104.14294): Self-Distillation with No Labels

2. **Implementation Resources**:
   - [timm library](https://github.com/rwightman/pytorch-image-models): ViT implementations
   - [Hugging Face Transformers](https://huggingface.co/docs/transformers/model_doc/vit): ViT documentation

---

## ✅ Lesson 1.2 Checklist

- [ ] Understand why transformers work for computer vision
- [ ] Implement patch embedding from scratch
- [ ] Add positional encoding and CLS token
- [ ] Compare different patch size configurations
- [ ] Analyze computational trade-offs
- [ ] Connect ViT concepts to DINO architecture

**Next**: 🔹 Lesson 1.3 - Contrastive Learning vs Knowledge Distillation
