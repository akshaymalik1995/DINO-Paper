# 🔹 Module 2, Lesson 2.2: Multi-Crop Data Augmentation Pipeline

## 📚 Learning Objectives
By the end of this lesson, you will:
- Understand DINO's multi-crop strategy in detail
- Implement global and local crop generation with proper augmentations
- Create asymmetric augmentation pipelines for student and teacher
- Build a complete, configurable multi-crop augmentation system
- Test and validate the augmentation pipeline with visualizations

---

## 🎯 DINO Multi-Crop Strategy Deep Dive

### Core Concept
DINO's multi-crop strategy is fundamental to its success. It generates multiple views of each image at different scales and resolutions, enabling the model to learn both global and local features simultaneously.

### Key Components

#### 1. **Global Crops (224×224)**
- **Purpose**: Capture overall object structure and context
- **Count**: 2 crops per image
- **Scale**: 50-100% of original image
- **Processing**: Both teacher and student networks see these

#### 2. **Local Crops (96×96)**  
- **Purpose**: Focus on fine-grained details and parts
- **Count**: 6-10 crops per image (paper uses 8)
- **Scale**: 5-50% of original image
- **Processing**: Only student network sees these

#### 3. **Asymmetric Processing**
- **Teacher**: Sees only global crops (stable, high-level features)
- **Student**: Sees all crops (learns from both global and local)

---

## 🔍 Multi-Crop Mathematical Foundation

### Loss Computation Strategy

```python
# Pseudocode for DINO multi-crop loss
def compute_multicrop_loss(student_outputs, teacher_outputs):
    """
    student_outputs: [global_1, global_2, local_1, ..., local_8]  # 10 outputs
    teacher_outputs: [global_1, global_2]                        # 2 outputs
    """
    total_loss = 0
    n_loss_terms = 0
    
    for teacher_out in teacher_outputs:  # Global crops only
        for student_out in student_outputs:  # All crops
            if not same_crop(teacher_out, student_out):
                loss = cross_entropy(student_out, teacher_out)
                total_loss += loss
                n_loss_terms += 1
    
    return total_loss / n_loss_terms
```

### Benefits of Multi-Crop Strategy

1. **Multi-Scale Learning**: Captures features at different scales
2. **Data Efficiency**: More views per image without storing more data
3. **Computational Efficiency**: Smaller local crops reduce computation
4. **Robustness**: Diverse views prevent overfitting to specific crops

---

## 🛠️ Implementation: Multi-Crop Augmentation System

### Core Augmentation Classes (`data/augmentations.py`)

```python
import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from PIL import Image, ImageFilter
import random
import numpy as np
from typing import List, Tuple, Union


class GaussianBlur:
    """
    Apply Gaussian Blur to PIL Image with random kernel size and sigma
    """
    def __init__(self, kernel_size_range: Tuple[int, int] = (9, 23), 
                 sigma_range: Tuple[float, float] = (0.1, 2.0)):
        self.kernel_size_range = kernel_size_range
        self.sigma_range = sigma_range
    
    def __call__(self, img: Image.Image) -> Image.Image:
        # Random kernel size (must be odd)
        kernel_size = random.randint(*self.kernel_size_range)
        if kernel_size % 2 == 0:
            kernel_size += 1
        
        # Random sigma
        sigma = random.uniform(*self.sigma_range)
        
        return img.filter(ImageFilter.GaussianBlur(radius=sigma))


class Solarization:
    """
    Solarization augmentation as used in DINO
    """
    def __init__(self, threshold: int = 128):
        self.threshold = threshold
    
    def __call__(self, img: Image.Image) -> Image.Image:
        return F.solarize(img, threshold=self.threshold)


class MultiCropAugmentation:
    """
    DINO Multi-crop augmentation strategy
    
    Generates multiple crops at different scales:
    - Global crops: Large crops covering most of the image
    - Local crops: Smaller crops focusing on image parts
    """
    
    def __init__(self, 
                 global_crops_number: int = 2,
                 local_crops_number: int = 8,
                 global_crops_scale: Tuple[float, float] = (0.4, 1.0),
                 local_crops_scale: Tuple[float, float] = (0.05, 0.4),
                 global_crop_size: int = 224,
                 local_crop_size: int = 96):
        
        self.global_crops_number = global_crops_number
        self.local_crops_number = local_crops_number
        self.global_crops_scale = global_crops_scale
        self.local_crops_scale = local_crops_scale
        self.global_crop_size = global_crop_size
        self.local_crop_size = local_crop_size
        
        # Color augmentation parameters (from DINO paper)
        self.color_jitter = transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],  # ImageNet stats
            std=[0.229, 0.224, 0.225]
        )
        
        # Global crop transforms
        self.global_transform_1 = transforms.Compose([
            transforms.RandomResizedCrop(
                global_crop_size, 
                scale=global_crops_scale,
                interpolation=transforms.InterpolationMode.BICUBIC
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            self.color_jitter,
            transforms.RandomGrayscale(p=0.2),
            GaussianBlur(kernel_size_range=(9, 23), sigma_range=(0.1, 2.0)),
            transforms.ToTensor(),
            self.normalize,
        ])
        
        self.global_transform_2 = transforms.Compose([
            transforms.RandomResizedCrop(
                global_crop_size, 
                scale=global_crops_scale,
                interpolation=transforms.InterpolationMode.BICUBIC
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            self.color_jitter,
            transforms.RandomGrayscale(p=0.2),
            GaussianBlur(kernel_size_range=(9, 23), sigma_range=(0.1, 2.0)),
            Solarization(),  # Only in second global crop
            transforms.ToTensor(),
            self.normalize,
        ])
        
        # Local crop transform
        self.local_transform = transforms.Compose([
            transforms.RandomResizedCrop(
                local_crop_size, 
                scale=local_crops_scale,
                interpolation=transforms.InterpolationMode.BICUBIC
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            self.color_jitter,
            transforms.RandomGrayscale(p=0.2),
            GaussianBlur(kernel_size_range=(5, 9), sigma_range=(0.1, 2.0)),
            transforms.ToTensor(),
            self.normalize,
        ])
    
    def __call__(self, image: Image.Image) -> List[torch.Tensor]:
        """
        Apply multi-crop augmentation to input image
        
        Args:
            image: PIL Image
            
        Returns:
            List of augmented crops [global_1, global_2, local_1, ..., local_n]
        """
        crops = []
        
        # Generate global crops
        for i in range(self.global_crops_number):
            if i == 0:
                crop = self.global_transform_1(image)
            else:
                crop = self.global_transform_2(image)
            crops.append(crop)
        
        # Generate local crops
        for _ in range(self.local_crops_number):
            crop = self.local_transform(image)
            crops.append(crop)
        
        return crops


class MultiCropAugmentationCIFAR:
    """
    Multi-crop augmentation adapted for CIFAR-10 (32x32 images)
    """
    
    def __init__(self, 
                 global_crops_number: int = 2,
                 local_crops_number: int = 6,  # Fewer local crops for small images
                 global_crops_scale: Tuple[float, float] = (0.7, 1.0),
                 local_crops_scale: Tuple[float, float] = (0.3, 0.7),
                 global_crop_size: int = 32,
                 local_crop_size: int = 16):
        
        self.global_crops_number = global_crops_number
        self.local_crops_number = local_crops_number
        
        # CIFAR-10 normalization
        self.normalize = transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465],  # CIFAR-10 stats
            std=[0.2023, 0.1994, 0.2010]
        )
        
        # Global crop transforms (adapted for small images)
        self.global_transform_1 = transforms.Compose([
            transforms.RandomResizedCrop(
                global_crop_size, 
                scale=global_crops_scale,
                interpolation=transforms.InterpolationMode.BICUBIC
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
            transforms.RandomGrayscale(p=0.2),
            # Smaller blur for small images
            GaussianBlur(kernel_size_range=(3, 7), sigma_range=(0.1, 1.0)),
            transforms.ToTensor(),
            self.normalize,
        ])
        
        self.global_transform_2 = transforms.Compose([
            transforms.RandomResizedCrop(
                global_crop_size, 
                scale=global_crops_scale,
                interpolation=transforms.InterpolationMode.BICUBIC
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
            transforms.RandomGrayscale(p=0.2),
            GaussianBlur(kernel_size_range=(3, 7), sigma_range=(0.1, 1.0)),
            Solarization(),
            transforms.ToTensor(),
            self.normalize,
        ])
        
        # Local crop transform
        self.local_transform = transforms.Compose([
            transforms.RandomResizedCrop(
                local_crop_size, 
                scale=local_crops_scale,
                interpolation=transforms.InterpolationMode.BICUBIC
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(0.3, 0.3, 0.3, 0.1),  # Slightly less aggressive
            transforms.RandomGrayscale(p=0.2),
            GaussianBlur(kernel_size_range=(3, 5), sigma_range=(0.1, 1.0)),
            transforms.ToTensor(),
            self.normalize,
        ])
    
    def __call__(self, image: Image.Image) -> List[torch.Tensor]:
        """Apply multi-crop augmentation for CIFAR-10"""
        crops = []
        
        # Global crops
        for i in range(self.global_crops_number):
            if i == 0:
                crop = self.global_transform_1(image)
            else:
                crop = self.global_transform_2(image)
            crops.append(crop)
        
        # Local crops
        for _ in range(self.local_crops_number):
            crop = self.local_transform(image)
            crops.append(crop)
        
        return crops


def get_multicrop_augmentation(config):
    """
    Factory function to create appropriate multi-crop augmentation
    
    Args:
        config: Configuration object
        
    Returns:
        MultiCrop augmentation instance
    """
    dataset = config.data.dataset.lower()
    
    if dataset == 'cifar10':
        return MultiCropAugmentationCIFAR(
            global_crops_number=config.data.global_crops_number,
            local_crops_number=config.data.local_crops_number,
            global_crops_scale=config.data.global_crops_scale,
            local_crops_scale=config.data.local_crops_scale,
            global_crop_size=config.data.global_crop_size,
            local_crop_size=config.data.local_crop_size,
        )
    
    else:  # ImageNet and others
        return MultiCropAugmentation(
            global_crops_number=config.data.global_crops_number,
            local_crops_number=config.data.local_crops_number,
            global_crops_scale=config.data.global_crops_scale,
            local_crops_scale=config.data.local_crops_scale,
            global_crop_size=config.data.global_crop_size,
            local_crop_size=config.data.local_crop_size,
        )
```

---

## 📊 DataLoader Integration (`data/dataloaders.py`)

```python
import torch
from torch.utils.data import DataLoader
from typing import List, Tuple, Any


class MultiCropDataLoader:
    """
    DataLoader wrapper for multi-crop augmentation
    
    Handles batching of variable-sized crop lists and provides utilities
    for separating global and local crops during training.
    """
    
    def __init__(self, dataset, config, shuffle=True):
        self.config = config
        self.global_crops_number = config.data.global_crops_number
        self.local_crops_number = config.data.local_crops_number
        
        self.dataloader = DataLoader(
            dataset,
            batch_size=config.data.batch_size,
            shuffle=shuffle,
            num_workers=config.data.num_workers,
            pin_memory=config.data.pin_memory,
            drop_last=True,
            collate_fn=self.collate_fn,
            persistent_workers=True if config.data.num_workers > 0 else False
        )
    
    def collate_fn(self, batch):
        """
        Custom collate function to handle multi-crop batches
        
        Args:
            batch: List of (crops, label) tuples
            
        Returns:
            (batched_crops, labels) where batched_crops is list of crop batches
        """
        crops_list = []
        labels = []
        
        for crops, label in batch:
            crops_list.append(crops)
            labels.append(label)
        
        # Transpose: from list of crop_lists to list of batches
        n_crops = len(crops_list[0])
        batched_crops = []
        
        for crop_idx in range(n_crops):
            crop_batch = torch.stack([crops_list[i][crop_idx] for i in range(len(crops_list))])
            batched_crops.append(crop_batch)
        
        labels = torch.tensor(labels)
        
        return batched_crops, labels
    
    def __iter__(self):
        return iter(self.dataloader)
    
    def __len__(self):
        return len(self.dataloader)
    
    def split_crops(self, crops: List[torch.Tensor]) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        Split crops into global and local
        
        Args:
            crops: List of crop tensors [global_1, global_2, local_1, ..., local_n]
            
        Returns:
            (global_crops, local_crops)
        """
        global_crops = crops[:self.global_crops_number]
        local_crops = crops[self.global_crops_number:]
        
        return global_crops, local_crops


def create_dino_dataloader(dataset, config, shuffle=True):
    """
    Create DataLoader configured for DINO training
    
    Args:
        dataset: Dataset with multi-crop augmentation
        config: Configuration object
        shuffle: Whether to shuffle data
        
    Returns:
        MultiCropDataLoader instance
    """
    return MultiCropDataLoader(dataset, config, shuffle)
```

---

## 🎨 Visualization Tools (`utils/visualization.py`)

```python
import torch
import matplotlib.pyplot as plt
import numpy as np
from typing import List
import torchvision.transforms as transforms


def denormalize_tensor(tensor: torch.Tensor, mean: List[float], std: List[float]) -> torch.Tensor:
    """
    Denormalize a tensor for visualization
    
    Args:
        tensor: Normalized tensor (C, H, W) or (B, C, H, W)
        mean: Normalization mean
        std: Normalization std
        
    Returns:
        Denormalized tensor
    """
    mean = torch.tensor(mean).view(-1, 1, 1)
    std = torch.tensor(std).view(-1, 1, 1)
    
    if tensor.dim() == 4:  # Batch dimension
        mean = mean.unsqueeze(0)
        std = std.unsqueeze(0)
    
    return tensor * std + mean


def visualize_multicrop_batch(crops: List[torch.Tensor], 
                             global_crops_number: int = 2,
                             sample_idx: int = 0,
                             save_path: str = None):
    """
    Visualize multi-crop augmentation for a single sample
    
    Args:
        crops: List of crop tensors from DataLoader
        global_crops_number: Number of global crops
        sample_idx: Which sample to visualize from the batch
        save_path: Optional path to save the visualization
    """
    # Split crops
    global_crops = crops[:global_crops_number]
    local_crops = crops[global_crops_number:]
    
    n_local = len(local_crops)
    total_crops = len(crops)
    
    # Create subplot grid
    fig, axes = plt.subplots(2, max(global_crops_number, n_local // 2), 
                            figsize=(15, 6))
    
    # Normalization stats (adjust based on dataset)
    mean = [0.485, 0.456, 0.406]  # ImageNet
    std = [0.229, 0.224, 0.225]
    
    # Plot global crops
    for i, crop in enumerate(global_crops):
        img = denormalize_tensor(crop[sample_idx], mean, std)
        img = torch.clamp(img, 0, 1)
        img = img.permute(1, 2, 0).numpy()
        
        axes[0, i].imshow(img)
        axes[0, i].set_title(f'Global Crop {i+1}\n{crop.shape[2]}×{crop.shape[3]}')
        axes[0, i].axis('off')
    
    # Hide unused global crop axes
    for i in range(global_crops_number, axes.shape[1]):
        axes[0, i].axis('off')
    
    # Plot local crops
    n_cols = axes.shape[1]
    for i, crop in enumerate(local_crops[:n_cols]):
        img = denormalize_tensor(crop[sample_idx], mean, std)
        img = torch.clamp(img, 0, 1)
        img = img.permute(1, 2, 0).numpy()
        
        axes[1, i].imshow(img)
        axes[1, i].set_title(f'Local Crop {i+1}\n{crop.shape[2]}×{crop.shape[3]}')
        axes[1, i].axis('off')
    
    # Hide unused local crop axes
    for i in range(len(local_crops), n_cols):
        axes[1, i].axis('off')
    
    plt.suptitle(f'DINO Multi-Crop Augmentation (Sample {sample_idx})', fontsize=16)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def visualize_augmentation_effects(dataset, config, n_samples: int = 4):
    """
    Visualize the effect of different augmentations
    
    Args:
        dataset: Dataset with multi-crop augmentation
        config: Configuration object
        n_samples: Number of samples to visualize
    """
    fig, axes = plt.subplots(n_samples, 6, figsize=(18, 3 * n_samples))
    
    for sample_idx in range(n_samples):
        crops, _ = dataset[sample_idx]
        
        # Show first 6 crops (2 global + 4 local)
        for crop_idx in range(min(6, len(crops))):
            crop = crops[crop_idx]
            
            # Denormalize
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]
            img = denormalize_tensor(crop, mean, std)
            img = torch.clamp(img, 0, 1)
            img = img.permute(1, 2, 0).numpy()
            
            axes[sample_idx, crop_idx].imshow(img)
            
            if sample_idx == 0:
                crop_type = "Global" if crop_idx < config.data.global_crops_number else "Local"
                axes[sample_idx, crop_idx].set_title(f'{crop_type} {crop_idx+1}')
            
            axes[sample_idx, crop_idx].axis('off')
    
    plt.suptitle('Multi-Crop Augmentation Diversity', fontsize=16)
    plt.tight_layout()
    plt.show()


def analyze_crop_statistics(dataloader, n_batches: int = 10):
    """
    Analyze statistics of the multi-crop augmentation
    
    Args:
        dataloader: MultiCropDataLoader
        n_batches: Number of batches to analyze
    """
    global_sizes = []
    local_sizes = []
    
    print("📊 Analyzing Multi-Crop Statistics...")
    
    for batch_idx, (crops, _) in enumerate(dataloader):
        if batch_idx >= n_batches:
            break
        
        global_crops, local_crops = dataloader.split_crops(crops)
        
        # Collect sizes
        for crop in global_crops:
            global_sizes.append(crop.shape[2:])  # (H, W)
        
        for crop in local_crops:
            local_sizes.append(crop.shape[2:])
    
    # Print statistics
    print(f"\n🌍 Global Crops Analysis ({len(global_sizes)} samples):")
    print(f"   Crop sizes: {set(global_sizes)}")
    print(f"   Number per image: {len(global_crops)}")
    
    print(f"\n🔍 Local Crops Analysis ({len(local_sizes)} samples):")
    print(f"   Crop sizes: {set(local_sizes)}")
    print(f"   Number per image: {len(local_crops)}")
    
    print(f"\n📈 Total crops per image: {len(crops)}")
    print(f"   Ratio local/global: {len(local_crops)/len(global_crops):.1f}")
```

---

## 🧪 **Hands-on Exercise**: Complete Multi-Crop Implementation

### Task
Implement and test the complete multi-crop augmentation pipeline for DINO.

### Step 1: Create Test Script (`scripts/test_augmentation.py`)

```python
import sys
import os
sys.path.append('.')

import torch
from data.datasets import get_dataset
from data.dataloaders import create_dino_dataloader
from utils.config import load_config
from utils.visualization import visualize_multicrop_batch, visualize_augmentation_effects, analyze_crop_statistics


def test_multicrop_augmentation():
    """Test the multi-crop augmentation pipeline"""
    
    print("🔬 Testing DINO Multi-Crop Augmentation Pipeline")
    print("=" * 50)
    
    # Load configuration
    config = load_config('cifar10_config')
    print(f"✓ Loaded config for dataset: {config.data.dataset}")
    
    # Create dataset
    train_dataset, _ = get_dataset(config)
    print(f"✓ Created dataset with {len(train_dataset)} samples")
    
    # Test single sample
    print("\n📷 Testing single sample augmentation...")
    crops, label = train_dataset[0]
    print(f"   Sample label: {label}")
    print(f"   Number of crops: {len(crops)}")
    print(f"   Global crops: {config.data.global_crops_number}")
    print(f"   Local crops: {config.data.local_crops_number}")
    
    # Print crop shapes
    for i, crop in enumerate(crops):
        crop_type = "Global" if i < config.data.global_crops_number else "Local"
        print(f"   {crop_type} crop {i+1}: {crop.shape}")
    
    # Create dataloader
    print("\n📦 Testing dataloader...")
    dataloader = create_dino_dataloader(train_dataset, config, shuffle=False)
    print(f"   Batch size: {config.data.batch_size}")
    print(f"   Number of batches: {len(dataloader)}")
    
    # Test batch
    crops_batch, labels_batch = next(iter(dataloader))
    print(f"   Batch crops: {len(crops_batch)}")
    print(f"   Batch labels: {labels_batch.shape}")
    
    # Test crop splitting
    global_crops, local_crops = dataloader.split_crops(crops_batch)
    print(f"   Global crops: {len(global_crops)}")
    print(f"   Local crops: {len(local_crops)}")
    
    # Visualize
    print("\n🎨 Creating visualizations...")
    visualize_multicrop_batch(crops_batch, config.data.global_crops_number)
    
    print("\n📊 Analyzing crop statistics...")
    analyze_crop_statistics(dataloader, n_batches=5)
    
    print("\n✅ Multi-crop augmentation test completed successfully!")
    
    return dataloader


def benchmark_augmentation_speed():
    """Benchmark the speed of multi-crop augmentation"""
    import time
    
    print("\n⚡ Benchmarking Augmentation Speed")
    print("=" * 50)
    
    config = load_config('cifar10_config')
    train_dataset, _ = get_dataset(config)
    dataloader = create_dino_dataloader(train_dataset, config, shuffle=False)
    
    # Warm up
    for i, (crops, labels) in enumerate(dataloader):
        if i >= 2:
            break
    
    # Benchmark
    start_time = time.time()
    n_batches = 10
    
    for i, (crops, labels) in enumerate(dataloader):
        if i >= n_batches:
            break
        
        # Simulate some processing
        global_crops, local_crops = dataloader.split_crops(crops)
    
    end_time = time.time()
    
    total_time = end_time - start_time
    time_per_batch = total_time / n_batches
    samples_per_second = (config.data.batch_size * n_batches) / total_time
    
    print(f"   Total time for {n_batches} batches: {total_time:.2f}s")
    print(f"   Time per batch: {time_per_batch:.3f}s")
    print(f"   Samples per second: {samples_per_second:.1f}")
    print(f"   Crops per second: {samples_per_second * (config.data.global_crops_number + config.data.local_crops_number):.1f}")


if __name__ == "__main__":
    # Test augmentation
    dataloader = test_multicrop_augmentation()
    
    # Benchmark speed
    benchmark_augmentation_speed()
    
    print("\n🎉 All tests passed! Multi-crop augmentation is ready for DINO training.")
```

### Step 2: Run the Test

```powershell
# Run the test script
python scripts/test_augmentation.py
```

### Step 3: Advanced Visualization (`notebooks/multicrop_analysis.ipynb`)

```python
# Jupyter notebook for detailed analysis
import sys
sys.path.append('..')

import torch
import matplotlib.pyplot as plt
import numpy as np
from data.datasets import get_dataset
from data.dataloaders import create_dino_dataloader
from utils.config import load_config
from utils.visualization import *

# Load data
config = load_config('cifar10_config')
train_dataset, _ = get_dataset(config)
dataloader = create_dino_dataloader(train_dataset, config)

# Interactive analysis
def interactive_multicrop_analysis():
    """Interactive analysis of multi-crop augmentation"""
    
    # 1. Visualize diversity across samples
    print("1. Visualizing augmentation diversity...")
    visualize_augmentation_effects(train_dataset, config, n_samples=6)
    
    # 2. Analyze scale distributions
    print("2. Analyzing scale distributions...")
    scales_global = []
    scales_local = []
    
    for i in range(100):  # Sample 100 images
        crops, _ = train_dataset[i]
        
        global_crops = crops[:config.data.global_crops_number]
        local_crops = crops[config.data.global_crops_number:]
        
        for crop in global_crops:
            scale = crop.shape[1] * crop.shape[2] / (32 * 32)  # CIFAR-10 base size
            scales_global.append(scale)
        
        for crop in local_crops:
            scale = crop.shape[1] * crop.shape[2] / (32 * 32)
            scales_local.append(scale)
    
    # Plot scale distributions
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.hist(scales_global, bins=20, alpha=0.7, label='Global crops')
    plt.xlabel('Scale (relative to original)')
    plt.ylabel('Count')
    plt.title('Global Crop Scale Distribution')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.hist(scales_local, bins=20, alpha=0.7, label='Local crops', color='orange')
    plt.xlabel('Scale (relative to original)')
    plt.ylabel('Count')
    plt.title('Local Crop Scale Distribution')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    print(f"Global crop scales: {np.min(scales_global):.2f} - {np.max(scales_global):.2f}")
    print(f"Local crop scales: {np.min(scales_local):.2f} - {np.max(scales_local):.2f}")

# Run analysis
interactive_multicrop_analysis()
```

---

## 🔧 Configuration Updates

### Update `configs/cifar10_config.yaml` with optimal multi-crop settings:

```yaml
# Enhanced CIFAR-10 configuration with optimized multi-crop settings
defaults:
  - base_config

experiment:
  name: "dino_cifar10_multicrop"

data:
  dataset: "cifar10"
  batch_size: 128  # Adjusted for multi-crop
  
  # Optimized multi-crop parameters for CIFAR-10
  global_crops_number: 2
  local_crops_number: 6  # Reduced for small images
  global_crops_scale: [0.7, 1.0]  # Higher minimum scale
  local_crops_scale: [0.3, 0.7]   # Adjusted range
  global_crop_size: 32
  local_crop_size: 16

model:
  backbone: "resnet18"
  projection_dim: 8192
  
training:
  batch_size_per_gpu: 64  # Per-GPU batch size
  epochs: 200
```

---

## 📊 Expected Results Analysis

### Performance Metrics to Track

1. **Augmentation Speed**: 
   - Target: >100 samples/second on modern GPU
   - Actual: Will depend on your hardware

2. **Memory Usage**:
   - Global crops: 2 × batch_size × 3 × 32 × 32
   - Local crops: 6 × batch_size × 3 × 16 × 16
   - Total per batch: ~8× single image memory

3. **Diversity Metrics**:
   - Scale distribution should match configured ranges
   - Visual diversity should be high across crops

### Troubleshooting Common Issues

```python
# Debug script for common multi-crop issues
def debug_multicrop_issues():
    """Debug common multi-crop augmentation issues"""
    
    print("🔧 Debugging Multi-Crop Issues")
    print("=" * 40)
    
    # Issue 1: Memory problems
    try:
        config = load_config('cifar10_config')
        train_dataset, _ = get_dataset(config)
        dataloader = create_dino_dataloader(train_dataset, config)
        
        crops, labels = next(iter(dataloader))
        print(f"✓ Memory test passed. Batch size: {labels.shape[0]}")
        
    except RuntimeError as e:
        print(f"❌ Memory error: {e}")
        print("   Try reducing batch_size or local_crops_number")
    
    # Issue 2: Speed problems
    import time
    start = time.time()
    
    for i, (crops, labels) in enumerate(dataloader):
        if i >= 5:
            break
    
    elapsed = time.time() - start
    speed = (5 * config.data.batch_size) / elapsed
    
    if speed < 50:
        print(f"⚠️  Slow augmentation: {speed:.1f} samples/sec")
        print("   Try reducing num_workers or simplifying augmentations")
    else:
        print(f"✓ Good speed: {speed:.1f} samples/sec")
    
    # Issue 3: Crop size problems
    for i, crop in enumerate(crops):
        expected_size = config.data.global_crop_size if i < config.data.global_crops_number else config.data.local_crop_size
        actual_size = crop.shape[2]
        
        if actual_size != expected_size:
            print(f"❌ Crop {i} size mismatch: expected {expected_size}, got {actual_size}")
        else:
            print(f"✓ Crop {i} size correct: {actual_size}")

# Run debug
debug_multicrop_issues()
```

---

## ✅ Lesson 2.2 Checklist

### Core Implementation
- [ ] Implemented GaussianBlur and Solarization augmentations
- [ ] Created MultiCropAugmentation class for ImageNet
- [ ] Created MultiCropAugmentationCIFAR for CIFAR-10
- [ ] Built MultiCropDataLoader with proper batching

### Visualization and Analysis
- [ ] Implemented crop visualization functions
- [ ] Created augmentation diversity analysis tools
- [ ] Built scale distribution analysis
- [ ] Added performance benchmarking

### Testing and Validation
- [ ] Tested single sample augmentation
- [ ] Validated batch processing
- [ ] Verified crop splitting functionality
- [ ] Checked memory usage and speed

### Configuration
- [ ] Updated CIFAR-10 config with optimal multi-crop settings
- [ ] Added debug utilities for troubleshooting
- [ ] Documented expected performance metrics

---

## 🎯 Key Takeaways

1. **Multi-crop Strategy**: Core to DINO's success - generates diverse views efficiently
2. **Asymmetric Processing**: Teacher sees global, student sees all crops
3. **Scale Awareness**: Different crop sizes capture different levels of detail
4. **Implementation Details**: Proper batching and data loading crucial for performance
5. **Configuration Flexibility**: Easy to adapt for different datasets and hardware

**Next**: 🔹 Lesson 2.3 - Backbone Architecture Implementation

Your multi-crop augmentation pipeline is now ready! In the next lesson, we'll implement the backbone networks (ResNet and ViT) with their projection heads, completing the core model architecture for DINO.
