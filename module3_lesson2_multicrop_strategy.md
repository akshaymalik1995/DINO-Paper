# Module 3, Lesson 2: Multi-Crop Strategy Implementation

## 🎯 Learning Objectives
By the end of this lesson, you will:
- Understand DINO's multi-crop augmentation strategy
- Implement global and local crop generation
- Build asymmetric augmentation pipelines for student and teacher
- Create efficient batch construction with multiple views per image

## 📚 Theoretical Background

### DINO's Multi-Crop Strategy

DINO employs a sophisticated multi-crop augmentation strategy:

1. **Global Crops**: Large crops (224×224) covering most of the image
2. **Local Crops**: Smaller crops (96×96) focusing on image details
3. **Asymmetric Processing**: Different crops for student vs. teacher
4. **Cross-Entropy Loss**: Student learns from teacher on all crop combinations

### Why Multi-Crop Works

- **Scale Invariance**: Model learns features at different scales
- **Local-Global Consistency**: Encourages consistent representations
- **Data Efficiency**: Multiple views from single image
- **Regularization**: Prevents overfitting to specific crop sizes

### Mathematical Framework

For an image **I**, we generate:
- **Global crops**: g₁, g₂ ∈ R^(224×224×3) 
- **Local crops**: l₁, l₂, ..., l_V ∈ R^(96×96×3)

**Loss computation**:
```
L = Σ Σ H(P_s(crop_s), P_t(crop_t))
```
Where:
- P_s, P_t are student/teacher probability distributions
- H is cross-entropy loss
- Sum over all student crops and teacher global crops

## 🛠️ Implementation

### Step 1: Multi-Crop Augmentation Pipeline

```python
# multicrop_augmentation.py
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import random
from typing import List, Tuple, Dict, Any
import cv2

class MultiCropAugmentation:
    """
    DINO Multi-crop augmentation strategy
    """
    def __init__(
        self,
        global_crop_size: int = 224,
        local_crop_size: int = 96,
        global_crop_scale: Tuple[float, float] = (0.4, 1.0),
        local_crop_scale: Tuple[float, float] = (0.05, 0.4),
        num_local_crops: int = 8,
        color_jitter_strength: float = 1.0,
        gaussian_blur_prob: float = 1.0,
        normalize: bool = True
    ):
        self.global_crop_size = global_crop_size
        self.local_crop_size = local_crop_size
        self.global_crop_scale = global_crop_scale
        self.local_crop_scale = local_crop_scale
        self.num_local_crops = num_local_crops
        
        # Build augmentation transforms
        self.global_transform1, self.global_transform2 = self._build_global_transforms(
            color_jitter_strength, gaussian_blur_prob, normalize
        )
        self.local_transform = self._build_local_transform(
            color_jitter_strength, gaussian_blur_prob, normalize
        )
    
    def _build_global_transforms(
        self, 
        color_jitter_strength: float, 
        gaussian_blur_prob: float,
        normalize: bool
    ) -> Tuple[transforms.Compose, transforms.Compose]:
        """Build global crop transforms"""
        
        # Normalization values for ImageNet
        normalize_transform = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ) if normalize else transforms.Lambda(lambda x: x)
        
        # Global transform 1 (stronger augmentation)
        global_transform1 = transforms.Compose([
            transforms.RandomResizedCrop(
                self.global_crop_size,
                scale=self.global_crop_scale,
                interpolation=Image.BICUBIC
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(
                brightness=0.4 * color_jitter_strength,
                contrast=0.4 * color_jitter_strength,
                saturation=0.2 * color_jitter_strength,
                hue=0.1 * color_jitter_strength
            ),
            transforms.RandomGrayscale(p=0.2),
            GaussianBlur(p=gaussian_blur_prob),
            transforms.ToTensor(),
            normalize_transform
        ])
        
        # Global transform 2 (lighter augmentation)
        global_transform2 = transforms.Compose([
            transforms.RandomResizedCrop(
                self.global_crop_size,
                scale=self.global_crop_scale,
                interpolation=Image.BICUBIC
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(
                brightness=0.4 * color_jitter_strength,
                contrast=0.4 * color_jitter_strength,
                saturation=0.2 * color_jitter_strength,
                hue=0.1 * color_jitter_strength
            ),
            transforms.RandomGrayscale(p=0.2),
            GaussianBlur(p=0.1),  # Lower blur probability
            Solarization(p=0.2),  # Add solarization
            transforms.ToTensor(),
            normalize_transform
        ])
        
        return global_transform1, global_transform2
    
    def _build_local_transform(
        self, 
        color_jitter_strength: float, 
        gaussian_blur_prob: float,
        normalize: bool
    ) -> transforms.Compose:
        """Build local crop transform"""
        
        normalize_transform = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ) if normalize else transforms.Lambda(lambda x: x)
        
        local_transform = transforms.Compose([
            transforms.RandomResizedCrop(
                self.local_crop_size,
                scale=self.local_crop_scale,
                interpolation=Image.BICUBIC
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(
                brightness=0.4 * color_jitter_strength,
                contrast=0.4 * color_jitter_strength,
                saturation=0.2 * color_jitter_strength,
                hue=0.1 * color_jitter_strength
            ),
            transforms.RandomGrayscale(p=0.2),
            GaussianBlur(p=gaussian_blur_prob * 0.5),  # Lower blur for local crops
            transforms.ToTensor(),
            normalize_transform
        ])
        
        return local_transform
    
    def __call__(self, image: Image.Image) -> Dict[str, torch.Tensor]:
        """
        Apply multi-crop augmentation to an image
        
        Returns:
            Dict with keys: 'global_crops', 'local_crops'
        """
        # Generate global crops
        global_crop1 = self.global_transform1(image)
        global_crop2 = self.global_transform2(image)
        global_crops = torch.stack([global_crop1, global_crop2])
        
        # Generate local crops
        local_crops = []
        for _ in range(self.num_local_crops):
            local_crop = self.local_transform(image)
            local_crops.append(local_crop)
        local_crops = torch.stack(local_crops)
        
        return {
            'global_crops': global_crops,
            'local_crops': local_crops
        }


class GaussianBlur:
    """
    Apply Gaussian Blur to PIL Image
    """
    def __init__(self, p: float = 0.5, radius_min: float = 0.1, radius_max: float = 2.0):
        self.prob = p
        self.radius_min = radius_min
        self.radius_max = radius_max

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() < self.prob:
            radius = random.uniform(self.radius_min, self.radius_max)
            return img.filter(ImageFilter.GaussianBlur(radius=radius))
        return img


class Solarization:
    """
    Apply solarization to PIL Image
    """
    def __init__(self, p: float = 0.2, threshold: int = 128):
        self.prob = p
        self.threshold = threshold

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() < self.prob:
            return ImageOps.solarize(img, threshold=self.threshold)
        return img


# Import required PIL modules
from PIL import ImageFilter, ImageOps
```

### Step 2: Efficient DataLoader with Multi-Crop

```python
# multicrop_dataloader.py
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
from typing import Dict, List, Any
import numpy as np

class MultiCropDataset(Dataset):
    """
    Dataset wrapper for multi-crop augmentation
    """
    def __init__(
        self, 
        base_dataset: Dataset,
        multicrop_transform: MultiCropAugmentation
    ):
        self.base_dataset = base_dataset
        self.multicrop_transform = multicrop_transform
        
    def __len__(self) -> int:
        return len(self.base_dataset)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # Get image from base dataset
        if hasattr(self.base_dataset, 'samples'):
            # ImageFolder-style dataset
            image_path, _ = self.base_dataset.samples[idx]
            image = Image.open(image_path).convert('RGB')
        else:
            # Custom dataset
            image, _ = self.base_dataset[idx]
        
        # Apply multi-crop augmentation
        crops = self.multicrop_transform(image)
        
        return crops


class MultiCropCollator:
    """
    Custom collate function for multi-crop batches
    """
    def __init__(self):
        pass
    
    def __call__(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """
        Collate batch of multi-crop samples
        
        Args:
            batch: List of dicts with 'global_crops' and 'local_crops'
            
        Returns:
            Dict with batched crops
        """
        # Stack global crops
        global_crops = torch.stack([item['global_crops'] for item in batch])
        # Shape: [batch_size, 2, 3, 224, 224]
        
        # Stack local crops
        local_crops = torch.stack([item['local_crops'] for item in batch])
        # Shape: [batch_size, num_local_crops, 3, 96, 96]
        
        # Reshape for easier processing
        batch_size = global_crops.shape[0]
        
        # Flatten global crops: [batch_size * 2, 3, 224, 224]
        global_crops_flat = global_crops.view(-1, *global_crops.shape[2:])
        
        # Flatten local crops: [batch_size * num_local_crops, 3, 96, 96]
        local_crops_flat = local_crops.view(-1, *local_crops.shape[2:])
        
        return {
            'global_crops': global_crops_flat,
            'local_crops': local_crops_flat,
            'batch_size': batch_size
        }


def create_multicrop_dataloader(
    dataset_path: str,
    batch_size: int = 32,
    num_workers: int = 4,
    global_crop_size: int = 224,
    local_crop_size: int = 96,
    num_local_crops: int = 8,
    **kwargs
) -> DataLoader:
    """
    Create a DataLoader with multi-crop augmentation
    """
    
    # Create base dataset
    base_dataset = ImageFolder(dataset_path)
    
    # Create multi-crop transform
    multicrop_transform = MultiCropAugmentation(
        global_crop_size=global_crop_size,
        local_crop_size=local_crop_size,
        num_local_crops=num_local_crops,
        **kwargs
    )
    
    # Wrap with multi-crop dataset
    multicrop_dataset = MultiCropDataset(base_dataset, multicrop_transform)
    
    # Create dataloader with custom collator
    dataloader = DataLoader(
        multicrop_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=MultiCropCollator(),
        pin_memory=True,
        drop_last=True
    )
    
    return dataloader
```

### Step 3: Integration with Student-Teacher Training

```python
# training_integration.py
import torch
import torch.nn as nn
from typing import Dict, Tuple
import torch.nn.functional as F

class MultiCropTrainer:
    """
    Trainer class handling multi-crop strategy with student-teacher networks
    """
    def __init__(
        self,
        student_teacher_model,
        temperature_student: float = 0.1,
        temperature_teacher: float = 0.04,
        center_momentum: float = 0.9
    ):
        self.model = student_teacher_model
        self.temperature_student = temperature_student
        self.temperature_teacher = temperature_teacher
        self.center_momentum = center_momentum
        
        # Initialize center for teacher outputs (prevents collapse)
        self.register_center = True
        self.center = None
        
    def forward_multicrop(
        self, 
        global_crops: torch.Tensor,
        local_crops: torch.Tensor,
        batch_size: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with multi-crop strategy
        
        Args:
            global_crops: [batch_size * 2, 3, 224, 224]
            local_crops: [batch_size * num_local, 3, 96, 96]
            batch_size: Original batch size
            
        Returns:
            student_outputs, teacher_outputs
        """
        
        # Resize local crops to match global crop size
        local_crops_resized = F.interpolate(
            local_crops, 
            size=global_crops.shape[-2:], 
            mode='bilinear', 
            align_corners=False
        )
        
        # Concatenate all crops for student
        all_crops = torch.cat([global_crops, local_crops_resized], dim=0)
        
        # Student forward on all crops
        student_outputs = self.model.forward_student(all_crops)
        
        # Teacher forward only on global crops
        teacher_outputs = self.model.forward_teacher(global_crops)
        
        return student_outputs, teacher_outputs
    
    def compute_dino_loss(
        self,
        student_outputs: torch.Tensor,
        teacher_outputs: torch.Tensor,
        batch_size: int,
        num_local_crops: int = 8
    ) -> torch.Tensor:
        """
        Compute DINO loss with centering
        
        Args:
            student_outputs: [batch_size * (2 + num_local), projection_dim]
            teacher_outputs: [batch_size * 2, projection_dim]
            batch_size: Original batch size
            num_local_crops: Number of local crops per image
        """
        
        # Apply temperature scaling
        student_probs = F.softmax(student_outputs / self.temperature_student, dim=1)
        teacher_probs = F.softmax(
            (teacher_outputs - self.center) / self.temperature_teacher, dim=1
        )
        
        # Reshape outputs
        # Student: separate global and local crops
        num_global = batch_size * 2
        student_global = student_probs[:num_global]
        student_local = student_probs[num_global:]
        
        # Teacher: only global crops
        teacher_global = teacher_probs
        
        # Compute loss: student learns from teacher
        total_loss = 0
        loss_count = 0
        
        # Student global crops learn from teacher global crops
        for i in range(2):  # 2 global crops
            for j in range(2):  # 2 teacher crops
                if i != j:  # Don't compare crop with itself
                    student_batch = student_global[i*batch_size:(i+1)*batch_size]
                    teacher_batch = teacher_global[j*batch_size:(j+1)*batch_size]
                    
                    loss = -torch.sum(teacher_batch * torch.log(student_batch + 1e-8), dim=1)
                    total_loss += loss.mean()
                    loss_count += 1
        
        # Student local crops learn from all teacher global crops
        num_local_per_image = num_local_crops
        for i in range(num_local_per_image):
            for j in range(2):  # 2 teacher global crops
                student_batch = student_local[i*batch_size:(i+1)*batch_size]
                teacher_batch = teacher_global[j*batch_size:(j+1)*batch_size]
                
                loss = -torch.sum(teacher_batch * torch.log(student_batch + 1e-8), dim=1)
                total_loss += loss.mean()
                loss_count += 1
        
        return total_loss / loss_count
    
    def update_center(self, teacher_outputs: torch.Tensor):
        """
        Update center with momentum (prevents mode collapse)
        """
        batch_center = torch.mean(teacher_outputs, dim=0, keepdim=True)
        
        if self.center is None:
            self.center = batch_center.clone()
        else:
            self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)
    
    def training_step(
        self,
        batch: Dict[str, torch.Tensor],
        optimizer: torch.optim.Optimizer,
        epoch: int
    ) -> Dict[str, float]:
        """
        Complete training step with multi-crop
        """
        
        # Extract batch data
        global_crops = batch['global_crops']
        local_crops = batch['local_crops']
        batch_size = batch['batch_size']
        
        # Forward pass
        student_outputs, teacher_outputs = self.forward_multicrop(
            global_crops, local_crops, batch_size
        )
        
        # Update center
        self.update_center(teacher_outputs.detach())
        
        # Compute loss
        loss = self.compute_dino_loss(
            student_outputs, teacher_outputs, batch_size, local_crops.shape[0] // batch_size
        )
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.get_student_parameters(), 3.0)
        
        optimizer.step()
        
        # Update teacher weights
        self.model.update_teacher(epoch)
        
        return {
            'loss': loss.item(),
            'student_norm': torch.norm(student_outputs).item(),
            'teacher_norm': torch.norm(teacher_outputs).item(),
            'center_norm': torch.norm(self.center).item() if self.center is not None else 0.0
        }
```

### Step 4: Visualization and Analysis Tools

```python
# visualization_tools.py
import matplotlib.pyplot as plt
import numpy as np
from torchvision.utils import make_grid
import torch
from PIL import Image

class MultiCropVisualizer:
    """
    Tools for visualizing multi-crop augmentations
    """
    
    @staticmethod
    def visualize_multicrop_sample(
        image: Image.Image,
        multicrop_transform: MultiCropAugmentation,
        save_path: str = None
    ):
        """
        Visualize multi-crop augmentation for a single image
        """
        
        # Apply multi-crop
        crops = multicrop_transform(image)
        global_crops = crops['global_crops']
        local_crops = crops['local_crops']
        
        # Create figure
        fig, axes = plt.subplots(3, max(len(global_crops), len(local_crops)//2), figsize=(15, 10))
        
        # Original image
        axes[0, 0].imshow(image)
        axes[0, 0].set_title("Original Image")
        axes[0, 0].axis('off')
        
        # Remove extra subplots in first row
        for i in range(1, axes.shape[1]):
            fig.delaxes(axes[0, i])
        
        # Global crops
        for i, crop in enumerate(global_crops):
            if i < axes.shape[1]:
                # Convert to displayable format
                crop_img = crop.permute(1, 2, 0)
                # Denormalize if normalized
                crop_img = crop_img * torch.tensor([0.229, 0.224, 0.225]) + torch.tensor([0.485, 0.456, 0.406])
                crop_img = torch.clamp(crop_img, 0, 1)
                
                axes[1, i].imshow(crop_img)
                axes[1, i].set_title(f"Global Crop {i+1}")
                axes[1, i].axis('off')
        
        # Remove extra global crop subplots
        for i in range(len(global_crops), axes.shape[1]):
            fig.delaxes(axes[1, i])
        
        # Local crops (show first few)
        num_local_show = min(len(local_crops), axes.shape[1])
        for i in range(num_local_show):
            crop = local_crops[i]
            # Convert to displayable format
            crop_img = crop.permute(1, 2, 0)
            # Denormalize if normalized
            crop_img = crop_img * torch.tensor([0.229, 0.224, 0.225]) + torch.tensor([0.485, 0.456, 0.406])
            crop_img = torch.clamp(crop_img, 0, 1)
            
            axes[2, i].imshow(crop_img)
            axes[2, i].set_title(f"Local Crop {i+1}")
            axes[2, i].axis('off')
        
        # Remove extra local crop subplots
        for i in range(num_local_show, axes.shape[1]):
            fig.delaxes(axes[2, i])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        plt.show()
    
    @staticmethod
    def visualize_batch(
        dataloader,
        num_samples: int = 4,
        save_path: str = None
    ):
        """
        Visualize a batch from multi-crop dataloader
        """
        
        # Get a batch
        batch = next(iter(dataloader))
        global_crops = batch['global_crops']
        local_crops = batch['local_crops']
        batch_size = batch['batch_size']
        
        # Show first few samples
        fig, axes = plt.subplots(num_samples, 4, figsize=(16, 4*num_samples))
        
        for i in range(min(num_samples, batch_size)):
            # Global crop 1
            img1 = global_crops[i*2].permute(1, 2, 0)
            img1 = img1 * torch.tensor([0.229, 0.224, 0.225]) + torch.tensor([0.485, 0.456, 0.406])
            img1 = torch.clamp(img1, 0, 1)
            axes[i, 0].imshow(img1)
            axes[i, 0].set_title(f"Sample {i+1}: Global 1")
            axes[i, 0].axis('off')
            
            # Global crop 2
            img2 = global_crops[i*2+1].permute(1, 2, 0)
            img2 = img2 * torch.tensor([0.229, 0.224, 0.225]) + torch.tensor([0.485, 0.456, 0.406])
            img2 = torch.clamp(img2, 0, 1)
            axes[i, 1].imshow(img2)
            axes[i, 1].set_title(f"Sample {i+1}: Global 2")
            axes[i, 1].axis('off')
            
            # Local crops (first two)
            num_local_per_image = local_crops.shape[0] // batch_size
            local1 = local_crops[i*num_local_per_image].permute(1, 2, 0)
            local1 = local1 * torch.tensor([0.229, 0.224, 0.225]) + torch.tensor([0.485, 0.456, 0.406])
            local1 = torch.clamp(local1, 0, 1)
            axes[i, 2].imshow(local1)
            axes[i, 2].set_title(f"Sample {i+1}: Local 1")
            axes[i, 2].axis('off')
            
            local2 = local_crops[i*num_local_per_image+1].permute(1, 2, 0)
            local2 = local2 * torch.tensor([0.229, 0.224, 0.225]) + torch.tensor([0.485, 0.456, 0.406])
            local2 = torch.clamp(local2, 0, 1)
            axes[i, 3].imshow(local2)
            axes[i, 3].set_title(f"Sample {i+1}: Local 2")
            axes[i, 3].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        plt.show()

    @staticmethod
    def plot_crop_statistics(dataloader, num_batches: int = 10):
        """
        Plot statistics about crop sizes and distributions
        """
        
        global_sizes = []
        local_sizes = []
        
        for i, batch in enumerate(dataloader):
            if i >= num_batches:
                break
                
            global_crops = batch['global_crops']
            local_crops = batch['local_crops']
            
            global_sizes.extend([224] * len(global_crops))  # All global crops are 224x224
            local_sizes.extend([96] * len(local_crops))    # All local crops are 96x96
        
        # Plot histogram
        plt.figure(figsize=(10, 6))
        plt.hist([global_sizes, local_sizes], bins=50, alpha=0.7, 
                label=['Global Crops', 'Local Crops'])
        plt.xlabel('Crop Size')
        plt.ylabel('Frequency')
        plt.title('Distribution of Crop Sizes')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()


def test_multicrop_implementation():
    """Test the multi-crop implementation"""
    
    # Create sample data
    from torchvision.datasets import FakeData
    from torchvision.transforms import ToPILImage
    
    # Create fake dataset
    fake_dataset = FakeData(size=100, image_size=(3, 256, 256))
    
    # Create multi-crop transform
    multicrop_transform = MultiCropAugmentation(
        global_crop_size=224,
        local_crop_size=96,
        num_local_crops=8
    )
    
    # Test on single image
    image, _ = fake_dataset[0]
    image_pil = ToPILImage()(image)
    
    crops = multicrop_transform(image_pil)
    print(f"Global crops shape: {crops['global_crops'].shape}")
    print(f"Local crops shape: {crops['local_crops'].shape}")
    
    # Create dataloader
    multicrop_dataset = MultiCropDataset(fake_dataset, multicrop_transform)
    dataloader = DataLoader(
        multicrop_dataset,
        batch_size=4,
        collate_fn=MultiCropCollator()
    )
    
    # Test batch
    batch = next(iter(dataloader))
    print(f"Batch global crops: {batch['global_crops'].shape}")
    print(f"Batch local crops: {batch['local_crops'].shape}")
    print(f"Batch size: {batch['batch_size']}")

if __name__ == "__main__":
    test_multicrop_implementation()
```

## 🧪 Hands-on Exercise: Implement Your Multi-Crop Strategy

### Exercise 1: Custom Augmentation Pipeline

Build a simplified multi-crop augmentation:

```python
# exercise1.py
import torch
import torchvision.transforms as transforms
from PIL import Image

class SimpleMultiCrop:
    def __init__(self, global_size=224, local_size=96, num_local=4):
        # TODO: Implement initialization
        pass
    
    def __call__(self, image):
        # TODO: Generate global and local crops
        # Return dict with 'global_crops' and 'local_crops'
        pass

# Test your implementation
transform = SimpleMultiCrop()
# Apply to a sample image and verify shapes
```

### Exercise 2: Asymmetric Augmentation

Implement different augmentation strengths for different crops:

```python
# exercise2.py
def create_asymmetric_transforms():
    """Create transforms with different augmentation strengths"""
    
    # Strong augmentation for global crop 1
    strong_transform = transforms.Compose([
        # TODO: Add strong augmentations
    ])
    
    # Medium augmentation for global crop 2
    medium_transform = transforms.Compose([
        # TODO: Add medium augmentations
    ])
    
    # Light augmentation for local crops
    light_transform = transforms.Compose([
        # TODO: Add light augmentations
    ])
    
    return strong_transform, medium_transform, light_transform

# Test different augmentation strengths
```

### Exercise 3: Batch Processing Analysis

Analyze the efficiency of your multi-crop pipeline:

```python
# exercise3.py
import time
import torch

def benchmark_multicrop_dataloader(dataloader, num_batches=10):
    """Benchmark the multi-crop dataloader"""
    
    start_time = time.time()
    
    for i, batch in enumerate(dataloader):
        if i >= num_batches:
            break
        
        # Simulate processing
        global_crops = batch['global_crops']
        local_crops = batch['local_crops']
        
        # TODO: Add timing and memory usage analysis
    
    end_time = time.time()
    
    print(f"Time per batch: {(end_time - start_time) / num_batches:.4f}s")
    # TODO: Add more detailed analysis

# Run benchmark
```

## 🔍 Key Insights

### Multi-Crop Strategy Benefits
1. **Scale Awareness**: Global and local crops provide different perspectives
2. **Data Efficiency**: Multiple views from single image increase training data
3. **Consistency Learning**: Model learns consistent representations across scales
4. **Regularization**: Prevents overfitting to specific image regions

### Implementation Considerations
1. **Memory Usage**: Multiple crops increase memory requirements
2. **Computational Cost**: More forward passes per image
3. **Augmentation Balance**: Different crops need different augmentation strengths
4. **Batch Construction**: Efficient collation for training

### Common Issues
1. **Memory Overflow**: Too many crops can exceed GPU memory
2. **Slow Training**: Inefficient augmentation can bottleneck training
3. **Inconsistent Normalization**: Different crops must use same normalization
4. **Loss Balancing**: Proper weighting of different crop combinations

## 📝 Summary

In this lesson, you learned:

✅ **Multi-Crop Strategy**: How DINO uses global and local crops for scale-invariant learning

✅ **Asymmetric Augmentation**: Different augmentation strengths for different crop types

✅ **Efficient Implementation**: Optimized data loading and batch processing

✅ **Training Integration**: How multi-crop feeds into student-teacher training

✅ **Visualization Tools**: Methods to analyze and debug your augmentation pipeline

### Next Steps
In the next lesson, we'll implement the projection heads and feature normalization strategies that complete the DINO architecture.

## 🔗 Additional Resources

- [DINO Paper - Multi-crop Strategy](https://arxiv.org/abs/2104.14294)
- [Data Augmentation in Computer Vision](https://journalofbigdata.springeropen.com/articles/10.1186/s40537-019-0197-0)
- [Efficient Data Loading in PyTorch](https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html)

---

**Next**: [Module 3, Lesson 3: Projection Heads and Feature Normalization](module3_lesson3_projection_heads.md)
