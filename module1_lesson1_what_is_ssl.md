# 🔹 Module 1, Lesson 1.1: What is Self-Supervised Learning?

## 📚 Learning Objectives
By the end of this lesson, you will:
- Understand the fundamentals of self-supervised learning (SSL)
- Distinguish between different SSL approaches: Contrastive, Predictive, and Distillation
- Recognize the advantages and limitations of SSL compared to supervised learning
- Apply SSL concepts to a practical comparison exercise

---

## 🎯 What is Self-Supervised Learning?

**Self-Supervised Learning (SSL)** is a machine learning paradigm where models learn meaningful representations from data without requiring manually labeled examples. Instead of relying on human annotations, SSL creates learning signals directly from the data itself through cleverly designed pretext tasks.

### 🔑 Key Concept: Creating Labels from Data

In SSL, we transform the **unlabeled data** into a **supervised learning problem** by:
1. **Creating pseudo-labels** from the data structure itself
2. **Designing pretext tasks** that force the model to understand data relationships
3. **Learning representations** that capture semantic meaning

---

## 🧩 Three Main Types of Self-Supervised Learning

### 1. 🔄 Contrastive Learning
**Core Idea**: Learn representations by contrasting similar and dissimilar examples.

**How it works**:
- Create **positive pairs** (augmented versions of the same image)
- Create **negative pairs** (different images)
- Train the model to pull positive pairs together and push negative pairs apart in embedding space

**Examples**:
- **SimCLR**: Contrastive Learning of Visual Representations
- **MoCo**: Momentum Contrast for Unsupervised Visual Representation Learning
- **SwAV**: Swapping Assignments between Multiple Views

**Mathematical Foundation**:
```
L_contrastive = -log(exp(sim(z_i, z_j)/τ) / Σ_k exp(sim(z_i, z_k)/τ))
```
Where `sim(·,·)` is cosine similarity and `τ` is temperature parameter.

### 2. 🎯 Predictive Learning
**Core Idea**: Learn by predicting missing or future parts of the data.

**How it works**:
- **Mask** parts of the input (pixels, patches, tokens)
- Train the model to **reconstruct** or **predict** the missing parts
- The model learns meaningful representations to solve the prediction task

**Examples**:
- **MAE**: Masked Autoencoders for Computer Vision
- **BEiT**: BERT Pre-Training of Image Transformers
- **Context Prediction**: Predicting relative positions of image patches

**Mathematical Foundation**:
```
L_predictive = MSE(f(x_masked), x_target)
```
Where `f(·)` is the prediction function and `x_target` is the ground truth.

### 3. 🎓 Knowledge Distillation
**Core Idea**: Learn from a teacher model without requiring negative examples.

**How it works**:
- Use two networks: **student** (learns) and **teacher** (guides)
- Teacher provides "soft targets" instead of hard labels
- Student learns to match teacher's output distributions
- **DINO belongs to this category!**

**Examples**:
- **BYOL**: Bootstrap Your Own Latent
- **DINO**: Self-Distillation with No Labels
- **SwAV**: Can also be viewed as distillation-based

**Mathematical Foundation**:
```
L_distillation = -Σ_i p_teacher(i) * log(p_student(i))
```
Where `p_teacher` and `p_student` are probability distributions.

---

## 🎯 Use Cases and Motivation

### Why Self-Supervised Learning?

#### 🚀 **Advantages**:
1. **Scalability**: Leverage vast amounts of unlabeled data
2. **Cost-Effective**: No expensive manual annotation required
3. **Generalization**: Learn robust, transferable representations
4. **Data Efficiency**: Better performance with limited labeled data downstream

#### 🎯 **Real-World Applications**:
- **Medical Imaging**: Learn from millions of X-rays without diagnoses
- **Satellite Imagery**: Understand earth patterns without geographic labels
- **Video Understanding**: Learn temporal dynamics without action labels
- **Natural Language**: BERT, GPT models use self-supervision

#### 📊 **Industry Impact**:
- **Meta's DINO**: Powers image search and content understanding
- **OpenAI's CLIP**: Text-image understanding without paired supervision
- **Google's SimCLR**: Improves transfer learning across domains

---

## ⚖️ Self-Supervised vs Supervised Learning

| Aspect | Self-Supervised Learning | Supervised Learning |
|--------|-------------------------|-------------------|
| **Data Requirements** | Unlabeled data only | Labeled datasets required |
| **Scalability** | Highly scalable | Limited by labeling cost |
| **Generalization** | Often better transfer | May overfit to specific labels |
| **Training Time** | Longer pretraining | Direct task training |
| **Label Efficiency** | Excellent with few labels | Requires many labels |
| **Representation Quality** | Rich, semantic features | Task-specific features |

### 📈 **Performance Comparison**:
- **Few-shot learning**: SSL often outperforms supervised baselines
- **Transfer learning**: SSL representations transfer better across domains
- **Robustness**: SSL models often more robust to domain shift

---

## 🔬 **Exercise**: Compare SSL vs Supervised on Small Dataset

### Objective
Implement and compare a simple contrastive SSL approach with supervised learning on CIFAR-10 with limited labels.

### Setup
```python
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
import matplotlib.pyplot as plt

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
```

### Part 1: Data Preparation
```python
class ContrastiveTransform:
    """Create two augmented views of the same image for contrastive learning"""
    def __init__(self):
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(32),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), 
                               (0.2023, 0.1994, 0.2010))
        ])
    
    def __call__(self, x):
        return [self.transform(x), self.transform(x)]

# Load CIFAR-10
train_dataset_ssl = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, 
    transform=ContrastiveTransform()
)

# Create small labeled subset (100 examples per class = 1000 total)
train_dataset_supervised = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, 
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), 
                           (0.2023, 0.1994, 0.2010))
    ])
)

# Create small subset for supervised learning
indices = []
targets = np.array(train_dataset_supervised.targets)
for class_idx in range(10):
    class_indices = np.where(targets == class_idx)[0][:100]  # 100 per class
    indices.extend(class_indices)

small_supervised_dataset = Subset(train_dataset_supervised, indices)
```

### Part 2: Simple Contrastive Model
```python
class SimpleEncoder(nn.Module):
    """Simple CNN encoder for CIFAR-10"""
    def __init__(self, output_dim=128):
        super().__init__()
        self.backbone = nn.Sequential(
            # Conv Block 1
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # Conv Block 2
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # Conv Block 3
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # Global Average Pool
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )
        
        # Projection head for contrastive learning
        self.projection_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )
    
    def forward(self, x):
        features = self.backbone(x)
        projections = self.projection_head(features)
        return nn.functional.normalize(projections, dim=1)

class ContrastiveLoss(nn.Module):
    """Simplified contrastive loss (InfoNCE)"""
    def __init__(self, temperature=0.5):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, z1, z2):
        batch_size = z1.size(0)
        
        # Compute similarity matrix
        z = torch.cat([z1, z2], dim=0)
        sim_matrix = torch.mm(z, z.t()) / self.temperature
        
        # Create labels (positive pairs)
        labels = torch.cat([torch.arange(batch_size), 
                           torch.arange(batch_size)]).long()
        
        # Mask out self-similarity
        mask = torch.eye(2 * batch_size).bool()
        sim_matrix.masked_fill_(mask, -float('inf'))
        
        # Compute loss
        loss = nn.functional.cross_entropy(sim_matrix, labels)
        return loss
```

### Part 3: Training Functions
```python
def train_contrastive(model, dataloader, epochs=50):
    """Train contrastive model"""
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = ContrastiveLoss()
    
    losses = []
    for epoch in range(epochs):
        epoch_loss = 0
        for batch_idx, (views, _) in enumerate(dataloader):
            optimizer.zero_grad()
            
            z1 = model(views[0])
            z2 = model(views[1])
            
            loss = criterion(z1, z2)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(dataloader)
        losses.append(avg_loss)
        
        if epoch % 10 == 0:
            print(f'Epoch {epoch}, Loss: {avg_loss:.4f}')
    
    return losses

def train_supervised(model, dataloader, epochs=50):
    """Train supervised model"""
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    
    # Modify model for classification
    model.classifier = nn.Linear(256, 10)  # 10 CIFAR-10 classes
    
    losses = []
    for epoch in range(epochs):
        epoch_loss = 0
        for batch_idx, (data, targets) in enumerate(dataloader):
            optimizer.zero_grad()
            
            features = model.backbone(data)
            outputs = model.classifier(features)
            
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(dataloader)
        losses.append(avg_loss)
        
        if epoch % 10 == 0:
            print(f'Epoch {epoch}, Loss: {avg_loss:.4f}')
    
    return losses
```

### Part 4: Evaluation
```python
def linear_evaluation(encoder, train_loader, test_loader, epochs=20):
    """Linear evaluation protocol for SSL model"""
    # Freeze encoder
    for param in encoder.backbone.parameters():
        param.requires_grad = False
    
    # Add linear classifier
    classifier = nn.Linear(256, 10)
    optimizer = optim.Adam(classifier.parameters(), lr=1e-2)
    criterion = nn.CrossEntropyLoss()
    
    # Train linear classifier
    encoder.eval()
    classifier.train()
    
    for epoch in range(epochs):
        for data, targets in train_loader:
            optimizer.zero_grad()
            
            with torch.no_grad():
                features = encoder.backbone(data)
            
            outputs = classifier(features)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
    
    # Evaluate
    classifier.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, targets in test_loader:
            features = encoder.backbone(data)
            outputs = classifier(features)
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
    
    accuracy = 100 * correct / total
    return accuracy

# Run the comparison
print("Training SSL model...")
ssl_model = SimpleEncoder()
ssl_loader = DataLoader(train_dataset_ssl, batch_size=64, shuffle=True)
ssl_losses = train_contrastive(ssl_model, ssl_loader)

print("Training supervised model...")
supervised_model = SimpleEncoder()
supervised_loader = DataLoader(small_supervised_dataset, batch_size=64, shuffle=True)
supervised_losses = train_supervised(supervised_model, supervised_loader)

# Evaluate both models
test_dataset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True,
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), 
                           (0.2023, 0.1994, 0.2010))
    ])
)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

ssl_accuracy = linear_evaluation(ssl_model, supervised_loader, test_loader)
print(f"SSL Linear Evaluation Accuracy: {ssl_accuracy:.2f}%")

# Note: supervised_model already has classifier trained
# For fair comparison, you would evaluate it directly
```

### 📊 Expected Results and Analysis

**Typical findings**:
1. **SSL model** may achieve 60-70% accuracy with linear evaluation
2. **Supervised model** with same amount of labeled data may achieve 55-65%
3. **Key insight**: SSL leverages ALL unlabeled data, not just the 1000 labeled examples

### 🤔 Discussion Questions
1. Why might SSL perform better with limited labels?
2. How would results change with more labeled data?
3. What happens if we use the full CIFAR-10 dataset for SSL pretraining?

---

## 🎯 Key Takeaways

1. **Self-supervised learning** creates learning signals from data structure itself
2. **Three main approaches**: Contrastive, Predictive, and Distillation
3. **Major advantage**: Leverages unlimited unlabeled data
4. **DINO uses distillation**: No negative examples needed
5. **Transfer learning**: SSL often provides better representations for downstream tasks

---

## 📚 Further Reading

1. **Original Papers**:
   - [SimCLR](https://arxiv.org/abs/2002.05709): A Simple Framework for Contrastive Learning
   - [BYOL](https://arxiv.org/abs/2006.07733): Bootstrap Your Own Latent
   - [DINO](https://arxiv.org/abs/2104.14294): Emerging Properties in Self-Supervised Vision Transformers

2. **Surveys**:
   - [Self-supervised Visual Feature Learning with Deep Neural Networks](https://arxiv.org/abs/1902.06162)
   - [A Survey on Self-Supervised Learning](https://arxiv.org/abs/2011.00362)

---

## ✅ Lesson 1.1 Checklist

- [ ] Understand the three types of SSL approaches
- [ ] Complete the SSL vs Supervised comparison exercise
- [ ] Analyze results and discuss findings
- [ ] Identify why SSL can be beneficial for limited labeled data scenarios
- [ ] Ready to move on to Vision Transformers primer

**Next**: 🔹 Lesson 1.2 - Vision Transformers (ViT) Primer
