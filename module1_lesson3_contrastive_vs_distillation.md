# 🔹 Module 1, Lesson 1.3: Contrastive Learning vs Knowledge Distillation

## 📚 Learning Objectives
By the end of this lesson, you will:
- Compare and contrast major self-supervised learning approaches
- Understand the progression from SimCLR/MoCo to BYOL to DINO
- Grasp why DINO doesn't need negative samples
- Master the mathematical foundations of knowledge distillation
- Complete a theory assessment on SSL approaches

---

## 🔄 Evolution of Self-Supervised Learning

### Timeline of Major Breakthroughs

```
2020: SimCLR/MoCo → Contrastive Learning Era
2020: BYOL → First successful non-contrastive method  
2021: DINO → Distillation-based vision transformers
2021: MAE → Masked autoencoder approaches
```

### 🎯 Core Philosophy Shift

**Contrastive Era**: "Learn by comparing positive and negative examples"
**Post-Contrastive**: "Learn by self-consistency and distillation"

---

## 🔥 Contrastive Learning: SimCLR & MoCo

### 📊 SimCLR (Simple Contrastive Learning)

**Core Idea**: Learn representations by maximizing agreement between differently augmented views of the same data

#### SimCLR Algorithm
1. **Sample** a minibatch of N examples
2. **Augment** each example twice → 2N augmented examples
3. **Encode** all augmented examples
4. **Contrast** positive pairs against negative pairs

#### Mathematical Foundation
```python
# SimCLR Loss (InfoNCE)
def simclr_loss(z_i, z_j, temperature=0.5):
    """
    z_i, z_j: normalized embeddings of positive pair
    """
    batch_size = z_i.size(0)
    
    # Compute similarity scores
    z = torch.cat([z_i, z_j], dim=0)  # 2N × d
    sim_matrix = torch.mm(z, z.t()) / temperature  # 2N × 2N
    
    # Create positive pair labels
    labels = torch.cat([torch.arange(batch_size), 
                       torch.arange(batch_size)]).long()
    
    # Mask out self-similarity on diagonal
    mask = torch.eye(2 * batch_size).bool()
    sim_matrix.masked_fill_(mask, -float('inf'))
    
    # CrossEntropy loss
    loss = F.cross_entropy(sim_matrix, labels)
    return loss
```

#### SimCLR Strengths & Weaknesses
**✅ Strengths**:
- Simple and effective
- Strong empirical results
- Clear theoretical foundation

**❌ Weaknesses**:
- Requires large batch sizes (8192+)
- Sensitive to negative sampling
- Computationally expensive

### 🔄 MoCo (Momentum Contrast)

**Key Innovation**: Maintain a queue of negative examples to avoid large batch size requirements

#### MoCo Components
1. **Query Encoder**: Online network (updated by gradient)
2. **Key Encoder**: momentum-updated network
3. **Memory Queue**: stores previous key representations

#### Mathematical Foundation
```python
class MoCo(nn.Module):
    def __init__(self, encoder, dim=128, K=65536, m=0.999, T=0.07):
        super().__init__()
        self.K = K  # queue size
        self.m = m  # momentum coefficient
        self.T = T  # temperature
        
        # Query encoder
        self.encoder_q = encoder
        
        # Key encoder (momentum)
        self.encoder_k = copy.deepcopy(encoder)
        for param in self.encoder_k.parameters():
            param.requires_grad = False
        
        # Queue
        self.register_buffer("queue", torch.randn(dim, K))
        self.queue = F.normalize(self.queue, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
    
    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """Momentum update of key encoder"""
        for param_q, param_k in zip(self.encoder_q.parameters(), 
                                   self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)
    
    def forward(self, im_q, im_k):
        # Query features
        q = self.encoder_q(im_q)
        q = F.normalize(q, dim=1)
        
        # Key features (no gradient)
        with torch.no_grad():
            self._momentum_update_key_encoder()
            k = self.encoder_k(im_k)
            k = F.normalize(k, dim=1)
        
        # Positive logits
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        
        # Negative logits
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])
        
        # Concatenate and apply temperature
        logits = torch.cat([l_pos, l_neg], dim=1) / self.T
        
        # Labels (positive is always index 0)
        labels = torch.zeros(logits.shape[0], dtype=torch.long)
        
        return logits, labels
```

---

## 🚀 The Non-Contrastive Revolution: BYOL

### 🎯 BYOL (Bootstrap Your Own Latent)

**Revolutionary Insight**: Learn good representations without negative examples!

#### BYOL Architecture
```
Online Network: encoder → projector → predictor
Target Network: encoder → projector (EMA of online)
```

#### Why BYOL Works (Theory)

**Key Insight**: Asymmetry prevents collapse
1. **Predictor asymmetry**: Only online network has predictor
2. **Target EMA**: Target network provides stable targets
3. **Stop gradient**: Prevents trivial solutions

#### Mathematical Foundation
```python
class BYOL(nn.Module):
    def __init__(self, encoder, hidden_dim=4096, output_dim=256, tau=0.99):
        super().__init__()
        self.tau = tau
        
        # Online network
        self.online_encoder = encoder
        self.online_projector = MLP(encoder.output_dim, hidden_dim, output_dim)
        self.online_predictor = MLP(output_dim, hidden_dim, output_dim)
        
        # Target network (EMA of online)
        self.target_encoder = copy.deepcopy(encoder)
        self.target_projector = copy.deepcopy(self.online_projector)
        
        # Freeze target network
        for param in self.target_encoder.parameters():
            param.requires_grad = False
        for param in self.target_projector.parameters():
            param.requires_grad = False
    
    @torch.no_grad()
    def update_target_network(self):
        """EMA update of target network"""
        for online_params, target_params in zip(
            self.online_encoder.parameters(), self.target_encoder.parameters()
        ):
            target_params.data = self.tau * target_params.data + (1 - self.tau) * online_params.data
        
        for online_params, target_params in zip(
            self.online_projector.parameters(), self.target_projector.parameters()
        ):
            target_params.data = self.tau * target_params.data + (1 - self.tau) * online_params.data
    
    def forward(self, x1, x2):
        # Online predictions
        online_proj_1 = self.online_projector(self.online_encoder(x1))
        online_proj_2 = self.online_projector(self.online_encoder(x2))
        
        online_pred_1 = self.online_predictor(online_proj_1)
        online_pred_2 = self.online_predictor(online_proj_2)
        
        # Target projections (stop gradient)
        with torch.no_grad():
            target_proj_1 = self.target_projector(self.target_encoder(x1))
            target_proj_2 = self.target_projector(self.target_encoder(x2))
        
        # Symmetric loss
        loss = (
            self.regression_loss(online_pred_1, target_proj_2) +
            self.regression_loss(online_pred_2, target_proj_1)
        ) / 2
        
        return loss
    
    def regression_loss(self, x, y):
        """Cosine similarity loss"""
        x = F.normalize(x, dim=-1, p=2)
        y = F.normalize(y, dim=-1, p=2)
        return 2 - 2 * (x * y).sum(dim=-1).mean()
```

#### BYOL's Impact
- **No negatives needed**: Simplified training
- **Stable training**: Less sensitive to hyperparameters  
- **Strong performance**: Competitive with contrastive methods

---

## 🎓 Knowledge Distillation: The Path to DINO

### 🧠 Traditional Knowledge Distillation

**Original Concept** (Hinton et al.): Transfer knowledge from teacher to student

#### Core Components
1. **Teacher Model**: Large, well-trained network
2. **Student Model**: Smaller network to be trained
3. **Soft Targets**: Teacher's output probabilities
4. **Temperature Scaling**: Softens probability distributions

#### Mathematical Foundation
```python
def knowledge_distillation_loss(student_logits, teacher_logits, temperature=3.0, alpha=0.7):
    """
    Combine hard targets (ground truth) with soft targets (teacher)
    """
    # Soften distributions
    student_probs = F.softmax(student_logits / temperature, dim=1)
    teacher_probs = F.softmax(teacher_logits / temperature, dim=1)
    
    # Distillation loss (KL divergence)
    distillation_loss = F.kl_div(
        F.log_softmax(student_logits / temperature, dim=1),
        teacher_probs,
        reduction='batchmean'
    ) * (temperature ** 2)
    
    return distillation_loss
```

### 🔄 Self-Distillation: DINO's Innovation

**DINO's Key Insight**: Use self-distillation for representation learning

#### How DINO Differs
1. **No ground truth labels**: Pure self-supervision
2. **Same architecture**: Teacher and student are identical
3. **EMA updates**: Teacher updated via exponential moving average
4. **Multi-crop strategy**: Different views for teacher vs student

#### DINO Architecture
```
Student Network: ViT → projection head
Teacher Network: ViT → projection head (EMA of student)
Loss: Cross-entropy between student and teacher outputs
```

---

## 🧮 Mathematical Foundations of Distillation

### Temperature Scaling

**Purpose**: Control the "sharpness" of probability distributions

```python
def temperature_scaling(logits, temperature):
    """
    Apply temperature scaling to logits
    
    Low temperature (< 1): Sharper distributions (more confident)
    High temperature (> 1): Softer distributions (less confident)
    """
    return logits / temperature

# Example
logits = torch.tensor([2.0, 1.0, 0.5])

# Different temperatures
temps = [0.5, 1.0, 2.0, 5.0]
for T in temps:
    probs = F.softmax(temperature_scaling(logits, T), dim=0)
    print(f"T={T}: {probs.numpy()}")

# Output:
# T=0.5: [0.8756 0.1064 0.0180]  # Sharp
# T=1.0: [0.6590 0.2424 0.0986]  # Normal
# T=2.0: [0.5033 0.3072 0.1896]  # Soft
# T=5.0: [0.3775 0.3264 0.2961]  # Very soft
```

### Centering Mechanism

**Problem**: Without proper regularization, both networks might collapse to the same output

**DINO's Solution**: Center the teacher outputs

```python
class CenteringMechanism:
    def __init__(self, output_dim, momentum=0.9):
        self.momentum = momentum
        self.center = torch.zeros(output_dim)
    
    @torch.no_grad()
    def update_center(self, teacher_output):
        """Update center with EMA"""
        batch_center = teacher_output.mean(dim=0)
        self.center = self.momentum * self.center + (1 - self.momentum) * batch_center
    
    def apply_centering(self, teacher_output):
        """Apply centering to teacher output"""
        return teacher_output - self.center
```

### DINO Loss Function

```python
def dino_loss(student_output, teacher_output, temperature_s=0.1, temperature_t=0.04):
    """
    DINO loss: cross-entropy between student and teacher
    
    Args:
        student_output: student network output (before softmax)
        teacher_output: teacher network output (before softmax)  
        temperature_s: student temperature (higher = softer)
        temperature_t: teacher temperature (lower = sharper)
    """
    # Apply temperature scaling
    student_probs = F.log_softmax(student_output / temperature_s, dim=1)
    teacher_probs = F.softmax(teacher_output / temperature_t, dim=1)
    
    # Cross-entropy loss
    loss = -torch.sum(teacher_probs * student_probs, dim=1).mean()
    
    return loss
```

---

## 🤔 Why DINO Doesn't Need Negative Samples

### The Collapse Problem

**Traditional Issue**: Without negative samples, networks collapse to trivial solutions

### DINO's Collapse Prevention

1. **Temperature Asymmetry**: 
   - Teacher: Low temperature (sharp, confident predictions)
   - Student: High temperature (soft, uncertain predictions)

2. **Centering**: Prevents all outputs from concentrating on one dimension

3. **EMA Updates**: Teacher provides stable targets

4. **Multi-crop Strategy**: Diverse augmentations prevent trivial solutions

### Theoretical Analysis

**Why this works**:
- **Teacher sharpness**: Provides confident, informative targets
- **Student softness**: Allows gradual learning without sudden jumps
- **Momentum updates**: Stabilizes training dynamics
- **Centering**: Ensures feature diversity

---

## 📋 **Theory Check**: SSL Approaches Quiz

### Question 1: Contrastive Learning
**Q**: What is the main computational bottleneck in SimCLR?
- A) Large model size
- B) Large batch size requirement
- C) Complex augmentations  
- D) Long training time

<details>
<summary>Answer</summary>
B) Large batch size requirement - SimCLR needs large batches (8192+) to have enough negative examples.
</details>

### Question 2: MoCo Innovation
**Q**: How does MoCo solve SimCLR's batch size problem?
- A) Using smaller models
- B) Queue of negative examples
- C) Different loss function
- D) Better augmentations

<details>
<summary>Answer</summary>
B) Queue of negative examples - MoCo maintains a queue of past representations as negatives.
</details>

### Question 3: BYOL Key Innovation
**Q**: What prevents collapse in BYOL without negative examples?
- A) Large batch sizes
- B) Strong augmentations
- C) Predictor asymmetry + EMA updates
- D) Temperature scaling

<details>
<summary>Answer</summary>
C) Predictor asymmetry + EMA updates - The predictor is only in the online network, and target network uses EMA.
</details>

### Question 4: Knowledge Distillation
**Q**: What does temperature scaling control in knowledge distillation?
- A) Learning rate
- B) Model size
- C) Sharpness of probability distributions
- D) Training time

<details>
<summary>Answer</summary>
C) Sharpness of probability distributions - Lower temperature = sharper, higher temperature = softer.
</details>

### Question 5: DINO Mechanism
**Q**: How does DINO prevent mode collapse?
- A) Negative sampling
- B) Large datasets
- C) Centering + temperature asymmetry
- D) Complex architectures

<details>
<summary>Answer</summary>
C) Centering + temperature asymmetry - Centering prevents concentration, temperature asymmetry stabilizes learning.
</details>

---

## 🔄 Implementation Comparison

### Practical Exercise: Implement Core Components

```python
class SSLComparison:
    """Compare core components of different SSL methods"""
    
    @staticmethod
    def simclr_loss(z1, z2, temperature=0.5):
        """SimCLR contrastive loss"""
        batch_size = z1.size(0)
        z = torch.cat([z1, z2], dim=0)
        
        # Cosine similarity
        sim = F.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=2)
        sim = sim / temperature
        
        # Positive pairs
        pos_mask = torch.block_diag(
            torch.ones(batch_size, batch_size) - torch.eye(batch_size),
            torch.ones(batch_size, batch_size) - torch.eye(batch_size)
        ).bool()
        
        # Negative pairs  
        neg_mask = ~(torch.eye(2 * batch_size).bool() | pos_mask)
        
        # InfoNCE loss
        pos_sim = sim[pos_mask].view(2 * batch_size, -1)
        neg_sim = sim[neg_mask].view(2 * batch_size, -1)
        
        logits = torch.cat([pos_sim, neg_sim], dim=1)
        labels = torch.zeros(2 * batch_size, dtype=torch.long)
        
        return F.cross_entropy(logits, labels)
    
    @staticmethod  
    def byol_loss(p1, z2, p2, z1):
        """BYOL regression loss (symmetric)"""
        return (
            2 - 2 * F.cosine_similarity(p1, z2, dim=-1).mean() +
            2 - 2 * F.cosine_similarity(p2, z1, dim=-1).mean()
        ) / 2
    
    @staticmethod
    def dino_loss(student_out, teacher_out, temp_s=0.1, temp_t=0.04):
        """DINO distillation loss"""
        student_probs = F.log_softmax(student_out / temp_s, dim=1)
        teacher_probs = F.softmax(teacher_out / temp_t, dim=1)
        return -(teacher_probs * student_probs).sum(dim=1).mean()

# Test different losses
batch_size = 32
embed_dim = 128

# Random embeddings
z1 = F.normalize(torch.randn(batch_size, embed_dim), dim=1)
z2 = F.normalize(torch.randn(batch_size, embed_dim), dim=1)

ssl_comp = SSLComparison()

# Compare losses
simclr_loss = ssl_comp.simclr_loss(z1, z2)
byol_loss = ssl_comp.byol_loss(z1, z2, z2, z1)  # Simplified
dino_loss = ssl_comp.dino_loss(z1, z2)

print(f"SimCLR Loss: {simclr_loss:.4f}")
print(f"BYOL Loss: {byol_loss:.4f}")  
print(f"DINO Loss: {dino_loss:.4f}")
```

---

## 🎯 Progression Summary

### Evolution of Ideas

1. **SimCLR/MoCo Era** (2020):
   - Contrastive learning dominance
   - Need for negative examples
   - Batch size and memory constraints

2. **BYOL Revolution** (2020):
   - First successful non-contrastive method
   - Asymmetry prevents collapse
   - Simpler training dynamics

3. **DINO Innovation** (2021):
   - Distillation for vision transformers
   - Multi-crop strategy
   - Outstanding attention maps

### Key Insights for DINO

1. **No negatives needed**: Following BYOL's insight
2. **Self-distillation**: Teacher-student paradigm
3. **Vision transformers**: Perfect match for global attention
4. **Multi-scale training**: Global and local crops

---

## 🎯 Key Takeaways

1. **SSL evolution**: Contrastive → Non-contrastive → Distillation
2. **Negative sampling**: Not always necessary with proper design
3. **Asymmetry**: Key to preventing collapse in non-contrastive methods
4. **Temperature scaling**: Critical for distillation-based approaches
5. **DINO innovation**: Combines best ideas in ViT framework

---

## 📚 Further Reading

1. **Contrastive Learning**:
   - [SimCLR](https://arxiv.org/abs/2002.05709): A Simple Framework for Contrastive Learning
   - [MoCo](https://arxiv.org/abs/1911.05722): Momentum Contrast for Unsupervised Visual Representation Learning

2. **Non-Contrastive Learning**:
   - [BYOL](https://arxiv.org/abs/2006.07733): Bootstrap Your Own Latent
   - [Understanding BYOL](https://arxiv.org/abs/2010.10241): Theoretical analysis

3. **Knowledge Distillation**:
   - [Distilling Knowledge](https://arxiv.org/abs/1503.02531): Original knowledge distillation paper
   - [DINO](https://arxiv.org/abs/2104.14294): Self-Distillation with No Labels

---

## ✅ Lesson 1.3 Checklist

- [ ] Understand contrastive learning (SimCLR, MoCo)
- [ ] Grasp BYOL's non-contrastive innovation  
- [ ] Master knowledge distillation principles
- [ ] Implement core loss functions for each approach
- [ ] Complete theory quiz on SSL approaches
- [ ] Understand DINO's position in SSL evolution

**Next**: 🔹 Lesson 1.4 - DINO Paper Deep Dive
