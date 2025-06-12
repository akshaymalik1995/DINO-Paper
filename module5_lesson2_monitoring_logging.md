# Module 5, Lesson 2: Training Monitoring and Logging

## 🎯 Learning Objectives
By the end of this lesson, you will:
- Implement comprehensive monitoring for DINO training
- Build real-time visualization tools for loss curves and feature embeddings
- Create interactive dashboards using Weights & Biases (WandB)
- Develop debugging tools for training diagnostics

## 📚 Theoretical Background

### Why Monitoring Matters in Self-Supervised Learning

**Unique Challenges in SSL**:
- **No Ground Truth**: Can't rely on validation accuracy
- **Mode Collapse Risk**: Need to detect feature degradation early
- **Complex Dynamics**: Student-teacher interactions are non-trivial
- **Long Training**: SSL often requires 100+ epochs

**Key Metrics to Monitor**:
1. **Training Loss**: Primary optimization signal
2. **Feature Diversity**: Prevent mode collapse
3. **Teacher Temperature**: Track adaptive scheduling
4. **EMA Momentum**: Monitor teacher update dynamics
5. **Gradient Norms**: Detect training instability
6. **Feature Visualization**: t-SNE/UMAP embeddings

### Real-time Monitoring Architecture

```
Training Loop → Metrics Collection → Dashboard Update
     ↓                    ↓                 ↓
Loss Computation    Feature Analysis    Visual Plots
Temperature         Gradient Stats      Embedding Maps
EMA Updates         Memory Usage        Loss Curves
```

## 🛠️ Implementation

### Step 1: Advanced Metrics Collection

```python
# monitoring/metrics_collector.py
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple
import time
from collections import defaultdict, deque
import psutil
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import wandb

class DINOMetricsCollector:
    """Comprehensive metrics collection for DINO training."""
    
    def __init__(self, config: Dict, log_freq: int = 100):
        self.config = config
        self.log_freq = log_freq
        
        # Metrics storage
        self.metrics_history = defaultdict(list)
        self.batch_metrics = defaultdict(deque, maxlen=1000)
        
        # Timing
        self.batch_times = deque(maxlen=100)
        self.start_time = time.time()
        
        # Feature analysis
        self.feature_buffer = deque(maxlen=1000)
        self.embedding_buffer = deque(maxlen=5000)
        
        # Gradient tracking
        self.gradient_norms = deque(maxlen=100)
        
    def update_batch_metrics(self, 
                           loss: float,
                           student_outputs: List[torch.Tensor],
                           teacher_outputs: List[torch.Tensor],
                           model: nn.Module,
                           optimizer: torch.optim.Optimizer,
                           batch_time: float) -> Dict[str, float]:
        """Update metrics for current batch."""
        
        metrics = {}
        
        # Basic training metrics
        metrics['loss'] = loss
        metrics['lr'] = optimizer.param_groups[0]['lr']
        metrics['batch_time'] = batch_time
        self.batch_times.append(batch_time)
        
        # Compute feature statistics
        feature_stats = self._compute_feature_stats(student_outputs, teacher_outputs)
        metrics.update(feature_stats)
        
        # Compute gradient statistics
        grad_stats = self._compute_gradient_stats(model)
        metrics.update(grad_stats)
        
        # System metrics
        system_stats = self._compute_system_stats()
        metrics.update(system_stats)
        
        # Store metrics
        for key, value in metrics.items():
            self.batch_metrics[key].append(value)
            
        return metrics
        
    def _compute_feature_stats(self, 
                              student_outputs: List[torch.Tensor],
                              teacher_outputs: List[torch.Tensor]) -> Dict[str, float]:
        """Compute feature-level statistics."""
        stats = {}
        
        # Concatenate all outputs
        student_features = torch.cat(student_outputs, dim=0)
        teacher_features = torch.cat(teacher_outputs, dim=0)
        
        with torch.no_grad():
            # Feature norms
            student_norms = torch.norm(student_features, dim=1)
            teacher_norms = torch.norm(teacher_features, dim=1)
            
            stats['student_feature_norm_mean'] = student_norms.mean().item()
            stats['student_feature_norm_std'] = student_norms.std().item()
            stats['teacher_feature_norm_mean'] = teacher_norms.mean().item()
            stats['teacher_feature_norm_std'] = teacher_norms.std().item()
            
            # Feature diversity (standard deviation across features)
            student_diversity = student_features.std(dim=0).mean().item()
            teacher_diversity = teacher_features.std(dim=0).mean().item()
            
            stats['student_feature_diversity'] = student_diversity
            stats['teacher_feature_diversity'] = teacher_diversity
            
            # Cosine similarity between student and teacher
            student_norm = F.normalize(student_features, dim=1)
            teacher_norm = F.normalize(teacher_features, dim=1)
            cosine_sim = (student_norm * teacher_norm).sum(dim=1).mean().item()
            
            stats['student_teacher_similarity'] = cosine_sim
            
            # Feature collapse detection (rank of feature matrix)
            try:
                feature_rank = torch.matrix_rank(student_features[:100]).item()
                stats['feature_rank'] = feature_rank
                stats['feature_rank_ratio'] = feature_rank / min(100, student_features.size(1))
            except:
                stats['feature_rank'] = 0
                stats['feature_rank_ratio'] = 0
                
            # Store features for visualization
            if len(self.feature_buffer) < self.feature_buffer.maxlen:
                self.feature_buffer.append(student_features[:32].cpu().numpy())
                
        return stats
        
    def _compute_gradient_stats(self, model: nn.Module) -> Dict[str, float]:
        """Compute gradient statistics."""
        stats = {}
        
        total_norm = 0.0
        param_count = 0
        grad_norms = []
        
        for name, param in model.named_parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
                param_count += 1
                grad_norms.append(param_norm.item())
                
        if param_count > 0:
            total_norm = total_norm ** (1. / 2)
            stats['grad_norm_total'] = total_norm
            stats['grad_norm_mean'] = np.mean(grad_norms)
            stats['grad_norm_std'] = np.std(grad_norms)
            stats['grad_norm_max'] = np.max(grad_norms)
            
            self.gradient_norms.append(total_norm)
        else:
            stats['grad_norm_total'] = 0.0
            stats['grad_norm_mean'] = 0.0
            stats['grad_norm_std'] = 0.0
            stats['grad_norm_max'] = 0.0
            
        return stats
        
    def _compute_system_stats(self) -> Dict[str, float]:
        """Compute system resource statistics."""
        stats = {}
        
        # Memory usage
        if torch.cuda.is_available():
            stats['gpu_memory_allocated'] = torch.cuda.memory_allocated() / 1024**3  # GB
            stats['gpu_memory_reserved'] = torch.cuda.memory_reserved() / 1024**3  # GB
            stats['gpu_memory_usage'] = torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated()
            
        # CPU and system memory
        stats['cpu_percent'] = psutil.cpu_percent()
        stats['memory_percent'] = psutil.virtual_memory().percent
        
        # Throughput
        if len(self.batch_times) > 0:
            stats['samples_per_sec'] = self.config.get('batch_size', 64) / np.mean(self.batch_times)
            
        return stats
        
    def get_epoch_summary(self) -> Dict[str, float]:
        """Get summary statistics for the epoch."""
        summary = {}
        
        for metric_name, values in self.batch_metrics.items():
            if len(values) > 0:
                summary[f'{metric_name}_mean'] = np.mean(values)
                summary[f'{metric_name}_std'] = np.std(values)
                summary[f'{metric_name}_min'] = np.min(values)
                summary[f'{metric_name}_max'] = np.max(values)
                
        return summary
        
    def should_log_visualizations(self, step: int) -> bool:
        """Check if we should create visualizations this step."""
        return step % (self.log_freq * 10) == 0
        
    def create_feature_visualization(self) -> Optional[plt.Figure]:
        """Create t-SNE visualization of learned features."""
        if len(self.feature_buffer) < 5:
            return None
            
        # Combine recent features
        features = np.vstack(list(self.feature_buffer)[-5:])
        
        if features.shape[0] < 50:
            return None
            
        # Reduce dimensionality with PCA first
        if features.shape[1] > 50:
            pca = PCA(n_components=50)
            features = pca.fit_transform(features)
            
        # t-SNE embedding
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, features.shape[0]//4))
        embedding = tsne.fit_transform(features[:500])  # Limit for speed
        
        # Create plot
        fig, ax = plt.subplots(figsize=(10, 8))
        scatter = ax.scatter(embedding[:, 0], embedding[:, 1], 
                           c=np.arange(len(embedding)), 
                           cmap='viridis', alpha=0.6)
        ax.set_title('DINO Feature Embeddings (t-SNE)')
        ax.set_xlabel('t-SNE Dimension 1')
        ax.set_ylabel('t-SNE Dimension 2')
        plt.colorbar(scatter, label='Sample Index')
        
        return fig
        
    def create_training_dashboard(self) -> plt.Figure:
        """Create comprehensive training dashboard."""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('DINO Training Dashboard', fontsize=16)
        
        # Loss curve
        if 'loss' in self.batch_metrics and len(self.batch_metrics['loss']) > 0:
            axes[0, 0].plot(self.batch_metrics['loss'])
            axes[0, 0].set_title('Training Loss')
            axes[0, 0].set_xlabel('Batch')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].grid(True)
            
        # Learning rate
        if 'lr' in self.batch_metrics and len(self.batch_metrics['lr']) > 0:
            axes[0, 1].plot(self.batch_metrics['lr'])
            axes[0, 1].set_title('Learning Rate')
            axes[0, 1].set_xlabel('Batch')
            axes[0, 1].set_ylabel('LR')
            axes[0, 1].grid(True)
            
        # Gradient norms
        if len(self.gradient_norms) > 0:
            axes[0, 2].plot(self.gradient_norms)
            axes[0, 2].set_title('Gradient Norm')
            axes[0, 2].set_xlabel('Batch')
            axes[0, 2].set_ylabel('Norm')
            axes[0, 2].grid(True)
            
        # Feature diversity
        if 'student_feature_diversity' in self.batch_metrics:
            axes[1, 0].plot(self.batch_metrics['student_feature_diversity'], label='Student')
            if 'teacher_feature_diversity' in self.batch_metrics:
                axes[1, 0].plot(self.batch_metrics['teacher_feature_diversity'], label='Teacher')
            axes[1, 0].set_title('Feature Diversity')
            axes[1, 0].set_xlabel('Batch')
            axes[1, 0].set_ylabel('Diversity')
            axes[1, 0].legend()
            axes[1, 0].grid(True)
            
        # Memory usage
        if 'gpu_memory_allocated' in self.batch_metrics:
            axes[1, 1].plot(self.batch_metrics['gpu_memory_allocated'])
            axes[1, 1].set_title('GPU Memory Usage (GB)')
            axes[1, 1].set_xlabel('Batch')
            axes[1, 1].set_ylabel('Memory (GB)')
            axes[1, 1].grid(True)
            
        # Throughput
        if 'samples_per_sec' in self.batch_metrics:
            axes[1, 2].plot(self.batch_metrics['samples_per_sec'])
            axes[1, 2].set_title('Training Throughput')
            axes[1, 2].set_xlabel('Batch')
            axes[1, 2].set_ylabel('Samples/sec')
            axes[1, 2].grid(True)
            
        plt.tight_layout()
        return fig


class WandBLogger:
    """Weights & Biases integration for DINO training."""
    
    def __init__(self, config: Dict, project_name: str = "dino-training"):
        self.config = config
        
        # Initialize wandb
        wandb.init(
            project=project_name,
            config=config,
            name=f"dino_{config['model']['backbone']}_{int(time.time())}"
        )
        
        # Watch model (optional, can be expensive)
        self.model_watched = False
        
    def watch_model(self, model: nn.Module):
        """Start watching model parameters."""
        if not self.model_watched:
            wandb.watch(model, log_freq=1000)
            self.model_watched = True
            
    def log_metrics(self, metrics: Dict[str, float], step: int):
        """Log metrics to wandb."""
        wandb.log(metrics, step=step)
        
    def log_images(self, images: Dict[str, plt.Figure], step: int):
        """Log matplotlib figures to wandb."""
        for name, fig in images.items():
            wandb.log({name: wandb.Image(fig)}, step=step)
            plt.close(fig)  # Free memory
            
    def log_histograms(self, data: Dict[str, np.ndarray], step: int):
        """Log histograms to wandb."""
        for name, values in data.items():
            wandb.log({f"{name}_hist": wandb.Histogram(values)}, step=step)
            
    def finish(self):
        """Finish wandb run."""
        wandb.finish()


class LiveDashboard:
    """Real-time training dashboard using matplotlib."""
    
    def __init__(self, metrics_collector: DINOMetricsCollector):
        self.metrics_collector = metrics_collector
        self.fig = None
        self.axes = None
        self.setup_plot()
        
    def setup_plot(self):
        """Setup interactive matplotlib dashboard."""
        plt.ion()  # Interactive mode
        self.fig, self.axes = plt.subplots(2, 2, figsize=(12, 8))
        self.fig.suptitle('DINO Training Live Dashboard')
        
    def update(self, step: int):
        """Update dashboard with latest metrics."""
        if step % 50 != 0:  # Update every 50 steps
            return
            
        # Clear axes
        for ax in self.axes.flat:
            ax.clear()
            
        metrics = self.metrics_collector.batch_metrics
        
        # Loss
        if 'loss' in metrics and len(metrics['loss']) > 0:
            self.axes[0, 0].plot(metrics['loss'])
            self.axes[0, 0].set_title('Training Loss')
            self.axes[0, 0].grid(True)
            
        # Feature diversity
        if 'student_feature_diversity' in metrics:
            self.axes[0, 1].plot(metrics['student_feature_diversity'], label='Student')
            if 'teacher_feature_diversity' in metrics:
                self.axes[0, 1].plot(metrics['teacher_feature_diversity'], label='Teacher')
            self.axes[0, 1].set_title('Feature Diversity')
            self.axes[0, 1].legend()
            self.axes[0, 1].grid(True)
            
        # Gradient norms
        if len(self.metrics_collector.gradient_norms) > 0:
            self.axes[1, 0].plot(self.metrics_collector.gradient_norms)
            self.axes[1, 0].set_title('Gradient Norm')
            self.axes[1, 0].grid(True)
            
        # Memory usage
        if 'gpu_memory_allocated' in metrics:
            self.axes[1, 1].plot(metrics['gpu_memory_allocated'])
            self.axes[1, 1].set_title('GPU Memory (GB)')
            self.axes[1, 1].grid(True)
            
        # Refresh display
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        
    def close(self):
        """Close dashboard."""
        plt.ioff()
        plt.close(self.fig)
```

### Step 2: Integration with Training Loop

```python
# training/monitored_trainer.py
import torch
from typing import Dict, Optional
import matplotlib.pyplot as plt
import time

from .complete_trainer import CompleteDINOTrainer
from ..monitoring.metrics_collector import DINOMetricsCollector, WandBLogger, LiveDashboard

class MonitoredDINOTrainer(CompleteDINOTrainer):
    """DINO trainer with comprehensive monitoring."""
    
    def __init__(self, config_path: str, enable_wandb: bool = True, enable_live_dashboard: bool = False):
        super().__init__(config_path)
        
        # Initialize monitoring
        self.metrics_collector = DINOMetricsCollector(
            config=self.config,
            log_freq=self.config['logging']['log_freq']
        )
        
        # WandB logging
        if enable_wandb and self.config['logging']['use_wandb']:
            self.wandb_logger = WandBLogger(
                config=self.config,
                project_name=self.config['logging']['project_name']
            )
            self.wandb_logger.watch_model(self.model.student)
        else:
            self.wandb_logger = None
            
        # Live dashboard
        if enable_live_dashboard:
            self.live_dashboard = LiveDashboard(self.metrics_collector)
        else:
            self.live_dashboard = None
            
        self.global_step = 0
        
    def train_epoch(self, epoch: int) -> dict:
        """Train epoch with comprehensive monitoring."""
        self.model.train()
        self.loss_fn.train()
        
        epoch_metrics = {
            'loss': 0.0,
            'lr': 0.0,
            'teacher_temp': 0.0,
            'ema_momentum': 0.0
        }
        
        num_batches = len(self.train_loader)
        
        for batch_idx, batch in enumerate(self.train_loader):
            batch_start_time = time.time()
            
            # Move data to device
            crops = [crop.to(self.device) for crop in batch['crops']]
            
            # Update learning rate for warmup
            if epoch < self.config['training']['warmup_epochs']:
                warmup_lr = self._get_warmup_lr(epoch, batch_idx, num_batches)
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = warmup_lr
                    
            # Update teacher temperature
            self.loss_fn.update_teacher_temp(epoch)
            
            # Update EMA momentum
            self.model.update_momentum_schedule(epoch, self.config['training']['epochs'])
            
            # Forward pass
            student_outputs = []
            teacher_outputs = []
            
            for crop in crops:
                student_out = self.model.student(crop)
                with torch.no_grad():
                    teacher_out = self.model.teacher(crop)
                student_outputs.append(student_out)
                teacher_outputs.append(teacher_out)
                
            # Compute loss
            loss = self.loss_fn(student_outputs, teacher_outputs)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.student.parameters(),
                self.config['training']['gradient_clip']
            )
            
            # Update student
            self.optimizer.step()
            
            # Update teacher (EMA)
            self.model.update_teacher()
            
            # Collect metrics
            batch_time = time.time() - batch_start_time
            batch_metrics = self.metrics_collector.update_batch_metrics(
                loss=loss.item(),
                student_outputs=student_outputs,
                teacher_outputs=teacher_outputs,
                model=self.model.student,
                optimizer=self.optimizer,
                batch_time=batch_time
            )
            
            # Add DINO-specific metrics
            batch_metrics.update({
                'teacher_temp': self.loss_fn.teacher_temp,
                'ema_momentum': self.model.momentum,
                'epoch': epoch,
                'batch_idx': batch_idx
            })
            
            # Log to wandb
            if self.wandb_logger and self.global_step % self.config['logging']['log_freq'] == 0:
                self.wandb_logger.log_metrics(batch_metrics, self.global_step)
                
            # Update live dashboard
            if self.live_dashboard:
                self.live_dashboard.update(self.global_step)
                
            # Log batch metrics
            if batch_idx % self.config['logging']['log_freq'] == 0:
                self.logger.info(
                    f"Epoch {epoch:3d} [{batch_idx:4d}/{num_batches:4d}] "
                    f"Loss: {loss.item():.4f} "
                    f"LR: {batch_metrics['lr']:.6f} "
                    f"Teacher Temp: {batch_metrics['teacher_temp']:.4f} "
                    f"Diversity: {batch_metrics.get('student_feature_diversity', 0):.4f} "
                    f"Memory: {batch_metrics.get('gpu_memory_allocated', 0):.1f}GB"
                )
                
            # Create visualizations periodically
            if (self.metrics_collector.should_log_visualizations(self.global_step) and 
                self.config['logging']['save_visualizations']):
                self._log_visualizations(epoch, self.global_step)
                
            # Track epoch metrics
            epoch_metrics['loss'] += loss.item()
            epoch_metrics['lr'] = batch_metrics['lr']
            epoch_metrics['teacher_temp'] = batch_metrics['teacher_temp']
            epoch_metrics['ema_momentum'] = batch_metrics['ema_momentum']
            
            self.global_step += 1
            
        # Average metrics over epoch
        epoch_metrics['loss'] /= num_batches
        
        # Log epoch summary
        epoch_summary = self.metrics_collector.get_epoch_summary()
        if self.wandb_logger:
            epoch_summary['epoch'] = epoch
            self.wandb_logger.log_metrics(epoch_summary, self.global_step)
            
        return epoch_metrics
        
    def _log_visualizations(self, epoch: int, step: int):
        """Create and log visualizations."""
        try:
            # Feature embedding visualization
            feature_fig = self.metrics_collector.create_feature_visualization()
            
            # Training dashboard
            dashboard_fig = self.metrics_collector.create_training_dashboard()
            
            # Log to wandb
            if self.wandb_logger:
                images = {}
                if feature_fig:
                    images['feature_embeddings'] = feature_fig
                if dashboard_fig:
                    images['training_dashboard'] = dashboard_fig
                    
                if images:
                    self.wandb_logger.log_images(images, step)
                    
            # Save locally
            if feature_fig:
                feature_fig.savefig(f'./plots/feature_embeddings_epoch_{epoch}.png', dpi=150, bbox_inches='tight')
                plt.close(feature_fig)
                
            if dashboard_fig:
                dashboard_fig.savefig(f'./plots/dashboard_epoch_{epoch}.png', dpi=150, bbox_inches='tight')
                plt.close(dashboard_fig)
                
        except Exception as e:
            self.logger.warning(f"Failed to create visualizations: {e}")
            
    def finish_training(self):
        """Clean up monitoring resources."""
        if self.wandb_logger:
            self.wandb_logger.finish()
            
        if self.live_dashboard:
            self.live_dashboard.close()
```

### Step 3: Training Analysis Tools

```python
# analysis/training_analyzer.py
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pandas as pd
from typing import Dict, List, Optional

class DINOTrainingAnalyzer:
    """Analyze DINO training runs."""
    
    def __init__(self, checkpoint_dir: str):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoints = self._load_checkpoints()
        
    def _load_checkpoints(self) -> List[Dict]:
        """Load all checkpoints from directory."""
        checkpoints = []
        
        for checkpoint_file in sorted(self.checkpoint_dir.glob("dino_epoch_*.pth")):
            checkpoint = torch.load(checkpoint_file, map_location='cpu')
            checkpoints.append(checkpoint)
            
        return checkpoints
        
    def plot_training_curves(self) -> plt.Figure:
        """Plot comprehensive training curves."""
        if not self.checkpoints:
            return None
            
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('DINO Training Analysis', fontsize=16)
        
        # Extract metrics
        epochs = [cp['epoch'] for cp in self.checkpoints]
        losses = [cp['metrics']['loss'] for cp in self.checkpoints]
        
        # Loss curve
        axes[0, 0].plot(epochs, losses, 'b-', linewidth=2)
        axes[0, 0].set_title('Training Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].grid(True)
        
        # Learning rate (if available)
        if 'lr' in self.checkpoints[0]['metrics']:
            lrs = [cp['metrics']['lr'] for cp in self.checkpoints]
            axes[0, 1].plot(epochs, lrs, 'g-', linewidth=2)
            axes[0, 1].set_title('Learning Rate')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('LR')
            axes[0, 1].grid(True)
            
        # Teacher temperature (if available)
        if 'teacher_temp' in self.checkpoints[0]['metrics']:
            temps = [cp['metrics']['teacher_temp'] for cp in self.checkpoints]
            axes[0, 2].plot(epochs, temps, 'r-', linewidth=2)
            axes[0, 2].set_title('Teacher Temperature')
            axes[0, 2].set_xlabel('Epoch')
            axes[0, 2].set_ylabel('Temperature')
            axes[0, 2].grid(True)
            
        # Feature diversity (if available)
        if 'student_feature_diversity_mean' in self.checkpoints[0]['metrics']:
            diversities = [cp['metrics']['student_feature_diversity_mean'] for cp in self.checkpoints]
            axes[1, 0].plot(epochs, diversities, 'm-', linewidth=2)
            axes[1, 0].set_title('Feature Diversity')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Diversity')
            axes[1, 0].grid(True)
            
        # Gradient norms (if available)
        if 'grad_norm_total_mean' in self.checkpoints[0]['metrics']:
            grad_norms = [cp['metrics']['grad_norm_total_mean'] for cp in self.checkpoints]
            axes[1, 1].plot(epochs, grad_norms, 'c-', linewidth=2)
            axes[1, 1].set_title('Gradient Norm')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Norm')
            axes[1, 1].grid(True)
            
        # Memory usage (if available)
        if 'gpu_memory_allocated_mean' in self.checkpoints[0]['metrics']:
            memory = [cp['metrics']['gpu_memory_allocated_mean'] for cp in self.checkpoints]
            axes[1, 2].plot(epochs, memory, 'orange', linewidth=2)
            axes[1, 2].set_title('GPU Memory Usage')
            axes[1, 2].set_xlabel('Epoch')
            axes[1, 2].set_ylabel('Memory (GB)')
            axes[1, 2].grid(True)
            
        plt.tight_layout()
        return fig
        
    def analyze_convergence(self) -> Dict[str, float]:
        """Analyze training convergence."""
        if len(self.checkpoints) < 10:
            return {}
            
        losses = [cp['metrics']['loss'] for cp in self.checkpoints]
        
        # Loss reduction
        initial_loss = np.mean(losses[:5])
        final_loss = np.mean(losses[-5:])
        loss_reduction = (initial_loss - final_loss) / initial_loss
        
        # Convergence stability (variance in last 20% of training)
        last_20_percent = losses[int(len(losses) * 0.8):]
        convergence_stability = 1.0 / (1.0 + np.var(last_20_percent))
        
        # Training efficiency (epochs to reach 90% of final improvement)
        target_loss = initial_loss - 0.9 * (initial_loss - final_loss)
        efficiency_epoch = None
        for i, loss in enumerate(losses):
            if loss <= target_loss:
                efficiency_epoch = i
                break
                
        efficiency_ratio = efficiency_epoch / len(losses) if efficiency_epoch else 1.0
        
        return {
            'loss_reduction': loss_reduction,
            'convergence_stability': convergence_stability,
            'training_efficiency': efficiency_ratio,
            'final_loss': final_loss,
            'total_epochs': len(losses)
        }
        
    def export_metrics_csv(self, save_path: str):
        """Export all metrics to CSV for further analysis."""
        if not self.checkpoints:
            return
            
        # Flatten metrics from all checkpoints
        data = []
        for cp in self.checkpoints:
            row = {'epoch': cp['epoch']}
            row.update(cp['metrics'])
            data.append(row)
            
        df = pd.DataFrame(data)
        df.to_csv(save_path, index=False)
        print(f"Metrics exported to {save_path}")
```

## 🧪 Practical Exercises

### Exercise 1: Basic Monitoring Setup
```python
# Set up monitored training
trainer = MonitoredDINOTrainer(
    config_path='config/cifar10_config.yaml',
    enable_wandb=True,
    enable_live_dashboard=True
)

# Train with monitoring
trainer.train()

# Analyze results
analyzer = DINOTrainingAnalyzer('./checkpoints')
convergence_stats = analyzer.analyze_convergence()
print(f"Training converged with {convergence_stats['loss_reduction']:.2%} loss reduction")
```

### Exercise 2: Custom Metrics
```python
# Add custom metrics to monitor specific aspects
class CustomMetricsCollector(DINOMetricsCollector):
    def update_batch_metrics(self, **kwargs):
        metrics = super().update_batch_metrics(**kwargs)
        
        # Add custom analysis
        student_outputs = kwargs['student_outputs']
        
        # Measure feature sparsity
        with torch.no_grad():
            features = torch.cat(student_outputs, dim=0)
            sparsity = (features.abs() < 0.01).float().mean().item()
            metrics['feature_sparsity'] = sparsity
            
        return metrics
```

### Exercise 3: Real-time Debugging
```python
# Set up alerts for training issues
class TrainingAlerter:
    def __init__(self, metrics_collector):
        self.metrics_collector = metrics_collector
        
    def check_training_health(self, step):
        """Check for common training issues."""
        recent_losses = list(self.metrics_collector.batch_metrics['loss'])[-50:]
        
        if len(recent_losses) > 10:
            # Check for loss explosion
            if recent_losses[-1] > 2 * recent_losses[0]:
                print("⚠️  WARNING: Loss explosion detected!")
                
            # Check for mode collapse
            recent_diversity = list(self.metrics_collector.batch_metrics['student_feature_diversity'])[-10:]
            if len(recent_diversity) > 5 and np.mean(recent_diversity) < 0.1:
                print("⚠️  WARNING: Possible mode collapse!")
```

## 🎯 Key Takeaways

1. **Comprehensive Monitoring**: Track loss, features, gradients, and system resources
2. **Real-time Visualization**: Use live dashboards for immediate feedback
3. **Professional Logging**: WandB integration for experiment tracking
4. **Training Diagnostics**: Detect mode collapse, gradient issues, and convergence problems
5. **Analysis Tools**: Post-training analysis for understanding training dynamics

## 🔍 What's Next?

In **Lesson 5.3**, we'll implement robust checkpoint management systems for handling long training runs, recovery from failures, and experiment reproducibility.

The monitoring system you've built provides unprecedented visibility into DINO training dynamics, enabling you to debug issues quickly and optimize training performance!
