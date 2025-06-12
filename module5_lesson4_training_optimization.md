# Module 5, Lesson 4: Training Optimization

## 🎯 Learning Objectives
By the end of this lesson, you will:
- Optimize DINO training for memory efficiency and speed
- Implement advanced optimization techniques for large-scale training
- Profile and debug training performance bottlenecks
- Scale DINO training to multiple GPUs and large datasets

## 📚 Theoretical Background

### DINO Training Challenges

**Memory Bottlenecks**:
- **Multi-Crop Strategy**: 10+ crops per image increase memory usage 5-10x
- **Large Backbones**: ViT-Large requires >24GB GPU memory
- **Teacher Copies**: EMA teacher doubles memory requirements
- **Gradient Accumulation**: Large effective batch sizes need more memory

**Computational Bottlenecks**:
- **Vision Transformers**: Quadratic complexity in sequence length
- **Multiple Forward Passes**: Student + Teacher for each crop
- **Attention Computation**: Self-attention is compute-intensive
- **Loss Computation**: Cross-entropy over large vocabularies

### Optimization Strategies

**Memory Optimization**:
1. **Mixed Precision Training**: FP16 reduces memory by ~50%
2. **Gradient Checkpointing**: Trade compute for memory
3. **Crop Batching**: Process crops in smaller batches
4. **Parameter Sharing**: Share computation between crops

**Speed Optimization**:
1. **Efficient Data Loading**: Multi-process, pin memory, prefetch
2. **Optimized Operators**: Use fused kernels when available
3. **Model Compilation**: JIT compilation for inference speedup
4. **Batch Size Scaling**: Find optimal batch size for throughput

**Distributed Training**:
1. **Data Parallel**: Distribute batches across GPUs
2. **Model Parallel**: Split large models across GPUs
3. **Pipeline Parallel**: Pipeline stages across devices
4. **Gradient Accumulation**: Simulate large batch sizes

## 🛠️ Implementation

### Step 1: Memory-Optimized Training

```python
# optimization/memory_optimizer.py
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from typing import List, Dict, Optional, Tuple
import gc
import psutil
import time

class MemoryOptimizedDINOTrainer:
    """Memory-optimized DINO trainer for large models."""
    
    def __init__(self, 
                 model: nn.Module,
                 loss_fn: nn.Module,
                 optimizer: torch.optim.Optimizer,
                 config: Dict,
                 device: torch.device):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.config = config
        self.device = device
        
        # Mixed precision setup
        self.use_amp = config.get('use_amp', True)
        self.scaler = GradScaler() if self.use_amp else None
        
        # Memory optimization settings
        self.gradient_checkpointing = config.get('gradient_checkpointing', False)
        self.crop_batch_size = config.get('crop_batch_size', 4)
        self.accumulation_steps = config.get('accumulation_steps', 1)
        
        # Enable gradient checkpointing
        if self.gradient_checkpointing:
            self._enable_gradient_checkpointing()
            
        # Memory monitoring
        self.memory_tracker = MemoryTracker()
        
    def _enable_gradient_checkpointing(self):
        """Enable gradient checkpointing for memory savings."""
        def checkpoint_forward(module, input):
            """Checkpointed forward pass."""
            return torch.utils.checkpoint.checkpoint(module, input, use_reentrant=False)
            
        # Apply to transformer blocks if using ViT
        if hasattr(self.model.student, 'backbone'):
            backbone = self.model.student.backbone
            if hasattr(backbone, 'blocks'):  # ViT-style architecture
                for block in backbone.blocks:
                    block.forward = lambda x, block=block: checkpoint_forward(block, x)
                    
    def train_step_optimized(self, 
                           crops: List[torch.Tensor],
                           epoch: int) -> Tuple[float, Dict[str, float]]:
        """Memory-optimized training step."""
        
        self.memory_tracker.start_step()
        
        # Initialize metrics
        total_loss = 0.0
        metrics = {}
        
        # Process crops in smaller batches to save memory
        num_crop_batches = (len(crops) + self.crop_batch_size - 1) // self.crop_batch_size
        
        # Gradient accumulation
        self.optimizer.zero_grad()
        
        all_student_outputs = []
        all_teacher_outputs = []
        
        for crop_batch_idx in range(num_crop_batches):
            start_idx = crop_batch_idx * self.crop_batch_size
            end_idx = min(start_idx + self.crop_batch_size, len(crops))
            crop_batch = crops[start_idx:end_idx]
            
            # Mixed precision forward pass
            with autocast(enabled=self.use_amp):
                # Student forward pass
                student_batch_outputs = []
                for crop in crop_batch:
                    student_out = self.model.student(crop)
                    student_batch_outputs.append(student_out)
                    
                # Teacher forward pass (no gradients)
                with torch.no_grad():
                    teacher_batch_outputs = []
                    for crop in crop_batch:
                        teacher_out = self.model.teacher(crop)
                        teacher_batch_outputs.append(teacher_out)
                        
                all_student_outputs.extend(student_batch_outputs)
                all_teacher_outputs.extend(teacher_batch_outputs)
                
            # Memory cleanup between crop batches
            if crop_batch_idx < num_crop_batches - 1:
                torch.cuda.empty_cache()
                
        # Compute loss on all outputs
        with autocast(enabled=self.use_amp):
            loss = self.loss_fn(all_student_outputs, all_teacher_outputs)
            
            # Scale loss for gradient accumulation
            scaled_loss = loss / self.accumulation_steps
            
        # Backward pass with mixed precision
        if self.use_amp:
            self.scaler.scale(scaled_loss).backward()
        else:
            scaled_loss.backward()
            
        total_loss += loss.item()
        
        # Update metrics
        metrics.update(self.memory_tracker.get_memory_stats())
        
        return total_loss, metrics
        
    def optimize_step(self) -> bool:
        """Optimized parameter update."""
        # Gradient clipping with mixed precision
        if self.use_amp:
            self.scaler.unscale_(self.optimizer)
            
        # Clip gradients
        torch.nn.utils.clip_grad_norm_(
            self.model.student.parameters(),
            self.config.get('gradient_clip', 3.0)
        )
        
        # Optimizer step
        if self.use_amp:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()
            
        # Update teacher
        self.model.update_teacher()
        
        # Memory cleanup
        torch.cuda.empty_cache()
        
        return True


class MemoryTracker:
    """Track GPU and system memory usage."""
    
    def __init__(self):
        self.peak_memory = 0
        self.start_memory = 0
        
    def start_step(self):
        """Start tracking memory for this step."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            self.start_memory = torch.cuda.memory_allocated()
            
    def get_memory_stats(self) -> Dict[str, float]:
        """Get current memory statistics."""
        stats = {}
        
        if torch.cuda.is_available():
            current_memory = torch.cuda.memory_allocated()
            peak_memory = torch.cuda.max_memory_allocated()
            reserved_memory = torch.cuda.memory_reserved()
            
            stats.update({
                'gpu_memory_current_gb': current_memory / 1024**3,
                'gpu_memory_peak_gb': peak_memory / 1024**3,
                'gpu_memory_reserved_gb': reserved_memory / 1024**3,
                'gpu_memory_utilization': current_memory / torch.cuda.max_memory_allocated() if torch.cuda.max_memory_allocated() > 0 else 0
            })
            
        # System memory
        memory_info = psutil.virtual_memory()
        stats.update({
            'system_memory_percent': memory_info.percent,
            'system_memory_available_gb': memory_info.available / 1024**3
        })
        
        return stats


class DataLoadingOptimizer:
    """Optimize data loading for DINO training."""
    
    @staticmethod
    def create_optimized_dataloader(dataset, 
                                  batch_size: int,
                                  num_workers: int = None,
                                  pin_memory: bool = True,
                                  prefetch_factor: int = 2) -> torch.utils.data.DataLoader:
        """Create optimized dataloader."""
        
        # Auto-determine optimal num_workers
        if num_workers is None:
            num_workers = min(8, torch.get_num_threads())
            
        # Custom collate function for multi-crop
        def optimized_collate(batch):
            """Optimized collation for multi-crop data."""
            crops_list = []
            
            for sample in batch:
                crops_list.append(sample['crops'])
                
            # Stack crops efficiently
            num_crops = len(crops_list[0])
            batched_crops = []
            
            for crop_idx in range(num_crops):
                crop_batch = torch.stack([crops[crop_idx] for crops in crops_list])
                batched_crops.append(crop_batch)
                
            return {'crops': batched_crops}
            
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            prefetch_factor=prefetch_factor,
            persistent_workers=num_workers > 0,
            collate_fn=optimized_collate,
            drop_last=True
        )


class ModelOptimizer:
    """Optimize model architecture for training efficiency."""
    
    @staticmethod
    def optimize_model(model: nn.Module, config: Dict) -> nn.Module:
        """Apply model optimizations."""
        
        # Compile model for speed (PyTorch 2.0+)
        if hasattr(torch, 'compile') and config.get('compile_model', False):
            model = torch.compile(model, mode='reduce-overhead')
            
        # Enable channels_last memory format for CNNs
        if config.get('channels_last', False):
            model = model.to(memory_format=torch.channels_last)
            
        # Fuse operations where possible
        if hasattr(model, 'fuse_model'):
            model.fuse_model()
            
        return model
        
    @staticmethod
    def optimize_attention(model: nn.Module, use_flash_attention: bool = True):
        """Optimize attention computation."""
        if not use_flash_attention:
            return model
            
        # Replace attention with flash attention if available
        try:
            from flash_attn import flash_attn_func
            
            def flash_attention_forward(self, x):
                # Simplified flash attention implementation
                B, N, C = x.shape
                
                # Compute Q, K, V
                qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
                q, k, v = qkv.permute(2, 0, 3, 1, 4)
                
                # Flash attention
                out = flash_attn_func(q, k, v, dropout_p=self.dropout)
                out = out.transpose(1, 2).reshape(B, N, C)
                
                return self.proj(out)
                
            # Replace attention in transformer blocks
            if hasattr(model, 'blocks'):
                for block in model.blocks:
                    if hasattr(block, 'attn'):
                        block.attn.forward = flash_attention_forward.__get__(block.attn)
                        
        except ImportError:
            print("Flash attention not available, using standard attention")
            
        return model
```

### Step 2: Performance Profiling and Debugging

```python
# optimization/profiler.py
import torch
import time
import numpy as np
from typing import Dict, List, Optional
import matplotlib.pyplot as plt
from contextlib import contextmanager
import cProfile
import pstats
import io

class DINOProfiler:
    """Comprehensive profiler for DINO training."""
    
    def __init__(self):
        self.timings = {}
        self.memory_usage = {}
        self.active_timers = {}
        
    @contextmanager
    def profile_block(self, name: str):
        """Profile a code block."""
        start_time = time.time()
        start_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        
        try:
            yield
        finally:
            end_time = time.time()
            end_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
            
            # Record timing
            if name not in self.timings:
                self.timings[name] = []
            self.timings[name].append(end_time - start_time)
            
            # Record memory
            if name not in self.memory_usage:
                self.memory_usage[name] = []
            self.memory_usage[name].append((end_memory - start_memory) / 1024**2)  # MB
            
    def profile_training_step(self, 
                            model: nn.Module,
                            crops: List[torch.Tensor],
                            loss_fn: nn.Module,
                            optimizer: torch.optim.Optimizer) -> Dict[str, float]:
        """Profile a complete training step."""
        
        step_timings = {}
        
        # Data movement
        with self.profile_block('data_movement'):
            crops_gpu = [crop.cuda() for crop in crops]
            
        # Student forward pass
        with self.profile_block('student_forward'):
            student_outputs = []
            for crop in crops_gpu:
                output = model.student(crop)
                student_outputs.append(output)
                
        # Teacher forward pass
        with self.profile_block('teacher_forward'):
            with torch.no_grad():
                teacher_outputs = []
                for crop in crops_gpu:
                    output = model.teacher(crop)
                    teacher_outputs.append(output)
                    
        # Loss computation
        with self.profile_block('loss_computation'):
            loss = loss_fn(student_outputs, teacher_outputs)
            
        # Backward pass
        with self.profile_block('backward_pass'):
            loss.backward()
            
        # Optimizer step
        with self.profile_block('optimizer_step'):
            optimizer.step()
            optimizer.zero_grad()
            
        # Teacher update
        with self.profile_block('teacher_update'):
            model.update_teacher()
            
        # Get latest timings
        for key, values in self.timings.items():
            if values:
                step_timings[f'{key}_time'] = values[-1]
                
        return step_timings
        
    def get_summary_stats(self) -> Dict[str, Dict[str, float]]:
        """Get summary statistics for all profiled operations."""
        stats = {}
        
        for operation, times in self.timings.items():
            if times:
                stats[operation] = {
                    'mean_time': np.mean(times),
                    'std_time': np.std(times),
                    'min_time': np.min(times),
                    'max_time': np.max(times),
                    'total_time': np.sum(times),
                    'count': len(times)
                }
                
        return stats
        
    def plot_performance_analysis(self) -> plt.Figure:
        """Create performance analysis plots."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('DINO Training Performance Analysis')
        
        # Timing breakdown
        operations = list(self.timings.keys())
        mean_times = [np.mean(self.timings[op]) for op in operations]
        
        axes[0, 0].bar(operations, mean_times)
        axes[0, 0].set_title('Average Time per Operation')
        axes[0, 0].set_ylabel('Time (seconds)')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Memory usage
        if self.memory_usage:
            memory_ops = list(self.memory_usage.keys())
            mean_memory = [np.mean(self.memory_usage[op]) for op in memory_ops]
            
            axes[0, 1].bar(memory_ops, mean_memory)
            axes[0, 1].set_title('Average Memory Usage per Operation')
            axes[0, 1].set_ylabel('Memory (MB)')
            axes[0, 1].tick_params(axis='x', rotation=45)
            
        # Time evolution
        if 'student_forward' in self.timings:
            axes[1, 0].plot(self.timings['student_forward'], label='Student Forward')
        if 'teacher_forward' in self.timings:
            axes[1, 0].plot(self.timings['teacher_forward'], label='Teacher Forward')
        if 'loss_computation' in self.timings:
            axes[1, 0].plot(self.timings['loss_computation'], label='Loss Computation')
            
        axes[1, 0].set_title('Timing Evolution')
        axes[1, 0].set_xlabel('Step')
        axes[1, 0].set_ylabel('Time (seconds)')
        axes[1, 0].legend()
        
        # Efficiency metrics
        if len(operations) > 0:
            efficiency_scores = []
            for op in operations:
                # Simple efficiency score (inverse of coefficient of variation)
                times = self.timings[op]
                if len(times) > 1:
                    cv = np.std(times) / np.mean(times)
                    efficiency_scores.append(1 / (1 + cv))
                else:
                    efficiency_scores.append(1.0)
                    
            axes[1, 1].bar(operations, efficiency_scores)
            axes[1, 1].set_title('Operation Consistency (Higher = Better)')
            axes[1, 1].set_ylabel('Consistency Score')
            axes[1, 1].tick_params(axis='x', rotation=45)
            
        plt.tight_layout()
        return fig
        
    def export_profile_data(self, filename: str):
        """Export profiling data to file."""
        import json
        
        data = {
            'timings': {k: v for k, v in self.timings.items()},
            'memory_usage': {k: v for k, v in self.memory_usage.items()},
            'summary_stats': self.get_summary_stats()
        }
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2, default=str)


class ThroughputOptimizer:
    """Optimize training throughput."""
    
    def __init__(self, model: nn.Module, device: torch.device):
        self.model = model
        self.device = device
        
    def find_optimal_batch_size(self, 
                               dataloader: torch.utils.data.DataLoader,
                               max_batch_size: int = 128,
                               target_memory_fraction: float = 0.9) -> int:
        """Find optimal batch size for maximum throughput."""
        
        if not torch.cuda.is_available():
            return dataloader.batch_size
            
        optimal_batch_size = dataloader.batch_size
        best_throughput = 0
        
        # Test different batch sizes
        test_sizes = [2**i for i in range(1, int(np.log2(max_batch_size)) + 1)]
        
        for batch_size in test_sizes:
            try:
                # Create test batch
                test_batch = self._create_test_batch(dataloader, batch_size)
                
                # Measure throughput
                throughput = self._measure_throughput(test_batch)
                
                # Check memory usage
                memory_fraction = torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated()
                
                if memory_fraction < target_memory_fraction and throughput > best_throughput:
                    best_throughput = throughput
                    optimal_batch_size = batch_size
                    
                # Clear memory
                torch.cuda.empty_cache()
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    break
                else:
                    raise e
                    
        return optimal_batch_size
        
    def _create_test_batch(self, dataloader, batch_size):
        """Create test batch with specified size."""
        # Get sample from dataloader
        sample_batch = next(iter(dataloader))
        crops = sample_batch['crops']
        
        # Replicate to desired batch size
        test_crops = []
        for crop in crops:
            # Repeat batch dimension
            current_batch_size = crop.size(0)
            repeat_factor = (batch_size + current_batch_size - 1) // current_batch_size
            repeated_crop = crop.repeat(repeat_factor, 1, 1, 1)[:batch_size]
            test_crops.append(repeated_crop.to(self.device))
            
        return test_crops
        
    def _measure_throughput(self, test_batch, num_iterations: int = 10) -> float:
        """Measure training throughput."""
        self.model.train()
        
        # Warmup
        for _ in range(3):
            with torch.no_grad():
                for crop in test_batch:
                    _ = self.model.student(crop)
                    
        # Measure
        torch.cuda.synchronize()
        start_time = time.time()
        
        for _ in range(num_iterations):
            for crop in test_batch:
                _ = self.model.student(crop)
                
        torch.cuda.synchronize()
        end_time = time.time()
        
        total_samples = len(test_batch) * test_batch[0].size(0) * num_iterations
        throughput = total_samples / (end_time - start_time)
        
        return throughput
```

### Step 3: Distributed Training Optimization

```python
# optimization/distributed_trainer.py
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from typing import Dict, Optional
import os

class DistributedDINOTrainer:
    """Distributed DINO training for multi-GPU setups."""
    
    def __init__(self, 
                 rank: int,
                 world_size: int,
                 config: Dict,
                 master_addr: str = 'localhost',
                 master_port: str = '12355'):
        self.rank = rank
        self.world_size = world_size
        self.config = config
        
        # Initialize distributed training
        os.environ['MASTER_ADDR'] = master_addr
        os.environ['MASTER_PORT'] = master_port
        
        dist.init_process_group(
            backend='nccl',
            rank=rank,
            world_size=world_size
        )
        
        # Set device
        torch.cuda.set_device(rank)
        self.device = torch.device(f'cuda:{rank}')
        
        print(f"Initialized distributed training: rank {rank}/{world_size}")
        
    def setup_distributed_model(self, model: nn.Module) -> nn.Module:
        """Setup model for distributed training."""
        model = model.to(self.device)
        
        # Wrap in DDP
        model = DDP(
            model,
            device_ids=[self.rank],
            output_device=self.rank,
            find_unused_parameters=False  # Set to True if needed
        )
        
        return model
        
    def setup_distributed_dataloader(self, dataset) -> torch.utils.data.DataLoader:
        """Setup distributed data loading."""
        
        # Create distributed sampler
        sampler = DistributedSampler(
            dataset,
            num_replicas=self.world_size,
            rank=self.rank,
            shuffle=True
        )
        
        # Create dataloader
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.config['training']['batch_size'] // self.world_size,
            sampler=sampler,
            num_workers=self.config['training']['num_workers'],
            pin_memory=True,
            collate_fn=self._collate_fn
        )
        
        return dataloader
        
    def _collate_fn(self, batch):
        """Custom collate function for multi-crop data."""
        crops_list = []
        
        for sample in batch:
            crops_list.append(sample['crops'])
            
        # Stack crops
        num_crops = len(crops_list[0])
        batched_crops = []
        
        for crop_idx in range(num_crops):
            crop_batch = torch.stack([crops[crop_idx] for crops in crops_list])
            batched_crops.append(crop_batch.to(self.device))
            
        return {'crops': batched_crops}
        
    def all_reduce_metrics(self, metrics: Dict[str, float]) -> Dict[str, float]:
        """Average metrics across all processes."""
        for key, value in metrics.items():
            tensor = torch.tensor(value, device=self.device)
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
            metrics[key] = tensor.item() / self.world_size
            
        return metrics
        
    def save_checkpoint(self, checkpoint_data: Dict, filepath: str):
        """Save checkpoint from rank 0 only."""
        if self.rank == 0:
            torch.save(checkpoint_data, filepath)
            
        # Synchronize all processes
        dist.barrier()
        
    def cleanup(self):
        """Cleanup distributed training."""
        dist.destroy_process_group()


def launch_distributed_training(config_path: str, 
                               num_gpus: int,
                               master_addr: str = 'localhost',
                               master_port: str = '12355'):
    """Launch distributed training across multiple GPUs."""
    
    def train_worker(rank: int):
        """Worker function for distributed training."""
        try:
            # Initialize distributed trainer
            trainer = DistributedDINOTrainer(
                rank=rank,
                world_size=num_gpus,
                config=config,
                master_addr=master_addr,
                master_port=master_port
            )
            
            # Setup model and data
            # ... (implementation specific to your setup)
            
            # Training loop
            # ... (your training code here)
            
        except Exception as e:
            print(f"Error in rank {rank}: {e}")
            raise
        finally:
            trainer.cleanup()
            
    # Launch processes
    mp.spawn(
        train_worker,
        args=(),
        nprocs=num_gpus,
        join=True
    )
```

## 🧪 Practical Exercises

### Exercise 1: Memory Optimization Benchmark
```python
# Benchmark memory usage with different optimization techniques
def benchmark_memory_optimizations():
    """Compare memory usage across optimization techniques."""
    
    configs = [
        {'name': 'Baseline', 'use_amp': False, 'gradient_checkpointing': False},
        {'name': 'Mixed Precision', 'use_amp': True, 'gradient_checkpointing': False},
        {'name': 'Grad Checkpointing', 'use_amp': False, 'gradient_checkpointing': True},
        {'name': 'Both Optimizations', 'use_amp': True, 'gradient_checkpointing': True}
    ]
    
    results = {}
    
    for config in configs:
        trainer = MemoryOptimizedDINOTrainer(model, loss_fn, optimizer, config, device)
        
        # Measure peak memory
        torch.cuda.reset_peak_memory_stats()
        loss, metrics = trainer.train_step_optimized(crops, epoch=0)
        peak_memory = torch.cuda.max_memory_allocated() / 1024**3  # GB
        
        results[config['name']] = {
            'peak_memory_gb': peak_memory,
            'loss': loss
        }
        
    return results
```

### Exercise 2: Throughput Optimization
```python
# Find optimal configuration for maximum throughput
def optimize_training_throughput(model, dataloader):
    """Find optimal settings for maximum training throughput."""
    
    optimizer = ThroughputOptimizer(model, device)
    profiler = DINOProfiler()
    
    # Find optimal batch size
    optimal_batch_size = optimizer.find_optimal_batch_size(dataloader)
    print(f"Optimal batch size: {optimal_batch_size}")
    
    # Profile training with optimal settings
    for step, batch in enumerate(dataloader):
        if step >= 10:  # Profile first 10 steps
            break
            
        metrics = profiler.profile_training_step(
            model, batch['crops'], loss_fn, optimizer
        )
        
    # Analyze results
    summary = profiler.get_summary_stats()
    fig = profiler.plot_performance_analysis()
    
    return optimal_batch_size, summary, fig
```

### Exercise 3: Distributed Training Setup
```python
# Setup distributed training
def setup_distributed_dino_training():
    """Setup and launch distributed DINO training."""
    
    if torch.cuda.device_count() < 2:
        print("Need at least 2 GPUs for distributed training")
        return
        
    # Launch distributed training
    launch_distributed_training(
        config_path='config/distributed_config.yaml',
        num_gpus=torch.cuda.device_count()
    )
```

## 🎯 Key Takeaways

1. **Memory Efficiency**: Mixed precision and gradient checkpointing reduce memory by 50-70%
2. **Speed Optimization**: Proper data loading and model compilation improve throughput by 20-40%
3. **Profiling Tools**: Systematic profiling identifies bottlenecks and optimization opportunities
4. **Distributed Training**: Scale to multiple GPUs with minimal code changes
5. **Adaptive Optimization**: Automatically find optimal batch sizes and configurations

## 🚀 Production Tips

### Memory Management Best Practices
```python
# Memory-efficient training tips
tips = {
    'gradient_accumulation': "Simulate large batch sizes without OOM",
    'crop_batching': "Process multi-crop in smaller batches",
    'checkpointing': "Trade compute for memory in transformers",
    'mixed_precision': "Use FP16 for 50% memory reduction",
    'empty_cache': "Call torch.cuda.empty_cache() between batches"
}
```

### Performance Monitoring
```python
# Set up continuous performance monitoring
class PerformanceMonitor:
    def __init__(self):
        self.throughput_history = []
        self.memory_history = []
        
    def log_performance(self, throughput, memory_usage):
        self.throughput_history.append(throughput)
        self.memory_history.append(memory_usage)
        
        # Alert if performance degrades
        if len(self.throughput_history) > 100:
            recent_avg = np.mean(self.throughput_history[-10:])
            baseline_avg = np.mean(self.throughput_history[:10])
            
            if recent_avg < 0.8 * baseline_avg:
                print("⚠️ Performance degradation detected!")
```

## 🔍 Module 5 Summary

You've now completed Module 5 with a production-ready DINO training system that includes:

✅ **Complete Training Implementation** - End-to-end training pipeline
✅ **Comprehensive Monitoring** - Real-time metrics and visualization  
✅ **Robust Checkpointing** - Fault-tolerant training with automatic recovery
✅ **Performance Optimization** - Memory and speed optimizations for large-scale training

In **Module 6**, we'll shift focus to evaluation and analysis, implementing k-NN classification, linear probing, and feature visualization to understand what DINO has learned!

The optimization techniques you've implemented can improve training speed by 2-3x and enable training of much larger models that wouldn't fit in memory otherwise. This is the same level of optimization used in production machine learning systems!
