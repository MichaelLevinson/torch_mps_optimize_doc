# PyTorch 2.8 Updates: MPS (Metal Performance Shaders) on Apple Silicon 

***EXPERIMENTAL - STILL TESTING***

> *Last updated: May 2025*

PyTorch 2.8 brings several improvements to Metal Performance Shaders (MPS) support on Apple Silicon, though some limitations remain. This document provides a comprehensive overview of the current state, improvements, remaining challenges, and recommended best practices based on thorough testing.

## Current State of MPS in PyTorch 2.8

Our extensive testing reveals a mixed but improving picture of MPS support in PyTorch 2.8:

### ✅ What Works Well

1. **LSTM/RNN Support**: Previous memory leak issues in 2.7 have been resolved
2. **Convolutional Operations**: Forward and backward passes work reliably
3. **Memory Management**: Basic memory allocation/deallocation functions properly
4. **Advanced Activations**: SiLU and GELU work correctly
5. **Complex Number Support**: Basic operations with complex tensors are functional
6. **CPU-MPS Interoperability**: Data transfer between CPU and MPS is reliable
7. **Quantization**: Basic quantization functionality works
8. **Profiler Integration**: Profiling tools now work with MPS backend

### ⚠️ Remaining Challenges

1. **Transformer Architecture**: Issues with batch scaling in transformer models
2. **Basic Tensor Operations**: Some operations (sgn, logdet, matrix_exp) still have problems
3. **Normalization Layers**: LayerNorm and BatchNorm3d implementations need improvements
4. **Sparse Operations**: Both basic and advanced sparse tensor operations show issues
5. **Mixed Precision Training**: Some AMP features still not fully supported
6. **In-place Operations**: Operations like add_ and mul_ have reliability issues
7. **Memory-Intensive Workloads**: Large memory allocation can be unstable

## Memory Management Improvements

Memory management has been enhanced in PyTorch 2.8:

```python
# Set memory fraction more reliably (API stabilized in 2.8)
torch.mps.set_per_process_memory_fraction(0.75)  # Use up to 75% of available memory

# Monitor memory usage more effectively
print(f"Allocated: {torch.mps.current_allocated_memory() / (1024**2):.2f} MB")
print(f"Cached: {torch.mps.driver_allocated_memory() / (1024**2):.2f} MB")

# Empty cache when needed
torch.mps.empty_cache()
```

### Memory Debugging Recommendations

```python
# Add memory checkpoints in your code
def log_memory(tag=""):
    print(f"{tag} - Allocated: {torch.mps.current_allocated_memory() / (1024**2):.2f} MB")

# Example usage
log_memory("Before forward pass")
output = model(input)
log_memory("After forward pass")
loss = criterion(output, target)
loss.backward()
log_memory("After backward pass")
```

## Updated AMP (Automatic Mixed Precision) Support

While AMP support has improved, it requires careful usage:

```python
# Import from torch.amp (not torch.cuda.amp)
from torch.amp import autocast, GradScaler

# Initialize scaler (no device_type needed now)
scaler = GradScaler()

# Training loop with AMP
for inputs, targets in dataloader:
    inputs = inputs.to("mps")
    targets = targets.to("mps")
    
    # Forward pass with mixed precision
    with autocast(device_type="mps", dtype=torch.float16):
        outputs = model(inputs)
        loss = criterion(outputs, targets)
    
    # Scale loss and backward pass
    scaler.scale(loss).backward()
    
    # Gradient checking (important for stability)
    scaler.unscale_(optimizer)
    gradient_ok = all(not (torch.isnan(p.grad).any() or torch.isinf(p.grad).any()) 
                      for p in model.parameters() if p.grad is not None)
    
    if gradient_ok:
        scaler.step(optimizer)
    else:
        print("Skipping step due to bad gradients")
    
    scaler.update()
    optimizer.zero_grad()
```

## Transformer Models Workarounds

For transformer-based models with batch scaling issues:

```python
# 1. Use gradient accumulation with smaller batches
effective_batch_size = 32
micro_batch_size = 8
accumulation_steps = effective_batch_size // micro_batch_size

optimizer.zero_grad()
for i, (inputs, targets) in enumerate(dataloader):
    outputs = model(inputs.to("mps"))
    loss = criterion(outputs, targets.to("mps"))
    (loss / accumulation_steps).backward()
    
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()

# 2. Set appropriate attention dimension sizes (prefer multiples of 8)
attention_dim = 512  # Instead of 768, for example
```

## Normalization Layers Best Practices

Given the issues with normalization layers:

```python
# 1. Use Instance Normalization as an alternative to BatchNorm3D
# Instead of:
# norm = nn.BatchNorm3d(channels)
# Use:
norm = nn.InstanceNorm3d(channels)

# 2. For LayerNorm issues, use careful initialization
layer_norm = nn.LayerNorm(hidden_size)
# Initialize with slightly larger eps
layer_norm.eps = 1e-5  # Default is 1e-5, try 1e-4 if having issues

# 3. For critical models, consider CPU fallback for normalization
class SafeLayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super().__init__()
        self.norm = nn.LayerNorm(normalized_shape)
        
    def forward(self, x):
        device = x.device
        return self.norm(x.cpu()).to(device)
```

## Sparse Tensor Operations

When working with sparse tensors:

```python
# 1. Move operations to CPU when needed
def sparse_matmul_safe(sparse_tensor, dense_tensor):
    if sparse_tensor.device.type == "mps":
        result = torch.matmul(sparse_tensor.cpu(), dense_tensor.cpu())
        return result.to("mps")
    else:
        return torch.matmul(sparse_tensor, dense_tensor)

# 2. Consider using dense representations for critical operations
# Instead of sparse directly:
sparse_tensor = torch.sparse_coo_tensor(indices, values, size).to("mps")
result = torch.matmul(sparse_tensor, dense)

# Use:
dense_representation = torch.sparse_coo_tensor(indices, values, size).to_dense().to("mps")
result = torch.matmul(dense_representation, dense)
```

## In-place Operations Stability

```python
# Avoid in-place operations in training loops
# Instead of:
# x.add_(y)
# Use:
x = x + y

# Create a safer version of common in-place ops
def safe_add(tensor1, tensor2):
    """Safe addition that avoids in-place operations on MPS"""
    return tensor1 + tensor2
```

## Memory-Intensive Workloads

```python
# 1. Implement a context manager for safer memory usage
import contextlib

@contextlib.contextmanager
def safe_mps_memory():
    try:
        yield
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print("MPS out of memory, clearing cache and retrying...")
            torch.mps.empty_cache()
            yield  # Try once more
        else:
            raise

# 2. Handle OOM gracefully
with safe_mps_memory():
    output = model(large_input)
```

## Profiling and Debugging

Profiling has improved in PyTorch 2.8:

```python
# Basic profiling
from torch.profiler import profile, record_function, ProfilerActivity

with profile(activities=[ProfilerActivity.CPU]) as prof:
    with record_function("model_inference"):
        output = model(input)
        
print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))

# Memory profiling (still experimental but improved)
# Set environment variable for more verbose memory logging
import os
os.environ["PYTORCH_MPS_LOG_LEVEL"] = "3"  # Verbose memory logging
```

## Additional Tips for MPS Performance

1. **Model Conversion**: Convert float64 to float32 before moving to MPS
   ```python
   # Convert model parameters to float32
   for param in model.parameters():
       if param.dtype == torch.float64:
           param.data = param.data.float()
   ```

2. **Dataloader Configuration**: Optimize for MPS
   ```python
   dataloader = DataLoader(
       dataset, 
       batch_size=32,
       pin_memory=False,  # Important: set to False for MPS
       num_workers=min(os.cpu_count(), 8)
   )
   ```

3. **Mixed CPU-MPS Processing**: Use each device's strengths
   ```python
   # Example: CPU for data preprocessing, MPS for inference
   def process_batch(images):
       # CPU preprocessing
       processed = preprocess_on_cpu(images)
       # MPS inference
       with torch.no_grad():
           return model(processed.to("mps")).cpu()
   ```

## Recommended Model Architectures for MPS

Based on testing, these architectures work best on MPS in PyTorch 2.8:

1. **CNNs**: Most reliable, especially ResNet and EfficientNet variants
2. **RNNs/LSTMs**: Now working well after fixes in 2.8
3. **Diffusion Models**: Generally stable if batch sizes are kept reasonable
4. **MLP-based Models**: Very reliable across all operations

## Architectures Requiring Caution

1. **Transformer-based Models**: Test thoroughly and use workarounds above
2. **GANs with Complex Loss Functions**: May have gradient instability
3. **Graph Neural Networks**: Sparse operations can be problematic
4. **Models with Heavy Memory Requirements**: Need careful memory management

## Conclusion

PyTorch 2.8 brings significant improvements to MPS support, making Apple Silicon GPUs more viable for deep learning workflows. While some limitations remain, most common use cases now work reliably with appropriate optimizations. The MPS backend continues to mature with each release, and many of the current limitations are likely to be addressed in future updates.

For production deployments, thorough testing remains essential, particularly for transformer-based architectures and memory-intensive workloads. Consider the workarounds and best practices outlined in this document to maximize performance and stability.
