"""
Common utilities for model optimization.
"""

import torch
import torch.nn as nn
import time
import os


def count_parameters(model):
    """
    Count the number of trainable parameters in a model.
    
    Args:
        model: PyTorch model
        
    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model_size_mb(model):
    """
    Calculate the size of a model in megabytes.
    
    Args:
        model: PyTorch model
        
    Returns:
        Model size in MB
    """
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    size_mb = (param_size + buffer_size) / (1024 ** 2)
    return size_mb


def benchmark_model(model, input_shape, device='cpu', num_runs=100, warmup_runs=10):
    """
    Benchmark model inference time.
    
    Args:
        model: PyTorch model
        input_shape: Shape of input tensor (excluding batch dimension)
        device: Device to run on ('cpu' or 'cuda')
        num_runs: Number of inference runs for benchmarking
        warmup_runs: Number of warmup runs
        
    Returns:
        Dictionary with timing statistics
    """
    model = model.to(device)
    model.eval()
    
    dummy_input = torch.randn(1, *input_shape).to(device)
    
    # Warmup
    with torch.no_grad():
        for _ in range(warmup_runs):
            _ = model(dummy_input)
    
    # Benchmark
    times = []
    with torch.no_grad():
        for _ in range(num_runs):
            start = time.time()
            _ = model(dummy_input)
            if device == 'cuda':
                torch.cuda.synchronize()
            end = time.time()
            times.append((end - start) * 1000)  # Convert to ms
    
    return {
        'mean_ms': sum(times) / len(times),
        'std_ms': (sum((x - sum(times) / len(times)) ** 2 for x in times) / len(times)) ** 0.5,
        'min_ms': min(times),
        'max_ms': max(times)
    }


def save_model(model, path, optimizer=None, epoch=None, metadata=None):
    """
    Save model checkpoint with optional training state.
    
    Args:
        model: PyTorch model
        path: Path to save checkpoint
        optimizer: Optional optimizer state
        epoch: Optional epoch number
        metadata: Optional dictionary of additional metadata
    """
    checkpoint = {
        'model_state_dict': model.state_dict(),
    }
    
    if optimizer is not None:
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()
    
    if epoch is not None:
        checkpoint['epoch'] = epoch
    
    if metadata is not None:
        checkpoint['metadata'] = metadata
    
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(checkpoint, path)
    print(f"Model saved to {path}")


def load_model(model, path, optimizer=None, device='cpu'):
    """
    Load model checkpoint.
    
    Args:
        model: PyTorch model to load state into
        path: Path to checkpoint
        optimizer: Optional optimizer to load state into
        device: Device to load model on
        
    Returns:
        Dictionary with loaded metadata
    """
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    metadata = {}
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if 'epoch' in checkpoint:
        metadata['epoch'] = checkpoint['epoch']
    
    if 'metadata' in checkpoint:
        metadata.update(checkpoint['metadata'])
    
    print(f"Model loaded from {path}")
    return metadata


def print_model_summary(model, input_shape=None):
    """
    Print a summary of the model architecture.
    
    Args:
        model: PyTorch model
        input_shape: Optional input shape for detailed summary
    """
    print("=" * 70)
    print("Model Summary")
    print("=" * 70)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Non-trainable parameters: {total_params - trainable_params:,}")
    print(f"Model size: {get_model_size_mb(model):.2f} MB")
    
    print("\nModel Architecture:")
    print("-" * 70)
    print(model)
    print("=" * 70)


class AverageMeter:
    """Computes and stores the average and current value."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
