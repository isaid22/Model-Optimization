"""
PyTorch Model Quantization Example
This module demonstrates dynamic and static quantization techniques.
"""

import torch
import torch.nn as nn
import torch.quantization as quantization
import time


class SimpleNet(nn.Module):
    """Simple neural network for demonstration."""
    
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(784, 256)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)
    
    def forward(self, x):
        x = x.view(-1, 784)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def dynamic_quantization(model):
    """
    Apply dynamic quantization to the model.
    Dynamic quantization quantizes weights ahead of time but quantizes activations dynamically.
    
    Args:
        model: PyTorch model to quantize
        
    Returns:
        Quantized model
    """
    quantized_model = torch.quantization.quantize_dynamic(
        model, {nn.Linear}, dtype=torch.qint8
    )
    return quantized_model


def static_quantization(model, example_inputs):
    """
    Apply static quantization to the model.
    Static quantization requires calibration with representative data.
    
    Args:
        model: PyTorch model to quantize
        example_inputs: Example inputs for calibration
        
    Returns:
        Quantized model
    """
    model.eval()
    model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
    model_prepared = torch.quantization.prepare(model)
    
    # Calibration step
    with torch.no_grad():
        for inputs in example_inputs:
            model_prepared(inputs)
    
    quantized_model = torch.quantization.convert(model_prepared)
    return quantized_model


def compare_model_sizes(original_model, quantized_model):
    """
    Compare sizes of original and quantized models.
    
    Args:
        original_model: Original model
        quantized_model: Quantized model
        
    Returns:
        Tuple of (original_size, quantized_size, compression_ratio)
    """
    # Save models temporarily to measure size
    torch.save(original_model.state_dict(), '/tmp/original_model.pth')
    torch.save(quantized_model.state_dict(), '/tmp/quantized_model.pth')
    
    import os
    original_size = os.path.getsize('/tmp/original_model.pth') / (1024 * 1024)  # MB
    quantized_size = os.path.getsize('/tmp/quantized_model.pth') / (1024 * 1024)  # MB
    compression_ratio = original_size / quantized_size
    
    # Clean up
    os.remove('/tmp/original_model.pth')
    os.remove('/tmp/quantized_model.pth')
    
    return original_size, quantized_size, compression_ratio


def benchmark_inference(model, input_tensor, num_runs=100):
    """
    Benchmark inference time of a model.
    
    Args:
        model: Model to benchmark
        input_tensor: Input tensor
        num_runs: Number of inference runs
        
    Returns:
        Average inference time in milliseconds
    """
    model.eval()
    with torch.no_grad():
        # Warmup
        for _ in range(10):
            _ = model(input_tensor)
        
        # Actual benchmark
        start_time = time.time()
        for _ in range(num_runs):
            _ = model(input_tensor)
        end_time = time.time()
    
    avg_time = ((end_time - start_time) / num_runs) * 1000  # Convert to ms
    return avg_time


if __name__ == "__main__":
    print("PyTorch Model Quantization Example")
    print("=" * 50)
    
    # Create original model
    model = SimpleNet()
    print(f"Original model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Create sample input
    sample_input = torch.randn(1, 28, 28)
    
    # Dynamic Quantization
    print("\n1. Dynamic Quantization")
    print("-" * 50)
    dynamic_quant_model = dynamic_quantization(model)
    
    orig_size, quant_size, ratio = compare_model_sizes(model, dynamic_quant_model)
    print(f"Original model size: {orig_size:.2f} MB")
    print(f"Quantized model size: {quant_size:.2f} MB")
    print(f"Compression ratio: {ratio:.2f}x")
    
    # Benchmark
    orig_time = benchmark_inference(model, sample_input)
    quant_time = benchmark_inference(dynamic_quant_model, sample_input)
    print(f"Original inference time: {orig_time:.3f} ms")
    print(f"Quantized inference time: {quant_time:.3f} ms")
    print(f"Speedup: {orig_time/quant_time:.2f}x")
    
    print("\nQuantization completed successfully!")
