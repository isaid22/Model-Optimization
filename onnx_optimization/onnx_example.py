"""
ONNX Model Optimization Example
This module demonstrates how to convert PyTorch models to ONNX format
and apply optimizations.
"""

import torch
import torch.nn as nn
import onnx
import onnxruntime as ort
import numpy as np


class SimpleModel(nn.Module):
    """Simple model for ONNX conversion demonstration."""
    
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, 10)
    
    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def export_to_onnx(
    model,
    dummy_input,
    onnx_path='model.onnx',
    opset_version=11
):
    """
    Export PyTorch model to ONNX format.
    
    Args:
        model: PyTorch model to export
        dummy_input: Example input tensor
        onnx_path: Path to save ONNX model
        opset_version: ONNX opset version
        
    Returns:
        Path to exported ONNX model
    """
    model.eval()
    
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    
    print(f"Model exported to {onnx_path}")
    return onnx_path


def verify_onnx_model(onnx_path):
    """
    Verify the exported ONNX model.
    
    Args:
        onnx_path: Path to ONNX model
        
    Returns:
        True if model is valid, False otherwise
    """
    try:
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
        print(f"ONNX model at {onnx_path} is valid!")
        return True
    except Exception as e:
        print(f"ONNX model validation failed: {e}")
        return False


def compare_outputs(pytorch_model, onnx_path, test_input):
    """
    Compare outputs between PyTorch and ONNX models.
    
    Args:
        pytorch_model: Original PyTorch model
        onnx_path: Path to ONNX model
        test_input: Test input tensor
        
    Returns:
        Dictionary with comparison results
    """
    # PyTorch inference
    pytorch_model.eval()
    with torch.no_grad():
        pytorch_output = pytorch_model(test_input).numpy()
    
    # ONNX inference
    ort_session = ort.InferenceSession(onnx_path)
    onnx_input = {ort_session.get_inputs()[0].name: test_input.numpy()}
    onnx_output = ort_session.run(None, onnx_input)[0]
    
    # Compare
    max_diff = np.max(np.abs(pytorch_output - onnx_output))
    mean_diff = np.mean(np.abs(pytorch_output - onnx_output))
    
    return {
        'max_difference': max_diff,
        'mean_difference': mean_diff,
        'outputs_match': np.allclose(pytorch_output, onnx_output, rtol=1e-03, atol=1e-05)
    }


def benchmark_onnx_inference(onnx_path, test_input, num_runs=100):
    """
    Benchmark ONNX model inference time.
    
    Args:
        onnx_path: Path to ONNX model
        test_input: Test input tensor
        num_runs: Number of inference runs
        
    Returns:
        Average inference time in milliseconds
    """
    import time
    
    ort_session = ort.InferenceSession(onnx_path)
    onnx_input = {ort_session.get_inputs()[0].name: test_input.numpy()}
    
    # Warmup
    for _ in range(10):
        _ = ort_session.run(None, onnx_input)
    
    # Benchmark
    start_time = time.time()
    for _ in range(num_runs):
        _ = ort_session.run(None, onnx_input)
    end_time = time.time()
    
    avg_time = ((end_time - start_time) / num_runs) * 1000
    return avg_time


def get_onnx_model_info(onnx_path):
    """
    Get information about ONNX model.
    
    Args:
        onnx_path: Path to ONNX model
        
    Returns:
        Dictionary with model information
    """
    onnx_model = onnx.load(onnx_path)
    
    # Get input/output info
    input_info = []
    for input_tensor in onnx_model.graph.input:
        shape = [dim.dim_value for dim in input_tensor.type.tensor_type.shape.dim]
        input_info.append({
            'name': input_tensor.name,
            'shape': shape
        })
    
    output_info = []
    for output_tensor in onnx_model.graph.output:
        shape = [dim.dim_value for dim in output_tensor.type.tensor_type.shape.dim]
        output_info.append({
            'name': output_tensor.name,
            'shape': shape
        })
    
    return {
        'ir_version': onnx_model.ir_version,
        'opset_version': onnx_model.opset_import[0].version,
        'inputs': input_info,
        'outputs': output_info,
        'num_nodes': len(onnx_model.graph.node)
    }


if __name__ == "__main__":
    print("ONNX Model Optimization Example")
    print("=" * 50)
    
    # Create and prepare model
    model = SimpleModel()
    model.eval()
    
    # Create dummy input
    dummy_input = torch.randn(1, 3, 32, 32)
    
    # Export to ONNX
    print("\n1. Exporting PyTorch model to ONNX")
    print("-" * 50)
    onnx_path = '/tmp/model.onnx'
    export_to_onnx(model, dummy_input, onnx_path)
    
    # Verify ONNX model
    print("\n2. Verifying ONNX model")
    print("-" * 50)
    is_valid = verify_onnx_model(onnx_path)
    
    # Get model info
    print("\n3. ONNX Model Information")
    print("-" * 50)
    info = get_onnx_model_info(onnx_path)
    print(f"IR Version: {info['ir_version']}")
    print(f"Opset Version: {info['opset_version']}")
    print(f"Number of nodes: {info['num_nodes']}")
    print(f"Inputs: {info['inputs']}")
    print(f"Outputs: {info['outputs']}")
    
    # Compare outputs
    print("\n4. Comparing PyTorch and ONNX outputs")
    print("-" * 50)
    test_input = torch.randn(1, 3, 32, 32)
    comparison = compare_outputs(model, onnx_path, test_input)
    print(f"Max difference: {comparison['max_difference']:.6f}")
    print(f"Mean difference: {comparison['mean_difference']:.6f}")
    print(f"Outputs match: {comparison['outputs_match']}")
    
    # Benchmark
    print("\n5. Benchmarking ONNX inference")
    print("-" * 50)
    onnx_time = benchmark_onnx_inference(onnx_path, test_input)
    print(f"Average ONNX inference time: {onnx_time:.3f} ms")
    
    print("\nONNX conversion and optimization completed successfully!")
