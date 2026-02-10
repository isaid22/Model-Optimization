# Model Optimization

This repository contains practical examples and implementations of various model optimization techniques for deep learning models. These techniques help reduce model size, improve inference speed, and maintain accuracy for deployment in resource-constrained environments.

## Overview

Model optimization is crucial for deploying deep learning models in production, especially on edge devices or mobile platforms. This repository demonstrates four key optimization techniques:

1. **Quantization** - Reduce model size and improve inference speed by converting weights and activations to lower precision
2. **Pruning** - Remove redundant parameters from the model to reduce size and computation
3. **Knowledge Distillation** - Train smaller student models to mimic larger teacher models
4. **ONNX Optimization** - Convert and optimize models for efficient cross-platform deployment

## Installation

Install the required dependencies:

```bash
pip install -r requirements.txt
```

### Requirements

- Python 3.8+
- PyTorch 2.0+
- ONNX 1.14+
- ONNXRuntime 1.15+
- NumPy 1.24+

## Optimization Techniques

### 1. Quantization

Quantization reduces model size by converting floating-point weights to lower precision (e.g., int8).

**Features:**
- Dynamic quantization (quantizes weights ahead of time, activations at runtime)
- Static quantization (requires calibration data)
- Model size reduction: 2-4x
- Inference speedup: 1.5-3x on CPU

**Usage:**
```bash
python quantization/quantization_example.py
```

**Example output:**
- Original model size: ~2 MB
- Quantized model size: ~0.5 MB
- Compression ratio: ~4x
- Inference speedup: ~2x

### 2. Pruning

Pruning removes less important weights or entire structures from the model.

**Features:**
- Magnitude-based pruning (L1 unstructured)
- Structured pruning (removes entire filters/channels)
- Configurable sparsity levels
- Support for making pruning permanent

**Usage:**
```bash
python pruning/pruning_example.py
```

**Benefits:**
- Reduces model parameters by 20-80%
- Can be combined with quantization for greater compression
- Maintains model accuracy with proper fine-tuning

### 3. Knowledge Distillation

Transfer knowledge from a large, complex teacher model to a smaller student model.

**Features:**
- Temperature-based soft target training
- Combined hard and soft target losses
- Configurable distillation parameters
- Model compression: 5-10x smaller

**Usage:**
```bash
python knowledge_distillation/distillation_example.py
```

**Benefits:**
- Smaller student model learns from teacher's knowledge
- Better performance than training student from scratch
- Significant parameter reduction (60-90%)

### 4. ONNX Optimization

Convert PyTorch models to ONNX format for optimized cross-platform deployment.

**Features:**
- PyTorch to ONNX conversion
- Model validation and verification
- Cross-platform inference
- Runtime optimization

**Usage:**
```bash
python onnx_optimization/onnx_example.py
```

**Benefits:**
- Deploy models across different frameworks and platforms
- Optimized inference engines (ONNX Runtime)
- Hardware-specific optimizations
- Consistent behavior across platforms

## Utilities

The `utils/` directory contains helper functions for:
- Parameter counting
- Model size calculation
- Inference benchmarking
- Model saving/loading
- Performance metrics

**Usage:**
```python
from utils.model_utils import count_parameters, get_model_size_mb, benchmark_model

# Count parameters
num_params = count_parameters(model)

# Get model size
size_mb = get_model_size_mb(model)

# Benchmark inference
stats = benchmark_model(model, input_shape=(3, 224, 224))
```

## Project Structure

```
Model-Optimization/
├── quantization/           # Quantization examples
│   └── quantization_example.py
├── pruning/               # Pruning examples
│   └── pruning_example.py
├── knowledge_distillation/ # Knowledge distillation examples
│   └── distillation_example.py
├── onnx_optimization/     # ONNX conversion examples
│   └── onnx_example.py
├── utils/                 # Utility functions
│   └── model_utils.py
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

## Best Practices

1. **Start with quantization** - Easiest to implement with good results
2. **Combine techniques** - Use quantization + pruning for maximum compression
3. **Measure everything** - Always benchmark model size, speed, and accuracy
4. **Fine-tune after optimization** - Recover any accuracy lost during optimization
5. **Test on target hardware** - Optimization benefits vary by platform

## Performance Comparison

| Technique | Size Reduction | Speed Improvement | Accuracy Impact |
|-----------|----------------|-------------------|-----------------|
| Quantization | 2-4x | 1.5-3x | Minimal |
| Pruning | 2-10x | 1.2-2x | Low-Medium |
| Distillation | 5-10x | 3-5x | Low-Medium |
| ONNX | - | 1.2-2x | None |
| Combined | 10-50x | 3-10x | Medium |

## Contributing

Contributions are welcome! Feel free to:
- Add new optimization techniques
- Improve existing examples
- Add support for more model architectures
- Enhance documentation

## References

- [PyTorch Quantization](https://pytorch.org/docs/stable/quantization.html)
- [PyTorch Pruning](https://pytorch.org/tutorials/intermediate/pruning_tutorial.html)
- [Knowledge Distillation Paper](https://arxiv.org/abs/1503.02531)
- [ONNX](https://onnx.ai/)

## License

This project is provided as-is for educational purposes.
