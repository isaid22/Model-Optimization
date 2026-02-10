"""
Model Pruning Example
This module demonstrates magnitude-based pruning and structured pruning techniques.
"""

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import copy


class ConvNet(nn.Module):
    """Convolutional neural network for demonstration."""
    
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(2)
    
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.maxpool(x)
        x = self.relu(self.conv2(x))
        x = self.maxpool(x)
        x = x.view(-1, 9216)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def magnitude_pruning(model, amount=0.3):
    """
    Apply L1 unstructured magnitude-based pruning.
    
    Args:
        model: PyTorch model to prune
        amount: Fraction of parameters to prune (0.0 to 1.0)
        
    Returns:
        Pruned model
    """
    # Apply pruning to convolutional and linear layers
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            prune.l1_unstructured(module, name='weight', amount=amount)
    
    return model


def structured_pruning(model, amount=0.3):
    """
    Apply structured pruning (removes entire filters/channels).
    
    Args:
        model: PyTorch model to prune
        amount: Fraction of structures to prune (0.0 to 1.0)
        
    Returns:
        Pruned model
    """
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            # Prune entire filters (structured pruning on dimension 0)
            prune.ln_structured(
                module, name='weight', amount=amount, n=2, dim=0
            )
        elif isinstance(module, nn.Linear):
            # Prune entire output neurons
            prune.ln_structured(
                module, name='weight', amount=amount, n=2, dim=0
            )
    
    return model


def remove_pruning_reparametrization(model):
    """
    Make pruning permanent by removing reparametrization.
    
    Args:
        model: Pruned model with reparametrization
        
    Returns:
        Model with permanent pruning
    """
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            try:
                prune.remove(module, 'weight')
            except (ValueError, AttributeError):
                pass
    return model


def calculate_sparsity(model):
    """
    Calculate the sparsity of the model.
    
    Args:
        model: PyTorch model
        
    Returns:
        Sparsity percentage
    """
    total_params = 0
    zero_params = 0
    
    for name, param in model.named_parameters():
        if 'weight' in name:
            total_params += param.numel()
            zero_params += (param == 0).sum().item()
    
    sparsity = 100.0 * zero_params / total_params
    return sparsity


def count_parameters(model):
    """
    Count total and non-zero parameters in model.
    
    Args:
        model: PyTorch model
        
    Returns:
        Tuple of (total_params, non_zero_params)
    """
    total_params = sum(p.numel() for p in model.parameters())
    non_zero_params = sum((p != 0).sum().item() for p in model.parameters())
    return total_params, non_zero_params


if __name__ == "__main__":
    print("Model Pruning Example")
    print("=" * 50)
    
    # Create original model
    model = ConvNet()
    total, non_zero = count_parameters(model)
    print(f"Original model: {total} total parameters, {non_zero} non-zero")
    print(f"Original sparsity: {calculate_sparsity(model):.2f}%")
    
    # Magnitude-based pruning
    print("\n1. Magnitude-based Pruning (30%)")
    print("-" * 50)
    model_mag_pruned = copy.deepcopy(model)
    model_mag_pruned = magnitude_pruning(model_mag_pruned, amount=0.3)
    
    total, non_zero = count_parameters(model_mag_pruned)
    print(f"After pruning: {total} total parameters, {non_zero} non-zero")
    print(f"Sparsity: {calculate_sparsity(model_mag_pruned):.2f}%")
    
    # Structured pruning
    print("\n2. Structured Pruning (20%)")
    print("-" * 50)
    model_struct_pruned = copy.deepcopy(model)
    model_struct_pruned = structured_pruning(model_struct_pruned, amount=0.2)
    
    total, non_zero = count_parameters(model_struct_pruned)
    print(f"After pruning: {total} total parameters, {non_zero} non-zero")
    print(f"Sparsity: {calculate_sparsity(model_struct_pruned):.2f}%")
    
    # Make pruning permanent
    print("\n3. Making Pruning Permanent")
    print("-" * 50)
    model_mag_pruned = remove_pruning_reparametrization(model_mag_pruned)
    print("Pruning masks removed, changes are now permanent")
    
    print("\nPruning completed successfully!")
