"""
Knowledge Distillation Example
This module demonstrates how to transfer knowledge from a large teacher model
to a smaller student model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class TeacherModel(nn.Module):
    """Large teacher model with more capacity."""
    
    def __init__(self):
        super(TeacherModel, self).__init__()
        self.fc1 = nn.Linear(784, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 10)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x):
        x = x.view(-1, 784)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.relu(self.fc3(x))
        x = self.fc4(x)
        return x


class StudentModel(nn.Module):
    """Smaller student model to be trained via distillation."""
    
    def __init__(self):
        super(StudentModel, self).__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = x.view(-1, 784)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class DistillationLoss(nn.Module):
    """
    Combined loss for knowledge distillation.
    Combines hard target loss and soft target loss from teacher.
    """
    
    def __init__(self, temperature=3.0, alpha=0.7):
        """
        Args:
            temperature: Temperature for softening probability distributions
            alpha: Weight for distillation loss (1-alpha for hard target loss)
        """
        super(DistillationLoss, self).__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.ce_loss = nn.CrossEntropyLoss()
    
    def forward(self, student_logits, teacher_logits, targets):
        """
        Calculate distillation loss.
        
        Args:
            student_logits: Output logits from student model
            teacher_logits: Output logits from teacher model
            targets: True labels
            
        Returns:
            Combined loss
        """
        # Soft target loss (distillation loss)
        soft_targets = F.softmax(teacher_logits / self.temperature, dim=1)
        soft_student = F.log_softmax(student_logits / self.temperature, dim=1)
        distillation_loss = F.kl_div(
            soft_student, soft_targets, reduction='batchmean'
        ) * (self.temperature ** 2)
        
        # Hard target loss (standard cross-entropy)
        student_loss = self.ce_loss(student_logits, targets)
        
        # Combined loss
        total_loss = (
            self.alpha * distillation_loss + 
            (1 - self.alpha) * student_loss
        )
        
        return total_loss


def train_student(
    student_model,
    teacher_model,
    train_loader,
    num_epochs=5,
    temperature=3.0,
    alpha=0.7,
    lr=0.001
):
    """
    Train student model using knowledge distillation.
    
    Args:
        student_model: Student model to train
        teacher_model: Pre-trained teacher model
        train_loader: DataLoader for training data
        num_epochs: Number of training epochs
        temperature: Temperature for distillation
        alpha: Weight for distillation loss
        lr: Learning rate
        
    Returns:
        Trained student model
    """
    teacher_model.eval()  # Teacher is in evaluation mode
    student_model.train()
    
    criterion = DistillationLoss(temperature=temperature, alpha=alpha)
    optimizer = optim.Adam(student_model.parameters(), lr=lr)
    
    for epoch in range(num_epochs):
        total_loss = 0.0
        for batch_idx, (data, targets) in enumerate(train_loader):
            optimizer.zero_grad()
            
            # Get predictions from both models
            with torch.no_grad():
                teacher_logits = teacher_model(data)
            student_logits = student_model(data)
            
            # Calculate loss and update
            loss = criterion(student_logits, teacher_logits, targets)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")
    
    return student_model


def compare_models(teacher_model, student_model):
    """
    Compare parameter counts of teacher and student models.
    
    Args:
        teacher_model: Teacher model
        student_model: Student model
        
    Returns:
        Dictionary with comparison metrics
    """
    teacher_params = sum(p.numel() for p in teacher_model.parameters())
    student_params = sum(p.numel() for p in student_model.parameters())
    
    compression_ratio = teacher_params / student_params
    
    return {
        'teacher_params': teacher_params,
        'student_params': student_params,
        'compression_ratio': compression_ratio,
        'parameter_reduction': ((teacher_params - student_params) / teacher_params) * 100
    }


if __name__ == "__main__":
    print("Knowledge Distillation Example")
    print("=" * 50)
    
    # Create teacher and student models
    teacher = TeacherModel()
    student = StudentModel()
    
    # Compare model sizes
    comparison = compare_models(teacher, student)
    print(f"Teacher parameters: {comparison['teacher_params']:,}")
    print(f"Student parameters: {comparison['student_params']:,}")
    print(f"Compression ratio: {comparison['compression_ratio']:.2f}x")
    print(f"Parameter reduction: {comparison['parameter_reduction']:.2f}%")
    
    # Create dummy data for demonstration
    print("\nCreating dummy training data...")
    dummy_data = torch.randn(100, 28, 28)
    dummy_targets = torch.randint(0, 10, (100,))
    
    # Create simple dataset and dataloader
    from torch.utils.data import TensorDataset, DataLoader
    dataset = TensorDataset(dummy_data, dummy_targets)
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    print("\nTraining student model via knowledge distillation...")
    print("-" * 50)
    trained_student = train_student(
        student, teacher, train_loader, 
        num_epochs=3, temperature=3.0, alpha=0.7
    )
    
    print("\nKnowledge distillation completed successfully!")
    print("The student model has learned from the teacher model.")
