import torch
import pytest
from omegaconf import OmegaConf
from src.model.builder import build_model

def test_resnet50_cifar_stem_output_shape():
    # Create dummy config
    cfg = OmegaConf.create({
        "model": {"num_classes": 100},
        "hardware": {"mode": "single"}
    })
    
    # Build model
    model = build_model(cfg)
    model.eval()
    
    # Dummy CIFAR-100 batch (Batch size 2, 3 channels, 32x32 pixels)
    dummy_input = torch.randn(2, 3, 32, 32)
    
    with torch.no_grad():
        output = model(dummy_input)
        
    # Check output shape: (Batch size, Num classes)
    assert output.shape == (2, 100), f"Expected shape (2, 100), got {output.shape}"
    
    # Check that conv1 was modified for CIFAR
    assert model.conv1.kernel_size == (3, 3), "Conv1 kernel should be 3x3 for CIFAR-100"
    assert model.conv1.stride == (1, 1), "Conv1 stride should be 1 for CIFAR-100"
