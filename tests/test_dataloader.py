import pytest
from omegaconf import OmegaConf
from src.data.dataloader import get_dataloaders

def test_dataloader_shapes():
    # Create minimal config
    cfg = OmegaConf.create({
        "dataset": {
            "batch_size": 4,
            "num_workers": 0,
            "pin_memory": False,
            "data_dir": "./data"
        },
        "hardware": {
            "mode": "single"
        }
    })
    
    train_loader, val_loader, train_sampler = get_dataloaders(cfg, rank=0, world_size=1)
    
    # Grab a single batch
    images, labels = next(iter(train_loader))
    
    # Check shapes
    assert images.shape == (4, 3, 32, 32), f"Expected image shape (4, 3, 32, 32), got {images.shape}"
    assert labels.shape == (4,), f"Expected labels shape (4,), got {labels.shape}"
    
    # Check sampler
    assert train_sampler is None, "Sampler should be None for single-GPU mode"
