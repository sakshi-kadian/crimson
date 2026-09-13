import os
import pytest
import torch.distributed as dist
from src.core.ddp_setup import DDPSetup

def test_ddp_setup_initialization():
    # Set up mock environment variables required by PyTorch DDP
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12345"
    os.environ["WORLD_SIZE"] = "1"
    os.environ["RANK"] = "0"
    
    # Initialize DDP using gloo backend (CPU compatible)
    ddp_setup = DDPSetup(backend_override="gloo")
    ddp_setup.setup()
    
    # Assert that PyTorch recognizes the distributed environment
    assert dist.is_initialized(), "DDP failed to initialize"
    
    # Clean up to prevent hanging or breaking other tests
    ddp_setup.cleanup()
    assert not dist.is_initialized(), "DDP failed to cleanup"
