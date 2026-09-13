import os
import pytest
import torch.distributed as dist
from src.core.ddp_setup import DDPSetup

def test_ddp_setup_initialization():
    # Set up mock environment variables required by PyTorch DDP
    env_vars = {
        "MASTER_ADDR": "localhost",
        "MASTER_PORT": "12345",
        "WORLD_SIZE": "1",
        "RANK": "0",
    }
    original = {k: os.environ.get(k) for k in env_vars}
    
    try:
        for k, v in env_vars.items():
            os.environ[k] = v

        # Initialize DDP using gloo backend (CPU compatible)
        ddp_setup = DDPSetup(backend_override="gloo")
        ddp_setup.setup()

        # Assert that PyTorch recognizes the distributed environment
        assert dist.is_initialized(), "DDP failed to initialize"

        # Clean up to prevent hanging or breaking other tests
        ddp_setup.cleanup()
        assert not dist.is_initialized(), "DDP failed to cleanup"
    finally:
        # Always restore original env vars to prevent test pollution
        for k, original_val in original.items():
            if original_val is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = original_val
