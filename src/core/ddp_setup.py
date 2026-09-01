import os
import torch
import torch.distributed as dist

class DDPSetup:
    def __init__(self, backend_override=None):
        self.backend_override = backend_override

    def setup(self):
        """
        Initializes the distributed process group.
        Auto-detects the appropriate backend if not provided.
        """
        if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
            print("Not running in distributed mode. DDPSetup skipped.")
            return

        if self.backend_override:
            backend = self.backend_override
        else:
            backend = "nccl" if torch.cuda.is_available() else "gloo"
        
        # Initialize the process group
        dist.init_process_group(backend=backend)
        
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        
        print(f"[Rank {rank}/{world_size}] Process group initialized using {backend} backend.")

    def cleanup(self):
        """
        Destroys the distributed process group.
        """
        if dist.is_initialized():
            dist.destroy_process_group()
            print("Process group destroyed.")
