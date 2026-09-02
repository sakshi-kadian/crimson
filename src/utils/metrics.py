import time
import torch

class ThroughputMeter:
    """
    Tracks samples processed per second.
    Useful for benchmarking hardware scaling efficiency.
    """
    def __init__(self):
        self.start_time = None
        self.total_samples = 0
        
    def start(self):
        self.start_time = time.time()
        self.total_samples = 0
        
    def update(self, batch_size):
        self.total_samples += batch_size
        
    def get_throughput(self):
        if self.start_time is None:
            return 0.0
        elapsed = time.time() - self.start_time
        if elapsed == 0:
            return 0.0
        return self.total_samples / elapsed


class GPUUtilizationTracker:
    """
    Queries NVML to track the physical GPU utilization percentage.
    Proves that the dataloader or network is not starving the GPU.
    Falls back to 0% gracefully on CPU.
    """
    def __init__(self):
        self.total_utilization = 0.0
        self.samples = 0
        self.has_cuda = torch.cuda.is_available()
        
    def sample(self, device_id=None):
        if not self.has_cuda:
            return 0.0
            
        try:
            # Requires PyTorch 2.0+
            # Returns an int (0-100) representing GPU utilization %
            util = torch.cuda.utilization(device_id)
            self.total_utilization += util
            self.samples += 1
            return util
        except Exception:
            return 0.0
            
    def get_average_utilization(self):
        if self.samples == 0:
            return 0.0
        return self.total_utilization / self.samples
        
    def reset(self):
        self.total_utilization = 0.0
        self.samples = 0
