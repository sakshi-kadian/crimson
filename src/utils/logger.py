import os
from torch.utils.tensorboard import SummaryWriter


class TensorBoardLogger:
    """
    Thin wrapper around TensorBoard's SummaryWriter.

    Only rank 0 writes logs — all other ranks get a no-op logger.
    This prevents 2 (or N) processes from fighting over the same log file,
    which would corrupt the TensorBoard event stream.
    """

    def __init__(self, log_dir: str, rank: int):
        self.rank = rank
        self.writer = None

        if rank == 0:
            os.makedirs(log_dir, exist_ok=True)
            self.writer = SummaryWriter(log_dir=log_dir)
            print(f"[Rank 0] TensorBoard logging to: {log_dir}")

    def log_scalar(self, tag: str, value: float, step: int):
        """Log a single scalar value (e.g. loss, accuracy, throughput)."""
        if self.writer is not None:
            self.writer.add_scalar(tag, value, step)

    def log_gpu_utilization(self, util_pct: float, step: int):
        """
        Log GPU utilization percentage to TensorBoard.
        Proves GPUs are not stalled on CPU dataloaders or NCCL barriers.
        """
        if self.writer is not None:
            self.writer.add_scalar("hardware/gpu_utilization_pct", util_pct, step)

    def close(self):
        """Flush and close the SummaryWriter. Call this at the end of training."""
        if self.writer is not None:
            self.writer.flush()
            self.writer.close()
