import os
from torch.utils.tensorboard import SummaryWriter

try:
    import wandb
except ImportError:
    wandb = None


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


class WandBLogger:
    """
    Weights & Biases logger with DDP-safe rank 0 isolation.

    W&B is the industry standard for tracking distributed ML experiments
    at production labs (Google DeepMind, Meta FAIR, OpenAI).

    Usage:
        python launcher.py logger=wandb

    Requires:
        pip install wandb
        wandb login  (run once to authenticate)
    """

    def __init__(self, project_name: str, cfg, rank: int):
        self.rank = rank
        self.run = None

        if rank == 0:
            try:
                import wandb
                self.run = wandb.init(
                    project=project_name,
                    config={
                        "learning_rate": cfg.model.learning_rate,
                        "epochs": cfg.model.epochs,
                        "batch_size": cfg.dataset.batch_size,
                        "architecture": cfg.model.name,
                        "num_classes": cfg.model.num_classes,
                        "hardware": cfg.hardware.mode,
                        "use_amp": cfg.hardware.use_amp,
                        "accumulation_steps": cfg.hardware.accumulation_steps,
                        "weight_decay": cfg.model.weight_decay,
                        "warmup_epochs": cfg.model.warmup_epochs,
                    },
                    reinit=True,
                )
                print(f"[Rank 0] W&B run initialized: {self.run.url}")
            except ImportError:
                print("[Rank 0] WARNING: wandb not installed. Run: pip install wandb")
                self.run = None

    def log_scalar(self, tag: str, value: float, step: int):
        """Log a scalar metric to W&B."""
        if self.run is not None:
            import wandb
            wandb.log({tag: value}, step=step)

    def log_gpu_utilization(self, util_pct: float, step: int):
        """Log GPU utilization to W&B."""
        if self.run is not None:
            import wandb
            wandb.log({"hardware/gpu_utilization_pct": util_pct}, step=step)

    def close(self):
        """Finish the W&B run cleanly."""
        if self.run is not None:
            import wandb
            wandb.finish()


def build_logger(cfg, rank: int):
    """
    Factory function: returns the correct logger based on cfg.logger.

    Config usage:
        logger: tensorboard   # default
        logger: wandb         # switch to W&B via CLI: python launcher.py logger=wandb
    """
    logger_type = getattr(cfg, "logger", "tensorboard")

    if logger_type == "wandb":
        return WandBLogger(
            project_name=getattr(cfg, "project_name", "crimson"),
            cfg=cfg,
            rank=rank,
        )
    else:
        return TensorBoardLogger(log_dir="logs/tensorboard", rank=rank)
