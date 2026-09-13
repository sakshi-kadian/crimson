import os
import torch
import torch.nn as nn
import hydra
from omegaconf import DictConfig, OmegaConf

from src.core.ddp_setup import DDPSetup
from src.data.dataloader import get_dataloaders
from src.model.builder import build_model
from src.utils.logger import TensorBoardLogger
from src.core.trainer import Trainer

@hydra.main(version_base="1.3", config_path="configs", config_name="config")
def main(cfg: DictConfig):
    print(f"Loaded config:\n{OmegaConf.to_yaml(cfg)}")
    
    if "seed" in cfg:
        torch.manual_seed(cfg.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(cfg.seed)
    
    is_ddp = cfg.hardware.mode == "ddp"
    if is_ddp:
        ddp_setup = DDPSetup(backend_override=cfg.hardware.backend)
        ddp_setup.setup()
        
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    
    # Dataloaders
    train_loader, val_loader, train_sampler = get_dataloaders(cfg, rank=rank, world_size=world_size)
    
    # Device determination
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{rank}" if is_ddp else "cuda:0")
    else:
        device = torch.device("cpu")
        
    # Model construction
    model = build_model(cfg, rank=rank, world_size=world_size)
    
    # Linear Scaling Rule
    accumulation_steps = getattr(cfg.hardware, "accumulation_steps", 1)
    effective_batch_multiplier = world_size * accumulation_steps
    actual_lr = cfg.model.learning_rate * effective_batch_multiplier
    
    # Optimizer & Criterion
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=actual_lr,
        momentum=0.9,
        weight_decay=cfg.model.weight_decay,
    )
    
    # LR Scheduler
    warmup_epochs = getattr(cfg.model, "warmup_epochs", 0)
    total_epochs = getattr(cfg.model, "epochs", 10)
    
    if warmup_epochs > 0:
        warmup = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_epochs)
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_epochs - warmup_epochs)
        scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs])
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_epochs)

    # Fault Tolerance: Resume from checkpoint if it exists
    start_epoch = 0
    best_acc = 0.0
    checkpoint_path = os.path.join(cfg.output_dir, "best_model.pth") if hasattr(cfg, "output_dir") else "outputs/best_model.pth"
    if os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
        if hasattr(model, "module"):
            model.module.load_state_dict(ckpt["model_state_dict"])
        else:
            model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "scheduler_state_dict" in ckpt and ckpt["scheduler_state_dict"] is not None:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        best_acc = ckpt.get("val_acc", 0.0)
        if rank == 0:
            print(f"[Rank {rank}] Resumed from checkpoint: Epoch {ckpt['epoch']} (Val Acc: {ckpt.get('val_acc', 0):.2f}%)")
    
    criterion = nn.CrossEntropyLoss()
    
    # Logger
    tb_dir = getattr(cfg, "logging", {}).get("tensorboard_dir", "logs/tensorboard") if hasattr(cfg, "logging") else "logs/tensorboard"
    logger = TensorBoardLogger(tb_dir, rank=rank)
    
    # Trainer initialization
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        rank=rank,
        logger=logger,
        cfg=cfg,
        train_sampler=train_sampler,
        scheduler=scheduler,
        start_epoch=start_epoch,
        best_acc=best_acc,
    )
    
    # Run training loop
    trainer.train()
    logger.close()
    
    if is_ddp:
        ddp_setup.cleanup()

if __name__ == "__main__":
    main()
