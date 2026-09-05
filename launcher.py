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
    
    # Optimizer & Criterion
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=cfg.model.learning_rate,
        momentum=0.9,
        weight_decay=cfg.model.weight_decay,
    )
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
    )
    
    # Run training loop
    trainer.train()
    logger.close()
    
    if is_ddp:
        ddp_setup.cleanup()

if __name__ == "__main__":
    main()
