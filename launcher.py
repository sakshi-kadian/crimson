import os
import hydra
from omegaconf import DictConfig, OmegaConf
from src.core.ddp_setup import DDPSetup

@hydra.main(version_base="1.3", config_path="configs", config_name="config")
def main(cfg: DictConfig):
    print(f"Loaded config: \n{OmegaConf.to_yaml(cfg)}")
    
    # Check if running in DDP mode based on hardware config
    is_ddp = cfg.hardware.mode == "ddp"
    
    if is_ddp:
        ddp_setup = DDPSetup(backend_override=cfg.hardware.backend)
        ddp_setup.setup()
    
    # Manual smoke test validation
    rank = int(os.environ.get("RANK", 0))
    print(f"Smoke Test: Running on rank {rank}!")
    
    if is_ddp:
        ddp_setup.cleanup()

if __name__ == "__main__":
    main()
