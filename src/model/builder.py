import torch
import torch.nn as nn
from torchvision.models import resnet50
from torch.nn.parallel import DistributedDataParallel as DDP

def build_model(cfg, rank=None, world_size=None):
    """
    Builds the ResNet-50 model, modifies the fully connected layer for CIFAR-100,
    moves it to the correct device, and wraps it in DDP if required.
    """
    # Load ResNet-50 without pre-trained weights (training from scratch)
    model = resnet50(weights=None)
    
    # Replace final layer for CIFAR-100 (100 classes)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, cfg.model.num_classes)
    
    is_ddp = cfg.hardware.mode == "ddp"
    has_cuda = torch.cuda.is_available()
    
    if is_ddp and has_cuda and rank is not None:
        # Real distributed GPU environment (Kaggle)
        device = torch.device(f"cuda:{rank}")
        model = model.to(device)
        model = DDP(model, device_ids=[rank])
    elif is_ddp and not has_cuda:
        # Local CPU smoke-testing environment via gloo
        model = DDP(model)
    elif has_cuda:
        # Single GPU mode
        model = model.to("cuda:0")
        
    return model
