import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler


# CIFAR-100 channel-wise mean and std (pre-computed over the full training set)
CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD  = (0.2675, 0.2565, 0.2761)


def get_dataloaders(cfg, rank=0, world_size=1):
    """
    Downloads CIFAR-100, applies transforms, and returns:
      - train_loader: uses DistributedSampler to shard data across GPUs
      - val_loader:   runs on all ranks but only rank 0 logs validation metrics
      - train_sampler: caller must call train_sampler.set_epoch(epoch) each epoch
                       to ensure correct reshuffling across DDP workers
    """
    # Training augmentation: standard CIFAR augmentation policy
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
    ])

    # Validation: only normalize, no augmentation
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
    ])

    # Download=True is a no-op if data already exists in data_dir
    train_dataset = torchvision.datasets.CIFAR100(
        root=cfg.dataset.data_dir,
        train=True,
        download=True,
        transform=train_transform,
    )

    val_dataset = torchvision.datasets.CIFAR100(
        root=cfg.dataset.data_dir,
        train=False,
        download=True,
        transform=val_transform,
    )

    is_distributed = cfg.hardware.mode == "ddp" and world_size > 1

    if is_distributed:
        # Shards the dataset across GPUs to prevent redundant computation
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
        )
        train_shuffle = False  # DistributedSampler handles shuffling
    else:
        train_sampler = None
        train_shuffle = True

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.dataset.batch_size,
        shuffle=train_shuffle,
        sampler=train_sampler,
        num_workers=cfg.dataset.num_workers,
        pin_memory=cfg.dataset.pin_memory,
        drop_last=True,  # Avoids uneven batch sizes in DDP
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.dataset.batch_size,
        shuffle=False,
        num_workers=cfg.dataset.num_workers,
        pin_memory=cfg.dataset.pin_memory,
    )

    return train_loader, val_loader, train_sampler
