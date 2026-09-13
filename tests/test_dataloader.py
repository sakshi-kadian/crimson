import pytest
from omegaconf import OmegaConf
from src.data.dataloader import get_dataloaders


def _make_cfg():
    return OmegaConf.create({
        "dataset": {
            "batch_size": 4,
            "num_workers": 0,
            "pin_memory": False,
            "data_dir": "./data"
        },
        "hardware": {"mode": "single"}
    })


def test_dataloader_batch_shapes():
    """Verifies train and val batches have correct image and label shapes."""
    train_loader, val_loader, _ = get_dataloaders(_make_cfg(), rank=0, world_size=1)

    images, labels = next(iter(train_loader))
    assert images.shape == (4, 3, 32, 32), f"Train image shape wrong: {images.shape}"
    assert labels.shape == (4,), f"Train label shape wrong: {labels.shape}"

    images, labels = next(iter(val_loader))
    assert images.shape == (4, 3, 32, 32), f"Val image shape wrong: {images.shape}"
    assert labels.shape == (4,), f"Val label shape wrong: {labels.shape}"


def test_set_epoch_produces_different_orderings():
    """Verifies that DistributedSampler.set_epoch() shuffles differently each epoch."""
    from torch.utils.data.distributed import DistributedSampler
    import torchvision
    import torchvision.transforms as transforms

    dataset = torchvision.datasets.CIFAR100(
        root="./data", train=True, download=True,
        transform=transforms.ToTensor()
    )

    sampler = DistributedSampler(dataset, num_replicas=2, rank=0, shuffle=True)

    sampler.set_epoch(0)
    order_epoch_0 = list(sampler)

    sampler.set_epoch(1)
    order_epoch_1 = list(sampler)

    assert order_epoch_0 != order_epoch_1, "set_epoch() produced identical data orderings — reshuffling is broken"
