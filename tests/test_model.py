import torch
import torch.nn as nn
import pytest
from omegaconf import OmegaConf
from src.model.builder import build_model


def _make_cfg():
    return OmegaConf.create({
        "model": {"num_classes": 100},
        "hardware": {"mode": "single"}
    })


def test_gradients_flow_through_model():
    """Verifies that loss.backward() actually computes non-zero gradients."""
    model = build_model(_make_cfg())
    model.train()

    dummy_input = torch.randn(2, 3, 32, 32)
    dummy_labels = torch.randint(0, 100, (2,))
    criterion = nn.CrossEntropyLoss()

    output = model(dummy_input)
    loss = criterion(output, dummy_labels)
    loss.backward()

    # At least one parameter should have a gradient
    grad_norms = [
        p.grad.norm().item()
        for p in model.parameters()
        if p.grad is not None
    ]
    assert len(grad_norms) > 0, "No gradients were computed"
    assert any(g > 0 for g in grad_norms), "All gradients are zero — backward pass is broken"


def test_checkpoint_save_load_roundtrip(tmp_path):
    """Verifies that a saved checkpoint can be reloaded with identical weights."""
    cfg = _make_cfg()
    model = build_model(cfg)

    # Save a checkpoint
    checkpoint_path = tmp_path / "best_model.pth"
    torch.save({"model_state_dict": model.state_dict(), "epoch": 1, "val_acc": 55.0}, checkpoint_path)

    # Load it into a fresh model
    fresh_model = build_model(cfg)
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    fresh_model.load_state_dict(ckpt["model_state_dict"])

    # Every single weight must be identical
    for (name, p1), (_, p2) in zip(model.named_parameters(), fresh_model.named_parameters()):
        assert torch.allclose(p1, p2), f"Checkpoint mismatch in layer: {name}"

    assert ckpt["epoch"] == 1
    assert ckpt["val_acc"] == 55.0
