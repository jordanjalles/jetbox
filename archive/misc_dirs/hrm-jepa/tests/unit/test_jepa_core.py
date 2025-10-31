"""Tests for JEPA core and objectives."""

import torch

from core.jepa_core import create_jepa_lite
from core.objectives import (
    ContrastiveLoss,
    JEPAObjective,
    LatentPredictionLoss,
    create_mask_block,
    create_mask_random,
)


def test_jepa_core_forward() -> None:
    """Test JEPA core forward pass with both modalities."""
    model = create_jepa_lite(latent_dim=512)
    model.eval()

    # Create dummy inputs
    images = torch.randn(2, 3, 224, 224)
    input_ids = torch.randint(0, 1000, (2, 128))
    attention_mask = torch.ones(2, 128)

    with torch.no_grad():
        outputs = model(images=images, input_ids=input_ids, attention_mask=attention_mask)

    # Check outputs
    assert "vision_latent" in outputs
    assert "text_latent" in outputs
    assert outputs["vision_latent"].shape == (2, 512)
    assert outputs["text_latent"].shape == (2, 512)


def test_jepa_core_vision_only() -> None:
    """Test JEPA with vision only."""
    model = create_jepa_lite(latent_dim=512)
    model.eval()

    images = torch.randn(2, 3, 224, 224)

    with torch.no_grad():
        outputs = model(images=images)

    assert "vision_latent" in outputs
    assert "text_latent" not in outputs


def test_jepa_core_text_only() -> None:
    """Test JEPA with text only."""
    model = create_jepa_lite(latent_dim=512)
    model.eval()

    input_ids = torch.randint(0, 1000, (2, 128))
    attention_mask = torch.ones(2, 128)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)

    assert "text_latent" in outputs
    assert "vision_latent" not in outputs


def test_contrastive_loss() -> None:
    """Test contrastive loss computation."""
    loss_fn = ContrastiveLoss(temperature=0.07)

    # Create dummy embeddings
    vision = torch.randn(4, 512)
    text = torch.randn(4, 512)

    loss = loss_fn(vision, text)

    assert loss.item() > 0
    assert not torch.isnan(loss)


def test_latent_prediction_loss_mse() -> None:
    """Test MSE-based latent prediction loss."""
    loss_fn = LatentPredictionLoss(loss_type="mse")

    predicted = torch.randn(4, 512)
    target = torch.randn(4, 512)

    loss = loss_fn(predicted, target)

    assert loss.item() > 0
    assert not torch.isnan(loss)


def test_latent_prediction_loss_cosine() -> None:
    """Test cosine-based latent prediction loss."""
    loss_fn = LatentPredictionLoss(loss_type="cosine")

    predicted = torch.randn(4, 512)
    target = torch.randn(4, 512)

    loss = loss_fn(predicted, target)

    assert loss.item() >= 0
    assert not torch.isnan(loss)


def test_latent_prediction_loss_with_mask() -> None:
    """Test latent prediction loss with masking."""
    loss_fn = LatentPredictionLoss(loss_type="mse")

    predicted = torch.randn(4, 196, 512)  # Batch, seq, dim
    target = torch.randn(4, 196, 512)
    mask = torch.zeros(4, 196)
    mask[:, :50] = 1.0  # Only predict first 50 positions

    loss = loss_fn(predicted, target, mask)

    assert loss.item() > 0
    assert not torch.isnan(loss)


def test_jepa_objective() -> None:
    """Test combined JEPA objective."""
    objective = JEPAObjective(
        temperature=0.07,
        prediction_loss_type="mse",
        contrastive_weight=1.0,
        prediction_weight=1.0,
    )

    vision_latents = torch.randn(4, 512)
    text_latents = torch.randn(4, 512)
    predicted_latents = torch.randn(4, 196, 512)
    target_latents = torch.randn(4, 196, 512)
    mask = torch.ones(4, 196)

    losses = objective(
        vision_latents=vision_latents,
        text_latents=text_latents,
        predicted_latents=predicted_latents,
        target_latents=target_latents,
        prediction_mask=mask,
    )

    assert "contrastive" in losses
    assert "prediction" in losses
    assert "total" in losses
    assert losses["total"].item() > 0


def test_create_mask_random() -> None:
    """Test random masking."""
    mask = create_mask_random(batch_size=4, seq_len=196, mask_ratio=0.15)

    assert mask.shape == (4, 196)
    assert 0.10 < mask.mean().item() < 0.20  # Approximately 15%
    assert ((mask == 0) | (mask == 1)).all()  # Binary mask


def test_create_mask_block() -> None:
    """Test block-wise masking."""
    mask = create_mask_block(batch_size=4, seq_len=196, block_size=8, num_blocks=3)

    assert mask.shape == (4, 196)
    assert mask.sum() > 0  # At least some masked
    assert ((mask == 0) | (mask == 1)).all()  # Binary mask


def test_jepa_gradient_flow() -> None:
    """Test that gradients flow through JEPA."""
    model = create_jepa_lite(latent_dim=512)
    model.train()

    images = torch.randn(2, 3, 224, 224)
    input_ids = torch.randint(0, 1000, (2, 128))

    outputs = model(images=images, input_ids=input_ids)

    # Simple loss
    loss = outputs["vision_latent"].sum() + outputs["text_latent"].sum()
    loss.backward()

    # Check gradients
    assert model.vision_encoder.latent_proj[0].weight.grad is not None
    assert model.text_encoder.latent_proj[0].weight.grad is not None
