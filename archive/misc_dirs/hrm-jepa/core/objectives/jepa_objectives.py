"""Training objectives for JEPA.

Implements predictive learning objectives that train the model to predict
masked/future latents without reconstructing pixels or tokens.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastiveLoss(nn.Module):
    """Contrastive loss for aligning vision and text latents."""

    def __init__(self, temperature: float = 0.07) -> None:
        """Initialize contrastive loss.

        Args:
            temperature: Temperature parameter for softmax
        """
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        vision_latents: torch.Tensor,
        text_latents: torch.Tensor,
    ) -> torch.Tensor:
        """Compute bidirectional contrastive loss.

        Args:
            vision_latents: Vision embeddings (batch, latent_dim)
            text_latents: Text embeddings (batch, latent_dim)

        Returns:
            Scalar loss value
        """
        batch_size = vision_latents.shape[0]

        # Normalize embeddings
        vision_latents = F.normalize(vision_latents, dim=-1)
        text_latents = F.normalize(text_latents, dim=-1)

        # Compute similarity matrix
        logits = torch.matmul(vision_latents, text_latents.T) / self.temperature

        # Labels: diagonal elements are positive pairs
        labels = torch.arange(batch_size, device=logits.device)

        # Loss in both directions
        loss_v2t = F.cross_entropy(logits, labels)
        loss_t2v = F.cross_entropy(logits.T, labels)

        return (loss_v2t + loss_t2v) / 2


class LatentPredictionLoss(nn.Module):
    """Loss for predicting masked latents (JEPA objective)."""

    def __init__(self, loss_type: str = "mse") -> None:
        """Initialize latent prediction loss.

        Args:
            loss_type: Type of loss ('mse' or 'cosine')
        """
        super().__init__()
        self.loss_type = loss_type

    def forward(
        self,
        predicted: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute prediction loss.

        Args:
            predicted: Predicted latents (batch, seq_len, dim) or (batch, dim)
            target: Target latents (same shape as predicted)
            mask: Optional mask indicating which positions to predict

        Returns:
            Scalar loss value
        """
        if self.loss_type == "mse":
            loss = F.mse_loss(predicted, target, reduction="none")

            if mask is not None:
                # Apply mask and average only over masked positions
                if loss.dim() == 3:  # (batch, seq_len, dim)
                    mask = mask.unsqueeze(-1)  # (batch, seq_len, 1)
                loss = (loss * mask).sum() / mask.sum().clamp(min=1.0)
            else:
                loss = loss.mean()

        elif self.loss_type == "cosine":
            # Normalize
            predicted_norm = F.normalize(predicted, dim=-1)
            target_norm = F.normalize(target, dim=-1)

            # Cosine similarity
            similarity = (predicted_norm * target_norm).sum(dim=-1)

            # Loss is 1 - similarity
            loss = 1 - similarity

            if mask is not None:
                loss = (loss * mask).sum() / mask.sum().clamp(min=1.0)
            else:
                loss = loss.mean()

        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")

        return loss


class JEPAObjective(nn.Module):
    """Combined JEPA training objective.

    Includes:
    1. Contrastive alignment between vision and text
    2. Masked latent prediction
    """

    def __init__(
        self,
        temperature: float = 0.07,
        prediction_loss_type: str = "mse",
        contrastive_weight: float = 1.0,
        prediction_weight: float = 1.0,
    ) -> None:
        """Initialize JEPA objective.

        Args:
            temperature: Temperature for contrastive loss
            prediction_loss_type: Type of prediction loss ('mse' or 'cosine')
            contrastive_weight: Weight for contrastive loss
            prediction_weight: Weight for prediction loss
        """
        super().__init__()
        self.contrastive_loss = ContrastiveLoss(temperature)
        self.prediction_loss = LatentPredictionLoss(prediction_loss_type)
        self.contrastive_weight = contrastive_weight
        self.prediction_weight = prediction_weight

    def forward(
        self,
        vision_latents: torch.Tensor | None = None,
        text_latents: torch.Tensor | None = None,
        predicted_latents: torch.Tensor | None = None,
        target_latents: torch.Tensor | None = None,
        prediction_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Compute JEPA losses.

        Args:
            vision_latents: Encoded vision (batch, latent_dim)
            text_latents: Encoded text (batch, latent_dim)
            predicted_latents: Predicted masked latents
            target_latents: Target latents for prediction
            prediction_mask: Mask for prediction loss

        Returns:
            Dictionary with loss components
        """
        losses = {}
        total_loss = 0.0

        # Contrastive alignment loss
        if vision_latents is not None and text_latents is not None:
            contrastive = self.contrastive_loss(vision_latents, text_latents)
            losses["contrastive"] = contrastive
            total_loss += self.contrastive_weight * contrastive

        # Masked prediction loss
        if predicted_latents is not None and target_latents is not None:
            prediction = self.prediction_loss(
                predicted_latents,
                target_latents,
                prediction_mask,
            )
            losses["prediction"] = prediction
            total_loss += self.prediction_weight * prediction

        losses["total"] = total_loss
        return losses


def create_mask_random(
    batch_size: int,
    seq_len: int,
    mask_ratio: float = 0.15,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Create random mask for latent prediction.

    Args:
        batch_size: Batch size
        seq_len: Sequence length
        mask_ratio: Fraction of tokens to mask
        device: Device to create tensor on

    Returns:
        Binary mask (batch, seq_len) where 1 = masked
    """
    num_masked = int(seq_len * mask_ratio)

    mask = torch.zeros(batch_size, seq_len, device=device)

    for i in range(batch_size):
        # Random indices to mask
        indices = torch.randperm(seq_len, device=device)[:num_masked]
        mask[i, indices] = 1.0

    return mask


def create_mask_block(
    batch_size: int,
    seq_len: int,
    block_size: int = 4,
    num_blocks: int = 3,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Create block-wise mask (for vision patches).

    Args:
        batch_size: Batch size
        seq_len: Sequence length
        block_size: Size of each masked block
        num_blocks: Number of blocks to mask
        device: Device to create tensor on

    Returns:
        Binary mask (batch, seq_len) where 1 = masked
    """
    mask = torch.zeros(batch_size, seq_len, device=device)

    for i in range(batch_size):
        for _ in range(num_blocks):
            # Random starting position
            start = torch.randint(0, max(1, seq_len - block_size + 1), (1,)).item()
            end = min(start + block_size, seq_len)
            mask[i, start:end] = 1.0

    return mask
