"""JEPA (Joint Embedding Predictive Architecture) core implementation.

This module implements the core JEPA architecture that learns unified
multimodal representations through predictive learning in latent space.
"""


import torch
import torch.nn as nn

from core.encoders import TextTransformer, VisionViT


class JEPACore(nn.Module):
    """Joint Embedding Predictive Architecture.

    Learns multimodal representations by predicting masked/future latents
    without reconstructing pixels or tokens.
    """

    def __init__(
        self,
        vision_encoder: VisionViT,
        text_encoder: TextTransformer,
        latent_dim: int = 512,
        predictor_depth: int = 4,
        predictor_heads: int = 8,
    ) -> None:
        """Initialize JEPA core.

        Args:
            vision_encoder: Vision encoder (ViT)
            text_encoder: Text encoder (Transformer)
            latent_dim: Dimension of joint latent space
            predictor_depth: Number of predictor transformer layers
            predictor_heads: Number of attention heads in predictor
        """
        super().__init__()
        self.vision_encoder = vision_encoder
        self.text_encoder = text_encoder
        self.latent_dim = latent_dim

        # Predictor: predicts target latents from context latents
        predictor_layer = nn.TransformerEncoderLayer(
            d_model=latent_dim,
            nhead=predictor_heads,
            dim_feedforward=latent_dim * 4,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.predictor = nn.TransformerEncoder(
            predictor_layer,
            num_layers=predictor_depth,
        )

        # Projection heads for alignment
        self.vision_proj = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, latent_dim),
        )

        self.text_proj = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, latent_dim),
        )

    def encode_vision(self, images: torch.Tensor) -> torch.Tensor:
        """Encode images to latent space.

        Args:
            images: Input images of shape (batch, channels, height, width)

        Returns:
            Vision latents of shape (batch, latent_dim)
        """
        z_vision = self.vision_encoder(images)
        z_vision = self.vision_proj(z_vision)
        # L2 normalize for contrastive learning
        z_vision = nn.functional.normalize(z_vision, dim=-1)
        return z_vision

    def encode_text(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Encode text to latent space.

        Args:
            input_ids: Token IDs of shape (batch, seq_len)
            attention_mask: Attention mask of shape (batch, seq_len)

        Returns:
            Text latents of shape (batch, latent_dim)
        """
        z_text = self.text_encoder(input_ids, attention_mask)
        z_text = self.text_proj(z_text)
        # L2 normalize for contrastive learning
        z_text = nn.functional.normalize(z_text, dim=-1)
        return z_text

    def predict_latent(
        self,
        context_latent: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Predict target latents from context.

        Args:
            context_latent: Context latents (batch, seq_len, latent_dim)
            mask: Optional mask for context tokens

        Returns:
            Predicted latents of same shape
        """
        # Add sequence dimension if needed
        if context_latent.dim() == 2:
            context_latent = context_latent.unsqueeze(1)

        # Apply predictor
        predicted = self.predictor(
            context_latent,
            src_key_padding_mask=mask,
        )

        return predicted

    def forward(
        self,
        images: torch.Tensor | None = None,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Forward pass for JEPA.

        Args:
            images: Input images (batch, channels, height, width)
            input_ids: Token IDs (batch, seq_len)
            attention_mask: Attention mask (batch, seq_len)

        Returns:
            Dictionary with encoded latents
        """
        outputs = {}

        if images is not None:
            outputs["vision_latent"] = self.encode_vision(images)

        if input_ids is not None:
            outputs["text_latent"] = self.encode_text(input_ids, attention_mask)

        return outputs


def create_jepa_lite(latent_dim: int = 512) -> JEPACore:
    """Create a lightweight JEPA model for RTX 3090.

    Args:
        latent_dim: Dimension of joint latent space

    Returns:
        Lightweight JEPA model
    """
    from core.encoders import create_text_transformer_lite, create_vit_lite

    vision_encoder = create_vit_lite(latent_dim=latent_dim)
    text_encoder = create_text_transformer_lite(latent_dim=latent_dim)

    return JEPACore(
        vision_encoder=vision_encoder,
        text_encoder=text_encoder,
        latent_dim=latent_dim,
        predictor_depth=4,
        predictor_heads=8,
    )
