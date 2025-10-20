"""Vision Transformer (ViT) encoder for JEPA architecture.

This module implements a lightweight Vision Transformer that encodes images
into latent representations for the joint embedding space.
"""


import torch
import torch.nn as nn


class PatchEmbedding(nn.Module):
    """Convert image into patches and embed them."""

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dim: int = 768,
    ) -> None:
        """Initialize patch embedding layer.

        Args:
            img_size: Input image size (assumes square)
            patch_size: Size of each patch
            in_channels: Number of input channels (3 for RGB)
            embed_dim: Dimension of patch embeddings
        """
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2

        # Convolutional projection of patches
        self.projection = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Convert image to patch embeddings.

        Args:
            x: Input tensor of shape (batch, channels, height, width)

        Returns:
            Patch embeddings of shape (batch, num_patches, embed_dim)
        """
        # x: (B, C, H, W) -> (B, embed_dim, H/P, W/P)
        x = self.projection(x)
        # (B, embed_dim, H/P, W/P) -> (B, embed_dim, num_patches)
        x = x.flatten(2)
        # (B, embed_dim, num_patches) -> (B, num_patches, embed_dim)
        x = x.transpose(1, 2)
        return x


class TransformerBlock(nn.Module):
    """Transformer encoder block with multi-head self-attention."""

    def __init__(
        self,
        embed_dim: int = 768,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
    ) -> None:
        """Initialize transformer block.

        Args:
            embed_dim: Embedding dimension
            num_heads: Number of attention heads
            mlp_ratio: Ratio of MLP hidden dim to embed_dim
            dropout: Dropout probability
        """
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(embed_dim)

        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply transformer block.

        Args:
            x: Input tensor of shape (batch, seq_len, embed_dim)

        Returns:
            Output tensor of same shape
        """
        # Self-attention with residual
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + attn_out

        # MLP with residual
        x = x + self.mlp(self.norm2(x))
        return x


class VisionViT(nn.Module):
    """Vision Transformer encoder for JEPA.

    Encodes images into a latent representation suitable for
    joint embedding with text.
    """

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        latent_dim: int = 512,
    ) -> None:
        """Initialize Vision Transformer.

        Args:
            img_size: Input image size
            patch_size: Patch size for embedding
            in_channels: Number of input channels
            embed_dim: Transformer embedding dimension
            depth: Number of transformer blocks
            num_heads: Number of attention heads
            mlp_ratio: MLP expansion ratio
            dropout: Dropout probability
            latent_dim: Final latent dimension for JEPA
        """
        super().__init__()
        self.patch_embed = PatchEmbedding(
            img_size, patch_size, in_channels, embed_dim
        )

        # CLS token for global representation
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # Position embeddings
        num_patches = self.patch_embed.num_patches
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, embed_dim)
        )

        self.dropout = nn.Dropout(dropout)

        # Transformer blocks
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
                for _ in range(depth)
            ]
        )

        self.norm = nn.LayerNorm(embed_dim)

        # Projection to JEPA latent space
        self.latent_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, latent_dim),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights."""
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode image to latent representation.

        Args:
            x: Input images of shape (batch, channels, height, width)

        Returns:
            Latent vectors of shape (batch, latent_dim)
        """
        batch_size = x.shape[0]

        # Patch embedding
        x = self.patch_embed(x)  # (B, num_patches, embed_dim)

        # Add CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # (B, num_patches+1, embed_dim)

        # Add positional encoding
        x = x + self.pos_embed
        x = self.dropout(x)

        # Apply transformer blocks
        for block in self.blocks:
            x = block(x)

        x = self.norm(x)

        # Use CLS token for global representation
        cls_output = x[:, 0]  # (B, embed_dim)

        # Project to JEPA latent space
        latent = self.latent_proj(cls_output)  # (B, latent_dim)

        return latent

    def get_patch_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        """Get patch-level embeddings (for JEPA masking).

        Args:
            x: Input images of shape (batch, channels, height, width)

        Returns:
            Patch embeddings of shape (batch, num_patches, embed_dim)
        """
        batch_size = x.shape[0]

        # Patch embedding
        x = self.patch_embed(x)

        # Add CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)

        # Add positional encoding
        x = x + self.pos_embed
        x = self.dropout(x)

        # Apply transformer blocks
        for block in self.blocks:
            x = block(x)

        x = self.norm(x)

        # Return all patch embeddings (excluding CLS)
        return x[:, 1:, :]  # (B, num_patches, embed_dim)


def create_vit_lite(latent_dim: int = 512) -> VisionViT:
    """Create a lightweight ViT for RTX 3090.

    Args:
        latent_dim: Dimension of JEPA latent space

    Returns:
        Lightweight ViT model
    """
    return VisionViT(
        img_size=224,
        patch_size=16,
        embed_dim=384,  # Smaller than ViT-Base
        depth=6,  # Fewer layers
        num_heads=6,
        mlp_ratio=4.0,
        dropout=0.1,
        latent_dim=latent_dim,
    )
