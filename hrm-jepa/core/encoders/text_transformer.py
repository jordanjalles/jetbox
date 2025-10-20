"""Text Transformer encoder for JEPA architecture.

This module implements a transformer-based text encoder that converts
text sequences into latent representations for the joint embedding space.
"""


import torch
import torch.nn as nn


class TextTransformer(nn.Module):
    """Transformer encoder for text in JEPA.

    Encodes text sequences into a latent representation suitable for
    joint embedding with vision.
    """

    def __init__(
        self,
        vocab_size: int = 30000,
        max_seq_len: int = 512,
        embed_dim: int = 512,
        depth: int = 6,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        latent_dim: int = 512,
    ) -> None:
        """Initialize Text Transformer.

        Args:
            vocab_size: Size of vocabulary
            max_seq_len: Maximum sequence length
            embed_dim: Transformer embedding dimension
            depth: Number of transformer blocks
            num_heads: Number of attention heads
            mlp_ratio: MLP expansion ratio
            dropout: Dropout probability
            latent_dim: Final latent dimension for JEPA
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len

        # Token embeddings
        self.token_embed = nn.Embedding(vocab_size, embed_dim)

        # Position embeddings
        self.pos_embed = nn.Parameter(torch.zeros(1, max_seq_len, embed_dim))

        self.dropout = nn.Dropout(dropout)

        # Transformer blocks
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=depth,
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
        nn.init.trunc_normal_(self.token_embed.weight, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Encode text to latent representation.

        Args:
            input_ids: Token IDs of shape (batch, seq_len)
            attention_mask: Attention mask of shape (batch, seq_len)
                          1 for real tokens, 0 for padding

        Returns:
            Latent vectors of shape (batch, latent_dim)
        """
        batch_size, seq_len = input_ids.shape

        # Token embeddings
        x = self.token_embed(input_ids)  # (B, seq_len, embed_dim)

        # Add positional embeddings
        x = x + self.pos_embed[:, :seq_len, :]
        x = self.dropout(x)

        # Create attention mask for transformer (inverted)
        if attention_mask is not None:
            # Convert to transformer format: True = ignore, False = attend
            mask = (attention_mask == 0).bool()
        else:
            mask = None

        # Apply transformer
        x = self.transformer(x, src_key_padding_mask=mask)
        x = self.norm(x)

        # Mean pooling over sequence (excluding padding)
        if attention_mask is not None:
            # Expand mask for broadcasting
            mask_expanded = attention_mask.unsqueeze(-1).float()
            # Sum embeddings and divide by sequence length
            sum_embeddings = (x * mask_expanded).sum(dim=1)
            sum_mask = mask_expanded.sum(dim=1).clamp(min=1.0)
            pooled = sum_embeddings / sum_mask
        else:
            # Simple mean if no mask provided
            pooled = x.mean(dim=1)  # (B, embed_dim)

        # Project to JEPA latent space
        latent = self.latent_proj(pooled)  # (B, latent_dim)

        return latent

    def get_token_embeddings(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Get token-level embeddings (for JEPA masking).

        Args:
            input_ids: Token IDs of shape (batch, seq_len)
            attention_mask: Attention mask of shape (batch, seq_len)

        Returns:
            Token embeddings of shape (batch, seq_len, embed_dim)
        """
        batch_size, seq_len = input_ids.shape

        # Token embeddings
        x = self.token_embed(input_ids)

        # Add positional embeddings
        x = x + self.pos_embed[:, :seq_len, :]
        x = self.dropout(x)

        # Create attention mask
        if attention_mask is not None:
            mask = (attention_mask == 0).bool()
        else:
            mask = None

        # Apply transformer
        x = self.transformer(x, src_key_padding_mask=mask)
        x = self.norm(x)

        return x  # (B, seq_len, embed_dim)


class SimpleTokenizer:
    """Simple character-level tokenizer for initial testing.

    In production, replace with BPE or WordPiece tokenizer.
    """

    def __init__(self, vocab_size: int = 30000) -> None:
        """Initialize tokenizer.

        Args:
            vocab_size: Maximum vocabulary size
        """
        self.vocab_size = vocab_size
        self.char_to_id = {"<PAD>": 0, "<UNK>": 1, "<CLS>": 2, "<SEP>": 3}
        self.id_to_char = {v: k for k, v in self.char_to_id.items()}
        self.next_id = 4

    def encode(self, text: str, max_length: int = 512) -> tuple[list[int], list[int]]:
        """Encode text to token IDs.

        Args:
            text: Input text
            max_length: Maximum sequence length

        Returns:
            Tuple of (token_ids, attention_mask)
        """
        # Add CLS token at start
        tokens = [self.char_to_id["<CLS>"]]
        attention = [1]

        # Encode characters
        for char in text[:max_length - 2]:  # Reserve space for CLS and SEP
            if char not in self.char_to_id:
                if self.next_id < self.vocab_size:
                    self.char_to_id[char] = self.next_id
                    self.id_to_char[self.next_id] = char
                    self.next_id += 1
                else:
                    # Use UNK if vocab full
                    char = "<UNK>"

            tokens.append(self.char_to_id.get(char, self.char_to_id["<UNK>"]))
            attention.append(1)

        # Add SEP token at end
        tokens.append(self.char_to_id["<SEP>"])
        attention.append(1)

        # Pad to max_length
        while len(tokens) < max_length:
            tokens.append(self.char_to_id["<PAD>"])
            attention.append(0)

        return tokens[:max_length], attention[:max_length]

    def decode(self, token_ids: list[int]) -> str:
        """Decode token IDs back to text.

        Args:
            token_ids: List of token IDs

        Returns:
            Decoded text
        """
        chars = []
        for token_id in token_ids:
            char = self.id_to_char.get(token_id, "<UNK>")
            if char not in ["<PAD>", "<CLS>", "<SEP>"]:
                chars.append(char)
        return "".join(chars)


def create_text_transformer_lite(latent_dim: int = 512) -> TextTransformer:
    """Create a lightweight text transformer for RTX 3090.

    Args:
        latent_dim: Dimension of JEPA latent space

    Returns:
        Lightweight text transformer model
    """
    return TextTransformer(
        vocab_size=30000,
        max_seq_len=256,  # Shorter for memory
        embed_dim=384,  # Match ViT-lite
        depth=6,
        num_heads=6,
        mlp_ratio=4.0,
        dropout=0.1,
        latent_dim=latent_dim,
    )
