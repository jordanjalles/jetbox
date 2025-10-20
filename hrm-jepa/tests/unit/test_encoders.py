"""Tests for JEPA encoders (ViT and Text Transformer)."""

import torch

from core.encoders import (
    SimpleTokenizer,
    create_text_transformer_lite,
    create_vit_lite,
)


def test_vision_vit_forward() -> None:
    """Test Vision ViT forward pass."""
    model = create_vit_lite(latent_dim=512)
    model.eval()

    # Create dummy image batch
    batch_size = 2
    images = torch.randn(batch_size, 3, 224, 224)

    # Forward pass
    with torch.no_grad():
        latents = model(images)

    # Check output shape
    assert latents.shape == (batch_size, 512)
    assert not torch.isnan(latents).any()


def test_vision_vit_patch_embeddings() -> None:
    """Test ViT patch-level embeddings."""
    model = create_vit_lite(latent_dim=512)
    model.eval()

    images = torch.randn(2, 3, 224, 224)

    with torch.no_grad():
        patch_embeds = model.get_patch_embeddings(images)

    # 224/16 = 14, so 14x14 = 196 patches
    assert patch_embeds.shape == (2, 196, 384)  # embed_dim=384 for lite


def test_text_transformer_forward() -> None:
    """Test Text Transformer forward pass."""
    model = create_text_transformer_lite(latent_dim=512)
    model.eval()

    # Create dummy token IDs
    batch_size = 2
    seq_len = 128
    input_ids = torch.randint(0, 1000, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len)

    # Forward pass
    with torch.no_grad():
        latents = model(input_ids, attention_mask)

    # Check output shape
    assert latents.shape == (batch_size, 512)
    assert not torch.isnan(latents).any()


def test_text_transformer_with_padding() -> None:
    """Test Text Transformer handles padding correctly."""
    model = create_text_transformer_lite(latent_dim=512)
    model.eval()

    batch_size = 2
    seq_len = 128
    input_ids = torch.randint(0, 1000, (batch_size, seq_len))

    # Create attention mask with padding
    attention_mask = torch.ones(batch_size, seq_len)
    attention_mask[0, 64:] = 0  # First sample has padding after position 64
    attention_mask[1, 96:] = 0  # Second sample has padding after position 96

    with torch.no_grad():
        latents = model(input_ids, attention_mask)

    assert latents.shape == (batch_size, 512)
    assert not torch.isnan(latents).any()


def test_simple_tokenizer() -> None:
    """Test simple character-level tokenizer."""
    tokenizer = SimpleTokenizer(vocab_size=1000)

    text = "Hello, world!"
    tokens, attention = tokenizer.encode(text, max_length=64)

    # Check lengths
    assert len(tokens) == 64
    assert len(attention) == 64

    # Check special tokens
    assert tokens[0] == tokenizer.char_to_id["<CLS>"]
    assert tokens[len(text) + 1] == tokenizer.char_to_id["<SEP>"]

    # Check padding
    assert all(t == tokenizer.char_to_id["<PAD>"] for t in tokens[len(text) + 2:])
    assert all(a == 0 for a in attention[len(text) + 2:])

    # Test decode
    decoded = tokenizer.decode(tokens)
    assert "Hello, world!" in decoded


def test_text_transformer_gradient_flow() -> None:
    """Test that gradients flow through text transformer."""
    model = create_text_transformer_lite(latent_dim=512)
    model.train()

    input_ids = torch.randint(0, 1000, (2, 128))
    attention_mask = torch.ones(2, 128)

    latents = model(input_ids, attention_mask)
    loss = latents.sum()
    loss.backward()

    # Check gradients exist
    assert model.token_embed.weight.grad is not None
    assert model.latent_proj[0].weight.grad is not None


def test_vision_vit_gradient_flow() -> None:
    """Test that gradients flow through ViT."""
    model = create_vit_lite(latent_dim=512)
    model.train()

    images = torch.randn(2, 3, 224, 224)

    latents = model(images)
    loss = latents.sum()
    loss.backward()

    # Check gradients exist
    assert model.patch_embed.projection.weight.grad is not None
    assert model.latent_proj[0].weight.grad is not None
