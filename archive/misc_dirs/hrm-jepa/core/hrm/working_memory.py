"""HRM Working Memory - fast-adapting task-specific layer.

Working memory maintains task-specific context and adapts quickly
to new tasks. Uses lightweight adapters over JEPA latents.
"""


import torch
import torch.nn as nn


class LoRAAdapter(nn.Module):
    """Low-Rank Adaptation (LoRA) for efficient fine-tuning."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        rank: int = 8,
        alpha: float = 16.0,
    ) -> None:
        """Initialize LoRA adapter.

        Args:
            in_dim: Input dimension
            out_dim: Output dimension
            rank: Rank of low-rank decomposition
            alpha: Scaling factor
        """
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        # Low-rank matrices
        self.lora_A = nn.Parameter(torch.zeros(in_dim, rank))
        self.lora_B = nn.Parameter(torch.zeros(rank, out_dim))

        # Initialize
        nn.init.kaiming_uniform_(self.lora_A, a=5**0.5)
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply LoRA adaptation.

        Args:
            x: Input tensor (..., in_dim)

        Returns:
            Adapted tensor (..., out_dim)
        """
        # x @ A @ B with scaling
        return (x @ self.lora_A @ self.lora_B) * self.scaling


class WorkingMemory(nn.Module):
    """Working Memory module for HRM.

    Fast-adapting layer that maintains task-specific context over
    JEPA latents. Uses LoRA adapters for parameter efficiency.
    """

    def __init__(
        self,
        latent_dim: int = 512,
        hidden_dim: int = 1024,
        num_layers: int = 2,
        lora_rank: int = 8,
        dropout: float = 0.1,
    ) -> None:
        """Initialize Working Memory.

        Args:
            latent_dim: Dimension of JEPA latents
            hidden_dim: Hidden dimension for reasoning
            num_layers: Number of processing layers
            lora_rank: Rank for LoRA adapters
            dropout: Dropout probability
        """
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim

        # Input projection (frozen base + LoRA adapter)
        self.input_proj_base = nn.Linear(latent_dim, hidden_dim)
        self.input_proj_lora = LoRAAdapter(latent_dim, hidden_dim, lora_rank)

        # Processing layers with LoRA
        self.layers = nn.ModuleList()
        self.lora_adapters = nn.ModuleList()

        for _ in range(num_layers):
            # Base layer (can be frozen during fast adaptation)
            layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=8,
                dim_feedforward=hidden_dim * 4,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.layers.append(layer)

            # LoRA adapter for this layer
            adapter = LoRAAdapter(hidden_dim, hidden_dim, lora_rank)
            self.lora_adapters.append(adapter)

        # Output projection
        self.output_proj = nn.Linear(hidden_dim, latent_dim)

        # Task state (episodic memory)
        self.register_buffer("task_state", torch.zeros(1, hidden_dim))
        self.register_buffer("step_count", torch.zeros(1, dtype=torch.long))

    def freeze_base(self) -> None:
        """Freeze base parameters (only train LoRA adapters)."""
        self.input_proj_base.requires_grad_(False)
        for layer in self.layers:
            layer.requires_grad_(False)
        # Keep LoRA and output trainable
        for adapter in self.lora_adapters:
            adapter.requires_grad_(True)
        self.output_proj.requires_grad_(True)

    def unfreeze_all(self) -> None:
        """Unfreeze all parameters."""
        self.requires_grad_(True)

    def reset_task_state(self) -> None:
        """Reset task state (start new task)."""
        self.task_state.zero_()
        self.step_count.zero_()

    def forward(
        self,
        latents: torch.Tensor,
        update_state: bool = True,
    ) -> torch.Tensor:
        """Process latents through working memory.

        Args:
            latents: JEPA latents (batch, latent_dim) or (batch, seq, latent_dim)
            update_state: Whether to update task state

        Returns:
            Processed latents (same shape as input)
        """
        input_shape = latents.shape
        if latents.dim() == 2:
            latents = latents.unsqueeze(1)  # Add sequence dim

        batch_size, seq_len, _ = latents.shape

        # Project to hidden dim (base + LoRA)
        x = self.input_proj_base(latents)
        x = x + self.input_proj_lora(latents)

        # Add task state as context
        task_state_expanded = self.task_state.expand(batch_size, -1, -1)
        x = torch.cat([task_state_expanded, x], dim=1)  # Prepend task state

        # Process through layers with LoRA
        for layer, adapter in zip(self.layers, self.lora_adapters):
            x_base = layer(x)
            x_adapted = adapter(x)
            x = x_base + x_adapted

        # Extract task state and sequence
        new_task_state = x[:, 0:1, :]  # First token is updated task state
        x = x[:, 1:, :]  # Rest is processed sequence

        # Update task state if requested
        if update_state:
            # Exponential moving average
            self.task_state = 0.9 * self.task_state + 0.1 * new_task_state.mean(0)
            self.step_count += 1

        # Project back to latent dim
        x = self.output_proj(x)

        # Restore original shape
        if len(input_shape) == 2:
            x = x.squeeze(1)

        return x

    def get_task_context(self) -> dict[str, torch.Tensor]:
        """Get current task context.

        Returns:
            Dictionary with task state and metadata
        """
        return {
            "task_state": self.task_state.clone(),
            "step_count": self.step_count.clone(),
        }

    def set_task_context(self, context: dict[str, torch.Tensor]) -> None:
        """Set task context (for loading saved state).

        Args:
            context: Dictionary with task state
        """
        if "task_state" in context:
            self.task_state.copy_(context["task_state"])
        if "step_count" in context:
            self.step_count.copy_(context["step_count"])
