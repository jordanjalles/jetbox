"""HRM Abstract Core - slow-updating long-term knowledge layer.

Abstract core maintains stable long-term reasoning patterns.
Updates only with explicit human approval after reflection.
"""

from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


class AbstractCore(nn.Module):
    """Abstract Core module for HRM.

    Slow-updating layer that maintains long-term knowledge and
    reasoning patterns. All updates require human approval.
    """

    def __init__(
        self,
        latent_dim: int = 512,
        hidden_dim: int = 2048,
        num_layers: int = 6,
        dropout: float = 0.05,  # Lower dropout for stable training
    ) -> None:
        """Initialize Abstract Core.

        Args:
            latent_dim: Dimension of JEPA latents
            hidden_dim: Hidden dimension for deep reasoning
            num_layers: Number of transformer layers
            dropout: Dropout probability (lower for stability)
        """
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim

        # Input projection
        self.input_proj = nn.Linear(latent_dim, hidden_dim)

        # Deep transformer for abstract reasoning
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=16,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        # Output projection
        self.output_proj = nn.Linear(hidden_dim, latent_dim)

        # Knowledge state (learned global context)
        self.knowledge_state = nn.Parameter(torch.zeros(1, 1, hidden_dim))

        # Update tracking
        self.register_buffer("update_count", torch.zeros(1, dtype=torch.long))
        self.register_buffer("last_update_time", torch.zeros(1, dtype=torch.long))

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights."""
        nn.init.trunc_normal_(self.knowledge_state, std=0.02)

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        """Process latents through abstract core.

        Args:
            latents: Input latents (batch, latent_dim) or (batch, seq, latent_dim)

        Returns:
            Processed latents (same shape as input)
        """
        input_shape = latents.shape
        if latents.dim() == 2:
            latents = latents.unsqueeze(1)  # Add sequence dim

        batch_size, seq_len, _ = latents.shape

        # Project to hidden dim
        x = self.input_proj(latents)

        # Add knowledge state as global context
        knowledge_expanded = self.knowledge_state.expand(batch_size, -1, -1)
        x = torch.cat([knowledge_expanded, x], dim=1)

        # Apply deep reasoning
        x = self.transformer(x)

        # Extract sequence (skip knowledge token)
        x = x[:, 1:, :]

        # Project back to latent dim
        x = self.output_proj(x)

        # Restore original shape
        if len(input_shape) == 2:
            x = x.squeeze(1)

        return x

    def create_update_proposal(
        self,
        rationale: str,
        evidence: list[str],
        expected_effects: str,
    ) -> dict[str, Any]:
        """Create update proposal for human review.

        Args:
            rationale: Why this update is needed
            evidence: Supporting evidence
            expected_effects: What will change

        Returns:
            Update proposal dictionary
        """
        # Save current state for rollback
        current_state = {
            "knowledge_state": self.knowledge_state.detach().cpu().clone(),
            "update_count": self.update_count.item(),
        }

        proposal = {
            "change_id": f"abstract_core_update_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "layer": "abstract_core",
            "timestamp": datetime.now().isoformat(),
            "rationale": rationale,
            "evidence": evidence,
            "expected_effects": expected_effects,
            "current_state": current_state,
            "revert_plan": "Restore from current_state checkpoint",
            "status": "pending_approval",
        }

        return proposal

    def apply_update(
        self,
        proposal: dict[str, Any],
        new_knowledge_state: torch.Tensor,
    ) -> None:
        """Apply approved update (only call after human approval).

        Args:
            proposal: The approved proposal
            new_knowledge_state: New knowledge state tensor
        """
        if proposal.get("status") != "approved":
            raise ValueError("Cannot apply unapproved update")

        # Update knowledge state
        self.knowledge_state.data.copy_(new_knowledge_state.to(self.knowledge_state.device))

        # Update metadata
        self.update_count += 1
        self.last_update_time.fill_(int(datetime.now().timestamp()))

    def revert_update(self, proposal: dict[str, Any]) -> None:
        """Revert to state before update.

        Args:
            proposal: The proposal containing current_state
        """
        if "current_state" not in proposal:
            raise ValueError("No state to revert to in proposal")

        state = proposal["current_state"]

        # Restore knowledge state
        self.knowledge_state.data.copy_(
            state["knowledge_state"].to(self.knowledge_state.device)
        )

        # Restore update count
        self.update_count.fill_(state["update_count"])

    def save_checkpoint(self, path: Path) -> None:
        """Save abstract core checkpoint.

        Args:
            path: Path to save checkpoint
        """
        checkpoint = {
            "state_dict": self.state_dict(),
            "update_count": self.update_count.item(),
            "last_update_time": self.last_update_time.item(),
            "config": {
                "latent_dim": self.latent_dim,
                "hidden_dim": self.hidden_dim,
            },
        }

        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, path)

    def load_checkpoint(self, path: Path) -> None:
        """Load abstract core checkpoint.

        Args:
            path: Path to checkpoint
        """
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        self.load_state_dict(checkpoint["state_dict"])
        self.update_count.fill_(checkpoint["update_count"])
        self.last_update_time.fill_(checkpoint["last_update_time"])

    def get_metadata(self) -> dict[str, Any]:
        """Get metadata about abstract core state.

        Returns:
            Dictionary with metadata
        """
        return {
            "update_count": self.update_count.item(),
            "last_update_time": datetime.fromtimestamp(
                self.last_update_time.item()
            ).isoformat() if self.last_update_time.item() > 0 else None,
            "knowledge_state_norm": self.knowledge_state.norm().item(),
        }
