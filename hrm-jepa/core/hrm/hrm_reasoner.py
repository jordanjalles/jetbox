"""HRM Reasoner - combines working memory, abstract core, and reflection.

This is the main HRM module that orchestrates hierarchical reasoning
over JEPA latents with explicit fast/slow thinking separation.
"""

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from core.hrm.abstract_core import AbstractCore
from core.hrm.reflection_loop import ReflectionLoop
from core.hrm.working_memory import WorkingMemory


class HRMReasoner(nn.Module):
    """Hierarchical Reasoning Model (HRM).

    Combines:
    - Working Memory: Fast-adapting task context
    - Abstract Core: Slow-updating long-term knowledge
    - Reflection Loop: Consistency monitoring and update proposals
    """

    def __init__(
        self,
        latent_dim: int = 512,
        wm_hidden_dim: int = 1024,
        wm_layers: int = 2,
        ac_hidden_dim: int = 2048,
        ac_layers: int = 6,
        lora_rank: int = 8,
        log_dir: Path = Path("logs/reflections"),
    ) -> None:
        """Initialize HRM Reasoner.

        Args:
            latent_dim: Dimension of JEPA latents
            wm_hidden_dim: Working memory hidden dimension
            wm_layers: Number of working memory layers
            ac_hidden_dim: Abstract core hidden dimension
            ac_layers: Number of abstract core layers
            lora_rank: LoRA rank for working memory
            log_dir: Directory for reflection logs
        """
        super().__init__()
        self.latent_dim = latent_dim

        # Working Memory (fast adaptation)
        self.working_memory = WorkingMemory(
            latent_dim=latent_dim,
            hidden_dim=wm_hidden_dim,
            num_layers=wm_layers,
            lora_rank=lora_rank,
        )

        # Abstract Core (slow updates, gated)
        self.abstract_core = AbstractCore(
            latent_dim=latent_dim,
            hidden_dim=ac_hidden_dim,
            num_layers=ac_layers,
        )

        # Reflection Loop (monitors consistency)
        self.reflection = ReflectionLoop(
            latent_dim=latent_dim,
            log_dir=log_dir,
        )

        # Fusion layer (combines WM and AC outputs)
        self.fusion = nn.Sequential(
            nn.Linear(latent_dim * 2, latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, latent_dim),
        )

    def forward(
        self,
        latents: torch.Tensor,
        use_working_memory: bool = True,
        use_abstract_core: bool = True,
        record_trace: bool = True,
    ) -> dict[str, torch.Tensor]:
        """Reason over latents using HRM.

        Args:
            latents: JEPA latents (batch, latent_dim) or (batch, seq, latent_dim)
            use_working_memory: Whether to use working memory
            use_abstract_core: Whether to use abstract core
            record_trace: Whether to record thought trace

        Returns:
            Dictionary with outputs and intermediate results
        """
        outputs = {"input": latents}

        # Working Memory reasoning
        if use_working_memory:
            wm_output = self.working_memory(latents)
            outputs["working_memory"] = wm_output
        else:
            wm_output = latents

        # Abstract Core reasoning
        if use_abstract_core:
            ac_output = self.abstract_core(latents)
            outputs["abstract_core"] = ac_output
        else:
            ac_output = latents

        # Fuse WM and AC outputs
        if use_working_memory and use_abstract_core:
            # Ensure same shape for fusion
            if (wm_output.dim() == 2 and ac_output.dim() == 2) or (wm_output.dim() == 3 and ac_output.dim() == 3):
                fused = self.fusion(torch.cat([wm_output, ac_output], dim=-1))
            else:
                # Shape mismatch, use WM only
                fused = wm_output

            outputs["fused"] = fused
        elif use_working_memory:
            fused = wm_output
            outputs["fused"] = fused
        elif use_abstract_core:
            fused = ac_output
            outputs["fused"] = fused
        else:
            fused = latents
            outputs["fused"] = fused

        # Record thought trace for reflection
        if record_trace and use_working_memory and use_abstract_core:
            # Use first item in batch for trace
            self.reflection.add_trace(
                input_latent=latents[0:1],
                working_memory_output=wm_output[0:1],
                abstract_core_output=ac_output[0:1],
            )

            # Check consistency
            is_inconsistent, score = self.reflection.detect_inconsistency(
                latents[0:1], wm_output[0:1], ac_output[0:1]
            )
            outputs["consistency_score"] = score
            outputs["is_inconsistent"] = is_inconsistent

        return outputs

    def fast_adapt(self, num_steps: int = 10) -> None:
        """Switch to fast adaptation mode (only train WM LoRA).

        Args:
            num_steps: Expected number of adaptation steps
        """
        # Freeze everything except WM LoRA
        self.working_memory.freeze_base()
        self.abstract_core.requires_grad_(False)
        self.reflection.requires_grad_(False)
        self.fusion.requires_grad_(False)

    def full_train(self) -> None:
        """Switch to full training mode (train everything)."""
        self.working_memory.unfreeze_all()
        self.abstract_core.requires_grad_(True)
        self.reflection.requires_grad_(True)
        self.fusion.requires_grad_(True)

    def freeze_abstract_core(self) -> None:
        """Freeze abstract core (normal operating mode)."""
        self.abstract_core.requires_grad_(False)

    def propose_abstract_core_update(self) -> dict[str, Any]:
        """Generate update proposal for abstract core.

        Returns:
            Update proposal for human review
        """
        return self.reflection.generate_update_proposal(
            self.working_memory,
            self.abstract_core,
        )

    def apply_abstract_core_update(
        self,
        proposal: dict[str, Any],
        new_knowledge_state: torch.Tensor,
    ) -> None:
        """Apply approved abstract core update.

        Args:
            proposal: Approved proposal
            new_knowledge_state: New knowledge state
        """
        self.abstract_core.apply_update(proposal, new_knowledge_state)

    def save_checkpoint(self, checkpoint_dir: Path) -> None:
        """Save HRM checkpoint.

        Args:
            checkpoint_dir: Directory to save checkpoints
        """
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Save working memory
        wm_path = checkpoint_dir / "working_memory.pth"
        torch.save(self.working_memory.state_dict(), wm_path)

        # Save abstract core (with metadata)
        ac_path = checkpoint_dir / "abstract_core.pth"
        self.abstract_core.save_checkpoint(ac_path)

        # Save reflection state
        ref_path = checkpoint_dir / "reflection.pth"
        torch.save(
            {
                "state_dict": self.reflection.state_dict(),
                "statistics": self.reflection.get_statistics(),
            },
            ref_path,
        )

        # Save fusion layer
        fusion_path = checkpoint_dir / "fusion.pth"
        torch.save(self.fusion.state_dict(), fusion_path)

        # Save reflection traces
        self.reflection.save_trace_log()

    def load_checkpoint(self, checkpoint_dir: Path) -> None:
        """Load HRM checkpoint.

        Args:
            checkpoint_dir: Directory containing checkpoints
        """
        # Load working memory
        wm_path = checkpoint_dir / "working_memory.pth"
        if wm_path.exists():
            self.working_memory.load_state_dict(
                torch.load(wm_path, map_location="cpu", weights_only=True)
            )

        # Load abstract core
        ac_path = checkpoint_dir / "abstract_core.pth"
        if ac_path.exists():
            self.abstract_core.load_checkpoint(ac_path)

        # Load reflection
        ref_path = checkpoint_dir / "reflection.pth"
        if ref_path.exists():
            checkpoint = torch.load(ref_path, map_location="cpu", weights_only=False)
            self.reflection.load_state_dict(checkpoint["state_dict"])

        # Load fusion layer
        fusion_path = checkpoint_dir / "fusion.pth"
        if fusion_path.exists():
            self.fusion.load_state_dict(
                torch.load(fusion_path, map_location="cpu", weights_only=True)
            )

    def get_status(self) -> dict[str, Any]:
        """Get HRM status and statistics.

        Returns:
            Status dictionary
        """
        return {
            "working_memory": {
                "task_steps": self.working_memory.step_count.item(),
                "task_state_norm": self.working_memory.task_state.norm().item(),
            },
            "abstract_core": self.abstract_core.get_metadata(),
            "reflection": self.reflection.get_statistics(),
        }


def create_hrm_lite(latent_dim: int = 512, log_dir: Path = Path("logs/reflections")) -> HRMReasoner:
    """Create a lightweight HRM for RTX 3090.

    Args:
        latent_dim: Dimension of JEPA latent space
        log_dir: Directory for reflection logs

    Returns:
        Lightweight HRM model
    """
    return HRMReasoner(
        latent_dim=latent_dim,
        wm_hidden_dim=512,  # Smaller for memory
        wm_layers=2,
        ac_hidden_dim=1024,  # Smaller than full spec
        ac_layers=4,  # Fewer layers
        lora_rank=8,
        log_dir=log_dir,
    )
