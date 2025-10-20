"""HRM Reflection Loop - monitors consistency and proposes updates.

The reflection loop tracks thought traces, detects inconsistencies,
and generates update proposals for the abstract core.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


class ThoughtTrace:
    """Stores a single reasoning trace."""

    def __init__(
        self,
        trace_id: str,
        input_latent: torch.Tensor,
        working_memory_output: torch.Tensor,
        abstract_core_output: torch.Tensor,
        timestamp: datetime | None = None,
    ) -> None:
        """Initialize thought trace.

        Args:
            trace_id: Unique identifier
            input_latent: Input to reasoning
            working_memory_output: Output from working memory
            abstract_core_output: Output from abstract core
            timestamp: When trace was created
        """
        self.trace_id = trace_id
        self.input_latent = input_latent.detach().cpu()
        self.working_memory_output = working_memory_output.detach().cpu()
        self.abstract_core_output = abstract_core_output.detach().cpu()
        self.timestamp = timestamp or datetime.now()

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary (for JSON serialization).

        Returns:
            Dictionary representation
        """
        return {
            "trace_id": self.trace_id,
            "timestamp": self.timestamp.isoformat(),
            "input_norm": self.input_latent.norm().item(),
            "wm_output_norm": self.working_memory_output.norm().item(),
            "ac_output_norm": self.abstract_core_output.norm().item(),
        }


class ReflectionLoop(nn.Module):
    """Reflection Loop for HRM.

    Monitors consistency between working memory and abstract core,
    detects patterns requiring abstract core updates, and generates
    update proposals for human approval.
    """

    def __init__(
        self,
        latent_dim: int = 512,
        trace_buffer_size: int = 1000,
        consistency_threshold: float = 0.7,
        log_dir: Path = Path("logs/reflections"),
    ) -> None:
        """Initialize Reflection Loop.

        Args:
            latent_dim: Dimension of latents
            trace_buffer_size: Max traces to keep in memory
            consistency_threshold: Threshold for consistency score
            log_dir: Directory for reflection logs
        """
        super().__init__()
        self.latent_dim = latent_dim
        self.trace_buffer_size = trace_buffer_size
        self.consistency_threshold = consistency_threshold
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Circular buffer for thought traces
        self.traces: list[ThoughtTrace] = []

        # Consistency detector (small network)
        self.consistency_net = nn.Sequential(
            nn.Linear(latent_dim * 3, 256),  # input, wm_out, ac_out
            nn.GELU(),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, 1),  # Consistency score
            nn.Sigmoid(),
        )

        # Counters
        self.register_buffer("total_traces", torch.zeros(1, dtype=torch.long))
        self.register_buffer("low_consistency_count", torch.zeros(1, dtype=torch.long))

    def add_trace(
        self,
        input_latent: torch.Tensor,
        working_memory_output: torch.Tensor,
        abstract_core_output: torch.Tensor,
    ) -> ThoughtTrace:
        """Add a new thought trace.

        Args:
            input_latent: Input to reasoning
            working_memory_output: Output from working memory
            abstract_core_output: Output from abstract core

        Returns:
            The created trace
        """
        trace_id = f"trace_{self.total_traces.item():06d}"

        trace = ThoughtTrace(
            trace_id=trace_id,
            input_latent=input_latent,
            working_memory_output=working_memory_output,
            abstract_core_output=abstract_core_output,
        )

        # Add to buffer (circular)
        self.traces.append(trace)
        if len(self.traces) > self.trace_buffer_size:
            self.traces.pop(0)

        self.total_traces += 1

        return trace

    def compute_consistency(
        self,
        input_latent: torch.Tensor,
        working_memory_output: torch.Tensor,
        abstract_core_output: torch.Tensor,
    ) -> torch.Tensor:
        """Compute consistency score between WM and AC outputs.

        Args:
            input_latent: Input to reasoning
            working_memory_output: Output from working memory
            abstract_core_output: Output from abstract core

        Returns:
            Consistency score (0-1, higher = more consistent)
        """
        # Concatenate all inputs
        combined = torch.cat(
            [input_latent, working_memory_output, abstract_core_output],
            dim=-1,
        )

        # Compute consistency
        score = self.consistency_net(combined)

        return score

    def detect_inconsistency(
        self,
        input_latent: torch.Tensor,
        working_memory_output: torch.Tensor,
        abstract_core_output: torch.Tensor,
    ) -> tuple[bool, float]:
        """Detect if outputs are inconsistent.

        Args:
            input_latent: Input to reasoning
            working_memory_output: Output from working memory
            abstract_core_output: Output from abstract core

        Returns:
            Tuple of (is_inconsistent, consistency_score)
        """
        score = self.compute_consistency(
            input_latent,
            working_memory_output,
            abstract_core_output,
        )

        score_value = score.mean().item()
        is_inconsistent = score_value < self.consistency_threshold

        if is_inconsistent:
            self.low_consistency_count += 1

        return is_inconsistent, score_value

    def analyze_traces(self, num_recent: int = 100) -> dict[str, Any]:
        """Analyze recent traces for patterns.

        Args:
            num_recent: Number of recent traces to analyze

        Returns:
            Analysis results
        """
        if not self.traces:
            return {"status": "no_traces"}

        recent = self.traces[-num_recent:]

        # Compute statistics
        inconsistency_rate = self.low_consistency_count.item() / max(
            1, self.total_traces.item()
        )

        analysis = {
            "total_traces": self.total_traces.item(),
            "traces_analyzed": len(recent),
            "inconsistency_rate": inconsistency_rate,
            "low_consistency_count": self.low_consistency_count.item(),
            "recommendation": "No update needed",
        }

        # Determine if update is warranted
        if inconsistency_rate > 0.3:  # >30% inconsistency
            analysis["recommendation"] = "Update warranted: High inconsistency rate"
        elif self.low_consistency_count.item() > 50:  # Absolute threshold
            analysis["recommendation"] = "Update warranted: Many inconsistent traces"

        return analysis

    def generate_update_proposal(
        self,
        working_memory: nn.Module,
        abstract_core: nn.Module,
    ) -> dict[str, Any]:
        """Generate update proposal based on trace analysis.

        Args:
            working_memory: Working memory module
            abstract_core: Abstract core module

        Returns:
            Update proposal for human review
        """
        analysis = self.analyze_traces()

        # Collect evidence
        evidence = [
            f"Analyzed {analysis['traces_analyzed']} recent traces",
            f"Inconsistency rate: {analysis['inconsistency_rate']:.2%}",
            f"Low consistency count: {analysis['low_consistency_count']}",
        ]

        # Rationale
        rationale = (
            f"Working memory has adapted to new patterns, but abstract core "
            f"is producing inconsistent outputs in {analysis['inconsistency_rate']:.1%} "
            f"of cases. Updating abstract core could improve long-term consistency."
        )

        # Expected effects
        expected_effects = (
            "Abstract core will better align with working memory patterns, "
            "reducing inconsistency and improving reasoning stability."
        )

        # Create proposal
        proposal = abstract_core.create_update_proposal(
            rationale=rationale,
            evidence=evidence,
            expected_effects=expected_effects,
        )

        proposal["analysis"] = analysis

        return proposal

    def save_trace_log(self, num_traces: int = 100) -> Path:
        """Save recent traces to log file.

        Args:
            num_traces: Number of recent traces to save

        Returns:
            Path to saved log file
        """
        if not self.traces:
            raise ValueError("No traces to save")

        recent = self.traces[-num_traces:]

        log_data = {
            "timestamp": datetime.now().isoformat(),
            "total_traces": self.total_traces.item(),
            "traces_saved": len(recent),
            "traces": [trace.to_dict() for trace in recent],
        }

        log_file = self.log_dir / f"traces_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(log_file, "w") as f:
            json.dump(log_data, f, indent=2)

        return log_file

    def load_trace_log(self, log_file: Path) -> None:
        """Load traces from log file.

        Args:
            log_file: Path to log file
        """
        with open(log_file) as f:
            log_data = json.load(f)

        # Note: This loads metadata only (not full tensors)
        # Full trace restoration would require saving tensor data
        self.total_traces.fill_(log_data["total_traces"])

    def get_statistics(self) -> dict[str, Any]:
        """Get reflection loop statistics.

        Returns:
            Dictionary with statistics
        """
        return {
            "total_traces": self.total_traces.item(),
            "traces_in_buffer": len(self.traces),
            "low_consistency_count": self.low_consistency_count.item(),
            "inconsistency_rate": (
                self.low_consistency_count.item() / max(1, self.total_traces.item())
            ),
            "consistency_threshold": self.consistency_threshold,
        }
