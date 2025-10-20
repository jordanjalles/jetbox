"""Tests for HRM components."""

import tempfile
from pathlib import Path

import torch

from core.hrm import (
    AbstractCore,
    ReflectionLoop,
    WorkingMemory,
    create_hrm_lite,
)


def test_working_memory_forward() -> None:
    """Test working memory forward pass."""
    wm = WorkingMemory(latent_dim=512, hidden_dim=1024, num_layers=2)
    wm.eval()

    latents = torch.randn(2, 512)

    with torch.no_grad():
        output = wm(latents, update_state=False)

    assert output.shape == (2, 512)
    assert not torch.isnan(output).any()


def test_working_memory_state_update() -> None:
    """Test working memory state updates."""
    wm = WorkingMemory(latent_dim=512)
    wm.eval()

    # Reset state
    wm.reset_task_state()
    assert wm.step_count.item() == 0

    # Process with state update
    latents = torch.randn(2, 512)
    with torch.no_grad():
        _ = wm(latents, update_state=True)

    assert wm.step_count.item() == 1

    # Process again
    with torch.no_grad():
        _ = wm(latents, update_state=True)

    assert wm.step_count.item() == 2


def test_working_memory_freeze() -> None:
    """Test freezing base parameters."""
    wm = WorkingMemory(latent_dim=512)

    # Freeze base
    wm.freeze_base()

    # Check that base is frozen
    assert not wm.input_proj_base.weight.requires_grad

    # Check that LoRA is trainable
    assert wm.input_proj_lora.lora_A.requires_grad
    assert wm.input_proj_lora.lora_B.requires_grad


def test_abstract_core_forward() -> None:
    """Test abstract core forward pass."""
    ac = AbstractCore(latent_dim=512, hidden_dim=2048, num_layers=6)
    ac.eval()

    latents = torch.randn(2, 512)

    with torch.no_grad():
        output = ac(latents)

    assert output.shape == (2, 512)
    assert not torch.isnan(output).any()


def test_abstract_core_update_proposal() -> None:
    """Test creating update proposal."""
    ac = AbstractCore(latent_dim=512)

    proposal = ac.create_update_proposal(
        rationale="Test update",
        evidence=["Evidence 1", "Evidence 2"],
        expected_effects="Expected improvements",
    )

    assert proposal["layer"] == "abstract_core"
    assert proposal["status"] == "pending_approval"
    assert "change_id" in proposal
    assert "current_state" in proposal


def test_abstract_core_apply_update() -> None:
    """Test applying approved update."""
    ac = AbstractCore(latent_dim=512, hidden_dim=256)

    # Create proposal
    proposal = ac.create_update_proposal(
        rationale="Test",
        evidence=["Test"],
        expected_effects="Test",
    )

    # Approve proposal
    proposal["status"] = "approved"

    # Create new knowledge state
    new_state = torch.randn_like(ac.knowledge_state)
    old_count = ac.update_count.item()

    # Apply update
    ac.apply_update(proposal, new_state)

    # Check update was applied
    assert ac.update_count.item() == old_count + 1
    assert torch.allclose(ac.knowledge_state, new_state)


def test_abstract_core_revert() -> None:
    """Test reverting an update."""
    ac = AbstractCore(latent_dim=512, hidden_dim=256)

    # Save original state
    original_state = ac.knowledge_state.data.clone()

    # Create and apply proposal
    proposal = ac.create_update_proposal("Test", ["Test"], "Test")
    proposal["status"] = "approved"
    new_state = torch.randn_like(ac.knowledge_state)
    ac.apply_update(proposal, new_state)

    # Revert
    ac.revert_update(proposal)

    # Check reverted to original
    assert torch.allclose(ac.knowledge_state, original_state)


def test_reflection_loop_add_trace() -> None:
    """Test adding thought traces."""
    reflection = ReflectionLoop(latent_dim=512)

    input_latent = torch.randn(1, 512)
    wm_output = torch.randn(1, 512)
    ac_output = torch.randn(1, 512)

    trace = reflection.add_trace(input_latent, wm_output, ac_output)

    assert trace.trace_id == "trace_000000"
    assert reflection.total_traces.item() == 1
    assert len(reflection.traces) == 1


def test_reflection_loop_consistency() -> None:
    """Test consistency detection."""
    reflection = ReflectionLoop(latent_dim=512, consistency_threshold=0.7)
    reflection.eval()

    input_latent = torch.randn(1, 512)
    wm_output = torch.randn(1, 512)
    ac_output = torch.randn(1, 512)

    with torch.no_grad():
        is_inconsistent, score = reflection.detect_inconsistency(
            input_latent, wm_output, ac_output
        )

    assert isinstance(is_inconsistent, bool)
    assert 0.0 <= score <= 1.0


def test_reflection_loop_analyze_traces() -> None:
    """Test trace analysis."""
    reflection = ReflectionLoop(latent_dim=512)

    # Add some traces
    for _ in range(10):
        input_latent = torch.randn(1, 512)
        wm_output = torch.randn(1, 512)
        ac_output = torch.randn(1, 512)
        reflection.add_trace(input_latent, wm_output, ac_output)

    analysis = reflection.analyze_traces()

    assert analysis["total_traces"] == 10
    assert "inconsistency_rate" in analysis
    assert "recommendation" in analysis


def test_hrm_reasoner_forward() -> None:
    """Test HRM reasoner forward pass."""
    hrm = create_hrm_lite(latent_dim=512)
    hrm.eval()

    latents = torch.randn(2, 512)

    with torch.no_grad():
        outputs = hrm(latents, record_trace=False)

    assert "fused" in outputs
    assert outputs["fused"].shape == (2, 512)


def test_hrm_reasoner_with_trace() -> None:
    """Test HRM reasoner with thought trace recording."""
    hrm = create_hrm_lite(latent_dim=512)
    hrm.eval()

    latents = torch.randn(2, 512)

    with torch.no_grad():
        outputs = hrm(latents, record_trace=True)

    assert "consistency_score" in outputs
    assert hrm.reflection.total_traces.item() == 1


def test_hrm_reasoner_modes() -> None:
    """Test switching between HRM modes."""
    hrm = create_hrm_lite(latent_dim=512)

    # Test fast adapt mode
    hrm.fast_adapt()
    assert not hrm.abstract_core.training

    # Test full train mode
    hrm.full_train()
    assert hrm.working_memory.training


def test_hrm_reasoner_checkpoint() -> None:
    """Test saving and loading HRM checkpoint."""
    hrm = create_hrm_lite(latent_dim=512)

    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_dir = Path(tmpdir) / "hrm_checkpoint"

        # Save checkpoint
        hrm.save_checkpoint(checkpoint_dir)

        # Check files exist
        assert (checkpoint_dir / "working_memory.pth").exists()
        assert (checkpoint_dir / "abstract_core.pth").exists()
        assert (checkpoint_dir / "reflection.pth").exists()
        assert (checkpoint_dir / "fusion.pth").exists()

        # Create new HRM and load
        hrm2 = create_hrm_lite(latent_dim=512)
        hrm2.load_checkpoint(checkpoint_dir)

        # Verify loaded correctly (check one parameter)
        assert torch.allclose(
            hrm.working_memory.input_proj_base.weight,
            hrm2.working_memory.input_proj_base.weight,
        )


def test_hrm_gradient_flow() -> None:
    """Test gradient flow through HRM."""
    hrm = create_hrm_lite(latent_dim=512)
    hrm.train()

    latents = torch.randn(2, 512)

    outputs = hrm(latents, record_trace=False)
    loss = outputs["fused"].sum()
    loss.backward()

    # Check gradients exist
    assert hrm.working_memory.output_proj.weight.grad is not None
    assert hrm.fusion[0].weight.grad is not None
