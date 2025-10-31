"""Compact storage utilities for limited disk space.

Strategies:
1. Quantize checkpoints to int8 (75% size reduction)
2. Store only essential state (drop optimizer if not training)
3. Compress logs with gzip
4. Limit trace buffer size
5. Periodic cleanup of old checkpoints
"""

import gzip
import json
import shutil
from pathlib import Path
from typing import Any

import torch


def quantize_checkpoint(
    checkpoint_path: Path,
    output_path: Path,
    quantize_level: str = "int8",
) -> dict[str, Any]:
    """Quantize checkpoint to reduce storage.

    Args:
        checkpoint_path: Path to original checkpoint
        output_path: Path to save quantized checkpoint
        quantize_level: Quantization level ('int8' or 'float16')

    Returns:
        Statistics about size reduction
    """
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    original_size = checkpoint_path.stat().st_size

    # Quantize state dict
    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
        quantized_state = {}

        for key, value in state_dict.items():
            if isinstance(value, torch.Tensor) and value.dtype == torch.float32:
                if quantize_level == "int8":
                    # Simple int8 quantization
                    scale = value.abs().max() / 127.0
                    quantized = (value / scale).round().to(torch.int8)
                    quantized_state[key] = {"data": quantized, "scale": scale}
                elif quantize_level == "float16":
                    quantized_state[key] = value.half()
                else:
                    quantized_state[key] = value
            else:
                quantized_state[key] = value

        checkpoint["state_dict"] = quantized_state
        checkpoint["quantized"] = True
        checkpoint["quantize_level"] = quantize_level

    # Save quantized
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, output_path)

    new_size = output_path.stat().st_size
    reduction = 1 - (new_size / original_size)

    return {
        "original_size_mb": original_size / 1024 / 1024,
        "new_size_mb": new_size / 1024 / 1024,
        "reduction_pct": reduction * 100,
    }


def compress_logs(log_dir: Path, keep_recent: int = 10) -> dict[str, Any]:
    """Compress old log files with gzip.

    Args:
        log_dir: Directory containing logs
        keep_recent: Number of recent logs to keep uncompressed

    Returns:
        Statistics about compression
    """
    log_files = sorted(log_dir.glob("*.log"), key=lambda p: p.stat().st_mtime)

    if len(log_files) <= keep_recent:
        return {"compressed": 0, "total_size_saved_mb": 0}

    to_compress = log_files[:-keep_recent]
    total_saved = 0
    compressed_count = 0

    for log_file in to_compress:
        if log_file.suffix == ".gz":
            continue  # Already compressed

        original_size = log_file.stat().st_size

        # Compress
        with open(log_file, "rb") as f_in:
            with gzip.open(f"{log_file}.gz", "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)

        compressed_size = Path(f"{log_file}.gz").stat().st_size
        total_saved += original_size - compressed_size

        # Remove original
        log_file.unlink()
        compressed_count += 1

    return {
        "compressed": compressed_count,
        "total_size_saved_mb": total_saved / 1024 / 1024,
    }


def cleanup_old_checkpoints(
    checkpoint_dir: Path,
    keep_best: int = 3,
    keep_recent: int = 5,
    metric_key: str = "loss",
) -> dict[str, Any]:
    """Clean up old checkpoints, keeping only best and most recent.

    Args:
        checkpoint_dir: Directory containing checkpoints
        keep_best: Number of best checkpoints to keep
        keep_recent: Number of recent checkpoints to keep
        metric_key: Metric to use for "best" (lower is better)

    Returns:
        Statistics about cleanup
    """
    checkpoints = list(checkpoint_dir.glob("*.pth"))

    if len(checkpoints) <= keep_best + keep_recent:
        return {"deleted": 0, "space_freed_mb": 0}

    # Load metadata
    checkpoint_info = []
    for ckpt_path in checkpoints:
        try:
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            metric = ckpt.get(metric_key, float("inf"))
            mtime = ckpt_path.stat().st_mtime
            size = ckpt_path.stat().st_size
            checkpoint_info.append(
                {
                    "path": ckpt_path,
                    "metric": metric,
                    "mtime": mtime,
                    "size": size,
                }
            )
        except Exception:
            # Skip corrupted checkpoints
            continue

    # Sort by metric (best first)
    by_metric = sorted(checkpoint_info, key=lambda x: x["metric"])
    best_paths = {x["path"] for x in by_metric[:keep_best]}

    # Sort by time (recent first)
    by_time = sorted(checkpoint_info, key=lambda x: x["mtime"], reverse=True)
    recent_paths = {x["path"] for x in by_time[:keep_recent]}

    # Keep union of best and recent
    keep_paths = best_paths | recent_paths

    # Delete the rest
    deleted = 0
    space_freed = 0

    for info in checkpoint_info:
        if info["path"] not in keep_paths:
            info["path"].unlink()
            deleted += 1
            space_freed += info["size"]

    return {
        "deleted": deleted,
        "space_freed_mb": space_freed / 1024 / 1024,
    }


def estimate_storage_usage(project_root: Path) -> dict[str, Any]:
    """Estimate storage usage by component.

    Args:
        project_root: Root directory of project

    Returns:
        Storage breakdown
    """

    def dir_size(path: Path) -> int:
        """Get total size of directory."""
        return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())

    checkpoints_dir = project_root / "checkpoints"
    logs_dir = project_root / "logs"
    data_dir = project_root / "data"

    usage = {
        "checkpoints_mb": (
            dir_size(checkpoints_dir) / 1024 / 1024 if checkpoints_dir.exists() else 0
        ),
        "logs_mb": dir_size(logs_dir) / 1024 / 1024 if logs_dir.exists() else 0,
        "data_mb": dir_size(data_dir) / 1024 / 1024 if data_dir.exists() else 0,
    }

    usage["total_mb"] = sum(usage.values())

    return usage


def suggest_cleanup_actions(
    project_root: Path,
    storage_limit_mb: int = 1000,
) -> list[str]:
    """Suggest cleanup actions based on current usage.

    Args:
        project_root: Root directory of project
        storage_limit_mb: Storage limit in MB

    Returns:
        List of suggested actions
    """
    usage = estimate_storage_usage(project_root)
    suggestions = []

    if usage["total_mb"] > storage_limit_mb:
        overage = usage["total_mb"] - storage_limit_mb
        suggestions.append(
            f"⚠️ Storage usage ({usage['total_mb']:.1f}MB) exceeds limit "
            f"({storage_limit_mb}MB) by {overage:.1f}MB"
        )

    if usage["checkpoints_mb"] > 500:
        suggestions.append(
            f"Checkpoints using {usage['checkpoints_mb']:.1f}MB. "
            "Consider: cleanup_old_checkpoints() or quantize_checkpoint()"
        )

    if usage["logs_mb"] > 100:
        suggestions.append(
            f"Logs using {usage['logs_mb']:.1f}MB. "
            "Consider: compress_logs()"
        )

    if not suggestions:
        suggestions.append(f"✓ Storage usage OK ({usage['total_mb']:.1f}MB)")

    return suggestions
