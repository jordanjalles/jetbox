"""Train JEPA text encoder on synthetic data.

Uses contrastive learning to align similar questions/answers in latent space.
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.encoders import SimpleTokenizer
from core.jepa_core import JEPACore, create_jepa_lite
from core.objectives.jepa_objectives import ContrastiveLoss


class SyntheticTextDataset(Dataset):
    """Dataset for synthetic text data."""

    def __init__(
        self,
        jsonl_file: Path,
        tokenizer: SimpleTokenizer,
        max_length: int = 256,
    ) -> None:
        """Initialize dataset.

        Args:
            jsonl_file: Path to JSONL file
            tokenizer: Tokenizer
            max_length: Maximum sequence length
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = []

        # Load JSONL
        with open(jsonl_file) as f:
            for line in f:
                self.samples.append(json.loads(line))

    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """Get a sample.

        Args:
            idx: Sample index

        Returns:
            Dict with tokenized question and answer
        """
        sample = self.samples[idx]

        # Tokenize question
        q_input_ids, q_attention_mask = self.tokenizer.encode(
            sample["question"], max_length=self.max_length
        )

        # Tokenize answer
        a_input_ids, a_attention_mask = self.tokenizer.encode(
            sample["answer"], max_length=self.max_length
        )

        return {
            "question_input_ids": torch.tensor(q_input_ids),
            "question_attention_mask": torch.tensor(q_attention_mask),
            "answer_input_ids": torch.tensor(a_input_ids),
            "answer_attention_mask": torch.tensor(a_attention_mask),
            "is_consistent": torch.tensor(
                1.0 if sample.get("is_consistent", True) else 0.0
            ),
            "category": sample["category"],
        }


def train_epoch(
    model: JEPACore,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: ContrastiveLoss,
    device: str = "cpu",
) -> dict[str, float]:
    """Train for one epoch.

    Args:
        model: JEPA model
        dataloader: Data loader
        optimizer: Optimizer
        criterion: Loss function
        device: Device to use

    Returns:
        Training metrics
    """
    model.train()
    total_loss = 0.0
    num_batches = 0

    for batch in dataloader:
        # Move to device
        q_input_ids = batch["question_input_ids"].to(device)
        q_attention_mask = batch["question_attention_mask"].to(device)
        a_input_ids = batch["answer_input_ids"].to(device)
        a_attention_mask = batch["answer_attention_mask"].to(device)

        # Encode question and answer
        q_latent = model.encode_text(q_input_ids, q_attention_mask)
        a_latent = model.encode_text(a_input_ids, a_attention_mask)

        # Contrastive loss: align question-answer pairs
        loss = criterion(q_latent, a_latent)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return {
        "loss": total_loss / num_batches if num_batches > 0 else 0.0,
    }


def evaluate(
    model: JEPACore,
    dataloader: DataLoader,
    criterion: ContrastiveLoss,
    device: str = "cpu",
) -> dict[str, float]:
    """Evaluate model.

    Args:
        model: JEPA model
        dataloader: Data loader
        criterion: Loss function
        device: Device to use

    Returns:
        Evaluation metrics
    """
    model.eval()
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch in dataloader:
            q_input_ids = batch["question_input_ids"].to(device)
            q_attention_mask = batch["question_attention_mask"].to(device)
            a_input_ids = batch["answer_input_ids"].to(device)
            a_attention_mask = batch["answer_attention_mask"].to(device)

            q_latent = model.encode_text(q_input_ids, q_attention_mask)
            a_latent = model.encode_text(a_input_ids, a_attention_mask)

            loss = criterion(q_latent, a_latent)

            total_loss += loss.item()
            num_batches += 1

    return {
        "loss": total_loss / num_batches if num_batches > 0 else 0.0,
    }


def train_jepa(
    data_file: Path,
    output_dir: Path = Path("checkpoints"),
    latent_dim: int = 512,
    batch_size: int = 16,
    num_epochs: int = 10,
    learning_rate: float = 1e-4,
    temperature: float = 0.1,
    device: str = "cpu",
    save_every: int = 5,
) -> None:
    """Train JEPA model.

    Args:
        data_file: Path to JSONL data file
        output_dir: Output directory for checkpoints
        latent_dim: Latent dimension
        batch_size: Batch size
        num_epochs: Number of epochs
        learning_rate: Learning rate
        temperature: Temperature for contrastive loss
        device: Device to use
        save_every: Save checkpoint every N epochs
    """
    print(f"🚀 Training JEPA text encoder")
    print(f"Data: {data_file}")
    print(f"Latent dim: {latent_dim}")
    print(f"Batch size: {batch_size}")
    print(f"Epochs: {num_epochs}")
    print(f"Learning rate: {learning_rate}")
    print(f"Device: {device}")
    print("-" * 60)

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create tokenizer and dataset
    tokenizer = SimpleTokenizer()
    dataset = SyntheticTextDataset(data_file, tokenizer)

    # Split into train/val
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    print(f"\n📊 Dataset:")
    print(f"  Total samples: {len(dataset)}")
    print(f"  Train: {len(train_dataset)}")
    print(f"  Val: {len(val_dataset)}")

    # Create model (text-only)
    model = create_jepa_lite(latent_dim=latent_dim)
    # Set vision encoder to None for text-only mode
    model.vision_encoder = None
    model.text_only = True
    model = model.to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\n🔧 Model:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")

    # Create optimizer and loss function
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    criterion = ContrastiveLoss(temperature=temperature)

    # Training loop
    best_val_loss = float("inf")
    training_history = []

    print(f"\n🏃 Training...")

    for epoch in range(1, num_epochs + 1):
        # Train
        train_metrics = train_epoch(
            model, train_loader, optimizer, criterion, device
        )

        # Evaluate
        val_metrics = evaluate(model, val_loader, criterion, device)

        # Log
        print(
            f"Epoch {epoch}/{num_epochs}: "
            f"train_loss={train_metrics['loss']:.4f}, "
            f"val_loss={val_metrics['loss']:.4f}"
        )

        # Save history
        training_history.append(
            {
                "epoch": epoch,
                "train_loss": train_metrics["loss"],
                "val_loss": val_metrics["loss"],
                "timestamp": datetime.now().isoformat(),
            }
        )

        # Save checkpoint if best
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            checkpoint_path = output_dir / "jepa_best.pth"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_metrics["loss"],
                    "latent_dim": latent_dim,
                },
                checkpoint_path,
            )
            print(f"  💾 Saved best checkpoint: {checkpoint_path}")

        # Save periodic checkpoint
        if epoch % save_every == 0:
            checkpoint_path = output_dir / f"jepa_epoch_{epoch}.pth"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_metrics["loss"],
                    "latent_dim": latent_dim,
                },
                checkpoint_path,
            )
            print(f"  💾 Saved checkpoint: {checkpoint_path}")

    # Save final checkpoint
    final_checkpoint = output_dir / "jepa_final.pth"
    torch.save(
        {
            "epoch": num_epochs,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_loss": val_metrics["loss"],
            "latent_dim": latent_dim,
        },
        final_checkpoint,
    )

    print(f"\n✅ Training complete!")
    print(f"Best val loss: {best_val_loss:.4f}")
    print(f"Final checkpoint: {final_checkpoint}")

    # Save training history
    history_file = output_dir / "training_history.json"
    with open(history_file, "w") as f:
        json.dump(training_history, f, indent=2)

    print(f"Training history: {history_file}")


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Train JEPA text encoder")
    parser.add_argument(
        "--data-file",
        type=Path,
        required=True,
        help="Path to JSONL data file",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("checkpoints"),
        help="Output directory for checkpoints",
    )
    parser.add_argument(
        "--latent-dim",
        type=int,
        default=512,
        help="Latent dimension",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size",
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=10,
        help="Number of epochs",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-4,
        help="Learning rate",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.1,
        help="Temperature for contrastive loss",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to use (cpu/cuda)",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=5,
        help="Save checkpoint every N epochs",
    )

    args = parser.parse_args()

    train_jepa(
        data_file=args.data_file,
        output_dir=args.output_dir,
        latent_dim=args.latent_dim,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        temperature=args.temperature,
        device=args.device,
        save_every=args.save_every,
    )


if __name__ == "__main__":
    main()
