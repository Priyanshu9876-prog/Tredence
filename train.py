"""
Part 2 – Sparsity Regularisation Loss
Part 3 – Model Definition, Training & Evaluation on CIFAR-10

Usage
-----
    python train.py                    # default settings
    python train.py --lambda_ 1e-3    # stronger sparsity pressure
    python train.py --epochs 30 --batch_size 256

Loss formulation
----------------
    Total Loss = CrossEntropyLoss + λ * SparsityLoss
    SparsityLoss = sum of all gate values (L1 norm of sigmoid(gate_scores))
                  across every PrunableLinear layer in the network.
"""

import argparse
import time
from typing import List

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

from prunable_layer import PrunableLinear


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class SelfPruningNet(nn.Module):
    """
    Feed-forward network for CIFAR-10 image classification.

    Architecture:
        Flatten → PrunableLinear(3072, 512) → BN → ReLU → Dropout
                → PrunableLinear(512,  256) → BN → ReLU → Dropout
                → PrunableLinear(256,  128) → BN → ReLU → Dropout
                → PrunableLinear(128,   10)
    """

    def __init__(self, dropout: float = 0.3) -> None:
        super().__init__()

        self.flatten = nn.Flatten()

        self.layers = nn.Sequential(
            PrunableLinear(3 * 32 * 32, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),

            PrunableLinear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),

            PrunableLinear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),

            PrunableLinear(128, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(self.flatten(x))

    def prunable_layers(self) -> List[PrunableLinear]:
        """Return all PrunableLinear sub-modules."""
        return [m for m in self.modules() if isinstance(m, PrunableLinear)]

    def sparsity_loss(self) -> torch.Tensor:
        """
        Part 2 – SparsityLoss: sum of all gate values (L1 norm of gates)
        across every PrunableLinear layer.
        """
        return sum(layer.sparsity_loss() for layer in self.prunable_layers())

    def print_pruning_stats(self) -> None:
        """Print per-layer fraction of pruned connections."""
        total_weights = 0
        total_pruned = 0
        for i, layer in enumerate(self.prunable_layers()):
            n = layer.weight.numel()
            pruned = layer.pruned_fraction() * n
            total_weights += n
            total_pruned += pruned
            print(
                f"  Layer {i + 1} ({layer.in_features}→{layer.out_features}): "
                f"{pruned / n * 100:.1f}% pruned"
            )
        print(
            f"  Overall: {total_pruned / total_weights * 100:.1f}% of connections pruned"
        )


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def get_dataloaders(batch_size: int, num_workers: int = 2):
    """Return CIFAR-10 train and test DataLoaders with standard augmentation."""

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])

    train_dataset = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=train_transform
    )
    test_dataset = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=test_transform
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    return train_loader, test_loader


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_one_epoch(
    model: SelfPruningNet,
    loader: torch.utils.data.DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    lambda_: float,
    device: torch.device,
) -> dict:
    """Run one epoch; return a dict with average losses and accuracy."""
    model.train()

    total_cls_loss = 0.0
    total_spar_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()

        logits = model(images)

        # Part 2 – combined loss
        cls_loss = criterion(logits, labels)
        spar_loss = model.sparsity_loss()
        loss = cls_loss + lambda_ * spar_loss

        loss.backward()
        optimizer.step()

        total_cls_loss += cls_loss.item()
        total_spar_loss += spar_loss.item()
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)

    n = len(loader)
    return {
        "cls_loss": total_cls_loss / n,
        "spar_loss": total_spar_loss / n,
        "train_acc": correct / total * 100,
    }


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(
    model: SelfPruningNet,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> dict:
    """Evaluate on the test set; return loss and accuracy."""
    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        total_loss += criterion(logits, labels).item()
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)

    return {
        "test_loss": total_loss / len(loader),
        "test_acc": correct / total * 100,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Self-Pruning Neural Network on CIFAR-10")
    parser.add_argument("--epochs",      type=int,   default=20,   help="Training epochs")
    parser.add_argument("--batch_size",  type=int,   default=128,  help="Mini-batch size")
    parser.add_argument("--lr",          type=float, default=1e-3, help="Learning rate (Adam)")
    parser.add_argument("--lambda_",     type=float, default=1e-4, help="Sparsity regularisation λ")
    parser.add_argument("--dropout",     type=float, default=0.3,  help="Dropout probability")
    parser.add_argument("--num_workers", type=int,   default=2,    help="DataLoader workers")
    parser.add_argument("--save",        type=str,   default="model.pt", help="Checkpoint path")
    return parser.parse_args()


def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")
    print(f"Config : epochs={args.epochs}, batch_size={args.batch_size}, "
          f"lr={args.lr}, lambda={args.lambda_}, dropout={args.dropout}\n")

    # Data
    train_loader, test_loader = get_dataloaders(args.batch_size, args.num_workers)

    # Model
    model = SelfPruningNet(dropout=args.dropout).to(device)
    print(model)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {total_params:,}\n")

    # Optimiser & scheduler
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0
    header = f"{'Epoch':>6} {'Train Acc':>10} {'Test Acc':>9} {'CLS Loss':>9} {'Spar Loss':>10} {'Time':>7}"
    print(header)
    print("-" * len(header))

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        train_stats = train_one_epoch(
            model, train_loader, optimizer, criterion, args.lambda_, device
        )
        test_stats = evaluate(model, test_loader, criterion, device)
        scheduler.step()

        elapsed = time.time() - t0
        print(
            f"{epoch:>6} "
            f"{train_stats['train_acc']:>9.2f}% "
            f"{test_stats['test_acc']:>8.2f}% "
            f"{train_stats['cls_loss']:>9.4f} "
            f"{train_stats['spar_loss']:>10.1f} "
            f"{elapsed:>6.1f}s"
        )

        if test_stats["test_acc"] > best_acc:
            best_acc = test_stats["test_acc"]
            torch.save(model.state_dict(), args.save)

    print(f"\nBest test accuracy: {best_acc:.2f}%")
    print(f"Checkpoint saved to: {args.save}\n")

    # Load best checkpoint and report final pruning stats
    model.load_state_dict(torch.load(args.save, map_location=device))
    print("Pruning statistics (gate < 0.5 → pruned):")
    model.print_pruning_stats()


if __name__ == "__main__":
    main()
