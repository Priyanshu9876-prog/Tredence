# Tredence Analytics — AI Engineering Intern Case Study

## Problem Title: The Self-Pruning Neural Network

A neural network that **learns to prune itself *during* training** on the CIFAR-10 dataset.

---

## Repository Structure

```
├── prunable_layer.py       # Part 1 – PrunableLinear custom layer
├── train.py                # Parts 2 & 3 – Loss formulation, model, training & evaluation
├── tests/
│   └── test_solution.py    # Unit tests (14 tests, all passing)
└── README.md
```

---

## Part 1 — PrunableLinear Layer (`prunable_layer.py`)

A custom replacement for `torch.nn.Linear` that learns which connections are important.

| Parameter | Shape | Role |
|-----------|-------|------|
| `weight` | `(out, in)` | Standard weight matrix |
| `bias` | `(out,)` | Standard bias |
| `gate_scores` | `(out, in)` | Learnable gating logits |

**Forward pass:**
```
gates          = sigmoid(gate_scores)      # squash to (0, 1)
pruned_weights = weight * gates            # element-wise mask
output         = pruned_weights @ xᵀ + bias
```

- `gate → 0` ≡ weight is **pruned** (multiplied by ≈ 0)
- `gate → 1` ≡ weight is **kept** (full strength)
- Gradients flow through both `weight` and `gate_scores` automatically

---

## Part 2 — Sparsity Regularisation Loss (`train.py`)

```
Total Loss = CrossEntropyLoss + λ × SparsityLoss
SparsityLoss = Σ sigmoid(gate_scores)  (sum over all PrunableLinear layers)
```

The **L1 norm** of gate values drives many gates toward zero:
- Higher `λ` → more aggressive pruning, lower accuracy ceiling
- Lower `λ` → lighter pruning, higher accuracy

---

## Part 3 — Model & Training (`train.py`)

### Architecture

```
CIFAR-10 image (3×32×32)
    ↓ Flatten
    → PrunableLinear(3072, 512) → BatchNorm → ReLU → Dropout
    → PrunableLinear(512,  256) → BatchNorm → ReLU → Dropout
    → PrunableLinear(256,  128) → BatchNorm → ReLU → Dropout
    → PrunableLinear(128,   10)   [logits]
```

### Running

```bash
# Install dependencies
pip install torch torchvision

# Train with default settings (20 epochs, λ=1e-4)
python train.py

# Stronger sparsity pressure
python train.py --lambda_ 1e-3

# Full options
python train.py --epochs 30 --batch_size 256 --lr 5e-4 --lambda_ 1e-4 --dropout 0.3
```

### Command-line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--epochs` | 20 | Training epochs |
| `--batch_size` | 128 | Mini-batch size |
| `--lr` | 1e-3 | Adam learning rate |
| `--lambda_` | 1e-4 | Sparsity regularisation strength |
| `--dropout` | 0.3 | Dropout probability |
| `--save` | model.pt | Checkpoint save path |

---

## Tests

```bash
python -m pytest tests/ -v
# 14 passed
```

Tests cover:
- Output shapes
- `gate_scores` registered as a model parameter with same shape as `weight`
- Gates are always in `[0, 1]`
- Gradient flow through both `weight` and `gate_scores`
- Gate = 0 → weight contribution vanishes
- Gate = 1 → output equals standard linear
- Sparsity loss is non-negative
- Full backward pass on combined loss

---

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| Sigmoid activation on gate scores | Smooth, differentiable squash to `(0,1)`; gradients flow everywhere |
| L1 norm as sparsity loss | L1 penalty is known to produce exact zeros (unlike L2) |
| Gate scores initialised at 0 | `sigmoid(0) = 0.5`; neither pruned nor fully active at start |
| Adam + CosineAnnealingLR | Adam handles the dual objective well; cosine schedule improves convergence |
| BatchNorm between prunable layers | Stabilises training when many weights are zeroed out mid-epoch |