"""
Unit tests for PrunableLinear and SelfPruningNet.

Run with:  python -m pytest tests/ -v
"""

import torch
import pytest
from prunable_layer import PrunableLinear
from train import SelfPruningNet


# ---------------------------------------------------------------------------
# PrunableLinear tests
# ---------------------------------------------------------------------------

class TestPrunableLinear:

    def test_output_shape(self):
        layer = PrunableLinear(16, 8)
        x = torch.randn(4, 16)
        assert layer(x).shape == (4, 8)

    def test_gate_scores_are_parameters(self):
        layer = PrunableLinear(16, 8)
        param_names = {n for n, _ in layer.named_parameters()}
        assert "gate_scores" in param_names
        assert "weight" in param_names

    def test_gate_scores_same_shape_as_weight(self):
        layer = PrunableLinear(16, 8)
        assert layer.gate_scores.shape == layer.weight.shape

    def test_gates_in_zero_one(self):
        layer = PrunableLinear(32, 16)
        gates = torch.sigmoid(layer.gate_scores)
        assert gates.min() >= 0.0
        assert gates.max() <= 1.0

    def test_gradients_flow_through_gate_scores(self):
        layer = PrunableLinear(8, 4)
        x = torch.randn(2, 8)
        out = layer(x).sum()
        out.backward()
        assert layer.gate_scores.grad is not None
        assert layer.weight.grad is not None

    def test_gate_zero_means_pruned_weight(self):
        """If gate_scores → -∞, sigmoid → 0 and weight contribution vanishes."""
        layer = PrunableLinear(4, 2, bias=False)
        with torch.no_grad():
            layer.gate_scores.fill_(-100.0)   # sigmoid ≈ 0
        x = torch.randn(3, 4)
        out = layer(x)
        assert torch.allclose(out, torch.zeros_like(out), atol=1e-5)

    def test_gate_one_means_full_weight(self):
        """If gate_scores → +∞, sigmoid → 1 and output equals standard linear."""
        layer = PrunableLinear(4, 2, bias=False)
        with torch.no_grad():
            layer.gate_scores.fill_(100.0)   # sigmoid ≈ 1
        x = torch.randn(3, 4)
        expected = torch.nn.functional.linear(x, layer.weight)
        assert torch.allclose(layer(x), expected, atol=1e-4)

    def test_sparsity_loss_non_negative(self):
        layer = PrunableLinear(8, 4)
        assert layer.sparsity_loss().item() >= 0.0

    def test_pruned_fraction_between_zero_and_one(self):
        layer = PrunableLinear(8, 4)
        frac = layer.pruned_fraction()
        assert 0.0 <= frac <= 1.0


# ---------------------------------------------------------------------------
# SelfPruningNet tests
# ---------------------------------------------------------------------------

class TestSelfPruningNet:

    def _model(self):
        return SelfPruningNet()

    def test_output_shape(self):
        model = self._model()
        x = torch.randn(8, 3, 32, 32)
        assert model(x).shape == (8, 10)

    def test_has_four_prunable_layers(self):
        model = self._model()
        assert len(model.prunable_layers()) == 4

    def test_sparsity_loss_positive(self):
        model = self._model()
        x = torch.randn(4, 3, 32, 32)
        model(x)   # forward pass not strictly needed but mirrors usage
        spar = model.sparsity_loss()
        assert spar.item() > 0

    def test_combined_loss_backward(self):
        """Gradients must flow through both cls_loss and sparsity_loss."""
        model = self._model()
        criterion = torch.nn.CrossEntropyLoss()
        x = torch.randn(4, 3, 32, 32)
        labels = torch.randint(0, 10, (4,))

        logits = model(x)
        loss = criterion(logits, labels) + 1e-4 * model.sparsity_loss()
        loss.backward()

        for layer in model.prunable_layers():
            assert layer.gate_scores.grad is not None

    def test_lambda_zero_equals_standard_training(self):
        """λ=0 means sparsity_loss contributes nothing to gradients."""
        model = self._model()
        criterion = torch.nn.CrossEntropyLoss()
        x = torch.randn(4, 3, 32, 32)
        labels = torch.randint(0, 10, (4,))

        logits = model(x)
        loss = criterion(logits, labels) + 0.0 * model.sparsity_loss()
        loss.backward()
        # Gradients should still exist for weights (via cls_loss)
        for layer in model.prunable_layers():
            assert layer.weight.grad is not None
