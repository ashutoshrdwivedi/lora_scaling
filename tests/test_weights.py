"""Tests for AdapterStore and BatchAssembler."""

import torch
import pytest
from lora_serving.config import LoraServingConfig
from lora_serving.weights.store import AdapterStore, LoraWeight
from lora_serving.weights.batch import BatchAssembler


@pytest.fixture
def config():
    return LoraServingConfig(
        model_name="intfloat/multilingual-e5-small",
        lora_rank=8,
        batch_size=4,
        max_seq_len=16,
        target_modules=["query", "value"],
        device=torch.device("cpu"),
        dtype=torch.float32,
    )


class TestAdapterStore:
    def test_load_synthetic(self, config):
        store = AdapterStore(config)
        store.load_synthetic("adapter_0", seed=0)
        assert len(store) == 1

    def test_get_returns_correct_shapes(self, config):
        store = AdapterStore(config)
        store.load_synthetic("adapter_0")
        w = store.get("adapter_0")
        L, H, R = config.num_layers, config.hidden_size, config.lora_rank
        assert w.wa.shape == (L, H, R)
        assert w.wb.shape == (L, R, H)

    def test_get_unknown_raises(self, config):
        store = AdapterStore(config)
        with pytest.raises(KeyError, match="not found"):
            store.get("nonexistent")

    def test_memory_bytes(self, config):
        store = AdapterStore(config)
        store.load_synthetic("a0")
        store.load_synthetic("a1")
        # Each adapter: 2 tensors (wa, wb) × num_layers × hidden_size × lora_rank × 4 bytes
        L, H, R = config.num_layers, config.hidden_size, config.lora_rank
        expected = 2 * (L * H * R * 4) * 2  # 2 adapters, 2 matrices each
        assert store.memory_bytes() == expected

    def test_multiple_adapters_are_independent(self, config):
        store = AdapterStore(config)
        store.load_synthetic("a0", seed=0)
        store.load_synthetic("a1", seed=1)
        w0 = store.get("a0")
        w1 = store.get("a1")
        # Different seeds should give different weights
        assert not torch.equal(w0.wa, w1.wa)


class TestBatchAssembler:
    def test_assemble_shapes(self, config):
        store = AdapterStore(config)
        for i in range(4):
            store.load_synthetic(f"adapter_{i}")

        assembler = BatchAssembler(store, config)
        ids = [f"adapter_{i}" for i in range(config.batch_size)]

        H, num_labels = config.hidden_size, 5
        lr_coefs = [torch.randn(1, num_labels, H) for _ in ids]
        lr_intercepts = [torch.zeros(1, num_labels) for _ in ids]

        lora_w, lr_w = assembler.assemble(ids, lr_coefs, lr_intercepts)

        assert len(lora_w) == config.num_layers
        for layer in lora_w:
            for module in config.target_modules:
                # Each list should have batch_size tensors of shape (1, H, R)
                assert len(layer.a[module]) == config.batch_size
                assert len(layer.b[module]) == config.batch_size

        B = config.batch_size
        assert lr_w.coef.shape == (B, num_labels, H)
        assert lr_w.intercept.shape == (B, num_labels)

    def test_assemble_pads_variable_labels(self, config):
        """Batch with different num_labels per sample should be padded to max."""
        store = AdapterStore(config)
        for i in range(2):
            store.load_synthetic(f"adapter_{i}")

        assembler = BatchAssembler(store, config)
        ids = ["adapter_0", "adapter_1"]
        H = config.hidden_size

        # Different label counts
        lr_coefs = [torch.randn(1, 3, H), torch.randn(1, 7, H)]
        lr_intercepts = [torch.zeros(1, 3), torch.zeros(1, 7)]

        _, lr_w = assembler.assemble(ids, lr_coefs, lr_intercepts)

        assert lr_w.coef.shape[1] == 7  # max labels
        assert lr_w.num_labels == [3, 7]
