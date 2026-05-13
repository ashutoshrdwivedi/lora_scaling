"""Numerical equivalence against HuggingFace PEFT.

For the same base encoder and the same A/B matrices, our batched multi-tenant
LoRA must produce the same pooled output as `peft.get_peft_model(...)` applied
to the same HuggingFace BertModel.

Three scenarios:
  1. Zero LoRA → matches the raw base HF model (no PEFT wrap).
  2. Single shared adapter across the batch → matches PEFT batched forward.
  3. Different adapter per sample → matches a per-sample PEFT loop with
     set_adapter().

Skipped if either CUDA or peft is unavailable.
"""

from __future__ import annotations

import pytest
import torch
from torch import Tensor
from transformers import AutoModel

from lora_serving.config import LoraServingConfig
from lora_serving.model.encoder import EncoderWithLora
from lora_serving.weights.batch import LayerwiseBatchedWeights
from lora_serving.weights.store import AdapterStore, LoraWeight

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="PEFT equivalence tests require CUDA",
)

peft = pytest.importorskip("peft", reason="peft not installed (install with `uv add --dev peft`)")
from peft import LoraConfig, get_peft_model  # noqa: E402


MODEL_NAME = "intfloat/multilingual-e5-small"
LORA_RANK = 8
TARGET_MODULES = ["query", "value"]
BATCH_SIZE = 4
SEQ_LEN = 16
ATOL = 1e-4
RTOL = 1e-4


# ---------- fixtures ----------

@pytest.fixture(scope="module")
def serving_config() -> LoraServingConfig:
    return LoraServingConfig(
        model_name=MODEL_NAME,
        lora_rank=LORA_RANK,
        batch_size=BATCH_SIZE,
        max_seq_len=SEQ_LEN,
        target_modules=TARGET_MODULES,
        device=torch.device("cuda:0"),
        dtype=torch.float32,
    )


@pytest.fixture(scope="module")
def our_model(serving_config) -> EncoderWithLora:
    m = EncoderWithLora.from_pretrained_serving(serving_config)
    m.eval()
    return m


@pytest.fixture(scope="module")
def base_hf_model(serving_config) -> torch.nn.Module:
    """Raw HF BertModel — same weights our model loaded from."""
    m = AutoModel.from_pretrained(MODEL_NAME, torch_dtype=serving_config.dtype)
    m.to(serving_config.device)
    m.eval()
    return m


@pytest.fixture(scope="module")
def batch_inputs(serving_config) -> tuple[Tensor, Tensor]:
    torch.manual_seed(0)
    input_ids = torch.randint(
        1, 30000, (BATCH_SIZE, SEQ_LEN), device=serving_config.device
    )
    attention_mask = torch.ones(
        BATCH_SIZE, SEQ_LEN, dtype=torch.long, device=serving_config.device
    )
    return input_ids, attention_mask


# ---------- helpers ----------

def _encode_pooled(
    model: EncoderWithLora,
    input_ids: Tensor,
    attention_mask: Tensor,
    lora_weights: list,
) -> Tensor:
    """Run our encoder up to the pooled CLS output (skip the LR head)."""
    B, S = input_ids.shape
    device = input_ids.device
    position_ids = torch.arange(S, device=device).unsqueeze(0).expand(B, -1)
    token_type_ids = torch.zeros(B, S, dtype=torch.long, device=device)

    x = (
        model.embeddings["word_embeddings"](input_ids)
        + model.embeddings["position_embeddings"](position_ids)
        + model.embeddings["token_type_embeddings"](token_type_ids)
    )
    x = model.embeddings["LayerNorm"](x)
    for layer, layer_lora in zip(model.encoder["layer"], lora_weights):
        x = layer(x, attention_mask, layer_lora)
    return model.pooler_act(model.pooler["dense"](x[:, 0]))


def _build_peft_model(base_state_dict_source: str, serving_config: LoraServingConfig):
    """Fresh HF BertModel wrapped with a PEFT LoRA adapter.

    lora_alpha = lora_rank so scaling = alpha/r = 1.0 — matches our impl, which
    has no scaling factor.
    """
    base = AutoModel.from_pretrained(base_state_dict_source, torch_dtype=serving_config.dtype)
    lora_cfg = LoraConfig(
        r=LORA_RANK,
        lora_alpha=LORA_RANK,
        target_modules=TARGET_MODULES,
        lora_dropout=0.0,
        bias="none",
    )
    peft_model = get_peft_model(base, lora_cfg)
    peft_model.to(serving_config.device)
    peft_model.eval()
    return peft_model


def _random_ab(
    serving_config: LoraServingConfig,
    seed: int,
) -> tuple[Tensor, Tensor]:
    """Random A (R, H) and B (H, R) per layer, with non-zero B so delta != 0.

    Returns:
        a_layers: (num_layers, R, H)  — PEFT-style "lora_A.weight" shape
        b_layers: (num_layers, H, R)  — PEFT-style "lora_B.weight" shape
    """
    g = torch.Generator(device="cpu").manual_seed(seed)
    H, R, L = serving_config.hidden_size, serving_config.lora_rank, serving_config.num_layers
    a = torch.randn(L, R, H, generator=g, dtype=serving_config.dtype) * 0.02
    b = torch.randn(L, H, R, generator=g, dtype=serving_config.dtype) * 0.02
    return a.to(serving_config.device), b.to(serving_config.device)


def _install_into_peft(peft_model, a_layers: Tensor, b_layers: Tensor) -> None:
    """Copy A/B weights into PEFT's lora_A.default / lora_B.default modules.

    PEFT stores `lora_A.<name>.weight` of shape (R, in_features) and
    `lora_B.<name>.weight` of shape (out_features, R). We have one (A, B) pair
    per layer that's shared across all TARGET_MODULES on that layer.
    """
    encoder_layers = peft_model.base_model.model.encoder.layer
    for i, layer in enumerate(encoder_layers):
        for module_name in TARGET_MODULES:
            target = getattr(layer.attention.self, module_name)
            target.lora_A["default"].weight.data.copy_(a_layers[i])
            target.lora_B["default"].weight.data.copy_(b_layers[i])


def _install_into_our_store(
    store: AdapterStore,
    adapter_id: str,
    a_layers: Tensor,
    b_layers: Tensor,
    serving_config: LoraServingConfig,
) -> None:
    """Mirror the PEFT A/B into our LoraWeight (transposed shapes)."""
    weight = LoraWeight(serving_config)
    # PEFT A is (R, H); ours is (H, R)  → transpose
    weight.wa.copy_(a_layers.transpose(1, 2))
    # PEFT B is (H, R); ours is (R, H)  → transpose
    weight.wb.copy_(b_layers.transpose(1, 2))
    store._store[adapter_id] = weight


def _assemble_lora_only(
    store: AdapterStore,
    adapter_ids: list[str],
    serving_config: LoraServingConfig,
) -> list[LayerwiseBatchedWeights]:
    """Build LayerwiseBatchedWeights for a batch, skipping LR-head assembly.

    BatchAssembler requires LR weights to compute max_labels; this helper exists
    so equivalence tests don't have to fabricate LR head data.
    """
    cfg = serving_config
    lora_weights = [LayerwiseBatchedWeights() for _ in range(cfg.num_layers)]
    for module in cfg.target_modules:
        for layer_idx in range(cfg.num_layers):
            lora_weights[layer_idx].a[module] = []
            lora_weights[layer_idx].b[module] = []
    for adapter_id in adapter_ids:
        weight = store.get(adapter_id)
        for layer_idx in range(cfg.num_layers):
            for module in cfg.target_modules:
                lora_weights[layer_idx].a[module].append(weight.wa[layer_idx].unsqueeze(0))
                lora_weights[layer_idx].b[module].append(weight.wb[layer_idx].unsqueeze(0))
    return lora_weights


# ---------- tests ----------

def test_zero_lora_matches_base_hf(serving_config, our_model, base_hf_model, batch_inputs):
    """With zero LoRA delta, our pooled output equals raw HF BertModel's pooler_output."""
    input_ids, attention_mask = batch_inputs

    # Zero adapter: wa random doesn't matter when wb is zero
    store = AdapterStore(serving_config)
    store.load_synthetic("zero", seed=0)  # load_synthetic uses wb=0 by default
    ids = ["zero"] * BATCH_SIZE
    lora_w = _assemble_lora_only(store, ids, serving_config)

    with torch.no_grad():
        our_pooled = _encode_pooled(our_model, input_ids, attention_mask, lora_w)
        hf_out = base_hf_model(input_ids=input_ids, attention_mask=attention_mask)

    assert torch.allclose(our_pooled, hf_out.pooler_output, atol=ATOL, rtol=RTOL), (
        f"max abs diff = {(our_pooled - hf_out.pooler_output).abs().max().item()}"
    )


def test_single_adapter_matches_peft(serving_config, our_model, batch_inputs):
    """Same adapter applied to every sample in the batch — match PEFT batched forward."""
    input_ids, attention_mask = batch_inputs

    a_layers, b_layers = _random_ab(serving_config, seed=42)

    # Install into PEFT
    peft_model = _build_peft_model(MODEL_NAME, serving_config)
    _install_into_peft(peft_model, a_layers, b_layers)

    # Install into our store
    store = AdapterStore(serving_config)
    _install_into_our_store(store, "adapter_0", a_layers, b_layers, serving_config)
    ids = ["adapter_0"] * BATCH_SIZE
    lora_w = _assemble_lora_only(store, ids, serving_config)

    with torch.no_grad():
        our_pooled = _encode_pooled(our_model, input_ids, attention_mask, lora_w)
        peft_out = peft_model(input_ids=input_ids, attention_mask=attention_mask)

    assert torch.allclose(our_pooled, peft_out.pooler_output, atol=ATOL, rtol=RTOL), (
        f"max abs diff = {(our_pooled - peft_out.pooler_output).abs().max().item()}"
    )


def test_multi_tenant_matches_peft_loop(serving_config, our_model, batch_inputs):
    """Each sample uses a different adapter; PEFT runs sample-by-sample, we run batched."""
    input_ids, attention_mask = batch_inputs

    # One distinct adapter per sample
    adapter_weights = [_random_ab(serving_config, seed=100 + i) for i in range(BATCH_SIZE)]

    # Build N PEFT models (one per adapter); run each sample through its own
    expected_pooled = []
    for i, (a, b) in enumerate(adapter_weights):
        peft_model = _build_peft_model(MODEL_NAME, serving_config)
        _install_into_peft(peft_model, a, b)
        with torch.no_grad():
            out = peft_model(
                input_ids=input_ids[i : i + 1],
                attention_mask=attention_mask[i : i + 1],
            )
        expected_pooled.append(out.pooler_output)
        del peft_model
        torch.cuda.empty_cache()
    expected = torch.cat(expected_pooled, dim=0)  # (B, H)

    # Our model: batched mixed-tenant forward
    store = AdapterStore(serving_config)
    for i, (a, b) in enumerate(adapter_weights):
        _install_into_our_store(store, f"adapter_{i}", a, b, serving_config)
    ids = [f"adapter_{i}" for i in range(BATCH_SIZE)]
    lora_w = _assemble_lora_only(store, ids, serving_config)

    with torch.no_grad():
        our_pooled = _encode_pooled(our_model, input_ids, attention_mask, lora_w)

    assert torch.allclose(our_pooled, expected, atol=ATOL, rtol=RTOL), (
        f"max abs diff = {(our_pooled - expected).abs().max().item()}"
    )
