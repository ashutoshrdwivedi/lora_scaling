"""Numerical validation of HFEncoderWithLora (hook-injected LoRA on stock HF models).

Uses tiny randomly-initialised checkpoints written to tmp_path, so every test
runs offline on CPU (and on GPU when available). Coverage:

  1. PEFT parity, BERT: hook-injected deltas match `peft.get_peft_model` on the
     same base weights and the same A/B matrices.
  2. PEFT parity, DeBERTa-v2 with relative attention + conv: same check on the
     disentangled-attention architecture (share_att_key=False so PEFT and the
     hooks target exactly the same content projections).
  3. Per-sample isolation, DeBERTa-v2 with share_att_key=True (the
     deberta-v2-xlarge configuration): zero-delta samples in a mixed batch
     reproduce the base forward exactly; nonzero samples diverge. Also
     exercises the hook guard that skips the shared relative-position
     projection calls.
  4. EncoderWithLora loader tolerates pooler-less checkpoints (ELECTRA).
"""

from __future__ import annotations

import pytest
import torch
from transformers import (
    AutoModel,
    BertConfig,
    BertModel,
    DebertaV2Config,
    DebertaV2Model,
    ElectraConfig,
    ElectraModel,
)

from lora_serving.config import LoraServingConfig
from lora_serving.model.encoder import EncoderWithLora
from lora_serving.model.hf_wrapper import HFEncoderWithLora
from lora_serving.weights.batch import BatchAssembler
from lora_serving.weights.store import AdapterStore

peft = pytest.importorskip("peft", reason="peft not installed")
from peft import LoraConfig, get_peft_model

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
LORA_RANK = 4
BATCH_SIZE = 4
SEQ_LEN = 16
ATOL = 1e-4
RTOL = 1e-4

TINY = dict(
    vocab_size=100,
    hidden_size=32,
    num_hidden_layers=2,
    num_attention_heads=4,
    intermediate_size=64,
    max_position_embeddings=64,
    pad_token_id=0,
)


def _save_tiny(model_cls, config, tmp_path) -> str:
    torch.manual_seed(0)
    model = model_cls(config)
    model.eval()
    path = str(tmp_path / "model")
    model.save_pretrained(path)
    return path


def _serving_config(model_path: str, target_modules: list[str]) -> LoraServingConfig:
    return LoraServingConfig(
        model_name=model_path,
        lora_rank=LORA_RANK,
        batch_size=BATCH_SIZE,
        max_seq_len=SEQ_LEN,
        target_modules=target_modules,
        device=DEVICE,
        dtype=torch.float32,
    )


def _store_with_adapters(cfg: LoraServingConfig, n: int, nonzero_b: bool = True):
    """AdapterStore with n random adapters. load_synthetic zeroes B; optionally
    re-randomise it so the delta path is actually exercised."""
    store = AdapterStore(cfg)
    for i in range(n):
        store.load_synthetic(f"adapter_{i}", seed=100 + i)
        if nonzero_b:
            w = store.get(f"adapter_{i}")
            torch.manual_seed(200 + i)
            for m in cfg.target_modules:
                torch.nn.init.normal_(w.wb[m], std=0.02)
    return store


def _inputs(cfg: LoraServingConfig):
    torch.manual_seed(7)
    input_ids = torch.randint(1, TINY["vocab_size"], (BATCH_SIZE, SEQ_LEN), device=DEVICE)
    attention_mask = torch.ones(BATCH_SIZE, SEQ_LEN, dtype=torch.long, device=DEVICE)
    return input_ids, attention_mask


def _peft_with_our_weights(model_path: str, cfg: LoraServingConfig, store, adapter_id: str,
                           peft_targets: list[str], layer_prefix: str):
    """PEFT-wrapped base model carrying the same A/B matrices as `adapter_id`.

    Our convention: delta = x @ wa @ wb with wa (H, R), wb (R, H).
    PEFT: delta = lora_B(lora_A(x)) * (alpha/r), lora_A.weight (r, H),
    lora_B.weight (H, r) — so wa = lora_A.weight.T and wb = lora_B.weight.T,
    with alpha = r making the scale 1.
    """
    base = AutoModel.from_pretrained(model_path, torch_dtype=torch.float32)
    lcfg = LoraConfig(
        r=LORA_RANK, lora_alpha=LORA_RANK, target_modules=peft_targets,
        lora_dropout=0.0, bias="none",
    )
    model = get_peft_model(base, lcfg, adapter_name="a")
    weight = store.get(adapter_id)
    sd = dict(model.named_modules())
    for logical, peft_name in zip(cfg.target_modules, peft_targets):
        for layer in range(cfg.num_layers):
            mod = sd[f"base_model.model.encoder.layer.{layer}.{layer_prefix}.{peft_name}"]
            with torch.no_grad():
                mod.lora_A["a"].weight.copy_(weight.wa[logical][layer].T)
                mod.lora_B["a"].weight.copy_(weight.wb[logical][layer].T)
    model.to(DEVICE)
    model.eval()
    return model


def _assert_parity(model_path: str, cfg: LoraServingConfig, peft_targets: list[str],
                   use_pooler_output: bool):
    """Shared-adapter batch: wrapper output must match PEFT batched forward."""
    store = _store_with_adapters(cfg, 1)
    wrapper = HFEncoderWithLora.from_pretrained_serving(cfg)
    assembler = BatchAssembler(store, cfg)
    lora_w = assembler.assemble_lora(["adapter_0"] * BATCH_SIZE)
    input_ids, attention_mask = _inputs(cfg)

    with torch.no_grad():
        ours = wrapper.encode_pooled(input_ids, attention_mask, lora_w)

    peft_model = _peft_with_our_weights(
        model_path, cfg, store, "adapter_0", peft_targets, "attention.self"
    )
    with torch.no_grad():
        out = peft_model(input_ids=input_ids, attention_mask=attention_mask)
    theirs = out.pooler_output if use_pooler_output else out.last_hidden_state[:, 0]

    torch.testing.assert_close(ours, theirs, atol=ATOL, rtol=RTOL)


def test_parity_bert(tmp_path):
    config = BertConfig(**TINY, type_vocab_size=2)
    path = _save_tiny(BertModel, config, tmp_path)
    cfg = _serving_config(path, ["query", "value"])
    _assert_parity(path, cfg, ["query", "value"], use_pooler_output=True)


def test_parity_deberta_v2_disentangled(tmp_path):
    config = DebertaV2Config(
        **TINY,
        type_vocab_size=0,
        relative_attention=True,
        position_buckets=4,
        pos_att_type=["p2c", "c2p"],
        share_att_key=False,
        conv_kernel_size=3,
    )
    path = _save_tiny(DebertaV2Model, config, tmp_path)
    cfg = _serving_config(path, ["query", "value"])
    _assert_parity(path, cfg, ["query_proj", "value_proj"], use_pooler_output=False)


def test_isolation_deberta_v2_shared_att_key(tmp_path):
    """deberta-v2-xlarge-shaped config: mixed batch where only some samples
    carry a nonzero delta. Zero-delta rows must equal the base forward."""
    config = DebertaV2Config(
        **TINY,
        type_vocab_size=0,
        relative_attention=True,
        position_buckets=4,
        pos_att_type=["p2c", "c2p"],
        share_att_key=True,
        conv_kernel_size=3,
    )
    path = _save_tiny(DebertaV2Model, config, tmp_path)
    cfg = _serving_config(path, ["query", "value"])

    store = AdapterStore(cfg)
    store.load_synthetic("zero", seed=1)  # B stays zero → delta = 0
    store.load_synthetic("live", seed=2)
    live = store.get("live")
    torch.manual_seed(3)
    for m in cfg.target_modules:
        torch.nn.init.normal_(live.wb[m], std=0.02)

    wrapper = HFEncoderWithLora.from_pretrained_serving(cfg)
    assembler = BatchAssembler(store, cfg)
    lora_w = assembler.assemble_lora(["live", "zero", "live", "zero"])
    input_ids, attention_mask = _inputs(cfg)

    with torch.no_grad():
        mixed = wrapper.encode_pooled(input_ids, attention_mask, lora_w)
        base = wrapper.encode_pooled(input_ids, attention_mask, lora_w, apply_lora=False)

    # zero-delta samples: exact base output (isolation by construction)
    torch.testing.assert_close(mixed[[1, 3]], base[[1, 3]], atol=ATOL, rtol=RTOL)
    # live samples must actually diverge from base
    assert not torch.allclose(mixed[[0, 2]], base[[0, 2]], atol=ATOL, rtol=RTOL)


def test_electra_loads_without_pooler(tmp_path):
    """ELECTRA checkpoints have no pooler; the strict-except-pooler loader must
    accept them and keep everything else strict."""
    config = ElectraConfig(**TINY, type_vocab_size=2, embedding_size=TINY["hidden_size"])
    path = _save_tiny(ElectraModel, config, tmp_path)
    cfg = LoraServingConfig(
        model_name=path,
        lora_rank=LORA_RANK,
        batch_size=BATCH_SIZE,
        max_seq_len=SEQ_LEN,
        target_modules=["query", "value"],
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    model = EncoderWithLora.from_pretrained_serving(cfg)
    # base weights must have loaded (not stayed at init): compare one tensor
    hf = AutoModel.from_pretrained(path, torch_dtype=torch.float32)
    torch.testing.assert_close(
        model.embeddings["word_embeddings"].weight,
        hf.state_dict()["embeddings.word_embeddings.weight"],
    )
