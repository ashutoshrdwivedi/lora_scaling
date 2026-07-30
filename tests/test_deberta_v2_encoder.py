from __future__ import annotations

import torch
from transformers import DebertaV2Config, DebertaV2Model

from lora_serving.config import LoraServingConfig
from lora_serving.model.deberta_v2 import DebertaV2EncoderWithLora
from lora_serving.weights.batch import LayerwiseBatchedWeights


def _tiny_deberta_config() -> DebertaV2Config:
    return DebertaV2Config(
        vocab_size=99,
        hidden_size=16,
        embedding_size=16,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=32,
        max_position_embeddings=32,
        relative_attention=True,
        pos_att_type=["c2p", "p2c"],
        type_vocab_size=2,
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
        conv_kernel_size=3,
    )


def _serving_config(hf_config: DebertaV2Config) -> LoraServingConfig:
    cfg = object.__new__(LoraServingConfig)
    cfg.model_name = "local-deberta-v2"
    cfg.lora_rank = 2
    cfg.batch_size = 2
    cfg.max_seq_len = 8
    cfg.target_modules = ["query_proj", "value_proj"]
    cfg.device = torch.device("cpu")
    cfg.dtype = torch.float32
    cfg.model_type = "deberta-v2"
    cfg.hidden_size = hf_config.hidden_size
    cfg.num_layers = hf_config.num_hidden_layers
    cfg.num_heads = hf_config.num_attention_heads
    cfg.head_dim = hf_config.hidden_size // hf_config.num_attention_heads
    cfg.intermediate_size = hf_config.intermediate_size
    cfg.layer_norm_eps = hf_config.layer_norm_eps
    cfg.hidden_act = hf_config.hidden_act
    cfg.embedding_size = hf_config.embedding_size
    cfg.type_vocab_size = hf_config.type_vocab_size
    cfg.relative_attention = hf_config.relative_attention
    cfg.max_relative_positions = hf_config.max_relative_positions
    if cfg.max_relative_positions < 1:
        cfg.max_relative_positions = hf_config.max_position_embeddings
    cfg.position_buckets = getattr(hf_config, "position_buckets", -1)
    cfg.pos_att_type = hf_config.pos_att_type if hf_config.pos_att_type is not None else []
    cfg.share_att_key = getattr(hf_config, "share_att_key", False)
    cfg.position_biased_input = getattr(hf_config, "position_biased_input", True)
    cfg.norm_rel_ebd = getattr(hf_config, "norm_rel_ebd", "none")
    return cfg


def test_deberta_v2_strictly_loads_hf_state_dict():
    hf_config = _tiny_deberta_config()
    serving_config = _serving_config(hf_config)

    hf_model = DebertaV2Model(hf_config)
    our_model = DebertaV2EncoderWithLora(hf_config, serving_config)

    missing, unexpected = our_model.load_state_dict(hf_model.state_dict(), strict=False)
    assert missing == []
    assert unexpected == []
    our_model.load_state_dict(hf_model.state_dict(), strict=True)


def test_deberta_v2_base_forward_matches_hf():
    torch.manual_seed(0)
    hf_config = _tiny_deberta_config()
    serving_config = _serving_config(hf_config)

    hf_model = DebertaV2Model(hf_config)
    our_model = DebertaV2EncoderWithLora(hf_config, serving_config)
    our_model.load_state_dict(hf_model.state_dict(), strict=True)
    hf_model.eval()
    our_model.eval()

    input_ids = torch.randint(1, hf_config.vocab_size, (serving_config.batch_size, serving_config.max_seq_len))
    input_ids[1, -2:] = 0
    attention_mask = input_ids.ne(0).long()
    token_type_ids = torch.zeros_like(input_ids)
    lora_weights = [LayerwiseBatchedWeights() for _ in range(serving_config.num_layers)]

    with torch.no_grad():
        embedding_output = our_model.embeddings(input_ids, attention_mask, token_type_ids)
        our_output = our_model.encoder(embedding_output, attention_mask, lora_weights, apply_lora=False)
        hf_output = hf_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        ).last_hidden_state

    assert torch.allclose(our_output, hf_output, atol=1e-5, rtol=1e-5)
