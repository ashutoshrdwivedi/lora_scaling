from __future__ import annotations

from dataclasses import dataclass, field

import torch
from transformers import AutoConfig


@dataclass
class LoraServingConfig:
    """All configuration needed to run LoRA serving. Pass this to every component.

    Dimensions are derived automatically from the HuggingFace model config —
    nothing is hardcoded.

    Example:
        config = LoraServingConfig(
            model_name="intfloat/multilingual-e5-small",
            lora_rank=8,
            batch_size=32,
            max_seq_len=512,
            target_modules=["query", "value"],
            device=torch.device("cuda:0"),
            dtype=torch.float32,
        )
    """

    model_name: str
    lora_rank: int
    batch_size: int
    max_seq_len: int
    target_modules: list[str]  # projection names to apply LoRA to, e.g. ["query", "value"]
    device: torch.device
    dtype: torch.dtype

    # Derived from model config — do not set manually
    hidden_size: int = field(init=False)
    num_layers: int = field(init=False)
    num_heads: int = field(init=False)
    head_dim: int = field(init=False)
    intermediate_size: int = field(init=False)
    layer_norm_eps: float = field(init=False)
    hidden_act: str = field(init=False)
    model_type: str = field(init=False)
    embedding_size: int = field(init=False)
    type_vocab_size: int = field(init=False)
    relative_attention: bool = field(init=False)
    max_relative_positions: int = field(init=False)
    position_buckets: int = field(init=False)
    pos_att_type: list[str] = field(init=False)
    share_att_key: bool = field(init=False)
    position_biased_input: bool = field(init=False)
    norm_rel_ebd: str = field(init=False)

    def __post_init__(self):
        hf = AutoConfig.from_pretrained(self.model_name)
        self.model_type = hf.model_type
        self.hidden_size = hf.hidden_size
        self.num_layers = hf.num_hidden_layers
        self.num_heads = hf.num_attention_heads
        self.head_dim = hf.hidden_size // hf.num_attention_heads
        self.intermediate_size = hf.intermediate_size
        self.layer_norm_eps = hf.layer_norm_eps
        self.hidden_act = getattr(hf, "hidden_act", "gelu")
        self.embedding_size = getattr(hf, "embedding_size", hf.hidden_size)
        self.type_vocab_size = getattr(hf, "type_vocab_size", 0)
        self.relative_attention = getattr(hf, "relative_attention", False)
        self.max_relative_positions = getattr(hf, "max_relative_positions", -1)
        if self.max_relative_positions < 1:
            self.max_relative_positions = getattr(hf, "max_position_embeddings", 0)
        self.position_buckets = getattr(hf, "position_buckets", -1)
        pos_att_type = getattr(hf, "pos_att_type", None)
        self.pos_att_type = pos_att_type if pos_att_type is not None else []
        self.share_att_key = getattr(hf, "share_att_key", False)
        self.position_biased_input = getattr(hf, "position_biased_input", True)
        self.norm_rel_ebd = getattr(hf, "norm_rel_ebd", "none")
