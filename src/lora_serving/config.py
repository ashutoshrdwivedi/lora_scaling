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
    vocab_size: int = field(init=False)
    pad_token_id: int = field(init=False)
    type_vocab_size: int = field(init=False)
    max_position_embeddings: int = field(init=False)
    model_type: str = field(init=False)

    def __post_init__(self):
        hf = AutoConfig.from_pretrained(self.model_name)
        self.model_type = hf.model_type
        self.hidden_size = getattr(hf, "hidden_size", getattr(hf, "d_model", None))
        self.num_layers = getattr(hf, "num_hidden_layers", getattr(hf, "num_layers", None))
        self.num_heads = getattr(hf, "num_attention_heads", getattr(hf, "num_heads", None))
        self.intermediate_size = getattr(hf, "intermediate_size", getattr(hf, "d_ff", None))
        self.layer_norm_eps = getattr(hf, "layer_norm_eps", getattr(hf, "layer_norm_epsilon", 1e-12))
        self.hidden_act = getattr(hf, "hidden_act", getattr(hf, "hidden_activation", "gelu"))
        self.vocab_size = hf.vocab_size
        self.pad_token_id = hf.pad_token_id or 0
        self.type_vocab_size = getattr(hf, "type_vocab_size", 1)
        self.max_position_embeddings = getattr(hf, "max_position_embeddings", self.max_seq_len)

        missing = [
            name
            for name in ("hidden_size", "num_layers", "num_heads", "intermediate_size")
            if getattr(self, name) is None
        ]
        if missing:
            raise ValueError(f"Could not derive {missing} from HuggingFace config for {self.model_name}")

        self.head_dim = self.hidden_size // self.num_heads
