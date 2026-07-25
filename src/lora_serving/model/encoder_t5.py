"""T5-family encoder with batched multi-tenant LoRA.

T5 is encoder-decoder by design, but its encoder stack is often used as a
standalone encoder. This module implements the T5 encoder block layout while
keeping the LoRA projection computation in the same order as encoder.py:
base projection, shrink with A, expand with B, add the batched delta.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from transformers import AutoConfig, PretrainedConfig, PreTrainedModel, T5EncoderModel

from lora_serving.config import LoraServingConfig
from lora_serving.ops.head import LRHeadOps
from lora_serving.ops.lora import LoraOps
from lora_serving.weights.batch import BatchedLRWeights, LayerwiseBatchedWeights


class T5LayerNorm(nn.Module):
    """T5 layer norm: scale-only RMS normalization."""

    def __init__(self, hidden_size: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: Tensor) -> Tensor:
        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(self.weight.dtype)


class T5AttentionWithLora(nn.Module):
    """T5 self-attention with the same batched LoRA computation as encoder.py."""

    def __init__(
        self,
        hf_config: PretrainedConfig,
        serving_config: LoraServingConfig,
        lora_ops: LoraOps,
        has_relative_attention_bias: bool = False,
    ):
        super().__init__()
        H = serving_config.hidden_size
        self.num_heads = serving_config.num_heads
        self.head_dim = getattr(hf_config, "d_kv", serving_config.head_dim)
        self.inner_dim = self.num_heads * self.head_dim
        if self.inner_dim != H:
            raise ValueError(
                "This serving path expects T5 inner attention dim to equal hidden_size "
                f"(got inner_dim={self.inner_dim}, hidden_size={H})."
            )

        self.target_modules = serving_config.target_modules
        self.lora_ops = lora_ops
        self.relative_attention_num_buckets = getattr(hf_config, "relative_attention_num_buckets", 32)
        self.relative_attention_max_distance = getattr(hf_config, "relative_attention_max_distance", 128)

        self.q = nn.Linear(H, H, bias=False)
        self.k = nn.Linear(H, H, bias=False)
        self.v = nn.Linear(H, H, bias=False)
        self.o = nn.Linear(H, H, bias=False)
        if has_relative_attention_bias:
            self.relative_attention_bias = nn.Embedding(
                self.relative_attention_num_buckets,
                self.num_heads,
            )

    @staticmethod
    def _relative_position_bucket(
        relative_position: Tensor,
        bidirectional: bool = True,
        num_buckets: int = 32,
        max_distance: int = 128,
    ) -> Tensor:
        relative_buckets = 0
        if bidirectional:
            num_buckets //= 2
            relative_buckets += (relative_position > 0).to(torch.long) * num_buckets
            relative_position = torch.abs(relative_position)
        else:
            relative_position = -torch.min(relative_position, torch.zeros_like(relative_position))

        max_exact = num_buckets // 2
        is_small = relative_position < max_exact
        relative_position_if_large = max_exact + (
            torch.log(relative_position.float() / max_exact)
            / math.log(max_distance / max_exact)
            * (num_buckets - max_exact)
        ).to(torch.long)
        relative_position_if_large = torch.min(
            relative_position_if_large,
            torch.full_like(relative_position_if_large, num_buckets - 1),
        )
        return relative_buckets + torch.where(is_small, relative_position, relative_position_if_large)

    def compute_bias(self, query_length: int, key_length: int, device: torch.device) -> Tensor:
        context_position = torch.arange(query_length, dtype=torch.long, device=device)[:, None]
        memory_position = torch.arange(key_length, dtype=torch.long, device=device)[None, :]
        relative_position = memory_position - context_position
        relative_position_bucket = self._relative_position_bucket(
            relative_position,
            bidirectional=True,
            num_buckets=self.relative_attention_num_buckets,
            max_distance=self.relative_attention_max_distance,
        )
        values = self.relative_attention_bias(relative_position_bucket)
        return values.permute(2, 0, 1).unsqueeze(0)

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor | None,
        position_bias: Tensor | None,
        lora_weights: LayerwiseBatchedWeights,
    ) -> tuple[Tensor, Tensor]:
        B, S, H = hidden_states.shape

        projections = {
            "query": self.q(hidden_states),
            "key": self.k(hidden_states),
            "value": self.v(hidden_states),
        }

        # Add LoRA delta to each target projection exactly as in encoder.py.
        for module in self.target_modules:
            if module not in projections:
                continue
            a = torch.cat(lora_weights.a[module], dim=0)
            b = torch.cat(lora_weights.b[module], dim=0)
            self.lora_ops.shrink(projections[module], a)
            self.lora_ops.expand(b)
            projections[module] = projections[module] + self.lora_ops.output

        def reshape(t: Tensor) -> Tensor:
            return t.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)

        q = reshape(projections["query"])
        k = reshape(projections["key"])
        v = reshape(projections["value"])

        scores = torch.matmul(q, k.transpose(-2, -1))

        if position_bias is None:
            if hasattr(self, "relative_attention_bias"):
                position_bias = self.compute_bias(S, S, hidden_states.device)
            else:
                position_bias = torch.zeros(
                    1,
                    self.num_heads,
                    S,
                    S,
                    device=hidden_states.device,
                    dtype=scores.dtype,
                )
            if attention_mask is not None:
                additive_mask = (1.0 - attention_mask.unsqueeze(1).unsqueeze(2).float()) * -10000.0
                position_bias = position_bias + additive_mask

        scores = scores + position_bias
        probs = torch.softmax(scores.float(), dim=-1).to(hidden_states.dtype)
        out = torch.matmul(probs, v)
        out = out.transpose(1, 2).contiguous().view(B, S, H)
        return self.o(out), position_bias


class T5DenseActDense(nn.Module):
    def __init__(self, hf_config: PretrainedConfig, serving_config: LoraServingConfig):
        super().__init__()
        H = serving_config.hidden_size
        I = serving_config.intermediate_size
        self.wi = nn.Linear(H, I, bias=False)
        self.wo = nn.Linear(I, H, bias=False)
        self.act = F.gelu if getattr(hf_config, "dense_act_fn", "relu") == "gelu" else F.relu

    def forward(self, hidden_states: Tensor) -> Tensor:
        return self.wo(self.act(self.wi(hidden_states)))


class T5DenseGatedActDense(nn.Module):
    def __init__(self, hf_config: PretrainedConfig, serving_config: LoraServingConfig):
        super().__init__()
        H = serving_config.hidden_size
        I = serving_config.intermediate_size
        self.wi_0 = nn.Linear(H, I, bias=False)
        self.wi_1 = nn.Linear(H, I, bias=False)
        self.wo = nn.Linear(I, H, bias=False)
        self.act = F.gelu if "gelu" in getattr(hf_config, "dense_act_fn", "gelu") else F.relu

    def forward(self, hidden_states: Tensor) -> Tensor:
        return self.wo(self.act(self.wi_0(hidden_states)) * self.wi_1(hidden_states))


class T5LayerWithLora(nn.Module):
    def __init__(
        self,
        hf_config: PretrainedConfig,
        serving_config: LoraServingConfig,
        lora_ops: LoraOps,
        has_relative_attention_bias: bool = False,
    ):
        super().__init__()
        dense_cls = T5DenseGatedActDense if getattr(hf_config, "is_gated_act", False) else T5DenseActDense
        self.layer = nn.ModuleList([
            nn.ModuleDict({
                "SelfAttention": T5AttentionWithLora(
                    hf_config,
                    serving_config,
                    lora_ops,
                    has_relative_attention_bias=has_relative_attention_bias,
                ),
                "layer_norm": T5LayerNorm(serving_config.hidden_size, serving_config.layer_norm_eps),
            }),
            nn.ModuleDict({
                "DenseReluDense": dense_cls(hf_config, serving_config),
                "layer_norm": T5LayerNorm(serving_config.hidden_size, serving_config.layer_norm_eps),
            }),
        ])

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor | None,
        position_bias: Tensor | None,
        lora_weights: LayerwiseBatchedWeights,
    ) -> tuple[Tensor, Tensor]:
        normed = self.layer[0]["layer_norm"](hidden_states)
        attn_out, position_bias = self.layer[0]["SelfAttention"](
            normed,
            attention_mask,
            position_bias,
            lora_weights,
        )
        hidden_states = hidden_states + attn_out

        normed = self.layer[1]["layer_norm"](hidden_states)
        hidden_states = hidden_states + self.layer[1]["DenseReluDense"](normed)
        return hidden_states, position_bias


class T5EncoderWithLora(PreTrainedModel):
    """Standalone T5 encoder with mean pooling and LR classification heads."""

    def __init__(self, hf_config: PretrainedConfig, serving_config: LoraServingConfig):
        super().__init__(hf_config)
        lora_ops = LoraOps(serving_config)

        self.shared = nn.Embedding(hf_config.vocab_size, serving_config.hidden_size)
        self.encoder = nn.ModuleDict({
            "block": nn.ModuleList([
                T5LayerWithLora(
                    hf_config,
                    serving_config,
                    lora_ops,
                    has_relative_attention_bias=(i == 0),
                )
                for i in range(serving_config.num_layers)
            ]),
            "final_layer_norm": T5LayerNorm(serving_config.hidden_size, serving_config.layer_norm_eps),
        })
        self.post_init()

    @classmethod
    def from_pretrained_serving(cls, serving_config: LoraServingConfig) -> "T5EncoderWithLora":
        hf_config = AutoConfig.from_pretrained(serving_config.model_name)
        model = cls(hf_config, serving_config)
        state_dict = T5EncoderModel.from_pretrained(
            serving_config.model_name,
            torch_dtype=serving_config.dtype,
        ).state_dict()
        unexpected = [key for key in state_dict if key.startswith("encoder.embed_tokens.")]
        for key in unexpected:
            state_dict.pop(key)
        model.load_state_dict(state_dict, strict=True)
        return model.to(serving_config.device)

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None,
        lora_weights: list[LayerwiseBatchedWeights],
        lr_weights: BatchedLRWeights,
        output_lr: Tensor,
        token_type_ids: Tensor | None = None,
    ) -> Tensor:
        del token_type_ids
        x = self.shared(input_ids)
        position_bias = None

        for layer, layer_lora in zip(self.encoder["block"], lora_weights):
            x, position_bias = layer(x, attention_mask, position_bias, layer_lora)

        x = self.encoder["final_layer_norm"](x)
        if attention_mask is None:
            pooled = x.mean(dim=1)
        else:
            mask = attention_mask.unsqueeze(-1).to(dtype=x.dtype)
            pooled = (x * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1e-9)

        LRHeadOps.predict_proba(
            pooled.unsqueeze(1),
            lr_weights.coef,
            lr_weights.intercept,
            output_lr,
        )
        return output_lr
