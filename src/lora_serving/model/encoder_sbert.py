"""Sentence-BERT-style bi-encoder with batched multi-tenant LoRA.

Sentence-BERT is a bi-encoder training and pooling family rather than a wholly
different Transformer block. This implementation keeps the same BERT-family
LoRA encoder layers as encoder.py and exposes SBERT-style masked mean pooling
over the token embeddings.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from lora_serving.config import LoraServingConfig
from lora_serving.model.encoder import EncoderWithLora
from lora_serving.ops.head import LRHeadOps
from lora_serving.weights.batch import BatchedLRWeights, LayerwiseBatchedWeights


class SentenceBertEncoderWithLora(EncoderWithLora):
    """BERT-compatible bi-encoder with SBERT masked mean pooling.

    The transformer stack and AttentionWithLora behavior are inherited from
    EncoderWithLora. After the final encoder layer, token embeddings are pooled
    with the attention mask and L2-normalized before the LR heads are applied.
    """

    @classmethod
    def from_pretrained_serving(cls, serving_config: LoraServingConfig) -> "SentenceBertEncoderWithLora":
        from transformers import AutoConfig, AutoModel

        hf_config = AutoConfig.from_pretrained(serving_config.model_name)
        model = cls(hf_config, serving_config)
        state_dict = AutoModel.from_pretrained(
            serving_config.model_name,
            torch_dtype=serving_config.dtype,
        ).state_dict()
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
        B, S = input_ids.shape
        device = input_ids.device

        position_ids = torch.arange(S, device=device).unsqueeze(0).expand(B, -1)
        if token_type_ids is None:
            token_type_ids = torch.zeros(B, S, dtype=torch.long, device=device)

        x = (
            self.embeddings["word_embeddings"](input_ids)
            + self.embeddings["position_embeddings"](position_ids)
            + self.embeddings["token_type_embeddings"](token_type_ids)
        )
        x = self.embeddings["LayerNorm"](x)

        for layer, layer_lora in zip(self.encoder["layer"], lora_weights):
            x = layer(x, attention_mask, layer_lora)

        if attention_mask is None:
            pooled = x.mean(dim=1)
        else:
            mask = attention_mask.unsqueeze(-1).to(dtype=x.dtype)
            pooled = (x * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1e-9)
        pooled = F.normalize(pooled, p=2, dim=-1)

        LRHeadOps.predict_proba(
            pooled.unsqueeze(1),
            lr_weights.coef,
            lr_weights.intercept,
            output_lr,
        )
        return output_lr
