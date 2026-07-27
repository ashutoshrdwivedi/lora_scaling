"""Batched multi-tenant LoRA on a stock HuggingFace encoder via forward hooks.

`EncoderWithLora` (encoder.py) reimplements the BERT-family forward, so it only
serves checkpoints whose state dict matches that architecture exactly. This
module takes the opposite approach for everything else: run the unmodified HF
model (`AutoModel`) and inject the same per-sample late-fusion delta
    y = W0·x + B_i·A_i·x
through forward hooks on the target projection Linears. The base architecture
is HF's own code — disentangled attention, convolutions, pre-LN variants all
come for free — while the LoRA path stays the identical batched-BMM
decomposition used everywhere else in this repo.

The AdapterStore / BatchAssembler / LRHeadOps pipeline is shared with
EncoderWithLora; only the base forward differs.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
from transformers import AutoModel

from lora_serving.config import LoraServingConfig
from lora_serving.ops.head import LRHeadOps
from lora_serving.weights.batch import BatchedLRWeights, LayerwiseBatchedWeights

# Per-layer module paths (relative to "encoder.layer.{i}.") for each logical
# target name. HF keeps BERT's naming across the post-LN family; DeBERTa-v2
# renames the content projections with a "_proj" suffix.
_TARGET_SUFFIXES: dict[str, dict[str, str]] = {
    "default": {
        "query": "attention.self.query",
        "key": "attention.self.key",
        "value": "attention.self.value",
    },
    "deberta-v2": {
        "query": "attention.self.query_proj",
        "key": "attention.self.key_proj",
        "value": "attention.self.value_proj",
    },
}


class HFEncoderWithLora(nn.Module):
    """Wraps a stock HF encoder; applies per-sample LoRA deltas via hooks.

    Presents the same forward signature as EncoderWithLora so
    benchmark/run.py can swap engines without touching the serving pipeline.

    Example:
        config = LoraServingConfig(model_name="microsoft/deberta-v2-xlarge", ...)
        model = HFEncoderWithLora.from_pretrained_serving(config)
        model.eval()
        output = model(input_ids, attention_mask, lora_weights, lr_weights, output_lr)
    """

    def __init__(self, base: nn.Module, serving_config: LoraServingConfig):
        super().__init__()
        self.base = base
        self.config = serving_config
        # Set per forward pass; hooks read these.
        self._batch_lora: list[LayerwiseBatchedWeights] | None = None
        self._expected_bs: tuple[int, int] | None = None  # (B, S) of the current batch

        suffixes = _TARGET_SUFFIXES.get(
            base.config.model_type, _TARGET_SUFFIXES["default"]
        )
        modules = dict(base.named_modules())
        for logical in serving_config.target_modules:
            if logical not in suffixes:
                raise ValueError(
                    f"target module '{logical}' has no mapping for "
                    f"model_type '{base.config.model_type}' "
                    f"(known: {sorted(suffixes)})"
                )
            for layer_idx in range(serving_config.num_layers):
                name = f"encoder.layer.{layer_idx}.{suffixes[logical]}"
                if name not in modules:
                    raise ValueError(
                        f"module '{name}' not found in {base.config.model_type} "
                        f"model — cannot attach LoRA hook"
                    )
                modules[name].register_forward_hook(
                    self._make_hook(layer_idx, logical)
                )

    def _make_hook(self, layer_idx: int, logical: str):
        def hook(module: nn.Module, inputs: tuple, output: Tensor):
            batch_lora = self._batch_lora
            if batch_lora is None:
                return None
            x = inputs[0]
            # Some architectures route batch-agnostic tensors through the same
            # projection — e.g. DeBERTa-v2 projects the shared relative-position
            # embeddings with query/key_proj when share_att_key is set. Those
            # calls carry no per-sample dimension, so per-tenant deltas do not
            # apply; only the (B, S, H) content call gets the LoRA path.
            if x.dim() != 3 or (x.shape[0], x.shape[1]) != self._expected_bs:
                return None
            layer_w = batch_lora[layer_idx]
            a = torch.cat(layer_w.a[logical], dim=0)  # (B, H, R)
            b = torch.cat(layer_w.b[logical], dim=0)  # (B, R, H)
            return output + torch.bmm(torch.bmm(x, a), b)

        return hook

    @classmethod
    def from_pretrained_serving(
        cls,
        serving_config: LoraServingConfig,
        attn_implementation: str | None = None,
    ) -> "HFEncoderWithLora":
        """Load the stock HF base model and attach LoRA hooks."""
        kwargs: dict = {"torch_dtype": serving_config.dtype}
        if attn_implementation is not None:
            kwargs["attn_implementation"] = attn_implementation
        base = AutoModel.from_pretrained(serving_config.model_name, **kwargs)
        base.eval()
        model = cls(base, serving_config)
        return model.to(serving_config.device)

    def encode_pooled(
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None,
        lora_weights: list[LayerwiseBatchedWeights],
        token_type_ids: Tensor | None = None,
        apply_lora: bool = True,
    ) -> Tensor:
        """Base forward with LoRA deltas, up to the (B, H) pooled output.

        Uses the checkpoint's own pooler when it has one (BERT/RoBERTa
        family); otherwise the raw CLS hidden state (ELECTRA, DeBERTa).
        """
        self._batch_lora = lora_weights if apply_lora else None
        self._expected_bs = tuple(input_ids.shape)
        try:
            kwargs = {}
            if token_type_ids is not None:
                kwargs["token_type_ids"] = token_type_ids
            out = self.base(
                input_ids=input_ids, attention_mask=attention_mask, **kwargs
            )
        finally:
            self._batch_lora = None
            self._expected_bs = None
        pooled = getattr(out, "pooler_output", None)
        if pooled is None:
            pooled = out.last_hidden_state[:, 0]
        return pooled

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None,
        lora_weights: list[LayerwiseBatchedWeights],
        lr_weights: BatchedLRWeights,
        output_lr: Tensor,
        token_type_ids: Tensor | None = None,
        apply_lora: bool = True,
    ) -> Tensor:
        """Full mixed-tenant forward: encoder + batched per-tenant LR heads.

        Same contract as EncoderWithLora.forward — fills and returns output_lr
        with per-sample logit scores (B, 1, max_labels).
        """
        pooled = self.encode_pooled(
            input_ids, attention_mask, lora_weights, token_type_ids, apply_lora
        )
        LRHeadOps.predict_proba(
            pooled.unsqueeze(1),
            lr_weights.coef,
            lr_weights.intercept,
            output_lr,
        )
        return output_lr
