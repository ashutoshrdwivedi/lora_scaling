"""DeBERTa-v2 encoder with disentangled attention and batched LoRA."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from transformers import PretrainedConfig, PreTrainedModel
from transformers.activations import ACT2FN
from transformers.models.deberta_v2.modeling_deberta_v2 import (
    StableDropout,
    XSoftmax,
    build_relative_position,
)

from lora_serving.config import LoraServingConfig
from lora_serving.ops.head import LRHeadOps
from lora_serving.ops.lora import LoraOps
from lora_serving.weights.batch import BatchedLRWeights, LayerwiseBatchedWeights


class DebertaV2EmbeddingsWithLora(nn.Module):
    """DeBERTa-v2 embeddings matching HuggingFace module names."""

    def __init__(self, hf_config: PretrainedConfig, serving_config: LoraServingConfig):
        super().__init__()
        H = serving_config.hidden_size
        E = serving_config.embedding_size
        pad_token_id = getattr(hf_config, "pad_token_id", 0)

        self.embedding_size = E
        self.position_biased_input = serving_config.position_biased_input
        self.word_embeddings = nn.Embedding(hf_config.vocab_size, E, padding_idx=pad_token_id)
        if self.position_biased_input:
            self.position_embeddings = nn.Embedding(hf_config.max_position_embeddings, E)
        else:
            self.position_embeddings = None
        if serving_config.type_vocab_size > 0:
            self.token_type_embeddings = nn.Embedding(serving_config.type_vocab_size, E)
        if E != H:
            self.embed_proj = nn.Linear(E, H, bias=False)
        self.LayerNorm = nn.LayerNorm(H, eps=serving_config.layer_norm_eps)
        self.dropout = StableDropout(getattr(hf_config, "hidden_dropout_prob", 0.0))
        self.config = hf_config
        self.register_buffer(
            "position_ids",
            torch.arange(hf_config.max_position_embeddings).expand((1, -1)),
            persistent=False,
        )

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None,
        token_type_ids: Tensor | None = None,
        position_ids: Tensor | None = None,
    ) -> Tensor:
        input_shape = input_ids.size()
        seq_length = input_shape[1]
        if position_ids is None:
            position_ids = self.position_ids[:, :seq_length]
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=input_ids.device)

        embeddings = self.word_embeddings(input_ids)
        if self.position_embeddings is not None:
            embeddings = embeddings + self.position_embeddings(position_ids.long())
        if hasattr(self, "token_type_embeddings"):
            embeddings = embeddings + self.token_type_embeddings(token_type_ids)
        if hasattr(self, "embed_proj"):
            embeddings = self.embed_proj(embeddings)
        embeddings = self.LayerNorm(embeddings)

        if attention_mask is not None:
            mask = attention_mask
            if mask.dim() != embeddings.dim():
                if mask.dim() == 4:
                    mask = mask.squeeze(1).squeeze(1)
                mask = mask.unsqueeze(2)
            embeddings = embeddings * mask.to(embeddings.dtype)
        return self.dropout(embeddings)


class DisentangledAttentionWithLora(nn.Module):
    """DeBERTa-v2 disentangled self-attention with batched LoRA deltas."""

    def __init__(self, hf_config: PretrainedConfig, serving_config: LoraServingConfig, lora_ops: LoraOps):
        super().__init__()
        H = serving_config.hidden_size
        self.num_attention_heads = serving_config.num_heads
        self.attention_head_size = getattr(hf_config, "attention_head_size", serving_config.head_dim)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.batch_size = serving_config.batch_size
        self.max_seq_len = serving_config.max_seq_len
        self.target_modules = serving_config.target_modules
        self.lora_ops = lora_ops

        self.query_proj = nn.Linear(H, self.all_head_size, bias=True)
        self.key_proj = nn.Linear(H, self.all_head_size, bias=True)
        self.value_proj = nn.Linear(H, self.all_head_size, bias=True)

        self.share_att_key = serving_config.share_att_key
        self.pos_att_type = serving_config.pos_att_type
        self.relative_attention = serving_config.relative_attention

        if self.relative_attention:
            self.position_buckets = serving_config.position_buckets
            self.max_relative_positions = serving_config.max_relative_positions
            self.pos_ebd_size = self.position_buckets if self.position_buckets > 0 else self.max_relative_positions
            self.pos_dropout = StableDropout(getattr(hf_config, "hidden_dropout_prob", 0.0))
            if not self.share_att_key:
                if "c2p" in self.pos_att_type:
                    self.pos_key_proj = nn.Linear(H, self.all_head_size, bias=True)
                if "p2c" in self.pos_att_type:
                    self.pos_query_proj = nn.Linear(H, self.all_head_size)

        self.dropout = StableDropout(getattr(hf_config, "attention_probs_dropout_prob", 0.0))

    def _linear_with_lora(
        self,
        module_name: str,
        linear: nn.Linear,
        x: Tensor,
        lora_weights: LayerwiseBatchedWeights,
        apply_lora: bool,
    ) -> Tensor:
        out = linear(x)
        if (
            not apply_lora
            or module_name not in self.target_modules
            or module_name not in lora_weights.a
        ):
            return out

        a = torch.cat(lora_weights.a[module_name], dim=0)
        b = torch.cat(lora_weights.b[module_name], dim=0)
        if x.size(0) == self.batch_size and x.size(1) == self.max_seq_len:
            self.lora_ops.shrink(x, a)
            self.lora_ops.expand(b)
            return out + self.lora_ops.output

        if x.size(0) == 1:
            x = x.expand(self.batch_size, -1, -1)
            out = out.expand(self.batch_size, -1, -1)
        elif x.size(0) != self.batch_size:
            raise ValueError(f"LoRA input batch {x.size(0)} does not match configured batch {self.batch_size}")

        self.lora_ops.shrink_relative(x, a)
        self.lora_ops.expand_relative(b)
        return out + self.lora_ops.output_relative

    def transpose_for_scores(self, x: Tensor, attention_heads: int) -> Tensor:
        new_x_shape = x.size()[:-1] + (attention_heads, -1)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))

    def transpose_relative_projection(self, x: Tensor, query_layer: Tensor) -> Tensor:
        x = self.transpose_for_scores(x, self.num_attention_heads)
        if x.size(0) == self.num_attention_heads:
            return x.repeat(query_layer.size(0) // self.num_attention_heads, 1, 1)
        return x

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor,
        lora_weights: LayerwiseBatchedWeights,
        query_states: Tensor | None = None,
        relative_pos: Tensor | None = None,
        rel_embeddings: Tensor | None = None,
        apply_lora: bool = True,
    ) -> Tensor:
        if query_states is None:
            query_states = hidden_states

        query_layer = self.transpose_for_scores(
            self._linear_with_lora("query_proj", self.query_proj, query_states, lora_weights, apply_lora),
            self.num_attention_heads,
        )
        key_layer = self.transpose_for_scores(
            self._linear_with_lora("key_proj", self.key_proj, hidden_states, lora_weights, apply_lora),
            self.num_attention_heads,
        )
        value_layer = self.transpose_for_scores(
            self._linear_with_lora("value_proj", self.value_proj, hidden_states, lora_weights, apply_lora),
            self.num_attention_heads,
        )

        scale_factor = 1
        if "c2p" in self.pos_att_type:
            scale_factor += 1
        if "p2c" in self.pos_att_type:
            scale_factor += 1
        scale = torch.sqrt(torch.tensor(query_layer.size(-1), dtype=torch.float, device=query_layer.device) * scale_factor)
        attention_scores = torch.bmm(query_layer, key_layer.transpose(-1, -2) / scale.to(query_layer.dtype))

        if self.relative_attention and rel_embeddings is not None:
            rel_embeddings = self.pos_dropout(rel_embeddings)
            rel_att = self.disentangled_attention_bias(
                query_layer,
                key_layer,
                relative_pos,
                rel_embeddings,
                scale_factor,
                lora_weights,
                apply_lora,
            )
            attention_scores = attention_scores + rel_att

        attention_scores = attention_scores.view(
            -1, self.num_attention_heads, attention_scores.size(-2), attention_scores.size(-1)
        )
        attention_probs = XSoftmax.apply(attention_scores, attention_mask, -1)
        attention_probs = self.dropout(attention_probs)
        context_layer = torch.bmm(
            attention_probs.view(-1, attention_probs.size(-2), attention_probs.size(-1)),
            value_layer,
        )
        context_layer = (
            context_layer.view(-1, self.num_attention_heads, context_layer.size(-2), context_layer.size(-1))
            .permute(0, 2, 1, 3)
            .contiguous()
        )
        return context_layer.view(context_layer.size()[:-2] + (-1,))

    def disentangled_attention_bias(
        self,
        query_layer: Tensor,
        key_layer: Tensor,
        relative_pos: Tensor | None,
        rel_embeddings: Tensor,
        scale_factor: int,
        lora_weights: LayerwiseBatchedWeights,
        apply_lora: bool,
    ) -> Tensor:
        if relative_pos is None:
            q = query_layer.size(-2)
            relative_pos = build_relative_position(
                q,
                key_layer.size(-2),
                bucket_size=self.position_buckets,
                max_position=self.max_relative_positions,
                device=query_layer.device,
            )
        if relative_pos.dim() == 2:
            relative_pos = relative_pos.unsqueeze(0).unsqueeze(0)
        elif relative_pos.dim() == 3:
            relative_pos = relative_pos.unsqueeze(1)
        elif relative_pos.dim() != 4:
            raise ValueError(f"Relative position ids must be of dim 2, 3, or 4. Got {relative_pos.dim()}.")

        att_span = self.pos_ebd_size
        relative_pos = relative_pos.long().to(query_layer.device)
        rel_embeddings = rel_embeddings[0 : att_span * 2, :].unsqueeze(0)

        if self.share_att_key:
            pos_query = self._linear_with_lora(
                "query_proj", self.query_proj, rel_embeddings, lora_weights, apply_lora
            )
            pos_key = self._linear_with_lora("key_proj", self.key_proj, rel_embeddings, lora_weights, apply_lora)
            pos_query_layer = self.transpose_relative_projection(pos_query, query_layer)
            pos_key_layer = self.transpose_relative_projection(pos_key, query_layer)
        else:
            if "c2p" in self.pos_att_type:
                pos_key = self._linear_with_lora(
                    "pos_key_proj", self.pos_key_proj, rel_embeddings, lora_weights, apply_lora
                )
                pos_key_layer = self.transpose_relative_projection(pos_key, query_layer)
            if "p2c" in self.pos_att_type:
                pos_query = self._linear_with_lora(
                    "pos_query_proj", self.pos_query_proj, rel_embeddings, lora_weights, apply_lora
                )
                pos_query_layer = self.transpose_relative_projection(pos_query, query_layer)

        score = 0
        if "c2p" in self.pos_att_type:
            scale = torch.sqrt(torch.tensor(pos_key_layer.size(-1), dtype=torch.float, device=query_layer.device) * scale_factor)
            c2p_att = torch.bmm(query_layer, pos_key_layer.transpose(-1, -2))
            c2p_pos = torch.clamp(relative_pos + att_span, 0, att_span * 2 - 1)
            c2p_att = torch.gather(
                c2p_att,
                dim=-1,
                index=c2p_pos.squeeze(0).expand([query_layer.size(0), query_layer.size(1), relative_pos.size(-1)]),
            )
            score += c2p_att / scale.to(c2p_att.dtype)

        if "p2c" in self.pos_att_type:
            scale = torch.sqrt(torch.tensor(pos_query_layer.size(-1), dtype=torch.float, device=query_layer.device) * scale_factor)
            if key_layer.size(-2) != query_layer.size(-2):
                r_pos = build_relative_position(
                    key_layer.size(-2),
                    key_layer.size(-2),
                    bucket_size=self.position_buckets,
                    max_position=self.max_relative_positions,
                    device=query_layer.device,
                )
                r_pos = r_pos.unsqueeze(0)
            else:
                r_pos = relative_pos
            p2c_pos = torch.clamp(-r_pos + att_span, 0, att_span * 2 - 1)
            p2c_att = torch.bmm(key_layer, pos_query_layer.transpose(-1, -2))
            p2c_att = torch.gather(
                p2c_att,
                dim=-1,
                index=p2c_pos.squeeze(0).expand([query_layer.size(0), key_layer.size(-2), key_layer.size(-2)]),
            ).transpose(-1, -2)
            score += p2c_att / scale.to(p2c_att.dtype)

        return score


class DebertaV2AttentionWithLora(nn.Module):
    def __init__(self, hf_config: PretrainedConfig, serving_config: LoraServingConfig, lora_ops: LoraOps):
        super().__init__()
        H = serving_config.hidden_size
        self.self = DisentangledAttentionWithLora(hf_config, serving_config, lora_ops)
        self.output = nn.ModuleDict({
            "dense": nn.Linear(H, H),
            "LayerNorm": nn.LayerNorm(H, eps=serving_config.layer_norm_eps),
            "dropout": StableDropout(getattr(hf_config, "hidden_dropout_prob", 0.0)),
        })
        self.config = hf_config

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor,
        lora_weights: LayerwiseBatchedWeights,
        query_states: Tensor | None = None,
        relative_pos: Tensor | None = None,
        rel_embeddings: Tensor | None = None,
        apply_lora: bool = True,
    ) -> Tensor:
        self_output = self.self(
            hidden_states,
            attention_mask,
            lora_weights,
            query_states=query_states,
            relative_pos=relative_pos,
            rel_embeddings=rel_embeddings,
            apply_lora=apply_lora,
        )
        if query_states is None:
            query_states = hidden_states
        attention_output = self.output["dense"](self_output)
        attention_output = self.output["dropout"](attention_output)
        return self.output["LayerNorm"](attention_output + query_states)


class DebertaV2LayerWithLora(nn.Module):
    def __init__(self, hf_config: PretrainedConfig, serving_config: LoraServingConfig, lora_ops: LoraOps):
        super().__init__()
        H = serving_config.hidden_size
        self.attention = DebertaV2AttentionWithLora(hf_config, serving_config, lora_ops)
        self.intermediate = nn.ModuleDict({
            "dense": nn.Linear(H, serving_config.intermediate_size),
        })
        self.intermediate_act = ACT2FN[serving_config.hidden_act]
        self.output = nn.ModuleDict({
            "dense": nn.Linear(serving_config.intermediate_size, H),
            "LayerNorm": nn.LayerNorm(H, eps=serving_config.layer_norm_eps),
            "dropout": StableDropout(getattr(hf_config, "hidden_dropout_prob", 0.0)),
        })

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor,
        lora_weights: LayerwiseBatchedWeights,
        query_states: Tensor | None = None,
        relative_pos: Tensor | None = None,
        rel_embeddings: Tensor | None = None,
        apply_lora: bool = True,
    ) -> Tensor:
        attention_output = self.attention(
            hidden_states,
            attention_mask,
            lora_weights,
            query_states=query_states,
            relative_pos=relative_pos,
            rel_embeddings=rel_embeddings,
            apply_lora=apply_lora,
        )
        intermediate_output = self.intermediate_act(self.intermediate["dense"](attention_output))
        layer_output = self.output["dense"](intermediate_output)
        layer_output = self.output["dropout"](layer_output)
        return self.output["LayerNorm"](layer_output + attention_output)


class DebertaV2ConvLayer(nn.Module):
    def __init__(self, hf_config: PretrainedConfig):
        super().__init__()
        kernel_size = getattr(hf_config, "conv_kernel_size", 3)
        groups = getattr(hf_config, "conv_groups", 1)
        self.conv_act = getattr(hf_config, "conv_act", "tanh")
        self.conv = nn.Conv1d(
            hf_config.hidden_size,
            hf_config.hidden_size,
            kernel_size,
            padding=(kernel_size - 1) // 2,
            groups=groups,
        )
        self.LayerNorm = nn.LayerNorm(hf_config.hidden_size, eps=hf_config.layer_norm_eps)
        self.dropout = StableDropout(getattr(hf_config, "hidden_dropout_prob", 0.0))
        self.config = hf_config

    def forward(self, hidden_states: Tensor, residual_states: Tensor, input_mask: Tensor) -> Tensor:
        out = self.conv(hidden_states.permute(0, 2, 1).contiguous()).permute(0, 2, 1).contiguous()
        rmask = (1 - input_mask).bool()
        out.masked_fill_(rmask.unsqueeze(-1).expand(out.size()), 0)
        out = ACT2FN[self.conv_act](self.dropout(out))

        layer_norm_input = residual_states + out
        output = self.LayerNorm(layer_norm_input).to(layer_norm_input)

        if input_mask.dim() != layer_norm_input.dim():
            if input_mask.dim() == 4:
                input_mask = input_mask.squeeze(1).squeeze(1)
            input_mask = input_mask.unsqueeze(2)
        return output * input_mask.to(output.dtype)


class DebertaV2EncoderStackWithLora(nn.Module):
    def __init__(self, hf_config: PretrainedConfig, serving_config: LoraServingConfig, lora_ops: LoraOps):
        super().__init__()
        self.layer = nn.ModuleList([
            DebertaV2LayerWithLora(hf_config, serving_config, lora_ops)
            for _ in range(serving_config.num_layers)
        ])
        self.relative_attention = serving_config.relative_attention
        self.max_relative_positions = serving_config.max_relative_positions
        self.position_buckets = serving_config.position_buckets
        self.norm_rel_ebd = [x.strip() for x in serving_config.norm_rel_ebd.lower().split("|")]
        if self.relative_attention:
            pos_ebd_size = self.position_buckets * 2 if self.position_buckets > 0 else self.max_relative_positions * 2
            self.rel_embeddings = nn.Embedding(pos_ebd_size, serving_config.hidden_size)
        if "layer_norm" in self.norm_rel_ebd:
            self.LayerNorm = nn.LayerNorm(serving_config.hidden_size, eps=serving_config.layer_norm_eps)
        self.conv = DebertaV2ConvLayer(hf_config) if getattr(hf_config, "conv_kernel_size", 0) > 0 else None

    def get_attention_mask(self, attention_mask: Tensor) -> Tensor:
        if attention_mask.dim() <= 2:
            extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            attention_mask = extended_attention_mask * extended_attention_mask.squeeze(-2).unsqueeze(-1)
        elif attention_mask.dim() == 3:
            attention_mask = attention_mask.unsqueeze(1)
        return attention_mask

    def get_rel_embedding(self) -> Tensor | None:
        rel_embeddings = self.rel_embeddings.weight if self.relative_attention else None
        if rel_embeddings is not None and "layer_norm" in self.norm_rel_ebd:
            rel_embeddings = self.LayerNorm(rel_embeddings)
        return rel_embeddings

    def get_rel_pos(
        self,
        hidden_states: Tensor,
        query_states: Tensor | None = None,
        relative_pos: Tensor | None = None,
    ) -> Tensor | None:
        if self.relative_attention and relative_pos is None:
            q = query_states.size(-2) if query_states is not None else hidden_states.size(-2)
            relative_pos = build_relative_position(
                q,
                hidden_states.size(-2),
                bucket_size=self.position_buckets,
                max_position=self.max_relative_positions,
                device=hidden_states.device,
            )
        return relative_pos

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor,
        lora_weights: list[LayerwiseBatchedWeights],
        apply_lora: bool = True,
    ) -> Tensor:
        input_mask = attention_mask if attention_mask.dim() <= 2 else attention_mask.sum(-2) > 0
        attention_mask = self.get_attention_mask(attention_mask)
        relative_pos = self.get_rel_pos(hidden_states)
        rel_embeddings = self.get_rel_embedding()
        output_states = hidden_states
        for i, (layer, layer_lora) in enumerate(zip(self.layer, lora_weights)):
            output_states = layer(
                output_states,
                attention_mask,
                layer_lora,
                relative_pos=relative_pos,
                rel_embeddings=rel_embeddings,
                apply_lora=apply_lora,
            )
            if i == 0 and self.conv is not None:
                output_states = self.conv(hidden_states, output_states, input_mask)
        return output_states


class DebertaV2EncoderWithLora(PreTrainedModel):
    """DeBERTa-v2 encoder with disentangled attention and batched LoRA."""

    def __init__(self, hf_config: PretrainedConfig, serving_config: LoraServingConfig):
        super().__init__(hf_config)
        lora_ops = LoraOps(serving_config)
        self.embeddings = DebertaV2EmbeddingsWithLora(hf_config, serving_config)
        self.encoder = DebertaV2EncoderStackWithLora(hf_config, serving_config, lora_ops)
        self.z_steps = 0
        self.config = hf_config
        self.post_init()

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
        if attention_mask is None:
            attention_mask = torch.ones(input_ids.shape, device=input_ids.device)
        embedding_output = self.embeddings(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        sequence_output = self.encoder(embedding_output, attention_mask, lora_weights, apply_lora)
        pooled = sequence_output[:, 0]
        LRHeadOps.predict_proba(
            pooled.unsqueeze(1),
            lr_weights.coef,
            lr_weights.intercept,
            output_lr,
        )
        return output_lr
