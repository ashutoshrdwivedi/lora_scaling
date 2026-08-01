import torch
from torch import Tensor

from lora_serving.config import LoraServingConfig


class LoraOps:
    """Batched LoRA operations using pre-allocated device buffers.

    LoRA applies a low-rank delta to a projection:
        y = W0 * x + B * A * x
    where A (shrink) reduces to lora_rank and B (expand) returns to hidden_size.

    Buffers are allocated once at init and reused every forward pass to avoid
    per-call memory allocation overhead.

    Usage:
        ops = LoraOps(config)
        ops.shrink(x, a_weights)   # fills internal buffer out_A
        ops.expand(b_weights)      # fills internal buffer out_B
        delta = ops.output         # (B, S, H) LoRA delta to add to base projection
    """

    def __init__(self, config: LoraServingConfig):
        B, S, H, R = config.batch_size, config.max_seq_len, config.hidden_size, config.lora_rank
        self._out_A = torch.empty(B, S, R, dtype=config.dtype, device=config.device)
        self._out_B = torch.empty(B, S, H, dtype=config.dtype, device=config.device)
        if getattr(config, "relative_attention", False):
            rel_span = getattr(config, "position_buckets", -1)
            if rel_span <= 0:
                rel_span = getattr(config, "max_relative_positions", 0)
            rel_seq_len = max(1, rel_span * 2)
        else:
            rel_seq_len = 1
        self._out_A_relative = torch.empty(B, rel_seq_len, R, dtype=config.dtype, device=config.device)
        self._out_B_relative = torch.empty(B, rel_seq_len, H, dtype=config.dtype, device=config.device)

    def shrink(self, x: Tensor, a_weights: Tensor) -> None:
        """First LoRA step: project input down to lora_rank.

        Args:
            x:         (B, S, H) — input hidden states
            a_weights: (B, H, R) — stacked A matrices for each sample in the batch
        """
        assert x.device == self._out_A.device, (
            f"x device {x.device} != buffer device {self._out_A.device}"
        )
        assert a_weights.device == self._out_A.device, (
            f"a_weights device {a_weights.device} != buffer device {self._out_A.device}"
        )
        # bmm with a mismatched `out=` silently resizes the buffer, defeating
        # the pre-allocation — inputs must match the init-time (B, S, H, R).
        B, S, R = self._out_A.shape
        H = self._out_B.shape[2]
        assert x.shape == (B, S, H), (
            f"x shape {tuple(x.shape)} != buffer shape {(B, S, H)}; "
            "LoraOps buffers are sized from config at init"
        )
        assert a_weights.shape == (B, H, R), (
            f"a_weights shape {tuple(a_weights.shape)} != expected {(B, H, R)}"
        )
        torch.bmm(x, a_weights, out=self._out_A)

    def shrink_relative(self, x: Tensor, a_weights: Tensor) -> None:
        """Project relative-position embeddings down to lora_rank."""
        assert x.device == self._out_A_relative.device, (
            f"x device {x.device} != relative buffer device {self._out_A_relative.device}"
        )
        assert a_weights.device == self._out_A_relative.device, (
            f"a_weights device {a_weights.device} != relative buffer device {self._out_A_relative.device}"
        )
        B, S, R = self._out_A_relative.shape
        H = self._out_B_relative.shape[2]
        assert x.shape == (B, S, H), (
            f"relative x shape {tuple(x.shape)} != buffer shape {(B, S, H)}"
        )
        assert a_weights.shape == (B, H, R), (
            f"a_weights shape {tuple(a_weights.shape)} != expected {(B, H, R)}"
        )
        torch.bmm(x, a_weights, out=self._out_A_relative)

    def expand(self, b_weights: Tensor) -> None:
        """Second LoRA step: project back up to hidden_size.

        Args:
            b_weights: (B, R, H) — stacked B matrices for each sample in the batch

        Call shrink() before expand(). Result is available via .output.
        """
        assert b_weights.device == self._out_B.device, (
            f"b_weights device {b_weights.device} != buffer device {self._out_B.device}"
        )
        B, S, R = self._out_A.shape
        H = self._out_B.shape[2]
        assert b_weights.shape == (B, R, H), (
            f"b_weights shape {tuple(b_weights.shape)} != expected {(B, R, H)}"
        )
        torch.bmm(self._out_A, b_weights, out=self._out_B)

    def expand_relative(self, b_weights: Tensor) -> None:
        """Project relative-position LoRA activations back to hidden_size."""
        assert b_weights.device == self._out_B_relative.device, (
            f"b_weights device {b_weights.device} != relative buffer device {self._out_B_relative.device}"
        )
        B, S, R = self._out_A_relative.shape
        H = self._out_B_relative.shape[2]
        assert b_weights.shape == (B, R, H), (
            f"b_weights shape {tuple(b_weights.shape)} != expected {(B, R, H)}"
        )
        torch.bmm(self._out_A_relative, b_weights, out=self._out_B_relative)

    @property
    def output(self) -> Tensor:
        """The (B, S, H) LoRA delta from the last expand() call."""
        return self._out_B

    @property
    def output_relative(self) -> Tensor:
        """The relative-position LoRA delta from the last expand_relative() call."""
        return self._out_B_relative
