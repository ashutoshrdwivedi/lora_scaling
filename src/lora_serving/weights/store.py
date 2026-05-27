from __future__ import annotations

import time
from collections import OrderedDict
from typing import Callable

import torch
from torch import Tensor

from lora_serving.config import LoraServingConfig

# Maps (layer_idx, module_name) → tensor key in the adapter .bin file.
# Callers provide this to make loading model-agnostic.
# Example for SetFit/adapters-library format:
#   key_fn = lambda i, mod: f"encoder.layer.{i}.attention.self.{mod}.loras.setfit_<id>.lora_A"
KeyResolverFn = Callable[[int, str], tuple[str, str]]  # returns (key_A, key_B)


class LoraWeight:
    """Stores the A and B low-rank matrices for one adapter, per target module per layer.

    Shapes (one entry per target module):
        wa[module]: (num_layers, hidden_size, lora_rank)  — A matrix (shrink)
        wb[module]: (num_layers, lora_rank, hidden_size)  — B matrix (expand)
    """

    def __init__(self, config: LoraServingConfig, device: torch.device | None = None):
        H, R, L = config.hidden_size, config.lora_rank, config.num_layers
        dev = device if device is not None else config.device
        self.wa: dict[str, Tensor] = {
            m: torch.zeros(L, H, R, dtype=config.dtype, device=dev)
            for m in config.target_modules
        }
        self.wb: dict[str, Tensor] = {
            m: torch.zeros(L, R, H, dtype=config.dtype, device=dev)
            for m in config.target_modules
        }

    def copy_from_tensors(
        self,
        a_by_module: dict[str, Tensor],
        b_by_module: dict[str, Tensor],
    ) -> None:
        """Load from column-major tensors (as stored in adapter .bin files).

        Args:
            a_by_module: module → (num_layers, lora_rank, hidden_size)  — A in column-major order
            b_by_module: module → (num_layers, hidden_size, lora_rank)  — B in column-major order
        """
        for m, a in a_by_module.items():
            self.wa[m].copy_(a.to(self.wa[m].device, self.wa[m].dtype).transpose(1, 2))
        for m, b in b_by_module.items():
            self.wb[m].copy_(b.to(self.wb[m].device, self.wb[m].dtype).transpose(1, 2))


class AdapterStore:
    """Pre-loaded cache of tenant LoRA adapters with optional two-tier (GPU + CPU) mode.

    Single-tier mode (default, max_gpu_adapters=None): all adapters live on GPU.
    Identical semantics to the original implementation.

    Two-tier mode (max_gpu_adapters=K): up to K adapters live in a GPU LRU cache;
    the rest live on CPU. On get() miss, the requested adapter is copied from CPU
    to GPU and the LRU GPU slot is evicted. Required for scaling beyond GPU memory
    (~25k adapters at rank-8 fp16 on 40GB) toward the host-RAM-limited ceiling
    (millions of adapters at 1.5 MB each).

    Example:
        store = AdapterStore(config, max_gpu_adapters=10_000)
        store.load_synthetic_cpu("tenant_42")     # cold storage on CPU
        weight = store.get("tenant_42")           # paged into GPU on first access
    """

    def __init__(self, config: LoraServingConfig, max_gpu_adapters: int | None = None):
        self._config = config
        # Single-tier (legacy) storage — only used when max_gpu_adapters is None.
        self._store: dict[str, LoraWeight] = {}
        # Two-tier storage — only used when max_gpu_adapters is an int.
        self._max_gpu = max_gpu_adapters
        self._gpu_cache: OrderedDict[str, LoraWeight] = OrderedDict()
        self._cpu_store: dict[str, LoraWeight] = {}
        # Paging stats (only meaningful in two-tier mode)
        self._hits = 0
        self._misses = 0
        self._evictions = 0
        self._page_in_us_sum = 0.0
        self._page_in_count = 0

    def load_from_file(
        self,
        adapter_id: str,
        path: str,
        key_fn: KeyResolverFn,
    ) -> None:
        """Load a LoRA adapter from a .bin file and cache it on GPU.

        Args:
            adapter_id: Unique identifier for this adapter (used in get()).
            path:       Path to the pytorch_adapter.bin file.
            key_fn:     Maps (layer_idx, module_name) → (key_A, key_B) in the state dict.
        """
        cfg = self._config
        state_dict = torch.load(path, map_location=cfg.device, weights_only=True)
        weight = LoraWeight(cfg)

        a_by_module: dict[str, Tensor] = {}
        b_by_module: dict[str, Tensor] = {}
        for module in cfg.target_modules:
            a_tensors, b_tensors = [], []
            for i in range(cfg.num_layers):
                key_a, key_b = key_fn(i, module)
                a_tensors.append(state_dict[key_a])
                b_tensors.append(state_dict[key_b])
            a_by_module[module] = torch.stack(a_tensors)
            b_by_module[module] = torch.stack(b_tensors)

        weight.copy_from_tensors(a_by_module, b_by_module)
        self._store[adapter_id] = weight

    def load_synthetic(self, adapter_id: str, seed: int | None = None) -> None:
        """Load a random LoRA adapter on GPU — useful for benchmarking without real files."""
        cfg = self._config
        if seed is not None:
            torch.manual_seed(seed)
        weight = LoraWeight(cfg)
        for m in cfg.target_modules:
            torch.nn.init.normal_(weight.wa[m], std=0.02)
            torch.nn.init.zeros_(weight.wb[m])  # B=0 means delta=0, safe for correctness tests
        self._store[adapter_id] = weight

    def load_synthetic_cpu(self, adapter_id: str, seed: int | None = None) -> None:
        """Load a random LoRA adapter into CPU cold storage (two-tier mode).

        Cheaper than load_synthetic when N is large (millions): no GPU memory,
        no kernel launches for init. Adapter is paged into GPU on first get().
        """
        cfg = self._config
        if seed is not None:
            torch.manual_seed(seed)
        cpu_device = torch.device("cpu")
        weight = LoraWeight(cfg, device=cpu_device)
        for m in cfg.target_modules:
            torch.nn.init.normal_(weight.wa[m], std=0.02)
            torch.nn.init.zeros_(weight.wb[m])
        self._cpu_store[adapter_id] = weight

    def get(self, adapter_id: str) -> LoraWeight:
        """Retrieve an adapter. In two-tier mode, pages in from CPU on miss."""
        # Single-tier fast path
        if self._max_gpu is None:
            if adapter_id not in self._store:
                raise KeyError(
                    f"Adapter '{adapter_id}' not found. "
                    f"Call load_from_file() or load_synthetic() first."
                )
            return self._store[adapter_id]

        # Two-tier path
        if adapter_id in self._gpu_cache:
            self._gpu_cache.move_to_end(adapter_id)
            self._hits += 1
            return self._gpu_cache[adapter_id]

        if adapter_id not in self._cpu_store:
            raise KeyError(
                f"Adapter '{adapter_id}' not in CPU cold storage. "
                f"Call load_synthetic_cpu() first."
            )

        self._misses += 1
        # Evict LRU if full
        if len(self._gpu_cache) >= self._max_gpu:
            self._gpu_cache.popitem(last=False)
            self._evictions += 1

        # Page in: allocate GPU LoraWeight and copy from CPU
        t0 = time.perf_counter()
        cpu_w = self._cpu_store[adapter_id]
        gpu_w = LoraWeight(self._config, device=self._config.device)
        for m in self._config.target_modules:
            gpu_w.wa[m].copy_(cpu_w.wa[m], non_blocking=True)
            gpu_w.wb[m].copy_(cpu_w.wb[m], non_blocking=True)
        # Note: non_blocking copies queue on the default stream; the GPU kernels that
        # consume these tensors will see them ready due to stream ordering.
        self._gpu_cache[adapter_id] = gpu_w
        self._page_in_us_sum += (time.perf_counter() - t0) * 1e6
        self._page_in_count += 1
        return gpu_w

    def __len__(self) -> int:
        if self._max_gpu is None:
            return len(self._store)
        return len(self._gpu_cache) + len(self._cpu_store)

    def memory_bytes(self) -> int:
        """Total GPU memory used by adapters currently resident in GPU cache."""
        total = 0
        if self._max_gpu is None:
            adapters = self._store.values()
        else:
            adapters = self._gpu_cache.values()
        for w in adapters:
            for m in w.wa:
                total += w.wa[m].nbytes + w.wb[m].nbytes
        return total

    def memory_gb(self) -> float:
        return self.memory_bytes() / 1e9

    def cpu_memory_bytes(self) -> int:
        """CPU bytes held by the cold store (two-tier mode only).

        Computed analytically per-adapter (avoids iterating millions of dicts):
        n * M * 2 * L * H * R * dtype_bytes.
        """
        cfg = self._config
        elem = torch.empty([], dtype=cfg.dtype).element_size()
        per_adapter = len(cfg.target_modules) * 2 * cfg.num_layers * cfg.hidden_size * cfg.lora_rank * elem
        return per_adapter * len(self._cpu_store)

    def cpu_memory_gb(self) -> float:
        return self.cpu_memory_bytes() / 1e9

    def stats(self) -> dict:
        """Paging stats (two-tier mode). Returns hit/miss counts and avg page-in latency."""
        total_gets = self._hits + self._misses
        return {
            "mode": "two-tier" if self._max_gpu is not None else "single-tier",
            "max_gpu_adapters": self._max_gpu,
            "gpu_resident": len(self._gpu_cache),
            "cpu_resident": len(self._cpu_store),
            "hits": self._hits,
            "misses": self._misses,
            "evictions": self._evictions,
            "hit_rate": self._hits / total_gets if total_gets else 0.0,
            "avg_page_in_us": self._page_in_us_sum / self._page_in_count if self._page_in_count else 0.0,
        }

    def reset_stats(self) -> None:
        """Reset paging counters (call between warmup and measurement)."""
        self._hits = 0
        self._misses = 0
        self._evictions = 0
        self._page_in_us_sum = 0.0
        self._page_in_count = 0
