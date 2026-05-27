"""Synthetic data generation for benchmarking without real adapter files."""

import torch

from lora_serving.config import LoraServingConfig
from lora_serving.weights.store import AdapterStore


def make_synthetic_adapters(store: AdapterStore, n: int, seed: int = 42) -> None:
    """Add n randomly-initialised LoRA adapters to the store (GPU-resident).

    Adapter IDs are "adapter_0", "adapter_1", ..., "adapter_{n-1}".
    A matrices are drawn from N(0, 0.02); B matrices are zero (so delta=0,
    safe for correctness checks).

    Args:
        store: AdapterStore to populate.
        n:     Number of adapters to generate.
        seed:  RNG seed for reproducibility.
    """
    for i in range(n):
        store.load_synthetic(f"adapter_{i}", seed=seed + i)


def make_synthetic_adapters_cpu(
    store: AdapterStore,
    n: int,
    seed: int = 42,
    progress_every: int = 50_000,
) -> None:
    """Add n random LoRA adapters into CPU cold storage — slow per-adapter path.

    Kept for compatibility; for n > 1000 use `make_synthetic_adapters_cpu_bulk`,
    which is ~100x faster due to bulk tensor allocation.
    """
    import time as _t
    t0 = _t.perf_counter()
    for i in range(n):
        store.load_synthetic_cpu(f"adapter_{i}", seed=seed + i)
        if progress_every and (i + 1) % progress_every == 0:
            elapsed = _t.perf_counter() - t0
            rate = (i + 1) / elapsed
            eta = (n - i - 1) / rate
            print(f"  populated {i+1:>9,}/{n:,} CPU adapters  "
                  f"({rate:.0f}/s, ETA {eta:.0f}s)")


def make_synthetic_adapters_cpu_bulk(
    store: AdapterStore,
    n: int,
    seed: int = 42,
) -> None:
    """Bulk-allocate n CPU adapters as views into one big tensor per (module, A/B).

    Memory layout: one (N, L, H, R) tensor per module for A, one (N, L, R, H) for B.
    Each per-adapter LoraWeight wraps slice views — no per-adapter allocation.
    Avoids 99% of the Python overhead in the per-adapter path; ~100x faster.

    At rank-8 fp16, 24 layers, hidden=1024, 2 target modules: ~1.57 MB/adapter
    so 1M adapters = 1.57 TB CPU RAM.
    """
    import time as _t
    from lora_serving.weights.store import LoraWeight

    cfg = store._config
    L, H, R = cfg.num_layers, cfg.hidden_size, cfg.lora_rank
    modules = cfg.target_modules
    dtype = cfg.dtype

    t0 = _t.perf_counter()
    total_gb = n * 2 * len(modules) * L * H * R * 2 / 1e9
    print(f"  Bulk-allocating {n:,} CPU adapters (~{total_gb:.1f} GB)...")
    # Use torch.zeros for both A and B: actual values are irrelevant for paging-
    # latency measurement (we only measure copy bandwidth + hit rates), and zeros_
    # is memory-bandwidth-bound while normal_ is single-thread PRNG bound.
    # All pages get faulted in so the H2D copy reflects a true RAM-resident workload.
    big_wa = {m: torch.zeros((n, L, H, R), dtype=dtype) for m in modules}
    big_wb = {m: torch.zeros((n, L, R, H), dtype=dtype) for m in modules}

    alloc_s = _t.perf_counter() - t0
    print(f"  Allocated + initialized in {alloc_s:.1f}s. Wiring {n:,} view objects...")

    # Construct N LoraWeight views — bypass __init__ to avoid per-instance allocation
    t1 = _t.perf_counter()
    for i in range(n):
        w = LoraWeight.__new__(LoraWeight)
        w.wa = {m: big_wa[m][i] for m in modules}
        w.wb = {m: big_wb[m][i] for m in modules}
        store._cpu_store[f"adapter_{i}"] = w
    wire_s = _t.perf_counter() - t1
    print(f"  Wired {n:,} views in {wire_s:.1f}s. Total: {_t.perf_counter() - t0:.1f}s")


def make_synthetic_inputs(config: LoraServingConfig, batch_size: int) -> dict:
    """Generate random tokenized inputs for a batch.

    Returns a dict with:
        input_ids:      (batch_size, max_seq_len) int64
        attention_mask: (batch_size, max_seq_len) int64  (all ones — no padding)
    """
    return {
        "input_ids": torch.randint(
            low=1,
            high=30000,  # safe for most vocab sizes
            size=(batch_size, config.max_seq_len),
            device=config.device,
        ),
        "attention_mask": torch.ones(
            batch_size, config.max_seq_len,
            dtype=torch.long,
            device=config.device,
        ),
    }


def make_synthetic_lr_weights(
    config: LoraServingConfig,
    num_labels: int = 10,
) -> tuple:
    """Generate random LR coef and intercept tensors for one tenant.

    Returns:
        coef:      (1, num_labels, hidden_size)
        intercept: (1, num_labels)
    """
    coef = torch.randn(1, num_labels, config.hidden_size, dtype=config.dtype, device=config.device)
    intercept = torch.zeros(1, num_labels, dtype=config.dtype, device=config.device)
    return coef, intercept
