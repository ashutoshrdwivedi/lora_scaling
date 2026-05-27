"""Scaling-beyond-GPU benchmark: two-tier AdapterStore at N ∈ {100k, 500k, 1M}.

Measures whether CPU-resident adapters paged into GPU on demand can serve
millions of distinct adapters at near-the-in-GPU-baseline latency.

Two access patterns:
  uniform   — each request samples an adapter uniformly at random from N
              (worst case for cache hit rate: hit_rate ≈ max_gpu / N)
  zipfian   — Zipf(α=1.0) over adapter IDs (realistic: a small head of
              popular tenants gets most requests, long cold tail)

Outputs CSV with columns:
  N, pattern, batch_size, max_gpu, p50_ms, p99_ms, throughput, hit_rate,
  avg_page_in_us, gpu_mem_gb, cpu_mem_gb

Run on A100:
    uv run python -m benchmarks.scaling.paging_sweep \\
        --adapter-counts 100000 500000 1000000 \\
        --max-gpu 20000 \\
        --batch-size 32 \\
        --warmup 30 --iters 200 \\
        --out benchmarks/results/paging_sweep.csv
"""

from __future__ import annotations

import argparse
import csv
import gc
import random
import time
from pathlib import Path

import numpy as np
import torch

from lora_serving.benchmark.synthetic import (
    make_synthetic_adapters_cpu_bulk,
    make_synthetic_inputs,
    make_synthetic_lr_weights,
)
from lora_serving.config import LoraServingConfig
from lora_serving.model.encoder import EncoderWithLora
from lora_serving.weights.batch import BatchAssembler
from lora_serving.weights.store import AdapterStore

DTYPE_MAP = {"fp16": torch.float16, "fp32": torch.float32, "bf16": torch.bfloat16}


def zipf_indices(n: int, alpha: float, size: int, rng: np.random.Generator) -> np.ndarray:
    """Draw `size` adapter indices in [0, n) from Zipf(alpha) over the population.

    Numpy's zipf draws from an unbounded distribution; we mod and offset to map
    into [0, n). The popular-head shape is preserved; the exact tail is truncated.
    """
    raw = rng.zipf(alpha, size=size * 4)  # oversample to avoid undersampling after mod
    raw = raw[raw >= 1] - 1
    raw = raw % n
    return raw[:size]


def run_config(
    model: EncoderWithLora,
    store: AdapterStore,
    assembler: BatchAssembler,
    config: LoraServingConfig,
    adapter_ids: list[str],
    lr_coefs: dict,
    lr_intercepts: dict,
    inputs: dict,
    output_lr: torch.Tensor,
    batch_size: int,
    pattern: str,
    warmup: int,
    iters: int,
    zipf_alpha: float = 1.0,
) -> dict:
    """One paging benchmark config. Returns metrics dict."""
    n = len(adapter_ids)
    rng = np.random.default_rng(0)
    device = config.device

    def sample_batch() -> list[str]:
        if pattern == "uniform":
            idxs = rng.integers(0, n, size=batch_size)
        elif pattern == "zipfian":
            idxs = zipf_indices(n, zipf_alpha, batch_size, rng)
        else:
            raise ValueError(f"unknown pattern: {pattern}")
        return [adapter_ids[int(i)] for i in idxs]

    def one_iter() -> tuple[float, float]:
        """Returns (assemble_ms, forward_ms). Mirrors run.py's split timing."""
        ids = sample_batch()
        coefs = [lr_coefs[a] for a in ids]
        intercepts = [lr_intercepts[a] for a in ids]
        t0 = time.perf_counter()
        lora_w, lr_w = assembler.assemble(ids, coefs, intercepts)
        assemble_ms = (time.perf_counter() - t0) * 1000
        start_evt = torch.cuda.Event(enable_timing=True)
        end_evt = torch.cuda.Event(enable_timing=True)
        start_evt.record()
        with torch.no_grad():
            model(inputs["input_ids"], inputs["attention_mask"], lora_w, lr_w, output_lr)
        end_evt.record()
        torch.cuda.synchronize(device)
        forward_ms = start_evt.elapsed_time(end_evt)
        return assemble_ms, forward_ms

    print(f"  Warmup ({warmup} iters, pattern={pattern})...")
    for _ in range(warmup):
        one_iter()
    store.reset_stats()
    torch.cuda.reset_peak_memory_stats(device)

    print(f"  Measuring ({iters} iters)...")
    assemble_lat = np.empty(iters)
    forward_lat = np.empty(iters)
    for i in range(iters):
        assemble_lat[i], forward_lat[i] = one_iter()

    total = assemble_lat + forward_lat
    stats = store.stats()
    peak_gpu_gb = torch.cuda.max_memory_allocated(device) / 1e9

    return {
        "N": n,
        "pattern": pattern,
        "batch_size": batch_size,
        "max_gpu": stats["max_gpu_adapters"],
        "p50_ms": round(float(np.percentile(total, 50)), 2),
        "p90_ms": round(float(np.percentile(total, 90)), 2),
        "p99_ms": round(float(np.percentile(total, 99)), 2),
        "mean_ms": round(float(np.mean(total)), 2),
        "assemble_mean_ms": round(float(np.mean(assemble_lat)), 3),
        "forward_mean_ms": round(float(np.mean(forward_lat)), 3),
        "throughput_samples_sec": round(batch_size / (np.mean(total) / 1000), 1),
        "hits": stats["hits"],
        "misses": stats["misses"],
        "evictions": stats["evictions"],
        "hit_rate": round(stats["hit_rate"], 4),
        "avg_page_in_us": round(stats["avg_page_in_us"], 1),
        "peak_gpu_mem_gb": round(peak_gpu_gb, 3),
        "gpu_cache_gb": round(store.memory_gb(), 3),
        "cpu_store_gb": round(store.cpu_memory_gb(), 3),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="BAAI/bge-m3")
    parser.add_argument("--dtype", choices=list(DTYPE_MAP), default="fp16")
    parser.add_argument("--adapter-counts", nargs="+", type=int,
                        default=[100_000, 500_000, 1_000_000])
    parser.add_argument("--max-gpu", type=int, default=20_000,
                        help="GPU cache capacity (in adapters)")
    parser.add_argument("--patterns", nargs="+", default=["uniform", "zipfian"])
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--num-labels", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--zipf-alpha", type=float, default=1.0)
    parser.add_argument("--out", default="benchmarks/results/paging_sweep.csv")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("Need CUDA. Aborting.")
        return

    device = torch.device("cuda:0")
    dtype = DTYPE_MAP[args.dtype]
    gpu_name = torch.cuda.get_device_name(0)
    print(f"GPU: {gpu_name}")
    print(f"Max GPU adapters (LRU cache size): {args.max_gpu:,}")

    config = LoraServingConfig(
        model_name=args.model,
        lora_rank=args.lora_rank,
        batch_size=args.batch_size,
        max_seq_len=args.seq_len,
        target_modules=["query", "value"],
        device=device,
        dtype=dtype,
    )

    print(f"Loading {args.model} ({dtype})...")
    model = EncoderWithLora.from_pretrained_serving(config)
    model.eval()

    inputs = make_synthetic_inputs(config, args.batch_size)
    output_lr = torch.zeros(args.batch_size, 1, args.num_labels, dtype=dtype, device=device)

    results = []
    for n_adapters in args.adapter_counts:
        per_adapter_mb = 24 * 2 * 1024 * args.lora_rank * 2 / 1e6  # rough analytic
        est_cpu_gb = per_adapter_mb * n_adapters / 1024
        print(f"\n=== N = {n_adapters:,} adapters (~{est_cpu_gb:.1f} GB CPU expected) ===")

        # Fresh store per N to keep paging stats clean
        store = AdapterStore(config, max_gpu_adapters=args.max_gpu)
        t0 = time.perf_counter()
        make_synthetic_adapters_cpu_bulk(store, n_adapters)
        print(f"  CPU population total: {time.perf_counter() - t0:.1f}s, "
              f"cpu_store_gb={store.cpu_memory_gb():.2f}")

        # Generate per-adapter LR weights (small, lives on GPU)
        adapter_ids = [f"adapter_{i}" for i in range(n_adapters)]
        # Sharing one set of LR weights across all adapters to keep memory bounded;
        # LR-head latency doesn't depend on per-adapter LR contents.
        shared_coef, shared_intercept = make_synthetic_lr_weights(config, args.num_labels)
        lr_coefs = {aid: shared_coef for aid in adapter_ids}
        lr_intercepts = {aid: shared_intercept for aid in adapter_ids}

        assembler = BatchAssembler(store, config)

        for pattern in args.patterns:
            print(f"\n  --- pattern = {pattern} ---")
            try:
                row = run_config(
                    model, store, assembler, config,
                    adapter_ids, lr_coefs, lr_intercepts,
                    inputs, output_lr,
                    batch_size=args.batch_size, pattern=pattern,
                    warmup=args.warmup, iters=args.iters,
                    zipf_alpha=args.zipf_alpha,
                )
                results.append(row)
                print(f"  p50={row['p50_ms']}ms  p99={row['p99_ms']}ms  "
                      f"tput={row['throughput_samples_sec']} sps  "
                      f"hit_rate={row['hit_rate']:.3f}  "
                      f"page_in={row['avg_page_in_us']:.0f}us  "
                      f"peak_gpu={row['peak_gpu_mem_gb']}GB")
            except torch.cuda.OutOfMemoryError as e:
                print(f"  OOM at N={n_adapters} pattern={pattern}: {e}")
                torch.cuda.empty_cache()
            except MemoryError as e:
                print(f"  Host OOM at N={n_adapters}: {e}")
                break

        # Free CPU store before moving to next N
        del store, assembler
        gc.collect()
        torch.cuda.empty_cache()

    if not results:
        print("No results collected.")
        return

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"\nSaved to {args.out}")

    print("\n=== Summary ===")
    for r in results:
        print(f"  N={r['N']:>10,}  {r['pattern']:>8s}  "
              f"p50={r['p50_ms']:>6.1f}ms  hit_rate={r['hit_rate']:.3f}  "
              f"tput={r['throughput_samples_sec']:>6.1f} sps")


if __name__ == "__main__":
    main()
