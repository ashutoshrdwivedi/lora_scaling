"""Standalone benchmark for multi-tenant LoRA serving on a GPU.

Run on RunPod (or any GPU machine):
    pip install uv
    uv sync
    uv run python -m lora_serving.benchmark.run

Sweep options:
    uv run python -m lora_serving.benchmark.run \\
        --model intfloat/multilingual-e5-small \\
        --adapters 100 500 1000 5000 \\
        --batch-sizes 16 32 64 128 \\
        --lora-ranks 8 16 32 \\
        --seq-len 128 \\
        --warmup 50 \\
        --iters 200 \\
        --out results.csv
"""

from __future__ import annotations

import argparse
import csv
import random
import time

import numpy as np
import torch

from lora_serving.benchmark.synthetic import (
    make_synthetic_adapters,
    make_synthetic_inputs,
    make_synthetic_lr_weights,
)
from lora_serving.config import LoraServingConfig
from lora_serving.model.encoder import EncoderWithLora
from lora_serving.weights.batch import BatchAssembler
from lora_serving.weights.store import AdapterStore


def run_single_config(
    model_name: str,
    num_adapters: int,
    batch_size: int,
    lora_rank: int,
    seq_len: int,
    warmup: int,
    iters: int,
    num_labels: int = 10,
) -> dict:
    """Run one benchmark configuration and return metrics."""
    device = torch.device("cuda:0")
    dtype = torch.float32

    config = LoraServingConfig(
        model_name=model_name,
        lora_rank=lora_rank,
        batch_size=batch_size,
        max_seq_len=seq_len,
        target_modules=["query", "value"],
        device=device,
        dtype=dtype,
    )

    # Load model
    print(f"  Loading base model ({model_name})...")
    model = EncoderWithLora.from_pretrained_serving(config)
    model.eval()

    # Populate adapter store with synthetic weights
    print(f"  Generating {num_adapters} synthetic adapters (rank={lora_rank})...")
    store = AdapterStore(config)
    make_synthetic_adapters(store, num_adapters)
    adapter_ids = [f"adapter_{i}" for i in range(num_adapters)]

    # Generate synthetic LR weights for each adapter
    lr_coefs = {}
    lr_intercepts = {}
    for aid in adapter_ids:
        coef, intercept = make_synthetic_lr_weights(config, num_labels)
        lr_coefs[aid] = coef
        lr_intercepts[aid] = intercept

    assembler = BatchAssembler(store, config)
    inputs = make_synthetic_inputs(config, batch_size)
    output_lr = torch.zeros(batch_size, 1, num_labels, dtype=dtype, device=device)

    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)

    def run_batch() -> None:
        # Sample a random batch of adapter IDs (mixed tenants)
        batch_ids = random.choices(adapter_ids, k=batch_size)
        coefs = [lr_coefs[aid] for aid in batch_ids]
        intercepts = [lr_intercepts[aid] for aid in batch_ids]

        lora_w, lr_w = assembler.assemble(batch_ids, coefs, intercepts)
        with torch.no_grad():
            model(
                inputs["input_ids"],
                inputs["attention_mask"],
                lora_w,
                lr_w,
                output_lr,
            )
        torch.cuda.synchronize(device)

    # Warmup
    print(f"  Warming up ({warmup} iters)...")
    for _ in range(warmup):
        run_batch()

    # Measure
    print(f"  Measuring ({iters} iters)...")
    latencies = []
    for _ in range(iters):
        t0 = time.perf_counter()
        run_batch()
        latencies.append(time.perf_counter() - t0)

    latencies_ms = np.array(latencies) * 1000
    peak_mem_gb = torch.cuda.max_memory_allocated(device) / 1e9
    adapter_mem_gb = store.memory_gb()
    throughput = batch_size / np.mean(latencies)

    return {
        "model": model_name,
        "num_adapters": num_adapters,
        "batch_size": batch_size,
        "lora_rank": lora_rank,
        "seq_len": seq_len,
        "p50_ms": round(float(np.percentile(latencies_ms, 50)), 2),
        "p90_ms": round(float(np.percentile(latencies_ms, 90)), 2),
        "p99_ms": round(float(np.percentile(latencies_ms, 99)), 2),
        "mean_ms": round(float(np.mean(latencies_ms)), 2),
        "throughput_samples_sec": round(throughput, 1),
        "peak_gpu_mem_gb": round(peak_mem_gb, 3),
        "adapter_cache_gb": round(adapter_mem_gb, 3),
    }


def print_table(rows: list[dict]) -> None:
    cols = list(rows[0].keys())
    widths = {c: max(len(c), max(len(str(r[c])) for r in rows)) for c in cols}
    sep = "  ".join("-" * widths[c] for c in cols)
    header = "  ".join(c.ljust(widths[c]) for c in cols)
    print(header)
    print(sep)
    for row in rows:
        print("  ".join(str(row[c]).ljust(widths[c]) for c in cols))


def main():
    parser = argparse.ArgumentParser(description="Multi-tenant LoRA serving benchmark")
    parser.add_argument("--model", default="intfloat/multilingual-e5-small")
    parser.add_argument("--adapters", nargs="+", type=int, default=[100, 500, 1000, 5000])
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[16, 32, 64, 128])
    parser.add_argument("--lora-ranks", nargs="+", type=int, default=[8, 16, 32])
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--num-labels", type=int, default=10)
    parser.add_argument("--out", default="results.csv")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("WARNING: No CUDA GPU detected. Benchmark requires a GPU.")
        return

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Sweep: {len(args.adapters)} adapter counts × {len(args.batch_sizes)} batch sizes × {len(args.lora_ranks)} lora ranks\n")

    results = []
    total = len(args.adapters) * len(args.batch_sizes) * len(args.lora_ranks)
    done = 0

    for num_adapters in args.adapters:
        for batch_size in args.batch_sizes:
            for lora_rank in args.lora_ranks:
                done += 1
                print(f"[{done}/{total}] adapters={num_adapters} batch={batch_size} rank={lora_rank}")
                row = run_single_config(
                    model_name=args.model,
                    num_adapters=num_adapters,
                    batch_size=batch_size,
                    lora_rank=lora_rank,
                    seq_len=args.seq_len,
                    warmup=args.warmup,
                    iters=args.iters,
                    num_labels=args.num_labels,
                )
                results.append(row)
                print(f"  p50={row['p50_ms']}ms  p90={row['p90_ms']}ms  p99={row['p99_ms']}ms  "
                      f"throughput={row['throughput_samples_sec']} samples/s  "
                      f"peak_gpu={row['peak_gpu_mem_gb']}GB\n")

    print("\n=== Results ===")
    print_table(results)

    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"\nSaved to {args.out}")


if __name__ == "__main__":
    main()
