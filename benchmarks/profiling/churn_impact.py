"""Measure adapter-registration impact under provider-managed churn.

This benchmark answers the Finding 7 reviewer question: if adapters are trained
by the SaaS provider rather than fine-tuned by tenants, how much production time
is spent registering adapters under realistic refresh churn?

For refresh churn, it measures both sequential delete+add replacement cycles and
one batched burst replacement after the pool is already at N=1000. In both
cases, every adapter entering the system is paired with one old adapter leaving
and the resident pool size remains fixed.

The run is intentionally capped at N=1000. Building that pool is setup only; the
reported metrics are replacement-churn costs, not cold-start/preload costs.

Example:
    uv run python -m benchmarks.profiling.churn_impact \\
        --model BAAI/bge-m3 --dtype fp16 --tickets-per-sec 30 \\
        --batch-size 32 --latefuse-serving-mean-ms 40.9 \\
        --peft-serving-mean-ms 764.2 \\
        --churn-iters 32 --burst-size 32 \\
        --out benchmarks/results/churn_impact_n1000.csv
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import random
import time
from pathlib import Path

import numpy as np
import torch

from benchmarks.baselines.peft_swap import (
    add_n_adapters,
    build_peft_model,
)
from lora_serving.benchmark.synthetic import make_synthetic_adapters
from lora_serving.config import LoraServingConfig
from lora_serving.weights.store import AdapterStore

DTYPE_MAP = {"fp16": torch.float16, "fp32": torch.float32, "bf16": torch.bfloat16}
N_ADAPTERS = 1000
SECONDS_PER_DAY = 86_400


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def stats_s(values: np.ndarray) -> dict[str, float]:
    return {
        "mean_s": float(np.mean(values)),
        "p50_s": float(np.percentile(values, 50)),
        "p95_s": float(np.percentile(values, 95)),
        "p99_s": float(np.percentile(values, 99)),
        "total_s": float(np.sum(values)),
    }


def time_call_s(fn, device: torch.device) -> float:
    sync(device)
    t0 = time.perf_counter()
    fn()
    sync(device)
    return time.perf_counter() - t0


def peft_replace_one(
    model,
    lora_cfg,
    device: torch.device,
    dtype: torch.dtype,
    old_id: str,
    new_id: str,
) -> None:
    model.set_adapter("adapter_0")
    model.delete_adapter(old_id)
    model.add_adapter(new_id, lora_cfg)
    model.to(device=device, dtype=dtype)


def latefuse_replace_one(store: AdapterStore, old_id: str, new_id: str, seed: int) -> None:
    del store._store[old_id]
    store.load_synthetic(new_id, seed=seed)


def measure_peft(
    model_name: str,
    dtype: torch.dtype,
    device: torch.device,
    lora_rank: int,
    churn_iters: int,
    burst_size: int,
) -> dict[str, float]:
    print(f"\n=== PEFT: setup pool with {N_ADAPTERS} adapters ===", flush=True)
    model, lora_cfg = build_peft_model(model_name, lora_rank, dtype, device)

    add_n_adapters(model, lora_cfg, N_ADAPTERS, device, dtype)
    sync(device)
    print("  setup complete; timing intentionally not recorded here", flush=True)

    if not hasattr(model, "delete_adapter"):
        raise RuntimeError("Installed PEFT version does not expose delete_adapter().")

    print(f"  measuring {churn_iters} delete+add churn cycles", flush=True)
    churn_latencies = np.empty(churn_iters)
    for i in range(churn_iters):
        old_id = f"adapter_{i + 1}"
        new_id = f"adapter_{N_ADAPTERS + i}"
        churn_latencies[i] = time_call_s(
            lambda old=old_id, new=new_id: peft_replace_one(
                model, lora_cfg, device, dtype, old, new
            ),
            device,
        )

    churn = stats_s(churn_latencies)
    print(
        f"  churn cycle mean={churn['mean_s'] * 1000:.2f}ms "
        f"p95={churn['p95_s'] * 1000:.2f}ms "
        f"total={churn['total_s']:.2f}s",
        flush=True,
    )

    burst_old_ids = [f"adapter_{churn_iters + 1 + i}" for i in range(burst_size)]
    burst_new_ids = [f"adapter_{N_ADAPTERS + churn_iters + i}" for i in range(burst_size)]

    def replace_burst() -> None:
        model.set_adapter("adapter_0")
        for old_id in burst_old_ids:
            model.delete_adapter(old_id)
        for new_id in burst_new_ids:
            model.add_adapter(new_id, lora_cfg)
        model.to(device=device, dtype=dtype)

    print(f"  measuring burst replacement of {burst_size} adapters", flush=True)
    burst_s = time_call_s(replace_burst, device)
    print(
        f"  burst total={burst_s:.2f}s "
        f"per_adapter={burst_s / burst_size * 1000:.2f}ms",
        flush=True,
    )

    peak_mem_gb = torch.cuda.max_memory_allocated(device) / 1e9
    del model
    torch.cuda.empty_cache()
    gc.collect()

    return {
        "churn_cycle_mean_s": churn["mean_s"],
        "churn_cycle_p50_s": churn["p50_s"],
        "churn_cycle_p95_s": churn["p95_s"],
        "churn_cycle_p99_s": churn["p99_s"],
        "churn_cycle_total_s": churn["total_s"],
        "churn_cycle_count": float(churn_iters),
        "burst_size": float(burst_size),
        "burst_total_s": burst_s,
        "burst_per_adapter_s": burst_s / burst_size,
        "peak_gpu_mem_gb": peak_mem_gb,
    }


def measure_latefuse(
    model_name: str,
    dtype: torch.dtype,
    device: torch.device,
    lora_rank: int,
    seq_len: int,
    churn_iters: int,
    burst_size: int,
) -> dict[str, float]:
    print(f"\n=== LateFuse: setup pool with {N_ADAPTERS} adapters ===", flush=True)
    config = LoraServingConfig(
        model_name=model_name,
        lora_rank=lora_rank,
        batch_size=1,
        max_seq_len=seq_len,
        target_modules=["query", "value"],
        device=device,
        dtype=dtype,
    )

    store = AdapterStore(config)
    make_synthetic_adapters(store, N_ADAPTERS)
    sync(device)
    print("  setup complete; timing intentionally not recorded here", flush=True)

    print(f"  measuring {churn_iters} delete+add AdapterStore churn cycles", flush=True)
    churn_latencies = np.empty(churn_iters)
    for i in range(churn_iters):
        old_id = f"adapter_{i}"
        new_id = f"adapter_{N_ADAPTERS + i}"
        seed = 42 + N_ADAPTERS + i
        churn_latencies[i] = time_call_s(
            lambda old=old_id, new=new_id, s=seed: latefuse_replace_one(
                store, old, new, s
            ),
            device,
        )
    churn = stats_s(churn_latencies)
    print(
        f"  AdapterStore churn mean={churn['mean_s'] * 1000:.2f}ms "
        f"p95={churn['p95_s'] * 1000:.2f}ms "
        f"total={churn['total_s']:.2f}s",
        flush=True,
    )

    burst_old_ids = [f"adapter_{churn_iters + i}" for i in range(burst_size)]
    burst_new_ids = [f"adapter_{N_ADAPTERS + churn_iters + i}" for i in range(burst_size)]
    burst_seeds = [42 + N_ADAPTERS + churn_iters + i for i in range(burst_size)]

    def replace_burst() -> None:
        for old_id in burst_old_ids:
            del store._store[old_id]
        for new_id, seed in zip(burst_new_ids, burst_seeds, strict=True):
            store.load_synthetic(new_id, seed=seed)

    print(f"  measuring burst replacement of {burst_size} adapters", flush=True)
    burst_s = time_call_s(replace_burst, device)
    print(
        f"  burst total={burst_s:.2f}s "
        f"per_adapter={burst_s / burst_size * 1000:.2f}ms",
        flush=True,
    )

    peak_mem_gb = torch.cuda.max_memory_allocated(device) / 1e9
    adapter_cache_gb = store.memory_gb()

    del store
    torch.cuda.empty_cache()
    gc.collect()

    return {
        "churn_cycle_mean_s": churn["mean_s"],
        "churn_cycle_p50_s": churn["p50_s"],
        "churn_cycle_p95_s": churn["p95_s"],
        "churn_cycle_p99_s": churn["p99_s"],
        "churn_cycle_total_s": churn["total_s"],
        "churn_cycle_count": float(churn_iters),
        "burst_size": float(burst_size),
        "burst_total_s": burst_s,
        "burst_per_adapter_s": burst_s / burst_size,
        "peak_gpu_mem_gb": peak_mem_gb,
        "adapter_cache_gb": adapter_cache_gb,
    }


def impact_rows(
    system: str,
    metrics: dict[str, float],
    tickets_per_sec: float,
    batch_size: int,
    serving_mean_ms: float,
    replicas: int,
    refresh_days: list[float],
) -> list[dict[str, float | str]]:
    tickets_per_day = tickets_per_sec * SECONDS_PER_DAY
    batches_per_day = tickets_per_day / batch_size
    serving_compute_s_per_day = batches_per_day * serving_mean_ms / 1000

    rows = []
    for days in refresh_days:
        adapter_updates_per_day = N_ADAPTERS / days
        register_s_per_day = (
            adapter_updates_per_day
            * metrics["churn_cycle_mean_s"]
            * replicas
        )
        total_busy_s_per_day = serving_compute_s_per_day + register_s_per_day
        register_share_pct = (
            100 * register_s_per_day / total_busy_s_per_day
            if total_busy_s_per_day else 0.0
        )
        register_wallclock_day_pct = 100 * register_s_per_day / SECONDS_PER_DAY

        rows.append(
            {
                "system": system,
                "num_adapters": N_ADAPTERS,
                "refresh_days": days,
                "adapter_updates_per_day": adapter_updates_per_day,
                "tickets_per_sec": tickets_per_sec,
                "batch_size": batch_size,
                "serving_mean_ms_source": "cli",
                "replicas": replicas,
                "serving_mean_ms": serving_mean_ms,
                "serving_compute_s_per_day": serving_compute_s_per_day,
                "churn_cycle_mean_ms": metrics["churn_cycle_mean_s"] * 1000,
                "churn_cycle_p50_ms": metrics["churn_cycle_p50_s"] * 1000,
                "churn_cycle_p95_ms": metrics["churn_cycle_p95_s"] * 1000,
                "churn_cycle_samples": metrics["churn_cycle_count"],
                "burst_size": metrics["burst_size"],
                "burst_total_s": metrics["burst_total_s"],
                "burst_per_adapter_ms": metrics["burst_per_adapter_s"] * 1000,
                "adapter_churn_s_per_day": register_s_per_day,
                "register_s_per_day": register_s_per_day,
                "register_share_pct": register_share_pct,
                "register_wallclock_day_pct": register_wallclock_day_pct,
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="BAAI/bge-m3")
    parser.add_argument("--dtype", choices=list(DTYPE_MAP), default="fp16")
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--serving-mean-ms",
        type=float,
        default=None,
        help="Mean per-batch serving latency from existing serving benchmarks. "
             "Used for both systems unless a system-specific value is set. "
             "This script does not measure serving latency.",
    )
    parser.add_argument(
        "--peft-serving-mean-ms",
        type=float,
        default=None,
        help="PEFT mean per-batch serving latency from the existing PEFT benchmark.",
    )
    parser.add_argument(
        "--latefuse-serving-mean-ms",
        type=float,
        default=None,
        help="LateFuse mean per-batch serving latency from the existing benchmark.",
    )
    parser.add_argument(
        "--churn-iters",
        type=int,
        default=32,
        help="Number of post-N=1000 delete+add replacement cycles to time. "
             "These live churn measurements are used for the churn-rate calculation.",
    )
    parser.add_argument(
        "--burst-size",
        type=int,
        default=32,
        help="Number of adapters to replace in one batched delete-then-add burst. "
             "This reports rollout blocking time; daily churn share still uses "
             "the sequential churn-cycle mean.",
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--tickets-per-sec", type=float, default=30.0)
    parser.add_argument("--replicas", type=int, default=1)
    parser.add_argument("--refresh-days", nargs="+", type=float, default=[7.0, 30.0, 90.0])
    parser.add_argument("--skip-peft", action="store_true")
    parser.add_argument("--skip-latefuse", action="store_true")
    parser.add_argument("--out", default="benchmarks/results/churn_impact_n1000.csv")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("Need CUDA for this benchmark.")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda:0")
    dtype = DTYPE_MAP[args.dtype]
    print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
    print(f"N fixed at {N_ADAPTERS}; batch_size={args.batch_size}", flush=True)
    if args.churn_iters <= 0:
        raise SystemExit("--churn-iters must be positive.")
    if args.churn_iters >= N_ADAPTERS:
        raise SystemExit("--churn-iters must be less than N=1000.")
    if args.burst_size <= 0:
        raise SystemExit("--burst-size must be positive.")
    if args.churn_iters + args.burst_size >= N_ADAPTERS:
        raise SystemExit("--churn-iters + --burst-size must be less than N=1000.")

    peft_serving_mean_ms = args.peft_serving_mean_ms or args.serving_mean_ms
    latefuse_serving_mean_ms = args.latefuse_serving_mean_ms or args.serving_mean_ms
    if not args.skip_peft and not peft_serving_mean_ms:
        raise SystemExit("Provide --peft-serving-mean-ms or --serving-mean-ms.")
    if not args.skip_latefuse and not latefuse_serving_mean_ms:
        raise SystemExit("Provide --latefuse-serving-mean-ms or --serving-mean-ms.")
    if peft_serving_mean_ms is not None and peft_serving_mean_ms <= 0:
        raise SystemExit("PEFT serving latency must be positive.")
    if latefuse_serving_mean_ms is not None and latefuse_serving_mean_ms <= 0:
        raise SystemExit("LateFuse serving latency must be positive.")

    raw: dict[str, dict[str, float]] = {}
    rows: list[dict] = []

    if not args.skip_peft:
        peft_metrics = measure_peft(
            args.model,
            dtype,
            device,
            args.lora_rank,
            args.churn_iters,
            args.burst_size,
        )
        raw["PEFT"] = peft_metrics
        rows.extend(
            impact_rows(
                "PEFT",
                peft_metrics,
                args.tickets_per_sec,
                args.batch_size,
                peft_serving_mean_ms,
                args.replicas,
                args.refresh_days,
            )
        )

    if not args.skip_latefuse:
        latefuse_metrics = measure_latefuse(
            args.model,
            dtype,
            device,
            args.lora_rank,
            args.seq_len,
            args.churn_iters,
            args.burst_size,
        )
        raw["LateFuse"] = latefuse_metrics
        rows.extend(
            impact_rows(
                "LateFuse",
                latefuse_metrics,
                args.tickets_per_sec,
                args.batch_size,
                latefuse_serving_mean_ms,
                args.replicas,
                args.refresh_days,
            )
        )

    if not rows:
        raise SystemExit("Nothing to write: both systems were skipped.")

    out = Path(args.out)
    write_csv(out, rows)
    json_out = out.with_suffix(".json")
    write_json(
        json_out,
        {
            "config": {
                "model": args.model,
                "dtype": args.dtype,
                "num_adapters": N_ADAPTERS,
                "batch_size": args.batch_size,
                "lora_rank": args.lora_rank,
                "seq_len": args.seq_len,
                "serving_mean_ms": args.serving_mean_ms,
                "peft_serving_mean_ms": peft_serving_mean_ms,
                "latefuse_serving_mean_ms": latefuse_serving_mean_ms,
                "churn_iters": args.churn_iters,
                "burst_size": args.burst_size,
                "tickets_per_sec": args.tickets_per_sec,
                "replicas": args.replicas,
                "refresh_days": args.refresh_days,
            },
            "raw_measurements": raw,
            "impact_rows": rows,
        },
    )

    print("\n=== Registration impact ===", flush=True)
    for row in rows:
        print(
            f"{row['system']:16s} refresh={row['refresh_days']:>5g}d "
            f"register={row['register_s_per_day']:.2f}s/day "
            f"serving_share={row['register_share_pct']:.2f}% "
            f"wallclock_share={row['register_wallclock_day_pct']:.3f}%",
            flush=True,
        )

    if len(raw) == 2:
        churn_speedup = raw["PEFT"]["churn_cycle_mean_s"] / raw["LateFuse"]["churn_cycle_mean_s"]
        print(f"\nChurn-cycle speedup: {churn_speedup:.1f}x", flush=True)

    print(f"\nSaved CSV:  {out}", flush=True)
    print(f"Saved JSON: {json_out}", flush=True)


if __name__ == "__main__":
    main()
