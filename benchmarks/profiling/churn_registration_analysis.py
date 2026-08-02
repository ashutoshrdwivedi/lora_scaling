"""Measure Zipf-induced AdapterStore churn for vanilla BatchAssembler serving.

The tenant population is fixed.  A Zipf distribution produces the request
stream; an LRU policy decides which adapters remain in the fixed-capacity GPU
store.  Every cache miss performs a real ``AdapterStore.load_synthetic`` call,
and every capacity eviction removes the corresponding real store entry.

The reported percentage answers the serial-cost question:

    runtime AdapterStore registration/eviction time
    ------------------------------------------------
    freshly benchmarked serving time + runtime AdapterStore registration/eviction time

The initial preload is deliberately excluded: it is startup provisioning, not
request-time churn.
"""

from __future__ import annotations

import argparse
import csv
import glob
import math
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


DTYPE_NAMES = {
    "float16": "fp16", "torch.float16": "fp16", "fp16": "fp16",
    "bfloat16": "bf16", "torch.bfloat16": "bf16", "bf16": "bf16",
    "float32": "fp32", "torch.float32": "fp32", "fp32": "fp32",
}

EXPERIMENT_PROFILE_PATHS = (
    "benchmarks/results/sweep_main.csv",
    "benchmarks/results/sweep_capacity.csv",
)


@dataclass(frozen=True)
class ServingProfile:
    source: str
    model: str
    dtype: str
    num_adapters: int
    batch_size: int
    lora_rank: int
    seq_len: int


@dataclass
class ChurnResult:
    request_samples: int
    request_batches: int
    request_hits: int
    request_misses: int
    admissions: int
    evictions: int
    registration_time_ms: float
    eviction_time_ms: float
    final_resident_adapters: int


def parse_float(row: dict[str, str], key: str) -> float | None:
    try:
        return float(row[key]) if row.get(key, "") else None
    except ValueError:
        return None


def parse_int(row: dict[str, str], key: str) -> int | None:
    value = parse_float(row, key)
    return int(value) if value is not None else None


def load_serving_profiles(patterns: Iterable[str]) -> list[ServingProfile]:
    """Read unique experiment shapes; CSV latency values are intentionally ignored."""
    profiles: list[ServingProfile] = []
    seen: set[tuple] = set()
    for pattern in patterns:
        for path_str in glob.glob(pattern, recursive=True):
            path = Path(path_str)
            with path.open(newline="") as handle:
                reader = csv.DictReader(handle)
                if not reader.fieldnames:
                    continue
                for row in reader:
                    num_adapters = parse_int(row, "num_adapters")
                    batch_size = parse_int(row, "batch_size")
                    lora_rank = parse_int(row, "lora_rank")
                    seq_len = parse_int(row, "seq_len")
                    model = row.get("model", "")
                    if row.get("status", "ok") != "ok":
                        continue
                    if not all((model, num_adapters, batch_size, lora_rank, seq_len)):
                        continue
                    profile = ServingProfile(
                        source=str(path), model=model, dtype=row.get("dtype", ""),
                        num_adapters=num_adapters, batch_size=batch_size,
                        lora_rank=lora_rank, seq_len=seq_len,
                    )
                    key = (profile.model, profile.dtype, profile.num_adapters,
                           profile.batch_size, profile.lora_rank, profile.seq_len)
                    if key not in seen:
                        seen.add(key)
                        profiles.append(profile)
    return profiles


def torch_dtype(dtype_name: str):
    import torch

    normalized = DTYPE_NAMES.get(dtype_name, dtype_name)
    dtypes = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}
    if normalized not in dtypes:
        raise ValueError(f"Unsupported dtype: {dtype_name}")
    return dtypes[normalized]


def sync_device(device) -> None:
    import torch

    if device.type == "cuda":
        torch.cuda.synchronize(device)


def harmonic_probs(num_tenants: int, alpha: float) -> np.ndarray:
    ranks = np.arange(1, num_tenants + 1, dtype=np.float64)
    probabilities = 1.0 / np.power(ranks, alpha)
    return probabilities / probabilities.sum()


def measure_serving_time(
    *,
    profile: ServingProfile,
    capacity: int,
    warmup: int,
    iters: int,
    target_modules: list[str],
    seed: int,
) -> dict:
    """Run the project's vanilla benchmark for this exact serving shape."""
    from lora_serving.benchmark.run import run_single_config

    return run_single_config(
        model_name=profile.model,
        num_adapters=capacity,
        batch_size=profile.batch_size,
        lora_rank=profile.lora_rank,
        seq_len=profile.seq_len,
        warmup=warmup,
        iters=iters,
        dtype=torch_dtype(profile.dtype),
        seed=seed,
        target_modules=target_modules,
    )


def replay_zipf_churn(
    *,
    profile: ServingProfile,
    num_tenants: int,
    capacity: int,
    duration_s: float,
    request_qps: float,
    zipf_alpha: float,
    target_modules: list[str],
    device_name: str,
    seed: int,
) -> ChurnResult:
    """Replay a Zipf request stream, mutating AdapterStore on every miss.

    The store starts with the most popular ``capacity`` tenants.  This makes
    the measured operations request-time churn only, excluding startup loads.
    """
    import torch

    from lora_serving.config import LoraServingConfig
    from lora_serving.weights.store import AdapterStore

    if num_tenants <= 0:
        raise ValueError("num_tenants must be positive")
    if capacity < 0:
        raise ValueError("capacity must be non-negative")
    capacity = min(capacity, num_tenants)
    device = torch.device(device_name)
    config = LoraServingConfig(
        model_name=profile.model, lora_rank=profile.lora_rank,
        batch_size=profile.batch_size, max_seq_len=profile.seq_len,
        target_modules=target_modules, device=device, dtype=torch_dtype(profile.dtype),
    )
    store = AdapterStore(config)
    resident: OrderedDict[int, None] = OrderedDict()

    # Warm start: rank 0 is the most-requested tenant.  Do not time this setup.
    for tenant in range(capacity):
        store.load_synthetic(str(tenant), seed=seed + tenant)
        resident[tenant] = None
    sync_device(device)

    rng = np.random.default_rng(seed)
    probabilities = harmonic_probs(num_tenants, zipf_alpha)
    num_batches = int(round(duration_s * request_qps / profile.batch_size))
    request_hits = request_misses = admissions = evictions = 0
    registration_time_ms = eviction_time_ms = 0.0

    def timed(operation) -> float:
        sync_device(device)
        start = time.perf_counter()
        operation()
        sync_device(device)
        return (time.perf_counter() - start) * 1000.0

    for batch_idx in range(num_batches):
        batch = rng.choice(num_tenants, size=profile.batch_size, p=probabilities)
        for tenant_raw in batch:
            tenant = int(tenant_raw)
            if tenant in resident:
                request_hits += 1
                resident.move_to_end(tenant)
                continue

            request_misses += 1
            if len(resident) == capacity and capacity:
                victim, _ = resident.popitem(last=False)
                eviction_time_ms += timed(lambda victim=victim: store.evict(str(victim)))
                evictions += 1
            if capacity:
                registration_time_ms += timed(
                    lambda tenant=tenant, batch_idx=batch_idx: store.load_synthetic(
                        str(tenant), seed=seed + num_tenants + batch_idx * profile.batch_size + tenant
                    )
                )
                resident[tenant] = None
                admissions += 1

    return ChurnResult(
        request_samples=num_batches * profile.batch_size,
        request_batches=num_batches,
        request_hits=request_hits,
        request_misses=request_misses,
        admissions=admissions,
        evictions=evictions,
        registration_time_ms=registration_time_ms,
        eviction_time_ms=eviction_time_ms,
        final_resident_adapters=len(store),
    )


def fmt_float(value: float | None, digits: int = 6) -> str:
    return "" if value is None or math.isnan(value) else f"{value:.{digits}f}"


def build_rows(args: argparse.Namespace) -> list[dict[str, str]]:
    profiles = [p for p in load_serving_profiles(EXPERIMENT_PROFILE_PATHS) if p.model in args.model]
    if args.batch_size:
        profiles = [p for p in profiles if p.batch_size in args.batch_size]
    if args.lora_rank:
        profiles = [p for p in profiles if p.lora_rank in args.lora_rank]
    if args.seq_len:
        profiles = [p for p in profiles if p.seq_len in args.seq_len]
    if args.max_profiles is not None:
        profiles = profiles[:args.max_profiles]
    if not profiles:
        raise SystemExit("No serving profiles found. Check --results-glob and filters.")

    rows: list[dict[str, str]] = []
    for profile_idx, profile in enumerate(profiles):
        for capacity in args.capacity or [profile.num_adapters]:
            capacity = min(capacity, args.num_tenants)
            if capacity <= 0:
                raise ValueError("capacity must be positive")
            serving = measure_serving_time(
                profile=profile,
                capacity=capacity,
                warmup=args.serving_warmup,
                iters=args.serving_iters,
                target_modules=args.target_modules,
                seed=args.seed + profile_idx * 1009,
            )
            for qps_idx, request_qps in enumerate(args.request_qps):
                churn = replay_zipf_churn(
                    profile=profile, num_tenants=args.num_tenants, capacity=capacity,
                    duration_s=args.duration_s, request_qps=request_qps,
                    zipf_alpha=args.zipf_alpha, target_modules=args.target_modules,
                    device_name=args.device, seed=args.seed + profile_idx * 1009 + qps_idx,
                )
                serving_time_ms = churn.request_batches * serving["mean_ms"]
                adapter_store_time_ms = churn.registration_time_ms + churn.eviction_time_ms
                total_time_ms = serving_time_ms + adapter_store_time_ms
                share = 100.0 * adapter_store_time_ms / total_time_ms if total_time_ms else 0.0
                hit_rate = churn.request_hits / churn.request_samples if churn.request_samples else 0.0
                rows.append({
                    "source": profile.source, "model": profile.model, "dtype": profile.dtype,
                    "profile_num_adapters": str(profile.num_adapters), "capacity": str(capacity),
                    "batch_size": str(profile.batch_size), "lora_rank": str(profile.lora_rank),
                    "seq_len": str(profile.seq_len), "serving_measurement": "run_single_config",
                    "mean_ms": fmt_float(serving["mean_ms"], 3),
                    "assemble_mean_ms": fmt_float(serving["assemble_mean_ms"], 3),
                    "forward_mean_ms": fmt_float(serving["forward_mean_ms"], 3),
                    "duration_s": fmt_float(args.duration_s, 1), "request_qps": fmt_float(request_qps, 1),
                    "zipf_alpha": fmt_float(args.zipf_alpha, 3), "num_tenants": str(args.num_tenants),
                    "warm_start": "top_capacity_zipf_ranks", "request_samples": str(churn.request_samples),
                    "request_batches": str(churn.request_batches), "request_hit_rate": fmt_float(hit_rate),
                    "request_misses": str(churn.request_misses), "admissions": str(churn.admissions),
                    "evictions": str(churn.evictions),
                    "registration_time_ms": fmt_float(churn.registration_time_ms),
                    "eviction_time_ms": fmt_float(churn.eviction_time_ms),
                    "adapter_store_time_ms": fmt_float(adapter_store_time_ms),
                    "serving_time_ms": fmt_float(serving_time_ms, 3),
                    "registration_share_pct": fmt_float(share),
                    "final_resident_adapters": str(churn.final_resident_adapters),
                })
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="benchmarks/results/churn_registration_analysis.csv")
    parser.add_argument("--model", nargs="+", default=["BAAI/bge-m3"])
    parser.add_argument("--batch-size", nargs="+", type=int)
    parser.add_argument("--lora-rank", nargs="+", type=int)
    parser.add_argument("--seq-len", nargs="+", type=int)
    parser.add_argument("--max-profiles", type=int)
    parser.add_argument("--num-tenants", type=int, default=50_000)
    parser.add_argument("--capacity", nargs="+", type=int)
    parser.add_argument("--zipf-alpha", type=float, default=1.1)
    parser.add_argument("--duration-s", type=float, default=600.0)
    parser.add_argument("--request-qps", nargs="+", type=float, default=[1000.0, 5000.0, 10000.0])
    parser.add_argument(
        "--serving-warmup", type=int, default=50,
        help="Warmup iterations passed to run_single_config.",
    )
    parser.add_argument(
        "--serving-iters", type=int, default=200,
        help="Measured iterations passed to run_single_config.",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--target-modules", nargs="+", default=["query", "value"])
    parser.add_argument("--seed", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.device != "cuda:0":
        raise SystemExit("run_single_config currently benchmarks only cuda:0; use --device cuda:0.")
    rows = build_rows(args)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {out}")


if __name__ == "__main__":
    main()
