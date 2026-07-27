#!/usr/bin/env python3
"""Aggregate the L40S run into the grid, rank, and vs-PEFT tables.

The N=28000 column lives in a separate CSV because the original sweep OOM'd on
it (allocator fragmentation, since fixed by exporting expandable_segments in
run_rebuttal_l40s.sh). The re-run used the identical --warmup 50 --iters 200
protocol, so the two files concatenate without a protocol caveat.

Usage: python summarize_l40s.py [--dir .]
"""
import argparse
import csv
import statistics
from collections import defaultdict
from pathlib import Path

SWEEPS = ["sweep_bgem3_l40s.csv", "sweep_bgem3_l40s_n28k_expseg.csv"]
NS = [100, 1000, 5000, 10000, 20000, 28000]
BS = [8, 16, 32, 64, 128]


def load_latefuse(d):
    agg = defaultdict(list)
    for fn in SWEEPS:
        p = d / fn
        if not p.exists():
            continue
        for r in csv.DictReader(p.open()):
            key = (int(r["num_adapters"]), int(r["batch_size"]), int(r["lora_rank"]))
            agg[key].append(r)
    return agg


def med(rows, col):
    return statistics.median(float(r[col]) for r in rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=".")
    args = ap.parse_args()
    d = Path(args.dir)

    lf = load_latefuse(d)
    out = []

    # Table A: the O(1)-in-N grid, rank 8.
    for N in NS:
        for B in BS:
            rows = lf.get((N, B, 8))
            if not rows:
                continue
            out.append({
                "table": "grid_rank8",
                "num_adapters": N,
                "batch_size": B,
                "lora_rank": 8,
                "p50_ms": round(med(rows, "p50_ms"), 2),
                "p99_ms": round(med(rows, "p99_ms"), 2),
                "throughput_samples_sec": round(med(rows, "throughput_samples_sec"), 1),
                "assemble_share_pct": round(med(rows, "assemble_share_pct"), 1),
                "peak_gpu_mem_gb": round(med(rows, "peak_gpu_mem_gb"), 3),
                "n_seeds": len(rows),
            })

    # Table B: rank sensitivity at the (1000, 32) operating point.
    for rk in [4, 8, 16, 32]:
        rows = lf.get((1000, 32, rk))
        if not rows:
            continue
        out.append({
            "table": "rank_sweep",
            "num_adapters": 1000,
            "batch_size": 32,
            "lora_rank": rk,
            "p50_ms": round(med(rows, "p50_ms"), 2),
            "p99_ms": round(med(rows, "p99_ms"), 2),
            "throughput_samples_sec": round(med(rows, "throughput_samples_sec"), 1),
            "assemble_share_pct": round(med(rows, "assemble_share_pct"), 1),
            "peak_gpu_mem_gb": round(med(rows, "peak_gpu_mem_gb"), 3),
            "n_seeds": len(rows),
        })

    # Table C: LateFuse vs PEFT at the cells the PEFT arm covers.
    peft = {}
    p = d / "peft_mixed_l40s.csv"
    if p.exists():
        for r in csv.DictReader(p.open()):
            peft[(int(r["num_adapters"]), int(r["batch_size"]))] = r
    for (N, B), pr in sorted(peft.items()):
        rows = lf.get((N, B, 8))
        if not rows:
            continue
        l50 = med(rows, "p50_ms")
        p50 = float(pr["p50_ms"])
        out.append({
            "table": "vs_peft_mixed",
            "num_adapters": N,
            "batch_size": B,
            "lora_rank": 8,
            "p50_ms": round(l50, 2),
            "peft_p50_ms": round(p50, 2),
            "speedup_x": round(p50 / l50, 2),
            "peft_add_adapter_total_s": round(float(pr["add_adapter_total_s"]), 2),
            "n_seeds": len(rows),
        })

    cols = ["table", "num_adapters", "batch_size", "lora_rank", "p50_ms", "p99_ms",
            "throughput_samples_sec", "assemble_share_pct", "peak_gpu_mem_gb",
            "peft_p50_ms", "speedup_x", "peft_add_adapter_total_s", "n_seeds"]
    dest = d / "summary_l40s.csv"
    with dest.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(out)
    print(f"wrote {dest} ({len(out)} rows)")

    grid = [r for r in out if r["table"] == "grid_rank8"]
    for B in BS:
        vals = [r["p50_ms"] for r in grid if r["batch_size"] == B]
        if vals:
            print(f"  B={B:<4} p50 {min(vals):6.2f}..{max(vals):6.2f} ms  "
                  f"spread {(max(vals) - min(vals)) / min(vals) * 100:+.1f}% over N=100..28000")


if __name__ == "__main__":
    main()
