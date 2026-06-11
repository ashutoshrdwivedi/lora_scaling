"""
Generate the paper's PDF figures from benchmark CSVs.

Produces:
    paper/figures/latency_adapters.pdf      -- p50 latency vs adapter count,
        one curve per batch size, mean +/- s.d. over the seeded repeats.
    paper/figures/throughput_baselines.pdf  -- QPS by system (PEFT grouped /
        mixed / homogeneous vs LateFuse) at B=32, N=100 and N=1,000.

Data sources are the same CSVs that feed Tables 2-3 (sweep_main.csv,
peft_*_sxm80.csv, sysname_sxm80_ref.csv), so figures and tables can't drift.
Multi-seed CSVs (a 'seed' column) are aggregated to mean +/- sample s.d.

Run:
    uv run --extra paper python paper/build_figures.py
"""

from __future__ import annotations

import csv
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parent.parent
RESULTS = REPO / "benchmarks" / "results"
FIG_DIR = REPO / "paper" / "figures"


def load_csv(name: str) -> list[dict[str, str]]:
    with (RESULTS / name).open() as f:
        return list(csv.DictReader(f))


def latency_adapters() -> Path:
    """p50 latency vs num_adapters, one line per batch size (r=8)."""
    # group seeded repeats: (batch, N) -> list of per-seed p50
    by_cfg: dict[tuple[int, int], list[float]] = defaultdict(list)
    for r in load_csv("sweep_main.csv"):
        if r["lora_rank"] != "8":
            continue
        key = (int(r["batch_size"]), int(r["num_adapters"]))
        by_cfg[key].append(float(r["p50_ms"]))

    batches = sorted({b for b, _ in by_cfg})
    fig, ax = plt.subplots(figsize=(5.0, 3.4))
    for b in batches:
        ns = sorted(n for bb, n in by_cfg if bb == b)
        means = [statistics.fmean(by_cfg[(b, n)]) for n in ns]
        stds = [statistics.stdev(by_cfg[(b, n)]) if len(by_cfg[(b, n)]) > 1
                else 0.0 for n in ns]
        ax.errorbar(ns, means, yerr=stds, marker="o", linewidth=1.3,
                    markersize=4, capsize=2, label=f"B = {b}")

    ax.set_xscale("log")
    ax.set_xlabel("Pre-loaded adapter count $N$")
    ax.set_ylabel("p50 latency (ms)")
    ax.grid(True, which="both", linewidth=0.4, alpha=0.5)
    ax.legend(frameon=False, loc="upper left", fontsize=8)
    ax.set_ylim(bottom=0)
    fig.tight_layout()

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = FIG_DIR / "latency_adapters.pdf"
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def throughput_baselines() -> Path:
    """QPS by system at B=32: PEFT grouped/mixed/homog vs LateFuse, N=100/1k."""
    systems = [
        ("PEFT grouped", load_csv("peft_grouped_sxm80.csv")),
        ("PEFT mixed", load_csv("peft_mixed_sxm80.csv")),
        ("PEFT homog.", load_csv("peft_homogeneous_sxm80.csv")),
        ("LateFuse", load_csv("sysname_sxm80_ref.csv")),
    ]

    def qps(rows: list[dict[str, str]], n: int) -> float:
        out = [float(r["throughput_samples_sec"]) for r in rows
               if r["num_adapters"] == str(n) and r["batch_size"] == "32"]
        assert len(out) == 1, f"expected 1 row at N={n} B=32, got {len(out)}"
        return out[0]

    fig, ax = plt.subplots(figsize=(5.0, 3.4))
    width = 0.38
    xs = range(len(systems))
    for n, shift in [(100, -width / 2), (1000, width / 2)]:
        vals = [qps(rows, n) for _, rows in systems]
        bars = ax.bar([x + shift for x in xs], vals, width,
                      label=f"$N$ = {n:,}")
        ax.bar_label(bars, fmt=lambda v: f"{v:.0f}" if v >= 10 else f"{v:.1f}",
                     fontsize=7, padding=2)

    ax.set_yscale("log")
    ax.set_xticks(list(xs), [name for name, _ in systems], fontsize=9)
    ax.set_ylabel("Throughput (samples/sec, log scale)")
    ax.grid(True, axis="y", which="both", linewidth=0.4, alpha=0.5)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = FIG_DIR / "throughput_baselines.pdf"
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> int:
    for fn in (latency_adapters, throughput_baselines):
        print(f"wrote {fn()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
