"""
Generate the paper's PDF figures from benchmark CSVs.

Produces paper/figures/latency_adapters.pdf showing p50 latency vs
adapter count, one curve per batch size. Data source is the same
sweep_main.csv that feeds Table 2, so figure and table can't drift.

Run:
    uv run --extra paper python paper/build_figures.py
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parent.parent
RESULTS = REPO / "benchmarks" / "results"
FIG_DIR = REPO / "paper" / "figures"


def load_sweep_main() -> list[dict[str, str]]:
    with (RESULTS / "sweep_main.csv").open() as f:
        return list(csv.DictReader(f))


def latency_adapters(rows: list[dict[str, str]]) -> Path:
    """p50 latency vs num_adapters, one line per batch size (r=8)."""
    by_batch: dict[int, list[tuple[int, float]]] = defaultdict(list)
    for r in rows:
        if r["lora_rank"] != "8":
            continue
        by_batch[int(r["batch_size"])].append(
            (int(r["num_adapters"]), float(r["p50_ms"]))
        )

    fig, ax = plt.subplots(figsize=(5.0, 3.4))
    for b in sorted(by_batch):
        pts = sorted(by_batch[b])
        xs = [n for n, _ in pts]
        ys = [p for _, p in pts]
        ax.plot(xs, ys, marker="o", linewidth=1.3, markersize=4,
                label=f"B = {b}")

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
    plt.close(fig)
    return out


def main() -> int:
    rows = load_sweep_main()
    out = latency_adapters(rows)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
