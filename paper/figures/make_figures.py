"""Generate the figures referenced by paper/main.tex.

Figure 2 (latency_adapters): p50 latency vs adapter count, one curve per batch size.

Written to paper/figures/ as PDF (preferred by LaTeX) and PNG.
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt

HERE = Path(__file__).parent
RESULTS = HERE.parent.parent / "benchmarks" / "results"


def load_csv(path: Path) -> list[dict]:
    with open(path) as f:
        return list(csv.DictReader(f))


def figure_latency_vs_adapters() -> None:
    """p50 latency vs N, one curve per batch size, fixed rank=8."""
    rows = load_csv(RESULTS / "sweep_main.csv")
    rows = [r for r in rows if int(r["lora_rank"]) == 8]

    by_batch: dict[int, list[tuple[int, float]]] = {}
    for r in rows:
        b = int(r["batch_size"])
        n = int(r["num_adapters"])
        p50 = float(r["p50_ms"])
        by_batch.setdefault(b, []).append((n, p50))

    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    for b in sorted(by_batch):
        pts = sorted(by_batch[b])
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        ax.plot(xs, ys, marker="o", label=f"B={b}", linewidth=1.4, markersize=5)

    ax.set_xscale("log")
    ax.set_xlabel("Pre-loaded adapter count $N$ (log scale)")
    ax.set_ylabel("p50 latency (ms)")
    ax.set_title("Latency is independent of adapter pool size")
    ax.grid(True, which="both", linestyle=":", linewidth=0.5, alpha=0.6)
    ax.legend(loc="center right", fontsize=8, ncol=1)
    ax.set_ylim(bottom=0)

    fig.tight_layout()
    fig.savefig(HERE / "latency_adapters.pdf")
    fig.savefig(HERE / "latency_adapters.png", dpi=150)
    plt.close(fig)
    print(f"wrote {HERE / 'latency_adapters.pdf'}")


if __name__ == "__main__":
    figure_latency_vs_adapters()
