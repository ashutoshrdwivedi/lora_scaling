"""Generate the two figures referenced by paper/main.tex.

Figure 2 (latency_adapters): p50 latency vs adapter count, one curve per batch size.
Figure 3 (throughput_baselines): grouped bar chart of throughput, ours vs PEFT.

Both are written to paper/figures/ as PDF (preferred by LaTeX) and PNG.
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


def figure_throughput_baselines() -> None:
    """Grouped bar: throughput at three batch sizes, ours vs PEFT, two N values."""
    main = load_csv(RESULTS / "sweep_main.csv")
    peft = load_csv(RESULTS / "peft_seq.csv")

    def get_ours(n: int, b: int) -> float:
        for r in main:
            if (int(r["num_adapters"]) == n
                and int(r["batch_size"]) == b
                and int(r["lora_rank"]) == 8):
                return float(r["throughput_samples_sec"])
        raise KeyError(f"ours N={n} B={b}")

    def get_peft(n: int, b: int) -> float:
        for r in peft:
            if int(r["num_adapters"]) == n and int(r["batch_size"]) == b:
                return float(r["throughput_samples_sec"])
        raise KeyError(f"peft N={n} B={b}")

    batches = [8, 32, 128]
    n_vals = [100, 1000]

    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.4), sharey=True)
    for ax, n in zip(axes, n_vals):
        ours = [get_ours(n, b) for b in batches]
        pefts = [get_peft(n, b) for b in batches]

        x = list(range(len(batches)))
        width = 0.36
        ax.bar([xi - width / 2 for xi in x], pefts, width, label="PEFT seq.",
               color="#c44e52", edgecolor="black", linewidth=0.4)
        ax.bar([xi + width / 2 for xi in x], ours, width, label="LateFuse",
               color="#4c72b0", edgecolor="black", linewidth=0.4)
        for xi, (p, o) in enumerate(zip(pefts, ours)):
            ax.text(xi - width / 2, p + 18, f"{p:.0f}", ha="center", fontsize=7.5)
            ax.text(xi + width / 2, o + 18, f"{o:.0f}", ha="center", fontsize=7.5)

        ax.set_xticks(x)
        ax.set_xticklabels([f"B={b}" for b in batches])
        ax.set_title(f"$N = {n:,}$ adapters")
        ax.grid(True, axis="y", linestyle=":", linewidth=0.5, alpha=0.6)
        ax.set_ylim(0, 1000)

    axes[0].set_ylabel("Throughput (samples / sec)")
    axes[1].legend(loc="upper left", fontsize=9)
    fig.suptitle("Throughput: LateFuse vs PEFT per-request swap (A100-40GB, rank=8)",
                 fontsize=10, y=1.02)
    fig.tight_layout()
    fig.savefig(HERE / "throughput_baselines.pdf", bbox_inches="tight")
    fig.savefig(HERE / "throughput_baselines.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {HERE / 'throughput_baselines.pdf'}")


if __name__ == "__main__":
    figure_latency_vs_adapters()
    figure_throughput_baselines()
