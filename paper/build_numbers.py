"""
Generate LaTeX fragments from benchmark CSVs/JSON so the paper stays in sync
with the data when benchmarks are re-run.

Outputs (overwrites on every run):
    paper/numbers.tex             -- \\newcommand macros (inline prose numbers)
    paper/table_main.tex          -- \\begin{tabular}...\\end{tabular} body
    paper/table_baselines.tex     -- ditto
    paper/table_accuracy.tex      -- ditto
    paper/table_accuracy_bge.tex  -- ditto (only once the bge-m3 run exists)

Run:
    uv run python paper/build_numbers.py

To wire the macros into main.tex (one-time):
    \\input{numbers.tex}                in the preamble
    \\input{table_main.tex}             inside the existing \\begin{table}...
    \\input{table_baselines.tex}
    \\input{table_accuracy.tex}

Prose numbers in main.tex should then reference the macros (e.g.
\\HeadlineQPS instead of '810'). All macros live in paper/numbers.tex with
sectioned comments showing their source CSV row.

----------------------------------------------------------------------------
SOURCE-OF-TRUTH MAP (one entry per number that appears in main.tex)
----------------------------------------------------------------------------
  Architecture / model:
      sweep_main.csv (model, dtype columns), plus a few constants pulled
      from the paper that should be re-derived from the model on GPU:
        BGE_NUM_PARAMS, MPNET_VANILLA_TRAINABLE.
        See SOURCES_TODO at the bottom of the printed summary.

  Adapter count / batch size sweeps (Table 2 top + middle blocks, Findings
  1/2/5, p50/p90/p99/QPS/GPU mem):
      benchmarks/results/sweep_main.csv

  Rank sweep (Table 2 bottom block, Finding 3):
      benchmarks/results/sweep_main.csv, rows at N=1000, B=32, r in {4,8,16,32}.
      Folded into the main sweep (run.py --extra-configs) so r=8 is measured
      exactly once and shared with the top/middle blocks.
      (Formerly a separate run, sweep_ranks.csv.)

  Capacity sweep (Finding 4 ceiling claim):
      benchmarks/results/sweep_capacity.csv  +  sweep_main.csv

  PEFT baselines (Table 3, Finding 7, baseline ceiling text):
      benchmarks/results/peft_grouped_sxm80.csv
      benchmarks/results/peft_homogeneous_sxm80.csv
      benchmarks/results/peft_base_sxm80.csv

  LateFuse rows for Table 3 (and speedup multipliers):
      benchmarks/results/sweep_main.csv, filtered to r=8 (5-seed mean).
      Same run as Table 2, so the two tables agree on overlapping configs.
      (Formerly a separate single run, sysname_sxm80_ref.csv.)

  Forward-pass FLOP / profiler / ablation (Finding 6):
      benchmarks/results/forward_breakdown.json

  Per-task accuracy (Table 1, Appendix A):
      benchmarks/quality/setfit_mpnet_multiseed.csv
      Aggregated per (dataset, method): mean over the 10 seeds. Vanilla uses
      its single LR (2e-5); LoRA uses one shared body_lr across all datasets
      (the LR with the highest across-dataset mean, currently 5e-4). Scores
      come from the 'score' column and respect the per-dataset 'metric'
      (e.g. AmazonCF = MCC), so the q+v adapter (294,912 params) matches the
      reported param count.

  Per-task accuracy, serving model (SetfitBge* macros, table_accuracy_bge):
      benchmarks/quality/setfit_bge_multiseed.csv
      Same protocol and aggregation as the mpnet file, run on BAAI/bge-m3
      (the model all serving benchmarks use). Optional until
      benchmarks/quality/run_setfit.sh has been run on the GPU host;
      macros and table are emitted only when the CSV exists.
"""

from __future__ import annotations

import csv
import json
import math
import statistics
from pathlib import Path
from typing import Any, Iterable

# ===========================================================================
# Required architecture / environment metadata
# ===========================================================================
# The script reads model_metadata.csv and env_metadata.csv unconditionally
# (no fallbacks). Both are produced by benchmarks/profiling/model_metadata.py
# on the GPU host. If either is missing the script aborts with a clear error.

# Hub identifiers used to look up rows in model_metadata.csv. If you point
# the metadata script at different hub names, update these to match.
BGE_NAME       = "BAAI/bge-m3"
MPNET_NAME_HUB = "sentence-transformers/paraphrase-mpnet-base-v2"

FP16_BYTES = 2
# Decimal GB/TB to match the paper's arithmetic
# (80 GB A100 / 22 TB at 20k models).
A100_80GB_BYTES = 80 * 10**9
HEADLINE_DTYPE  = "float16"

# ===========================================================================
# Paths
# ===========================================================================
REPO    = Path(__file__).resolve().parent.parent
RESULTS = REPO / "benchmarks" / "results"
QUALITY = REPO / "benchmarks" / "quality"
OUT     = REPO / "paper"


# ===========================================================================
# CSV / formatting helpers
# ===========================================================================

def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open() as f:
        return list(csv.DictReader(f))


def find_row(rows: list[dict[str, str]], **filters: Any) -> dict[str, str]:
    out = [r for r in rows
           if all(str(r[k]) == str(v) for k, v in filters.items())]
    if len(out) != 1:
        raise LookupError(f"expected 1 row for {filters}, got {len(out)}")
    return out[0]


SWEEP_GROUP_KEYS = ("model", "dtype", "num_adapters", "batch_size",
                    "lora_rank", "seq_len")


def aggregate_seeds(rows: list[dict[str, str]],
                    group_keys: tuple[str, ...] = SWEEP_GROUP_KEYS,
                    ) -> list[dict[str, str]]:
    """Collapse multi-seed sweep CSVs to one row per configuration.

    run.py --seeds writes one row per (config, seed). Group by config, replace
    each numeric metric with its mean over seeds, and add '<col>_std' (sample
    s.d.; 0.0 when n=1) plus 'n_seeds'. Single-run CSVs pass through with
    values unchanged, so downstream find_row lookups work for both formats.
    """
    groups: dict[tuple, list[dict[str, str]]] = {}
    for r in rows:
        groups.setdefault(tuple(r[k] for k in group_keys), []).append(r)
    # warmup/iters are protocol constants, not measurements -- pass through
    skip = set(group_keys) | {"seed", "warmup", "iters"}
    out: list[dict[str, str]] = []
    for grp in groups.values():
        agg = {k: v for k, v in grp[0].items() if k != "seed"}
        agg["n_seeds"] = str(len(grp))
        for col in agg.copy():
            if col in skip or col == "n_seeds":
                continue
            try:
                vals = [float(g[col]) for g in grp]
            except (ValueError, KeyError):
                continue
            agg[col] = repr(statistics.fmean(vals))
            agg[col + "_std"] = repr(
                statistics.stdev(vals) if len(vals) > 1 else 0.0)
        out.append(agg)
    return out


def std_or_todo(row: dict[str, str], col: str, dp: int = 1) -> str:
    """s.d.-over-seeds macro value; 'TODO' until the multi-seed re-run lands."""
    if int(row.get("n_seeds", "1")) > 1:
        return fmt_f(row[col + "_std"], dp)
    return "TODO"


def _fmt_lr(lr: float) -> str:
    """0.00002 -> '2\\times10^{-5}' (LaTeX math, used inside the table caption)."""
    mant, exp = f"{lr:.0e}".split("e")
    return rf"{int(float(mant))}\times10^{{{int(exp)}}}"


def aggregate_setfit(rows: list[dict[str, str]]) -> tuple[list[dict[str, str]], str, str]:
    """Collapse the multiseed sweep to one score per (dataset, method).

    Mean over seeds. Vanilla uses its single LR; LoRA uses one shared body_lr
    across all datasets (the LR whose across-dataset mean is highest). Reads
    the 'score' column, so per-dataset metrics (AmazonCF = MCC) are respected.
    Returns (agg_rows, vanilla_lr_tex, lora_lr_tex); agg_rows carry an
    'accuracy' field so the downstream renderers stay unchanged.

    If the sweep includes 'frozen' rows (head-only fit on pre-trained
    embeddings, no body LR), they are aggregated too.
    """
    groups: dict[tuple, list[float]] = {}
    for r in rows:
        groups.setdefault(
            (r["dataset"], r["method"], float(r["body_lr"])), []
        ).append(float(r["score"]))
    means = {k: statistics.fmean(v) for k, v in groups.items()}
    datasets = sorted({k[0] for k in means})

    vanilla_lrs = sorted({k[2] for k in means if k[1] == "vanilla"})
    if len(vanilla_lrs) != 1:
        raise ValueError(f"expected one vanilla LR, got {vanilla_lrs}")
    vanilla_lr = vanilla_lrs[0]

    lora_lrs = sorted({k[2] for k in means if k[1] == "lora"})
    lora_lr = max(
        lora_lrs,
        key=lambda lr: statistics.fmean([means[(d, "lora", lr)] for d in datasets]),
    )

    frozen_lrs = sorted({k[2] for k in means if k[1] == "frozen"})

    agg: list[dict[str, str]] = []
    for d in datasets:
        if frozen_lrs:
            agg.append({"dataset": d, "method": "frozen", "lora_rank": "0",
                        "accuracy": f"{means[(d, 'frozen', frozen_lrs[0])]:.6f}"})
        agg.append({"dataset": d, "method": "vanilla", "lora_rank": "0",
                    "accuracy": f"{means[(d, 'vanilla', vanilla_lr)]:.6f}"})
        agg.append({"dataset": d, "method": "lora", "lora_rank": "8",
                    "accuracy": f"{means[(d, 'lora', lora_lr)]:.6f}"})
    return agg, _fmt_lr(vanilla_lr), _fmt_lr(lora_lr)


def load_model_arch() -> dict[str, dict[str, str]]:
    """Read model_metadata.csv into {model_name: {key: value}}.

    Required input -- the script aborts if it's not present. Generate it
    with benchmarks/profiling/model_metadata.py on the GPU host.
    """
    path = RESULTS / "model_metadata.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"missing {path}\n"
            "  Run benchmarks/profiling/model_metadata.py on a GPU host to\n"
            "  produce it, then copy it back and re-run this script."
        )
    data: dict[str, dict[str, str]] = {}
    with path.open() as f:
        for row in csv.DictReader(f):
            data.setdefault(row["model"], {})[row["key"]] = row["value"]
    return data


def load_env() -> dict[str, str]:
    """Read env_metadata.csv (single row) into a flat dict.

    Required input -- aborts if missing. Source: model_metadata.py.
    """
    path = RESULTS / "env_metadata.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"missing {path}\n"
            "  Run benchmarks/profiling/model_metadata.py on a GPU host."
        )
    with path.open() as f:
        return next(csv.DictReader(f))


def arch_int(arch: dict[str, str], model: str, key: str) -> int:
    """Pull an int field from the per-model arch dict; clear error if missing."""
    if key not in arch or not str(arch[key]).strip():
        raise KeyError(
            f"model_metadata.csv: model={model!r} is missing required key {key!r}"
        )
    return int(arch[key])


def num_comma(n: int) -> str:
    """Format an int with LaTeX-friendly thousands separator: 47000 -> '47{,}000'."""
    return f"{n:,}".replace(",", "{,}")


def fmt_f(v: float | str, dp: int) -> str:
    return f"{float(v):.{dp}f}"


def fmt_int(v: float | str) -> str:
    return str(round(float(v)))


# Macro-name suffix maps. LaTeX macros can't contain digits, so each numeric
# axis value gets a word-form name. Used in macro names AND nowhere else.
N_SUFFIX = {
    100:    "Hundred",
    1000:   "OneK",
    5000:   "FiveK",
    10000:  "TenK",
    20000:  "TwentyK",
    40000:  "FortyK",
    47000:  "FortySevenK",
}
B_SUFFIX = {
    8:   "Eight",
    16:  "Sixteen",
    32:  "ThirtyTwo",
    64:  "SixtyFour",
    128: "OneTwentyEight",
}
R_SUFFIX = {
    4:  "Four",
    8:  "Eight",
    16: "Sixteen",
    32: "ThirtyTwo",
}


# ===========================================================================
# Macro collector
# ===========================================================================

class Macros:
    """Collects \\newcommand definitions in declaration order, with comments."""

    def __init__(self) -> None:
        self._chunks: list[str] = []
        self._defined: set[str] = set()

    def section(self, title: str, source: str = "") -> None:
        line = f"\n% ===== {title} =====\n"
        if source:
            line += f"% source: {source}\n"
        self._chunks.append(line)

    def add(self, name: str, value: Any, comment: str = "") -> None:
        if name in self._defined:
            raise ValueError(f"duplicate macro: {name}")
        self._defined.add(name)
        suffix = f"  % {comment}" if comment else ""
        self._chunks.append(f"\\newcommand{{\\{name}}}{{{value}}}{suffix}\n")

    def render(self) -> str:
        header = (
            "% AUTO-GENERATED by paper/build_numbers.py -- DO NOT EDIT.\n"
            "% Regenerate with:  uv run python paper/build_numbers.py\n"
            "% Macros are grouped by topic; the comment above each block names\n"
            "% the CSV/JSON source so it's clear what to re-run.\n"
        )
        return header + "".join(self._chunks)


# Macro-name tag per SetFit dataset; shared by the mpnet and bge-m3 blocks.
SETFIT_TASKS = [
    ("SSTTwo",      "SetFit/sst2"),
    ("SSTFive",     "SetFit/sst5"),
    ("CR",          "SetFit/CR"),
    ("AmazonCF",    "SetFit/amazon_counterfactual_en"),
    ("Emotion",     "SetFit/emotion"),
    ("EnronSpam",   "SetFit/enron_spam"),
    ("AGNews",      "SetFit/ag_news"),
]


def emit_setfit_macros(m: Macros, setfit: list[dict[str, str]],
                       prefix: str) -> None:
    """Per-task + mean accuracy macros from one aggregated SetFit sweep.

    prefix="" emits the original mpnet names (Setfit<Task>Vanilla, ...);
    prefix="Bge" emits SetfitBge<Task>Vanilla etc. for the serving model.
    Frozen (no-adaptation baseline) macros are emitted when the sweep has
    'frozen' rows.
    """
    have_frozen = any(r["method"] == "frozen" for r in setfit)
    van_total = 0.0
    lora_total = 0.0
    frozen_total = 0.0
    drops_pp: list[float] = []
    for tag, key in SETFIT_TASKS:
        van = float(next(r for r in setfit
                         if r["dataset"] == key and r["method"] == "vanilla"
                         )["accuracy"])
        lora = float(next(r for r in setfit
                          if r["dataset"] == key and r["method"] == "lora"
                          )["accuracy"])
        # Delta from the rounded cells, so it equals the column subtraction a
        # reader does in the table (CR's true mean diff otherwise rounds to a
        # different last digit than 0.xxx-rounded(vanilla) - 0.xxx-rounded(lora)).
        delta = round(lora, 3) - round(van, 3)
        if have_frozen:
            frozen = float(next(r for r in setfit
                                if r["dataset"] == key and r["method"] == "frozen"
                                )["accuracy"])
            m.add(f"Setfit{prefix}{tag}Frozen", fmt_f(frozen, 3))
            frozen_total += frozen
        m.add(f"Setfit{prefix}{tag}Vanilla", fmt_f(van, 3))
        m.add(f"Setfit{prefix}{tag}Lora",    fmt_f(lora, 3))
        # Delta with explicit minus sign (paper writes it as -0.016)
        m.add(f"Setfit{prefix}{tag}Delta",   fmt_f(delta, 3))
        van_total += van
        lora_total += lora
        drops_pp.append(-delta * 100)

    van_mean = van_total / len(SETFIT_TASKS)
    lora_mean = lora_total / len(SETFIT_TASKS)
    if have_frozen:
        frozen_mean = frozen_total / len(SETFIT_TASKS)
        m.add(f"Setfit{prefix}MeanFrozen", fmt_f(frozen_mean, 3))
        m.add(f"Setfit{prefix}LoraGainOverFrozenPP",
              fmt_f((lora_mean - frozen_mean) * 100, 1),
              comment="LoRA mean minus frozen-encoder mean (pp)")
    m.add(f"Setfit{prefix}MeanVanilla",     fmt_f(van_mean, 3))
    m.add(f"Setfit{prefix}MeanLora",        fmt_f(lora_mean, 3))
    m.add(f"Setfit{prefix}MeanDelta",       fmt_f(lora_mean - van_mean, 3))
    m.add(f"Setfit{prefix}RetainsPct",
          fmt_f(100 * lora_mean / van_mean, 1))
    m.add(f"Setfit{prefix}MeanDropPP",
          fmt_f((van_mean - lora_mean) * 100, 2))
    m.add(f"Setfit{prefix}MinDropPP", fmt_f(min(drops_pp), 1),
          comment="smallest per-task accuracy drop (pp)")
    m.add(f"Setfit{prefix}MaxDropPP", fmt_f(max(drops_pp), 1),
          comment="largest per-task accuracy drop (pp)")


# ===========================================================================
# Build
# ===========================================================================

def build() -> None:
    # ---- Load architecture + env metadata (required) -----------------------
    arch = load_model_arch()
    env  = load_env()
    for name in (BGE_NAME, MPNET_NAME_HUB):
        if name not in arch:
            raise KeyError(
                f"model_metadata.csv has no rows for {name!r}. "
                "Re-run benchmarks/profiling/model_metadata.py with "
                "--model 'name:tm1,tm2' covering this model."
            )
    BGE_HIDDEN_SIZE    = arch_int(arch[BGE_NAME],       BGE_NAME,       "hidden_size")
    BGE_NUM_LAYERS     = arch_int(arch[BGE_NAME],       BGE_NAME,       "num_layers")
    BGE_NUM_HEADS      = arch_int(arch[BGE_NAME],       BGE_NAME,       "num_heads")
    BGE_NUM_PARAMS     = arch_int(arch[BGE_NAME],       BGE_NAME,       "total_params")
    BGE_TARGET_MODULES = arch_int(arch[BGE_NAME],       BGE_NAME,       "num_target_modules")
    MPNET_HIDDEN_SIZE       = arch_int(arch[MPNET_NAME_HUB], MPNET_NAME_HUB, "hidden_size")
    MPNET_NUM_LAYERS        = arch_int(arch[MPNET_NAME_HUB], MPNET_NAME_HUB, "num_layers")
    MPNET_TARGET_MODULES    = arch_int(arch[MPNET_NAME_HUB], MPNET_NAME_HUB, "num_target_modules")
    MPNET_VANILLA_TRAINABLE = arch_int(arch[MPNET_NAME_HUB], MPNET_NAME_HUB, "total_params")

    # ---- Load benchmark CSVs / JSON ----------------------------------------
    sweep_main     = aggregate_seeds(load_csv(RESULTS / "sweep_main.csv"))
    sweep_capacity = aggregate_seeds(load_csv(RESULTS / "sweep_capacity.csv"))
    peft_grouped   = load_csv(RESULTS / "peft_grouped_sxm80.csv")
    peft_homog     = load_csv(RESULTS / "peft_homogeneous_sxm80.csv")
    peft_base      = load_csv(RESULTS / "peft_base_sxm80.csv")
    # PEFT native mixed-batch (adapter_names) baseline: optional until the
    # first run_sxm80.sh re-run produces it (run.py mode added 2026-06-10).
    peft_mixed_path = RESULTS / "peft_mixed_sxm80.csv"
    peft_mixed     = (load_csv(peft_mixed_path)
                      if peft_mixed_path.exists() else None)
    # LateFuse rows for Table 3 / the speedup multipliers come from the same
    # 5-seed sweep_main run as Table 2, filtered to the headline rank (r=8), so
    # the two tables report identical numbers for overlapping configs. (Was a
    # separate single run, sysname_sxm80_ref.csv, which drifted ~5% in QPS.)
    sysname_ref    = [r for r in sweep_main if r["lora_rank"] == "8"]
    fwd            = json.loads((RESULTS / "forward_breakdown.json").read_text())
    setfit_raw     = load_csv(QUALITY / "setfit_mpnet_multiseed.csv")
    setfit, setfit_vanilla_lr, setfit_lora_lr = aggregate_setfit(setfit_raw)
    # BGE-m3 accuracy run (same protocol, serving model): optional until
    # benchmarks/quality/run_setfit.sh has produced the CSV.
    setfit_bge_path = QUALITY / "setfit_bge_multiseed.csv"
    setfit_bge = setfit_bge_vanilla_lr = setfit_bge_lora_lr = None
    if setfit_bge_path.exists():
        setfit_bge, setfit_bge_vanilla_lr, setfit_bge_lora_lr = (
            aggregate_setfit(load_csv(setfit_bge_path)))

    m = Macros()

    # =======================================================================
    # 1. Architecture constants (sourced from model_metadata.csv)
    # =======================================================================
    m.section("Architecture (BGE-m3, mpnet)",
              "benchmarks/results/model_metadata.csv (params, dims, heads)")
    m.add("BgeHidden", BGE_HIDDEN_SIZE)
    m.add("BgeLayers", BGE_NUM_LAYERS)
    m.add("BgeHeads",  BGE_NUM_HEADS)
    m.add("BgeParamsM", round(BGE_NUM_PARAMS / 1e6),
          comment=f"= round({BGE_NUM_PARAMS} / 1e6)")
    bge_fp16_gb = round(BGE_NUM_PARAMS * FP16_BYTES / 1e9, 2)  # 1.14: displayed per-copy size
    m.add("BgeFpSixteenGB",
          fmt_f(bge_fp16_gb, 2),
          comment="per-model fp16 size, decimal GB")

    m.add("MpnetHidden", MPNET_HIDDEN_SIZE)
    m.add("MpnetLayers", MPNET_NUM_LAYERS)
    m.add("MpnetVanillaParamsM", round(MPNET_VANILLA_TRAINABLE / 1e6),
          comment=f"= round({MPNET_VANILLA_TRAINABLE} / 1e6)")

    # =======================================================================
    # 2. LoRA per-adapter params / storage (BGE-m3 at each rank)
    # =======================================================================
    m.section(
        "LoRA per-adapter params and storage (BGE-m3)",
        "analytic: 2 * M * L * d * r  (A is r×d, B is d×r, both trainable)",
    )

    def lora_params(rank: int, d=BGE_HIDDEN_SIZE, L=BGE_NUM_LAYERS,
                    M=BGE_TARGET_MODULES) -> int:
        return 2 * M * L * d * rank

    def lora_bytes(rank: int) -> int:
        return lora_params(rank) * FP16_BYTES

    for r, tag in R_SUFFIX.items():
        m.add(f"LoraParamsR{tag}",     num_comma(lora_params(r)))
        m.add(f"LoraBytesR{tag}MB",    fmt_f(lora_bytes(r) / 1e6, 3))

    # Full Q+V fine-tune param count and the LoRA reduction multiplier
    full_qv = (BGE_TARGET_MODULES * BGE_NUM_LAYERS
               * BGE_HIDDEN_SIZE * BGE_HIDDEN_SIZE)
    m.add("FullQVParams", num_comma(full_qv))
    m.add("LoraReductionVsQVRankEight",
          round(full_qv / lora_params(8)),
          comment="= (M*L*d*d) / (2*M*L*d*r) at r=8")
    m.add("StorageReductionFullVsLoraRankEight",
          round(BGE_NUM_PARAMS * FP16_BYTES / lora_bytes(8)),
          comment=f"= (BGE_NUM_PARAMS * 2) / bytes(r=8); "
                  f"reflects BGE_NUM_PARAMS = {BGE_NUM_PARAMS}")

    # =======================================================================
    # 3. Aggregate storage / capacity numbers (intro + complexity §)
    # =======================================================================
    m.section("Aggregate storage / per-GPU capacity",
              "derived from arch constants and sweep_main.csv")

    # Derived from the *displayed* per-copy size (1.14 GB) so the intro arithmetic
    # reproduces exactly: 1.14 GB x 20,000 = 22.8 TB; / 80 GB = 285 A100s. Using
    # full-precision bytes here gave 22.7 TB / 284 GPUs, which 1.14 x 20,000 cannot
    # reproduce -- the 2-dp per-copy figure rounds up from 1.1355.
    storage_20k_bytes = bge_fp16_gb * 1e9 * 20_000
    m.add("StorageTwentyKTB",     fmt_f(storage_20k_bytes / 1e12, 1))
    m.add("StorageTwentyKGPUs",   math.ceil(storage_20k_bytes / A100_80GB_BYTES))

    m.add("CacheNHundredMB",
          fmt_int(100 * lora_bytes(8) / 1e6),
          comment="100 adapters at r=8 fp16")
    m.add("CacheNTwentyKGB",
          fmt_f(20_000 * lora_bytes(8) / 1e9, 1),
          comment="20k adapters at r=8 fp16")

    # Empirical 47k capacity (largest N in sweep_main); ranks 16/32 are
    # back-calculated by the linear-in-r scaling shown in Finding 4.
    cap_r8 = max(int(r["num_adapters"]) for r in sweep_main)
    m.add("CapacityRankEight",      num_comma(cap_r8))
    m.add("CapacityRankSixteen",    num_comma(cap_r8 // 2))
    m.add("CapacityRankThirtyTwo",  num_comma(cap_r8 // 4))

    # =======================================================================
    # 4. Main N-sweep (B=32, r=8) — Table 2 top block, Findings 1/4
    # =======================================================================
    m.section("Adapter-count sweep (B=32, r=8)",
              "benchmarks/results/sweep_main.csv")
    n_main_rows: dict[int, dict[str, str]] = {}
    for n, tag in N_SUFFIX.items():
        row = find_row(sweep_main,
                       num_adapters=str(n), batch_size="32", lora_rank="8")
        n_main_rows[n] = row
        m.add(f"PFiftyAtN{tag}",        fmt_f(row["p50_ms"], 1))
        m.add(f"PFiftyAtN{tag}Std",     std_or_todo(row, "p50_ms"),
              comment="sample s.d. over seeds")
        m.add(f"PNinetyAtN{tag}",       fmt_f(row["p90_ms"], 1))
        m.add(f"PNinetyNineAtN{tag}",   fmt_f(row["p99_ms"], 1))
        m.add(f"QpsAtN{tag}",           fmt_int(row["throughput_samples_sec"]))
        m.add(f"GpuGBAtN{tag}",         fmt_f(row["peak_gpu_mem_gb"], 2))
        m.add(f"AssemblePctAtN{tag}",   fmt_f(row["assemble_share_pct"], 1))
        m.add(f"FwdMsAtN{tag}",         fmt_f(row["forward_p50_ms"], 1))

    # Derived: drift and ceiling claims from Finding 1
    p50_lo = float(n_main_rows[100]["p50_ms"])
    p50_hi = float(n_main_rows[47000]["p50_ms"])
    m.add("PFiftyDriftPctNHundredToFortySevenK",
          fmt_f(abs(p50_hi - p50_lo) / p50_lo * 100, 1),
          comment="|p50_47k - p50_100| / p50_100 * 100 (unsigned; prose says 'varies by <= X%')")
    m.add("SweepNumSeeds", n_main_rows[100].get("n_seeds", "1"),
          comment="seeded repeats per config in sweep_main")
    m.add("CustomerMultiplierHundredToFortySevenK",
          round(47000 / 100), comment="= 47000 / 100")

    p99_ceiling_n_sweep = max(
        float(r["p99_ms"]) for r in sweep_main
        if r["batch_size"] == "32" and r["lora_rank"] == "8"
    )
    m.add("PNinetyNineCeilingNSweep", fmt_f(p99_ceiling_n_sweep, 1),
          comment="max p99 over N at B=32 r=8")
    fwd_mean_b32_r8 = statistics.fmean(
        float(r["forward_p50_ms"]) for r in sweep_main
        if r["batch_size"] == "32" and r["lora_rank"] == "8")
    m.add("ForwardMsConstantBThirtyTwo", fmt_f(fwd_mean_b32_r8, 1),
          comment="mean forward_p50 across N at B=32 r=8")
    asm_max_b32_r8 = max(float(r["assemble_share_pct"]) for r in sweep_main
                         if r["batch_size"] == "32" and r["lora_rank"] == "8")
    m.add("AssemblePctCeilingBThirtyTwo", fmt_f(asm_max_b32_r8, 1),
          comment="max assemble_share_pct across N at B=32 r=8")

    # Base + activation buffer overhead at the per-GPU ceiling (Finding 4).
    # Derived from N=47000 row: peak GPU mem minus adapter-cache footprint.
    base_act_gb = (float(n_main_rows[47000]["peak_gpu_mem_gb"])
                   - float(n_main_rows[47000]["adapter_cache_gb"]))
    m.add("BaseActivationOverheadGB", fmt_f(base_act_gb, 2),
          comment="peak_gpu - adapter_cache @ N=47000 B=32 r=8")

    # =======================================================================
    # 5. Batch-size sweep (N=1000, r=8) — Table 2 middle, Findings 2/5
    # =======================================================================
    m.section("Batch-size sweep (N=1000, r=8)",
              "benchmarks/results/sweep_main.csv")
    b_rows: dict[int, dict[str, str]] = {}
    for b, tag in B_SUFFIX.items():
        row = find_row(sweep_main,
                       num_adapters="1000", batch_size=str(b), lora_rank="8")
        b_rows[b] = row
        m.add(f"PFiftyAtB{tag}",        fmt_f(row["p50_ms"], 1))
        m.add(f"PNinetyAtB{tag}",       fmt_f(row["p90_ms"], 1))
        m.add(f"PNinetyNineAtB{tag}",   fmt_f(row["p99_ms"], 1))
        m.add(f"QpsAtB{tag}",           fmt_int(row["throughput_samples_sec"]))
        m.add(f"GpuGBAtB{tag}",         fmt_f(row["peak_gpu_mem_gb"], 2))
        m.add(f"AssemblePctAtB{tag}",   fmt_f(row["assemble_share_pct"], 1))

    # p99/p50 ratios (Finding 5)
    for b in (32, 64):
        ratio = float(b_rows[b]["p99_ms"]) / float(b_rows[b]["p50_ms"])
        m.add(f"PNinetyNineOverPFiftyAtB{B_SUFFIX[b]}", fmt_f(ratio, 2))

    # Throughput at B=64 (the batch size at which the encoder forward
    # becomes GPU-bound and throughput stops climbing)
    m.add("QpsAtBSixtyFourSatVal",
          fmt_int(b_rows[64]["throughput_samples_sec"]),
          comment="QPS at the saturation knee, B=64 N=1000 r=8")

    # =======================================================================
    # 6. Rank sweep (N=1000, B=32) — Table 2 bottom, Finding 3
    # =======================================================================
    m.section("Rank sweep (N=1000, B=32)",
              "benchmarks/results/sweep_main.csv (--extra-configs rank cells)")
    r_rows: dict[int, dict[str, str]] = {}
    for r, tag in R_SUFFIX.items():
        row = find_row(sweep_main,
                       num_adapters="1000", batch_size="32", lora_rank=str(r))
        r_rows[r] = row
        m.add(f"PFiftyAtR{tag}",        fmt_f(row["p50_ms"], 1))
        m.add(f"PFiftyAtR{tag}Std",     std_or_todo(row, "p50_ms"),
              comment="sample s.d. over seeds")
        m.add(f"PNinetyAtR{tag}",       fmt_f(row["p90_ms"], 1))
        m.add(f"PNinetyNineAtR{tag}",   fmt_f(row["p99_ms"], 1))
        m.add(f"PNinetyNineAtR{tag}Std", std_or_todo(row, "p99_ms"),
              comment="sample s.d. over seeds")
        m.add(f"QpsAtR{tag}",           fmt_int(row["throughput_samples_sec"]))
        m.add(f"GpuGBAtR{tag}",         fmt_f(row["peak_gpu_mem_gb"], 2))
        m.add(f"FwdMsAtR{tag}",         fmt_f(row["forward_p50_ms"], 1))

    # Non-monotonic spread of p99 across ranks (Finding 3 tail-noise caveat)
    p99_by_rank = [float(r_rows[r]["p99_ms"]) for r in R_SUFFIX]
    m.add("PNinetyNineRankSpreadPct",
          fmt_int((max(p99_by_rank) - min(p99_by_rank)) / min(p99_by_rank) * 100),
          comment="(max - min) / min p99 over ranks, N=1000 B=32")

    # =======================================================================
    # 7. Forward pass breakdown (Finding 6)
    # =======================================================================
    m.section("Forward-pass breakdown",
              "benchmarks/results/forward_breakdown.json")
    m.add("FlopBaseGFlopsPerLayer",
          fmt_f(fwd["flop_ratio"]["base_GFLOPs_per_layer"], 2))
    m.add("FlopLoraGFlopsPerLayer",
          fmt_f(fwd["flop_ratio"]["lora_GFLOPs_per_layer"], 3))
    m.add("FlopBaseToLoraRatio",
          fmt_int(fwd["flop_ratio"]["base_to_lora_ratio"]))
    m.add("FlopMaxLoraSavingsPct",
          fmt_f(100 / fwd["flop_ratio"]["base_to_lora_ratio"], 2),
          comment="ceiling savings if LoRA kernel were free")

    m.add("ProfBaseLinearPct",
          fmt_f(fwd["profiler_categories"]["base_linear"]["share_pct"], 1))
    m.add("ProfLoraBmmIsolatedPct",
          fmt_f(fwd["bmm_isolation"]["lora_bmm_self_share_pct"], 2),
          comment="LoRA-only bmm share (attn/LR head subtracted)")
    m.add("ProfTotalCudaMs",
          fmt_f(fwd["profiler_total_ms"], 1))

    m.add("AblationLoraOnMs",
          fmt_f(fwd["ablation"]["lora_on_mean_ms"], 1))
    m.add("AblationLoraCostMs",
          fmt_f(fwd["ablation"]["lora_cost_mean_ms"], 1))
    m.add("AblationLoraCostPct",
          fmt_f(fwd["ablation"]["lora_cost_share_pct"], 1))

    # =======================================================================
    # 8. PEFT baselines (Table 3, baseline text, Finding 7)
    # =======================================================================
    m.section("PEFT baseline (no-adapter ceiling)",
              "benchmarks/results/peft_base_sxm80.csv")
    for b in (8, 32, 128):
        row = find_row(peft_base, batch_size=str(b))
        m.add(f"PeftBaseQpsAtB{B_SUFFIX[b]}",
              fmt_int(row["throughput_samples_sec"]))
        m.add(f"PeftBasePFiftyAtB{B_SUFFIX[b]}",
              fmt_f(row["p50_ms"], 1))

    m.section("PEFT grouped baseline",
              "benchmarks/results/peft_grouped_sxm80.csv")
    PEFT_NB = [(100, 8), (100, 32), (100, 128),
               (1000, 8), (1000, 32), (1000, 128)]
    for n, b in PEFT_NB:
        row = find_row(peft_grouped, num_adapters=str(n), batch_size=str(b))
        suf = f"N{N_SUFFIX[n]}B{B_SUFFIX[b]}"
        m.add(f"PeftGroupedPFifty{suf}",  fmt_int(row["p50_ms"]))
        m.add(f"PeftGroupedQps{suf}",     fmt_f(row["throughput_samples_sec"], 1))
        # Cold-start: add_adapter_total_s (per-N constant)
    # add_adapter_total_s is the same for each row of a given N; emit once per N
    for n in (100, 1000):
        row = find_row(peft_grouped, num_adapters=str(n), batch_size="8")
        m.add(f"PeftGroupedAddAdapterSecN{N_SUFFIX[n]}",
              fmt_f(row["add_adapter_total_s"], 1))

    if peft_mixed is not None:
        m.section("PEFT mixed-batch baseline (adapter_names API)",
                  "benchmarks/results/peft_mixed_sxm80.csv")
        for n, b in PEFT_NB:
            row = find_row(peft_mixed, num_adapters=str(n), batch_size=str(b))
            suf = f"N{N_SUFFIX[n]}B{B_SUFFIX[b]}"
            m.add(f"PeftMixedPFifty{suf}",  fmt_f(row["p50_ms"], 1))
            m.add(f"PeftMixedQps{suf}",     fmt_f(row["throughput_samples_sec"], 1))

    m.section("PEFT homogeneous baseline",
              "benchmarks/results/peft_homogeneous_sxm80.csv")
    for n, b in PEFT_NB:
        row = find_row(peft_homog, num_adapters=str(n), batch_size=str(b))
        suf = f"N{N_SUFFIX[n]}B{B_SUFFIX[b]}"
        m.add(f"PeftHomogPFifty{suf}",  fmt_int(row["p50_ms"]))
        m.add(f"PeftHomogQps{suf}",     fmt_f(row["throughput_samples_sec"], 1))
    for n in (100, 1000):
        row = find_row(peft_homog, num_adapters=str(n), batch_size="8")
        m.add(f"PeftHomogAddAdapterSecN{N_SUFFIX[n]}",
              fmt_f(row["add_adapter_total_s"], 1))

    m.section("LateFuse reference for Table 3",
              "benchmarks/results/sweep_main.csv (5-seed mean, r=8)")
    for n, b in PEFT_NB:
        row = find_row(sysname_ref, num_adapters=str(n), batch_size=str(b))
        suf = f"N{N_SUFFIX[n]}B{B_SUFFIX[b]}"
        m.add(f"SysRefPFifty{suf}",  fmt_f(row["p50_ms"], 1))
        m.add(f"SysRefQps{suf}",     fmt_int(row["throughput_samples_sec"]))

    # =======================================================================
    # 9. Speedup multipliers (paper uses QPS ratios)
    # =======================================================================
    m.section("Speedup multipliers (LateFuse QPS / PEFT QPS)",
              "derived from sweep_main (5-seed, r=8) vs peft_grouped/homog")

    def qps(rows, n: int, b: int) -> float:
        return float(find_row(rows, num_adapters=str(n),
                              batch_size=str(b))["throughput_samples_sec"])

    speedups_grouped: list[int] = []
    speedups_homog: list[float] = []
    for n, b in PEFT_NB:
        suf = f"N{N_SUFFIX[n]}B{B_SUFFIX[b]}"
        sp_g = qps(sysname_ref, n, b) / qps(peft_grouped, n, b)
        sp_h = qps(sysname_ref, n, b) / qps(peft_homog, n, b)
        m.add(f"SpeedupGrouped{suf}", round(sp_g))
        m.add(f"SpeedupHomog{suf}",   fmt_f(sp_h, 1))
        speedups_grouped.append(round(sp_g))
        speedups_homog.append(sp_h)

    m.add("SpeedupGroupedMin", min(speedups_grouped),
          comment="min over ALL measured (N,B) grouped speedups")
    m.add("SpeedupGroupedMax", max(speedups_grouped),
          comment="max over ALL measured (N,B) grouped speedups")

    if peft_mixed is not None:
        speedups_mixed = []
        for n, b in PEFT_NB:
            suf = f"N{N_SUFFIX[n]}B{B_SUFFIX[b]}"
            sp = qps(sysname_ref, n, b) / qps(peft_mixed, n, b)
            speedups_mixed.append(sp)
            m.add(f"SpeedupMixed{suf}", fmt_f(sp, 1))
        m.add("SpeedupMixedMin", fmt_f(min(speedups_mixed), 1),
              comment="min over ALL measured (N,B) mixed speedups")
        m.add("SpeedupMixedMax", fmt_f(max(speedups_mixed), 1),
              comment="max over ALL measured (N,B) mixed speedups")

    # Cold-start: PEFT add_adapter loop seconds vs LateFuse AdapterStore preload.
    # The LateFuse side (column 'adapter_load_total_s' in sweep_main.csv) only
    # exists if benchmarks have been re-run after the run.py instrumentation
    # change. Emit TODO if missing so the macro is well-defined either way.
    sys_load_row = next(
        (r for r in sweep_main
         if r["num_adapters"] == "1000" and r["batch_size"] == "32"
         and r["lora_rank"] == "8"
         and r.get("adapter_load_total_s", "").strip() not in ("", "0", "0.0")),
        None,
    )
    if sys_load_row is not None:
        sys_load_s = float(sys_load_row["adapter_load_total_s"])
        peft_add_s = float(find_row(peft_grouped, num_adapters="1000",
                                    batch_size="8")["add_adapter_total_s"])
        m.add("LateFuseColdLoadSecNOneK",
              fmt_f(sys_load_s, 2),
              comment="AdapterStore preload time, N=1000 from sweep_main.csv")
        sys_load_row_100 = next(
            (r for r in sweep_main
             if r["num_adapters"] == "100" and r["batch_size"] == "32"
             and r["lora_rank"] == "8"
             and r.get("adapter_load_total_s", "").strip() not in ("", "0", "0.0")),
            None,
        )
        if sys_load_row_100 is not None:
            m.add("LateFuseColdLoadSecNHundred",
                  fmt_f(float(sys_load_row_100["adapter_load_total_s"]), 2),
                  comment="AdapterStore preload time, N=100 from sweep_main.csv")
        m.add("ColdStartSpeedupNOneK",
              round(peft_add_s / sys_load_s),
              comment=f"PEFT add_adapter ({peft_add_s:.1f}s) / "
                      f"LateFuse preload ({sys_load_s:.2f}s)")

        # Min cold-start speedup across every (N) where both LateFuse and PEFT
        # have a paired measurement at B=32 r=8. add_adapter_total_s is
        # B-independent, so picking B=32 just selects the canonical row.
        cold_ratios: list[tuple[int, float]] = []
        for n in sorted({int(r["num_adapters"]) for r in peft_grouped}):
            try:
                lf_row = find_row(sweep_main, num_adapters=str(n),
                                  batch_size="32", lora_rank="8")
                pf_row = find_row(peft_grouped, num_adapters=str(n),
                                  batch_size="32", lora_rank="8")
            except LookupError:
                continue
            lf_s = float(lf_row.get("adapter_load_total_s") or 0)
            pf_s = float(pf_row.get("add_adapter_total_s") or 0)
            if lf_s > 0 and pf_s > 0:
                cold_ratios.append((n, pf_s / lf_s))
        if cold_ratios:
            min_n, min_ratio = min(cold_ratios, key=lambda x: x[1])
            m.add("ColdStartSpeedupMin", round(min_ratio),
                  comment=f"min across {[n for n,_ in cold_ratios]}; min at N={min_n}")
    else:
        m.add("LateFuseColdLoadSecNOneK", "TODO",
              comment="re-run benchmarks/run_sxm80.sh after run.py instrumentation")
        m.add("ColdStartSpeedupNOneK", "TODO",
              comment="needs adapter_load_total_s column in sweep_main.csv")
        m.add("ColdStartSpeedupMin", "TODO",
              comment="needs adapter_load_total_s column in sweep_main.csv")

    # Finding 7 ratios: PEFT homog QPS collapse N=100 -> N=1000
    homog_collapse = [
        qps(peft_homog, 100, b) / qps(peft_homog, 1000, b)
        for b in (8, 32, 128)
    ]
    m.add("PeftHomogCollapseMin", fmt_f(min(homog_collapse), 1))
    m.add("PeftHomogCollapseMax", fmt_f(max(homog_collapse), 1))

    # =======================================================================
    # 10. SetFit accuracy (Table 1 / Appendix A)
    # =======================================================================
    m.section("SetFit accuracy (mpnet, n=8 per class, 10-seed mean)",
              "benchmarks/quality/setfit_mpnet_multiseed.csv "
              "(vanilla @2e-5; LoRA @ shared best LR; scores respect per-dataset metric)")
    m.add("SetfitVanillaLR", setfit_vanilla_lr,
          comment="vanilla body LR (single LR in the sweep)")
    m.add("SetfitLoraLR", setfit_lora_lr,
          comment="shared LoRA body LR = argmax across-dataset mean")
    emit_setfit_macros(m, setfit, prefix="")

    # MPNet-specific param counts (used in Table 1 footer + Appendix A)
    mpnet_lora_params = (2 * MPNET_TARGET_MODULES * MPNET_NUM_LAYERS
                         * MPNET_HIDDEN_SIZE * 8)
    m.add("MpnetLoraParams",   num_comma(mpnet_lora_params))
    m.add("MpnetParamReductionRankEight",
          round(MPNET_VANILLA_TRAINABLE / mpnet_lora_params),
          comment="= MPNET_VANILLA_TRAINABLE / mpnet_lora_params")

    # =======================================================================
    # 10b. SetFit accuracy on the serving model (BGE-m3) — optional until
    #      benchmarks/quality/run_setfit.sh has been run on the GPU host
    # =======================================================================
    if setfit_bge is not None:
        m.section("SetFit accuracy (bge-m3, n=8 per class, 10-seed mean)",
                  "benchmarks/quality/setfit_bge_multiseed.csv "
                  "(same protocol as the mpnet sweep, on the serving model)")
        m.add("SetfitBgeVanillaLR", setfit_bge_vanilla_lr,
              comment="vanilla body LR (single LR in the sweep)")
        m.add("SetfitBgeLoraLR", setfit_bge_lora_lr,
              comment="shared LoRA body LR = argmax across-dataset mean")
        emit_setfit_macros(m, setfit_bge, prefix="Bge")
        m.add("BgeParamReductionRankEight",
              round(BGE_NUM_PARAMS / lora_params(8)),
              comment="= BGE_NUM_PARAMS / lora_params(r=8); "
                      "vanilla SetFit trains the full encoder")

    # =======================================================================
    # 11. Headline / abstract-friendly aliases
    # =======================================================================
    m.section("Benchmark protocol (warmup/iters from CSV rows)", "")
    sys_proto_row = n_main_rows[47000]
    peft_proto_row = find_row(peft_grouped, num_adapters="1000",
                              batch_size="32")
    # If the run pre-dates the warmup/iters CSV columns, fall back to "?" so
    # the macro is defined and the prose still compiles after a stale run.
    def _proto(row, key, fallback="?"):
        v = row.get(key, "").strip()
        return v if v else fallback
    m.add("SysWarmup",  _proto(sys_proto_row,  "warmup"),
          comment="from sweep_main.csv N=47000 row")
    m.add("SysIters",   _proto(sys_proto_row,  "iters"))
    m.add("PeftWarmup", _proto(peft_proto_row, "warmup"),
          comment="from peft_grouped_sxm80.csv N=1000 B=32 row")
    m.add("PeftIters",  _proto(peft_proto_row, "iters"))

    m.section("Headline aliases (abstract/intro/conclusion)", "")
    m.add("HeadlineCapacity",     num_comma(cap_r8))
    m.add("HeadlineQps",          fmt_int(n_main_rows[47000]["throughput_samples_sec"]))
    m.add("HeadlinePNinetyNine",  fmt_int(n_main_rows[47000]["p99_ms"]))
    m.add("HeadlineBatch",        32)
    m.add("HeadlineRank",         8)
    m.add("HeadlineDtype",        HEADLINE_DTYPE)
    m.add("HeadlineGPU",          env.get("gpu_name", ""))
    m.add("EnvTorchVersion",      env.get("torch_version", ""))
    m.add("EnvCudaVersion",       env.get("cuda_version", ""))
    m.add("EnvTransformersVersion", env.get("transformers_version", ""))
    m.add("EnvPeftVersion",       env.get("peft_version", ""))
    m.add("HeadlineSpeedupLow",   min(speedups_grouped))
    m.add("HeadlineSpeedupHigh",  max(speedups_grouped))

    # =======================================================================
    # Write outputs
    # =======================================================================
    (OUT / "numbers.tex").write_text(m.render())
    (OUT / "table_main.tex").write_text(
        render_table_main(sweep_main))
    (OUT / "table_baselines.tex").write_text(
        render_table_baselines(peft_grouped, peft_mixed, peft_homog, peft_base,
                               sysname_ref))
    (OUT / "table_accuracy.tex").write_text(render_table_accuracy(
        setfit, MPNET_HIDDEN_SIZE, MPNET_NUM_LAYERS,
        MPNET_TARGET_MODULES, MPNET_VANILLA_TRAINABLE))
    if setfit_bge is not None:
        (OUT / "table_accuracy_bge.tex").write_text(render_table_accuracy(
            setfit_bge, BGE_HIDDEN_SIZE, BGE_NUM_LAYERS,
            BGE_TARGET_MODULES, BGE_NUM_PARAMS))

    # Stdout summary
    storage_reduction_x = round(
        BGE_NUM_PARAMS * FP16_BYTES / lora_bytes(8)
    )
    print_summary(m._defined, sweep_main, peft_grouped, sysname_ref,
                  fwd, setfit, env, storage_reduction_x,
                  have_setfit_bge=setfit_bge is not None)


# ===========================================================================
# Table renderers
# ===========================================================================

GEN_HEADER = (
    "% AUTO-GENERATED by paper/build_numbers.py -- DO NOT EDIT.\n"
    "% Regenerate with:  uv run python paper/build_numbers.py\n"
)


def render_table_main(sweep_main) -> str:
    """tab:main_results body: N sweep + B sweep + r sweep, all at the same op-point.

    All three blocks read from sweep_main, including the rank cells (folded in
    via run.py --extra-configs), so the shared N=1000/B=32/r=8 operating point is
    measured exactly once and the rank block cannot read slower than that anchor
    from cross-run thermal drift.

    p50/p99 cells are mean +/- s.d. over the seeded repeats (p90 dropped:
    never discussed in prose, and the column width pays for the +/- cells).
    """
    lines: list[str] = []
    lines.append(r"\begin{tabular}{rrr rr r r}")
    lines.append(r"\toprule")
    lines.append(r"$N$ & $\mathcal{B}$ & $r$ & p50 (ms) & p99 (ms) & QPS & GPU \\")
    lines.append(r"    &     &     &     &     & (s/s) & (GB) \\")
    lines.append(r"\midrule")

    def pm(row, col, dp=1):
        return f"{fmt_f(row[col], dp)}$\\pm${std_or_todo(row, col, dp)}"

    def row_line(row, n, b, r):
        return (
            f"{num_comma(n)} & {b} & {r} & "
            f"{pm(row, 'p50_ms')} & "
            f"{pm(row, 'p99_ms')} & "
            f"{fmt_int(row['throughput_samples_sec'])} & "
            f"{fmt_f(row['peak_gpu_mem_gb'], 2)} \\\\"
        )

    # adapter-count sweep (B=32, r=8) -- top block
    for n in (100, 1000, 5000, 10000, 20000, 40000, 47000):
        row = find_row(sweep_main, num_adapters=str(n),
                       batch_size="32", lora_rank="8")
        lines.append(row_line(row, n, 32, 8))
    lines.append(r"\midrule")

    # batch-size sweep (N=1000, r=8); skip B=32 (in top block already)
    for b in (8, 16, 64, 128):
        row = find_row(sweep_main, num_adapters="1000",
                       batch_size=str(b), lora_rank="8")
        lines.append(row_line(row, 1000, b, 8))
    lines.append(r"\midrule")

    # rank sweep (N=1000, B=32); skip r=8 (in top block already).
    # These rows come from the same sweep_main run as r=8 (run.py
    # --extra-configs), measured back-to-back at the operating point.
    for r in (4, 16, 32):
        row = find_row(sweep_main, num_adapters="1000",
                       batch_size="32", lora_rank=str(r))
        lines.append(row_line(row, 1000, 32, r))

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    return GEN_HEADER + "\n".join(lines) + "\n"


def render_table_baselines(peft_grouped, peft_mixed, peft_homog, peft_base,
                           sysname_ref) -> str:
    """tab:baselines body: PEFT base/grouped/mixed/homog, LateFuse, speedups."""
    lines: list[str] = []
    lines.append(r"\begin{tabular}{lrrrr}")
    lines.append(r"\toprule")
    lines.append(r" & \multicolumn{2}{c}{$N = 100$} & \multicolumn{2}{c}{$N = 1{,}000$} \\")
    lines.append(r"\cmidrule(lr){2-3} \cmidrule(lr){4-5}")
    lines.append(r"System & p50 (ms) & QPS & p50 (ms) & QPS \\")
    lines.append(r"\midrule")

    BATCHES = [8, 32, 128]

    def emit_block(label_fn, csv_rows, p50_dp=0):
        for b in BATCHES:
            n100 = find_row(csv_rows, num_adapters="100", batch_size=str(b))
            n1k  = find_row(csv_rows, num_adapters="1000", batch_size=str(b))
            label = label_fn(b)
            lines.append(
                f"{label} & "
                f"{fmt_int(n100['p50_ms']) if p50_dp == 0 else fmt_f(n100['p50_ms'], p50_dp):>5} & "
                f"{fmt_f(n100['throughput_samples_sec'], 1):>6} & "
                f"{num_comma(round(float(n1k['p50_ms']))) if p50_dp == 0 else fmt_f(n1k['p50_ms'], p50_dp):>5} & "
                f"{fmt_f(n1k['throughput_samples_sec'], 1):>5} \\\\"
            )

    # PEFT base: bare model, no adapters -- the N-independent throughput ceiling.
    # Reported in both N blocks identically since it registers no adapters.
    for b in BATCHES:
        row = find_row(peft_base, num_adapters="1", batch_size=str(b))
        p50 = fmt_int(row["p50_ms"])
        qps = fmt_f(row["throughput_samples_sec"], 1)
        lines.append(
            f"Base ($\\mathcal{{B}}{{=}}{b}$) & "
            f"{p50:>5} & {qps:>6} & {p50:>5} & {qps:>5} \\\\"
        )
    lines.append(r"\midrule")

    emit_block(
        lambda b: f"PEFT grouped ($\\mathcal{{B}}{{=}}{b}$)",
        peft_grouped,
    )
    lines.append(r"\midrule")
    if peft_mixed is not None:
        emit_block(
            lambda b: f"PEFT mixed ($\\mathcal{{B}}{{=}}{b}$)",
            peft_mixed,
        )
        lines.append(r"\midrule")
    emit_block(
        lambda b: f"PEFT homog. ($\\mathcal{{B}}{{=}}{b}$)",
        peft_homog,
    )
    lines.append(r"\midrule")

    # LateFuse block (p50 with one decimal place, QPS as integer for headline)
    for b in BATCHES:
        n100 = find_row(sysname_ref, num_adapters="100", batch_size=str(b))
        n1k  = find_row(sysname_ref, num_adapters="1000", batch_size=str(b))
        bold = (b in (32, 128))
        def fmt_qps(x):
            v = fmt_int(x)
            return f"\\textbf{{{v}}}" if bold else v
        lines.append(
            rf"\sysname{{}} ($\mathcal{{B}}{{=}}{b}$) & "
            f"{fmt_f(n100['p50_ms'], 1)} & {fmt_qps(n100['throughput_samples_sec'])} & "
            f"{fmt_f(n1k['p50_ms'], 1)} & {fmt_qps(n1k['throughput_samples_sec'])} \\\\"
        )
    lines.append(r"\midrule")

    # Speedup rows span all three measured batch sizes (8/32/128) so the
    # abstract's min/max ranges are traceable in the table -- e.g. the mixed
    # maximum (B=8, N=1000) would otherwise be invisible here. Primary row is vs
    # PEFT's native mixed-batch API (the like-for-like baseline the abstract
    # leads with; bold); secondary is vs grouped per-adapter swapping (the
    # worst-case ceiling, plain). Falls back to grouped-only if mixed is absent.
    SPEEDUP_BATCHES = (8, 32, 128)

    def qps(rows, n, b):
        return float(find_row(rows, num_adapters=str(n),
                              batch_size=str(b))["throughput_samples_sec"])

    if peft_mixed is not None:
        for b in SPEEDUP_BATCHES:
            sp100 = qps(sysname_ref, 100, b) / qps(peft_mixed, 100, b)
            sp1k  = qps(sysname_ref, 1000, b) / qps(peft_mixed, 1000, b)
            # Speedup is a QPS ratio, so it goes in the QPS sub-column with the
            # p50 cell left blank (empty first cell of each N block).
            lines.append(
                rf"\textbf{{Speedup\textsubscript{{mixed}}}} ($\mathcal{{B}}{{=}}{b}$) & "
                rf" & \textbf{{{fmt_f(sp100, 1)}$\times$}} & "
                rf" & \textbf{{{fmt_f(sp1k, 1)}$\times$}} \\"
            )

    # The secondary Speedup_grouped rows are suppressed to save vertical space in
    # tab:baselines (6-page body limit); the grouped multipliers still appear in
    # Finding 7's prose, so no data is lost. To restore the three rows in the
    # table, flip SHOW_GROUPED_SPEEDUP to True. (When there is no mixed baseline,
    # grouped acts as the primary baseline and is always shown.)
    SHOW_GROUPED_SPEEDUP = False
    if SHOW_GROUPED_SPEEDUP or peft_mixed is None:
        if peft_mixed is not None:
            lines.append(r"\addlinespace")
        for b in SPEEDUP_BATCHES:
            sp100 = round(qps(sysname_ref, 100, b) / qps(peft_grouped, 100, b))
            sp1k  = round(qps(sysname_ref, 1000, b) / qps(peft_grouped, 1000, b))
            label   = r"Speedup\textsubscript{grouped}"
            cell100 = rf"{sp100}$\times$"
            cell1k  = rf"{sp1k}$\times$"
            if peft_mixed is None:  # no mixed row -> grouped is primary, bold it
                label   = rf"\textbf{{{label}}}"
                cell100 = rf"\textbf{{{cell100}}}"
                cell1k  = rf"\textbf{{{cell1k}}}"
            lines.append(
                rf"{label} ($\mathcal{{B}}{{=}}{b}$) & "
                rf" & {cell100} & "
                rf" & {cell1k} \\"
            )
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    return GEN_HEADER + "\n".join(lines) + "\n"


def render_table_accuracy(setfit, hidden: int, layers: int,
                          target_modules: int,
                          vanilla_trainable: int) -> str:
    """tab:accuracy body: per-task SetFit vanilla vs LoRA-r8.

    Arch params select the model (mpnet for table_accuracy.tex, bge-m3 for
    table_accuracy_bge.tex); they only affect the trainable-param footer row.
    A leading 'Frozen' column (head-only fit, no body adaptation) is added
    when the sweep includes frozen rows.
    """
    DATASETS = [
        ("SST-2",      "SetFit/sst2"),
        ("SST-5",      "SetFit/sst5"),
        ("CR",         "SetFit/CR"),
        ("Amazon-CF",  "SetFit/amazon_counterfactual_en"),
        ("Emotion",    "SetFit/emotion"),
        ("EnronSpam",  "SetFit/enron_spam"),
        ("AG News",    "SetFit/ag_news"),
    ]
    have_frozen = any(r["method"] == "frozen" for r in setfit)
    lines: list[str] = []
    if have_frozen:
        lines.append(r"\begin{tabular}{lrrrr}")
        lines.append(r"\toprule")
        lines.append(r"Dataset & Frozen & Full FT & LoRA-$r$8 & $\Delta$ \\")
    else:
        lines.append(r"\begin{tabular}{lrrr}")
        lines.append(r"\toprule")
        lines.append(r"Dataset & Full FT & LoRA-$r$8 & $\Delta$ \\")
    lines.append(r"\midrule")

    van_sum = lora_sum = frozen_sum = 0.0
    for label, key in DATASETS:
        van = float(next(r for r in setfit
                         if r["dataset"] == key and r["method"] == "vanilla"
                         )["accuracy"])
        lora = float(next(r for r in setfit
                          if r["dataset"] == key and r["method"] == "lora"
                          )["accuracy"])
        delta = round(lora, 3) - round(van, 3)  # match the table's column subtraction
        delta_str = f"$-${abs(delta):.3f}" if delta < 0 else f"{delta:.3f}"
        frozen_cell = ""
        if have_frozen:
            frozen = float(next(r for r in setfit
                                if r["dataset"] == key and r["method"] == "frozen"
                                )["accuracy"])
            frozen_cell = f"{frozen:.3f} & "
            frozen_sum += frozen
        lines.append(
            f"{label:<14s} & {frozen_cell}{van:.3f} & {lora:.3f} & {delta_str} \\\\"
        )
        van_sum += van
        lora_sum += lora

    van_mean = van_sum / len(DATASETS)
    lora_mean = lora_sum / len(DATASETS)
    mean_delta = lora_mean - van_mean
    mean_delta_str = (f"$-${abs(mean_delta):.3f}"
                      if mean_delta < 0 else f"{mean_delta:.3f}")
    frozen_mean_cell = ""
    if have_frozen:
        frozen_mean_cell = rf"\textbf{{{frozen_sum / len(DATASETS):.3f}}} & "
    lines.append(r"\midrule")
    lines.append(
        rf"\textbf{{Mean}}  & {frozen_mean_cell}\textbf{{{van_mean:.3f}}} & "
        rf"\textbf{{{lora_mean:.3f}}} & {mean_delta_str} \\"
    )

    # Trainable-param footer row (sourced from model_metadata.csv)
    lora_params_r8 = 2 * target_modules * layers * hidden * 8
    reduction = round(vanilla_trainable / lora_params_r8)
    van_M = round(vanilla_trainable / 1e6)
    lora_K = round(lora_params_r8 / 1000)
    frozen_params_cell = "0 & " if have_frozen else ""
    lines.append(
        f"Params & {frozen_params_cell}{van_M}M & {lora_K}K & "
        rf"$1/{reduction}$ \\"
    )
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    return GEN_HEADER + "\n".join(lines) + "\n"


# ===========================================================================
# Stdout summary
# ===========================================================================

def print_summary(macros_defined, sweep_main, peft_grouped, sysname_ref,
                  fwd, setfit, env, storage_reduction,
                  have_setfit_bge: bool = False) -> None:
    print(f"Wrote {len(macros_defined)} macros to paper/numbers.tex")
    tables = ("paper/table_main.tex, paper/table_baselines.tex, "
              "paper/table_accuracy.tex")
    if have_setfit_bge:
        tables += ", paper/table_accuracy_bge.tex"
    print(f"Wrote tabular bodies to {tables}")
    print()

    print("== ENVIRONMENT (from env_metadata.csv) ==")
    for k in ("gpu_name", "torch_version", "cuda_version",
              "transformers_version", "peft_version", "captured_at_utc"):
        if k in env:
            print(f"  {k}: {env[k]}")
    print()

    print("== HEADLINE NUMBERS (re-derived this run) ==")
    n47 = find_row(sweep_main, num_adapters="47000",
                   batch_size="32", lora_rank="8")
    print(f"  N=47000 B=32 r=8: p50={n47['p50_ms']} ms  "
          f"p99={n47['p99_ms']} ms  QPS={n47['throughput_samples_sec']}  "
          f"GPU={n47['peak_gpu_mem_gb']} GB")
    qps_sys_32_1k = float(find_row(sysname_ref, num_adapters="1000",
                                   batch_size="32")["throughput_samples_sec"])
    qps_peft_32_1k = float(find_row(peft_grouped, num_adapters="1000",
                                    batch_size="32")["throughput_samples_sec"])
    print(f"  Speedup at N=1000 B=32 (sys/peft-grouped): "
          f"{qps_sys_32_1k / qps_peft_32_1k:.1f}x")
    print(f"  LoRA self-CUDA share: "
          f"{fwd['bmm_isolation']['lora_bmm_self_share_pct']:.2f}%")
    print(f"  LoRA wall-clock share (ablation): "
          f"{fwd['ablation']['lora_cost_share_pct']:.2f}%")
    print()

    missing_cold_load = not any(
        r.get("adapter_load_total_s", "").strip() not in ("", "0", "0.0")
        for r in sweep_main)
    if missing_cold_load or not have_setfit_bge:
        print("== ITEMS NOT CAPTURED IN CURRENT CSVS ==")
        if missing_cold_load:
            print("  LateFuse cold-load wall time (column adapter_load_total_s):")
            print("    src/lora_serving/benchmark/run.py is instrumented; the column")
            print("    will appear once benchmarks/run_sxm80.sh is re-run. Until then,")
            print("    \\ColdStartSpeedupNOneK and \\LateFuseColdLoadSecNOneK are TODO.")
        if not have_setfit_bge:
            print("  SetFit accuracy on bge-m3 (SetfitBge* macros, "
                  "table_accuracy_bge.tex):")
            print("    run benchmarks/quality/run_setfit.sh on the GPU host to")
            print("    produce benchmarks/quality/setfit_bge_multiseed.csv, then")
            print("    re-run this script.")
        print()


if __name__ == "__main__":
    build()
