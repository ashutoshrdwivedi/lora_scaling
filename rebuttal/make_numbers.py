#!/usr/bin/env python3
"""Single source of truth for every number quoted in the EMNLP rebuttal.

Reads the raw result files under benchmarks/results/ and emits:
  rebuttal/numbers.json  -- flat {key: value} registry, machine-checkable
  rebuttal/NUMBERS.md    -- human-readable fact sheet

No number may appear in the rebuttal text unless it appears here (or in
check.py's literal allowlist). Run `python rebuttal/make_numbers.py` then
`python rebuttal/check.py`.

Protocol note: sweeps are 5 seeds x (warmup 50 / iters 200), fp16, seq 128.
Capacity probes are single-seed at B=32 only. PEFT arms are warmup 10 /
iters 50 on the same node.
"""
from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "benchmarks" / "results"
OUT_JSON = Path(__file__).resolve().parent / "numbers.json"
OUT_MD = Path(__file__).resolve().parent / "NUMBERS.md"

# ---------------------------------------------------------------- loading


def read_csv(path: Path, measured_only: bool = True) -> list[dict]:
    """Rows from a benchmark CSV, by default only the ones that measured.

    The runner records an OOM'd config as a row with status="oom" and blank
    metric columns, so the file says which cells were attempted and failed
    rather than leaving a silent gap. Those rows must not reach an aggregate:
    a blank p50 is not a zero, and in a capacity probe an OOM row carries the
    HIGHEST num_adapters, so a max() over unfiltered rows would report a
    ceiling the card demonstrably could not reach.

    Pass measured_only=False to see the failures too -- the capacity check
    below uses them to confirm the probe actually bracketed the ceiling. CSVs
    written before the column exists have no status and are all treated as
    measured.
    """
    with open(path) as fh:
        rows = list(csv.DictReader(fh))
    if measured_only:
        rows = [r for r in rows if r.get("status", "ok") == "ok"]
    return rows


def read_json(path: Path) -> dict:
    with open(path) as fh:
        return json.load(fh)


def num(x):
    return float(x) if x not in ("", None) else None


def _gib(gb: float) -> float:
    """Decimal GB -> binary GiB. Both units appear in this file; neither is
    allowed to borrow the other's name."""
    return gb * 1e9 / 1024**3


def _gpu_total_gb(env_path) -> float | None:
    """Card capacity as the driver reports it, from a run's env_metadata.csv."""
    if env_path is None or not env_path.exists():
        return None
    rows = read_csv(env_path)
    return num(rows[0].get("gpu_total_mem_gb")) if rows else None


# Each config: sweep CSV, capacity CSV(s), PEFT-mixed CSV, metadata CSV.
#
# ONE FILE PER ARTIFACT. Every path below is written by the matching
# benchmarks/run_rebuttal_*.sh, so the script is the single source of truth for
# its row. Earlier revisions also listed hand-run patch files --
# sweep_deberta_capacity2.csv, sweep_bgem3_l40s_n28k_expseg.csv -- created when
# a probe grid missed the ceiling or a column OOM'd mid-session. They are gone
# from this list on purpose. Do not add that kind of file back: a second source
# for one number is how the L40S row ended up with its ceiling in one CSV, its
# 28k latencies in another, and its provenance only in a comment.
#
# The capacity lists stay list-typed because one probe can legitimately have
# several arms in one file -- see run_rebuttal_l40s.sh, which measures both
# allocators into a single CSV keyed by 'alloc_conf'.
CONFIGS = {
    "bgem3_a100": dict(
        label="bge-m3 / A100-80GB (paper)",
        model="BAAI/bge-m3",
        sweep=RES / "sweep_main.csv",
        capacity=[RES / "sweep_capacity.csv"],
        peft_mixed=RES / "peft_mixed_sxm80.csv",
        meta=RES / "model_metadata.csv",
        env=RES / "env_metadata.csv",
        targets="query+value",
    ),
    "electra": dict(
        label="ELECTRA-large / A100-80GB",
        model="google/electra-large-discriminator",
        sweep=RES / "rebuttal_electra/sweep_electra_a100.csv",
        capacity=[RES / "rebuttal_electra/sweep_electra_capacity.csv"],
        peft_mixed=RES / "rebuttal_electra/peft_mixed_electra.csv",
        meta=RES / "rebuttal_electra/model_metadata.csv",
        env=RES / "rebuttal_electra/env_metadata.csv",
        targets="query+value",
    ),
    "deberta": dict(
        label="DeBERTa-v2-xlarge / A100-80GB",
        model="microsoft/deberta-v2-xlarge",
        sweep=RES / "rebuttal_deberta/sweep_deberta_a100.csv",
        capacity=[RES / "rebuttal_deberta/sweep_deberta_capacity.csv"],
        peft_mixed=RES / "rebuttal_deberta/peft_mixed_deberta.csv",
        meta=RES / "rebuttal_deberta/model_metadata.csv",
        env=RES / "rebuttal_deberta/env_metadata.csv",
        targets="value",
    ),
    "xlmr_xl": dict(
        label="XLM-RoBERTa-XL / A100-80GB",
        model="facebook/xlm-roberta-xl",
        sweep=RES / "rebuttal_xlmr_xl/sweep_xlmrxl_a100.csv",
        capacity=[RES / "rebuttal_xlmr_xl/sweep_xlmrxl_capacity.csv"],
        peft_mixed=RES / "rebuttal_xlmr_xl/peft_mixed_xlmrxl.csv",
        meta=RES / "rebuttal_xlmr_xl/model_metadata.csv",
        env=RES / "rebuttal_xlmr_xl/env_metadata.csv",
        targets="query+value",
    ),
    "l40s": dict(
        label="bge-m3 / L40S-48GB",
        model="BAAI/bge-m3",
        sweep=RES / "rebuttal_l40s/sweep_bgem3_l40s.csv",
        capacity=[RES / "rebuttal_l40s/sweep_capacity_l40s.csv"],
        peft_mixed=RES / "rebuttal_l40s/peft_mixed_l40s.csv",
        meta=RES / "rebuttal_l40s/model_metadata.csv",
        env=RES / "rebuttal_l40s/env_metadata.csv",
        targets="query+value",
    ),
}

BASE_RANK = 8

# The allocator setting the rebuttal scripts ship, and therefore the one the
# reported ceilings are measured under. Rows carry it in 'alloc_conf'; see the
# two-arm capacity probe in benchmarks/run_rebuttal_l40s.sh.
SHIPPED_ALLOC_CONF = "expandable_segments:True"


# ---------------------------------------------------------------- helpers


def cell(rows, n, b, col, r=BASE_RANK):
    """Mean-across-seeds value of `col` for one (N, B, r) cell.

    Mean, not median: paper/build_numbers.py's aggregate_seeds() uses
    statistics.fmean, and the rebuttal must aggregate identically or the new
    rows will not be comparable to the paper's bge-m3 row.
    """
    vals = [
        num(x[col])
        for x in rows
        if int(x["num_adapters"]) == n
        and int(x["batch_size"]) == b
        and int(x["lora_rank"]) == r
    ]
    return statistics.fmean(vals) if vals else None


def cell_p50(rows, n, b, r=BASE_RANK):
    return cell(rows, n, b, "p50_ms", r)


def seed_sd_pct(rows, n, b, r=BASE_RANK):
    vals = [
        num(x["p50_ms"])
        for x in rows
        if int(x["num_adapters"]) == n
        and int(x["batch_size"]) == b
        and int(x["lora_rank"]) == r
    ]
    if len(vals) < 2:
        return None
    return 100.0 * statistics.stdev(vals) / statistics.mean(vals)


def analyse_config(cfg: dict) -> dict:
    sweep = read_csv(cfg["sweep"])
    out: dict = {"label": cfg["label"], "targets": cfg["targets"]}

    ns = sorted({int(r["num_adapters"]) for r in sweep})
    bs = sorted({int(r["batch_size"]) for r in sweep})
    ranks = sorted({int(r["lora_rank"]) for r in sweep})
    out["sweep_n_values"] = ns
    out["sweep_batch_sizes"] = bs
    out["sweep_ranks"] = ranks
    out["sweep_rows"] = len(sweep)

    # --- Spread: worst-case total (max-min)/min of p50 across the N sweep,
    #     taken over batch sizes. Conservative: folds in non-monotone seed
    #     and cache-locality variation, and is anchored at N=100 (noisiest).
    spreads = {}
    for b in bs:
        series = [(n, cell_p50(sweep, n, b)) for n in ns]
        series = [(n, v) for n, v in series if v is not None]
        if len(series) < 2:
            continue
        vals = [v for _, v in series]
        spreads[b] = 100.0 * (max(vals) - min(vals)) / min(vals)
    out["spread_pct_by_batch"] = {str(k): round(v, 2) for k, v in spreads.items()}
    out["spread_pct_worst"] = round(max(spreads.values()), 2)
    out["spread_pct_worst_batch"] = max(spreads, key=spreads.get)

    # --- Noise floor: between-seed s.d. as % of p50, worst cell.
    noise = {}
    for n in ns:
        for b in bs:
            sd = seed_sd_pct(sweep, n, b)
            if sd is not None:
                noise[(n, b)] = sd
    worst = max(noise, key=noise.get)
    out["seed_sd_pct_worst"] = round(noise[worst], 2)
    out["seed_sd_pct_worst_cell"] = {"n": worst[0], "b": worst[1]}

    # --- Rank insensitivity (Finding 3): r in {4,8,16,32} at N=1000, B=32.
    rank_cells = {r: cell_p50(sweep, 1000, 32, r) for r in ranks}
    rank_cells = {r: v for r, v in rank_cells.items() if v is not None}
    if len(rank_cells) > 1:
        rv = list(rank_cells.values())
        out["rank_p50_ms"] = {str(k): round(v, 2) for k, v in rank_cells.items()}
        out["rank_spread_pct"] = round(100.0 * (max(rv) - min(rv)) / min(rv), 2)

    # --- Ceiling: highest N that ran, from the capacity probes (B=32).
    cap_rows = []
    cap_all = []
    for p in cfg["capacity"]:
        if p.exists():
            cap_rows += read_csv(p)
            cap_all += read_csv(p, measured_only=False)
    cap32 = [r for r in cap_rows if int(r["batch_size"]) == 32]

    # A probe is only informative if it found the edge. If every cell fit, the
    # ceiling is a lower bound, not a measurement -- the grid stopped too low.
    # This is the mirror of the sweep's --require-complete check: there a
    # missing cell is the fault, here a missing FAILURE is.
    probed = [r for r in cap_all if int(r["batch_size"]) == 32]
    if probed and any("status" in r for r in probed):
        if not any(r.get("status") == "oom" for r in probed):
            top = max(int(r["num_adapters"]) for r in probed)
            print(
                f"  WARNING [{cfg['label']}]: capacity probe never OOM'd; every "
                f"cell up to N={top} fit. The reported ceiling is a lower bound, "
                "not the measured limit -- raise the probe grid and re-run."
            )
        elif not any(r.get("status") == "ok" for r in probed):
            low = min(int(r["num_adapters"]) for r in probed)
            print(
                f"  WARNING [{cfg['label']}]: capacity probe found no fitting "
                f"cell -- even N={low} OOM'd, so the grid starts above the "
                "ceiling and brackets nothing. No ceiling is reported for this "
                "config. Lower the probe grid and re-run."
            )
    # The L40S probe measures BOTH allocator settings into one CSV, told apart
    # by 'alloc_conf'. A plain max() over that file would report whichever arm
    # happened to reach higher without naming which -- right by luck today
    # (expandable_segments always wins) but not something to leave implicit in
    # a number that goes to reviewers. Pin the ceiling to the shipped setting.
    # CSVs written before the column exists fall through unchanged.
    if any("alloc_conf" in r for r in cap32):
        shipped = [r for r in cap32 if r.get("alloc_conf") == SHIPPED_ALLOC_CONF]
        if shipped:
            cap32 = shipped
        else:
            print(
                f"  WARNING [{cfg['label']}]: no capacity rows at "
                f"{SHIPPED_ALLOC_CONF!r}; ceiling falls back to whatever ran. "
                "Did the sweep script export PYTORCH_CUDA_ALLOC_CONF?"
            )
    if cap32:
        top = max(cap32, key=lambda r: int(r["num_adapters"]))
        ceil_n = int(top["num_adapters"])
        # If the sweep itself reached higher (bge-m3 47k lives in the sweep),
        # prefer the larger.
        if ns[-1] > ceil_n and cell_p50(sweep, ns[-1], 32) is not None:
            ceil_n = ns[-1]
            ceil_p50 = cell_p50(sweep, ceil_n, 32)
            ceil_mem = statistics.median(
                [
                    num(r["peak_gpu_mem_gb"])
                    for r in sweep
                    if int(r["num_adapters"]) == ceil_n and int(r["batch_size"]) == 32
                ]
            )
        else:
            ceil_p50 = num(top["p50_ms"])
            ceil_mem = num(top["peak_gpu_mem_gb"])
        out["ceiling_n"] = ceil_n
        out["ceiling_p50_ms_b32"] = round(ceil_p50, 2)
        out["ceiling_peak_mem_gb"] = round(ceil_mem, 1)

        # Default-allocator ceiling, when the probe measured both arms (today
        # only L40S). This is the 26k side of the "+7.7% free capacity" claim,
        # which otherwise lives in prose with nothing to trace back to. Emitting
        # it lets rebuttal/check.py verify the comparison instead of
        # whitelisting the number. Absent on CSVs without an 'alloc_conf'
        # column, in which case the claim is unverifiable and check.py must
        # keep the literal allowed.
        default32 = [
            r
            for r in cap_rows
            if int(r["batch_size"]) == 32 and r.get("alloc_conf") == "default"
        ]
        if default32:
            ceil_default = max(int(r["num_adapters"]) for r in default32)
            out["ceiling_n_default_alloc"] = ceil_default
            out["ceiling_gain_from_expandable_segments_pct"] = round(
                100.0 * (ceil_n - ceil_default) / ceil_default, 1
            )
        base = cell_p50(sweep, 1000, 32)
        out["n1000_p50_ms_b32"] = round(base, 2)
        out["at_ceiling_vs_n1000_pct"] = round(100.0 * (ceil_p50 - base) / base, 2)

        # The ceiling probe is B=32-only on the A100 configs; the L40S has a
        # full 5-seed x 5-batch re-run at its ceiling. Where all batch sizes
        # exist, report the worst one -- that is the binding case for any
        # "within X% of N=1,000" claim.
        by_b = {}
        for b in bs:
            ceil_b = cell_p50(cap_rows, ceil_n, b) or cell_p50(sweep, ceil_n, b)
            base_b = cell_p50(sweep, 1000, b)
            if ceil_b and base_b:
                by_b[b] = 100.0 * (ceil_b - base_b) / base_b
        out["at_ceiling_pct_by_batch"] = {str(k): round(v, 2) for k, v in by_b.items()}
        out["at_ceiling_pct_worst"] = round(max(by_b.values()), 2)
        out["at_ceiling_worst_batch"] = max(by_b, key=by_b.get)
        out["at_ceiling_batches_measured"] = sorted(by_b)

    # --- Speedup vs PEFT mixed-batch, over the cells PEFT covers.
    #     THROUGHPUT ratio (ours/PEFT samples/sec) -- the paper's definition in
    #     build_numbers.py (SpeedupMixed*). The p50-latency ratio is kept
    #     alongside it because the two differ by a few tenths and mixing them
    #     across table rows would be an unforced inconsistency.
    peft = read_csv(cfg["peft_mixed"])
    sp_tput, sp_lat = {}, {}
    for r in peft:
        n, b = int(r["num_adapters"]), int(r["batch_size"])
        ours_q = cell(sweep, n, b, "throughput_samples_sec")
        ours_p = cell_p50(sweep, n, b)
        if ours_q:
            sp_tput[(n, b)] = ours_q / num(r["throughput_samples_sec"])
        if ours_p:
            sp_lat[(n, b)] = num(r["p50_ms"]) / ours_p
    if sp_tput:
        out["speedup_cells"] = {f"N{n}_B{b}": round(v, 2) for (n, b), v in sorted(sp_tput.items())}
        out["speedup_min"] = round(min(sp_tput.values()), 1)
        out["speedup_max"] = round(max(sp_tput.values()), 1)
        out["speedup_basis"] = "throughput ratio (paper convention)"
    if sp_lat:
        out["speedup_latency_cells"] = {
            f"N{n}_B{b}": round(v, 2) for (n, b), v in sorted(sp_lat.items())
        }
        out["speedup_latency_min"] = round(min(sp_lat.values()), 1)
        out["speedup_latency_max"] = round(max(sp_lat.values()), 1)

    # --- Adapter geometry from model metadata.
    meta = {r["key"]: r["value"] for r in read_csv(cfg["meta"]) if r["model"] == cfg["model"]}
    if meta:
        out["hidden_size"] = int(meta["hidden_size"])
        out["num_layers"] = int(meta["num_layers"])
        out["total_params_m"] = round(int(meta["total_params"]) / 1e6, 1)
        out["base_fp16_gb"] = round(int(meta["total_params"]) * 2 / 1e9, 2)
        b8 = int(meta["lora_bytes_fp16_r8"])
        out["bytes_per_adapter_r8"] = b8
        out["params_per_adapter_r8"] = int(meta["lora_params_r8"])
        # Decimal MB is what the draft quotes; MiB kept so the two never get
        # silently mixed (1.57 MB == 1.50 MiB for a 24x1024 query+value pair).
        out["mb_per_adapter_r8"] = round(b8 / 1e6, 2)
        out["mib_per_adapter_r8"] = round(b8 / 1024**2, 2)
        # Two units, never one. `_gb` is decimal (1e9), `_gib` binary (2^30);
        # nvidia-smi and torch report decimal, so a GiB value wearing a GB
        # label makes the memory budget stop adding up (68.8 + 1.14 != 76.2,
        # while 73.9 + 1.14 ~= 76.2 once both sides are decimal).
        out["base_fp16_gib"] = round(int(meta["total_params"]) * 2 / 1024**3, 2)
        if "ceiling_n" in out:
            out["store_gb_at_ceiling"] = round(b8 * out["ceiling_n"] / 1e9, 1)
            out["store_gib_at_ceiling"] = round(b8 * out["ceiling_n"] / 1024**3, 1)
        else:
            out["store_gb_at_ceiling"] = None
            out["store_gib_at_ceiling"] = None
        # Card capacity as the driver reports it, and what the tenant ceiling
        # leaves unused. Headroom is an upper bound: it is measured at the
        # largest pool that ran, and the next probe point OOM'd.
        total_gb = _gpu_total_gb(cfg.get("env"))
        if total_gb:
            out["gpu_total_gb"] = round(total_gb, 1)
            out["gpu_total_gib"] = round(_gib(total_gb), 1)
            if out.get("ceiling_peak_mem_gb"):
                peak_gib = _gib(out["ceiling_peak_mem_gb"])
                out["ceiling_peak_mem_gib"] = round(peak_gib, 1)
                out["headroom_upper_bound_gib"] = round(_gib(total_gb) - peak_gib, 1)
        # Analytic FLOP ratio bounding what a free LoRA kernel could recover.
        out["flop_ratio_3d_over_r"] = int(3 * out["hidden_size"] / BASE_RANK)
        out["max_recoverable_flop_pct"] = round(100.0 / out["flop_ratio_3d_over_r"], 2)

    # --- Memory-vs-load: does batch size move the tenant ceiling?
    #     Use the largest N whose peak-memory readings are consistent across
    #     batch sizes (an N re-run in a separate process can carry a different
    #     allocator high-water mark, which would fake a huge delta).
    big_n = ns[-1]
    mems = {
        b: statistics.median(
            [
                num(r["peak_gpu_mem_gb"])
                for r in sweep
                if int(r["num_adapters"]) == big_n
                and int(r["batch_size"]) == b
                and int(r["lora_rank"]) == BASE_RANK
            ]
            or [float("nan")]
        )
        for b in bs
    }
    mems = {b: v for b, v in mems.items() if v == v}
    if mems:
        out["mem_vs_batch_n"] = big_n
        out["mem_vs_batch_gb"] = {str(k): round(v, 1) for k, v in mems.items()}
        out["mem_vs_batch_delta_gb"] = round(max(mems.values()) - min(mems.values()), 1)

    # --- Registration cost (our side): adapter store fill, seconds/adapter.
    loads = [
        (int(r["num_adapters"]), num(r["adapter_load_total_s"]))
        for r in sweep
        if r.get("adapter_load_total_s") and int(r["num_adapters"]) == 1000
    ]
    if loads:
        per = statistics.median([t / n for n, t in loads])
        out["register_ms_per_adapter_at_n1000"] = round(per * 1000, 2)

    # --- PEFT registration: add_adapter total time, showing O(N^2).
    adds = sorted(
        {(int(r["num_adapters"]), num(r["add_adapter_total_s"])) for r in peft if r.get("add_adapter_total_s")}
    )
    if len(adds) >= 2:
        out["peft_add_total_s"] = {str(n): round(t, 1) for n, t in adds}
        (n0, t0), (n1, t1) = adds[0], adds[-1]
        out["peft_add_growth_ratio"] = round((t1 / n1) / (t0 / n0), 1)
        # marginal cost of the N-th add under a quadratic total: 2*T/N
        out["peft_marginal_add_s_at_top"] = round(2 * t1 / n1, 1)
        if "register_ms_per_adapter_at_n1000" in out:
            out["peft_vs_ours_add_ratio"] = int(
                round(2 * t1 / n1 / (out["register_ms_per_adapter_at_n1000"] / 1000), -2)
            )
    return out


# ------------------------------------------------------- assembly benches


def analyse_assembly(path: Path) -> dict:
    d = read_json(path)
    agg = d["aggregated"]
    batches = sorted(int(b) for b in agg)
    out = {
        "model": d["config"]["model"],
        "gpu": d["gpu"],
        "cpu": d.get("env", {}).get("cpu"),
        "cpu_cgroup_quota": d.get("env", {}).get("cpu_cgroup_quota"),
        "warmup": d["config"]["warmup"],
        "iters": d["config"]["iters"],
        "seeds": len(d["config"]["seeds"]),
        "num_adapters": d["config"]["num_adapters"],
        "batch_sizes": batches,
    }

    def g(b, variant, field):
        return agg[str(b)][variant][field]["mean"]

    # Single-stream throughput = batch / end-to-end latency (assemble->forward
    # sequentially). This is a per-replica compute bound, not a load test.
    for variant in ("baseline", "indexsel", "indexsel_compile"):
        tput = {b: b / (g(b, variant, "total_mean_ms") / 1000.0) for b in batches}
        out[f"{variant}_tput_by_batch"] = {str(k): int(round(v)) for k, v in tput.items()}
        out[f"{variant}_tput_max"] = int(round(max(tput.values())))
        out[f"{variant}_tput_at_large_batches"] = [
            int(round(tput[b])) for b in batches if b >= 64
        ]
    out["speedup_by_batch"] = {
        str(b): round(g(b, "baseline", "total_mean_ms") / g(b, "indexsel", "total_mean_ms"), 2)
        for b in batches
    }
    out["speedup_min"] = round(min(out["speedup_by_batch"].values()), 2)
    out["speedup_max"] = round(max(out["speedup_by_batch"].values()), 2)

    for variant in ("baseline", "indexsel"):
        shares = {b: g(b, variant, "assemble_share_pct") for b in batches}
        out[f"{variant}_assemble_share_pct"] = {str(k): round(v, 1) for k, v in shares.items()}
        out[f"{variant}_assemble_share_max"] = round(max(shares.values()), 1)
        out[f"{variant}_assemble_share_min"] = round(min(shares.values()), 1)
        big = [v for b, v in shares.items() if b >= 64]
        if big:
            out[f"{variant}_assemble_share_b64plus"] = [round(min(big), 1), round(max(big), 1)]
        tails = {b: g(b, variant, "p99_over_p50") for b in batches}
        out[f"{variant}_tail_ratio"] = {str(k): round(v, 2) for k, v in tails.items()}
        out[f"{variant}_tail_ratio_max"] = round(max(tails.values()), 2)
        if "scatter_share_pct" in agg[str(batches[0])][variant]:
            sc = {b: g(b, variant, "scatter_share_pct") for b in batches}
            out[f"{variant}_scatter_share_pct"] = {str(k): round(v, 1) for k, v in sc.items()}
            out[f"{variant}_scatter_share_min"] = round(min(sc.values()), 1)
            out[f"{variant}_scatter_share_max"] = round(max(sc.values()), 1)
            out[f"{variant}_scatter_ms"] = {
                str(b): round(g(b, variant, "scatter_mean_ms"), 2) for b in batches
            }
    return out


# ---------------------------------------------------------- mixed rank


def analyse_mixed_rank(path: Path) -> dict:
    """Mixed-rank arms: does flat-in-N survive a heterogeneous fleet?

    Two quantities carry the answer. The PADDING TAX is what a low-rank tenant
    pays for sharing a batch with high-rank tenants (mixed vs a uniform batch at
    the fleet's lowest rank) -- the cost the uniform-rank limitation is about.
    The OVERHEAD VS R_MAX is whether a mixed batch costs anything beyond its
    padded shape; zero-padding makes the batch tensors identical to a uniform
    r_max batch, so this is the check that no gather cost sneaks in.

    The N-spread of the mixed arm is computed the same way as the main configs'
    spread_pct_worst -- (max-min)/min over the pool sweep -- so the mixed-rank
    curve and the paper's Figure 3 curve are directly comparable numbers.
    """
    d = read_json(path)
    out = {
        "model": d["config"]["model"],
        "gpu": d["gpu"],
        "batch_size": d["config"]["batch_size"],
        "adapters": d["config"]["adapters"],
        "assembler": d["config"]["assembler"],
        "warmup": d["config"]["warmup"],
        "iters": d["config"]["iters"],
        "seeds": len(d["seeds"]),
        "mixes": [m["label"] for m in d["mixes"]],
        # The exactness gate ran before any timing; if padding had changed the
        # served result the benchmark would have aborted, so this is the parity
        # number backing "zero-padding is exact", not a tolerance we chose after
        # seeing the data.
        # fp32 only. The `_serving_dtype` sibling is deliberately excluded: it
        # is the same comparison run in fp16, where the native and padded paths
        # use different GEMM shapes and so accumulate in different orders. It is
        # a real number and it is carried below under its own key, but folding
        # it into `max()` here would report accumulation noise as the parity of
        # the padding identity and put an ~1e-1 where an exact 0.0 belongs.
        "exactness_max_abs_delta": max(
            v
            for e in d["exactness"]
            for k, v in e.items()
            if k.startswith("max_abs_delta")
            and not k.endswith("_serving_dtype")
        ),
        "exactness_gate_dtype": d["exactness"][0].get("gate_dtype", "fp32"),
        # What padding costs a tenant in the dtype actually served, as a
        # fraction of the logit scale. Quote this, not the fp32 zero, when the
        # claim is about what a mixed-rank fleet does to outputs in production.
        "exactness_serving_dtype_max_abs_delta": max(
            (
                e["max_abs_delta_padded_vs_native_serving_dtype"]
                for e in d["exactness"]
                if "max_abs_delta_padded_vs_native_serving_dtype" in e
            ),
            default=None,
        ),
        "exactness_serving_dtype_pct_of_logit_scale": max(
            (
                100.0
                * e["max_abs_delta_padded_vs_native_serving_dtype"]
                / e["serving_dtype_mean_abs_logit"]
                for e in d["exactness"]
                if e.get("serving_dtype_mean_abs_logit")
            ),
            default=None,
        ),
    }

    agg = d["aggregated"]

    def p50(n: int, arm: str):
        key = f"{n}|{arm}"
        return agg[key]["total_p50_ms"]["mean"] if key in agg else None

    by_mix: dict = {}
    taxes, fwd_taxes, overheads, store_ratios, at_max, spreads = [], [], [], [], [], []
    for mix in d["mixes"]:
        label = mix["label"]
        cells = {}
        for key, dv in d["derived"].items():
            n_str, mix_label = key.split("|", 1)
            if mix_label != label:
                continue
            cells[n_str] = {
                "padding_tax_pct": round(dv["padded_padding_tax_vs_rmin_pct"], 2),
                "padding_tax_forward_pct": round(dv["padded_padding_tax_forward_pct"], 2),
                "overhead_vs_rmax_pct": round(dv["padded_overhead_vs_rmax_pct"], 2),
                "native_padding_tax_pct": round(dv["native_padding_tax_vs_rmin_pct"], 2),
                "native_store_vs_padded": round(dv["native_store_vs_padded"], 3),
                "batches_at_fleet_max_pct": round(dv["padded_batches_at_fleet_max_pct"], 1),
                "uniform_rank_delta_pct": round(dv["uniform_rank_delta_pct"], 2),
            }
            taxes.append(dv["padded_padding_tax_vs_rmin_pct"])
            fwd_taxes.append(dv["padded_padding_tax_forward_pct"])
            overheads.append(dv["padded_overhead_vs_rmax_pct"])
            store_ratios.append(dv["native_store_vs_padded"])
            at_max.append(dv["padded_batches_at_fleet_max_pct"])
        if not cells:
            continue

        # Flat-in-N for the mixed fleet itself: the reviewer's literal question.
        mixed_arm = f"mixed_{label}_padded"
        series = [v for n in d["config"]["adapters"] if (v := p50(n, mixed_arm))]
        entry = {"by_n": cells}
        if len(series) > 1:
            entry["p50_ms_by_n"] = {
                str(n): round(p50(n, mixed_arm), 2)
                for n in d["config"]["adapters"]
                if p50(n, mixed_arm)
            }
            spread = 100.0 * (max(series) - min(series)) / min(series)
            entry["spread_pct_over_n"] = round(spread, 2)
            spreads.append(spread)
        by_mix[label] = entry

    out["by_mix"] = by_mix
    if taxes:
        out["padding_tax_pct_max"] = round(max(taxes), 2)
        out["padding_tax_pct_min"] = round(min(taxes), 2)
        out["padding_tax_forward_pct_max"] = round(max(fwd_taxes), 2)
        out["overhead_vs_rmax_pct_max"] = round(max(overheads), 2)
        out["overhead_vs_rmax_pct_min"] = round(min(overheads), 2)
        out["native_store_vs_padded_min"] = round(min(store_ratios), 3)
        out["batches_at_fleet_max_pct_min"] = round(min(at_max), 1)
    if spreads:
        out["spread_pct_over_n_max"] = round(max(spreads), 2)
    if d["oom_cells"]:
        out["oom_cells"] = len(d["oom_cells"])
    return out


# ------------------------------------------------------- registration churn

CHURN_DIR = RES / "churn_registration_a100"

# Only the blocking arm is quotable as production behaviour: deploy/server/
# reload.py takes the same inference lock the serving path holds before it
# invokes the reload callback, so registration is serialised against inference
# in the shipped system. The background arm bounds what an overlapped design
# could buy, and at what tail cost; see CAVEATS.md in the results directory.
CHURN_MODE = "blocking"


def _t95(n: int) -> float:
    """Two-sided 95% t multiplier for n observations (n-1 df), n <= 10."""
    return {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571}.get(n, 1.96)


def _churn_group(rows: list[dict], keys: tuple[str, ...]) -> dict:
    out: dict = {}
    for r in rows:
        if r["churn_mode"] != CHURN_MODE:
            continue
        out.setdefault(tuple(r[k] for k in keys), []).append(r)
    return out


def _churn_cell(rows: list[dict]) -> dict:
    """One (rate, N) cell, averaged over seeds.

    Registration cost is reported as the within-run median (replace_p50_ms),
    not the mean. One seed of the 1 update/s ceiling cell pays a single cold
    first-touch registration that lifts that run's *mean* to 20.7 ms while its
    median stays at 11.8 ms; a mean over means would report that one stall as
    if it were the typical cost. The stall is not hidden -- replace_p95_ms
    carries it -- and the paper reports both columns.
    """
    f = lambda c: statistics.fmean(float(x[c]) for x in rows)

    def first(*names):
        """First column present. The corrected harness renamed two columns
        after the primary CSVs were written, so both schemas are in the tree."""
        for n in names:
            if n in rows[0]:
                return f(n)
        raise KeyError(names)

    return {
        "seeds": len(rows),
        "batch_size": int(rows[0]["batch_size"]),
        # share of pod wall-clock capacity spent registering rather than
        # serving -- a ratio of accumulated times, NOT a per-request mean.
        "registration_share_pct": round(f("registration_share_pct"), 2),
        "register_p50_ms": round(f("replace_p50_ms"), 2),
        "register_p95_ms": round(f("replace_p95_ms"), 2),
        "register_mean_ms": round(f("replace_mean_ms"), 2),
        "serve_p50_ms": round(f("serve_p50_ms"), 2),
        "serve_p99_ms": round(f("serve_p99_ms"), 2),
        # serve_throughput_samples_sec excludes the intervals the loop spent
        # registering, so under blocking churn it overstates what the pod
        # delivers. Recompute the honest denominator: served / wall clock.
        "throughput_while_serving": int(round(first(
            "serve_throughput_samples_sec", "serve_throughput_while_serving"))),
        "delivered_throughput": int(round(statistics.fmean(
            float(x["batches_served"]) * int(x["batch_size"]) / float(x["wall_s"])
            for x in rows))),
        "achieved_admission_rate": round(f("achieved_admission_rate"), 2),
        "rate_attainment": round(f("rate_attainment"), 3),
        "sustainable_rate_at_1pct": round(f("sustainable_rate_at_1pct"), 2),
        "saturation_rate": round(f("saturation_rate"), 1),
        "peak_gpu_mem_gb": round(f("peak_gpu_mem_gb"), 2),
        "adapter_cache_gb": round(f("adapter_cache_gb"), 2),
    }


def _head_install_ms() -> dict:
    """Classification-head install cost, from the host-2 diagnostic run."""
    txt = (CHURN_DIR / "host2_replication" / "diag_breakdown.txt").read_text()
    synth, pinned = [], []
    for line in txt.splitlines():
        if "on-device head synthesis" in line:
            synth.append(float(line.split()[-2]))
        elif "pinned head copy" in line:
            pinned.append(float(line.split()[-2]))
    return {
        "head_synth_ms_range": [min(synth), max(synth)],
        "head_pinned_copy_ms_range": [min(pinned), max(pinned)],
        "head_install_ms": round(statistics.fmean(pinned), 2),
        "head_repeats": len(pinned),
    }


def analyse_churn() -> dict:
    """Adapter registration measured while the same process keeps serving.

    Answers "what does a live adapter update cost a serving pod, and does that
    cost grow with the resident pool?" The registration path is the production
    AdapterStore.load_from_file: adapter file on disk -> GPU weights.
    """
    paths = read_csv(CHURN_DIR / "churn_registration_paths.csv")
    rate = read_csv(CHURN_DIR / "churn_rate_sweep.csv")
    ceil = read_csv(CHURN_DIR / "churn_rate_sweep_ceiling.csv")
    host2 = read_csv(CHURN_DIR / "host2_replication" / "host2_registration_paths.csv")

    out: dict = {
        "model": paths[0]["model"],
        "gpu": "NVIDIA A100-SXM4-80GB",
        "mode": CHURN_MODE,
        "batch_size": int(paths[0]["batch_size"]),
        "lora_rank": int(paths[0]["lora_rank"]),
        "note": "blocking arm only; see CAVEATS.md alongside the CSVs",
    }
    out.update(_head_install_ms())

    # ---- registration cost by path, at the two pool sizes
    by_path: dict = {}
    for (path, resident), rows in _churn_group(paths, ("registration_path", "resident")).items():
        by_path.setdefault(path, {})[resident] = _churn_cell(rows)
    out["by_path"] = by_path

    # ---- the O(1)-in-N claim, on the production file path
    small, big = "1000", "47000"
    f_small, f_big = by_path["file"][small], by_path["file"][big]
    out["pool_small"] = int(small)
    out["pool_large"] = int(big)
    out["pool_growth"] = int(big) // int(small)
    out["file_ms_small"] = f_small["register_mean_ms"]
    out["file_ms_large"] = f_big["register_mean_ms"]
    out["file_drift_pct"] = round(
        100.0 * (f_big["register_mean_ms"] - f_small["register_mean_ms"])
        / f_small["register_mean_ms"], 2)

    # 95% CI half-width on each cell's seed mean; quote the wider of the two.
    cis = []
    for resident in (small, big):
        vals = [float(r["replace_mean_ms"]) for r in paths
                if r["churn_mode"] == CHURN_MODE and r["registration_path"] == "file"
                and r["resident"] == resident]
        cis.append(_t95(len(vals)) * statistics.stdev(vals) / len(vals) ** 0.5)
    out["file_ci95_ms"] = round(max(cis), 2)

    # ---- second host: the scaling replicates, the constant does not
    h2: dict = {}
    for (path, resident), rows in _churn_group(host2, ("registration_path", "resident")).items():
        h2.setdefault(path, {})[resident] = _churn_cell(rows)
    out["host2_by_path"] = h2
    out["host2_file_ms_small"] = h2["file"][small]["register_mean_ms"]
    out["host2_file_ms_large"] = h2["file"][big]["register_mean_ms"]
    out["host2_file_drift_pct"] = round(
        100.0 * (h2["file"][big]["register_mean_ms"] - h2["file"][small]["register_mean_ms"])
        / h2["file"][small]["register_mean_ms"], 2)
    out["host2_slower_pct"] = round(
        100.0 * (h2["file"][small]["register_mean_ms"] - f_small["register_mean_ms"])
        / f_small["register_mean_ms"], 1)

    # ---- admission-rate sweep: what churn costs a pod's serving capacity
    sweep: dict = {}
    for (r_target, resident), rows in _churn_group(
            rate + ceil, ("target_admission_rate", "resident")).items():
        sweep.setdefault(str(float(r_target)), {})[resident] = _churn_cell(rows)
    out["rate_sweep"] = sweep

    quoted = [(r, n) for r in ("1.0", "10.0") for n in (small, big)]
    out["rate_sweep_quoted"] = [r for r, _ in quoted]
    out["register_p50_ms_quoted_range"] = [
        min(sweep[r][n]["register_p50_ms"] for r, n in quoted),
        max(sweep[r][n]["register_p50_ms"] for r, n in quoted),
    ]
    out["serve_p99_ms_quoted_range"] = [
        min(sweep[r][n]["serve_p99_ms"] for r, n in quoted),
        max(sweep[r][n]["serve_p99_ms"] for r, n in quoted),
    ]
    out["share_pct_at_1_per_s"] = [sweep["1.0"][n]["registration_share_pct"] for n in (small, big)]
    out["share_pct_at_10_per_s"] = [sweep["10.0"][n]["registration_share_pct"] for n in (small, big)]
    out["sustainable_rate_at_1pct"] = round(statistics.fmean(
        sweep[r][n]["sustainable_rate_at_1pct"] for r, n in quoted), 2)
    out["saturation_rate"] = round(statistics.fmean(
        sweep[r][n]["saturation_rate"] for r, n in quoted), 0)
    return out


# ------------------------------------------------- allocator capacity probe

ALLOC_PROBE = dict(
    key="bgem3_a100_alloc",
    model="BAAI/bge-m3",
    sweep=RES / "rebuttal_bgem3_alloc/sweep_capacity_bgem3_alloc.csv",
    meta=RES / "rebuttal_bgem3_alloc/model_metadata.csv",
    env=RES / "rebuttal_bgem3_alloc/env_metadata.csv",
)


def analyse_capacity_probe(cfg: dict) -> dict:
    """Where the tenant ceiling actually sits, per CUDA allocator setting.

    Run as two arms in one CSV keyed by `alloc_conf`, so the only difference
    between them is PYTORCH_CUDA_ALLOC_CONF. Each arm walks N upward until the
    allocation fails; the OOM rows are kept (status="oom") so the file records
    that the probe bracketed the ceiling rather than merely stopping.
    """
    rows_all = read_csv(cfg["sweep"], measured_only=False)
    meta = {r["key"]: r["value"] for r in read_csv(cfg["meta"]) if r["model"] == cfg["model"]}
    b8 = int(meta["lora_bytes_fp16_r8"])
    total_gb = _gpu_total_gb(cfg["env"])

    out: dict = {
        "model": cfg["model"],
        "source": str(cfg["sweep"].relative_to(ROOT)),
        "analytic_bytes_per_adapter": b8,
        "base_fp16_gb": round(int(meta["total_params"]) * 2 / 1e9, 3),
        "base_fp16_gib": round(_gib(int(meta["total_params"]) * 2 / 1e9), 2),
        "batch_size": int(rows_all[0]["batch_size"]),
        "lora_rank": int(rows_all[0]["lora_rank"]),
        "gpu_total_gb": round(total_gb, 1),
        "gpu_total_gib": round(_gib(total_gb), 1),
        "arms": {},
    }

    peaks: dict[int, float] = {}
    p50s: list[float] = []
    for conf in sorted({r["alloc_conf"] for r in rows_all}):
        rows = [r for r in rows_all if r["alloc_conf"] == conf]
        ok = sorted(int(r["num_adapters"]) for r in rows if r.get("status", "ok") == "ok")
        oom = sorted(int(r["num_adapters"]) for r in rows if r.get("status") == "oom")
        ceiling = max(ok)
        by_n = {int(r["num_adapters"]): r for r in rows if r.get("status", "ok") == "ok"}
        peak = {n: round(num(by_n[n]["peak_gpu_mem_gb"]), 3) for n in ok}
        peaks.update(peak)
        p50 = {n: round(num(by_n[n]["p50_ms"]), 2) for n in ok}
        p50s += list(p50.values())
        out["arms"][conf] = {
            # The claim is only as good as the bracket: the arm must have an
            # OOM strictly above its best success, or "ceiling" just means
            # "where we stopped asking".
            "bracketed": bool(oom) and min(oom) > ceiling,
            "ceiling_n": ceiling,
            "n_ok": ok,
            "n_oom": oom,
            "p50_ms": {str(n): v for n, v in p50.items()},
            "peak_gpu_mem_gb": {str(n): v for n, v in peak.items()},
            "peak_at_ceiling_gb": peak[ceiling],
            "peak_at_ceiling_gib": round(_gib(peak[ceiling]), 1),
            "store_gb_at_ceiling": round(b8 * ceiling / 1e9, 1),
            "store_gib_at_ceiling": round(b8 * ceiling / 1024**3, 1),
        }

    # Latency must not move as the pool fills; if it does, the ceiling is not
    # the only thing the allocator setting changed.
    out["p50_range_ms"] = [min(p50s), max(p50s)]
    out["p50_spread_pct"] = round(100.0 * (max(p50s) - min(p50s)) / min(p50s), 2)

    # Peak memory vs N is a straight line; the slope is the real per-adapter
    # cost (analytic weights plus allocator overhead) and the intercept is the
    # tenant-independent base. Both arms share the same physics, so the fit
    # runs over the union of measured pool sizes.
    ns = sorted(peaks)
    slope, intercept = statistics.linear_regression(ns, [peaks[n] * 1e9 for n in ns])
    resid = [abs(peaks[n] * 1e9 - (slope * n + intercept)) for n in ns]
    out["fit_n_points"] = len(ns)
    out["fit_bytes_per_adapter"] = round(slope)
    out["fit_fixed_bytes"] = round(intercept)
    out["fit_fixed_overhead_gb"] = round(
        (intercept - int(meta["total_params"]) * 2) / 1e9, 3)
    out["fit_overhead_bytes_per_adapter"] = round(slope) - b8
    out["fit_overhead_pct_per_adapter"] = round(100.0 * (slope - b8) / b8, 2)
    out["fit_max_abs_resid_mb"] = round(max(resid) / 1e6, 2)
    return out


def analyse_breakdown() -> dict:
    d = read_json(RES / "forward_breakdown.json")
    return {
        "ablation_lora_on_ms": round(d["ablation"]["lora_on_mean_ms"], 1),
        "ablation_lora_off_ms": round(d["ablation"]["lora_off_mean_ms"], 1),
        "ablation_lora_cost_ms": round(d["ablation"]["lora_cost_mean_ms"], 1),
        "ablation_lora_share_pct": round(d["ablation"]["lora_cost_share_pct"], 1),
        "profiler_base_linear_pct": round(d["profiler_categories"]["base_linear"]["share_pct"], 1),
        "profiler_lora_bmm_self_pct": round(d["bmm_isolation"]["lora_bmm_self_share_pct"], 2),
        "flop_ratio_base_to_lora": int(d["flop_ratio"]["base_to_lora_ratio"]),
        "max_recoverable_flop_pct": round(100.0 / d["flop_ratio"]["base_to_lora_ratio"], 2),
        "batch_size": d["config"]["batch_size"],
        "num_adapters": d["config"]["num_adapters"],
    }


# ------------------------------------------------------------ aggregation


def main() -> None:
    reg: dict = {"configs": {}, "assembly": {}, "breakdown": analyse_breakdown()}
    if (CHURN_DIR / "churn_registration_paths.csv").exists():
        reg["churn"] = analyse_churn()
    if ALLOC_PROBE["sweep"].exists():
        reg["capacity_probe"] = {ALLOC_PROBE["key"]: analyse_capacity_probe(ALLOC_PROBE)}
    for key, cfg in CONFIGS.items():
        reg["configs"][key] = analyse_config(cfg)

    for tag, fname in [
        ("minilm", "assembly_bench_minilm.json"),
        ("bgem3", "assembly_bench.json"),
    ]:
        p = RES / "rebuttal_assembly" / fname
        if p.exists():
            reg["assembly"][tag] = analyse_assembly(p)

    # Written by benchmarks/run_rebuttal_mixed_rank.sh. The key is omitted
    # entirely until that pod session runs, rather than emitted empty: this file
    # is committed, and a regenerate that has no new measurements to report
    # should leave it byte-identical.
    mixed_rank = {}
    for tag, fname in [
        ("bgem3_4_16", "mixed_rank_4_16.json"),
        ("bgem3_spread", "mixed_rank_spread.json"),
        ("bgem3_4_16_cpuasm", "mixed_rank_4_16_cpuasm.json"),
    ]:
        p = RES / "rebuttal_mixed_rank" / fname
        if p.exists():
            mixed_rank[tag] = analyse_mixed_rank(p)
    if mixed_rank:
        reg["mixed_rank"] = mixed_rank

    # ---- cross-config roll-ups, the phrasings the rebuttal actually uses
    new = ["electra", "deberta", "xlmr_xl", "l40s"]
    cs = reg["configs"]
    roll = {
        "spread_pct_min_over_new": min(cs[k]["spread_pct_worst"] for k in new),
        "spread_pct_max_over_new": max(cs[k]["spread_pct_worst"] for k in new),
        "at_ceiling_max_over_new": max(cs[k]["at_ceiling_pct_worst"] for k in new),
        "at_ceiling_min_over_new": min(cs[k]["at_ceiling_pct_worst"] for k in new),
        "at_ceiling_max_all": max(cs[k]["at_ceiling_pct_worst"] for k in cs),
        "speedup_min_over_new": min(cs[k]["speedup_min"] for k in new),
        "speedup_max_over_new": max(cs[k]["speedup_max"] for k in new),
        "speedup_max_a100": max(cs[k]["speedup_max"] for k in ["electra", "deberta", "xlmr_xl"]),
        "params_min_m": min(cs[k]["total_params_m"] for k in new),
        "params_max_m": max(cs[k]["total_params_m"] for k in new),
        "rank_spread_max_over_new": max(cs[k]["rank_spread_pct"] for k in new),
        "flop_ratios": {k: cs[k]["flop_ratio_3d_over_r"] for k in ["electra", "deberta", "xlmr_xl"]},
        "flop_recoverable_pct": {
            k: cs[k]["max_recoverable_flop_pct"] for k in ["electra", "deberta", "xlmr_xl"]
        },
    }
    reg["rollup"] = roll

    # ---- derived quantities the prose states explicitly, computed here so
    #      they are checkable rather than done in a reviewer's head.
    b = cs["bgem3_a100"]
    ours_ms = b["register_ms_per_adapter_at_n1000"]
    pool = 1000
    reg["derived"] = {
        # entire 1,000-adapter pool replaced once an hour, our side
        "churn_pool_size": pool,
        "churn_period_s": 3600,
        "churn_refill_s": round(ours_ms * pool / 1000.0, 2),
        "churn_share_of_wallclock_pct": round(100.0 * ours_ms * pool / 1000.0 / 3600.0, 3),
        # PEFT's marginal add at N=1,000 vs ours, same pool
        "peft_marginal_add_s": b["peft_marginal_add_s_at_top"],
        "peft_vs_ours_add_ratio": b["peft_vs_ours_add_ratio"],
        # delta-path arithmetic as a share of one projection: 2r/d
        "delta_share_of_projection_pct_r8": round(100.0 * 2 * 8 / b["hidden_size"], 2),
        "delta_share_of_projection_pct_r32": round(100.0 * 2 * 32 / b["hidden_size"], 2),
        # halving the tenant ceiling per doubling of r (Finding 4)
        "ceiling_r8": b["ceiling_n"],
        "ceiling_r16": b["ceiling_n"] // 2,
        "ceiling_r32": b["ceiling_n"] // 4,
        # adapter store as a share of the A100's 80 GB at the ceiling
        "store_gb_at_ceiling": b["store_gb_at_ceiling"],
        "store_gib_at_ceiling": b["store_gib_at_ceiling"],
        # The at-ceiling column is anchored at N=1,000, so the multiplier the
        # "flat p50" claim has to survive is ceiling/1000, not ceiling/100.
        "pool_growth_vs_n1000_max": round(max(c["ceiling_n"] for c in cs.values()) / 1000),
        "pool_growth_vs_n100_max": round(max(c["ceiling_n"] for c in cs.values()) / 100),
    }

    OUT_JSON.write_text(json.dumps(reg, indent=2, sort_keys=True) + "\n")

    # ---- fact sheet
    L = ["# Rebuttal fact sheet (generated — do not edit by hand)", ""]
    L.append("Regenerate with `python rebuttal/make_numbers.py`. Every number quoted in")
    L.append("the rebuttal must trace to a line here.")
    L.append("")
    L.append("## Per-configuration sweep results")
    L.append("")
    L.append(
        "| Config | Params (L, d) | Spread across sweep | At ceiling vs N=1,000 "
        "| Speedup vs PEFT-mixed | Ceiling @ r=8 | MB/adapter | Peak mem @ ceiling |"
    )
    L.append("|---|---|---|---|---|---|---|---|")
    for k, c in cs.items():
        L.append(
            f"| {c['label']} | {c['total_params_m']}M ({c['num_layers']}, {c['hidden_size']}) "
            f"| {c['spread_pct_worst']}% (B={c['spread_pct_worst_batch']}) "
            f"| {c['at_ceiling_vs_n1000_pct']:+.2f}% "
            f"| {c['speedup_min']}–{c['speedup_max']}× "
            f"| {c['ceiling_n']:,} | {c['mb_per_adapter_r8']} | {c['ceiling_peak_mem_gb']} GB |"
        )
    L.append("")
    for k, c in cs.items():
        L.append(f"### {k} — {c['label']}")
        L.append(f"- targets: `{c['targets']}`, sweep N={c['sweep_n_values']}, B={c['sweep_batch_sizes']}")
        L.append(f"- rows: {c['sweep_rows']} (5 seeds x N x B, plus rank cells r={c['sweep_ranks']})")
        L.append(f"- spread by batch: {c['spread_pct_by_batch']}")
        L.append(
            f"- ceiling {c['ceiling_n']:,}: p50 {c['ceiling_p50_ms_b32']} ms vs "
            f"{c['n1000_p50_ms_b32']} ms at N=1,000 -> {c['at_ceiling_vs_n1000_pct']:+.2f}%"
        )
        L.append(f"- rank cells (N=1000, B=32): {c.get('rank_p50_ms')} -> spread {c.get('rank_spread_pct')}%")
        L.append(f"- speedup cells (throughput ratio, paper convention): {c['speedup_cells']}")
        L.append(f"- speedup cells (p50-latency ratio, for reference): {c['speedup_latency_cells']}")
        L.append(
            f"- worst between-seed s.d.: {c['seed_sd_pct_worst']}% at "
            f"N={c['seed_sd_pct_worst_cell']['n']}, B={c['seed_sd_pct_worst_cell']['b']}"
        )
        L.append(
            f"- peak mem vs batch at N={c['mem_vs_batch_n']:,}: {c['mem_vs_batch_gb']} "
            f"(delta {c['mem_vs_batch_delta_gb']} GB)"
        )
        L.append(f"- at ceiling by batch: {c.get('at_ceiling_pct_by_batch')} (worst {c.get('at_ceiling_pct_worst')}% at B={c.get('at_ceiling_worst_batch')})")
        L.append(f"- MB/adapter r=8: {c['mb_per_adapter_r8']} MB = {c['mib_per_adapter_r8']} MiB ({c['params_per_adapter_r8']:,} params)")
        L.append(f"- store at ceiling: {c['store_gb_at_ceiling']} GB; 3d/r = {c['flop_ratio_3d_over_r']}")
        L.append(f"- registration: {c.get('register_ms_per_adapter_at_n1000')} ms/adapter at N=1,000")
        L.append(f"- PEFT add_adapter totals: {c.get('peft_add_total_s')} s")
        L.append("")

    L.append("## Roll-ups used in prose")
    L.append("")
    for k, v in roll.items():
        L.append(f"- `{k}`: {v}")
    L.append("")

    L.append("## Assembly benchmark (re-measured at the house 50/200 protocol)")
    L.append("")
    for tag, a in reg["assembly"].items():
        L.append(f"### {tag} — {a['model']}")
        L.append(f"- {a['gpu']}, {a['cpu']} (cgroup quota {a['cpu_cgroup_quota']})")
        L.append(
            f"- N={a['num_adapters']}, warmup {a['warmup']} / iters {a['iters']}, "
            f"{a['seeds']} seeds, B={a['batch_sizes']}"
        )
        L.append(f"- baseline single-stream throughput: {a['baseline_tput_by_batch']}")
        L.append(f"- index_select throughput: {a['indexsel_tput_by_batch']}")
        L.append(f"- end-to-end speedup: {a['speedup_by_batch']} (range {a['speedup_min']}–{a['speedup_max']}x)")
        L.append(f"- assembly share, baseline: {a['baseline_assemble_share_pct']}")
        L.append(f"- assembly share, index_select: {a['indexsel_assemble_share_pct']}")
        L.append(f"- tail p99/p50, baseline: {a['baseline_tail_ratio']}")
        L.append(f"- tail p99/p50, index_select: {a['indexsel_tail_ratio']}")
        if "baseline_scatter_share_pct" in a:
            L.append(f"- result-scatter share, baseline: {a['baseline_scatter_share_pct']}")
            L.append(f"- result-scatter share, index_select: {a['indexsel_scatter_share_pct']}")
            L.append(f"- result-scatter time (ms), baseline: {a['baseline_scatter_ms']}")
        L.append("")

    if reg.get("mixed_rank"):
        L.append("## Mixed-rank serving (yeZ9 Q5 / W1)")
        L.append("")
        L.append(
            "Padding tax = mixed batch vs a uniform batch at the fleet's LOWEST "
            "rank (what a low-rank tenant pays for sharing a batch). Overhead vs "
            "r_max = cost beyond the padded shape; ~0 means padding adds nothing "
            "the uniform max-rank sweep did not already measure."
        )
        L.append("")
        for tag, a in reg["mixed_rank"].items():
            L.append(f"### {tag} — {a['model']}")
            L.append(f"- {a['gpu']}, B={a['batch_size']}, assembler={a['assembler']}")
            L.append(
                f"- N={a['adapters']}, warmup {a['warmup']} / iters {a['iters']}, "
                f"{a['seeds']} seeds, mixes={a['mixes']}"
            )
            L.append(
                f"- padding exactness: max|delta logit| = "
                f"{a['exactness_max_abs_delta']:.2e} (asserted before timing)"
            )
            for label, entry in a["by_mix"].items():
                L.append(f"- mix {label}:")
                if "p50_ms_by_n" in entry:
                    L.append(f"    - p50 by N: {entry['p50_ms_by_n']}")
                    L.append(f"    - spread across N: {entry['spread_pct_over_n']}%")
                for n, c in entry["by_n"].items():
                    L.append(
                        f"    - N={n}: padding tax {c['padding_tax_pct']:+}% "
                        f"(forward {c['padding_tax_forward_pct']:+}%), "
                        f"vs uniform r_max {c['overhead_vs_rmax_pct']:+}%, "
                        f"uniform r_min->r_max {c['uniform_rank_delta_pct']:+}%, "
                        f"native store {c['native_store_vs_padded']}x, "
                        f"batches at fleet max {c['batches_at_fleet_max_pct']}%"
                    )
            if "padding_tax_pct_max" in a:
                L.append(
                    f"- worst padding tax: {a['padding_tax_pct_max']:+}% "
                    f"(best {a['padding_tax_pct_min']:+}%); worst forward-only "
                    f"{a['padding_tax_forward_pct_max']:+}%"
                )
                L.append(
                    f"- overhead vs uniform r_max: "
                    f"{a['overhead_vs_rmax_pct_min']:+}% to "
                    f"{a['overhead_vs_rmax_pct_max']:+}%"
                )
                L.append(
                    f"- native store vs pre-padded: "
                    f"{a['native_store_vs_padded_min']}x; batches at fleet max "
                    f"rank >= {a['batches_at_fleet_max_pct_min']}%"
                )
            if "spread_pct_over_n_max" in a:
                L.append(
                    f"- worst mixed-fleet spread across N: "
                    f"{a['spread_pct_over_n_max']}%"
                )
            if "oom_cells" in a:
                L.append(f"- WARNING: {a['oom_cells']} cell(s) OOM'd and are missing")
            L.append("")

    L.append("## LoRA-path ablation (Finding 6)")
    L.append("")
    for k, v in reg["breakdown"].items():
        L.append(f"- `{k}`: {v}")
    L.append("")

    OUT_MD.write_text("\n".join(L) + "\n")
    print(f"wrote {OUT_JSON}")
    print(f"wrote {OUT_MD}")


if __name__ == "__main__":
    main()
