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


def read_csv(path: Path) -> list[dict]:
    with open(path) as fh:
        return list(csv.DictReader(fh))


def read_json(path: Path) -> dict:
    with open(path) as fh:
        return json.load(fh)


def num(x):
    return float(x) if x not in ("", None) else None


# Each config: sweep CSV, capacity CSV(s), PEFT-mixed CSV, metadata CSV.
CONFIGS = {
    "bgem3_a100": dict(
        label="bge-m3 / A100-80GB (paper)",
        model="BAAI/bge-m3",
        sweep=RES / "sweep_main.csv",
        capacity=[RES / "sweep_capacity.csv"],
        peft_mixed=RES / "peft_mixed_sxm80.csv",
        meta=RES / "model_metadata.csv",
        targets="query+value",
    ),
    "electra": dict(
        label="ELECTRA-large / A100-80GB",
        model="google/electra-large-discriminator",
        sweep=RES / "rebuttal_electra/sweep_electra_a100.csv",
        capacity=[RES / "rebuttal_electra/sweep_electra_capacity.csv"],
        peft_mixed=RES / "rebuttal_electra/peft_mixed_electra.csv",
        meta=RES / "rebuttal_electra/model_metadata.csv",
        targets="query+value",
    ),
    "deberta": dict(
        label="DeBERTa-v2-xlarge / A100-80GB",
        model="microsoft/deberta-v2-xlarge",
        sweep=RES / "rebuttal_deberta/sweep_deberta_a100.csv",
        capacity=[
            RES / "rebuttal_deberta/sweep_deberta_capacity.csv",
            RES / "rebuttal_deberta/sweep_deberta_capacity2.csv",
        ],
        peft_mixed=RES / "rebuttal_deberta/peft_mixed_deberta.csv",
        meta=RES / "rebuttal_deberta/model_metadata.csv",
        targets="value",
    ),
    "xlmr_xl": dict(
        label="XLM-RoBERTa-XL / A100-80GB",
        model="facebook/xlm-roberta-xl",
        sweep=RES / "rebuttal_xlmr_xl/sweep_xlmrxl_a100.csv",
        capacity=[RES / "rebuttal_xlmr_xl/sweep_xlmrxl_capacity.csv"],
        peft_mixed=RES / "rebuttal_xlmr_xl/peft_mixed_xlmrxl.csv",
        meta=RES / "rebuttal_xlmr_xl/model_metadata.csv",
        targets="query+value",
    ),
    "l40s": dict(
        label="bge-m3 / L40S-48GB",
        model="BAAI/bge-m3",
        sweep=RES / "rebuttal_l40s/sweep_bgem3_l40s.csv",
        # the 28k cell OOM'd under the default allocator and was re-run
        # separately under PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
        capacity=[
            RES / "rebuttal_l40s/sweep_capacity_l40s.csv",
            RES / "rebuttal_l40s/sweep_bgem3_l40s_n28k_expseg.csv",
        ],
        peft_mixed=RES / "rebuttal_l40s/peft_mixed_l40s.csv",
        meta=RES / "rebuttal_l40s/model_metadata.csv",
        targets="query+value",
    ),
}

BASE_RANK = 8


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
    for p in cfg["capacity"]:
        if p.exists():
            cap_rows += read_csv(p)
    cap32 = [r for r in cap_rows if int(r["batch_size"]) == 32]
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
        b8 = int(meta["lora_bytes_fp16_r8"])
        out["bytes_per_adapter_r8"] = b8
        out["params_per_adapter_r8"] = int(meta["lora_params_r8"])
        # Decimal MB is what the draft quotes; MiB kept so the two never get
        # silently mixed (1.57 MB == 1.50 MiB for a 24x1024 query+value pair).
        out["mb_per_adapter_r8"] = round(b8 / 1e6, 2)
        out["mib_per_adapter_r8"] = round(b8 / 1024**2, 2)
        out["store_gb_at_ceiling"] = (
            round(b8 * out["ceiling_n"] / 1024**3, 1) if "ceiling_n" in out else None
        )
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
    for key, cfg in CONFIGS.items():
        reg["configs"][key] = analyse_config(cfg)

    for tag, fname in [
        ("minilm", "assembly_bench_minilm.json"),
        ("bgem3", "assembly_bench.json"),
    ]:
        p = RES / "rebuttal_assembly" / fname
        if p.exists():
            reg["assembly"][tag] = analyse_assembly(p)

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
