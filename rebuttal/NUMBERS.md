# Rebuttal fact sheet (generated — do not edit by hand)

Regenerate with `python rebuttal/make_numbers.py`. Every number quoted in
the rebuttal must trace to a line here.

## Per-configuration sweep results

| Config | Params (L, d) | Spread across sweep | At ceiling vs N=1,000 | Speedup vs PEFT-mixed | Ceiling @ r=8 | MB/adapter | Peak mem @ ceiling |
|---|---|---|---|---|---|---|---|
| bge-m3 / A100-80GB (paper) | 567.8M (24, 1024) | 3.31% (B=128) | -0.37% | 5.6–21.2× | 47,000 | 1.57 | 76.2 GB |
| ELECTRA-large / A100-80GB | 334.1M (24, 1024) | 1.36% (B=8) | -0.16% | 6.2–20.9× | 51,000 | 1.57 | 82.1 GB |
| DeBERTa-v2-xlarge / A100-80GB | 884.6M (24, 1536) | 1.16% (B=8) | +0.43% | 2.4–7.1× | 64,000 | 1.18 | 79.9 GB |
| XLM-RoBERTa-XL / A100-80GB | 3482.5M (36, 2560) | 0.7% (B=128) | -0.38% | 2.3–19.5× | 12,000 | 5.9 | 78.6 GB |
| bge-m3 / L40S-48GB | 567.8M (24, 1024) | 1.6% (B=8) | +0.20% | 3.1–32.7× | 28,000 | 1.57 | 45.9 GB |

### bgem3_a100 — bge-m3 / A100-80GB (paper)
- targets: `query+value`, sweep N=[100, 1000, 5000, 10000, 20000, 40000, 47000], B=[8, 16, 32, 64, 128]
- rows: 190 (5 seeds x N x B, plus rank cells r=[4, 8, 16, 32])
- spread by batch: {'8': 0.48, '16': 0.87, '32': 0.86, '64': 1.82, '128': 3.31}
- ceiling 47,000: p50 39.95 ms vs 40.1 ms at N=1,000 -> -0.37%
- rank cells (N=1000, B=32): {'4': 40.41, '8': 40.1, '16': 40.13, '32': 39.93} -> spread 1.19%
- speedup cells (throughput ratio, paper convention): {'N100_B8': 6.38, 'N100_B32': 8.92, 'N100_B128': 5.65, 'N1000_B8': 21.21, 'N1000_B32': 18.64, 'N1000_B128': 11.35}
- speedup cells (p50-latency ratio, for reference): {'N100_B8': 6.34, 'N100_B32': 9.25, 'N100_B128': 6.14, 'N1000_B8': 21.55, 'N1000_B32': 19.0, 'N1000_B128': 12.41}
- worst between-seed s.d.: 2.2% at N=100, B=128
- peak mem vs batch at N=47,000: {'8': 76.1, '16': 76.1, '32': 76.2, '64': 76.3, '128': 76.6} (delta 0.5 GB)
- at ceiling by batch: {'8': -0.05, '16': -0.57, '32': -0.37, '64': 0.37, '128': 0.65} (worst 0.65% at B=128)
- MB/adapter r=8: 1.57 MB = 1.5 MiB (786,432 params)
- store at ceiling: 73.9 GB; 3d/r = 384
- registration: 0.29 ms/adapter at N=1,000
- PEFT add_adapter totals: {'100': 14.2, '1000': 1229.2} s

### electra — ELECTRA-large / A100-80GB
- targets: `query+value`, sweep N=[100, 1000, 5000, 10000, 20000, 47000], B=[8, 16, 32, 64, 128]
- rows: 165 (5 seeds x N x B, plus rank cells r=[4, 8, 16, 32])
- spread by batch: {'8': 1.36, '16': 1.06, '32': 1.18, '64': 1.21, '128': 0.77}
- ceiling 51,000: p50 36.95 ms vs 37.01 ms at N=1,000 -> -0.16%
- rank cells (N=1000, B=32): {'4': 37.28, '8': 37.01, '16': 37.01, '32': 37.27} -> spread 0.73%
- speedup cells (throughput ratio, paper convention): {'N100_B8': 6.21, 'N100_B32': 10.2, 'N100_B128': 6.59, 'N1000_B8': 20.49, 'N1000_B32': 20.88, 'N1000_B128': 13.19}
- speedup cells (p50-latency ratio, for reference): {'N100_B8': 6.3, 'N100_B32': 10.62, 'N100_B128': 7.18, 'N1000_B8': 20.8, 'N1000_B32': 21.41, 'N1000_B128': 14.22}
- worst between-seed s.d.: 2.87% at N=47000, B=16
- peak mem vs batch at N=47,000: {'8': 75.6, '16': 75.7, '32': 75.7, '64': 75.9, '128': 76.1} (delta 0.5 GB)
- at ceiling by batch: {'32': -0.16} (worst -0.16% at B=32)
- MB/adapter r=8: 1.57 MB = 1.5 MiB (786,432 params)
- store at ceiling: 80.2 GB; 3d/r = 384
- registration: 0.32 ms/adapter at N=1,000
- PEFT add_adapter totals: {'100': 15.6, '1000': 1208.5} s

### deberta — DeBERTa-v2-xlarge / A100-80GB
- targets: `value`, sweep N=[100, 1000, 5000, 10000, 20000, 40000, 60000], B=[8, 16, 32, 64, 128]
- rows: 190 (5 seeds x N x B, plus rank cells r=[4, 8, 16, 32])
- spread by batch: {'8': 1.16, '16': 0.35, '32': 0.43, '64': 0.47, '128': 0.29}
- ceiling 64,000: p50 76.09 ms vs 75.76 ms at N=1,000 -> +0.43%
- rank cells (N=1000, B=32): {'4': 75.92, '8': 75.76, '16': 75.84, '32': 76.08} -> spread 0.41%
- speedup cells (throughput ratio, paper convention): {'N100_B8': 2.71, 'N100_B32': 3.33, 'N100_B128': 2.37, 'N1000_B8': 7.12, 'N1000_B32': 5.77, 'N1000_B128': 3.91}
- speedup cells (p50-latency ratio, for reference): {'N100_B8': 2.72, 'N100_B32': 3.34, 'N100_B128': 2.41, 'N1000_B8': 7.15, 'N1000_B32': 5.82, 'N1000_B128': 3.96}
- worst between-seed s.d.: 2.26% at N=5000, B=8
- peak mem vs batch at N=60,000: {'8': 74.6, '16': 74.8, '32': 75.1, '64': 75.8, '128': 77.1} (delta 2.5 GB)
- at ceiling by batch: {'32': 0.43} (worst 0.43% at B=32)
- MB/adapter r=8: 1.18 MB = 1.12 MiB (589,824 params)
- store at ceiling: 75.5 GB; 3d/r = 576
- registration: 0.3 ms/adapter at N=1,000
- PEFT add_adapter totals: {'100': 7.7, '1000': 564.2} s

### xlmr_xl — XLM-RoBERTa-XL / A100-80GB
- targets: `query+value`, sweep N=[100, 1000, 2000, 4000, 6000, 8000, 11000], B=[8, 16, 32, 64, 128]
- rows: 190 (5 seeds x N x B, plus rank cells r=[4, 8, 16, 32])
- spread by batch: {'8': 0.47, '16': 0.41, '32': 0.31, '64': 0.31, '128': 0.7}
- ceiling 12,000: p50 122.17 ms vs 122.64 ms at N=1,000 -> -0.38%
- rank cells (N=1000, B=32): {'4': 123.35, '8': 122.64, '16': 122.95, '32': 123.46} -> spread 0.67%
- speedup cells (throughput ratio, paper convention): {'N100_B8': 4.53, 'N100_B32': 3.5, 'N100_B128': 2.28, 'N1000_B8': 19.52, 'N1000_B32': 8.14, 'N1000_B128': 4.18}
- speedup cells (p50-latency ratio, for reference): {'N100_B8': 4.61, 'N100_B32': 3.55, 'N100_B128': 2.34, 'N1000_B8': 19.7, 'N1000_B32': 8.18, 'N1000_B128': 4.3}
- worst between-seed s.d.: 0.59% at N=100, B=8
- peak mem vs batch at N=11,000: {'8': 72.5, '16': 72.6, '32': 72.7, '64': 73.0, '128': 73.5} (delta 1.0 GB)
- at ceiling by batch: {'32': -0.38} (worst -0.38% at B=32)
- MB/adapter r=8: 5.9 MB = 5.62 MiB (2,949,120 params)
- store at ceiling: 70.8 GB; 3d/r = 960
- registration: 0.23 ms/adapter at N=1,000
- PEFT add_adapter totals: {'100': 21.3, '1000': 1742.5} s

### l40s — bge-m3 / L40S-48GB
- targets: `query+value`, sweep N=[100, 1000, 5000, 10000, 20000, 28000], B=[8, 16, 32, 64, 128]
- rows: 165 (5 seeds x N x B, plus rank cells r=[4, 8, 16, 32])
- spread by batch: {'8': 1.6, '16': 1.05, '32': 0.84, '64': 0.99, '128': 0.23}
- ceiling 28,000: p50 39.3 ms vs 39.22 ms at N=1,000 -> +0.20%
- rank cells (N=1000, B=32): {'4': 39.22, '8': 39.22, '16': 39.22, '32': 39.25} -> spread 0.07%
- speedup cells (throughput ratio, paper convention): {'N100_B8': 7.67, 'N100_B32': 5.53, 'N100_B128': 3.06, 'N1000_B8': 32.67, 'N1000_B32': 13.09, 'N1000_B128': 6.21}
- speedup cells (p50-latency ratio, for reference): {'N100_B8': 7.83, 'N100_B32': 5.65, 'N100_B128': 3.26, 'N1000_B8': 33.05, 'N1000_B32': 13.31, 'N1000_B128': 6.62}
- worst between-seed s.d.: 3.41% at N=100, B=16
- peak mem vs batch at N=28,000: {'8': 45.8, '16': 45.8, '32': 45.9, '64': 46.0, '128': 46.3} (delta 0.5 GB)
- at ceiling by batch: {'8': 1.29, '16': -0.61, '32': 0.2, '64': 0.11, '128': -0.09} (worst 1.29% at B=8)
- MB/adapter r=8: 1.57 MB = 1.5 MiB (786,432 params)
- store at ceiling: 44.0 GB; 3d/r = 384
- registration: 0.21 ms/adapter at N=1,000
- PEFT add_adapter totals: {'100': 10.6, '1000': 918.8} s

## Roll-ups used in prose

- `spread_pct_min_over_new`: 0.7
- `spread_pct_max_over_new`: 1.6
- `at_ceiling_max_over_new`: 1.29
- `at_ceiling_min_over_new`: -0.38
- `at_ceiling_max_all`: 1.29
- `speedup_min_over_new`: 2.3
- `speedup_max_over_new`: 32.7
- `speedup_max_a100`: 20.9
- `params_min_m`: 334.1
- `params_max_m`: 3482.5
- `rank_spread_max_over_new`: 0.73
- `flop_ratios`: {'electra': 384, 'deberta': 576, 'xlmr_xl': 960}
- `flop_recoverable_pct`: {'electra': 0.26, 'deberta': 0.17, 'xlmr_xl': 0.1}

## Assembly benchmark (re-measured at the house 50/200 protocol)

### minilm — sentence-transformers/all-MiniLM-L6-v2
- NVIDIA A100-SXM4-80GB, AMD EPYC 7742 64-Core Processor (cgroup quota 27.2)
- N=2000, warmup 50 / iters 200, 5 seeds, B=[8, 16, 32, 64, 128, 256]
- baseline single-stream throughput: {'8': 1463, '16': 2445, '32': 4051, '64': 4925, '128': 5192, '256': 4912}
- index_select throughput: {'8': 1647, '16': 3156, '32': 6713, '64': 12152, '128': 14255, '256': 15315}
- end-to-end speedup: {'8': 1.13, '16': 1.29, '32': 1.66, '64': 2.47, '128': 2.75, '256': 3.12} (range 1.13–3.12x)
- assembly share, baseline: {'8': 15.3, '16': 24.8, '32': 40.7, '64': 58.0, '128': 61.6, '256': 65.3}
- assembly share, index_select: {'8': 4.1, '16': 3.9, '32': 4.6, '64': 4.4, '128': 3.1, '256': 1.8}
- tail p99/p50, baseline: {'8': 1.2, '16': 1.21, '32': 1.17, '64': 1.24, '128': 1.48, '256': 5.28}
- tail p99/p50, index_select: {'8': 1.13, '16': 1.17, '32': 1.13, '64': 1.11, '128': 1.06, '256': 1.03}
- result-scatter share, baseline: {'8': 2.7, '16': 4.0, '32': 6.1, '64': 7.2, '128': 7.5, '256': 7.1}
- result-scatter share, index_select: {'8': 3.0, '16': 5.1, '32': 9.6, '64': 16.1, '128': 18.1, '256': 18.8}
- result-scatter time (ms), baseline: {'8': 0.15, '16': 0.27, '32': 0.51, '64': 1.0, '128': 2.0, '256': 3.97}

### bgem3 — BAAI/bge-m3
- NVIDIA A100-SXM4-80GB, AMD EPYC 7742 64-Core Processor (cgroup quota 27.2)
- N=2000, warmup 50 / iters 200, 5 seeds, B=[8, 16, 32, 64, 128]
- baseline single-stream throughput: {'8': 374, '16': 596, '32': 776, '64': 766, '128': 768}
- index_select throughput: {'8': 427, '16': 838, '32': 1202, '64': 1322, '128': 1379}
- end-to-end speedup: {'8': 1.14, '16': 1.41, '32': 1.55, '64': 1.73, '128': 1.8} (range 1.14–1.8x)
- assembly share, baseline: {'8': 14.8, '16': 29.8, '32': 36.8, '64': 42.8, '128': 45.0}
- assembly share, index_select: {'8': 2.2, '16': 2.2, '32': 1.8, '64': 1.0, '128': 0.5}
- tail p99/p50, baseline: {'8': 1.12, '16': 1.14, '32': 1.18, '64': 4.29, '128': 2.76}
- tail p99/p50, index_select: {'8': 1.08, '16': 1.1, '32': 1.02, '64': 1.02, '128': 1.01}
- result-scatter share, baseline: {'8': 0.7, '16': 1.0, '32': 1.3, '64': 1.3, '128': 1.3}
- result-scatter share, index_select: {'8': 0.8, '16': 1.4, '32': 2.0, '64': 2.1, '128': 2.1}
- result-scatter time (ms), baseline: {'8': 0.15, '16': 0.27, '32': 0.53, '64': 1.08, '128': 2.12}

## Mixed-rank serving (yeZ9 Q5 / W1)

Padding tax = mixed batch vs a uniform batch at the fleet's LOWEST rank (what a low-rank tenant pays for sharing a batch). Overhead vs r_max = cost beyond the padded shape; ~0 means padding adds nothing the uniform max-rank sweep did not already measure.

### bgem3_4_16 — BAAI/bge-m3
- NVIDIA A100-SXM4-80GB, B=32, assembler=indexsel
- N=[100, 1000, 5000, 10000, 20000], warmup 50 / iters 200, 5 seeds, mixes=['4+16']
- padding exactness: max|delta logit| = 0.00e+00 (asserted before timing)
- mix 4+16:
    - p50 by N: {'100': 26.8, '1000': 26.85, '5000': 26.88, '10000': 26.87, '20000': 26.77}
    - spread across N: 0.41%
    - N=100: padding tax -0.37% (forward -0.39%), vs uniform r_max +0.19%, uniform r_min->r_max -0.56%, native store 0.625x, batches at fleet max 100.0%
    - N=1000: padding tax -0.77% (forward -0.79%), vs uniform r_max +0.06%, uniform r_min->r_max -0.83%, native store 0.625x, batches at fleet max 100.0%
    - N=5000: padding tax -0.82% (forward -0.78%), vs uniform r_max -0.15%, uniform r_min->r_max -0.68%, native store 0.625x, batches at fleet max 100.0%
    - N=10000: padding tax -0.76% (forward -0.78%), vs uniform r_max -0.21%, uniform r_min->r_max -0.54%, native store 0.625x, batches at fleet max 100.0%
    - N=20000: padding tax -0.79% (forward -0.83%), vs uniform r_max -0.21%, uniform r_min->r_max -0.58%, native store 0.625x, batches at fleet max 100.0%
- worst padding tax: -0.37% (best -0.82%); worst forward-only -0.39%
- overhead vs uniform r_max: -0.21% to +0.19%
- native store vs pre-padded: 0.625x; batches at fleet max rank >= 100.0%
- worst mixed-fleet spread across N: 0.41%

### bgem3_spread — BAAI/bge-m3
- NVIDIA A100-SXM4-80GB, B=32, assembler=indexsel
- N=[100, 1000, 5000, 10000], warmup 50 / iters 200, 5 seeds, mixes=['4+8+16+32']
- padding exactness: max|delta logit| = 0.00e+00 (asserted before timing)
- mix 4+8+16+32:
    - p50 by N: {'100': 27.08, '1000': 27.3, '5000': 27.22, '10000': 27.18}
    - spread across N: 0.78%
    - N=100: padding tax +0.59% (forward +0.61%), vs uniform r_max +0.14%, uniform r_min->r_max +0.44%, native store 0.469x, batches at fleet max 100.0%
    - N=1000: padding tax +0.68% (forward +0.71%), vs uniform r_max +0.21%, uniform r_min->r_max +0.46%, native store 0.469x, batches at fleet max 100.0%
    - N=5000: padding tax +0.65% (forward +0.6%), vs uniform r_max +0.21%, uniform r_min->r_max +0.44%, native store 0.469x, batches at fleet max 100.0%
    - N=10000: padding tax +0.39% (forward +0.29%), vs uniform r_max +0.06%, uniform r_min->r_max +0.33%, native store 0.469x, batches at fleet max 100.0%
- worst padding tax: +0.68% (best +0.39%); worst forward-only +0.71%
- overhead vs uniform r_max: +0.06% to +0.21%
- native store vs pre-padded: 0.469x; batches at fleet max rank >= 100.0%
- worst mixed-fleet spread across N: 0.78%

### bgem3_4_16_cpuasm — BAAI/bge-m3
- NVIDIA A100-SXM4-80GB, B=32, assembler=baseline
- N=[1000], warmup 50 / iters 200, 5 seeds, mixes=['4+16']
- padding exactness: max|delta logit| = 0.00e+00 (asserted before timing)
- mix 4+16:
    - N=1000: padding tax -1.96% (forward -0.57%), vs uniform r_max +0.56%, uniform r_min->r_max -2.51%, native store 0.625x, batches at fleet max 100.0%
- worst padding tax: -1.96% (best -1.96%); worst forward-only -0.57%
- overhead vs uniform r_max: +0.56% to +0.56%
- native store vs pre-padded: 0.625x; batches at fleet max rank >= 100.0%

## LoRA-path ablation (Finding 6)

- `ablation_lora_on_ms`: 26.4
- `ablation_lora_off_ms`: 24.0
- `ablation_lora_cost_ms`: 2.4
- `ablation_lora_share_pct`: 9.0
- `profiler_base_linear_pct`: 50.6
- `profiler_lora_bmm_self_pct`: 4.94
- `flop_ratio_base_to_lora`: 384
- `max_recoverable_flop_pct`: 0.26
- `batch_size`: 32
- `num_adapters`: 1000

