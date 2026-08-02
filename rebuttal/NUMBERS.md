# Rebuttal fact sheet (generated — do not edit by hand)

Regenerate with `python rebuttal/make_numbers.py`. Every number quoted in
the rebuttal must trace to a line here.

## Per-configuration sweep results

| Config | Params (L, d) | Spread across sweep | At ceiling vs N=1,000 | Speedup vs PEFT-mixed | Ceiling @ r=8 | MB/adapter | Peak mem @ ceiling |
|---|---|---|---|---|---|---|---|
| bge-m3 / A100-80GB (paper) | 567.8M (24, 1024) | 3.31% (B=128) | -0.37% | 5.6–21.2× | 47,000 | 1.57 | 76.2 GB |
| ELECTRA-large / A100-80GB | 334.1M (24, 1024) | 1.36% (B=8) | -0.16% | 6.2–20.9× | 51,000 | 1.57 | 82.1 GB |
| DeBERTa-v2-xlarge / A100-80GB | 884.6M (24, 1536) | 1.32% (B=8) | +0.55% | 2.4–7.3× | 55,000 | 1.18 | 69.0 GB |
| XLM-RoBERTa-XL / A100-80GB | 3482.5M (36, 2560) | 0.69% (B=16) | +0.22% | 2.9–19.9× | 12,000 | 5.9 | 79.8 GB |
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
- store at ceiling: 68.8 GB; 3d/r = 384
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
- store at ceiling: 74.7 GB; 3d/r = 384
- registration: 0.32 ms/adapter at N=1,000
- PEFT add_adapter totals: {'100': 15.6, '1000': 1208.5} s

### deberta — DeBERTa-v2-xlarge / A100-80GB
- targets: `value`, sweep N=[100, 1000, 5000, 10000, 20000, 40000], B=[8, 16, 32, 64, 128]
- rows: 165 (5 seeds x N x B, plus rank cells r=[4, 8, 16, 32])
- spread by batch: {'8': 1.32, '16': 0.65, '32': 0.55, '64': 0.48, '128': 0.34}
- ceiling 55,000: p50 75.8 ms vs 75.38 ms at N=1,000 -> +0.55%
- rank cells (N=1000, B=32): {'4': 75.64, '8': 75.38, '16': 75.42, '32': 75.61} -> spread 0.35%
- speedup cells (throughput ratio, paper convention): {'N100_B8': 2.72, 'N100_B32': 3.34, 'N100_B128': 2.42, 'N1000_B8': 7.26, 'N1000_B32': 5.88, 'N1000_B128': 3.98}
- speedup cells (p50-latency ratio, for reference): {'N100_B8': 2.75, 'N100_B32': 3.35, 'N100_B128': 2.46, 'N1000_B8': 7.25, 'N1000_B32': 5.9, 'N1000_B128': 4.04}
- worst between-seed s.d.: 0.36% at N=100, B=8
- peak mem vs batch at N=40,000: {'8': 50.4, '16': 50.5, '32': 50.9, '64': 51.5, '128': 52.9} (delta 2.5 GB)
- at ceiling by batch: {'32': 0.55} (worst 0.55% at B=32)
- MB/adapter r=8: 1.18 MB = 1.12 MiB (589,824 params)
- store at ceiling: 60.4 GB; 3d/r = 576
- registration: 0.24 ms/adapter at N=1,000
- PEFT add_adapter totals: {'100': 7.0, '1000': 566.0} s

### xlmr_xl — XLM-RoBERTa-XL / A100-80GB
- targets: `query+value`, sweep N=[100, 1000, 2000, 4000, 6000, 8000, 11000], B=[8, 16, 32, 64, 128]
- rows: 190 (5 seeds x N x B, plus rank cells r=[4, 8, 16, 32])
- spread by batch: {'8': 0.66, '16': 0.69, '32': 0.62, '64': 0.62, '128': 0.56}
- ceiling 12,000: p50 137.6 ms vs 137.29 ms at N=1,000 -> +0.22%
- rank cells (N=1000, B=32): {'4': 138.16, '8': 137.29, '16': 137.58, '32': 138.33} -> spread 0.75%
- speedup cells (throughput ratio, paper convention): {'N100_B8': 5.93, 'N100_B32': 4.86, 'N100_B128': 2.92, 'N1000_B8': 19.89, 'N1000_B32': 9.16, 'N1000_B128': 5.31}
- speedup cells (p50-latency ratio, for reference): {'N100_B8': 6.1, 'N100_B32': 4.94, 'N100_B128': 3.04, 'N1000_B8': 19.82, 'N1000_B32': 9.3, 'N1000_B128': 5.5}
- worst between-seed s.d.: 0.49% at N=8000, B=64
- peak mem vs batch at N=11,000: {'8': 73.5, '16': 73.6, '32': 73.7, '64': 74.0, '128': 74.5} (delta 1.0 GB)
- at ceiling by batch: {'32': 0.22} (worst 0.22% at B=32)
- MB/adapter r=8: 5.9 MB = 5.62 MiB (2,949,120 params)
- store at ceiling: 65.9 GB; 3d/r = 960
- registration: 0.28 ms/adapter at N=1,000
- PEFT add_adapter totals: {'100': 26.8, '1000': 1904.5} s

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
- store at ceiling: 41.0 GB; 3d/r = 384
- registration: 0.21 ms/adapter at N=1,000
- PEFT add_adapter totals: {'100': 10.6, '1000': 918.8} s

## Roll-ups used in prose

- `spread_pct_min_over_new`: 0.69
- `spread_pct_max_over_new`: 1.6
- `at_ceiling_max_over_new`: 1.29
- `at_ceiling_min_over_new`: -0.16
- `at_ceiling_max_all`: 1.29
- `speedup_min_over_new`: 2.4
- `speedup_max_over_new`: 32.7
- `speedup_max_a100`: 20.9
- `params_min_m`: 334.1
- `params_max_m`: 3482.5
- `rank_spread_max_over_new`: 0.75
- `flop_ratios`: {'electra': 384, 'deberta': 576, 'xlmr_xl': 960}
- `flop_recoverable_pct`: {'electra': 0.26, 'deberta': 0.17, 'xlmr_xl': 0.1}

## Assembly benchmark (re-measured at the house 50/200 protocol)

### minilm — sentence-transformers/all-MiniLM-L6-v2
- NVIDIA A100-SXM4-80GB, Intel(R) Xeon(R) Platinum 8470 (cgroup quota 20.4)
- N=2000, warmup 50 / iters 200, 5 seeds, B=[8, 16, 32, 64, 128, 256]
- baseline single-stream throughput: {'8': 2102, '16': 3535, '32': 5741, '64': 6309, '128': 6804, '256': 6500}
- index_select throughput: {'8': 2375, '16': 4585, '32': 9675, '64': 12479, '128': 14378, '256': 15342}
- end-to-end speedup: {'8': 1.13, '16': 1.3, '32': 1.69, '64': 1.98, '128': 2.11, '256': 2.36} (range 1.13–2.36x)
- assembly share, baseline: {'8': 15.1, '16': 24.1, '32': 39.1, '64': 50.4, '128': 53.2, '256': 57.9}
- assembly share, index_select: {'8': 4.9, '16': 4.8, '32': 5.8, '64': 4.1, '128': 2.9, '256': 1.8}
- tail p99/p50, baseline: {'8': 1.04, '16': 1.05, '32': 1.1, '64': 1.09, '128': 1.22, '256': 5.14}
- tail p99/p50, index_select: {'8': 1.06, '16': 1.07, '32': 1.04, '64': 1.02, '128': 1.01, '256': 1.01}
- result-scatter share, baseline: {'8': 2.9, '16': 4.2, '32': 6.2, '64': 6.6, '128': 7.0, '256': 6.5}
- result-scatter share, index_select: {'8': 3.2, '16': 5.4, '32': 9.9, '64': 12.0, '128': 13.8, '256': 13.9}
- result-scatter time (ms), baseline: {'8': 0.11, '16': 0.2, '32': 0.37, '64': 0.71, '128': 1.41, '256': 2.72}

### bgem3 — BAAI/bge-m3
- NVIDIA A100-SXM4-80GB, Intel(R) Xeon(R) Platinum 8470 (cgroup quota 20.4)
- N=2000, warmup 50 / iters 200, 5 seeds, B=[8, 16, 32, 64, 128]
- baseline single-stream throughput: {'8': 514, '16': 806, '32': 885, '64': 910, '128': 925}
- index_select throughput: {'8': 601, '16': 1052, '32': 1214, '64': 1336, '128': 1401}
- end-to-end speedup: {'8': 1.17, '16': 1.3, '32': 1.37, '64': 1.47, '128': 1.51} (range 1.17–1.51x)
- assembly share, baseline: {'8': 16.1, '16': 25.3, '32': 28.4, '64': 32.7, '128': 34.5}
- assembly share, index_select: {'8': 2.5, '16': 2.2, '32': 1.4, '64': 0.8, '128': 0.5}
- tail p99/p50, baseline: {'8': 1.11, '16': 1.08, '32': 2.84, '64': 3.42, '128': 2.38}
- tail p99/p50, index_select: {'8': 1.08, '16': 1.03, '32': 1.0, '64': 1.0, '128': 1.0}
- result-scatter share, baseline: {'8': 0.8, '16': 1.1, '32': 1.2, '64': 1.1, '128': 1.2}
- result-scatter share, index_select: {'8': 0.9, '16': 1.5, '32': 1.6, '64': 1.6, '128': 1.7}
- result-scatter time (ms), baseline: {'8': 0.13, '16': 0.23, '32': 0.42, '64': 0.81, '128': 1.63}

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

