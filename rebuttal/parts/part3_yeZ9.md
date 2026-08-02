We thank Reviewer yeZ9 for an exceptionally thorough, insightful review. We address all questions (Q1–Q6) and weaknesses (W1–W6) below, with full multi-model/GPU results in our General Response.

**Q1/Q3/W5 (End-to-End Latency & High-QPS CPU Assembly).** Reported latencies are already end-to-end (CPU gather + GPU forward). Result scatter (per-tenant slicing) adds only 0.15–2.12 ms (**0.7–1.3% of latency**). For high QPS, our pure-PyTorch GPU-resident assembler (`index_select`) lifts MiniLM single-stream throughput from 5,192 to 15,315 samples/s, cuts assembly share from 58–65% to 1.8–4.4% ($B \ge 64$), and stabilizes tail latency (p99/p50 from 5.28× to 1.03×). On bge-m3 it gives 1.14–1.80× speedup. Paper figures use the CPU assembler and are conservative.

**Q2 (VRAM Footprint & KV-Cache).** To clarify, because encoders do not require a KV cache, sequence length affects only transient activations cleared per batch. VRAM footprint is determined by adapter weights ($2MLdr$ parameters; 1.57 MB/adapter at $r{=}8$ fp16 for bge-m3, 5.90 MB for XLM-R-XL). As shown in the table below (A100-80GB GPU), peak memory at the per-GPU capacity ceiling is dominated by the adapter store, leaving predictable headroom before OOM:

| Model ($B{=}32, r{=}8$) | Base Weights | Adapter Store | Peak VRAM | Headroom |
|---|---|---|---|---|
| bge-m3 ($N{=}47,000$ ceiling) | 1.14 GB | 68.8 GB | 76.2 GB | 3.8 GB |
| XLM-R-XL ($N{=}12,000$ ceiling) | 6.96 GB | 65.9 GB | 78.6 GB | 1.4 GB |

Scaling batch size $8 \to 128$ increases peak memory by only 0.5 GB on bge-m3 and 1.0 GB on XLM-R-XL (72.5 $\to$ 73.5 GB), demonstrating that capacity is governed by adapter geometry, essentially independent of serving load.

**Q4/W2 (Generality Across Models, Hardware & Nodes).** As shown in the General Response table, headline claims replicate on ELECTRA-large (334M), DeBERTa-v2-xlarge (885M), XLM-RoBERTa-XL (3.5B), and an L40S-48GB GPU (p50 flat within +1.29% at ceiling; 2.3–32.7× vs PEFT). On **W2**, multi-node horizontal scaling (Appendix C) shards adapter pools by tenant with zero inter-GPU sync; we concede empirical multi-node evaluation remains a design claim.

**Q5/W1 (Uniform Rank & Mixed-Rank Fleets).** Pad-to-max is latency-neutral because Finding 3 sweeps uniform rank $r \in \{4,8,16,32\}$ (worst-case padding): p50 moves 40.41 $\to$ 39.93 ms (1.19%, within noise) and delta FLOPs add only 1.56% ($r{=}8$) to 6.25% ($r{=}32$) of one projection. In production, mixed-rank fleets are served via per-rank AdapterStores with rank-bucketed batching: zero padding, zero waste, zero kernel changes. Higher rank reduces tenant density (ceiling 47,000 at $r{=}8 \to$ 11,750 at $r{=}32$, Finding 4), not compute efficiency.

**W3 (Custom Kernel Optimality Ceiling).** Our 9.0% bound is **measured via ablation** (disabling the LoRA path saves 2.4 ms of 26.4 ms wall-clock time), not purely analytic. Disentangling FLOPs shows $3d/r = 384$, bounding free-kernel FLOP gains at 0.26% (shrinking to 0.10% on XLM-R-XL). Profiling charges 4.94% of GPU time to LoRA `bmm` vs 50.6% to base projections. No custom-kernel engine (Punica, S-LoRA, vLLM) supports encoder multi-tenant LoRA; porting custom kernels is future work.

**Q6 (Adapter Registration Churn).** Registration is an $O(1)$ memcpy into AdapterStore taking **0.29 ms/adapter** (flat in $N$). Replacing 1,000 adapters hourly costs 0.29 s/hour (**0.008% wall-clock**), off the serving path. In contrast, PEFT's `add_adapter` rescans module dicts in $O(N^2)$ (14.2 s for 100 vs 1,229.2 s for 1,000; ~2.5 s marginal add).

**W4/W6 (Accuracy & Downstream Workloads).** **W4:** $N{=}8$ follows SetFit protocol to show numerical equivalence ($<10^{-4}$ relative tolerance vs PEFT); serving benchmarks (latency/VRAM) depend on tensor shapes, not training sample count. **W6:** bge-m3 is natively an embedding/retrieval encoder. We concede full end-to-end reranking load tests remain unmeasured.
