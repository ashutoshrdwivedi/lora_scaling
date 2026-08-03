We thank Reviewer yeZ9 for an exceptionally thorough, insightful review. We address all questions (Q1–Q6) and weaknesses (W1–W6) below, with full multi-model/GPU results added     in our General Response.

**Q1/Q3/W5 (End-to-End Latency & High-QPS CPU Assembly).** Reported latencies are already end-to-end (CPU gather + GPU forward). Result scatter (per-tenant slicing) adds only 0.15–2.12 ms (**0.7–1.3% of latency**). For high QPS, our pure-PyTorch GPU-resident assembler (`index_select`) lifts MiniLM single-stream throughput from 5,192 to 15,315 samples/s, cuts assembly share from 58–65% to 1.8–4.4% ($B \ge 64$), and stabilizes tail latency (p99/p50 from 5.28× to 1.03×). On bge-m3 it gives 1.14–1.80× speedup. Paper figures use the CPU assembler and are conservative.

**Q2 (VRAM Footprint & KV-Cache).** To clarify: encoders have no KV cache, so sequence length affects only transient activations, freed per batch. VRAM is set by adapter geometry ($2MLdr$; at $r{=}8$ fp16, 1.50 MiB/adapter for bge-m3, 5.62 MiB for XLM-R-XL). On our A100 (79.3 GiB usable per CUDA):

| Model ($B{=}32, r{=}8$) | Base | Adapter Store | Peak VRAM | Headroom |
|---|---|---|---|---|
| bge-m3 ($N{=}52,000$) | 1.06 GiB | 76.2 GiB | 78.4 GiB | 0.9 GiB |
| XLM-R-XL ($N{=}12,000$) | 6.49 GiB | 65.9 GiB | 73.2 GiB | 6.1 GiB |

Peak is predictable: base $+\ N\times$adapter$\times1.0133 + 0.137$ GB (1.33% alloc padding) fits 6 points ($N{=}46$–52k) to **within 0.36 MB**. Load barely moves it: $B{:}8\to128$ costs 0.5 GB (bge-m3), 1.0 GB (XLM-R-XL). The bge-m3 row is a new direct probe ($B{=}32$, single-seed, `expandable_segments:True`): **52,000** fits, 53,000 OOMs, p50 flat to 0.68%; 

**Q4/W2 (Generality Across Models, Hardware & Nodes).** Headline claims replicate on ELECTRA-large (334M), DeBERTa-v2-xlarge (885M), XLM-RoBERTa-XL (3.5B), and an L40S-48GB GPU (p50 flat within +1.29% at ceiling; 2.3–32.7× vs PEFT). On **W2**, multi-node horizontal scaling (Appendix C) shards adapter pools by tenant with zero inter-GPU sync; we concede empirical multi-node evaluation remains a design claim.

**Q5/W1 (Uniform Rank & Mixed-Rank Fleets).** This follows from our finding 3: p50 latency is flat across ranks, so padding a mixed batch to the fleet's maximum rank should have limited latency impact while the encoder base path dominates the forward pass.

However, We agree it deserves a direct test, so we measured it on bge-m3/A100 ($B{=}32$, 5 seeds, $N$ swept to 20,000), serving one heterogeneous fleet two ways: a pre-padded store, and per-rank stores that pad only during batch assembly. 

The overhead the reviewer was concerned about is small in our measurements. Across the reviewer’s $\{4,16\}$ example and a wider $\{4,8,16,32\}$ fleet, pre-padding adds at most **0.21%** mean end-to-end latency over a uniform batch already running at the fleet's maximum rank. Latency also remains flat as the adapter pool grows: the worst p50 spread across $N$ is 0.78%. Since 100.0% of sampled batches include the fleet maximum rank, these are worst-case mixed
batches rather than favorable averages

 The practical tradeoff is memory, not latency scaling. Per-rank stores cut resident adapter memory to $0.625\times$ for $\{4,16\}$ and $0.469\times$ for $\{4,8,16,32\}$, but move the padding into the gather at a small latency cost.
 
 Our submitted Limitations paragraph described this conservatively as wasted compute; based on these measurements, we will revise it at camera-ready to distinguish the latency-neutral pre-padding path from the memory-saving per-rank storage path

**W3 (Custom Kernel Optimality Ceiling).** Our 9.0% bound is **measured via ablation** (disabling the LoRA path saves 2.4 ms of 26.4 ms forward time), not purely analytic. Disentangling FLOPs shows $3d/r = 384$, bounding free-kernel FLOP gains at 0.26% (shrinking to 0.10% on XLM-R-XL). Profiling charges 4.94% of GPU time to LoRA `bmm` vs 50.6% to base projections. No custom-kernel engine (Punica, S-LoRA, vLLM) supports encoder multi-tenant LoRA; porting them is future work.

**Q6 (Registration Churn: Production Impact).** We ran the interleaved add-under-load experiment the reviewer asks for (bge-m3/A100, $B{=}32$, 3 seeds, 60 cells), timing the *production* registration path — `AdapterStore.load_from_file`, adapter file $\to$ GPU — under serving load. Registration costs **11.9 ms/adapter** and is **flat in $N$**: 11.90 ms at $N{=}1{,}000$ against 11.98 ms at $N{=}47{,}000$ ($+0.7$% across a 47$\times$ pool-size range, 95% CI $\pm$0.10 ms). Replicating on a second, independent A100 host reproduces the flatness exactly ($+0.67$%: 14.92 $\to$ 15.02 ms) at a 25% higher constant — the cost is CPU-bound deserialization, so the absolute figure tracks host CPU while the $O(1)$ scaling does not. We quote the faster host throughout; the same 10 replacements/s measurement on the slower one reads 15.1% rather than 12.1%. The timed region also installs the tenant's classification head, measured separately at 0.02 ms and so immaterial. This is 41$\times$ Finding 7's 0.29 ms, which times pool construction with no file I/O on *either* system — the like-for-like comparison against `add_adapter`.

Driving replacements at a fixed rate — churn as an independent variable rather than an emergent function of (Zipf $\alpha$, tenant count, capacity) — registration takes **1.2% of serving wall-clock at 1 replacement/s** and **12.1% at 10/s**, unchanged at the capacity ceiling (12.15% at $N{=}47{,}000$). For calibration: a 47,000-tenant pool in which *every* tenant retrains daily generates 0.54 replacements/s, and hourly retraining of all 47,000 generates 13/s. A full 1,000-adapter pool replaced once an hour costs 11.9 s of 3,600 s, i.e. **0.33%**. Because tenants stay resident rather than demand-paged (pools beyond one GPU shard by tenant, Appendix C), churn is bounded by retraining cadence, not request traffic; we measure the ceiling regardless — synchronous admission saturates at **85 adapters/s**.

The $O(N^2)$ contrast is undisturbed: PEFT's `add_adapter` rescans per-layer `ModuleDict`s (14.2 s for 100 adapters vs 1,229.2 s for 1,000; $\approx$2.5 s marginal at $N{=}1{,}000$), some **210$\times$** our per-adapter cost. The residual 11.9 ms is CPU-side deserialization, not I/O or transfer: a cold page cache adds only 9%, and the host-to-device copy itself is 0.2 ms. All figures are synchronous, matching what we ship — the hot-reload service holds the inference lock across registration. Overlapping on a copy stream is a proposed optimization, not current behaviour: it holds p50 flat ($<$0.03% of serving time) at a tail cost (p99 28 $\to$ 55 ms at 10/s).

**W4 (8 examples/class accuracy).** We agree this does not establish that rank-8 LoRA is the optimal adaptation choice in full-data regimes. Our accuracy contribution is narrower: under the standard SetFit few-shot protocol, rank-8 LoRA
  on bge-m3 retains 98.1% of full fine-tuning's mean accuracy and improves over a frozen encoder, which validates that the served workload is meaningful rather than a synthetic latency-only case. Separately, our serving path is numerically
  faithful to PEFT on real checkpoints ($<10^{-4}$ relative error). The main systems claims — latency, throughput, tenant ceiling, and cold-start — depend on tensor shapes $(L,d,r)$ and batch structure, not on the number of training
  examples. We will make this scope explicit.


  **W6 (Downstream Workloads).** bge-m3 is itself a retrieval/embedding encoder, so embedding generation uses the same benchmarked path with the classification head omitted. Cross-encoder reranking is also an encoder forward over a query-document pair, so we expect the same serving behavior, though we did not run an end-to-end reranking load test. 
  For the camera-ready pip-installable library, we will expose task-level wrappers for common encoder workloads, so users can apply the same serving path without having to write it themselves.

