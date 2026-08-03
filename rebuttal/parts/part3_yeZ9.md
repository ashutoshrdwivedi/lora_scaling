We thank Reviewer yeZ9 for an exceptionally thorough, insightful review. We address all questions (Q1–Q6) and weaknesses (W1–W6) below, with full multi-model/GPU results added     in our General Response.

**Q1/Q3/W5 (End-to-End Latency & High-QPS CPU Assembly).** Reported latencies are already end-to-end (CPU gather + GPU forward). Result scatter (per-tenant slicing) adds only 0.15–2.12 ms (**0.7–1.3% of latency**). For high QPS, our pure-PyTorch GPU-resident assembler (`index_select`) lifts MiniLM single-stream throughput from 5,192 to 15,315 samples/s, cuts assembly share from 58–65% to 1.8–4.4% ($B \ge 64$), and stabilizes tail latency (p99/p50 from 5.28× to 1.03×). On bge-m3 it gives 1.14–1.80× speedup. Paper figures use the CPU assembler and are conservative.

**Q2 (VRAM Footprint & KV-Cache).** To clarify, Encoders don't have KV cache. At $r{=}8$ one bge-m3 adapter is 786,432 params = 1.57 MB fp16. 47,000 adapters carry 73.9 GB of parameters. Measured peak at $B{=}32$ is 76.2 GB, 3.1% above the parameter count.

**Q4/W2 (Generality Across Models, Hardware & Nodes).** Headline claims replicate on ELECTRA-large (334M), DeBERTa-v2-xlarge (885M), XLM-RoBERTa-XL (3.5B), and an L40S-48GB GPU (p50 flat within +1.29% at ceiling; 2.3–32.7× vs PEFT). On **W2**, multi-node horizontal scaling (Appendix C) shards adapter pools by tenant with zero inter-GPU sync; we concede empirical multi-node evaluation remains a design claim.

**Q5/W1 (Uniform Rank & Mixed-Rank Fleets).** This follows from our finding 3: p50 latency is flat across ranks, so padding a mixed batch to the fleet's maximum rank should have limited latency impact while the encoder base path dominates the forward pass.

However, We agree it deserves a direct test, so we measured it on bge-m3/A100 ($B{=}32$, 5 seeds, $N$ swept to 20,000), serving one heterogeneous fleet two ways: a pre-padded store, and per-rank stores that pad only during batch assembly. 

The overhead the reviewer was concerned about is small in our measurements. Across the reviewer’s $\{4,16\}$ example and a wider $\{4,8,16,32\}$ fleet, pre-padding adds at most **0.21%** mean end-to-end latency over a uniform batch already running at the fleet's maximum rank. Latency also remains flat as the adapter pool grows: the worst p50 spread across $N$ is 0.78%. Since 100.0% of sampled batches include the fleet maximum rank, these are worst-case mixed
batches rather than favorable averages

 The practical tradeoff is memory, not latency scaling. Per-rank stores cut resident adapter memory to $0.625\times$ for $\{4,16\}$ and $0.469\times$ for $\{4,8,16,32\}$, but move the padding into the gather at a small latency cost.
 
 Our submitted Limitations paragraph described this conservatively as wasted compute; based on these measurements, we will revise it at camera-ready to distinguish the latency-neutral pre-padding path from the memory-saving per-rank storage path

**W3 (Custom Kernel Optimality Ceiling).** Our 9.0% bound is **measured via ablation** (disabling the LoRA path saves 2.4 ms of 26.4 ms forward time), not purely analytic. Disentangling FLOPs shows $3d/r = 384$, bounding free-kernel FLOP gains at 0.26% (shrinking to 0.10% on XLM-R-XL). Profiling charges 4.94% of GPU time to LoRA `bmm` vs 50.6% to base projections. No custom-kernel engine (Punica, S-LoRA, vLLM) supports encoder multi-tenant LoRA; porting them is future work.

**Q6 (Registration Churn: Production Impact).** We thank the reviewer for this suggestion. We added a file-based adapter-replacement churn benchmark that measures the fraction of serving wall-clock capacity consumed by complete adapter replacement (registration plus eviction), using blocking replacement.

At 1,000 resident adapters, mean replacement share is 0.1091% at 0.1 admissions/s, 1.2065% at 1.0 admissions/s, and 12.0671% at 10.0 admissions/s. The corresponding mean achieved admission rates are 0.0830, 0.9830, and 9.9803 admissions/s. At 10.0 admissions/s, mean replacement cost is 11.7720 ms per adapter update, while mean p99 serving latency is 27.9677 ms.

At the 47,000-adapter capacity ceiling, the mean replacement share at 10.0 admissions/s is 12.1524%, with a mean achieved rate of 9.9810 admissions/s, mean replacement cost of 11.7350 ms per adapter update, and mean p99 serving latency of 27.5057 ms. The close agreement with the 1,000-resident setting indicates that the measured per-update serving impact remains stable at the evaluated capacity ceiling.

We also evaluate a cold file-load setting at 1,000 resident adapters and 10.0 admissions/s. The file path has mean replacement cost of 12.9970 ms and mean replacement share of 13.2834%. The pinned-file path has mean replacement cost of 8.5693 ms and mean replacement share of 8.7877%.

These results translate the cold-start result into an operational quantity: a deployment can map its per-pod adapter-update rate to the measured percentage of serving capacity consumed by complete adapter replacement.

**W4 (8 examples/class accuracy).** We agree this does not establish that rank-8 LoRA is the optimal adaptation choice in full-data regimes. Our accuracy contribution is narrower: under the standard SetFit few-shot protocol, rank-8 LoRA
  on bge-m3 retains 98.1% of full fine-tuning's mean accuracy and improves over a frozen encoder, which validates that the served workload is meaningful rather than a synthetic latency-only case. Separately, our serving path is numerically
  faithful to PEFT on real checkpoints ($<10^{-4}$ relative error). The main systems claims — latency, throughput, tenant ceiling, and cold-start — depend on tensor shapes $(L,d,r)$ and batch structure, not on the number of training
  examples. We will make this scope explicit.


  **W6 (Downstream Workloads).** bge-m3 is itself a retrieval/embedding encoder, so embedding generation uses the same benchmarked path with the classification head omitted. Cross-encoder reranking is also an encoder forward over a query-document pair, so we expect the same serving behavior, though we did not run an end-to-end reranking load test. 
  For the camera-ready pip-installable library, we will expose task-level wrappers for common encoder workloads, so users can apply the same serving path without having to write it themselves.

