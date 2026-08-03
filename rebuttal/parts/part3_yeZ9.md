We thank Reviewer yeZ9 for an exceptionally thorough, insightful review. We address all questions (Q1–Q6) and weaknesses (W1–W6) below, with full multi-model/GPU results added in our General Response.

### Q1/Q3/W5 (End-to-End Latency & High-QPS CPU Assembly).
Reported latencies are already end-to-end (CPU gather + GPU forward). Result scatter (per-tenant slicing) adds only 0.15–2.12 ms (**0.7–1.3% of latency**). For high QPS, CPU bottleneck is not a hard ceiling, we implemented the pure-PyTorch GPU-resident assembler (`index_select`), it lifts MiniLM single-stream throughput from 5,192 to 15,315 samples/s, cuts assembly share from 58–65% to 1.8–4.4% ($B \ge 64$), and stabilizes tail latency (p99/p50 from 5.28× to 1.03×). On bge-m3 it gives 1.14–1.80× speedup. Beyond a single replica, we expect deployment to be horizontal: replicas are sharded by tenant (Appendix C), so aggregate QPS is spread across pods rather than absorbed by one host's CPU.

### Q2 (VRAM Footprint & KV-Cache).
To clarify, Encoders don't have KV cache. At $r{=}8$ one bge-m3 adapter is 786,432 params = 1.57 MB fp16. 47,000 adapters carry 73.9 GB of parameters. Measured peak at $B{=}32$ is 76.2 GB, 3.1% above the parameter count.

### Q4/W2 (Generality Across Models, Hardware & Nodes).
Headline claims replicate on ELECTRA-large (334M), DeBERTa-v2-xlarge (885M), XLM-RoBERTa-XL (3.5B), and an L40S-48GB GPU (p50 flat within +1.29% at ceiling; 2.3–32.7× vs PEFT). On **W2**, multi-node horizontal scaling (Appendix C) shards adapter pools by tenant with zero inter-GPU sync; we concede empirical multi-node evaluation remains a design claim.

### Q5/W1 (Uniform Rank & Mixed-Rank Fleets).
This follows from our finding 3: p50 latency is flat across ranks, so padding a mixed batch to the fleet's maximum rank should have limited latency impact while the encoder base path dominates the forward pass. However, we agree it deserves a direct test, so we measured it on bge-m3/A100 ($B{=}32$, 5 seeds, $N$ swept to 20,000), serving a heterogeneous fleet two ways: a pre-padded store, and per-rank stores that pad only during batch assembly. 

The overhead is small in our measurements. Across the $\{4,16\}$ example and a wider $\{4,8,16,32\}$ fleet, pre-padding adds at most *0.21%* mean end-to-end latency over a uniform batch already running at the fleet's maximum rank. Latency also remains flat as the adapter pool grows: the worst p50 spread across $N$ is 0.78%. Since 100.0% of sampled batches include the fleet maximum rank, these are worst-case mixed batches.

The tradeoff is memory: per-rank stores use $0.625\times$ memory for ${4,16}$ and $0.469\times$ for ${4,8,16,32}$, at a small gather-time coststorage path

### Q6 (Registration Churn: Production Impact).
We measured interleaved “add adapter while serving” on bge-m3/A100 ($B{=}32$, three seeds, 60 cells), timing the production-style AdapterStore.load_from_file path from adapter file to GPU weights while inference continues. Registration is effectively $O(1)$ in pool size: 11.90 ms at 1,000 adapters versus 11.98 ms at 47,000 (0.7%; 95% CI ±0.10 ms). A second A100 replicated this (14.92 vs 15.02 ms; 0.67%). Installing the classification head takes 0.02 ms and is immaterial. This is 41× Finding 7’s 0.29 ms because that figure measures synthetic pool construction without adapter-file I/O. 

The table below shows adapter registration share as percentage of total serving time, under different admission rates : 

 Admission rate | Resident adapters | Registration share | Mean update cost | Mean p99 serving latency |
|---:|---:|---:|---:|---:|
| 1/s | 1,000 | 1.21% | 11.94 ms | 27.66 ms |
| 1/s | 47,000 | 1.52% | 11.93 ms | 27.34 ms |
| 10/s | 1,000 | 12.07% | 11.77 ms | 27.97 ms |
| 10/s | 47,000 | 12.15% | 11.74 ms | 27.51 ms |


### W4 (8 examples/class accuracy).
We agree this does not establish that rank-8 LoRA is the optimal adaptation choice in full-data regimes. Our accuracy contribution is narrower: under the standard SetFit few-shot protocol, rank-8 LoRA on bge-m3 retains 98.1% of full fine-tuning's mean accuracy and beats a frozen encoder, validating a realistic served workload. Separately, our serving path matches PEFT checkpoints ($<10^{-4}$ relative error). The main claims (latency, throughput, tenant ceiling, and cold-start) depend on tensor shapes $(L,d,r)$ and batch structure, not on the training set size.

### W6 (Downstream Workloads).
bge-m3 embedding generation uses the same benchmarked path without the classification head. Cross-encoder reranking is also an encoder forward over a query-document pair, so we expect the same serving behavior, though we did not run an end-to-end reranking load test. For the camera-ready pip-installable library, we will expose task-level wrappers for common encoder workloads.