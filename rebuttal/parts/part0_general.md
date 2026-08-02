We thank all reviewers for their careful reading. All three reviews rate Soundness 4. Across the reviews the main requests are (a) evidence beyond a single model, (b) evidence beyond a single GPU, and (c) characterization of the CPU-side assembly ceiling. Since submission we ran new experiments on all three axes using the paper's exact protocol (5 seeds, fp16, seq 128, $r{=}8$, warmup 50 / 200 timed iters).

**(a) Generality across encoder architectures and scale, and (b) across GPU class.** We hold the system fixed (without any model or hardware specific code) and vary the encoder and the GPU. Both headline results replicate on every configuration we tried: p50 stays flat in $N$ up to the memory ceiling, and the speedup over PEFT-mixed stays multi-fold:

| Config | Params (L, d) | GPU | p50 at ceiling vs $N{=}1000$ | Spread over sweep | Ceiling @ $r{=}8$ | Speedup vs PEFT-mixed |
|---|---|---|---|---|---|---|
| bge-m3 (paper) | 568M (24, 1024) | A100-80GB | +0.65% | 3.31% | 47,000 | 5.6–21.2× |
| ELECTRA-large | 334M (24, 1024) | A100-80GB | −0.16% | 1.36% | 51,000 | 6.2–20.9× |
| DeBERTa-v2-xlarge | 885M (24, 1536) | A100-80GB | +0.43% | 1.16% | 64,000 | 2.4–7.1× |
| XLM-RoBERTa-XL | 3.5B (36, 2560) | A100-80GB | −0.38% | 0.70% | 12,000 | 2.3–19.5× |
| bge-m3 | 568M (24, 1024) | L40S-48GB | +1.29% | 1.60% | 28,000 | 3.1–32.7× |


**Key Results & Replicated Findings:**
* **"p50 at ceiling vs $N{=}1000$":** Evaluates the $O(1)$-in-$N$ claim directly. Growing the pool by up to 64× over $N{=}1000$ moves p50 by at most **+1.29%**, and on two configs it is *negative*.
* **"Spread over sweep":** Measures the total max−min latency across the full sweep at its worst batch size. Because it is anchored at $N{=}100$, our noisiest cell (where seed-to-seed variance reaches 3.41% on the L40S versus $\le 0.5\%$ across the rest of the sweep), this metric naturally overstates any apparent trend.
* **"Speedup vs PEFT-mixed":** Compares throughput (samples/sec) against PEFT across the test settings PEFT is able to run ($N \in \{100, 1000\}$ and batch sizes $B \in \{8, 32, 128\}$).
* **"Ceiling @ $r{=}8$":** Represents GPU memory capacity limits. VRAM simply fills as per-adapter bytes × $N$ (5.90 MB/adapter for XLM-R-XL → 12,000; 1.18 MB for DeBERTa → 64,000), verified by an OOM error at the next test point.
* **Rank-insensitivity (Finding 3):** Sweeping $r \in \{4,8,16,32\}$ leaves p50 flat within 0.73% across all four new configs (measured in our broader sweep data beyond the table's $r{=}8$).

**(c) CPU assembly ceiling.** Our reported latencies are end-to-end (CPU assembly + GPU forward). To probe CPU bottlenecking in high-QPS regimes, we evaluated `all-MiniLM-L6-v2` (a smaller model where low GPU compute latency shifts the primary bottleneck to host CPU assembly):
* **Baseline (CPU Assembler):** Caps single-stream throughput at **5,192 samples/s**, with assembly consuming 58–65% of total latency for $B \geq 64$.
* **GPU-Resident Assembler (`index_select`):** We evaluated the GPU-resident assembler proposed in our Limitations. Built in pure PyTorch (`index_select`), it shifts assembly directly onto the GPU: throughput reaches **12,152–15,315 samples/s** (2.47–3.12× speedup), assembly overhead falls to 1.8–4.4%, and tail latency stabilizes (p99/p50 drops from 5.28× to 1.03×). On bge-m3, throughput improves by 1.14× ($B{=}8$) to 1.80× ($B{=}128$).
* **Validation & Conservatism:** Testing across two host CPUs confirms baseline throughput tracks CPU speed, whereas the GPU assembler reproduces identical high throughput on both. Our paper uses the baseline assembler and is therefore conservative.

**Notes on the table above:**
* **Ceiling Probes:** Only the ceiling points for the three new A100 configs are single-seed probes at $B{=}32$; only `bge-m3` and `L40S` evaluate all five batch sizes at ceiling.
* **L40S Allocator Optimization:** PyTorch's default caching allocator tops out at 27,000 adapters. Setting `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` reclaims fragmented VRAM to reach 28,000 at identical p50 (**3.7% more tenants for free**). Table figures use this setting.
* **DeBERTa Target Modules:** Adapters target value projections (`value_proj`) only, as DeBERTa-v2 shares `query_proj` with batch-shared relative-position embeddings (`share_att_key`). Value-only output matches PEFT to $2\times10^{-5}$ relative tolerance (unit-tested). Because fewer modules are wrapped, PEFT incurs less mixed-batch overhead, making our speedup row conservative.

At camera-ready (+1 page) we will consolidate (a)+(b) into a new subsection, "Generalization Across Models and Hardware", and fold (c) into Finding 5 and the Limitations. We hope these address the main concerns.
