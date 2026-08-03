We thank the reviewer for the encouraging assessment and for suggesting a broader evaluation across model families. We agree that extending validation across diverse encoders strengthens our findings, and we have executed these experiments.

### Only one model tested 
Holding the system fixed, we re-ran the paper's full grid (same protocol, 5 seeds) on three further encoders and a second GPU class: ELECTRA-large (334M, a replaced-token-detection discriminator), DeBERTa-v2-xlarge (885M, disentangled attention), XLM-RoBERTa-XL (3.5B, 36 layers), and bge-m3 on an L40S-48GB. The results table is present in our general response. Three things replicate everywhere:

- **$O(1)$ in tenant count.** Growing the pool to the memory ceiling moves p50 by at most **+1.29%** relative to $N{=}1000$.

- **The speedup over PEFT's mixed-batch API persists**: 2.3–20.9× on A100 and up to 32.7× on the L40S.
- **Rank-insensitivity** (Finding 3) holds on all four new configurations, p50 flat within 0.73% across $r \in \{4,8,16,32\}$.

Notably the property is *strongest* at the largest scale: XLM-RoBERTa-XL has the flattest profile of all five configurations (0.70% total spread across the sweep), which is what the decomposition predicts. The base forward dominates more heavily as the encoder grows, so the per-tenant delta path matters less.

These become a new subsection, "Generalization Across Models and Hardware", at camera-ready, using the additional page.

### Would we release the code as a library?
Yes. The benchmark and reference implementation are already at the anonymized link; at camera-ready we will de-anonymize the repository and publish a pip-installable package, with task-level wrappers for common encoder workloads. Since submission we have also added a generic attachment path that hooks any HuggingFace encoder exposing standard projection modules, so applying the method to a new model requires no model-specific code, that is how two of the three new models above were run.
