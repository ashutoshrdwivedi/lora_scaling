We thank the reviewer for the encouraging assessment, and for identifying single-model evaluation as the paper's thinnest axis. We agree, and we have measured it.

**Only one model tested.** Holding the system fixed, we re-ran the paper's full grid (same protocol, 5 seeds) on three further encoders and a second GPU class: ELECTRA-large (334M, a replaced-token-detection discriminator rather than a masked LM), DeBERTa-v2-xlarge (885M, disentangled attention), XLM-RoBERTa-XL (3.5B, 36 layers), and bge-m3 on an L40S-48GB. The table is in our general response. Three things replicate everywhere:

- **$O(1)$ in tenant count.** Growing the pool to the memory ceiling moves p50 by at most **+1.27%** relative to $N{=}1000$; on ELECTRA it is 2.57% *below*, and on DeBERTa −0.25%. There is no upward trend in $N$ in any configuration.
- **The speedup over PEFT's mixed-batch API persists**: 2.4–22.8× on A100 and up to 32.4× on the L40S.
- **Rank-insensitivity** (Finding 3) holds on all four new configurations, p50 flat within 2.46% across $r \in \{4,8,16,32\}$.

Notably the property is *strongest* at the largest scale: XLM-RoBERTa-XL has the flattest profile of all five configurations (0.69% total spread across the sweep), which is what the decomposition predicts — the base forward dominates more heavily as the encoder grows, so the per-tenant delta path matters less. We present the 3.5B result as a stress test rather than a recommended deployment.

These become a new subsection, "Generalization Across Models and Hardware", at camera-ready, using the additional page.

**Would we release the code as a library?** Yes. The benchmark and reference implementation are already at the anonymized link; at camera-ready we will de-anonymize the repository and publish a pip-installable package with a stable API for the three components a deployment needs: the adapter store, the batch assembler, and the encoder attachment. Since submission we have also added a generic attachment path that hooks any HuggingFace encoder exposing standard projection modules, so applying the method to a new model requires no model-specific code — that is how two of the three new models above were run, and it is what makes the packaging worth doing.
