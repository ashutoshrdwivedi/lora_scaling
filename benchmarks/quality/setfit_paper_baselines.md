# SetFit paper Table 2 — N=8, paraphrase-mpnet-base-v2 (SETFIT_MPNET)

Source: Tunstall et al. 2022, "Efficient Few-Shot Learning Without Prompts" (arxiv 2209.11055), Table 2.
Reported as mean ± std (subscript) across multiple seeds.

| Dataset    | SetFit MPNet (N=8) | Metric    |
|------------|--------------------|-----------|
| SST-5      | 43.6 ± 3.0         | accuracy  |
| AmazonCF   | 40.3 ± 11.8        | **MCC**   |
| CR         | 88.5 ± 1.9         | accuracy  |
| Emotion    | 48.8 ± 4.5         | accuracy  |
| EnronSpam  | 90.1 ± 3.4         | accuracy  |
| AG News    | 82.9 ± 2.8         | accuracy  |
| Average†   | 62.3 ± 4.9         | mean      |

† AG News is excluded from the paper's average (T-Few has AGNews in training set).

## Notes for our comparison

- **AmazonCF uses MCC, not accuracy.** Our setfit_compare.py currently reports accuracy via SetFit's default metric. Either switch to MCC for this dataset or note the metric mismatch.
- **SST-2 is not in Table 2** — paper uses SST-5 instead. Our SST-2 numbers can't be paper-anchored.
- **All paper numbers use 10 seeds**; ours currently use 1 (seed=42). Expect our single-seed point estimate to land within ±1 std of the paper number when reproducing well.
- **Same base model**: paraphrase-mpnet-base-v2 (110M params).
- **Same training config**: N=8 per class, default SetFit hyperparameters.
