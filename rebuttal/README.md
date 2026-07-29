# EMNLP 2026 Industry Track — rebuttal

**Paste from `parts/`. Nothing else in this directory is submission-ready.**

Each file in `parts/` is one OpenReview comment: Markdown + LaTeX, under the
5,000-character limit. Post `part0_general.md` as the general response (or at
the top of each per-reviewer reply if the portal only allows per-review
threads), then the four per-reviewer parts.

| File | Goes to | Chars |
|---|---|---|
| `parts/part0_general.md` | general response / all reviewers | 4,582 |
| `parts/part1_5qtX.md` | Reviewer 5qtX | 2,239 |
| `parts/part2_aNEv.md` | Reviewer aNEv | 3,113 |
| `parts/part3a_yeZ9_questions.md` | Reviewer yeZ9 (Q1–Q6) | 4,904 |
| `parts/part3b_yeZ9_weaknesses.md` | Reviewer yeZ9 (W1–W6) | 4,973 |

## No number gets typed by hand

```
python rebuttal/make_numbers.py    # CSVs -> numbers.json + NUMBERS.md
python rebuttal/check.py           # verify every part; exit 1 on any problem
```

- **`make_numbers.py`** reads the raw result files under `benchmarks/results/`
  and computes every quantity the rebuttal quotes. It is the single source of
  truth. Output: `numbers.json` (machine-checkable) and `NUMBERS.md` (a fact
  sheet to read while writing).
- **`check.py`** enforces the character limit and scans every numeric token in
  every part, failing on any number that does not trace back to `numbers.json`
  or to a short, justified list of non-measurements (batch sizes, question
  labels, and so on). A hallucinated or stale number cannot pass.

Re-run both after **any** edit to a part. If `check.py` flags a number, the
number is wrong or its source is missing from the registry — do not silence it
by extending `ALLOWED` unless the value genuinely is not a measurement.

## Conventions the registry pins down

These bit the first draft, so they are now enforced in one place:

- **Speedups are throughput ratios**, matching `paper/build_numbers.py`
  (`SpeedupMixed*`), not p50-latency ratios. The script reproduces the paper's
  published 5.6–21.2× for bge-m3 exactly, which is how we know the new rows are
  computed the same way. `speedup_latency_*` is kept alongside for reference.
- **Seeds aggregate by mean**, matching `aggregate_seeds()` in the paper build.
- **Two different latency claims, never conflated.** `at_ceiling_*` compares p50
  at the pool ceiling against $N{=}1000$ — this is where the O(1)-in-N claim
  lives. `spread_*` is the total max−min across the whole sweep, anchored at
  $N{=}100$, the noisiest cell we measure; it is a conservative bound, not the
  claim.
- **MB is decimal MB**, not MiB (1.57 MB = 1.50 MiB per bge-m3 adapter).
- The A100 ceiling probes are **single-seed at $B{=}32$ only**; only bge-m3 and
  the L40S have all five batch sizes at their ceiling.

## Working notes (not for submission)

`rebuttal_response.md` and `rebuttal_answers_draft.md` are the earlier drafts.
They contain reasoning, reviewer-by-reviewer strategy, and caveats worth
keeping — but **their numbers predate the re-measured assembly benchmark and
the convention fixes above**. Read them for argument, never for figures.
