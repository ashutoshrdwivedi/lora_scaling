# Caveats for the churn registration results

Findings from review of the harness after these CSVs were produced. The code in
`src/lora_serving/benchmark/churn.py` has been corrected; **these CSVs were
generated before those corrections**, so the notes below apply to the data in
this directory. Nothing here changes the headline figures, but two columns
should not be read at face value.

## 1. `blocking` is the production mode, not `background`

`deploy/server/reload.py:119` acquires the same inference lock the serving path
holds before invoking the reload callback, so registration is serialized against
inference in the shipped system. **Quote the `blocking` rows as the production
figures.** The `background` rows bound what an overlapped implementation could
buy and at what tail cost; they are a proposed design, not current behaviour.

## 2. Registration time includes on-device head synthesis

The timed region installs the tenant's classification head as well as the
adapter, which is correct — a tenant is not servable without one. But
`make_synthetic_lr_weights()` builds it on the GPU with a curand `randn` rather
than copying it from host, so the measured cost is not purely file-load + H2D.

**Resolved by direct measurement — see §8, raw output in
`host2_replication/diag_breakdown.txt`.** Across three repeats, on-device synthesis
costs 0.0190–0.0194 ms against 0.0173–0.0178 ms for the equivalent pinned H2D
copy, so the contamination is **0.0015–0.0018 ms, or 0.013–0.015%** of the
11.897 ms figure. (An earlier conservative bound
derived from the `synthetic` arm total put it at ≤3.7%; the direct measurement is
~260× tighter.) The primary results stand unmodified. All three arms pay the
identical head cost regardless, so between-arm comparisons were never affected.
The harness now installs the head by pinned copy.

## 3. `serve_throughput_samples_sec` is not delivered throughput

That column is `batch_size / mean(serve_latency)` — throughput *while serving*.
It excludes the intervals the loop spent registering, so under blocking churn it
overstates what the pod actually delivers. Delivered throughput is
`batches_served × batch_size / wall_s`:

| target rate | reported | delivered | gap |
|---|---|---|---|
| 0.1 /s | 1151.4 | 1121.3 | −2.6% |
| 1 /s | 1162.0 | 1118.9 | −3.7% |
| 10 /s | 1160.0 | 995.5 | −14.2% |
| 100 /s | 987.3 | 18.1 | −98.2% |

(blocking, N=1,000, mean of 3 seeds). The harness now emits both columns as
`serve_throughput_while_serving` and `delivered_throughput_samples_sec`.

## 4. Share columns are wall-time-weighted ratios of totals

`registration_share_pct` and `registration_work_share_pct` are accumulated time
over accumulated time, so a long registration contributes proportionally more.
That is the right construction for this question — the quantity of interest is
*pod wall-clock capacity unavailable to inference* — but it is **not** a mean
per-request percentage and should not be described as one. Registration here
happens between batches, independently of requests, so a per-request share would
not be meaningful anyway. Tail registration latency is reported separately
(`replace_p95_ms`) so individual stalls stay visible rather than being averaged
into the share.

## 5. At saturation, background rows mix measurement windows

`admissions_completed` was snapshotted at the end of the timed window, but the
latency arrays were built after the worker backlog drained — so replacement
totals and work share could include registrations that finished *after* the
window. Negligible where the target rate is attained (at 10/s: drain 0.02 s,
0 shed) but material at saturation (at 100/s: drain 2.98 s, 1,851 shed).

**Affected rows: `churn_rate_sweep.csv`, `churn_mode=background`,
`target_admission_rate=100`.** Treat that cell's `replace_*` and
`registration_work_share_pct` as approximate. Fixed in the harness.

## 6. `admissions_requested` undercounts arrivals

It incremented only when an admission was accepted inline or queued; shed
arrivals were counted separately. True generated demand is
`admissions_requested + admissions_shed`. Renamed to `admissions_accepted` in
the harness, with `admissions_arrived` added for the total.

## 7. No hit-rate metric, and Zipf alpha barely matters

The module docstring claimed an achieved hit rate was reported. None is
collected, and none would be informative: requests always draw from the current
resident pool, so every request hits by construction. Zipf alpha shapes only
which resident adapters are popular, and since replacement walks slots
round-robin rather than by popularity, alpha has little influence on the result.
Docstring corrected.

## 8. Second-host replication, and what is host-dependent

Artifacts: `host2_replication/` — CSV, run and setup logs, the diagnostic
scripts under `scripts/`, their captured output in `diag_breakdown.txt`, the
smoke run that first exposed the pinned-path pathology in `smoke.csv`, and a
`SHA256SUMS` manifest generated on the pod and verified after transfer.

`host2_replication/` holds an independent rerun of phase 1 on a different A100
(different CPU, local container disk rather than the network `/workspace`
mount), using the corrected harness — head installed by pinned H2D copy rather
than synthesised on device.

| | host 1 (primary) | host 2 |
|---|---|---|
| file, N=1,000 | 11.90 ms | 14.92 ms |
| file, N=47,000 | 11.98 ms | 15.02 ms |
| drift over 47× pool growth | +0.7% | +0.67% |

**The O(1)-in-N property replicates exactly. The constant does not** — host 2 is
25% slower. That is consistent with the cost being CPU-side `torch.load`
deserialization -- `torch.load`→cuda alone is 14.5–15.1 ms on host 2 against
~11.9 ms for the whole registration on host 1 (`diag_breakdown.txt`), so the absolute figure tracks host CPU and
storage while the scaling claim does not depend on either. Anyone rerunning the
artifact on different hardware should expect the constant to move and the
flatness to hold.

Direct measurement on host 2 also settles §2 above: on-device head synthesis
costs 0.0190–0.0194 ms against 0.0173–0.0178 ms for the pinned copy, so the
contamination of the primary CSVs is **0.0015–0.0018 ms — 0.013–0.015%** of the
11.897 ms figure, far inside the ≤3.7% bound quoted there. Three repeats are
recorded in `diag_breakdown.txt`. The primary results stand
unmodified.

`file_pinned` is excluded from the host-2 run: copying the non-contiguous
transposed tensor into page-locked memory costs 130–150 ms on that host against
~0.3 ms on host 1, which saturates the arm. The 7.4 ms `file_pinned` figure in
`churn_registration_paths.csv` is therefore host 1 only and must not be quoted
as a general result -- it has been removed from the Q6 rebuttal text.
