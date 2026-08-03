## Important review findings

### 1. Background mode does not match the current production implementation

The comment says background mode is “what a real server does,” but the shipped reloader acquires the same inference lock used by serving before calling the reload callback. That makes the current production path behaviorally closer to blocking mode.

Background mode is a proposed optimized design, not presently the production implementation.

### 2. The classification-head description is incorrect

The benchmark says the classification head is a host-side tensor copied to the GPU. It actually calls `make_synthetic_lr_weights()`, which constructs `coef` and `intercept` directly on the GPU, including a GPU `torch.randn`, at [synthetic.py](/workspace/lora_scaling/src/lora_serving/benchmark/synthetic.py:47).

Because that call is inside the timed registration region at [churn.py](/workspace/lora_scaling/src/lora_serving/benchmark/churn.py:222), registration time still includes synthetic GPU random-number generation. The measurement is consequently not purely file loading plus host-to-device transfer.

### 3. The reported serving throughput excludes blocking time

`serve_throughput_samples_sec` is calculated from the mean of `serve_latencies`. Those latencies only cover assembly and forward execution. They do not include inline registration intervals.

In blocking mode, this is not actual delivered throughput under churn. The appropriate delivered throughput would be approximately:

\[
\frac{\text{batches served}\times\text{batch size}}{\text{wall seconds}}
\]

The CSV contains enough information to calculate this, but it is not currently reported as such.

### 4. The share metrics are ratios of totals

Both registration-share columns use accumulated time. They are intentionally wall-time-weighted, so a long registration contributes proportionally more.

For this particular reviewer question, that is arguably appropriate: a 500 ms registration really does consume more production capacity than a 5 ms registration. But it is not a mean per-request percentage and should not be described that way.

Here registration occurs between batches independently of requests, so “per-request registration share” is not especially natural. The clearer production metric is:

> Percentage of pod wall-clock capacity unavailable to inference because of adapter updates.

Tail registration latency should be reported separately so a few stalls are visible rather than hidden inside the share.

### 5. Background work-share accounting mixes measurement windows

`admissions_completed` is captured at the end of the timed window, before the worker backlog is drained. However, replacement latency arrays are constructed after draining and therefore include registrations completed after the measurement window.

As a result, achieved admission rate describes in-window completions, while replacement totals and work share can include post-window work. This matters at saturation, when the backlog is large.

### 6. `admissions_requested` is misleadingly named

It increments only when an admission is accepted for inline execution or queued. Shed target arrivals are counted separately. Consequently:

\[
\text{actual generated demand}
\approx
\text{admissions_requested}+\text{admissions_shed}
\]

The existing name can be mistaken for total target arrivals.

### 7. The claimed hit-rate output does not exist

The module documentation says the achieved hit rate is reported. In this workload, requests always select the current resident pool, so the hit rate is effectively 100%. No hit/miss metric is actually collected or written.

Zipf α only changes resident-adapter popularity and likely has limited relevance because replacements are selected round-robin rather than based on popularity.

## Overall assessment

Conceptually, this is the right benchmark for your proposed reviewer response:

> We update \(X\) adapters per day; after accounting for replication and pod placement, that produces \(Y\) registrations per pod per second; at that rate, registration consumes \(Z\%\) of pod serving capacity.

The strongest measurements are the file-based blocking replacement cost, achieved admission rate, blocked wall-time share, and serving latency under churn.

Before using its results in a paper, I would be cautious about the background-mode production claim, the GPU-generated classification-head cost, the throughput denominator, and the mixed measurement window at saturation.