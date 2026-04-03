# Kubernetes Deep Dive — How Every Piece Fits Together

This document explains every K8s component in the deployment, why it exists, how it connects to the others, and where the application-level code (like the consistent-hash ring) maps to infrastructure concepts. Written for someone who knows Python/ML but is learning K8s.

---

## The Big Picture: What K8s Actually Does For Us

Without K8s, you'd SSH into a GPU server, `pip install`, run `uvicorn`, and pray it doesn't crash. K8s automates:

1. **"Run N copies of my server on N GPU machines"** → Deployment
2. **"Route traffic evenly across them"** → Service + Ingress
3. **"Restart it if it crashes"** → Liveness Probe
4. **"Don't send traffic until it's ready"** → Readiness Probe
5. **"Add more copies when busy, remove when idle"** → HPA
6. **"Store configuration and secrets separately from code"** → ConfigMap + Secret

Every YAML file in `deploy/k8s/` tells K8s one of these things.

---

## Layer-by-Layer: How a Request Flows

Here's what happens when a client sends `POST /predict` with `X-Tenant-Id: tenant_42`:

```
 Client (curl / SDK)
     │
     │  HTTPS request with header: X-Tenant-Id: tenant_42
     ▼
 ┌───────────────────────────────────────────────────────┐
 │  1. DigitalOcean Load Balancer (cloud-managed)        │
 │     - Public IP / DNS endpoint                        │
 │     - SSL termination                                 │
 │     - Created automatically by NGINX Ingress          │
 └──────────────────────┬────────────────────────────────┘
                        │  HTTP (internal)
                        ▼
 ┌───────────────────────────────────────────────────────┐
 │  2. NGINX Ingress Controller  (ingress.yaml)          │
 │     - Runs as pods inside K8s (you install it once)   │
 │     - Reads Ingress resources to learn routing rules  │
 │     - OUR KEY: consistent-hash on X-Tenant-Id header  │
 │       hash("tenant_42") → always picks Pod 2          │
 └──────────────────────┬────────────────────────────────┘
                        │  HTTP to port 8080
                        ▼
 ┌───────────────────────────────────────────────────────┐
 │  3. Service  (service.yaml)                           │
 │     - Virtual IP that K8s maintains internally        │
 │     - Knows which pods are "ready" (passed probes)    │
 │     - NGINX bypasses this for routing (goes direct)   │
 │     - But other internal services use it to reach us  │
 └──────────────────────┬────────────────────────────────┘
                        │
          ┌─────────────┼──────────────┐
          ▼             ▼              ▼
 ┌──────────────┐┌──────────────┐┌──────────────┐
 │ Pod 1 (GPU)  ││ Pod 2 (GPU)  ││ Pod 3 (GPU)  │
 │              ││  ◄── HERE    ││              │
 │ FastAPI:8080 ││ FastAPI:8080 ││ FastAPI:8080 │
 │ Batcher      ││ Batcher      ││ Batcher      │
 │ Model+LoRAs  ││ Model+LoRAs  ││ Model+LoRAs  │
 └──────┬───────┘└──────┬───────┘└──────┬───────┘
        │               │               │
        └───────────────┼───────────────┘
                        ▼
 ┌───────────────────────────────────────────────────────┐
 │  4. Redis  (redis.yaml)                               │
 │     - Separate pod, no GPU needed                     │
 │     - Only used for pub/sub messaging                 │
 │     - All inference pods subscribe to same channel    │
 └───────────────────────────────────────────────────────┘
```

---

## Each K8s Resource Explained

### `namespace.yaml` — The Folder

**What:** A namespace is like a folder for K8s resources. Everything we create goes into the `lora-serving` namespace.

**Why:** Isolation. If someone else deploys a "redis" service in the `default` namespace, it won't collide with ours. Also makes cleanup easy: `kubectl delete namespace lora-serving` removes everything.

**Analogy:** Like a Python virtual environment, but for K8s resources.

---

### `configmap.yaml` — The `.env` File

**What:** A ConfigMap stores key-value pairs that get injected as environment variables into our pods. The Secret does the same but for sensitive values (base64-encoded, access-controlled).

**Why:** Decouples config from code. Change `LORA_MAX_BATCH_SIZE` from 32 to 64 without rebuilding the Docker image — just edit the ConfigMap and restart pods.

**How it connects to code:** In `deploy/server/config.py`, the `ServerConfig` class uses `pydantic-settings` to read these environment variables:
```python
class ServerConfig(BaseSettings):
    model_config = {"env_prefix": "LORA_"}
    max_batch_size: int = 32  # reads LORA_MAX_BATCH_SIZE from env
```

In `deployment.yaml`, we inject them:
```yaml
envFrom:
  - configMapRef:
      name: lora-serving-config   # ← all keys become env vars
  - secretRef:
      name: lora-serving-s3-creds # ← S3 keys, kept secret
```

---

### `deployment.yaml` — The Actual Server Pods

**What:** A Deployment tells K8s "run N replicas of this container image, on nodes with GPUs, with these resource limits."

**Key concepts:**

#### GPU Scheduling
```yaml
resources:
  limits:
    nvidia.com/gpu: "1"   # Request exactly 1 GPU
```
K8s has a plugin (NVIDIA Device Plugin, pre-installed on DO GPU nodes) that exposes GPUs as schedulable resources. When your pod says "I need 1 GPU," K8s finds a node with a free GPU and schedules the pod there. Two pods cannot share the same GPU (by default).

#### The Three Probes (Critical for GPU workloads)

```
Timeline of a pod starting up:
 
 t=0s     Pod created, container starts
 t=5s     Startup probe begins checking /ready every 10s
 t=15s    Still loading model... probe fails, but that's OK (30 retries)
 t=45s    Model loaded! /ready returns 200
           → Startup probe passes
           → Readiness probe takes over → pod receives traffic
           → Liveness probe starts (every 30s)
 t=300s   GPU OOM kills the process → /health fails
           → Liveness probe fails 3x → K8s restarts the pod
```

| Probe | Endpoint | Purpose | What happens if it fails |
|---|---|---|---|
| **Startup** | `GET /ready` | "Are you done loading the model?" | K8s waits patiently (up to 5 min) |
| **Readiness** | `GET /ready` | "Can you handle traffic right now?" | K8s stops sending traffic to this pod |
| **Liveness** | `GET /health` | "Are you still alive?" | K8s kills and restarts the pod |

**Why startup probe matters:** Loading a base model + 1000 adapters takes 30-60 seconds. Without a startup probe, K8s thinks the pod is broken after ~30s and kills it in a restart loop.

#### Rolling Update Strategy
```yaml
strategy:
  type: RollingUpdate
  rollingUpdate:
    maxUnavailable: 0   # never kill an existing pod before the new one is ready
    maxSurge: 1          # create 1 new pod at a time
```
This means during a deployment update (new Docker image), K8s:
1. Starts 1 new pod
2. Waits for it to pass readiness (model loaded)
3. Only then kills 1 old pod
4. Repeats

You never have zero capacity during a deploy.

---

### `service.yaml` — The Internal DNS Name

**What:** A Service gives your set of pods a stable DNS name and virtual IP. Other things inside K8s can reach your pods at `lora-serving.lora-serving.svc.cluster.local:80`.

**Session Affinity:**
```yaml
sessionAffinity: ClientIP
```
This is a **basic** sticky routing fallback. It means: if the same client IP sends multiple requests, K8s tries to route them to the same pod. But this is weak — it's based on source IP (which might be a NAT gateway shared by many tenants). The real sticky routing happens at the Ingress layer.

**Why we still need it:** Internal services (like a metrics scraper) that call our service directly don't go through Ingress. The session affinity gives them some stickiness too.

---

### `ingress.yaml` — The Smart Router (Where Sticky Routing Lives)

**What:** An Ingress resource tells the NGINX Ingress Controller how to route external HTTP traffic to your Service. It's the "front door" of your application.

**Where NGINX runs:**
NGINX Ingress Controller is a separate Deployment (that you install once on the cluster). It runs as regular pods (no GPU needed) and watches for Ingress resources. When you create our `ingress.yaml`, NGINX picks it up and configures itself.

```
You install NGINX once:
  kubectl apply -f .../deploy/static/provider/do/deploy.yaml
  → Creates pods in "ingress-nginx" namespace
  → Creates a DigitalOcean Load Balancer (public IP)

You create an Ingress resource:
  kubectl apply -f deploy/k8s/ingress.yaml
  → NGINX reads it and adds routing rules
```

**The consistent-hash annotation — this is the money shot:**
```yaml
annotations:
  nginx.ingress.kubernetes.io/upstream-hash-by: "$http_x_tenant_id"
```

This single line tells NGINX: "Take the value of the `X-Tenant-Id` HTTP header, hash it, and use the hash to pick which backend pod to route to."

**How it works internally:**
1. NGINX discovers the list of backend pod IPs from the Service (e.g., `[10.0.1.5, 10.0.1.6, 10.0.1.7]`)
2. When a request arrives with `X-Tenant-Id: tenant_42`:
   - NGINX computes `hash("tenant_42") = 8837291...`
   - `8837291 % 3 pods = pod index 1` → routes to `10.0.1.6`
3. Next request with `X-Tenant-Id: tenant_42` → same hash → same pod
4. Request with `X-Tenant-Id: tenant_99` → different hash → might be a different pod

**What happens when pods scale:**
Without the `upstream-hash-by-subset` annotation, adding a 4th pod changes the modulo from `% 3` to `% 4`, which remaps ~75% of tenants. With it:
```yaml
nginx.ingress.kubernetes.io/upstream-hash-by-subset: "true"
```
NGINX uses a consistent-hash ring (same concept as our `routing.py`), so adding a 4th pod only remaps ~25% of tenants.

---

### How `routing.py` (App-Level) Relates to NGINX (Infra-Level)

**They solve the same problem at different layers:**

| | NGINX (Infra) | `routing.py` (App) |
|---|---|---|
| **Where** | Runs in `ingress-nginx` namespace as a reverse proxy | Runs inside each inference pod |
| **What it routes** | HTTP requests from clients | Nothing by default — it's a library |
| **How** | Hashes `X-Tenant-Id` header | Hashes any string (tenant_id) |
| **When you'd use it** | Always — it's the primary routing mechanism | For smart pre-loading decisions |

**When would you actually use `routing.py`?**

Scenario: You have 10,000 tenants but each GPU pod can only hold 3,000 adapters. You need to decide which adapters to pre-load on which pod.

```python
# On each pod at startup:
ring = ConsistentHashRing()
ring.add_pod(PodInfo(pod_name="pod-0", pod_ip="10.0.1.5"))
ring.add_pod(PodInfo(pod_name="pod-1", pod_ip="10.0.1.6"))
ring.add_pod(PodInfo(pod_name="pod-2", pod_ip="10.0.1.7"))

# "Which adapters should I pre-load?"
my_pod_name = os.environ["HOSTNAME"]  # K8s sets this
for tenant_id in all_tenant_ids:
    target = ring.get_pod(tenant_id)
    if target.pod_name == my_pod_name:
        store.load_from_file(tenant_id, ...)  # I own this tenant
```

This way, `pod-0` only loads tenants that NGINX will route to it. The consistent-hash algorithm in `routing.py` matches NGINX's, so they agree on the assignment.

**Right now, this isn't wired up in `app.py`** — all pods load all adapters. But the `routing.py` module is there for when you hit the memory limit and need to partition adapters across pods.

---

### `redis.yaml` — The Notification Bus

**What:** A single Redis instance used purely for pub/sub messaging. Not a cache, not a datastore — just a message broker.

**How it connects:**

```
Training Pipeline finishes retraining tenant_42's adapter
    │
    │ Uploads new pytorch_adapter.bin to DigitalOcean Spaces (S3)
    │
    │ Publishes to Redis:
    │   PUBLISH lora:adapter_updates '{"adapter_id":"tenant_42","s3_key":"..."}'
    │
    ▼
┌─────────────────────────────────────────┐
│ Redis pod (redis.lora-serving.svc:6379) │
│                                         │
│  Channel: lora:adapter_updates          │
│     │                                   │
│     ├──→ Pod 1 subscriber (reload.py)   │
│     ├──→ Pod 2 subscriber (reload.py)   │
│     └──→ Pod 3 subscriber (reload.py)   │
└─────────────────────────────────────────┘
         │          │          │
         ▼          ▼          ▼
   All pods download the new adapter from S3
   and reload it into GPU memory (thread-safe)
```

**Why all pods reload, not just one:** Because we don't know which pod NGINX will route `tenant_42` to. If pods scale or a pod restarts, the tenant might land on a different pod. So all pods should have the latest adapter.

**In production:** Replace this single Redis pod with DigitalOcean Managed Redis. It's $15/month and gives you HA + persistence.

---

### `hpa.yaml` — The Autoscaler

**What:** HorizontalPodAutoscaler watches a metric and adjusts the replica count of your Deployment.

**The GPU problem:** Standard K8s only knows about CPU and memory. It can't see GPU utilization. To scale on GPU load, you need:

1. **NVIDIA DCGM Exporter** — a DaemonSet that reads GPU metrics and exposes them to Prometheus
2. **Prometheus** — collects the metrics
3. **Prometheus Adapter** — translates Prometheus metrics into K8s custom metrics API

This is a lot of setup, so the HPA currently falls back to CPU utilization, which is a decent proxy (tokenization + batch assembly are CPU-bound).

**Conservative scaling:**
```yaml
scaleUp:
  stabilizationWindowSeconds: 60    # wait 1 min before adding pod
  policies:
    - type: Pods
      value: 1                      # add only 1 pod at a time
      periodSeconds: 120            # at most every 2 min
scaleDown:
  stabilizationWindowSeconds: 300   # wait 5 min before removing pod
```

GPU pods take ~60s to start and cost ~$2/hour. We scale up cautiously (1 pod every 2 min) and scale down very slowly (5 min cooldown) to avoid thrashing.

---

## Common Interview Questions About This Architecture

**Q: What happens if NGINX routes tenant_42 to Pod 2, but Pod 2 doesn't have that adapter loaded?**
A: The FastAPI endpoint returns 404 ("Adapter not found"). In the current implementation, all pods pre-load all adapters from the `LORA_PRELOAD_ADAPTER_IDS` list. For production with thousands of tenants, you'd add on-demand loading: if the adapter isn't cached, download and load it on the fly (adds ~200ms to the first request).

**Q: What happens if a pod dies mid-request?**
A: NGINX detects the pod is gone (readiness probe fails), removes it from the upstream list, and re-hashes. The client gets a 502 and retries. The consistent-hash ring remaps only ~1/N tenants to other pods.

**Q: Why not use gRPC instead of REST?**
A: For encoder models with short inputs/outputs, the HTTP overhead is negligible compared to GPU inference time (~50ms). gRPC shines when you're streaming tokens (LLM generation) or have extremely high QPS (>10K/s). REST is simpler to debug with curl.

**Q: Why is there only 1 uvicorn worker?**
A: Each worker would try to load the base model onto the GPU. With 1 GPU per pod, you can only have 1 model instance. The async batcher handles concurrency — multiple HTTP requests are served by 1 worker via asyncio, then flushed as a batch to the single GPU.
