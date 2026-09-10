# Forge Pipeline

A CI/CD pipeline that trains a real convolutional network on a real public dataset,
tests it, verifies its own export, containerises it, and deploys it alongside a Java
service — then loads the deployed endpoint until it breaks and profiles why.

**~3,100 lines · 30 tests passing · verified on Apple M-series, CPython 3.13.9, torch 2.13, Temurin 21**

Every number here was measured on a run and written to `outputs/`. Nothing is estimated.

---

## The finding

The load test was supposed to answer "how much can it take?" It answered something
more specific.

`loadtest/harness.py`, ramping concurrency against the deployed service, SLA p99 ≤ 100ms:

| concurrency | rps | p50 ms | p95 ms | p99 ms | max ms |
|---:|---:|---:|---:|---:|---:|
| 1 | 1,188 | 0.77 | 1.05 | 2.37 | 17.16 |
| 2 | **1,512** | 1.20 | 1.71 | 4.05 | 15.83 |
| 4 | 1,413 | 2.43 | 4.01 | 10.79 | 56.65 |
| 8 | 1,418 | 5.11 | 7.45 | 19.29 | 51.49 |
| 16 | 1,277 | 11.43 | 17.05 | 41.37 | 81.80 |
| 32 | 1,315 | 22.91 | 32.22 | 69.16 | 124.15 |
| 64 | 1,174 | 51.50 | 72.18 | **134.70** | 152.29 |

**Breaking point: p99 crosses 100ms at concurrency 64, 1,174 rps.**
**Peak throughput: 1,512 rps, reached at concurrency 2 and never exceeded.**

Throughput is flat from two clients onward while latency grows linearly. That shape has
a specific meaning, and `loadtest/profile.py` tests it rather than asserting it.

### Little's Law holds to 0.0%

In a closed system, `concurrency = throughput × residence time`. Every step of the ramp
is an independent test of that identity:

| concurrency | rps | mean latency | predicted concurrency | error |
|---:|---:|---:|---:|---:|
| 1 | 1,188 | 0.84 ms | 1.00 | 0.0% |
| 2 | 1,512 | 1.32 ms | 2.00 | 0.1% |
| 4 | 1,413 | 2.83 ms | 4.00 | 0.0% |
| 8 | 1,418 | 5.64 ms | 8.00 | 0.0% |
| 16 | 1,277 | 12.53 ms | 16.00 | 0.0% |
| 32 | 1,315 | 24.36 ms | 32.03 | 0.1% |
| 64 | 1,174 | 54.46 ms | 63.94 | 0.1% |

**Mean error 0.0% across seven independent steps.** That identifies the plateau exactly:
a saturated closed-loop queue sitting at its service-rate ceiling. Not a memory leak,
not a deadlock, not a misconfigured thread pool — the server is simply serving as fast
as it can, and every extra client is standing in line.

It also caught a bug in my own harness. The first run showed a *constant* 25.7% error at
every step, and a constant error is the signature of a measurement artifact rather than
a property of the system. It was: throughput counted requests that completed while the
workers were draining, but divided by the sleep duration alone. Timestamping every
completion and measuring both over the identical window took the error to 0.0%.

## The second finding: batching does nothing for this model

Received wisdom is that batching buys 5–10× on inference. Measured per-instance
inference time:

| batch | per-instance inference |
|---:|---:|
| 1 | **0.303 ms** |
| 8 | 0.383 ms |
| 32 | 0.423 ms |
| 128 | 0.451 ms |

It gets *worse* with size. Over real HTTP batching does show a 1.2× win — but that win
is amortised HTTP framing and serialisation, not model efficiency, which is exactly why
it does not grow with batch size. In-process, with no network to amortise, batching is a
net loss.

**Why:** batching pays when there is a large fixed per-call cost to spread — kernel
launch, host-to-device transfer, Python dispatch. A 468k-parameter CNN pinned to one CPU
thread has almost none of that, so per-item compute dominates and building a larger input
tensor costs more than it saves. On a GPU this would very likely invert.

**Consequence:** throughput here is bought with process concurrency, not batch size.

## The third finding, reported as inconclusive

The natural next hypothesis is that the ceiling is per-process, so N workers give N×
throughput. Measured: **4 workers gave 1.14× of an ideal 4×.**

That is not evidence against horizontal scaling, because the test is confounded and the
profiler says so: 10 cores against a demand of 13 (4 server processes + 8 client threads
+ 1 idle service). The load generator is co-located with the service, so the machine is
CPU-oversubscribed and the measurement cannot separate *"horizontal scaling doesn't
work"* from *"this laptop ran out of cores"*. Settling it needs a separate
load-generation host, which is stated as an open item rather than papered over.

---

## The model

Fashion-MNIST — 70,000 real 28×28 greyscale images, 10 classes, fetched from its public
source. Not MNIST: MNIST is saturated at 99.5% and a pipeline regression would hide
underneath it.

| | |
|---|---|
| architecture | ForgeCNN, 2 conv blocks + head, **468,010 parameters** |
| split | 54,000 train / 6,000 val / 10,000 test, seeded, deterministic |
| training | 8 epochs, AdamW + OneCycle, ~2 min on MPS |
| best val accuracy | 0.9413 |
| **test accuracy** | **0.9344** on 10,000 held-out images |

### Calibration — and a result that goes the wrong way

| | raw | calibrated |
|---|---:|---:|
| ECE | 0.0333 | **0.0140** |
| mean confidence | 0.9016 | 0.9481 |

**Temperature fitted to 0.7206 — below 1.0, meaning the model is *under*-confident.**
That is the opposite of the usual cross-entropy failure mode, where networks are
notoriously over-confident. It is traceable to `label_smoothing=0.05` in the training
loss: smoothing caps the target probability at 0.955 and explicitly teaches the network
never to be fully certain. The calibration step then has to sharpen rather than soften.

Worst confusions, all semantically sensible: Shirt→T-shirt/top (92), T-shirt/top→Shirt
(72), Shirt→Coat (48).

---

## What the pipeline guarantees

Six properties, each enforced by code and covered by a test:

1. **Provenance.** Every artifact carries a manifest: dataset fingerprint, git sha,
   seed, hyperparameters, environment, metrics. "Which model is serving?" has an exact
   answer.
2. **The export verifies itself.** `train/export.py` re-loads its own TorchScript output
   and asserts agreement with the eager model — max logit difference **3.58e-06** over
   512 real samples — or refuses to write the file. Tracing silently diverging is a real
   failure mode and this is the only place it can be caught.
3. **Integrity before activation.** The serving registry checks the artifact's sha256
   against the manifest. A truncated file fails at load, not at a user's request.
4. **A failed deploy is not an outage.** The new model is fully built and smoke-tested
   before it replaces the live one; on any failure the previous model keeps serving, and
   rollback is one call.
5. **Readiness is separate from liveness.** Ready requires loaded *and* warmed and
   returns **503** so a load balancer removes the instance. Liveness never touches the
   model — if it did, a model problem would become a restart loop.
6. **Language-agnostic.** The same pipeline builds, tests and containerises a Java
   Spring Boot feature service beside the Python one. Nothing in the pipeline knows which
   is which.

Warmup is part of readiness because the cold penalty is real and measured: **4–6× on the
first inference** (1.597ms vs 0.266ms warm at batch 1). Serving before warming means the
load balancer's first requests are the slowest the process will ever produce.

---

## Layout

| path | lines | what |
|---|---:|---|
| `train/` | 736 | data, model, training, evaluation + calibration, verified export |
| `serve/` | 548 | registry, warmup, health, FastAPI app, drawing UI |
| `services/java-service/` | 234 | Spring Boot feature service |
| `loadtest/` | 776 | ramp harness, hypothesis profiler, report consolidator |
| `tests/` | 388 | 30 tests |
| `infra/` + CI + Makefile | ~420 | Dockerfiles, compose, Fly, GitHub Actions |

## Run it

```bash
make setup
make all           # train -> evaluate -> export -> test
make serve         # http://localhost:8200
```

The UI is a drawing canvas. Draw a shirt, watch it come back with a full probability
distribution — then press **Run load ramp** and watch throughput flatten while p99
climbs, live, in the browser.

```bash
make loadtest      # ramp to SLA breach, then profile the cause
make report        # consolidate everything into outputs/results.json
make docker        # both services in containers
```

Deploy: `fly deploy --config infra/fly.toml --dockerfile infra/Dockerfile.serve`.
The autoscale trigger in `fly.toml` is taken directly from the load test — soft limit 25
concurrent requests per machine, which keeps p99 near 20ms with headroom before the
measured breaking point at 64.

## Known limits

- Trained on MPS, served on CPU. Accuracy is device-independent; every latency number
  is not.
- **The horizontal-scaling hypothesis is untested on real separate hosts.** The one
  measurement is confounded by CPU oversubscription and is reported as inconclusive.
- Autoscaling is configured but the trigger itself is unproven — the breaking point was
  measured against a single instance.
- No GPU path. The batching finding would very likely invert on one, because
  kernel-launch overhead is precisely the fixed cost that is missing on CPU.
- The load generator shares a machine with the service, which caps measurable throughput.
- CI's quality gate is a floor (accuracy > 0.85 on a 2-epoch run), not the full-run
  target. It catches a broken pipeline, not a subtle regression.

See `DECISIONS.md` for why each choice was made and what was rejected.
