# Decisions — Forge Pipeline

Every non-obvious choice, the alternatives, and why they lost.

---

## D-01 · Fashion-MNIST, not MNIST and not CIFAR-10

**Chose:** Fashion-MNIST.

- *MNIST* is saturated. A small CNN gets 99.4%, so a real pipeline regression — shuffled
  labels, broken normalisation, a bad export — moves the metric by less than run-to-run
  noise and CI cannot gate on it. A dataset where the model lands at 93% leaves room for
  a regression to be visible.
- *CIFAR-10* is 3-channel 32×32 and takes several times longer to train for the same
  pipeline lesson. This project is about the pipeline; a longer train makes the pipeline
  harder to test, which is backwards.

**Requirement it had to meet:** downloadable without credentials, so CI works on a clean
runner with no secrets.

## D-02 · A real CNN, but under 500k parameters

**Chose:** 468,010 parameters, ~2 minutes to train.

**Why not larger:** a pipeline whose model cannot be trained inside a CI job cannot be
tested by that pipeline. The constraint is testability, not accuracy.

**Why not a stand-in (logistic regression, a constant):** the pipeline's claim is that
it trains, exports, versions, verifies and serves a real neural network. Weights that do
not have to be learned would make every step downstream vacuous.

**Enforced by a test** (`test_model_stays_small_enough_to_train_in_ci`) so the constraint
cannot be lost by accident.

## D-03 · Three splits, and the test set is touched exactly once

**Chose:** 54,000 / 6,000 / 10,000, cut with a fixed seed.

**Why:** early stopping selects a model. A model selected on the same data it is scored
on has a score that is optimistic by construction. Validation drives early stopping and
temperature fitting; test is opened once, in `evaluate.py`, after training is over.

**Why the split is seeded:** two CI runs with different validation sets are not
comparable, and "did this commit help?" is the question the pipeline exists to answer.

**Normalisation constants come from the training split only** — computing them over the
full dataset would leak test statistics into every training batch. A small leak, but a
real one, and invisible afterwards.

## D-04 · Every random source seeded in one place

**Chose:** `set_seed()` covering `random`, `numpy`, `torch`, and CUDA.

**Why one function:** missing any single source produces a run that is *almost*
reproducible, which is worse than one that obviously is not — the classic case being
numpy seeded but torch not, so the data order matches while the weights differ, and the
difference gets attributed to the change under test.

**Two tests, in opposite directions:** same seed must give identical weights, *and*
different seeds must give different ones. The second catches a seeding bug that pins
every run to one draw.

## D-05 · OneCycle, AdamW, label smoothing, gradient clipping

- **OneCycle over a step schedule.** A step schedule tuned for 90 epochs would spend this
  entire 8-epoch run at its initial learning rate. OneCycle warms up and anneals inside a
  short run, which is what makes 8 epochs sufficient.
- **AdamW over Adam.** Decoupled weight decay. Adam's L2 term interacts with the adaptive
  step size, so the effective decay differs per parameter — which is not what "weight
  decay" is supposed to mean.
- **Label smoothing 0.05.** Improves generalisation, and it is *why* the model ends up
  under-confident with a fitted temperature of 0.72. That is a real, traceable
  consequence of a deliberate choice, and it is documented rather than explained away.
- **Gradient clipping at 5.0.** Cheap insurance. One exploding batch would wreck a run
  that CI then reports as a model regression, sending you to look in the wrong place.

## D-06 · TorchScript tracing, verified — not scripting, not ONNX

**Chose:** `torch.jit.trace` plus `optimize_for_inference`, then re-load and compare.

- *Tracing vs scripting:* the forward pass is a straight-line sequence of modules with no
  data-dependent control flow, so tracing captures it exactly and avoids TorchScript's
  restrictions on acceptable Python.
- *ONNX:* the right answer if the serving runtime were not PyTorch. It is, so ONNX adds
  a conversion and a second runtime for nothing.

**The verification is the real decision.** Tracing will happily produce a graph that
differs from the eager model if there *is* hidden control flow, and nothing complains
until production predictions are wrong. So export re-loads its own output and asserts
agreement on 512 real samples (measured max logit difference **3.58e-06**) or deletes
the file and fails.

**Export on CPU regardless of training device:** a graph traced on MPS or CUDA can carry
device assumptions that only surface when it is loaded elsewhere.

## D-07 · Temperature scaling, fitted on validation with LBFGS

**Chose:** one scalar, fitted by minimising NLL on validation.

**Why calibration at all:** anything that routes on confidence — an abstention gate, a
human-review queue, a fallback — is built on a lie if the model's confidence does not
match its accuracy. ECE 0.0333 → 0.0140 is a 58% reduction for one parameter.

**Why LBFGS:** the problem is one-dimensional, smooth and convex. It converges in a
handful of iterations where SGD would need a learning rate nobody wants to tune.

**Why optimise `log T`:** keeps T positive without a constraint or a clamp.

**Why not on test:** fitting a parameter on the data used to report the result is the
definition of a leak.

## D-08 · A manifest travels with every artifact

**Chose:** dataset fingerprint, git sha, seed, hyperparameters, environment, full epoch
history, metrics, export checksum — all in one JSON beside the weights.

**Why:** a model file with no provenance cannot be rolled back to with confidence, cannot
be reproduced, and cannot be checked against what is currently serving. The manifest is
what turns "a .pt file" into "a deployable".

**The temperature lives here too**, not only in the evaluation report, because serving
needs it. A number a human is supposed to copy from a report into a config is a number
that will eventually be copied wrong.

## D-09 · The registry verifies, smoke-tests, then swaps

**Chose:** load → checksum → smoke test → atomic swap, with the predecessor retained.

- **Checksum before activation:** a file that does not match the manifest is not the
  model that was evaluated, so its reported accuracy describes something else.
- **Smoke test before activation:** loading is not the same as working. A shape mismatch
  or a missing operator surfaces on the first forward pass, and that pass should be ours.
- **Swap last:** assigning first and validating after leaves a broken model live for the
  duration of the validation.
- **Keep the predecessor:** this is what makes a bad deploy a *failed deploy* rather than
  an outage.

## D-10 · Warmup is part of readiness, not an optimisation

**Measured cold penalty: 4–6×** on the first inference (1.597ms vs 0.266ms warm, batch 1).

**Why it belongs in readiness:** a service that reports ready before warming sends the
load balancer's first requests into the slowest inferences the process will ever run —
which is precisely when an autoscaler starts adding instances because latency looks bad,
making the situation worse.

**Why warm every batch shape:** allocator behaviour and kernel selection differ by shape.
Warming only at batch 1 leaves the first batch-32 request cold.

## D-11 · Three health endpoints, not one

- **live** — a failure gets the container KILLED. It must never depend on the model, a
  disk or a downstream, or a slow dependency causes a restart loop that guarantees the
  outage it was meant to prevent.
- **ready** — a failure removes the instance from the load balancer without killing it.
  This is where "loaded and warmed" belongs, and it returns **503**, because 200 with
  `{"ready": false}` keeps traffic arriving.
- **startup** — suppresses liveness during a slow boot so a 20-second model load is not
  repeatedly killed at 10.

Collapsing these into one endpoint is the most common way a deploy becomes a restart loop.

## D-12 · `torch.set_num_threads(1)`

**Chose:** one torch thread per worker process.

**Why:** torch defaults to one thread per core. Under load that means every concurrent
request tries to use every core, and the resulting oversubscription makes throughput
*fall* as concurrency rises. The service scales by handling more requests, not by making
one request use the whole machine.

**Consequence, accepted:** single-request latency is higher than it could be. That is the
right trade for a service measured on throughput and p99 rather than on best-case latency.

## D-13 · Closed-loop load generation, not open-loop

**Chose:** N workers, each waiting for a response before sending the next.

**Why:** it matches a real client pool. Open-loop (fire at a fixed rate regardless of
responses) measures something different, and against an already-saturated service it
mostly measures the generator's own queue depth.

**Why the harness measures its own overhead first:** a load test whose client is the
bottleneck measures the client. Reported at p50 0.23ms / p99 0.49ms, well below the
service's numbers.

**Why every step discards a warm phase:** the previous step's backlog is still draining,
and letting it leak in makes each step's numbers a blend of two states.

**The bug this decision did not prevent:** throughput was originally computed over the
sleep duration while latencies were collected over sleep + drain. Fixed by timestamping
every completion and computing both over the identical window — 25.7% error → 0.0%.

## D-14 · Test three hypotheses against each other, not one

**Chose:** `profile.py` states H1/H2/H3 explicitly and tests each.

**Why:** "throughput plateaus" has several plausible causes and picking the most
plausible one is how a wrong diagnosis becomes a wasted quarter. Little's Law is
falsifiable and it held to 0.0%, which rules out leaks and deadlocks rather than merely
making them seem unlikely.

**Why H3 starts a real second service:** the per-process hypothesis is also the *remedy*,
so it is worth measuring rather than reasoning about. It came back confounded, and
reporting it as inconclusive with the CPU budget attached is more useful than either
claiming it works or claiming it does not.

## D-15 · Artifacts are copied into the image, never trained in it

**Chose:** `COPY artifacts/ ...` in the Dockerfile.

**Why:** the image should contain the model that was tested, byte for byte. Training
inside the build produces a different model on every build, so what CI verified and what
ships are two different artifacts that merely share a commit.

## D-16 · CPU-only torch wheel in the runtime image

**Chose:** `--index-url https://download.pytorch.org/whl/cpu`.

**Why:** the default wheel bundles CUDA and is roughly 2.5GB against ~200MB for the CPU
build. Shipping CUDA to a container that will never see a GPU is the largest avoidable
cost in most ML images, and it slows every pull, every cold start and every deploy.

## D-17 · The autoscale trigger comes from the measurement

**Chose:** `soft_limit = 25` concurrent requests per machine in `fly.toml`.

**Why that number:** the load test puts p99 at 41ms at concurrency 16 and 69ms at 32,
crossing the 100ms SLA at 64. A soft limit of 25 keeps p99 near 20ms with real headroom
before the measured breaking point.

**What makes this defensible:** it is derived from a run, not chosen because it is a
round number. It is also the honest weak spot — the breaking point was measured against
a *single* instance, so the trigger is grounded but the autoscaling behaviour itself is
still unproven.
