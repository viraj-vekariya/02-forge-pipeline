"""Ramp real load at the deployed service until it breaches its SLA.

This is the measurement the project exists to make. "It's deployed" is a claim anyone
can make; "here is the request rate at which p99 crosses 100ms, and here is what
breaks first" is a claim that requires actually doing it.

Method
------
Closed-loop with a fixed worker count per step, which is the model that matches a real
client pool: N callers each waiting for a response before sending the next. Open-loop
(fire at a fixed rate regardless of responses) measures something different and, on a
service that is already saturated, mostly measures the load generator's queue.

Each step raises concurrency, runs for a fixed duration, and records the full latency
distribution. The step where p99 crosses the SLA is the breaking point.

Two properties that keep it honest:

* The generator's own overhead is measured first and reported, because a load test
  whose client is the bottleneck measures the client.
* Every step is preceded by a short warm phase whose results are discarded, so the
  previous step's queue does not leak into the next step's numbers.

Run:  python3 loadtest/harness.py --base http://127.0.0.1:8200
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional

import httpx

ROOT = Path(__file__).resolve().parent.parent
SLA_P99_MS = 100.0
CONCURRENCY_STEPS = (1, 2, 4, 8, 16, 32, 64)


@dataclass
class StepResult:
    concurrency: int
    duration_sec: float
    requests: int
    errors: int
    throughput_rps: float
    mean_ms: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    max_ms: float
    server_mean_total_ms: float
    server_mean_inference_ms: float
    server_inference_share: float
    breached: bool


def _percentile(values: List[float], p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    return s[min(len(s) - 1, int(len(s) * p))]


def _payload() -> Dict[str, List[float]]:
    """A fixed, valid payload. Generated once and reused so that neither payload
    construction nor JSON encoding of varying data lands inside the timed section."""
    return {"pixels": [float((i * 7) % 256) for i in range(784)]}


def _worker(base: str, payload: Dict, stop: threading.Event,
            samples: List[tuple], errors: List[int], lock: threading.Lock) -> None:
    """Record (completion_timestamp, latency_ms) per request.

    The timestamp is not decoration. An earlier version collected latencies for the
    whole post-warm period but divided by the sleep duration alone, so requests that
    completed while the workers were draining after stop.set() were counted against a
    shorter window. That inflated throughput by a CONSTANT ~25%, which showed up as a
    flat 25% error in Little's Law across every step - a systematic offset, which is
    the signature of a measurement bug rather than a property of the service.
    Timestamping lets throughput and latency be computed over the identical window.
    """
    # One client per worker with keep-alive. A fresh connection per request would make
    # this a TCP handshake benchmark.
    with httpx.Client(timeout=30.0, limits=httpx.Limits(max_connections=4)) as client:
        local: List[tuple] = []
        local_errors = 0
        while not stop.is_set():
            t0 = time.perf_counter()
            try:
                r = client.post(f"{base}/predict", json=payload)
                done = time.perf_counter()
                if r.status_code == 200:
                    local.append((done, (done - t0) * 1000))
                else:
                    local_errors += 1
            except Exception:                    # noqa: BLE001
                local_errors += 1
        # Merge once at the end rather than locking per request; contending on a lock
        # inside the timed loop would make the harness measure itself.
        with lock:
            samples.extend(local)
            errors.append(local_errors)


def measure_client_overhead(base: str, payload: Dict, n: int = 200) -> Dict[str, float]:
    """How fast can the generator go against a trivial endpoint?

    If this number is close to the service's measured latency, the harness is the
    bottleneck and every number below it is meaningless.
    """
    samples = []
    with httpx.Client(timeout=10.0) as client:
        for _ in range(20):
            client.get(f"{base}/health/live")
        for _ in range(n):
            t0 = time.perf_counter()
            client.get(f"{base}/health/live")
            samples.append((time.perf_counter() - t0) * 1000)
    return {"health_p50_ms": round(_percentile(samples, 0.5), 3),
            "health_p99_ms": round(_percentile(samples, 0.99), 3),
            "samples": n}


def run_step(base: str, concurrency: int, duration: float, warm: float) -> StepResult:
    payload = _payload()
    samples: List[tuple] = []
    errors: List[int] = []
    lock = threading.Lock()
    stop = threading.Event()

    workers = [threading.Thread(target=_worker,
                                args=(base, payload, stop, samples, errors, lock),
                                daemon=True)
               for _ in range(concurrency)]
    for w in workers:
        w.start()

    # Discard the warm phase: the previous step's backlog is still draining.
    time.sleep(warm)
    with lock:
        samples.clear()
        errors.clear()

    with httpx.Client(timeout=10.0) as c:
        c.get(f"{base}/metrics")                 # reset reference point
    window_start = time.perf_counter()
    time.sleep(duration)
    window_end = time.perf_counter()

    stop.set()
    for w in workers:
        w.join(timeout=30)

    # Keep only completions inside the measurement window, and measure throughput over
    # exactly that window. Both numbers now describe the same interval.
    with lock:
        in_window = [(t, ms) for t, ms in samples if window_start <= t <= window_end]
    latencies = [ms for _, ms in in_window]
    elapsed = window_end - window_start

    with httpx.Client(timeout=10.0) as c:
        server = c.get(f"{base}/metrics").json()["latency"]

    total_errors = sum(errors)
    n = len(latencies)
    p99 = _percentile(latencies, 0.99)

    return StepResult(
        concurrency=concurrency,
        duration_sec=round(elapsed, 2),
        requests=n,
        errors=total_errors,
        throughput_rps=round(n / elapsed, 1),
        mean_ms=round(statistics.fmean(latencies), 3) if latencies else 0.0,
        p50_ms=round(_percentile(latencies, 0.50), 3),
        p95_ms=round(_percentile(latencies, 0.95), 3),
        p99_ms=round(p99, 3),
        max_ms=round(max(latencies), 3) if latencies else 0.0,
        server_mean_total_ms=server["mean_total_ms"],
        server_mean_inference_ms=server["mean_inference_ms"],
        server_inference_share=server["inference_share"],
        breached=p99 > SLA_P99_MS,
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8200")
    ap.add_argument("--duration", type=float, default=6.0)
    ap.add_argument("--warm", type=float, default=1.5)
    ap.add_argument("--sla", type=float, default=SLA_P99_MS)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    base = args.base.rstrip("/")

    steps = (1, 4, 16) if args.quick else CONCURRENCY_STEPS
    duration = 2.0 if args.quick else args.duration
    warm = 0.5 if args.quick else args.warm

    with httpx.Client(timeout=10.0) as c:
        try:
            ready = c.get(f"{base}/health/ready")
        except Exception as exc:                 # noqa: BLE001
            print(f"ERROR: service unreachable at {base} ({exc})", file=sys.stderr)
            return 1
        if ready.status_code != 200:
            print(f"ERROR: service is not ready: {ready.json()}", file=sys.stderr)
            return 1
        model = c.get(f"{base}/model").json()["info"]

    print(f"target   {base}")
    print(f"model    {model['version']}  acc={model['test_accuracy']}")
    print(f"SLA      p99 <= {args.sla:.0f}ms\n")

    overhead = measure_client_overhead(base, _payload())
    print(f"harness overhead (GET /health/live): p50 {overhead['health_p50_ms']}ms  "
          f"p99 {overhead['health_p99_ms']}ms\n")

    print(f"  {'conc':>5} {'rps':>9} {'p50':>8} {'p95':>8} {'p99':>8} "
          f"{'max':>8} {'err':>5} {'inf%':>6}")
    results: List[StepResult] = []
    breaking_point: Optional[StepResult] = None

    for concurrency in steps:
        r = run_step(base, concurrency, duration, warm)
        results.append(r)
        flag = "  <-- SLA BREACH" if r.breached else ""
        print(f"  {r.concurrency:>5} {r.throughput_rps:>9,.0f} {r.p50_ms:>8.2f} "
              f"{r.p95_ms:>8.2f} {r.p99_ms:>8.2f} {r.max_ms:>8.2f} {r.errors:>5} "
              f"{r.server_inference_share * 100:>5.0f}%{flag}")
        if r.breached and breaking_point is None:
            breaking_point = r

    peak = max(results, key=lambda r: r.throughput_rps)
    baseline = results[0]

    print()
    if breaking_point:
        print(f"  BREAKING POINT: p99 crossed {args.sla:.0f}ms at concurrency "
              f"{breaking_point.concurrency} ({breaking_point.throughput_rps:,.0f} rps)")
    else:
        print(f"  SLA never breached up to concurrency {steps[-1]}")
    print(f"  peak throughput {peak.throughput_rps:,.0f} rps at concurrency "
          f"{peak.concurrency}")
    print(f"  throughput gain 1 -> {peak.concurrency} workers: "
          f"{peak.throughput_rps / baseline.throughput_rps:.2f}x "
          f"(ideal would be {peak.concurrency}x)")
    print(f"  inference is {peak.server_inference_share * 100:.0f}% of server time "
          f"at peak - the rest is framework and serialisation")

    out = {
        "target": base,
        "model": model,
        "sla_p99_ms": args.sla,
        "harness_overhead": overhead,
        "steps": [asdict(r) for r in results],
        "breaking_point": asdict(breaking_point) if breaking_point else None,
        "peak": asdict(peak),
        "scaling": {
            "workers": peak.concurrency,
            "actual_gain": round(peak.throughput_rps / baseline.throughput_rps, 2),
            "ideal_gain": peak.concurrency,
            "efficiency": round(
                (peak.throughput_rps / baseline.throughput_rps) / peak.concurrency, 3),
        },
    }
    dest = ROOT / "outputs" / "loadtest.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=2) + "\n")
    print(f"\n  wrote {dest.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
