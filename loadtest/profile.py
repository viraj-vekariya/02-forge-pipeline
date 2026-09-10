"""Root-cause the saturation the ramp found, instead of guessing at it.

harness.py establishes THAT throughput plateaus. This establishes WHY, by testing
three explanations against each other rather than asserting the most plausible one.

  H1  The service is a saturated single-server queue.
      Prediction: Little's Law holds - concurrency = throughput x latency - so
      latency rises linearly with concurrency while throughput stays flat. If the
      residuals are small, the plateau is a service-rate limit, not a bug.

  H2  The model forward pass is the service rate.
      Prediction: measured single-request inference time should be close to
      1 / peak_throughput. If it is much smaller, the framework is the limit instead.

  H3  The limit is per-process, so more processes should scale it.
      Prediction: N uvicorn workers gives ~N x throughput. This is the falsifiable
      one, and it is also the fix - so it is tested by actually starting a second
      service and measuring, not by reasoning about it.

Run:  python3 loadtest/profile.py
"""

from __future__ import annotations

import json
import os
import signal
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Dict, List

import httpx

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _payload() -> Dict:
    return {"pixels": [float((i * 7) % 256) for i in range(784)]}


def _percentile(values: List[float], p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    return s[min(len(s) - 1, int(len(s) * p))]


def check_littles_law(loadtest: Dict) -> Dict:
    """H1: in a closed system, concurrency = throughput x residence time.

    Every step of the ramp is an independent observation of that identity. If it holds
    across all of them, the service is a queue at its service-rate ceiling and the flat
    throughput is the expected behaviour of a saturated server - not a leak, not a
    deadlock, not a thread-pool misconfiguration.
    """
    rows = []
    for step in loadtest["steps"]:
        predicted = step["throughput_rps"] * (step["mean_ms"] / 1000.0)
        actual = step["concurrency"]
        rows.append({
            "concurrency": actual,
            "throughput_rps": step["throughput_rps"],
            "mean_latency_ms": step["mean_ms"],
            "predicted_concurrency": round(predicted, 2),
            "relative_error": round(abs(predicted - actual) / actual, 4),
        })
    errors = [r["relative_error"] for r in rows]
    return {"hypothesis": "closed-system queue at its service-rate ceiling",
            "rows": rows,
            "mean_relative_error": round(statistics.fmean(errors), 4),
            "max_relative_error": round(max(errors), 4),
            "holds": max(errors) < 0.15}


def measure_service_rate(base: str, n: int = 400) -> Dict:
    """H2: time one request at concurrency 1 and compare 1/latency to peak throughput."""
    payload = _payload()
    totals, infers, pres = [], [], []
    with httpx.Client(timeout=30.0) as c:
        for _ in range(30):
            c.post(f"{base}/predict", json=payload)
        for _ in range(n):
            t0 = time.perf_counter()
            r = c.post(f"{base}/predict", json=payload).json()
            totals.append((time.perf_counter() - t0) * 1000)
            infers.append(r["timing_ms"]["inference"])
            pres.append(r["timing_ms"]["preprocess"])
    mean_total = statistics.fmean(totals)
    mean_inf = statistics.fmean(infers)
    return {
        "samples": n,
        "client_observed_mean_ms": round(mean_total, 3),
        "server_inference_mean_ms": round(mean_inf, 3),
        "server_preprocess_mean_ms": round(statistics.fmean(pres), 3),
        "inference_share_of_request": round(mean_inf / mean_total, 3),
        "implied_max_rps_single_threaded": round(1000.0 / mean_total, 1),
        "implied_max_rps_if_only_inference": round(1000.0 / mean_inf, 1),
    }


def _bench(base: str, concurrency: int, duration: float) -> Dict:
    payload, lat, errs = _payload(), [], [0]
    stop = threading.Event()
    lock = threading.Lock()

    def worker():
        local = []
        with httpx.Client(timeout=30.0) as c:
            while not stop.is_set():
                t0 = time.perf_counter()
                try:
                    r = c.post(f"{base}/predict", json=payload)
                    if r.status_code == 200:
                        local.append((time.perf_counter() - t0) * 1000)
                    else:
                        errs[0] += 1
                except Exception:                # noqa: BLE001
                    errs[0] += 1
        with lock:
            lat.extend(local)

    ws = [threading.Thread(target=worker, daemon=True) for _ in range(concurrency)]
    for w in ws:
        w.start()
    time.sleep(1.0)
    with lock:
        lat.clear()
    t0 = time.perf_counter()
    time.sleep(duration)
    elapsed = time.perf_counter() - t0
    stop.set()
    for w in ws:
        w.join(timeout=20)
    return {"throughput_rps": round(len(lat) / elapsed, 1),
            "p50_ms": round(_percentile(lat, 0.5), 3),
            "p99_ms": round(_percentile(lat, 0.99), 3),
            "errors": errs[0]}


def cpu_budget(concurrency: int, workers: int) -> Dict:
    """Who is competing for cores during H3.

    This matters because the test co-locates the load generator with the service. On a
    laptop that is not a neutral setup: N server processes plus N client threads plus
    the single-worker service already running can exceed the core count, at which point
    adding workers subtracts throughput. Reporting the budget is what separates
    "horizontal scaling does not work" from "this machine ran out of cores".
    """
    cores = os.cpu_count() or 1
    return {
        "physical_cores_reported": cores,
        "server_processes_under_test": workers,
        "client_threads": concurrency,
        "other_service_processes": 1,
        "demand": workers + concurrency + 1,
        "oversubscribed": (workers + concurrency + 1) > cores,
    }


def test_worker_scaling(single_base: str, workers: int = 4,
                        port: int = 8299, duration: float = 6.0) -> Dict:
    """H3: start the SAME service with N uvicorn workers and measure.

    This is the hypothesis that is worth money, because if it holds it is also the
    remedy: the ceiling is per-process, so the deployment scales by adding processes
    (or replicas), not by tuning the model.
    """
    env = dict(os.environ, FORGE_TORCH_THREADS="1", FORGE_LOG_LEVEL="WARNING")
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "serve.app:app", "--host", "127.0.0.1",
         "--port", str(port), "--workers", str(workers), "--log-level", "warning"],
        cwd=str(ROOT), env=env,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        start_new_session=True)          # own process group, so we can kill the pool
    multi_base = f"http://127.0.0.1:{port}"
    try:
        for _ in range(90):
            try:
                if httpx.get(f"{multi_base}/health/ready", timeout=2).status_code == 200:
                    break
            except Exception:                    # noqa: BLE001
                pass
            time.sleep(1)
        else:
            return {"error": f"{workers}-worker service never became ready"}

        conc = 8
        budget = cpu_budget(conc, workers)
        one = _bench(single_base, conc, duration)
        many = _bench(multi_base, conc, duration)
        gain = many["throughput_rps"] / one["throughput_rps"]
        return {
            "hypothesis": "the throughput ceiling is per-process",
            "cpu_budget": budget,
            "concurrency": conc,
            "workers_1": one,
            f"workers_{workers}": many,
            "throughput_gain": round(gain, 2),
            "ideal_gain": workers,
            "scaling_efficiency": round(gain / workers, 3),
            "p99_change": round(many["p99_ms"] / one["p99_ms"], 2),
            "holds": gain > 1.5,
            "confounded_by_oversubscription": budget["oversubscribed"],
        }
    finally:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        proc.wait(timeout=20)


def main() -> int:
    base = os.environ.get("FORGE_BASE", "http://127.0.0.1:8200")
    loadtest_path = ROOT / "outputs" / "loadtest.json"
    if not loadtest_path.exists():
        print("run loadtest/harness.py first", file=sys.stderr)
        return 1
    loadtest = json.loads(loadtest_path.read_text())

    print("H1  is this a saturated queue?  (Little's Law: concurrency = rps x latency)")
    little = check_littles_law(loadtest)
    for r in little["rows"]:
        print(f"      conc {r['concurrency']:>3}  rps {r['throughput_rps']:>7,.0f}  "
              f"latency {r['mean_latency_ms']:>7.2f}ms  ->  predicted conc "
              f"{r['predicted_concurrency']:>6.2f}  err {r['relative_error'] * 100:>5.1f}%")
    print(f"      mean error {little['mean_relative_error'] * 100:.1f}%  "
          f"-> {'HOLDS' if little['holds'] else 'DOES NOT HOLD'}\n")

    print("H2  is the model the service rate?")
    rate = measure_service_rate(base)
    print(f"      one request end-to-end   {rate['client_observed_mean_ms']:.3f} ms")
    print(f"      of which model forward   {rate['server_inference_mean_ms']:.3f} ms "
          f"({rate['inference_share_of_request'] * 100:.0f}%)")
    print(f"      implied ceiling          {rate['implied_max_rps_single_threaded']:,.0f} rps")
    print(f"      measured peak            {loadtest['peak']['throughput_rps']:,.0f} rps\n")

    print("H3  is the ceiling per-process?  (starting a 4-worker instance)")
    scaling = test_worker_scaling(base)
    if "error" in scaling:
        print(f"      {scaling['error']}")
    else:
        b = scaling["cpu_budget"]
        print(f"      cores {b['physical_cores_reported']}, demand {b['demand']} "
              f"({b['server_processes_under_test']} server + {b['client_threads']} client "
              f"+ {b['other_service_processes']} idle) "
              f"-> {'OVERSUBSCRIBED' if b['oversubscribed'] else 'within budget'}")
        print(f"      1 worker   {scaling['workers_1']['throughput_rps']:>8,.0f} rps  "
              f"p99 {scaling['workers_1']['p99_ms']:>7.2f}ms")
        print(f"      4 workers  {scaling['workers_4']['throughput_rps']:>8,.0f} rps  "
              f"p99 {scaling['workers_4']['p99_ms']:>7.2f}ms")
        print(f"      gain {scaling['throughput_gain']}x of an ideal {scaling['ideal_gain']}x "
              f"({scaling['scaling_efficiency'] * 100:.0f}% efficiency) "
              f"-> {'HOLDS' if scaling['holds'] else 'DOES NOT HOLD'}")

    report = {"littles_law": little, "service_rate": rate, "worker_scaling": scaling,
              "measured_peak_rps": loadtest["peak"]["throughput_rps"]}
    dest = ROOT / "outputs" / "profile.json"
    dest.write_text(json.dumps(report, indent=2) + "\n")
    print(f"\n  wrote {dest.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
