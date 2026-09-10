"""Warm the model before it is declared ready.

The first inference through a TorchScript module is materially slower than the
thousandth: lazy kernel selection, allocator growth and CPU cache population all
happen once. If the service reports ready before that, the load balancer sends real
traffic into the slowest requests the process will ever serve - which is precisely
when an autoscaler is adding instances because latency looks bad, making it worse.

So warmup is part of readiness, not an optimisation.
"""

from __future__ import annotations

import logging
import time
from typing import Dict, List

import torch

log = logging.getLogger("forge.warmup")


def warm(module: torch.jit.ScriptModule, batch_sizes: List[int] | None = None,
         iterations: int = 12) -> Dict[str, object]:
    """Run inference at each batch size the service will actually see.

    Per batch size, because the allocator and kernel choice differ by shape: warming
    only at batch 1 leaves the first batch-32 request cold.
    """
    batch_sizes = batch_sizes or [1, 8, 32]
    timings: Dict[str, Dict[str, float]] = {}
    started = time.perf_counter()

    with torch.no_grad():
        for bs in batch_sizes:
            x = torch.zeros(bs, 1, 28, 28)
            samples = []
            for i in range(iterations):
                t0 = time.perf_counter()
                module(x)
                samples.append((time.perf_counter() - t0) * 1000)
            # The first iteration is the cold one; reporting both makes the size of
            # the effect visible instead of averaging it away.
            timings[f"batch_{bs}"] = {
                "first_ms": round(samples[0], 3),
                "warm_mean_ms": round(sum(samples[1:]) / max(1, len(samples) - 1), 3),
                "warm_min_ms": round(min(samples[1:]), 3),
                "cold_penalty_x": round(samples[0] / max(1e-9, min(samples[1:])), 2),
            }

    total = (time.perf_counter() - started) * 1000
    log.info("warmup complete in %.0fms: %s", total, timings)
    return {"total_ms": round(total, 1), "iterations": iterations, "by_batch": timings}
