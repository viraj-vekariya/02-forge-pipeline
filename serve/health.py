"""Three distinct health signals, because orchestrators use them for opposite things.

* **live**    - is the process alive? A failing liveness probe gets the container
                KILLED and restarted. It must therefore never depend on a model, a
                disk or a downstream service; if it did, a slow dependency would cause
                a restart loop that guarantees the outage it was meant to prevent.
* **ready**   - can this instance serve traffic right now? A failing readiness probe
                removes it from the load balancer WITHOUT killing it. This is where
                "model loaded and warmed" belongs.
* **startup** - has the process finished booting? Suppresses liveness during a slow
                start so a model that takes 20 seconds to load is not repeatedly
                killed at 10.

Collapsing these into one endpoint is the most common way a deploy turns into a
restart loop.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass
class HealthState:
    started_at: float = field(default_factory=time.time)
    model_loaded: bool = False
    warmed: bool = False
    last_error: str = ""
    warmup_report: Optional[Dict] = None

    def live(self) -> Dict[str, object]:
        return {"status": "alive", "uptime_sec": round(time.time() - self.started_at, 1)}

    def ready(self) -> Dict[str, object]:
        ok = self.model_loaded and self.warmed
        return {
            "status": "ready" if ok else "not_ready",
            "ready": ok,
            "model_loaded": self.model_loaded,
            "warmed": self.warmed,
            "last_error": self.last_error,
        }

    def startup(self) -> Dict[str, object]:
        return {"status": "started" if self.model_loaded else "starting",
                "model_loaded": self.model_loaded,
                "elapsed_sec": round(time.time() - self.started_at, 1)}
