"""Versioned model loading, with rollback.

The pipeline's whole promise is that a bad model can be undone. That requires three
things this class provides and a plain `torch.load` does not:

* **Identity.** Every loaded model carries the manifest that produced it - dataset
  fingerprint, git sha, hyperparameters, test accuracy. "Which model is serving?" must
  have an exact answer, not a filename.
* **Verification before activation.** The artifact's sha256 is checked against the
  manifest before it is allowed to serve. A truncated download or a half-written file
  should fail at load, not at the first request.
* **Atomic swap with rollback.** The new model is loaded and smoke-tested fully before
  it replaces the live one. If anything fails the previous model keeps serving, which
  is the difference between a failed deploy and an outage.
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import torch

log = logging.getLogger("forge.registry")


@dataclass
class LoadedModel:
    version: str
    module: torch.jit.ScriptModule
    manifest: Dict
    temperature: float
    classes: List[str]
    mean: float
    std: float
    loaded_at: float
    sha256: str

    def info(self) -> Dict[str, object]:
        return {
            "version": self.version,
            "git_sha": self.manifest.get("git_sha"),
            "architecture": self.manifest.get("model", {}).get("architecture"),
            "parameters": self.manifest.get("model", {}).get("total"),
            "test_accuracy": self.manifest.get("evaluation", {}).get("test_accuracy"),
            "temperature": self.temperature,
            "dataset_fingerprint": self.manifest.get("dataset", {}).get("fingerprint"),
            "sha256": self.sha256[:16],
            "loaded_at": round(self.loaded_at, 3),
            "age_sec": round(time.time() - self.loaded_at, 1),
        }


class ModelRegistry:
    def __init__(self, artifacts_dir: Path, verify_checksum: bool = True) -> None:
        self.dir = Path(artifacts_dir)
        self.verify_checksum = verify_checksum
        self._current: Optional[LoadedModel] = None
        self._previous: Optional[LoadedModel] = None
        self._lock = threading.Lock()
        self.load_count = 0
        self.rollback_count = 0

    # -- helpers -------------------------------------------------------------

    @staticmethod
    def _sha256(path: Path) -> str:
        h = hashlib.sha256()
        with path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()

    def _build(self) -> LoadedModel:
        manifest_path = self.dir / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"no manifest at {manifest_path}; run the training job")
        manifest = json.loads(manifest_path.read_text())

        export = manifest.get("export")
        if not export:
            raise ValueError("manifest has no export section; run train/export.py")

        artifact = self.dir / export["file"]
        if not artifact.exists():
            raise FileNotFoundError(f"manifest references {artifact} which does not exist")

        digest = self._sha256(artifact)
        if self.verify_checksum and digest != export["sha256"]:
            # A mismatch means the file on disk is not the file that was evaluated.
            # Serving it would mean the reported accuracy describes a different model.
            raise ValueError(
                f"checksum mismatch for {artifact.name}: manifest says "
                f"{export['sha256'][:16]}..., file is {digest[:16]}...")

        module = torch.jit.load(str(artifact), map_location="cpu")
        module.eval()

        norm = export.get("normalisation", {})
        return LoadedModel(
            version=manifest["version"],
            module=module,
            manifest=manifest,
            temperature=float(manifest.get("evaluation", {}).get("temperature", 1.0)),
            classes=export.get("classes", []),
            mean=float(norm.get("mean", 0.0)),
            std=float(norm.get("std", 1.0)),
            loaded_at=time.time(),
            sha256=digest,
        )

    @staticmethod
    def _smoke_test(model: LoadedModel) -> None:
        """Prove the model can actually infer before it is allowed to serve traffic.

        Loading successfully is not the same as working: a shape mismatch, a missing
        operator on this platform, or a corrupted graph all surface on the first
        forward pass, and the first forward pass should be ours, not a user's.
        """
        with torch.no_grad():
            out = model.module(torch.zeros(2, 1, 28, 28))
        if out.shape != (2, len(model.classes)):
            raise ValueError(f"smoke test: expected (2,{len(model.classes)}), got {tuple(out.shape)}")
        if not torch.isfinite(out).all():
            raise ValueError("smoke test: model produced non-finite logits")

    # -- public --------------------------------------------------------------

    def load(self) -> LoadedModel:
        """Load and activate. On any failure the previous model keeps serving."""
        candidate = self._build()
        self._smoke_test(candidate)
        with self._lock:
            # Swap only after the candidate is fully built AND smoke-tested. Assigning
            # first and validating after would leave a broken model live for the
            # duration of the validation.
            self._previous = self._current
            self._current = candidate
            self.load_count += 1
        log.info("activated model %s (acc=%s)", candidate.version,
                 candidate.manifest.get("evaluation", {}).get("test_accuracy"))
        return candidate

    def rollback(self) -> Optional[LoadedModel]:
        with self._lock:
            if self._previous is None:
                return None
            self._current, self._previous = self._previous, self._current
            self.rollback_count += 1
        log.warning("rolled back to model %s", self._current.version)
        return self._current

    @property
    def current(self) -> Optional[LoadedModel]:
        return self._current

    def require(self) -> LoadedModel:
        model = self._current
        if model is None:
            raise RuntimeError("no model loaded")
        return model

    def state(self) -> Dict[str, object]:
        return {
            "current": self._current.info() if self._current else None,
            "previous": self._previous.info() if self._previous else None,
            "loads": self.load_count,
            "rollbacks": self.rollback_count,
            "rollback_available": self._previous is not None,
        }
