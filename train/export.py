"""Export the trained model to a serving artifact, and verify the export.

The verification is the point. An export step that does not check its own output is a
step that can silently ship a broken model - TorchScript tracing in particular will
happily produce a graph that differs from the eager model if the forward pass has a
data-dependent branch, and nothing complains until production predictions are wrong.

So this file exports, then re-loads the exported artifact and asserts it agrees with
the eager model to within floating-point tolerance on real data. If it does not, the
export fails loudly rather than writing a file.

Run:  python3 train/export.py
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
_here = str(Path(__file__).resolve().parent)
sys.path[:] = [p for p in sys.path if p not in ("", _here)]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from train import data as data_mod                # noqa: E402
from train.model import build                     # noqa: E402

ARTIFACTS = ROOT / "artifacts"
AGREEMENT_TOLERANCE = 1e-4


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    manifest_path = ARTIFACTS / "manifest.json"
    manifest = json.loads(manifest_path.read_text())

    # Export on CPU regardless of what trained the model. The serving container has no
    # accelerator, and a graph traced on MPS or CUDA can carry device assumptions that
    # only fail when it is loaded somewhere else.
    device = torch.device("cpu")
    model = build().to(device)
    model.load_state_dict(torch.load(ARTIFACTS / "model.pt", map_location=device))
    model.eval()

    splits = data_mod.load_splits()
    sample_x, _ = splits.test.tensors
    example = sample_x[:8].to(device)

    # torch.jit.trace, not script: the forward pass is a straight-line sequence of
    # modules with no data-dependent control flow, so tracing captures it exactly and
    # avoids TorchScript's restrictions on the Python it will accept.
    with torch.no_grad():
        traced = torch.jit.trace(model, example)
    traced = torch.jit.optimize_for_inference(traced)

    scripted_path = ARTIFACTS / "model_traced.pt"
    traced.save(str(scripted_path))

    # --- the verification --------------------------------------------------
    reloaded = torch.jit.load(str(scripted_path), map_location=device)
    reloaded.eval()
    check = sample_x[:512].to(device)
    with torch.no_grad():
        eager_out = model(check)
        traced_out = reloaded(check)

    max_diff = (eager_out - traced_out).abs().max().item()
    agree = torch.equal(eager_out.argmax(1), traced_out.argmax(1))

    if max_diff > AGREEMENT_TOLERANCE or not agree:
        scripted_path.unlink(missing_ok=True)
        print(f"EXPORT FAILED: traced model disagrees with eager model "
              f"(max logit diff {max_diff:.2e}, labels agree={agree})", file=sys.stderr)
        return 1

    manifest["export"] = {
        "format": "torchscript-traced",
        "file": scripted_path.name,
        "sha256": sha256(scripted_path),
        "bytes": scripted_path.stat().st_size,
        "exported_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "verified_on_samples": int(check.shape[0]),
        "max_logit_difference": float(f"{max_diff:.3e}"),
        "argmax_agreement": agree,
        "normalisation": {"mean": data_mod.TRAIN_MEAN, "std": data_mod.TRAIN_STD},
        "classes": data_mod.CLASSES,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    print(f"  exported  {scripted_path.name}  "
          f"({scripted_path.stat().st_size / 1024:.0f} KB)")
    print(f"  verified  {check.shape[0]} samples, max logit diff {max_diff:.2e}, "
          f"argmax agreement {agree}")
    print(f"  sha256    {manifest['export']['sha256'][:16]}...")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
