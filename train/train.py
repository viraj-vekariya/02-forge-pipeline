"""Training run. Seeded, checkpointed, early-stopped, and it writes a manifest.

The manifest is the part that matters to the pipeline. A trained model on disk with no
record of which data, which code, which seed and which metrics produced it is not
deployable - you cannot roll back to it, you cannot reproduce it, and you cannot tell
whether the thing currently serving traffic is the thing you tested. Every artifact
this writes carries its own provenance.

Run:  python3 train/train.py [--epochs 8] [--quick]
"""

from __future__ import annotations

import argparse
import json
import platform
import random
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent.parent
# Running this file as a script puts train/ on sys.path ahead of the repo root, where
# the name `train` then resolves to train.py instead of to the train package. Removing
# the script directory makes `python3 train/train.py` and `import train.train` agree.
_here = str(Path(__file__).resolve().parent)
sys.path[:] = [p for p in sys.path if p not in ("", _here)]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from train import data as data_mod           # noqa: E402
from train.model import build, parameter_count, pick_device  # noqa: E402

ARTIFACTS = ROOT / "artifacts"


def set_seed(seed: int) -> None:
    """Everything that draws a random number, in one place.

    Missing any one of these makes a run unreproducible in a way that is very hard to
    find later - the classic case being numpy seeded but torch not, so the weights
    differ while the data order matches.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       cwd=ROOT, stderr=subprocess.DEVNULL).decode().strip()
    except Exception:                          # noqa: BLE001 - not every checkout is a repo
        return "nogit"


@dataclass
class EpochRecord:
    epoch: int
    train_loss: float
    val_loss: float
    val_accuracy: float
    seconds: float
    lr: float


@torch.no_grad()
def evaluate_loader(model: nn.Module, loader, device, criterion) -> Dict[str, float]:
    model.eval()
    total_loss, correct, seen = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        total_loss += criterion(logits, y).item() * y.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        seen += y.size(0)
    return {"loss": total_loss / seen, "accuracy": correct / seen}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--seed", type=int, default=data_mod.SEED)
    ap.add_argument("--quick", action="store_true", help="2 epochs, for CI smoke")
    args = ap.parse_args()

    if args.quick:
        args.epochs = 2

    set_seed(args.seed)
    device = pick_device()
    splits = data_mod.load_splits()
    loaders = data_mod.loaders(splits, args.batch_size)

    model = build().to(device)
    params = parameter_count(model)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimiser = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    # OneCycle rather than a step schedule: it warms up and then anneals within a
    # single short run, which is what makes 8 epochs enough. A step schedule tuned for
    # 90 epochs would spend this entire run at its initial rate.
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimiser, max_lr=args.lr, epochs=args.epochs,
        steps_per_epoch=len(loaders["train"]), pct_start=0.3)

    print(f"device={device.type}  params={params['total']:,}  "
          f"train={len(splits.train):,}  val={len(splits.val):,}")

    history: List[EpochRecord] = []
    best_acc, best_state, stale = 0.0, None, 0
    run_started = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_started = time.time()
        running, seen = 0.0, 0
        for x, y in loaders["train"]:
            x, y = x.to(device), y.to(device)
            optimiser.zero_grad(set_to_none=True)   # cheaper than zeroing in place
            loss = criterion(model(x), y)
            loss.backward()
            # Gradient clipping: cheap insurance. A single bad batch producing an
            # exploding gradient would otherwise wreck a run that CI then reports as
            # a model regression, sending you looking in the wrong place.
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimiser.step()
            scheduler.step()
            running += loss.item() * y.size(0)
            seen += y.size(0)

        val = evaluate_loader(model, loaders["val"], device, criterion)
        rec = EpochRecord(epoch, running / seen, val["loss"], val["accuracy"],
                          round(time.time() - epoch_started, 2),
                          scheduler.get_last_lr()[0])
        history.append(rec)
        print(f"  epoch {epoch}/{args.epochs}  train_loss {rec.train_loss:.4f}  "
              f"val_loss {rec.val_loss:.4f}  val_acc {rec.val_accuracy:.4f}  "
              f"{rec.seconds:.1f}s")

        if val["accuracy"] > best_acc:
            best_acc, stale = val["accuracy"], 0
            # Copy to CPU: keeping the best state on an accelerator holds a second
            # full copy of the model in the memory the next epoch wants.
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= args.patience:
                print(f"  early stop: no val improvement in {args.patience} epochs")
                break

    assert best_state is not None
    model.load_state_dict(best_state)

    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    version = f"{time.strftime('%Y%m%d-%H%M%S')}-{git_sha()}"
    torch.save(best_state, ARTIFACTS / "model.pt")

    manifest = {
        "version": version,
        "git_sha": git_sha(),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "dataset": {"name": "FashionMNIST", "fingerprint": splits.fingerprint,
                    **splits.sizes()},
        "model": {"architecture": "ForgeCNN", **params},
        "hyperparameters": {k: v for k, v in vars(args).items() if k != "quick"},
        "environment": {"python": platform.python_version(), "torch": torch.__version__,
                        "device": device.type, "platform": platform.platform()},
        "history": [asdict(h) for h in history],
        "best_val_accuracy": round(best_acc, 4),
        "epochs_run": len(history),
        "total_seconds": round(time.time() - run_started, 1),
    }
    (ARTIFACTS / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    print(f"\n  best val accuracy {best_acc:.4f} after {len(history)} epochs")
    print(f"  wrote artifacts/model.pt + artifacts/manifest.json (version {version})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
