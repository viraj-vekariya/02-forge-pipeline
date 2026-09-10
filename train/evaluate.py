"""Holdout evaluation and calibration. The test set is touched here and nowhere else.

Accuracy alone is not enough to decide whether a model is fit to serve, for two
reasons this file addresses:

1. **Which errors.** A confusion matrix says whether the model is confusing shirts with
   coats (expected, and mostly harmless) or shirts with sandals (a signal that
   something is broken). CI can gate on that; it cannot gate on a single number.

2. **Whether its confidence means anything.** A network trained with cross-entropy is
   usually over-confident: it says 0.95 and is right 0.85 of the time. Anything that
   routes on confidence - an abstention gate, a human-review queue, a fallback - is
   built on a lie unless the model is calibrated. Temperature scaling fixes most of it
   with a single scalar fitted on the validation set, never on test.

Run:  python3 train/evaluate.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
_here = str(Path(__file__).resolve().parent)
sys.path[:] = [p for p in sys.path if p not in ("", _here)]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from train import data as data_mod                       # noqa: E402
from train.model import build, pick_device               # noqa: E402

ARTIFACTS = ROOT / "artifacts"
OUTPUTS = ROOT / "outputs"


@torch.no_grad()
def collect_logits(model: nn.Module, loader, device) -> Tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    logits, labels = [], []
    for x, y in loader:
        logits.append(model(x.to(device)).cpu())
        labels.append(y)
    return torch.cat(logits), torch.cat(labels)


def fit_temperature(logits: torch.Tensor, labels: torch.Tensor,
                    max_iter: int = 200) -> float:
    """One scalar T, fitted by minimising NLL on the VALIDATION set.

    Fitting on test would be selecting a hyperparameter on the data used to report the
    result, which is the definition of a leak. LBFGS because the problem is
    one-dimensional, smooth and convex - it converges in a handful of iterations where
    SGD would need a learning rate nobody wants to tune.
    """
    log_t = torch.zeros(1, requires_grad=True)     # optimise log T so T stays positive
    optimiser = torch.optim.LBFGS([log_t], lr=0.1, max_iter=max_iter)
    criterion = nn.CrossEntropyLoss()

    def closure():
        optimiser.zero_grad()
        loss = criterion(logits / log_t.exp(), labels)
        loss.backward()
        return loss

    optimiser.step(closure)
    return float(log_t.exp().item())


def expected_calibration_error(probs: torch.Tensor, labels: torch.Tensor,
                               bins: int = 15) -> float:
    """ECE: average gap between confidence and accuracy, weighted by bin population.

    Equal-width bins, which is the standard formulation. It under-weights the
    high-confidence region where most predictions actually live, so ECE is a summary
    and the reliability table below it is the real evidence.
    """
    confidence, prediction = probs.max(dim=1)
    correct = prediction.eq(labels).float()
    edges = torch.linspace(0, 1, bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (confidence > lo) & (confidence <= hi)
        if mask.sum() == 0:
            continue
        ece += (mask.float().mean() * (correct[mask].mean() - confidence[mask].mean()).abs()).item()
    return ece


def reliability_table(probs: torch.Tensor, labels: torch.Tensor,
                      bins: int = 10) -> List[Dict[str, float]]:
    confidence, prediction = probs.max(dim=1)
    correct = prediction.eq(labels).float()
    edges = torch.linspace(0, 1, bins + 1)
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (confidence > lo) & (confidence <= hi)
        if mask.sum() == 0:
            continue
        rows.append({"bin": f"{lo:.1f}-{hi:.1f}", "n": int(mask.sum()),
                     "confidence": round(confidence[mask].mean().item(), 4),
                     "accuracy": round(correct[mask].mean().item(), 4)})
    return rows


def per_class_metrics(pred: torch.Tensor, labels: torch.Tensor) -> List[Dict[str, object]]:
    out = []
    for i, name in enumerate(data_mod.CLASSES):
        tp = int(((pred == i) & (labels == i)).sum())
        fp = int(((pred == i) & (labels != i)).sum())
        fn = int(((pred != i) & (labels == i)).sum())
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        out.append({"class": name, "support": tp + fn, "precision": round(precision, 4),
                    "recall": round(recall, 4), "f1": round(f1, 4)})
    return out


def top_confusions(pred: torch.Tensor, labels: torch.Tensor, k: int = 5):
    n = len(data_mod.CLASSES)
    matrix = np.zeros((n, n), dtype=int)
    for t, p in zip(labels.tolist(), pred.tolist()):
        matrix[t][p] += 1
    pairs = [(matrix[i][j], data_mod.CLASSES[i], data_mod.CLASSES[j])
             for i in range(n) for j in range(n) if i != j]
    pairs.sort(reverse=True)
    return ([{"true": t, "predicted": p, "count": int(c)} for c, t, p in pairs[:k]],
            matrix.tolist())


def main() -> int:
    device = pick_device()
    splits = data_mod.load_splits()
    loaders = data_mod.loaders(splits)

    model = build().to(device)
    model.load_state_dict(torch.load(ARTIFACTS / "model.pt", map_location=device))

    val_logits, val_labels = collect_logits(model, loaders["val"], device)
    test_logits, test_labels = collect_logits(model, loaders["test"], device)

    temperature = fit_temperature(val_logits, val_labels)

    raw = F.softmax(test_logits, dim=1)
    calibrated = F.softmax(test_logits / temperature, dim=1)
    pred = test_logits.argmax(1)

    accuracy = pred.eq(test_labels).float().mean().item()
    ece_raw = expected_calibration_error(raw, test_labels)
    ece_cal = expected_calibration_error(calibrated, test_labels)
    confusions, matrix = top_confusions(pred, test_labels)

    report = {
        "test_accuracy": round(accuracy, 4),
        "test_size": int(len(test_labels)),
        "temperature": round(temperature, 4),
        "ece_raw": round(ece_raw, 4),
        "ece_calibrated": round(ece_cal, 4),
        "ece_improvement": round(ece_raw - ece_cal, 4),
        "mean_confidence_raw": round(raw.max(1).values.mean().item(), 4),
        "mean_confidence_calibrated": round(calibrated.max(1).values.mean().item(), 4),
        "per_class": per_class_metrics(pred, test_labels),
        "top_confusions": confusions,
        "reliability_calibrated": reliability_table(calibrated, test_labels),
        "confusion_matrix": matrix,
        "classes": data_mod.CLASSES,
    }

    OUTPUTS.mkdir(parents=True, exist_ok=True)
    (OUTPUTS / "evaluation.json").write_text(json.dumps(report, indent=2) + "\n")

    # Temperature is needed at serving time, so it belongs in the manifest that
    # travels with the model - not in a report a human reads and then forgets to copy.
    manifest_path = ARTIFACTS / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["evaluation"] = {k: report[k] for k in
                              ("test_accuracy", "temperature", "ece_raw", "ece_calibrated")}
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    print(f"  test accuracy      {accuracy:.4f} on {len(test_labels):,} held-out images")
    print(f"  temperature        {temperature:.4f}")
    print(f"  ECE  {ece_raw:.4f} -> {ece_cal:.4f}  (improvement {ece_raw - ece_cal:+.4f})")
    print(f"  mean confidence    {raw.max(1).values.mean():.4f} -> "
          f"{calibrated.max(1).values.mean():.4f}")
    print("  worst confusions:")
    for c in confusions[:3]:
        print(f"    {c['true']:<12} -> {c['predicted']:<12} {c['count']:>4}")
    print(f"\n  wrote outputs/evaluation.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
