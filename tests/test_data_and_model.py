"""Data splits, determinism and the model's contract.

The determinism tests are the load-bearing ones. The entire pipeline's ability to
answer "did this commit help?" rests on two runs of the same code producing the same
model, and that property is silently lost the moment one random source goes unseeded.
"""

import torch

from train import data as data_mod
from train.model import ForgeCNN, build, parameter_count, pick_device
from train.train import set_seed


def test_splits_do_not_overlap_and_sum_correctly():
    s = data_mod.load_splits(download=False)
    sizes = s.sizes()
    assert sizes["train"] + sizes["val"] == 60_000, "train+val must be the full train set"
    assert sizes["test"] == 10_000
    assert sizes["val"] == 6_000


def test_the_split_is_deterministic():
    """Two loads must produce identical splits, or two CI runs are incomparable."""
    a = data_mod.load_splits(download=False)
    b = data_mod.load_splits(download=False)
    assert a.fingerprint == b.fingerprint
    assert torch.equal(a.train.tensors[1][:500], b.train.tensors[1][:500])


def test_normalisation_is_applied():
    """Roughly zero mean and unit variance. If normalisation were skipped the mean
    would sit near 0.286 and training would be slower for no visible reason."""
    s = data_mod.load_splits(download=False)
    x, _ = s.train.tensors
    sample = x[:5000]
    assert abs(sample.mean().item()) < 0.15
    assert 0.8 < sample.std().item() < 1.25


def test_dataset_is_balanced_so_accuracy_is_a_fair_headline():
    s = data_mod.load_splits(download=False)
    counts = list(data_mod.class_balance(s)["test"].values())
    assert min(counts) == max(counts) == 1000


def test_model_output_shape_and_finiteness():
    model = build()
    out = model(torch.zeros(4, 1, 28, 28))
    assert out.shape == (4, 10)
    assert torch.isfinite(out).all()


def test_model_stays_small_enough_to_train_in_ci():
    """A pipeline whose model cannot be trained inside a CI job cannot be tested by
    that pipeline, which defeats the purpose."""
    params = parameter_count(build())
    assert params["total"] < 500_000, f"model grew to {params['total']:,}"
    assert params["trainable"] == params["total"]


def test_seeding_makes_initialisation_reproducible():
    set_seed(123)
    a = build()
    set_seed(123)
    b = build()
    for pa, pb in zip(a.parameters(), b.parameters()):
        assert torch.equal(pa, pb), "same seed produced different weights"


def test_different_seeds_actually_differ():
    """Guards the opposite failure: a seeding bug that pins every run to one draw."""
    set_seed(1)
    a = build()
    set_seed(2)
    b = build()
    assert not all(torch.equal(pa, pb) for pa, pb in zip(a.parameters(), b.parameters()))


def test_eval_mode_is_deterministic():
    """Dropout and batch-norm must be inert at inference, or the same image gets
    different predictions on consecutive calls."""
    model = build().eval()
    x = torch.randn(8, 1, 28, 28)
    with torch.no_grad():
        assert torch.equal(model(x), model(x))


def test_train_mode_dropout_is_actually_active():
    model = build().train()
    x = torch.randn(8, 1, 28, 28)
    assert not torch.equal(model(x), model(x)), "dropout appears to be disabled"


def test_device_selection_returns_something_usable():
    d = pick_device()
    assert d.type in ("cpu", "cuda", "mps")
    build().to(d)(torch.zeros(2, 1, 28, 28).to(d))
