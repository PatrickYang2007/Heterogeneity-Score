"""Tests for per-base sequence attributions (src/attribute.py).

The finding these support: attribution on this model is easy to get silently
wrong in two places -- Captum has no DeepLIFT rule for GELU and skips it without
warning, and the sigmoid head rescales every gradient-based attribution by a
per-window constant. These lock down the pieces that make both correct, plus the
exactness of ISM, which is the ground truth everything else is judged against.
"""
from collections import Counter

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import pytest

from attribute import (dinuc_shuffle, shuffled_references, LogitView,
                       register_gelu_rule, build_inputs, ism_attributions,
                       gradxinput_attributions, project)
from model import HeterogeneityScoreModel


def doublets(a):
    return Counter(zip(np.asarray(a)[:-1].tolist(), np.asarray(a)[1:].tolist()))


# --------------------------- dinucleotide shuffle ---------------------------

def test_dinuc_shuffle_preserves_doublet_counts():
    # The whole point of this reference: same dinucleotide (hence GC, hence CpG)
    # content, different arrangement. A composition-preserving null is what makes
    # the attributions mean "beyond composition" -- and composition alone already
    # gets Pearson ~0.60 on this task, so the distinction is the entire question.
    rng = np.random.default_rng(0)
    seq = rng.integers(0, 4, size=400)
    for row in dinuc_shuffle(seq, 5, rng):
        assert doublets(row) == doublets(seq)


def test_dinuc_shuffle_preserves_length_and_composition():
    rng = np.random.default_rng(1)
    seq = rng.integers(0, 4, size=200)
    shuf = dinuc_shuffle(seq, 4, rng)
    assert shuf.shape == (4, 200)
    for row in shuf:
        assert np.array_equal(np.bincount(row, minlength=4),
                              np.bincount(seq, minlength=4))


def test_dinuc_shuffle_actually_permutes_and_varies():
    # A "shuffle" that returns the input, or the same output every time, would
    # pass the counts test above while making every attribution meaningless.
    rng = np.random.default_rng(2)
    seq = rng.integers(0, 4, size=300)
    shuf = dinuc_shuffle(seq, 6, rng)
    assert not any(np.array_equal(r, seq) for r in shuf)
    assert len({tuple(r) for r in shuf}) == 6


def test_dinuc_shuffle_handles_a_single_repeated_base():
    # Degenerate graph: one vertex, all self-edges. The Eulerian walk must still
    # terminate and return the right length rather than stranding itself.
    rng = np.random.default_rng(3)
    seq = np.zeros(50, dtype=int)
    shuf = dinuc_shuffle(seq, 2, rng)
    assert shuf.shape == (2, 50)
    assert (shuf == 0).all()


def test_shuffled_references_are_valid_one_hot_and_keep_Ns():
    # N is encoded as an all-zero column. It must stay all-zero in the reference
    # so it contributes no delta and therefore collects no attribution -- an N
    # that turned into a real base in the reference would invent signal.
    rng = np.random.default_rng(4)
    onehot = np.zeros((4, 80), dtype=np.float32)
    onehot[rng.integers(0, 4, 80), np.arange(80)] = 1.0
    onehot[:, 7] = 0.0
    refs = shuffled_references(onehot, 5, rng)

    assert refs.shape == (5, 4, 80)
    assert refs[:, :, 7].sum() == 0
    real = np.arange(80) != 7
    assert (refs[:, :, real].sum(axis=1) == 1).all()


# --------------------------------- LogitView --------------------------------

def _model(**kw):
    torch.manual_seed(0)
    kw.setdefault("num_blocks", 2)
    kw.setdefault("num_filters", 8)
    return HeterogeneityScoreModel(dropout=0.0, in_channels=4, pool=2, **kw).eval()


def test_logitview_is_the_same_model_without_the_sigmoid():
    m = _model(bounded=True)
    x = torch.randn(4, 4, 64)
    with torch.no_grad():
        assert torch.allclose(torch.sigmoid(LogitView(m)(x)), m(x), atol=1e-6)


def test_logitview_is_a_noop_for_a_linear_head():
    # The raw-label / summed-bin models are already unbounded; wrapping them must
    # change nothing at all.
    m = _model(bounded=False)
    x = torch.randn(4, 4, 64)
    with torch.no_grad():
        assert torch.equal(LogitView(m)(x), m(x))


def test_logitview_handles_the_center_pool_head():
    # center_pool adds final_norm and doubles the head width; LogitView rebuilds
    # the forward pass by hand, so it has to reproduce both.
    m = _model(bounded=True, center_pool=True)
    x = torch.randn(4, 4, 64)
    with torch.no_grad():
        assert torch.allclose(torch.sigmoid(LogitView(m)(x)), m(x), atol=1e-6)


def test_sigmoid_only_rescales_gradient_attributions():
    # Documented behavior, and the reason --keep-sigmoid is a warning and not an
    # error: the sigmoid is a monotone scalar map at the very end, so it
    # multiplies a whole window's attributions by one constant. The SHAPE of the
    # profile is untouched (r == 1); only the scale moves, which is what makes
    # magnitudes non-comparable ACROSS windows.
    m = _model(bounded=True)
    x = torch.randn(4, 96).abs()
    a = gradxinput_attributions(m, x)[1]
    b = gradxinput_attributions(LogitView(m), x)[1]
    assert np.corrcoef(a, b)[0, 1] == pytest.approx(1.0, abs=1e-6)
    ratio = b[np.abs(a) > 1e-12] / a[np.abs(a) > 1e-12]
    assert ratio.std() / np.abs(ratio.mean()) < 1e-4


# ------------------------------- the GELU rule ------------------------------

def test_register_gelu_rule_installs_the_rescale_rule():
    # Captum 0.7.0 ships no GELU entry and `_can_register_hook` silently skips
    # unlisted modules, so without this every BatchNorm->GELU->Conv block would
    # fall back to plain gradients while still being reported as DeepLIFT.
    from captum.attr._core import deep_lift as dl
    dl.SUPPORTED_NON_LINEAR.pop(nn.GELU, None)
    assert register_gelu_rule()
    assert dl.SUPPORTED_NON_LINEAR[nn.GELU] is dl.nonlinear


def test_register_gelu_rule_is_idempotent():
    from captum.attr._core import deep_lift as dl
    register_gelu_rule()
    before = dict(dl.SUPPORTED_NON_LINEAR)
    register_gelu_rule()
    assert dl.SUPPORTED_NON_LINEAR == before


# ---------------------------------- ISM -------------------------------------

class LinearScorer(nn.Module):
    """f(x) = sum_i w[base_i, i]. Its exact per-base contribution is known, so
    ISM has a closed-form answer to be checked against."""

    def __init__(self, w):
        super().__init__()
        self.w = nn.Parameter(torch.tensor(w, dtype=torch.float32),
                              requires_grad=False)

    def forward(self, x):
        return (x[:, :4] * self.w).sum(dim=(1, 2), keepdim=False).unsqueeze(-1)


def test_ism_recovers_exact_contributions_on_a_known_linear_model():
    rng = np.random.default_rng(5)
    L = 12
    w = rng.normal(size=(4, L)).astype(np.float32)
    model = LinearScorer(w).eval()

    x = torch.zeros(4, L)
    x[torch.from_numpy(rng.integers(0, 4, L)), torch.arange(L)] = 1.0

    hyp = ism_attributions(model, x, batch_size=7)   # batch_size < 4*L on purpose
    assert np.allclose(hyp, w - w.mean(axis=0, keepdims=True), atol=1e-5)


def test_ism_hypothetical_is_mean_centered_per_position():
    # Centering is what makes ISM comparable to DeepLIFT/SHAP hypothetical
    # contributions and removes the constant f(wildtype) offset.
    m = _model(bounded=True)
    x = torch.zeros(4, 64)
    x[torch.randint(0, 4, (64,)), torch.arange(64)] = 1.0
    hyp = ism_attributions(m, x)
    assert np.allclose(hyp.mean(axis=0), 0.0, atol=1e-6)


def test_ism_leaves_the_region_mask_channel_alone():
    # Channel 5 is positional metadata, not sequence. Mutating it would measure
    # something with no genomic meaning, so ISM must only touch channels 0-3.
    seen = []

    class Recorder(nn.Module):
        def forward(self, x):
            seen.append(x[:, 4].clone())
            return x[:, :4].sum(dim=(1, 2), keepdim=False).unsqueeze(-1)

    L = 16
    x = torch.zeros(5, L)
    x[torch.randint(0, 4, (L,)), torch.arange(L)] = 1.0
    x[4, 6:10] = 1.0
    ism_attributions(Recorder().eval(), x)
    mask = torch.cat(seen)
    assert (mask == x[4].expand_as(mask)).all()


# ------------------------------ projection / io -----------------------------

def test_project_reads_off_the_observed_base():
    hyp = np.arange(12, dtype=np.float32).reshape(4, 3)
    onehot = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1], [0, 0, 0]], dtype=np.float32)
    assert project(hyp, onehot).tolist() == [3.0, 1.0, 8.0]


def test_project_gives_zero_at_N_positions():
    hyp = np.ones((4, 5), dtype=np.float32)
    onehot = np.ones((4, 5), dtype=np.float32)
    onehot[:, 2] = 0.0
    assert project(hyp, onehot)[2] == 0.0


def test_build_inputs_crops_and_masks_like_the_dataset():
    # Must match GenomicDataset exactly or the attributions describe a different
    # input than the one the model was trained on.
    df = pd.DataFrame({"sequence": ["A" * 8 + "CGTACGTA" + "T" * 8]})
    x = build_inputs(df, crop=8, region_mask=True, region_width=4)
    assert x.shape == (1, 5, 8)
    assert "".join("ACGT"[i] for i in x[0, :4].argmax(0).tolist()) == "CGTACGTA"
    assert x[0, 4].tolist() == [0, 0, 1, 1, 1, 1, 0, 0]


def test_build_inputs_encodes_N_as_an_all_zero_column():
    df = pd.DataFrame({"sequence": ["ACNGT"]})
    x = build_inputs(df, crop=None, region_mask=False, region_width=4)
    assert x[0, :, 2].sum() == 0
    assert x[0, :, 0].tolist() == [1, 0, 0, 0]
