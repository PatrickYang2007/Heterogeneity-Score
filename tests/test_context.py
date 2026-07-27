"""Tests for the context-width (crop) and center-anchored-pooling changes.

The finding these support: the label is a ~256 bp local property, so the model
should be able to (a) see a narrower window without regenerating data and (b) read
features directly over the labeled region instead of only a window-wide average.
"""
import torch
import pytest

from config import effective_width, max_blocks_for
from model import HeterogeneityScoreModel, center_crop, region_mask_channel


# ----------------------------- center_crop -----------------------------

def test_center_crop_keeps_the_middle():
    x = torch.arange(10).reshape(1, 10)
    assert center_crop(x, 4).tolist() == [[3, 4, 5, 6]]


def test_center_crop_is_a_noop_when_not_narrower():
    x = torch.arange(8).reshape(1, 8)
    assert torch.equal(center_crop(x, 8), x)
    assert torch.equal(center_crop(x, 99), x)
    assert torch.equal(center_crop(x, None), x)
    assert torch.equal(center_crop(x, 0), x)


def test_center_crop_works_on_channel_first_tensors():
    # predict.py crops a (batch, channels, length) tensor; the dataset crops a
    # (rows, length) one. Cropping is on the last axis either way.
    x = torch.arange(2 * 3 * 10).reshape(2, 3, 10)
    assert center_crop(x, 4).shape == (2, 3, 4)
    assert torch.equal(center_crop(x, 4), x[..., 3:7])


def test_cropped_window_still_centers_the_labeled_region():
    # widen_windows.py centers each window on its region, so the region must stay
    # centered after cropping -- otherwise the mask would mark the wrong bases.
    mask_full = region_mask_channel(2048, 16)
    assert torch.equal(center_crop(mask_full, 256), region_mask_channel(256, 16))


# ----------------------------- config helpers -----------------------------

@pytest.mark.parametrize("window,crop,expected", [
    (2048, None, 2048),      # no crop
    (2048, 256, 256),        # normal case
    (2048, 4096, 2048),      # crop wider than the window is a no-op
    (2048, 2048, 2048),
    (None, 256, None),       # raw 16 bp path has no window to crop
])
def test_effective_width(window, crop, expected):
    assert effective_width(window, crop) == expected


@pytest.mark.parametrize("length,pool,expected", [
    (2048, 2, 11),
    (256, 2, 8),
    (32, 2, 5),
    (16, 2, 4),
    (2048, 1, None),         # no pooling -> no depth ceiling
])
def test_max_blocks_for(length, pool, expected):
    assert max_blocks_for(length, pool) == expected


def test_max_blocks_matches_the_models_own_guard():
    # The guard in forward() and the ceiling train.py checks against must agree,
    # or train.py would accept a config that then dies mid-epoch.
    length, pool = 256, 2
    cap = max_blocks_for(length, pool)
    HeterogeneityScoreModel(dropout=0.0, num_filters=2, num_blocks=cap,
                            pool=pool).eval()(torch.randn(1, 4, length))
    too_deep = HeterogeneityScoreModel(dropout=0.0, num_filters=2,
                                       num_blocks=cap + 1, pool=pool).eval()
    with pytest.raises(ValueError, match="too short"):
        too_deep(torch.randn(1, 4, length))


# ----------------------------- center pooling -----------------------------

def test_center_pool_doubles_the_head_input():
    plain = HeterogeneityScoreModel(dropout=0.0, num_filters=8, num_blocks=3,
                                    pool=2)
    ctr = HeterogeneityScoreModel(dropout=0.0, num_filters=8, num_blocks=3,
                                  pool=2, center_pool=True)
    assert ctr.fc.in_features == 2 * plain.fc.in_features


def test_center_pool_forward_shape_and_bounds():
    m = HeterogeneityScoreModel(dropout=0.0, num_filters=8, num_blocks=3, pool=2,
                                center_pool=True, bounded=True).eval()
    out = m(torch.randn(4, 4, 256))
    assert out.shape == (4, 1)
    assert torch.all((out >= 0) & (out <= 1))


def test_center_pool_actually_reads_the_center():
    # Perturbing the middle of the window must move a center-pooled model's
    # output. This is the property the plain attention pool lacks: it averages
    # over every position, so the labeled region is 1/N of what the head sees.
    torch.manual_seed(0)
    m = HeterogeneityScoreModel(dropout=0.0, num_filters=8, num_blocks=3, pool=2,
                                center_pool=True, bounded=False).eval()
    x = torch.zeros(1, 4, 256)
    x[0, 0, :] = 1.0
    y = x.clone()
    y[0, :, 120:136] = torch.tensor([0.0, 0.0, 0.0, 1.0]).reshape(4, 1)
    with torch.no_grad():
        assert not torch.allclose(m(x), m(y), atol=1e-6)


def test_center_pool_checkpoints_do_not_load_into_a_plain_model():
    # Different head shape, so a mismatched --center-pool at eval time must fail
    # loudly rather than silently mis-scoring.
    ctr = HeterogeneityScoreModel(dropout=0.0, num_filters=8, num_blocks=3,
                                  pool=2, center_pool=True)
    plain = HeterogeneityScoreModel(dropout=0.0, num_filters=8, num_blocks=3,
                                    pool=2)
    with pytest.raises(RuntimeError):
        plain.load_state_dict(ctr.state_dict())


def test_plain_model_state_dict_is_unchanged_by_the_center_pool_option():
    # center_pool=False must leave the architecture byte-identical to before, so
    # every existing checkpoint still loads.
    m = HeterogeneityScoreModel(dropout=0.0, num_filters=32, num_blocks=3, pool=2)
    keys = set(m.state_dict())
    assert "fc.weight" in keys and m.fc.in_features == 32 * 4
