"""Tests for HeterogeneityScoreModel plumbing.

These don't test that the model *learns* (that's stochastic and slow); they pin
down the wiring that silently breaks training or checkpoint loading: output
shapes, the bounded/linear head, channel doubling, the configurable depth, the
depth-vs-length guard, and — critically after the block-loop refactor — that the
state_dict keys stay compatible with existing checkpoints.
"""
import torch
import pytest

from model import HeterogeneityScoreModel


def make_model(**kw):
    kw.setdefault("dropout", 0.0)
    return HeterogeneityScoreModel(**kw)


def test_forward_output_shape():
    model = make_model(num_filters=8, num_blocks=3, pool=2).eval()
    x = torch.randn(5, 4, 64)
    out = model(x)
    assert out.shape == (5, 1)


def test_bounded_output_in_unit_interval():
    model = make_model(num_filters=8, num_blocks=3, pool=2, bounded=True).eval()
    out = model(torch.randn(8, 4, 64))
    assert torch.all(out >= 0.0) and torch.all(out <= 1.0)


def test_unbounded_output_runs():
    # linear head: just has to run and give the right shape (value unconstrained)
    model = make_model(num_filters=8, num_blocks=3, pool=2, bounded=False).eval()
    out = model(torch.randn(8, 4, 64))
    assert out.shape == (8, 1)
    assert torch.isfinite(out).all()


@pytest.mark.parametrize("num_blocks", [1, 2, 3, 4])
def test_depth_is_configurable(num_blocks):
    model = make_model(num_filters=4, num_blocks=num_blocks, pool=2).eval()
    # final-block channels double each block: 4 * 2**(num_blocks-1)
    assert model.fc.in_features == 4 * (2 ** (num_blocks - 1))
    length = 2 ** num_blocks * 4  # comfortably above the pooling floor
    out = model(torch.randn(2, 4, length))
    assert out.shape == (2, 1)


def test_too_deep_for_length_raises():
    # 5 blocks with pool=2 needs length >= 32; 16 is too short -> clear error
    model = make_model(num_filters=4, num_blocks=5, pool=2).eval()
    with pytest.raises(ValueError, match="too short"):
        model(torch.randn(2, 4, 16))


def test_pool1_has_no_length_constraint():
    # pool=1 never shrinks length, so deep models work on short inputs
    model = make_model(num_filters=4, num_blocks=5, pool=1).eval()
    out = model(torch.randn(2, 4, 16))
    assert out.shape == (2, 1)


def test_default_state_dict_keys_are_backward_compatible():
    # The default 3-block/32-filter model must keep block1/block2/block3 naming
    # so previously trained checkpoints still load.
    model = make_model(num_filters=32, num_blocks=3, pool=2)
    keys = set(model.state_dict().keys())
    for expected in [
        "block1.0.weight", "block1.2.weight",   # BatchNorm, Conv of block 1
        "block2.2.weight", "block3.2.weight",   # Conv of blocks 2 and 3
        "pool.scores.0.weight", "fc.weight",
    ]:
        assert expected in keys


def test_state_dict_round_trip():
    a = make_model(num_filters=8, num_blocks=3, pool=2).eval()
    b = make_model(num_filters=8, num_blocks=3, pool=2).eval()
    b.load_state_dict(a.state_dict())          # must not raise (keys/shapes match)
    x = torch.randn(3, 4, 64)
    with torch.no_grad():
        assert torch.allclose(a(x), b(x), atol=1e-6)
