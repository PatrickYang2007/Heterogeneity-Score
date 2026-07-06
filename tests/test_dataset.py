"""Tests for GenomicDataset: parquet -> (one-hot tensor, score) with correct
shape and N handling. Uses a tiny temp parquet, no real data.
"""
import pandas as pd
import pytest

from model import GenomicDataset, region_mask_channel


@pytest.fixture
def tiny_parquet(tmp_path):
    path = tmp_path / "tiny.parquet"
    pd.DataFrame({
        "sequence": ["ACGT", "NNNN", "acgt"],   # mixed case + all-N row
        "score": [0.5, 0.2, 0.7],
    }).to_parquet(path)
    return str(path)


def test_length_and_targets(tiny_parquet):
    ds = GenomicDataset(tiny_parquet)
    assert len(ds) == 3
    _, y0 = ds[0]
    assert float(y0) == pytest.approx(0.5)


def test_item_shape_is_channels_first(tiny_parquet):
    ds = GenomicDataset(tiny_parquet)
    x, _ = ds[0]
    assert tuple(x.shape) == (4, 4)              # (channels=4, length=4)


def test_base_one_hot_columns(tiny_parquet):
    ds = GenomicDataset(tiny_parquet)
    x, _ = ds[0]                                  # "ACGT"
    assert x[:, 0].tolist() == [1, 0, 0, 0]       # A
    assert x[:, 3].tolist() == [0, 0, 0, 1]       # T


def test_all_n_row_is_zero(tiny_parquet):
    ds = GenomicDataset(tiny_parquet)
    x, _ = ds[1]                                  # "NNNN"
    assert x.sum().item() == 0.0


def test_lowercase_is_upper_cased(tiny_parquet):
    ds = GenomicDataset(tiny_parquet)
    x, _ = ds[2]                                  # "acgt" -> treated as ACGT
    assert x[:, 0].tolist() == [1, 0, 0, 0]


def test_region_mask_adds_fifth_channel(tiny_parquet):
    # region_mask=True appends a 5th channel marking the central region_width
    # positions; the four base channels are unchanged.
    ds = GenomicDataset(tiny_parquet, region_mask=True, region_width=2)
    x, _ = ds[0]                                  # "ACGT" -> length 4
    assert tuple(x.shape) == (5, 4)              # (4 base + 1 mask, length)
    assert x[:4, 0].tolist() == [1, 0, 0, 0]     # base channels intact (A)
    # central region_width=2 of length 4 -> positions [1, 3): cols 1 and 2 set
    assert x[4].tolist() == [0, 1, 1, 0]


def test_region_mask_channel_helper_centers_region():
    mask = region_mask_channel(length=10, region_width=4)
    assert tuple(mask.shape) == (1, 10)
    # central 4 of 10 -> positions [3, 7)
    assert mask[0].tolist() == [0, 0, 0, 1, 1, 1, 1, 0, 0, 0]
    assert mask.sum().item() == 4.0
