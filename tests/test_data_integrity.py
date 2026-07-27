"""Tests for the data-integrity fixes in prepare_data.py.

These cover the two bugs that silently corrupted the training distribution:
annotation-duplicated regions being kept as independent rows, and the MACS2
peak filter running on a mis-sorted track and emitting degenerate whole-arm
"peaks". All synthetic -- no genome or bedgraph file needed.
"""
import pandas as pd
import pytest

from prepare_data import dedup_regions, filter_by_macs2_peaks, check_peaks_sane


def frame(rows):
    return pd.DataFrame(rows, columns=["chrom", "start", "end", "score"])


@pytest.fixture
def write_peaks(tmp_path):
    """Write a narrowPeak file (with MACS2's track header) and return its path."""
    def _write(peaks):
        path = tmp_path / "peaks.bed"
        with open(path, "w") as f:
            f.write('track type=narrowPeak name="peaks"\n')
            for _, r in peaks.iterrows():
                f.write(f"{r.chrom}\t{r.start}\t{r.end}\tpeak\t10\t.\t0\t0\t0\t0\n")
        return str(path)
    return _write


def test_dedup_collapses_annotation_duplicates():
    # Same interval emitted twice (two overlapping genes), same score.
    df = frame([
        ("chr1", 100, 116, 1.0),
        ("chr1", 100, 116, 1.0),
        ("chr1", 116, 132, 0.5),
    ])
    out = dedup_regions(df)
    assert len(out) == 2
    assert list(out.start) == [100, 116]
    assert list(out.score) == [1.0, 0.5]


def test_dedup_is_a_noop_when_already_unique():
    df = frame([("chr1", 100, 116, 1.0), ("chr1", 116, 132, 0.5)])
    assert len(dedup_regions(df)) == 2


def test_dedup_keeps_same_coords_on_different_chroms():
    df = frame([("chr1", 100, 116, 1.0), ("chr2", 100, 116, 0.5)])
    assert len(dedup_regions(df)) == 2


def test_dedup_refuses_conflicting_scores():
    # If duplicates disagree they are not redundant copies, so silently keeping
    # one would fabricate a label. Must raise instead of guessing.
    df = frame([("chr1", 100, 116, 1.0), ("chr1", 100, 116, 0.2)])
    with pytest.raises(ValueError, match="more than one distinct"):
        dedup_regions(df)


def test_peak_filter_keeps_only_overlapping_regions(write_peaks):
    df = frame([
        ("chr1", 100, 116, 1.0),   # inside the peak
        ("chr1", 900, 916, 1.0),   # outside
        ("chr2", 100, 116, 1.0),   # chromosome has no peaks
    ])
    peaks = pd.DataFrame({"chrom": ["chr1"], "start": [50], "end": [200]})
    out = filter_by_macs2_peaks(df, write_peaks(peaks))
    assert list(out.start) == [100]


def test_peak_filter_handles_nested_and_overlapping_peaks(write_peaks):
    # The real peaks.bed had overlapping and fully nested intervals; merging must
    # not lose a region that only overlaps the enclosing one.
    df = frame([("chr1", 10, 26, 1.0), ("chr1", 500, 516, 1.0),
                ("chr1", 5000, 5016, 1.0)])
    peaks = pd.DataFrame({"chrom": ["chr1"] * 3,
                          "start": [0, 100, 200],
                          "end": [1000, 300, 250]})   # nested inside the first
    out = filter_by_macs2_peaks(df, write_peaks(peaks))
    assert list(out.start) == [10, 500]


def test_peak_filter_boundary_is_half_open(write_peaks):
    # A region touching a peak only at the boundary does not overlap it.
    df = frame([("chr1", 200, 216, 1.0), ("chr1", 84, 100, 1.0)])
    peaks = pd.DataFrame({"chrom": ["chr1"], "start": [100], "end": [200]})
    out = filter_by_macs2_peaks(df, write_peaks(peaks))
    assert len(out) == 0


def test_check_peaks_sane_flags_the_degenerate_call():
    # The shape of the real failure: a handful of peaks, one spanning 203 Mb.
    bad = pd.DataFrame({"chrom": ["chr2"], "start": [667440], "end": [203739568]})
    assert check_peaks_sane(bad) is False


def test_check_peaks_sane_accepts_a_normal_call():
    good = pd.DataFrame({"chrom": ["chr1"] * 2000,
                         "start": range(0, 2000 * 10_000, 10_000),
                         "end": range(1_000, 2000 * 10_000 + 1_000, 10_000)})
    assert check_peaks_sane(good) is True
