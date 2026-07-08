"""Attach the raw specificity-entropy label to the existing per-region splits.

The bedgraph `score` (0-1) that train.py currently regresses is a clipped,
inverted transform of an underlying continuous signal: it piles up at exactly
1.0 for ~41% of rows, which forces every anti-saturation hack in the pipeline
(sigmoid + TARGET_CLIP, spike downsampling, inverse-density weighting, output
bias seeding). The file data/specificity_entropy_weighted.npy is that underlying
signal, un-clipped and un-saturated -- a genome-wide, per-16bp-bin track over
chr1-22 + chrX + chrY (chrM excluded), in FASTA-index order, with floor(len/16)
bins per chromosome. Its length (193,016,852) matches that tiling exactly, and a
region-center -> bin mapping correlates with the current label at Spearman ~0.80
(negative: high raw = high specificity = LOW heterogeneity).

This script maps each row of data/{split}{suffix}.parquet to its genome bin,
reads the raw value, and writes data/{split}{suffix}_raw.parquet with the SAME
columns but `score` replaced by log1p(raw) (the raw signal is heavy-tailed, up to
~3000, so a log stabilizes the regression target). Writing it into the `score`
column means GenomicDataset needs no changes -- train.py just points at the _raw
files. The original label is kept as `orig_score` and the untransformed signal as
`raw` for diagnostics.

Note on direction: this predicts specificity (log1p of the raw signal) directly,
so the target runs OPPOSITE to "heterogeneity". Correlation magnitude is the
comparable metric against prior runs; flip the sign here if you want the
heterogeneity orientation.
"""
import argparse

import numpy as np
import pandas as pd

DATA_DIR = "data"
FAI_PATH = f"{DATA_DIR}/GRCh38.primary_assembly.genome.fa.fai"
SIGNAL_PATH = f"{DATA_DIR}/specificity_entropy_weighted.npy"
BIN_SIZE = 16
# The track's bin order: main chromosomes in FASTA-index order, chrM excluded.
MAIN_CHROMS = [f"chr{i}" for i in range(1, 23)] + ["chrX", "chrY"]


def build_bin_offsets(fai_path):
    """Cumulative per-chromosome bin offset into the flattened signal array.

    Each chromosome contributes floor(len / BIN_SIZE) bins; a position's global
    bin index is offset[chrom] + pos // BIN_SIZE. Order and floor() must match how
    the track was generated -- verified against the array length (sum of bins ==
    len(signal)) before trusting the labels.
    """
    fai = pd.read_csv(fai_path, sep="\t", header=None, usecols=[0, 1],
                      names=["chrom", "len"])
    fai = fai[fai.chrom.isin(MAIN_CHROMS)].set_index("chrom").reindex(MAIN_CHROMS)
    bins = (fai["len"] // BIN_SIZE).astype("int64")
    offset = bins.cumsum().shift(1).fillna(0).astype("int64")
    return offset, int(bins.sum())


def attach(split, suffix, signal, offset, n_bins):
    in_path = f"{DATA_DIR}/{split}{suffix}.parquet"
    out_path = f"{DATA_DIR}/{split}{suffix}_raw.parquet"
    df = pd.read_parquet(in_path)

    known = df.chrom.isin(offset.index)
    if not known.all():
        dropped = sorted(df.chrom[~known].unique())
        print(f"  {split}: dropping {int((~known).sum()):,} rows on off-track "
              f"chroms {dropped}")
        df = df[known].reset_index(drop=True)

    center = ((df.start + df.end) // 2).astype("int64")
    gbin = df.chrom.map(offset).astype("int64").to_numpy() + (center.to_numpy() // BIN_SIZE)
    # Clamp the last bin of each chromosome (floor() drops the final partial bin,
    # so a region whose center lands there would index one past the chrom block).
    gbin = np.clip(gbin, 0, n_bins - 1)

    raw = np.asarray(signal[gbin], dtype=np.float64)
    df["orig_score"] = df["score"]
    df["raw"] = raw.astype(np.float32)
    df["score"] = np.log1p(raw).astype(np.float32)   # regression target

    df.to_parquet(out_path, index=False)
    print(f"  {split}: {len(df):,} rows -> {out_path}")
    print(f"    log1p(raw)  min/mean/max = {df.score.min():.3f} / "
          f"{df.score.mean():.3f} / {df.score.max():.3f}")
    print(f"    raw==0 rows: {int((df.raw == 0).sum()):,} "
          f"({100 * float((df.raw == 0).mean()):.2f}%)")
    return df


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--suffix", default="_w2048",
                    help="split-file suffix to relabel (default: _w2048)")
    ap.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    args = ap.parse_args()

    offset, n_bins = build_bin_offsets(FAI_PATH)
    signal = np.load(SIGNAL_PATH, mmap_mode="r")
    if n_bins != signal.shape[0]:
        raise SystemExit(
            f"bin-count mismatch: fai gives {n_bins:,} bins but signal has "
            f"{signal.shape[0]:,}. The chrom order / floor() convention does not "
            f"match how the track was built -- refusing to write mislabeled data.")
    print(f"signal: {signal.shape[0]:,} bins  (matches fai tiling)")

    for split in args.splits:
        attach(split, args.suffix, signal, offset, n_bins)


if __name__ == "__main__":
    main()
