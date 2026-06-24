"""Testbed for the "summed-bin" idea.

The usual setup is one-region-one-score: a WINDOW-bp window centered on a single
16 bp region, labeled with that region's score. This script instead tiles each
chromosome into NON-overlapping WINDOW-bp bins:

    0-256, 256-512, 512-768, ...

For each bin it SUMS the scores of every 16 bp region whose center falls inside
it, and uses that sum as the label for the bin's 256 bp sequence. So one 256 bp
bin containing 16 regions becomes a single (sequence, summed_score) example.

Binning is O(n): each region computes its own bin id from its center
(center // WINDOW), then a groupby sums per bin. Input order does not matter and
there is no neighbor searching.

This is a separate, optional path. It writes data/{split}_agg{WINDOW}.parquet and
leaves the normal train/val/test{_w*}.parquet files untouched. Flip AGGREGATE in
train.py to train on these instead.

Note: the label is a SUM (range ~0..#regions_in_bin), so it is NOT bounded in
[0,1] like a per-region score. The model must be built with bounded=False so its
output is linear instead of sigmoid-squashed (train.py handles this when
AGGREGATE is on). No MACS2 peak filtering is applied here: the summation itself
downweights low-signal bins.
"""
import os
import pandas as pd
from pyfaidx import Fasta

from prepare_data import TRAIN_CHROMS, VAL_CHROMS, TEST_CHROMS

DATA_DIR = "data"
BEDGRAPH_PATH = f"{DATA_DIR}/entropy_specificity_onGreaterThan1_stitched_annotated_complete.bedgraph"
GENOME_PATH = f"{DATA_DIR}/GRCh38.primary_assembly.genome.fa"

# Bin width in bp. Each bin's label is the sum of the scores of the 16 bp regions
# whose center falls in [bin_start, bin_start + WINDOW).
WINDOW = 256


def extract_bin(chrom_seq, bin_start, window):
    """Return the WINDOW-bp sequence for a bin, padded with 'N' at chrom ends."""
    bin_end = bin_start + window
    left_pad = max(0, -bin_start)
    right_pad = max(0, bin_end - len(chrom_seq))
    core = chrom_seq[max(0, bin_start):min(len(chrom_seq), bin_end)]
    return "N" * left_pad + core + "N" * right_pad


def build_bins(bedgraph_path, genome_path, window):
    print("Loading bedgraph...")
    df = pd.read_csv(
        bedgraph_path,
        sep="\t",
        header=None,
        usecols=[0, 1, 2, 3],
        names=["chrom", "start", "end", "score"],
        dtype={"chrom": str, "start": float, "end": float, "score": float},
    )
    df["start"] = df["start"].round().astype(int)
    df["end"] = df["end"].round().astype(int)
    print(f"  {len(df):,} regions loaded")

    # Stamp each region with its bin id from its own center (one vectorized pass),
    # then sum scores per bin. No sorting or neighbor lookups needed.
    center = (df["start"] + df["end"]) // 2
    df["bin"] = (center // window).astype(int)
    agg = (
        df.groupby(["chrom", "bin"])
        .agg(score=("score", "sum"), n_regions=("score", "size"))
        .reset_index()
    )
    agg["start"] = (agg["bin"] * window).astype(int)
    agg["end"] = agg["start"] + window
    print(f"  {len(agg):,} bins with >=1 region "
          f"(mean {agg['n_regions'].mean():.1f} regions/bin, "
          f"summed score range {agg['score'].min():.2f}-{agg['score'].max():.2f})")

    print("Extracting bin sequences...")
    genome = Fasta(genome_path)
    sequences = pd.Series(index=agg.index, dtype=str)
    for chrom, group in agg.groupby("chrom"):
        if chrom not in genome:
            sequences[group.index] = "N" * window
            continue
        chrom_seq = genome[chrom][:].seq.upper()
        sequences[group.index] = [extract_bin(chrom_seq, s, window) for s in group["start"]]
    agg["sequence"] = sequences

    lengths = agg["sequence"].str.len()
    assert (lengths == window).all(), f"got widths {sorted(lengths.unique())[:5]}"
    return agg


def main():
    agg = build_bins(BEDGRAPH_PATH, GENOME_PATH, WINDOW)

    # n_regions is kept so you can later check whether the label is being driven
    # by how many regions a bin happened to contain, or filter sparse bins.
    cols = ["chrom", "start", "end", "sequence", "score", "n_regions"]
    splits = {
        "train": agg[agg["chrom"].isin(TRAIN_CHROMS)],
        "val": agg[agg["chrom"].isin(VAL_CHROMS)],
        "test": agg[~agg["chrom"].isin(TRAIN_CHROMS + VAL_CHROMS)],
    }
    for split, sdf in splits.items():
        out_path = f"{DATA_DIR}/{split}_agg{WINDOW}.parquet"
        sdf[cols].to_parquet(out_path, index=False)
        print(f"[{split}] {len(sdf):,} bins -> {out_path}")
    print("Done.")


if __name__ == "__main__":
    main()
