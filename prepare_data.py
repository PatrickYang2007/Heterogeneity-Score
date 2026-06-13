import pandas as pd
from pyfaidx import Fasta
import numpy as np
import subprocess
import os
from collections import defaultdict


def one_hot_encode(sequences):
    mapping = {
        "A": [1.0, 0.0, 0.0, 0.0],
        "C": [0.0, 1.0, 0.0, 0.0],
        "G": [0.0, 0.0, 1.0, 0.0],
        "T": [0.0, 0.0, 0.0, 1.0],
    }
    # N and other IUPAC ambiguity codes map to an all-zero ("no information")
    # vector instead of raising a KeyError.
    unknown = [0.0, 0.0, 0.0, 0.0]
    return np.array([[mapping.get(base, unknown) for base in seq] for seq in sequences])


def run_macs2_bdgpeakcall(bedgraph_path, out_dir, cutoff=2.0, min_length=200, max_gap=100):
    """Run MACS2 bdgpeakcall on a bedgraph and return path to the output peak BED file."""
    peak_path = os.path.join(out_dir, "peaks.bed")
    cmd = [
        "macs2", "bdgpeakcall",
        "-i", bedgraph_path,
        "-c", str(cutoff),
        "-l", str(min_length),
        "-g", str(max_gap),
        "-o", peak_path,
    ]
    subprocess.run(cmd, check=True)
    return peak_path


def filter_by_macs2_peaks(df, peak_path):
    """Keep only rows in df that overlap a MACS2 peak."""
    peaks = pd.read_csv(
        peak_path, sep="\t", header=None, comment="#",
        usecols=[0, 1, 2],
        names=["chrom", "start", "end"],
    )
    peak_intervals = defaultdict(list)
    for _, row in peaks.iterrows():
        peak_intervals[row["chrom"]].append((row["start"], row["end"]))

    def overlaps_peak(row):
        for ps, pe in peak_intervals.get(row["chrom"], []):
            if row["start"] < pe and row["end"] > ps:
                return True
        return False

    mask = df.apply(overlaps_peak, axis=1)
    return df[mask]


def prepare_data(bedgraph_path, genome_path, out_dir, train_chroms, val_chroms,
                 cutoff=2.0, min_length=200, max_gap=100, peak_path=None):
    print("Loading bedgraph...")
    df = pd.read_csv(
        bedgraph_path,
        sep="\t",
        header=None,
        usecols=[0, 1, 2, 3],
        names=["chrom", "start", "end", "score"],
        # start/end are read as float because some rows store coordinates in
        # scientific notation (e.g. "7.2e+07"), which int parsing rejects.
        dtype={"chrom": str, "start": float, "end": float, "score": float},
    )
    df["start"] = df["start"].round().astype(int)
    df["end"] = df["end"].round().astype(int)
    print(f"  {len(df):,} regions loaded")

    print("Extracting sequences...")
    genome = Fasta(genome_path)
    sequences = pd.Series(index=df.index, dtype=str)
    for chrom, group in df.groupby("chrom"):
        if chrom not in genome:
            sequences[group.index] = "N" * 16
            continue
        chrom_seq = genome[chrom][:].seq.upper()
        sequences[group.index] = [chrom_seq[s:e] for s, e in zip(group["start"], group["end"])]
    df["sequence"] = sequences
    print(f"  Done. Example: {df['sequence'].iloc[0]}")

    if peak_path is not None:
        print(f"  Using pre-called peaks from {peak_path}")
    else:
        print("  Running MACS2 bdgpeakcall...")
        peak_path = run_macs2_bdgpeakcall(bedgraph_path, out_dir, cutoff, min_length, max_gap)
        print(f"  Peaks written to {peak_path}")

    before = len(df)
    df = filter_by_macs2_peaks(df, peak_path)
    print(f"  MACS2 filter: kept {len(df):,} / {before:,} regions")

    cols = ["chrom", "start", "end", "sequence", "score"]
    train_df = df[df["chrom"].isin(train_chroms)][cols]
    val_df   = df[df["chrom"].isin(val_chroms)][cols]
    test_df  = df[~df["chrom"].isin(train_chroms + val_chroms)][cols]

    print(f"  Train: {len(train_df):,} rows  ({train_chroms})")
    print(f"  Val:   {len(val_df):,} rows  ({val_chroms})")
    print(f"  Test:  {len(test_df):,} rows  (everything else)")

    train_df.to_parquet(f"{out_dir}/train.parquet", index=False)
    val_df.to_parquet(f"{out_dir}/val.parquet",     index=False)
    test_df.to_parquet(f"{out_dir}/test.parquet",   index=False)
    print(f"Saved to {out_dir}/")


# ---------------------------------------------------------------------------
# Configuration - edit these for future data prep runs
# ---------------------------------------------------------------------------
DATA_DIR = "data"
BEDGRAPH_PATH = f"{DATA_DIR}/entropy_specificity_onGreaterThan1_stitched_annotated_complete.bedgraph"
GENOME_PATH = f"{DATA_DIR}/GRCh38.primary_assembly.genome.fa"
OUT_DIR = DATA_DIR

# Chromosome-level train/val/test split. Splitting by whole chromosome (rather
# than randomly shuffling rows) keeps the model from seeing sequence near a
# validation/test region during training, which would make val/test scores
# look better than they really are.
#   - TEST_CHROMS held out entirely (chr8/chr9 follow the common DeepSEA/Basset
#     convention for a genomics test set).
#   - VAL_CHROMS used for early stopping / LR scheduling (chr2 is large,
#     chr19 is small and gene-dense, giving validation a size mix).
#   - TRAIN_CHROMS is everything else, computed automatically below.
TEST_CHROMS = ["chr8", "chr9"]
VAL_CHROMS = ["chr2", "chr19"]
ALL_CHROMS = [f"chr{i}" for i in range(1, 23)] + ["chrX"]
TRAIN_CHROMS = [c for c in ALL_CHROMS if c not in TEST_CHROMS + VAL_CHROMS]

# MACS2 bdgpeakcall parameters (see run_macs2_bdgpeakcall above).
# MACS2_CUTOFF is on the same scale as the bedgraph "score" column (0-1 here,
# not raw signal) - a region counts as part of a peak once its score crosses
# this value. 0.75 keeps regions in roughly the top quartile+ of scores.
MACS2_CUTOFF = 0.75
MACS2_MIN_LENGTH = 200
MACS2_MAX_GAP = 100


def main():
    prepare_data(
        bedgraph_path=BEDGRAPH_PATH,
        genome_path=GENOME_PATH,
        out_dir=OUT_DIR,
        train_chroms=TRAIN_CHROMS,
        val_chroms=VAL_CHROMS,
        cutoff=MACS2_CUTOFF,
        min_length=MACS2_MIN_LENGTH,
        max_gap=MACS2_MAX_GAP,
    )


if __name__ == "__main__":
    main()

