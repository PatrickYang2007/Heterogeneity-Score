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
    return np.array([[mapping[base] for base in seq] for seq in sequences])


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
        dtype={"chrom": str, "start": int, "end": int, "score": float},
    )
    print(f"  {len(df):,} regions loaded")

    print("Extracting sequences...")
    genome = Fasta(genome_path)
    sequences = []
    for chrom, group in df.groupby("chrom"):
        if chrom not in genome:
            sequences.extend(["N" * 16] * len(group))
            continue
        chrom_seq = genome[chrom][:].seq.upper()
        seqs = [chrom_seq[s:e] for s, e in zip(group["start"], group["end"])]
        sequences.extend(seqs)
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

