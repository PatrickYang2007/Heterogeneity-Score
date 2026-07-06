"""Re-extract wider sequence windows from the existing train/val/test splits.

The raw regions are 16 bp, which is too little context for the model to learn
from (train loss plateaus immediately and Pearson r stalls around ~0.44). This
script keeps the exact same rows, scores, and chromosome split, but replaces the
16 bp `sequence` with a wider window of `WINDOW` bp centered on each region,
re-extracted from the genome FASTA. It does NOT re-run MACS2 / peak filtering,
so it's cheap to sweep different window sizes.

Output: data/{split}_w{WINDOW}.parquet, which train.py / eval_report.py can point at.
"""
import os
import pandas as pd
from pyfaidx import Fasta

from prepare_data import extract_window
from config import WINDOW

DATA_DIR = "data"
GENOME_PATH = f"{DATA_DIR}/GRCh38.primary_assembly.genome.fa"
SPLITS = ["train", "val", "test"]


def widen_split(split, genome, window):
    in_path = f"{DATA_DIR}/{split}.parquet"
    out_path = f"{DATA_DIR}/{split}_w{window}.parquet"
    df = pd.read_parquet(in_path)
    print(f"[{split}] {len(df):,} rows -> {window} bp windows")

    seqs = pd.Series(index=df.index, dtype=str)
    for chrom, group in df.groupby("chrom"):
        if chrom not in genome:
            seqs[group.index] = "N" * window
            continue
        chrom_seq = genome[chrom][:].seq.upper()
        seqs[group.index] = [
            extract_window(chrom_seq, s, e, window)
            for s, e in zip(group["start"], group["end"])
        ]
    df["sequence"] = seqs

    lengths = df["sequence"].str.len()
    assert (lengths == window).all(), f"got widths {sorted(lengths.unique())[:5]}"
    df.to_parquet(out_path, index=False)
    print(f"[{split}] wrote {out_path}")


def main():
    genome = Fasta(GENOME_PATH)
    for split in SPLITS:
        if not os.path.exists(f"{DATA_DIR}/{split}.parquet"):
            print(f"[{split}] missing, skipping")
            continue
        widen_split(split, genome, WINDOW)
    print("Done.")


if __name__ == "__main__":
    main()