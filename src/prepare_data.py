import argparse
import pandas as pd
from pyfaidx import Fasta
import numpy as np
import shlex
import subprocess
import os

from config import DEDUP_REGIONS, PEAK_FILTER


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


# Index encoding (A=0, C=1, G=2, T=3, anything else=4 for N/ambiguity codes).
# Used by GenomicDataset to store sequences compactly as int8 and build the
# one-hot tensor on the fly, so memory stays flat as the window width grows.
_BASE_TO_IDX = np.full(256, 4, dtype=np.int8)
for _b, _i in {"A": 0, "C": 1, "G": 2, "T": 3}.items():
    _BASE_TO_IDX[ord(_b)] = _i
    _BASE_TO_IDX[ord(_b.lower())] = _i


def encode_indices(sequences, chunk=200_000):
    """Encode a list of equal-length sequences to an int8 array (N, L).

    Encodes in chunks rather than one shot. The obvious one-liner --
    `np.frombuffer("".join(sequences).encode())` -- materializes the ENTIRE
    dataset twice as transient objects: a single joined str and then its ascii
    bytes copy. At 5.2M x 2048 bp that is ~11 GB each, ~22 GB of transient on top
    of the ~11 GB list of strings and the ~11 GB output array, which is what blew
    the 32 G training jobs. Chunking caps the transient at chunk*L bytes (~410 MB
    at the default) and is otherwise identical.
    """
    n = len(sequences)
    if n == 0:
        return np.empty((0, 0), dtype=np.int8)
    length = len(sequences[0])
    out = np.empty((n, length), dtype=np.int8)
    for i in range(0, n, chunk):
        block = sequences[i:i + chunk]
        buf = np.frombuffer("".join(block).encode("ascii"), dtype=np.uint8)
        out[i:i + len(block)] = _BASE_TO_IDX[buf].reshape(len(block), length)
    return out


def extract_window(chrom_seq, start, end, window):
    """Return a sequence of length `window` centered on [start, end).

    Out-of-bounds positions at chromosome ends are padded with 'N' so every
    returned sequence has exactly `window` bases. If `window` is None or <= the
    region width, the original [start, end) slice is returned unchanged.
    """
    if window is None or window <= (end - start):
        return chrom_seq[start:end]
    center = (start + end) // 2
    new_start = center - window // 2
    new_end = new_start + window
    left_pad = max(0, -new_start)
    right_pad = max(0, new_end - len(chrom_seq))
    core = chrom_seq[max(0, new_start):min(len(chrom_seq), new_end)]
    return "N" * left_pad + core + "N" * right_pad


def load_bedgraph(bedgraph_path, annotations=False):
    """Read the annotated bedgraph into a DataFrame.

    Shared by prepare_data and aggregate_bins so both parse coordinates the same
    way. start/end are read as float because some rows store coordinates in
    scientific notation (e.g. "7.2e+07"), which int parsing rejects; they are
    rounded back to int here.

    Columns 5 and 6 are the annotation this file was built from: a cCRE / gene-part
    class (PLS, dELS, pELS, intron, exon, 5primeUTR, 3primeUTR, CA, CA-CTCF,
    CA-TF, CA-H3K4me3, TF) and a gene symbol. `annotations=True` keeps them. They
    are NOT model inputs -- the model predicts from sequence alone -- but they are
    what makes rows duplicate (one row per overlapping annotation, see
    dedup_regions) and they explain a large share of the label, so they are worth
    carrying for stratified evaluation. Class means on the full file range from
    0.38 (5primeUTR) and 0.41 (PLS) to 0.97 (CA), i.e. a class-only predictor is a
    real baseline a sequence model should be beating.
    """
    cols = [0, 1, 2, 3]
    names = ["chrom", "start", "end", "score"]
    if annotations:
        cols += [4, 5]
        names += ["feature", "gene"]
    df = pd.read_csv(
        bedgraph_path,
        sep="\t",
        header=None,
        usecols=cols,
        names=names,
        dtype={"chrom": str, "start": float, "end": float, "score": float,
               "feature": str, "gene": str},
    )
    df["start"] = df["start"].round().astype(int)
    df["end"] = df["end"].round().astype(int)
    print(f"  {len(df):,} regions loaded")
    return df


def dedup_regions(df):
    """Collapse rows that describe the same genomic interval.

    The bedgraph is annotated, not deduplicated: it emits one row per
    (interval x overlapping annotation), so an interval covered by two genes --
    or by both a gene-part class and a cCRE class -- appears 2+ times carrying the
    SAME score. On the current splits 37-39% of rows are such duplicates, with up
    to 9 copies of a single locus.

    Why this matters rather than just wasting compute: multiplicity is not random.
    It tracks annotation density, which anti-correlates with the label (in train,
    multiplicity 1 -> mean score 0.79, 3 -> 0.63, 5 -> 0.49), so leaving the
    duplicates in silently oversamples the low-score tail 3-5x in training and
    makes val/test metrics per-row instead of per-locus.

    Duplicates are verified to agree on `score` before collapsing (0 conflicting
    groups across all three splits at the time of writing); a mismatch means the
    duplicates are NOT redundant and this function refuses to guess.
    """
    key = ["chrom", "start", "end"]
    n_before = len(df)
    conflicts = df.groupby(key, sort=False)["score"].nunique()
    n_conflict = int((conflicts > 1).sum())
    if n_conflict:
        raise ValueError(
            f"{n_conflict:,} duplicate intervals carry more than one distinct "
            f"score, so they are not redundant copies. Deduplicating would "
            f"silently pick one; resolve the label first (or set "
            f"DEDUP_REGIONS=False to keep the old behavior).")
    out = df.drop_duplicates(subset=key).reset_index(drop=True)
    print(f"  dedup: {n_before:,} rows -> {len(out):,} unique intervals "
          f"({1 - len(out) / n_before:.1%} were duplicates)")
    return out


def extract_sequences(frame, genome, per_group, missing_fill):
    """Map each row of `frame` to a sequence, reading each chromosome's FASTA once.

    Groups `frame` by chrom, upper-cases the chromosome sequence, and delegates
    to `per_group(chrom_seq, group)` -> list of sequences (in group order) for
    the actual slicing. Rows on a chromosome absent from `genome` are filled with
    `missing_fill`. Shared by prepare_data, widen_windows, and aggregate_bins,
    which differ only in how they slice a window out of `chrom_seq`.
    """
    seqs = pd.Series(index=frame.index, dtype=str)
    for chrom, group in frame.groupby("chrom"):
        if chrom not in genome:
            seqs[group.index] = missing_fill
            continue
        chrom_seq = genome[chrom][:].seq.upper()
        seqs[group.index] = per_group(chrom_seq, group)
    return seqs


def sort_bedgraph(bedgraph_path, out_path):
    """Write a position-sorted (chrom, start) copy of the bedgraph.

    The source file is sorted by ANNOTATION CLASS first -- it walks the whole
    genome once per class (3primeUTR, 5primeUTR, CA, ..., pELS), giving 5.77M
    chromosome switches and 143k backward position steps. MACS2 bdgpeakcall
    consumes a coverage track as a stream and assumes it is position-sorted, so on
    the unsorted file it produced 46 garbage "peaks" spanning up to 203 Mb,
    overlapping and nested (see PEAK_FILTER in config.py). Sorting first is the
    minimum needed for bdgpeakcall to mean anything.
    """
    print(f"  sorting bedgraph by position -> {out_path}")
    subprocess.run(f"sort -k1,1 -k2,2n {shlex.quote(bedgraph_path)} "
                   f"> {shlex.quote(out_path)}", shell=True, check=True)
    return out_path


def run_macs2_bdgpeakcall(bedgraph_path, out_dir, cutoff=2.0, min_length=200,
                          max_gap=100, sort_first=True):
    """Run MACS2 bdgpeakcall on a bedgraph and return path to the output peak BED file.

    `sort_first` position-sorts the input into out_dir first (see sort_bedgraph);
    leave it on -- bdgpeakcall's output is meaningless on an unsorted track.
    """
    if sort_first:
        bedgraph_path = sort_bedgraph(
            bedgraph_path, os.path.join(out_dir, "sorted_for_macs2.bedgraph"))
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


def check_peaks_sane(peaks, max_span_mb=10.0, min_peaks=1000):
    """Warn loudly when a peak file looks like bdgpeakcall failed.

    The failure mode seen here is not an exception -- MACS2 exits 0 and writes a
    plausible-looking narrowPeak file. What gives it away is the shape: a handful
    of peaks, each tens or hundreds of Mb, overlapping and nested. Regulatory
    peaks are kb-scale and there should be many thousands, so those two checks
    separate a real call from the degenerate one.
    """
    spans_mb = (peaks["end"] - peaks["start"]) / 1e6
    problems = []
    if len(peaks) < min_peaks:
        problems.append(f"only {len(peaks):,} peaks (expected >= {min_peaks:,})")
    n_huge = int((spans_mb > max_span_mb).sum())
    if n_huge:
        problems.append(f"{n_huge} peak(s) span > {max_span_mb:g} Mb "
                        f"(largest {spans_mb.max():.1f} Mb)")
    if problems:
        print("  !! WARNING: peak file looks degenerate: " + "; ".join(problems))
        print("  !! bdgpeakcall exits 0 on a mis-sorted or unsuitable track and "
              "still writes output. Filtering on this would drop data on an "
              "arbitrary basis -- inspect data/peaks.bed before trusting it.")
    return not problems


def filter_by_macs2_peaks(df, peak_path):
    """Keep only rows in df that overlap a MACS2 peak.

    Vectorized per chromosome: peaks are merged into disjoint intervals and each
    region's start is located with a binary search, which is O(n log p) instead of
    the O(n * p) Python row-loop this replaces (that scan was ~10.7M rows x 46
    peaks). Behavior is unchanged -- a region is kept iff it overlaps some peak.
    """
    peaks = pd.read_csv(
        peak_path, sep="\t", header=None, comment="#",
        usecols=[0, 1, 2],
        names=["chrom", "start", "end"],
    )
    check_peaks_sane(peaks)

    keep = np.zeros(len(df), dtype=bool)
    for chrom, pk in peaks.groupby("chrom", sort=False):
        rows = np.flatnonzero((df["chrom"] == chrom).to_numpy())
        if not len(rows):
            continue
        # Merge overlapping/nested peaks into disjoint intervals so a single
        # binary search per region is enough.
        pk = pk.sort_values("start")
        starts, ends = [], []
        for s, e in zip(pk["start"].to_numpy(), pk["end"].to_numpy()):
            if ends and s <= ends[-1]:
                ends[-1] = max(ends[-1], e)
            else:
                starts.append(s); ends.append(e)
        starts, ends = np.asarray(starts), np.asarray(ends)
        r_start = df["start"].to_numpy()[rows]
        r_end = df["end"].to_numpy()[rows]
        # Candidate interval is the last one starting at or before the region.
        i = np.searchsorted(starts, r_start, side="right") - 1
        ok = i >= 0
        i_ok = np.clip(i, 0, len(starts) - 1)
        # Overlap test: region [start, end) intersects peak [ps, pe).
        keep[rows] = ok & (r_start < ends[i_ok]) & (r_end > starts[i_ok])
    return df[keep]


def prepare_data(bedgraph_path, genome_path, out_dir, train_chroms, val_chroms,
                 cutoff=2.0, min_length=200, max_gap=100, peak_path=None,
                 window=None, dedup=True, peak_filter=False,
                 keep_annotations=True, chroms=None):
    print("Loading bedgraph...")
    df = load_bedgraph(bedgraph_path, annotations=keep_annotations)

    # Optional chromosome restriction, for cheap dress-rehearsal runs that
    # exercise the whole pipeline (dedup -> sequence extraction -> split ->
    # parquet) in minutes instead of hours before committing to the full genome.
    if chroms:
        before = len(df)
        df = df[df["chrom"].isin(chroms)].reset_index(drop=True)
        print(f"  --chroms {chroms}: {before:,} -> {len(df):,} regions")

    # Collapse annotation-duplicated intervals BEFORE sequence extraction: the
    # duplicates are identical rows, so extracting each one's sequence twice is
    # both wasted work and a silent reweighting of the training distribution.
    if dedup:
        print("Deduplicating regions...")
        df = dedup_regions(df)

    print("Extracting sequences...")
    genome = Fasta(genome_path)
    df["sequence"] = extract_sequences(
        df, genome,
        lambda chrom_seq, group: [
            extract_window(chrom_seq, s, e, window)
            for s, e in zip(group["start"], group["end"])
        ],
        missing_fill="N" * 16,
    )
    print(f"  Done. Example: {df['sequence'].iloc[0]}")

    # Peak filtering is opt-in (see PEAK_FILTER in config.py). On this bedgraph the
    # original always-on filter discarded ~48% of regions on the basis of 46
    # degenerate multi-Mb "peaks", which is why it now has to be asked for.
    if not peak_filter:
        print("  MACS2 peak filter: OFF (PEAK_FILTER=False) -- keeping all regions")
    else:
        if peak_path is not None:
            print(f"  Using pre-called peaks from {peak_path}")
        else:
            print("  Running MACS2 bdgpeakcall...")
            peak_path = run_macs2_bdgpeakcall(bedgraph_path, out_dir, cutoff,
                                              min_length, max_gap)
            print(f"  Peaks written to {peak_path}")

        before = len(df)
        df = filter_by_macs2_peaks(df, peak_path)
        print(f"  MACS2 filter: kept {len(df):,} / {before:,} regions")

    cols = ["chrom", "start", "end", "sequence", "score"]
    # Annotation columns ride along for stratified evaluation (see load_bedgraph);
    # they are never fed to the model, which predicts from sequence alone.
    cols += [c for c in ("feature", "gene") if c in df.columns]
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

# This script writes the BASE splits at the bedgraph's raw 16 bp region width.
# Widening is widen_windows.py's job: it re-extracts wider windows from these
# splits (reading config.WINDOW) without redoing the bedgraph parse, dedup, or
# peak steps, so window-size sweeps are cheap. Pass --window N here only if you
# deliberately want wide base splits -- at 2048 bp that is ~13 GB of sequence
# strings for 6.5M regions, which will not fit the job's 32-64 G ask.


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    # Default None = keep the bedgraph's raw 16 bp regions. This is deliberately
    # NOT config.WINDOW: widening belongs to widen_windows.py, which re-extracts
    # wider windows from these base splits without re-running the expensive
    # bedgraph/dedup/peak steps. Doing it here instead would (a) duplicate that
    # work and (b) blow the job's memory -- 6.5M regions x 2048 bp is ~13 GB of
    # sequence strings before pandas overhead, against a 32 G ask. Pass
    # --window N only if you deliberately want wide base splits.
    ap.add_argument("--window", type=int, default=None,
                    help="width of the sequence stored per region (default: the "
                         "raw 16 bp region; widen later with widen_windows.py)")
    ap.add_argument("--dedup", dest="dedup", action="store_true")
    ap.add_argument("--no-dedup", dest="dedup", action="store_false",
                    help="keep annotation-duplicated rows (the old behavior)")
    ap.set_defaults(dedup=DEDUP_REGIONS)
    ap.add_argument("--peak-filter", dest="peak_filter", action="store_true",
                    help="run MACS2 bdgpeakcall and keep only peak-overlapping "
                         "regions (see PEAK_FILTER in config.py -- it is broken "
                         "on this bedgraph)")
    ap.set_defaults(peak_filter=PEAK_FILTER)
    ap.add_argument("--chroms", nargs="+", default=None,
                    help="restrict to these chromosomes (dress-rehearsal runs)")
    return ap.parse_args()


def main():
    args = parse_args()
    prepare_data(
        bedgraph_path=BEDGRAPH_PATH,
        genome_path=GENOME_PATH,
        out_dir=OUT_DIR,
        train_chroms=TRAIN_CHROMS,
        val_chroms=VAL_CHROMS,
        cutoff=MACS2_CUTOFF,
        min_length=MACS2_MIN_LENGTH,
        max_gap=MACS2_MAX_GAP,
        window=args.window,
        dedup=args.dedup,
        peak_filter=args.peak_filter,
        chroms=args.chroms,
    )


if __name__ == "__main__":
    main()

