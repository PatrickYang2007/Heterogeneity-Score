"""Non-learned baselines the CNN has to beat to be worth its complexity.

Nothing in this repo previously answered "is a 14M-parameter conv net doing
anything a handful of numbers can't?", and the answer turns out to be "not much":
on a random test sample, a linear model over 5 GC/CpG composition features reaches
Pearson ~0.61 / Spearman ~0.47 against the b5_f64 CNN's 0.68 / 0.53, and ~59% of
the CNN's own output variance is reproducible from those same 5 numbers. Any
future architecture change should be judged against these numbers, not against
zero.

Three baselines are reported, cheapest first:

  composition  linear model on GC fraction, CpG fraction and CpG observed/expected
               over the central `--crop` bp. Run it at several widths: this is how
               the "the label is a ~256 bp local property" finding was made -- the
               fit peaks near 256 bp and decays out to 2048 bp.

  kmer         ridge regression on normalized k-mer frequencies. Strictly more
               expressive than `composition` (k=1,2 recover GC/CpG) and still has
               no notion of motif position, so the gap between this and the CNN is
               the part of the signal that needs sequence ORDER, i.e. the part a
               conv net is actually for.

  smoothed     the label's own local average -- for each region, the mean label of
               its neighbours within +/- `--half` bp, EXCLUDING itself. This is
               not a sequence model; it is the information ceiling implied by the
               label's spatial autocorrelation. It answers "if the model could only
               resolve position, not sequence, how well would it do?" On test at
               +/-1024 bp (the current 2048 bp window) it scores ~0.79, which the
               0.68 CNN does not reach.

Usage (CPU, minutes):
    python src/baselines.py --sample 20000
    python src/baselines.py --which composition --widths 128 256 512 1024 2048
"""
import argparse
import sys

import numpy as np
import pyarrow.parquet as pq
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from prepare_data import encode_indices

DATA_DIR = "data"


def sample_split(path, n, seed=0, columns=("sequence", "score")):
    """Random rows spread across every row group of a parquet file.

    Reading the first n rows would sample one contiguous stretch of one
    chromosome, which is not representative -- the label varies a lot along the
    genome. Sampling within each row group keeps memory bounded (these files are
    up to 3 GB) while still covering the whole split.
    """
    pf = pq.ParquetFile(path)
    n_groups = pf.metadata.num_row_groups
    per_group = max(1, n // n_groups)
    rng = np.random.default_rng(seed)
    frames = []
    for i in range(n_groups):
        t = pf.read_row_group(i, columns=list(columns))
        idx = np.sort(rng.choice(t.num_rows, size=min(per_group, t.num_rows),
                                 replace=False))
        # Subset in pandas, not with pyarrow.Table.take: a row group of 2048 bp
        # sequences exceeds the 2 GB offset limit of a non-large string array and
        # take() fails with "offset overflow while concatenating arrays".
        frames.append(t.to_pandas().iloc[idx].reset_index(drop=True))
        del t
    return pd.concat(frames, ignore_index=True)


def fit_predict(Xtr, ytr, Xte, lam=0.0):
    """Ridge (lam>0) or OLS (lam=0) with an unpenalized intercept."""
    A = np.c_[Xtr, np.ones(len(Xtr))].astype(np.float64)
    G = A.T @ A
    if lam:
        G += lam * np.eye(A.shape[1])
        G[-1, -1] -= lam                      # never penalize the intercept
    w = np.linalg.solve(G, A.T @ ytr)
    return np.c_[Xte, np.ones(len(Xte))] @ w


def report(name, pred, y):
    r = pearsonr(pred, y)[0]
    rho = spearmanr(pred, y)[0]
    print(f"  {name:<26} pearson={r:.4f}  spearman={rho:.4f}", flush=True)
    return {"baseline": name, "pearson": float(r), "spearman": float(rho)}


# ----------------------------- composition -----------------------------

def composition_features(idx, width=None):
    """GC frac, CpG frac, CpG observed/expected over the central `width` bp.

    `idx` is the int8 encoding from prepare_data.encode_indices (A/C/G/T = 0-3,
    N = 4). CpG observed/expected normalizes CpG count by what GC content alone
    would predict, which is the classic CpG-island statistic and the single
    strongest feature here (r ~ -0.60 with the label).
    """
    L = idx.shape[1]
    if width and width < L:
        idx = idx[:, L // 2 - width // 2: L // 2 + width // 2]
    isC, isG = idx == 1, idx == 2
    n = np.maximum((idx != 4).sum(1), 1)
    nC, nG = isC.sum(1), isG.sum(1)
    cpg = (isC[:, :-1] & isG[:, 1:]).sum(1)
    gc = (nC + nG) / n
    cpg_frac = cpg / n
    expected = np.maximum(nC * nG / n, 1e-9)
    return np.c_[gc, cpg_frac, cpg / expected, gc ** 2, cpg_frac ** 2]


def run_composition(Itr, ytr, Ite, yte, widths):
    print("\n[composition] linear model on GC / CpG / CpG o-e")
    out = []
    for w in widths:
        pred = fit_predict(composition_features(Itr, w), ytr,
                           composition_features(Ite, w))
        out.append(report(f"central {w} bp", pred, yte))
    return out


# ----------------------------- k-mer -----------------------------

def kmer_features(idx, k, width=None):
    """Normalized frequencies of all 4**k k-mers; k-mers touching an N are skipped."""
    L = idx.shape[1]
    if width and width < L:
        idx = idx[:, L // 2 - width // 2: L // 2 + width // 2]
    n, L = idx.shape
    code = np.zeros((n, L - k + 1), dtype=np.int32)
    ok = np.ones((n, L - k + 1), dtype=bool)
    for j in range(k):
        col = idx[:, j:L - k + 1 + j]
        code = code * 4 + np.clip(col, 0, 3)
        ok &= col != 4
    out = np.zeros((n, 4 ** k), dtype=np.float32)
    for i in range(n):
        out[i] = np.bincount(code[i][ok[i]], minlength=4 ** k)
    return out / np.maximum(out.sum(1, keepdims=True), 1)


def run_kmer(Itr, ytr, Ite, yte, ks, width, lam):
    print(f"\n[kmer] ridge on k-mer frequencies (central {width} bp, lambda={lam:g})")
    out = []
    for k in ks:
        pred = fit_predict(kmer_features(Itr, k, width), ytr,
                           kmer_features(Ite, k, width), lam=lam)
        out.append(report(f"{k}-mer ({4 ** k} feats)", pred, yte))
    return out


# ----------------------------- smoothed label -----------------------------

def run_smoothed(split, halves, data_dir=DATA_DIR):
    """Leave-one-out local mean of the LABEL -- the spatial-autocorrelation ceiling.

    Uses the un-widened {split}.parquet because it carries coordinates. Each
    region is scored by the mean label of its neighbours within +/- half bp,
    itself excluded -- without that exclusion the "prediction" contains its own
    answer and the correlation is meaningless at small radii.

    Read this as an upper bound on any model whose spatial resolution is `half`:
    it uses the true labels of neighbours, which a sequence model does not get.
    """
    print(f"\n[smoothed] leave-one-out local mean of the label ({split} split)")
    df = pd.read_parquet(f"{data_dir}/{split}.parquet",
                         columns=["chrom", "start", "end", "score"])
    df = df.drop_duplicates(subset=["chrom", "start"]).sort_values(["chrom", "start"])
    out = []
    for half in halves:
        preds, trues = [], []
        for _, d in df.groupby("chrom", sort=False):
            pos = ((d.start + d.end) // 2).to_numpy()
            y = d.score.to_numpy().astype(float)
            cs = np.concatenate([[0.0], np.cumsum(y)])
            lo = np.searchsorted(pos, pos - half, "left")
            hi = np.searchsorted(pos, pos + half, "right")
            cnt = hi - lo - 1                       # drop self
            tot = cs[hi] - cs[lo] - y
            keep = cnt > 0                          # no neighbours -> undefined
            preds.append(tot[keep] / cnt[keep])
            trues.append(y[keep])
        out.append(report(f"+/-{half} bp neighbours",
                          np.concatenate(preds), np.concatenate(trues)))
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--window", type=int, default=2048,
                   help="which {split}_w{WINDOW}.parquet to read (default 2048)")
    p.add_argument("--raw-label", action="store_true",
                   help="use the _raw (log1p specificity) target instead of the score")
    p.add_argument("--sample", type=int, default=20000,
                   help="rows sampled per split for the sequence baselines")
    p.add_argument("--which", nargs="+", default=["composition", "kmer", "smoothed"],
                   choices=["composition", "kmer", "smoothed"])
    p.add_argument("--widths", type=int, nargs="+",
                   default=[64, 128, 256, 512, 1024, 2048],
                   help="context widths for the composition baseline")
    p.add_argument("--ks", type=int, nargs="+", default=[3, 5],
                   help="k values for the k-mer baseline")
    p.add_argument("--kmer-width", type=int, default=256,
                   help="context width for the k-mer baseline (default 256)")
    p.add_argument("--lam", type=float, default=1e-2, help="ridge penalty for k-mer")
    p.add_argument("--halves", type=int, nargs="+",
                   default=[64, 256, 1024], help="radii for the smoothed-label ceiling")
    p.add_argument("--data-dir", default=DATA_DIR)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    suffix = f"_w{args.window}" + ("_raw" if args.raw_label else "")
    results = []

    if {"composition", "kmer"} & set(args.which):
        print(f"Sampling {args.sample:,} rows per split from "
              f"{args.data_dir}/{{train,test}}{suffix}.parquet ...", flush=True)
        tr = sample_split(f"{args.data_dir}/train{suffix}.parquet", args.sample,
                          args.seed)
        te = sample_split(f"{args.data_dir}/test{suffix}.parquet", args.sample,
                          args.seed + 1)
        ytr, yte = tr.score.to_numpy(), te.score.to_numpy()
        Itr = encode_indices(tr.sequence.str.upper().tolist()).astype(np.int8)
        Ite = encode_indices(te.sequence.str.upper().tolist()).astype(np.int8)
        del tr, te
        print(f"  train n={len(ytr):,}  test n={len(yte):,}  "
              f"(test label mean {yte.mean():.3f}, std {yte.std():.3f})")
        if "composition" in args.which:
            results += run_composition(Itr, ytr, Ite, yte, args.widths)
        if "kmer" in args.which:
            results += run_kmer(Itr, ytr, Ite, yte, args.ks, args.kmer_width, args.lam)

    if "smoothed" in args.which and not args.raw_label:
        results += run_smoothed("test", args.halves, args.data_dir)

    print("\nBeat these before believing an architecture change helped.")
    return results


if __name__ == "__main__":
    sys.exit(0 if main() is not None else 1)
