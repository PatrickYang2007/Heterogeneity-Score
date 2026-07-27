# Heterogeneity Score

A 1D convolutional neural network that predicts a per-region "heterogeneity score"
directly from genomic DNA sequence. The model is trained on a bedgraph of scored
regions, with a chromosome-level train/val/test split, and is built in PyTorch.

## Overview

The pipeline turns a scored bedgraph into `(sequence, score)` examples, encodes
the DNA as one-hot, and trains a CNN to regress the score from sequence alone.

```
bedgraph (chrom, start, end, score, feature, gene)
      │
      ▼
prepare_data.py ──► dedup ──► [optional MACS2 filter] ──► chrom split ──► data/{train,val,test}.parquet
      │
      ▼  (optional: widen the 16 bp regions to a larger context window)
widen_windows.py ──► data/{train,val,test}_w256.parquet
      │
      ▼
train.py ──► model.py (CNN) + trainer.py (Trainer) ──► best_model.pt
      │
      ▼
eval_report.py / predict.py        baselines.py (what the model must beat)
```

> **Read [`docs/FINDINGS.md`](docs/FINDINGS.md) before running anything.** An audit
> found two data-prep bugs that were corrupting the dataset (annotation-duplicated
> rows, and a MACS2 filter that discarded ~48% of the data on 46 degenerate
> multi-megabase "peaks"), and established that the current CNN only narrowly beats
> a six-parameter GC/CpG baseline. Both bugs are fixed here, but **the data must be
> regenerated** and metrics from before the fix are not comparable to metrics after.

## Data representation

How sequence and score are presented to the model is **a choice, not a fixed
property of the data** — it's set in `src/config.py` and regenerated per run, so
the description below is the *default* configuration, not the only one.

The raw input is fixed by the bedgraph: each region is 16 bp with a single score
(0–1). Everything downstream is configurable:

- **Window width** (`WINDOW` in `src/config.py`, currently **2048 bp**; set to
  `None` to keep the raw 16 bp). 16 bp is too little context to learn from, so
  `widen_windows.py` re-extracts a wider window centered on each region's
  midpoint, padding with `N` at chromosome ends. Re-run it to change the width.
- **Context crop** (`CROP` / `--crop`). Sweeps context width *without*
  regenerating the multi-GB parquets: the row still stores `WINDOW` bp, the model
  is handed the central `CROP` bp. Because windows are region-centered, the label's
  region stays centered after cropping. Measured composition fit peaks at ~256 bp
  of context and decays out to 2048 bp, so 2048 is likely 8× more sequence than
  the signal supports — see [`docs/FINDINGS.md`](docs/FINDINGS.md) §3.
- **Label / representation** (`AGGREGATE` in `src/config.py`). `False` keeps the
  per-region score pinned to its original region (the flanking bases are context
  only); `True` switches to a summed-bin label on a different scale. See
  [Two experiments](#two-experiments) — this changes both the data file and the
  model's output head.

The DNA encoding itself is fixed in code: sequences are one-hot encoded
(`A/C/G/T` → 4-dim, `N` → all-zero) into a `4 × L` tensor.

The split is by **whole chromosome** (not random rows) so the model never sees
sequence near a validation/test region during training:

- **Test:** chr8, chr9 (held out entirely)
- **Validation:** chr2, chr19 (early stopping / LR scheduling)
- **Train:** everything else (chr1, 3–7, 10–18, 20–22, chrX)

## Model

`HeterogeneityScoreModel` (in `src/model.py`) stacks `num_blocks` pooled conv
blocks (BatchNorm → GELU → Conv1d → Dropout → MaxPool) that grow the receptive
field geometrically, followed by an attention-pooling layer and a linear head.
Channels double each block (`num_filters`, `num_filters*2`, ...). Training uses
AdamW, MSE loss, gradient clipping, `ReduceLROnPlateau`, and saves the checkpoint
with the best validation Pearson correlation.

Model capacity is configurable without editing layers, via flags on `train.py`:

| Flag | Meaning | Default |
|---|---|---|
| `--num-filters` | width (channels in the first block) | 32 |
| `--num-blocks` | depth (number of conv blocks) | 3 |
| `--crop` | use only the central N bp of each window | off |
| `--center-pool` | head also reads the features over the labeled region | off |

`--center-pool` addresses a structural mismatch: the label describes the central
16 bp, but `AttentionPool` returns a softmax-weighted **average** over every output
position (64 of them at 2048 bp with 5 blocks), so the labeled region contributes
~1/64 of what the head sees. With the flag on, the head gets
`[attention_pool(x) ; x[:, :, center]]` — global context *and* the labeled region.
It changes the head's shape, so pass the same flag to `eval_report.py`/`predict.py`.

```bash
sbatch slurm/train.sbatch --num-blocks 5 --num-filters 64
```

Runs with non-default capacity save to a tagged checkpoint (e.g.
`best_model_w2048_b5_f64.pt`) so sweeps don't overwrite each other. Pass the same
`--num-filters`/`--num-blocks` to `eval_report.py`/`predict.py` when loading such
a model. Note: with `pool=2` each block halves the length, so keep
`num_blocks <= log2(WINDOW) - 2` (e.g. <= 9 for a 2048 bp window).

## Two experiments

The repo supports two ways of relating sequence to score, toggled by the
`AGGREGATE` flag in `src/config.py` (or `--aggregate`/`--no-aggregate` on
`train.py`):

| | Per-region (default) | Summed-bin |
|---|---|---|
| Data script | `widen_windows.py` | `aggregate_bins.py` |
| Data files | `data/{split}_w{WINDOW}.parquet` | `data/{split}_agg{WINDOW}.parquet` |
| Label | the single region's score (0–1) | **sum** of all region scores in a non-overlapping `WINDOW` bp bin |
| Model output | sigmoid (bounded 0–1) | linear (unbounded) |
| `AGGREGATE` | `False` | `True` |

**Per-region:** a `WINDOW` bp window centered on each 16 bp region, labeled with
that region's score.

**Summed-bin:** the genome is tiled into non-overlapping `WINDOW` bp bins
(`0–256`, `256–512`, …); each bin's label is the sum of the scores of the 16 bp
regions whose center falls inside it. Because the label is a sum (range
~0..#regions/bin) rather than a 0–1 score, the model's final sigmoid is dropped
(`bounded=False`, handled automatically when `AGGREGATE=True`). Note that
MSE/loss values are **not** comparable between the two modes because the labels
live on different scales — compare them by Pearson/Spearman correlation instead.

## Data integrity (read before regenerating)

Two `prepare_data.py` steps were corrupting the dataset; both are fixed and both
are toggles in `src/config.py`. Full evidence in [`docs/FINDINGS.md`](docs/FINDINGS.md).

- **`DEDUP_REGIONS`** (default **on**). The bedgraph is annotated, not
  deduplicated — it emits one row per (interval × overlapping annotation), so the
  same 16 bp interval appears up to 9 times with the same score. **37–39% of rows
  in every split are duplicates.** Multiplicity anti-correlates with the label
  (multiplicity 1 → mean score 0.79, 5 → 0.49), so leaving them in oversamples the
  low-score tail 3–5× in training and makes val/test metrics per-row rather than
  per-locus. Deduplication refuses to run if duplicates disagree on the score.

- **`PEAK_FILTER`** (default **off**). The MACS2 filter used to be unconditional
  and dropped ~48% of regions — on the basis of 46 "peaks" spanning up to 203 Mb,
  overlapping and nested. `bdgpeakcall` needs a position-sorted track, and this
  bedgraph is sorted by *annotation class* first (it restarts the genome 12 times).
  MACS2 exits 0 on that input and writes a plausible-looking file anyway.
  `prepare_data.py` now sorts before calling MACS2 and warns via
  `check_peaks_sane()`, but inspect `data/peaks.bed` before re-enabling this.

Annotation columns (`feature`, `gene`) now ride along into the parquets for
stratified evaluation. They are **never** model inputs — the model predicts from
sequence alone.

## Baselines

`src/baselines.py` computes what any architecture change has to beat. Run it
alongside every eval; a model that doesn't pull away from the composition baseline
hasn't learned sequence grammar.

```bash
python src/baselines.py --sample 20000
```

| baseline | what it measures | test Pearson |
|---|---|---|
| composition (GC/CpG, central 256 bp) | how far 6 numbers get you | 0.604 |
| k-mer ridge | composition without positional information | 0.597 |
| smoothed label (±1024 bp, leave-one-out) | ceiling from label autocorrelation alone | 0.785 |
| current CNN (`w2048_b5_f64_mask`) | | 0.680 |

The CNN sits between them: barely above the composition baseline, and **below the
ceiling implied by its own window width**. That gap is the headroom.

## Class imbalance (per-region)

The per-region labels are bounded [0, 1] but **pile up at exactly 1.0** (~41% of
rows). Under MSE that spike dominates the gradient and pulls predictions toward
the high mean, compressing the output's dynamic range (a pred-vs-observed
best-fit slope well below 1). The sigmoid head + `TARGET_CLIP` and the output-bias
seeding (both in `train.py`) already lean against this; `BALANCE_SPIKE` in
`src/config.py` adds an optional, more direct lever:

- `BALANCE_SPIKE` (`--balance` / `--no-balance`, default **off**) thins the
  `score >= SPIKE_THRESHOLD` pile-up down to `SPIKE_KEEP_FRAC` (`--cap-frac`) of
  its rows. **Only the training split is rebalanced** — val/test always keep the
  real distribution, so their metrics stay comparable across runs. Ignored for
  the summed-bin (`AGGREGATE`) path, which has no 1.0 spike.

Balanced runs save under a `_bal{pct}` checkpoint tag (e.g.
`best_model_w2048_b5_f64_mask_bal30.pt` keeps 30% of the spike), so they don't
overwrite the unbalanced baseline. Pass the matching `--balance`/`--cap-frac`
only at train time; `eval_report.py`/`predict.py` need no balancing flags.

## Usage

Dependencies: `torch`, `pandas`, `numpy`, `pyfaidx`, `scipy`, `matplotlib`, and
`macs2` (for peak calling). Place the genome FASTA and bedgraph in `data/`
(both are git-ignored).

All experiment settings live in `src/config.py` (`WINDOW`, `AGGREGATE`); set them
once and every script reads from there. The Slurm scripts in `slurm/` `cd` to the
project root and run `python src/<script>.py`, so submit them from the repo root.

### 1. Prepare data
```bash
sbatch slurm/prepare_data.sbatch     # bedgraph -> data/{train,val,test}.parquet
sbatch slurm/widen_windows.sbatch    # -> data/{split}_w{WINDOW}.parquet (per-region)
```

For the summed-bin experiment instead (set `AGGREGATE = True` in `src/config.py`):
```bash
sbatch slurm/aggregate_bins.sbatch   # -> data/{split}_agg{WINDOW}.parquet
```

### 2. Train
After the matching data exists for the current `WINDOW`/`AGGREGATE`:
```bash
sbatch slurm/train.sbatch            # -> Models/best_model_{w,agg}{WINDOW}.pt + loss curve
sbatch slurm/train.sbatch --balance  # per-region: thin the score=1.0 spike (train only)

# context-width sweep -- no data regeneration needed, ~8x cheaper per epoch
sbatch slurm/train.sbatch --num-blocks 5 --num-filters 64 --crop 256 --center-pool
```

Crop / center-pool runs save under their own tags (`_c256`, `_ctr`), so they never
overwrite the full-window baseline. Pass the same `--crop`/`--center-pool` to
`eval_report.py` and `predict.py`.

### 3. Evaluate / predict
```bash
# full eval report (metrics + diagnostic plots + summary.txt) -> Models/eval/<tag>/
sbatch slurm/eval.sbatch --weights Models/best_model_w2048.pt --window 2048

python src/predict.py data/test_w2048.parquet \
  --weights Models/best_model_w2048.pt --window 2048 --output preds.tsv
# add --aggregate when the weights came from a summed-bin model
```

## Repository layout

```
src/      Python modules (config, model, training, data prep)
slurm/    Slurm submission scripts (.sbatch); submit from the repo root
tests/    pytest test suite (uses synthetic data only)
Models/   saved checkpoints (*.pt), eval reports, and loss curves
logs/     Slurm .out/.err job logs (git-ignored)
data/     genome FASTA, bedgraph, and parquet splits (git-ignored)
```

| File | Purpose |
|---|---|
| `src/config.py` | shared experiment settings (`WINDOW`, `CROP`, `AGGREGATE`, `REGION_MASK`, `CENTER_POOL`, `DEDUP_REGIONS`, `PEAK_FILTER`, `BALANCE_SPIKE`) |
| `src/prepare_data.py` | bedgraph → sequences, dedup, optional peak filter, chrom split, parquet |
| `src/widen_windows.py` | re-extract wider context windows from existing splits |
| `src/aggregate_bins.py` | summed-bin experiment: tile genome, sum scores per bin |
| `src/model.py` | CNN, attention pooling, `GenomicDataset`, dataloader |
| `src/trainer.py` | `Trainer` (training/validation loops, checkpointing) |
| `src/train.py` | training entry point, hyperparameters, loss-curve plot |
| `src/eval_report.py` | full eval report: metrics + diagnostic plots + summary |
| `src/baselines.py` | composition / k-mer / smoothed-label baselines to beat |
| `src/predict.py` | run inference on new sequences |
| `slurm/*.sbatch` | Slurm submission scripts |
| `docs/FINDINGS.md` | audit: data bugs, baselines, and what to try next |

## Tests

A pytest suite covers the deterministic, easy-to-break parts: DNA encoding,
`extract_window` centering/padding, the bin summing, model wiring (output shape,
sigmoid vs linear head, configurable depth, and loading old checkpoints), the
hand-rolled eval metrics, and `GenomicDataset`. The tests build their own small
synthetic data, so no real genome/bedgraph files are needed.

```bash
sbatch slurm/test.sbatch     # CPU-only; installs pytest if missing
```

## Notes

- `data/`, model checkpoints (`*.pt`), plots (`*.png`), and Slurm logs are
  git-ignored; only code is tracked.
- Changing `WINDOW` in `src/config.py` means regenerating that size's data once
  (`widen_windows.sbatch` or `aggregate_bins.sbatch`); after that you can flip
  `AGGREGATE` freely without regenerating.
