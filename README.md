# Homogeneity Score

A 1D convolutional neural network that predicts a per-region "homogeneity score"
directly from genomic DNA sequence. The model is trained on a bedgraph of scored
regions, with a chromosome-level train/val/test split, and is built in PyTorch.

## Overview

The pipeline turns a scored bedgraph into `(sequence, score)` examples, encodes
the DNA as one-hot, and trains a CNN to regress the score from sequence alone.

```
bedgraph (chrom, start, end, score)
      │
      ▼
prepare_data.py ──► MACS2 peak filtering ──► chrom split ──► data/{train,val,test}.parquet
      │
      ▼  (optional: widen the 16 bp regions to a larger context window)
widen_windows.py ──► data/{train,val,test}_w256.parquet
      │
      ▼
train.py ──► model.py (CNN) + model_train.py (Trainer) ──► best_model.pt
      │
      ▼
evaluate.py / predict.py
```

## Data representation

Each raw region is 16 bp with a single score (0–1). Because 16 bp is too little
context for the model to learn from, `widen_windows.py` re-extracts a wider
window (default **256 bp**) centered on each region's midpoint, padding with `N`
at chromosome ends. The score stays pinned to the original region — the extra
flanking bases are context only. Sequences are one-hot encoded
(`A/C/G/T` → 4-dim, `N` → all-zero) into a `4 × L` tensor.

The split is by **whole chromosome** (not random rows) so the model never sees
sequence near a validation/test region during training:

- **Test:** chr8, chr9 (held out entirely)
- **Validation:** chr2, chr19 (early stopping / LR scheduling)
- **Train:** everything else (chr1, 3–7, 10–18, 20–22, chrX)

## Model

`HomogeneityScoreModel` (in `src/model.py`) stacks `num_blocks` pooled conv
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
`AGGREGATE` flag in `train.py`:

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
```

### 3. Evaluate / predict
```bash
python -c "import sys; sys.path.insert(0, 'src'); from evaluate import evaluate; \
  evaluate('Models/best_model_w256_perRegion.pt', 'data/test_w256.parquet', 'data/val_w256.parquet')"

python src/predict.py data/test_w256.parquet \
  --weights Models/best_model_w256_perRegion.pt --output preds.tsv
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
| `src/config.py` | shared experiment settings (`WINDOW`, `AGGREGATE`) |
| `src/prepare_data.py` | bedgraph → sequences, MACS2 peak filter, chrom split, parquet |
| `src/widen_windows.py` | re-extract wider context windows from existing splits |
| `src/aggregate_bins.py` | summed-bin experiment: tile genome, sum scores per bin |
| `src/model.py` | CNN, attention pooling, `GenomicDataset`, dataloader |
| `src/model_train.py` | `Trainer` (training/validation loops, checkpointing) |
| `src/train.py` | training entry point and hyperparameters |
| `src/evaluate.py` | metrics (Pearson/Spearman) and scatter/loss plots |
| `src/eval_report.py` | full eval report: metrics + diagnostic plots + summary |
| `src/predict.py` | run inference on new sequences |
| `slurm/*.sbatch` | Slurm submission scripts |

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
