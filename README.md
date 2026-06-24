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

`HomogeneityScoreModel` (in `model.py`) is three pooled conv blocks
(BatchNorm → GELU → Conv1d → Dropout → MaxPool) that grow the receptive field
geometrically, followed by an attention-pooling layer and a linear head. Training
uses AdamW, MSE loss, gradient clipping, `ReduceLROnPlateau`, and saves the
checkpoint with the best validation Pearson correlation.

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

### 1. Prepare data
```bash
python prepare_data.py            # bedgraph -> data/{train,val,test}.parquet
python widen_windows.py           # -> data/{split}_w256.parquet (per-region context)
```

For the summed-bin experiment instead:
```bash
python aggregate_bins.py          # -> data/{split}_agg256.parquet
```

### 2. Train
Set `WINDOW` and `AGGREGATE` at the top of `train.py`, then:
```bash
python train.py                   # writes best_model.pt, loss_curves.png
```

### 3. Evaluate / predict
```bash
python -c "from evaluate import evaluate; \
  evaluate('best_model.pt', 'data/test_w256.parquet', 'data/val_w256.parquet')"

python predict.py data/test_w256.parquet --weights best_model.pt --output preds.tsv
# add --aggregate when the weights came from a summed-bin model
```

On an HPC cluster, `prepare_data.sbatch` and `train.sbatch` wrap these as Slurm
jobs (edit the conda env / partitions for your system).

## Repository layout

| File | Purpose |
|---|---|
| `prepare_data.py` | bedgraph → sequences, MACS2 peak filter, chrom split, parquet |
| `widen_windows.py` | re-extract wider context windows from existing splits |
| `aggregate_bins.py` | summed-bin experiment: tile genome, sum scores per bin |
| `model.py` | CNN, attention pooling, `GenomicDataset`, dataloader |
| `model_train.py` | `Trainer` (training/validation loops, checkpointing) |
| `train.py` | training entry point and hyperparameters |
| `evaluate.py` | metrics (Pearson/Spearman) and scatter/loss plots |
| `predict.py` | run inference on new sequences |
| `*.sbatch` | Slurm submission scripts |

## Notes

- `data/`, model checkpoints (`*.pt`), plots (`*.png`), and Slurm logs are
  git-ignored; only code is tracked.
- Changing `WINDOW` to a new size means regenerating that size's data once
  (`widen_windows.py` or `aggregate_bins.py`); after that you can flip `AGGREGATE`
  freely without regenerating.
