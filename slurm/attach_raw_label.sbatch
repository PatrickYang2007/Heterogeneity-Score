#!/usr/bin/env bash
#SBATCH --job-name=attach_raw_label
#SBATCH --output=logs/attach_raw_label.%j.out
#SBATCH --error=logs/attach_raw_label.%j.err
#SBATCH --partition=ycga
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=pattyyang1@gmail.com

# Rebuild data/{split}_w{WINDOW}_raw.parquet from the CURRENT splits, attaching the
# un-saturated specificity signal from data/specificity_entropy_weighted.npy as the
# regression target. Must be re-run whenever prepare_data/widen_windows regenerate
# the splits -- the _raw files are a copy of them with `score` swapped, so a stale
# _raw set silently trains on the old row population.
#
# 96 G because attach_raw_label reads every column, including the 2048 bp sequence:
# ~11 GB of Python strings for the 5.2M-row train split, plus the parquet write
# buffers on top. A 32 G ask is what OOM-killed the training jobs at this size.

module load miniconda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate heterogeneityScore

cd "/vast/palmer/pi/garg/Patrick/Heterogeneity Score"
python src/attach_raw_label.py "$@"
