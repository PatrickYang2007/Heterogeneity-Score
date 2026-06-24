#!/usr/bin/env bash
#SBATCH --job-name=aggregate_bins
#SBATCH --output=aggregate_bins.%j.out
#SBATCH --error=aggregate_bins.%j.err
#SBATCH --partition=ycga
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=pattyyang1@gmail.com


module load miniconda
conda activate homogeneityScore

cd "/vast/palmer/pi/garg/Patrick/Homogeneity Score"
python src/aggregate_bins.py
