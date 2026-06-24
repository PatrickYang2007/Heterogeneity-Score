#!/usr/bin/env bash
#SBATCH --job-name=widen_windows
#SBATCH --output=logs/widen_windows.%j.out
#SBATCH --error=logs/widen_windows.%j.err
#SBATCH --partition=ycga
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=pattyyang1@gmail.com


module load miniconda
conda activate homogeneityScore

cd "/vast/palmer/pi/garg/Patrick/Homogeneity Score"
python src/widen_windows.py
