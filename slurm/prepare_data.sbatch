#!/usr/bin/env bash
#SBATCH --job-name=prepare_data
#SBATCH --output=logs/prepare_data.%j.out
#SBATCH --error=logs/prepare_data.%j.err
#SBATCH --partition=ycga
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=pattyyang1@gmail.com


module load miniconda
# Initialize conda's shell functions so `conda activate` doesn't emit a
# "Run 'conda init' before 'conda deactivate'" warning to stderr.
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate heterogeneityScore

cd "/vast/palmer/pi/garg/Patrick/Heterogeneity Score"
python src/prepare_data.py
