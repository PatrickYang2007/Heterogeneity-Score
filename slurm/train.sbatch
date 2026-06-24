#!/usr/bin/env bash
#SBATCH --job-name=train_model
#SBATCH --output=logs/train_model.%j.out
#SBATCH --error=logs/train_model.%j.err
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=pattyyang1@gmail.com

module load miniconda
# Initialize conda's shell functions so `conda activate` doesn't emit a
# "Run 'conda init' before 'conda deactivate'" warning to stderr.
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate homogeneityScore

cd "/vast/palmer/pi/garg/Patrick/Homogeneity Score"
# Args after the script name are passed through, e.g.:
#   sbatch slurm/train.sbatch                       (per-region, from config)
#   sbatch slurm/train.sbatch --aggregate           (summed-bin)
#   sbatch slurm/train.sbatch --window 256 --no-aggregate
python src/train.py "$@"
