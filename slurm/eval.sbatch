#!/usr/bin/env bash
#SBATCH --job-name=eval_report
#SBATCH --output=logs/eval_report.%j.out
#SBATCH --error=logs/eval_report.%j.err
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=pattyyang1@gmail.com

module load miniconda
# Initialize conda's shell functions so `conda activate` doesn't emit a
# "Run 'conda init' before 'conda deactivate'" warning to stderr.
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate homogeneityScore

cd "/vast/palmer/pi/garg/Patrick/Homogeneity Score"
# Args are passed through, e.g.:
#   sbatch slurm/eval.sbatch --weights Models/best_model_w256_perRegion.pt \
#       --window 256 --tag w256_perRegion
python src/eval_report.py "$@"
