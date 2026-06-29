#!/usr/bin/env bash
#SBATCH --job-name=tests
#SBATCH --output=logs/tests.%j.out
#SBATCH --error=logs/tests.%j.err
#SBATCH --partition=ycga
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=pattyyang1@gmail.com

module load miniconda
# Initialize conda's shell functions so `conda activate` doesn't emit a
# "Run 'conda init' before 'conda deactivate'" warning to stderr.
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate heterogeneityScore

cd "/vast/palmer/pi/garg/Patrick/Heterogeneity Score"
# pytest is a dev-only dependency; install it into the env if it isn't there.
python -c "import pytest" 2>/dev/null || python -m pip install -q pytest

# The tests are fast and CPU-only (no GPU, no real data needed).
python -m pytest tests/ -v
