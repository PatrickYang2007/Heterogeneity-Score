#!/usr/bin/env bash
#SBATCH --job-name=attribute
#SBATCH --output=logs/attribute.%j.out
#SBATCH --error=logs/attribute.%j.err
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=pattyyang1@gmail.com

module load miniconda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate heterogeneityScore

cd "/vast/palmer/pi/garg/Patrick/Heterogeneity Score"
# Args pass through to src/attribute.py, e.g.:
#   sbatch slurm/attribute.sbatch data/val_w2048.parquet \
#       --weights Models/best_model_w2048_b5_f64_mask_monpearson.pt \
#       --window 2048 --num-blocks 5 --num-filters 64 \
#       --method ism --method deeplift --n-regions 256 --out attr_val.npz
#
# Cost note: ISM is 4*L forward passes PER window (8192 at 2048 bp), so runtime
# scales with n_regions * window. 256 windows at 2048 bp is ~2M forward passes
# through a 5-block/64-filter stack -- fine on a GPU, hours on CPU. Start small
# and check the printed ISM-vs-DeepLIFT agreement before scaling up.
python src/attribute.py "$@"
