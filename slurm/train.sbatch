#!/usr/bin/env bash
#SBATCH --job-name=train_model
#SBATCH --output=train_model.%j.out
#SBATCH --error=train_model.%j.err
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=pattyyang1@gmail.com

module load miniconda
conda activate homogeneityScore

cd "/vast/palmer/pi/garg/Patrick/Homogeneity Score"
python src/train.py
