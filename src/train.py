import os
import argparse
import torch
from model import HomogeneityScoreModel
from model_train import Trainer
from model import make_dataloader
from evaluate import plot_loss_curves
from config import WINDOW as CFG_WINDOW, AGGREGATE as CFG_AGGREGATE

DATA_DIR = "data"
OUT_DIR = "Models"   # checkpoints and loss curve are written here

NUM_FILTERS = 32
KER_SIZE = 5
DROPOUT = 0.3
LR = 1e-3
WEIGHT_DECAY = 1e-4
GRAD_CLIP = 1.0
PATIENCE = 10
EARLY_STOPPING = True
EPOCHS = 100
BATCH_SIZE = 64


def parse_args():
    # window/aggregate default to config.py, but can be overridden per run so two
    # jobs (e.g. per-region and summed-bin) can train in parallel without sharing
    # one global config value. --no-aggregate forces the per-region path.
    parser = argparse.ArgumentParser(description="Train HomogeneityScoreModel")
    parser.add_argument("--window", type=int, default=CFG_WINDOW,
                        help=f"sequence window width (default from config: {CFG_WINDOW})")
    parser.add_argument("--aggregate", dest="aggregate", action="store_true",
                        help="train on the summed-bin data (linear output)")
    parser.add_argument("--no-aggregate", dest="aggregate", action="store_false",
                        help="train on the per-region data (sigmoid output)")
    parser.set_defaults(aggregate=CFG_AGGREGATE)
    return parser.parse_args()


def main():
    args = parse_args()
    window, aggregate = args.window, args.aggregate

    # Per-block max-pool factor. Use 2 for wide windows (grows receptive field);
    # 1 falls back to the original no-pooling model for 16 bp inputs.
    pool = 2 if window else 1

    if aggregate:
        suffix = f"_agg{window}"
    else:
        suffix = f"_w{window}" if window else ""
    print(f"Training: window={window}  aggregate={aggregate}  -> data{suffix}.parquet")

    train_loader = make_dataloader(f"{DATA_DIR}/train{suffix}.parquet", batch_size = BATCH_SIZE)
    val_loader = make_dataloader(f"{DATA_DIR}/val{suffix}.parquet", batch_size = BATCH_SIZE, shuffle = False)

    # Each experiment saves to its own checkpoint/plot under Models/ so parallel
    # runs don't overwrite each other (e.g. best_model_w2048.pt vs _agg2048.pt).
    os.makedirs(OUT_DIR, exist_ok=True)
    checkpoint_path = f"{OUT_DIR}/best_model{suffix}.pt"

    # Summed-bin labels are unbounded, so drop the final sigmoid (bounded=False).
    model = HomogeneityScoreModel(dropout = DROPOUT, ker_size = KER_SIZE,
                                  num_filters = NUM_FILTERS, pool = pool,
                                  bounded = not aggregate)

    trainer = Trainer(model, train_loader, val_loader, num_epochs=EPOCHS, lr=LR,
                      weight_decay=WEIGHT_DECAY, grad_clip=GRAD_CLIP, patience=PATIENCE,
                      early_stopping=EARLY_STOPPING, checkpoint_path=checkpoint_path)
    train_losses, val_losses = trainer.fit()

    plot_loss_curves(train_losses, val_losses, out_dir=OUT_DIR,
                     filename=f"loss_curves{suffix}.png")
    print(f"best val pearson: {trainer.best_val_corr:.4f}  (val loss at that epoch tracked separately)")
    print(f"model saved to {checkpoint_path}")


if __name__ == "__main__":
    main()
