import os
import torch
from model import HomogeneityScoreModel
from model_train import Trainer
from model import make_dataloader
from evaluate import plot_loss_curves
from config import WINDOW, AGGREGATE

DATA_DIR = "data"
OUT_DIR = "Models"   # checkpoints and loss curve are written here

NUM_FILTERS = 32
KER_SIZE = 5
# Per-block max-pool factor. Use 2 for wide windows (grows receptive field);
# 1 falls back to the original no-pooling model for 16 bp inputs.
POOL = 2 if WINDOW else 1
DROPOUT = 0.3
LR = 1e-3
WEIGHT_DECAY = 1e-4
GRAD_CLIP = 1.0
PATIENCE = 10
EARLY_STOPPING = False
EPOCHS = 100
BATCH_SIZE = 64


def main():
    if AGGREGATE:
        suffix = f"_agg{WINDOW}"
    else:
        suffix = f"_w{WINDOW}" if WINDOW else ""
    train_loader = make_dataloader(f"{DATA_DIR}/train{suffix}.parquet", batch_size = BATCH_SIZE)
    val_loader = make_dataloader(f"{DATA_DIR}/val{suffix}.parquet", batch_size = BATCH_SIZE, shuffle = False)

    # Each experiment saves to its own checkpoint under Models/ so runs don't
    # overwrite each other, e.g. Models/best_model_w256.pt vs best_model_agg256.pt.
    os.makedirs(OUT_DIR, exist_ok=True)
    checkpoint_path = f"{OUT_DIR}/best_model{suffix}.pt"

    # Summed-bin labels are unbounded, so drop the final sigmoid (bounded=False).
    model = HomogeneityScoreModel(dropout = DROPOUT, ker_size = KER_SIZE,
                                  num_filters = NUM_FILTERS, pool = POOL,
                                  bounded = not AGGREGATE)

    trainer = Trainer(model, train_loader, val_loader, num_epochs=EPOCHS, lr=LR,
                      weight_decay=WEIGHT_DECAY, grad_clip=GRAD_CLIP, patience=PATIENCE,
                      early_stopping=EARLY_STOPPING, checkpoint_path=checkpoint_path)
    train_losses, val_losses = trainer.fit()

    plot_loss_curves(train_losses, val_losses, out_dir=OUT_DIR)
    print(f"best val pearson: {trainer.best_val_corr:.4f}  (val loss at that epoch tracked separately)")
    print(f"model saved to {checkpoint_path}")


if __name__ == "__main__":
    main()
