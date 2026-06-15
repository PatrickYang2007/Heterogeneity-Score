import torch
from model import HomogeneityScoreModel
from model_train import Trainer
from model import make_dataloader
from evaluate import plot_loss_curves

DATA_DIR = "data"
OUT_DIR = "."

# Sequence window width. Must match what was produced by widen_windows.py /
# prepare_data.py. Set to None to train on the original 16 bp regions
# (train_<split>.parquet); otherwise the wide files train_w{WINDOW}.parquet etc.
WINDOW = 256

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
    suffix = f"_w{WINDOW}" if WINDOW else ""
    train_loader = make_dataloader(f"{DATA_DIR}/train{suffix}.parquet", batch_size = BATCH_SIZE)
    val_loader = make_dataloader(f"{DATA_DIR}/val{suffix}.parquet", batch_size = BATCH_SIZE, shuffle = False)

    model = HomogeneityScoreModel(dropout = DROPOUT, ker_size = KER_SIZE,
                                  num_filters = NUM_FILTERS, pool = POOL)

    trainer = Trainer(model, train_loader, val_loader, num_epochs=EPOCHS, lr=LR,
                      weight_decay=WEIGHT_DECAY, grad_clip=GRAD_CLIP, patience=PATIENCE,
                      early_stopping=EARLY_STOPPING)
    train_losses, val_losses = trainer.fit()

    plot_loss_curves(train_losses, val_losses, out_dir=OUT_DIR)
    print(f"best val pearson: {trainer.best_val_corr:.4f}  (val loss at that epoch tracked separately)")
    print(f"model saved to best_model.pt")


if __name__ == "__main__":
    main()
