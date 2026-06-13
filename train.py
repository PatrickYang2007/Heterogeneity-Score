import torch
from model import HomogeneityScoreModel
from model_train import Trainer
from model import make_dataloader
from evaluate import plot_loss_curves

DATA_DIR = "data"
OUT_DIR = "."

NUM_FILTERS = 32
KER_SIZE = 5
DROPOUT = 0.3
LR = 1e-3
WEIGHT_DECAY = 1e-4
GRAD_CLIP = 1.0
PATIENCE = 10
EARLY_STOPPING = False
EPOCHS = 100
BATCH_SIZE = 64


def main():
    train_loader = make_dataloader(f"{DATA_DIR}/train.parquet", batch_size = BATCH_SIZE)
    val_loader = make_dataloader(f"{DATA_DIR}/val.parquet", batch_size = BATCH_SIZE, shuffle = False)

    model = HomogeneityScoreModel(dropout = DROPOUT, ker_size = KER_SIZE, num_filters = NUM_FILTERS)

    trainer = Trainer(model, train_loader, val_loader, num_epochs=EPOCHS, lr=LR,
                      weight_decay=WEIGHT_DECAY, grad_clip=GRAD_CLIP, patience=PATIENCE,
                      early_stopping=EARLY_STOPPING)
    train_losses, val_losses = trainer.fit()

    plot_loss_curves(train_losses, val_losses, out_dir=OUT_DIR)
    print(f"best val loss: {trainer.best_val_loss:.4f}")
    print(f"model saved to best_model.pt")


if __name__ == "__main__":
    main()
