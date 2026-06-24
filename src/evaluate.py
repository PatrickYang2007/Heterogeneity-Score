import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr

from model import HomogeneityScoreModel, make_dataloader


def get_predictions(model, loader, device):
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            preds = model(x).squeeze(1).cpu().tolist()
            all_preds.extend(preds)
            all_targets.extend(y.tolist())

    return np.array(all_preds), np.array(all_targets)


def plot_scatter(preds, targets, split, out_dir):
    pearson, _ = pearsonr(preds, targets)
    spearman, _ = spearmanr(preds, targets)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(targets, preds, alpha=0.3, s=5, color='steelblue')
    ax.set_xlabel("true score")
    ax.set_ylabel("predicted score")
    ax.set_title(f"{split}  pearson={pearson:.3f}  spearman={spearman:.3f}")

    lims = [min(targets.min(), preds.min()), max(targets.max(), preds.max())]
    ax.plot(lims, lims, 'r--', linewidth=1)

    plt.tight_layout()
    plt.savefig(f"{out_dir}/{split}_scatter.png", dpi=150)
    plt.close()
    return pearson, spearman


def plot_loss_curves(train_losses, val_losses, out_dir):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(train_losses, label="train")
    ax.plot(val_losses, label="val")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"{out_dir}/loss_curves.png", dpi=150)
    plt.close()


def evaluate(weight_file, test_parquet, val_parquet, out_dir=".",
             num_filters=32, ker_size=5, dropout=0.3, pool=2, bounded=True):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # bounded must match how the model was trained: False for summed-bin
    # (aggregate) weights, True for the per-region score model.
    model = HomogeneityScoreModel(dropout=dropout, ker_size=ker_size,
                                  num_filters=num_filters, pool=pool, bounded=bounded)
    model.load_state_dict(torch.load(weight_file, map_location=device))
    model = model.to(device)

    test_loader = make_dataloader(test_parquet, shuffle=False)
    val_loader = make_dataloader(val_parquet, shuffle=False)

    for split, loader in [("val", val_loader), ("test", test_loader)]:
        preds, targets = get_predictions(model, loader, device)
        pearson, spearman = plot_scatter(preds, targets, split, out_dir)
        print(f"{split}  pearson={pearson:.4f}  spearman={spearman:.4f}")
