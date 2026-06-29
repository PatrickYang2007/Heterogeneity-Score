"""Generate an evaluation report (metrics + plots) for a trained model.

Produces, per split (val/test), the plots that are standard for a
sequence-to-function regression model in genomics, plus a metrics summary:

  1. pred_vs_true     hexbin density of predicted vs observed, with identity and
                      best-fit lines (the core regression diagnostic).
  2. residuals        residual (pred - true) vs true, to expose bias / range
                      where the model under- or over-predicts.
  3. distributions    overlaid histograms of true vs predicted, to check the
                      model isn't collapsing toward the mean.
  4. calibration      true binned into deciles vs mean prediction in each bin.
  5. per_chrom        Pearson r per chromosome (the split is by chromosome, so
                      this shows consistency across held-out chromosomes).
  6. roc_pr           ROC and precision-recall curves after binarizing the score
                      at --threshold (only for the bounded per-region model;
                      AUPRC is the more honest summary under class imbalance).

Metrics reported: Pearson, Spearman, R^2, RMSE, MAE, and (if binarizable)
AUROC + AUPRC. Everything is computed with numpy/scipy so no extra deps.

Run via slurm/eval.sbatch, e.g.:
  sbatch slurm/eval.sbatch --weights Models/best_model_w256_perRegion.pt \
      --window 256 --tag w256_perRegion
"""
import os
import json
import argparse

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr, rankdata

from model import HeterogeneityScoreModel, make_dataloader
from config import REGION_MASK as CFG_REGION_MASK, REGION_WIDTH as CFG_REGION_WIDTH


# ----------------------------- metrics (numpy) -----------------------------

def regression_metrics(preds, targets):
    pearson, _ = pearsonr(preds, targets)
    spearman, _ = spearmanr(preds, targets)
    ss_res = np.sum((targets - preds) ** 2)
    ss_tot = np.sum((targets - targets.mean()) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    rmse = np.sqrt(np.mean((targets - preds) ** 2))
    mae = np.mean(np.abs(targets - preds))
    return {"pearson": pearson, "spearman": spearman, "r2": r2,
            "rmse": rmse, "mae": mae}


def auroc(scores, labels):
    """AUROC via the rank (Mann-Whitney U) identity. labels in {0,1}."""
    n_pos = labels.sum()
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = rankdata(scores)
    return (ranks[labels == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def pr_curve(scores, labels):
    """Precision/recall arrays and average precision (AUPRC)."""
    order = np.argsort(-scores)
    labels = labels[order]
    tp = np.cumsum(labels)
    fp = np.cumsum(1 - labels)
    precision = tp / (tp + fp)
    recall = tp / labels.sum()
    ap = np.sum((recall - np.concatenate([[0], recall[:-1]])) * precision)
    return precision, recall, ap


def roc_curve_np(scores, labels):
    order = np.argsort(-scores)
    labels = labels[order]
    tp = np.cumsum(labels)
    fp = np.cumsum(1 - labels)
    tpr = tp / labels.sum()
    fpr = fp / (len(labels) - labels.sum())
    return fpr, tpr


# ----------------------------- prediction -----------------------------

def get_predictions(model, loader, device):
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            preds.extend(model(x).squeeze(1).cpu().tolist())
            targets.extend(y.tolist())
    return np.array(preds), np.array(targets)


# ----------------------------- plots -----------------------------

def plot_pred_vs_true(preds, targets, m, split, out_dir):
    fig, ax = plt.subplots(figsize=(6, 6))
    hb = ax.hexbin(targets, preds, gridsize=60, bins="log", cmap="viridis", mincnt=1)
    fig.colorbar(hb, ax=ax, label="log10(count)")
    lims = [min(targets.min(), preds.min()), max(targets.max(), preds.max())]
    ax.plot(lims, lims, "r--", lw=1, label="identity")
    a, b = np.polyfit(targets, preds, 1)
    xs = np.array(lims)
    ax.plot(xs, a * xs + b, "k-", lw=1, label=f"fit (slope={a:.2f})")
    ax.set_xlabel("observed score")
    ax.set_ylabel("predicted score")
    ax.set_title(f"{split}: pred vs observed\n"
                 f"r={m['pearson']:.3f}  rho={m['spearman']:.3f}  "
                 f"R2={m['r2']:.3f}  RMSE={m['rmse']:.3f}")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(f"{out_dir}/{split}_pred_vs_true.png", dpi=150)
    plt.close(fig)


def plot_residuals(preds, targets, split, out_dir):
    resid = preds - targets
    fig, ax = plt.subplots(figsize=(6, 5))
    hb = ax.hexbin(targets, resid, gridsize=60, bins="log", cmap="magma", mincnt=1)
    fig.colorbar(hb, ax=ax, label="log10(count)")
    ax.axhline(0, color="cyan", lw=1)
    ax.set_xlabel("observed score")
    ax.set_ylabel("residual (pred - observed)")
    ax.set_title(f"{split}: residuals")
    fig.tight_layout()
    fig.savefig(f"{out_dir}/{split}_residuals.png", dpi=150)
    plt.close(fig)


def plot_distributions(preds, targets, split, out_dir):
    fig, ax = plt.subplots(figsize=(6, 4))
    bins = np.linspace(min(targets.min(), preds.min()),
                       max(targets.max(), preds.max()), 60)
    ax.hist(targets, bins=bins, alpha=0.5, label="observed", density=True)
    ax.hist(preds, bins=bins, alpha=0.5, label="predicted", density=True)
    ax.set_xlabel("score")
    ax.set_ylabel("density")
    ax.set_title(f"{split}: score distributions")
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{out_dir}/{split}_distributions.png", dpi=150)
    plt.close(fig)


def plot_calibration(preds, targets, split, out_dir, n_bins=10):
    # Bin observed values into quantiles; plot mean predicted vs mean observed.
    edges = np.quantile(targets, np.linspace(0, 1, n_bins + 1))
    edges = np.unique(edges)
    idx = np.clip(np.digitize(targets, edges[1:-1]), 0, len(edges) - 2)
    mean_true, mean_pred = [], []
    for b in range(len(edges) - 1):
        sel = idx == b
        if sel.any():
            mean_true.append(targets[sel].mean())
            mean_pred.append(preds[sel].mean())
    mean_true, mean_pred = np.array(mean_true), np.array(mean_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    lims = [min(mean_true.min(), mean_pred.min()), max(mean_true.max(), mean_pred.max())]
    ax.plot(lims, lims, "r--", lw=1, label="perfect")
    ax.plot(mean_true, mean_pred, "o-", color="steelblue", label="binned mean")
    ax.set_xlabel("mean observed (quantile bin)")
    ax.set_ylabel("mean predicted")
    ax.set_title(f"{split}: calibration")
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{out_dir}/{split}_calibration.png", dpi=150)
    plt.close(fig)


def plot_per_chrom(preds, targets, chroms, split, out_dir):
    rows = []
    for c in pd.unique(chroms):
        sel = chroms == c
        if sel.sum() > 2:
            r, _ = pearsonr(preds[sel], targets[sel])
            rows.append((c, r, sel.sum()))
    rows.sort(key=lambda t: t[1])
    labels = [r[0] for r in rows]
    vals = [r[1] for r in rows]
    fig, ax = plt.subplots(figsize=(max(6, len(labels) * 0.5), 4))
    ax.bar(labels, vals, color="teal")
    ax.axhline(np.mean(vals), color="k", ls="--", lw=1, label=f"mean={np.mean(vals):.3f}")
    ax.set_ylabel("Pearson r")
    ax.set_title(f"{split}: per-chromosome Pearson")
    ax.tick_params(axis="x", rotation=45)
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{out_dir}/{split}_per_chrom.png", dpi=150)
    plt.close(fig)


def plot_roc_pr(preds, targets, threshold, split, out_dir):
    labels = (targets >= threshold).astype(int)
    if labels.sum() == 0 or labels.sum() == len(labels):
        print(f"  [{split}] threshold {threshold} gives one class only; "
              f"skipping ROC/PR")
        return None
    au = auroc(preds, labels)
    prec, rec, ap = pr_curve(preds, labels)
    fpr, tpr = roc_curve_np(preds, labels)
    baseline = labels.mean()  # prevalence = PR baseline

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))
    ax1.plot(fpr, tpr, color="darkorange", label=f"AUROC={au:.3f}")
    ax1.plot([0, 1], [0, 1], "k--", lw=1)
    ax1.set_xlabel("false positive rate")
    ax1.set_ylabel("true positive rate")
    ax1.set_title(f"{split}: ROC (score >= {threshold})")
    ax1.legend(loc="lower right")

    ax2.plot(rec, prec, color="purple", label=f"AUPRC={ap:.3f}")
    ax2.axhline(baseline, color="gray", ls="--", lw=1,
                label=f"baseline={baseline:.3f}")
    ax2.set_xlabel("recall")
    ax2.set_ylabel("precision")
    ax2.set_title(f"{split}: precision-recall")
    ax2.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(f"{out_dir}/{split}_roc_pr.png", dpi=150)
    plt.close(fig)
    return {"auroc": au, "auprc": ap, "positive_rate": baseline,
            "n_positive": int(labels.sum())}


# ----------------------------- driver -----------------------------

def evaluate_split(model, split, parquet, device, threshold, bounded, out_dir,
                   region_mask=False, region_width=16):
    print(f"[{split}] {parquet}")
    loader = make_dataloader(parquet, shuffle=False,
                             region_mask=region_mask, region_width=region_width)
    preds, targets = get_predictions(model, loader, device)
    chroms = pd.read_parquet(parquet, columns=["chrom"])["chrom"].to_numpy()

    m = regression_metrics(preds, targets)
    plot_pred_vs_true(preds, targets, m, split, out_dir)
    plot_residuals(preds, targets, split, out_dir)
    plot_distributions(preds, targets, split, out_dir)
    plot_calibration(preds, targets, split, out_dir)
    plot_per_chrom(preds, targets, chroms, split, out_dir)
    if bounded:
        cls = plot_roc_pr(preds, targets, threshold, split, out_dir)
        if cls:
            m.update(cls)

    print(f"  " + "  ".join(f"{k}={v:.4f}" for k, v in m.items()))
    return m


def write_summary(results, out_dir, args, suffix):
    """Write a small human-readable summary.txt alongside the plots."""
    lines = []
    lines.append("=" * 60)
    lines.append("  HETEROGENEITY SCORE - EVALUATION SUMMARY")
    lines.append("=" * 60)
    lines.append(f"weights      : {args.weights}")
    lines.append(f"data         : data/{{split}}{suffix}.parquet")
    lines.append(f"mode         : {'summed-bin (linear)' if args.aggregate else 'per-region (sigmoid)'}")
    lines.append(f"window       : {args.window}")
    lines.append("")
    header = f"{'metric':<14}" + "".join(f"{s:>12}" for s in results)
    lines.append(header)
    lines.append("-" * len(header))
    keys = ["pearson", "spearman", "r2", "rmse", "mae", "auroc", "auprc"]
    label = {"pearson": "Pearson r", "spearman": "Spearman rho", "r2": "R^2",
             "rmse": "RMSE", "mae": "MAE", "auroc": "AUROC", "auprc": "AUPRC"}
    for k in keys:
        if any(k in results[s] for s in results):
            row = f"{label[k]:<14}"
            for s in results:
                v = results[s].get(k)
                row += f"{v:>12.4f}" if v is not None else f"{'-':>12}"
            lines.append(row)
    lines.append("")
    lines.append("Plots in this folder (per split):")
    lines.append("  pred_vs_true  - predicted vs observed (density); slope < 1 = mean regression")
    lines.append("  calibration   - binned mean pred vs observed; flat = compressed range")
    lines.append("  residuals     - error vs observed; structure = systematic bias")
    lines.append("  distributions - observed vs predicted spread")
    lines.append("  per_chrom     - Pearson per held-out chromosome (consistency)")
    lines.append("  roc_pr        - ROC / precision-recall at score threshold (per-region only)")
    text = "\n".join(lines) + "\n"
    with open(f"{out_dir}/summary.txt", "w") as f:
        f.write(text)
    print("\n" + text)


def main():
    p = argparse.ArgumentParser(description="Evaluation report for a trained model")
    p.add_argument("--weights", required=True, help="model checkpoint (.pt)")
    p.add_argument("--window", type=int, required=True, help="window/bin size of the data")
    p.add_argument("--aggregate", action="store_true",
                   help="weights are from a summed-bin model (linear output)")
    p.add_argument("--tag", default=None, help="output subfolder name under Models/eval/")
    p.add_argument("--threshold", type=float, default=0.75,
                   help="score threshold for binary ROC/PR (per-region only)")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--num-filters", type=int, default=32)
    p.add_argument("--num-blocks", type=int, default=3,
                   help="must match the trained model's depth")
    p.add_argument("--ker-size", type=int, default=5)
    p.add_argument("--dropout", type=float, default=0.3)
    args = p.parse_args()

    suffix = f"_agg{args.window}" if args.aggregate else f"_w{args.window}"
    tag = args.tag or suffix.lstrip("_")
    out_dir = f"Models/eval/{tag}"
    os.makedirs(out_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pool = 2 if args.window else 1
    # Mirror train.py: per-region weights carry the region-mask channel (5 input
    # channels), the summed-bin path does not.
    region_mask = CFG_REGION_MASK and not args.aggregate
    in_channels = 5 if region_mask else 4
    model = HeterogeneityScoreModel(dropout=args.dropout, ker_size=args.ker_size,
                                  in_channels=in_channels,
                                  num_filters=args.num_filters, num_blocks=args.num_blocks,
                                  pool=pool, bounded=not args.aggregate)
    model.load_state_dict(torch.load(args.weights, map_location=device))
    model = model.to(device)
    print(f"Loaded {args.weights} (device={device}, bounded={not args.aggregate})")

    results = {}
    for split in ["val", "test"]:
        parquet = f"{args.data_dir}/{split}{suffix}.parquet"
        if not os.path.exists(parquet):
            print(f"[{split}] {parquet} missing, skipping")
            continue
        results[split] = evaluate_split(model, split, parquet, device,
                                        args.threshold, not args.aggregate, out_dir,
                                        region_mask=region_mask,
                                        region_width=CFG_REGION_WIDTH)

    with open(f"{out_dir}/metrics.json", "w") as f:
        json.dump(results, f, indent=2)
    write_summary(results, out_dir, args, suffix)
    print(f"Wrote plots + metrics.json + summary.txt to {out_dir}/")


if __name__ == "__main__":
    main()
