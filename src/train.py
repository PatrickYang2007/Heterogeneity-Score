import os
import math
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from model import HeterogeneityScoreModel, make_dataloader
from trainer import Trainer
from config import (WINDOW as CFG_WINDOW, AGGREGATE as CFG_AGGREGATE,
                    REGION_MASK as CFG_REGION_MASK, REGION_WIDTH as CFG_REGION_WIDTH,
                    BALANCE_SPIKE as CFG_BALANCE_SPIKE,
                    SPIKE_THRESHOLD as CFG_SPIKE_THRESHOLD,
                    SPIKE_KEEP_FRAC as CFG_SPIKE_KEEP_FRAC)


def plot_loss_curves(train_losses, val_losses, out_dir, filename="loss_curves.png"):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(train_losses, label="train")
    ax.plot(val_losses, label="val")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{out_dir}/{filename}", dpi=150)
    plt.close(fig)

DATA_DIR = "data"
OUT_DIR = "Models"   # checkpoints and loss curve are written here

NUM_FILTERS = 32      # width: channels in the first conv block
NUM_BLOCKS = 3        # depth: number of conv blocks (channels double each block)
KER_SIZE = 5
DROPOUT = 0.2
LR = 1e-3
WEIGHT_DECAY = 1e-4
GRAD_CLIP = 1.0
PATIENCE = 25
EARLY_STOPPING = True
EPOCHS = 100
BATCH_SIZE = 256
# Per-region labels are bounded [0, 1] but pile up at exactly 1.0 (~41% of rows).
# A sigmoid can only reach 1.0 in the limit, so an unclipped target drags the
# output onto the saturating rail where the gradient vanishes and the model
# collapses to a constant. Squashing the target into this range keeps the
# optimum on the steep part of the sigmoid. Only applied to the bounded head.
TARGET_CLIP = (0.02, 0.98)


def parse_args():
    # window/aggregate default to config.py, but can be overridden per run so two
    # jobs (e.g. per-region and summed-bin) can train in parallel without sharing
    # one global config value. --no-aggregate forces the per-region path.
    parser = argparse.ArgumentParser(description="Train HeterogeneityScoreModel")
    parser.add_argument("--window", type=int, default=CFG_WINDOW,
                        help=f"sequence window width (default from config: {CFG_WINDOW})")
    parser.add_argument("--aggregate", dest="aggregate", action="store_true",
                        help="train on the summed-bin data (linear output)")
    parser.add_argument("--no-aggregate", dest="aggregate", action="store_false",
                        help="train on the per-region data (sigmoid output)")
    parser.set_defaults(aggregate=CFG_AGGREGATE)
    # Model capacity, overridable per run so complexity sweeps don't need edits.
    parser.add_argument("--num-filters", type=int, default=NUM_FILTERS,
                        help=f"width: first-block channels (default {NUM_FILTERS})")
    parser.add_argument("--num-blocks", type=int, default=NUM_BLOCKS,
                        help=f"depth: number of conv blocks (default {NUM_BLOCKS})")
    # Train-only spike rebalancing (per-region path); defaults from config.py.
    parser.add_argument("--balance", dest="balance", action="store_true",
                        help="thin the score>=threshold spike in the TRAIN split")
    parser.add_argument("--no-balance", dest="balance", action="store_false",
                        help="train on the real (unbalanced) distribution")
    parser.set_defaults(balance=CFG_BALANCE_SPIKE)
    parser.add_argument("--cap-frac", type=float, default=CFG_SPIKE_KEEP_FRAC,
                        help=f"fraction of spike rows to keep (default {CFG_SPIKE_KEEP_FRAC})")
    parser.add_argument("--cap-threshold", type=float, default=CFG_SPIKE_THRESHOLD,
                        help=f"score >= this is the spike to thin (default {CFG_SPIKE_THRESHOLD})")
    return parser.parse_args()


def main():
    args = parse_args()
    window, aggregate = args.window, args.aggregate
    num_filters, num_blocks = args.num_filters, args.num_blocks
    # Spike rebalancing only makes sense for the bounded per-region label (the
    # summed-bin label has no 1.0 pile-up), so force it off in aggregate mode.
    balance = args.balance and not aggregate

    # Per-block max-pool factor. Use 2 for wide windows (grows receptive field);
    # 1 falls back to the original no-pooling model for 16 bp inputs.
    pool = 2 if window else 1

    # Region-mask channel only applies to the per-region path (the summed-bin
    # label has no single region to mark). When on, the input gains a 5th channel,
    # so the model's first conv must take in_channels=5.
    region_mask = CFG_REGION_MASK and not aggregate
    in_channels = 5 if region_mask else 4

    # Data suffix depends only on window/aggregate (architecture doesn't change
    # the data). The arch tag is appended to OUTPUT names only when capacity is
    # non-default, so complexity sweeps get distinct checkpoints without
    # renaming the existing default-arch files.
    if aggregate:
        suffix = f"_agg{window}"
    else:
        suffix = f"_w{window}" if window else ""
    arch = ""
    if (num_blocks, num_filters) != (NUM_BLOCKS, NUM_FILTERS):
        arch = f"_b{num_blocks}_f{num_filters}"
    # Mask runs save under a distinct name so they don't overwrite (or get
    # confused with) the no-mask baseline's checkpoint and loss curve. Balanced
    # runs get their own tag too (e.g. _bal30 = spike thinned to 30% kept).
    feat = "_mask" if region_mask else ""
    if balance:
        feat += f"_bal{int(round(args.cap_frac * 100))}"
    print(f"Training: window={window}  aggregate={aggregate}  "
          f"blocks={num_blocks}  filters={num_filters}  region_mask={region_mask}  "
          f"balance={balance}"
          + (f" (keep {args.cap_frac:g} of score>={args.cap_threshold:g})" if balance else "")
          + f"  -> data{suffix}.parquet")

    # Balancing thins only the TRAIN spike; val keeps the real distribution so its
    # loss/Pearson stay comparable to non-balanced runs.
    train_loader = make_dataloader(f"{DATA_DIR}/train{suffix}.parquet", batch_size = BATCH_SIZE,
                                   region_mask = region_mask, region_width = CFG_REGION_WIDTH,
                                   balance = balance, cap_threshold = args.cap_threshold,
                                   cap_frac = args.cap_frac)
    val_loader = make_dataloader(f"{DATA_DIR}/val{suffix}.parquet", batch_size = BATCH_SIZE,
                                 shuffle = False, region_mask = region_mask,
                                 region_width = CFG_REGION_WIDTH)

    # Each experiment saves to its own checkpoint/plot under Models/ so parallel
    # runs don't overwrite each other (e.g. best_model_w2048.pt vs _agg2048.pt,
    # or best_model_w2048_b5_f64.pt for a deeper/wider sweep).
    os.makedirs(OUT_DIR, exist_ok=True)
    checkpoint_path = f"{OUT_DIR}/best_model{suffix}{arch}{feat}.pt"

    # Summed-bin labels are unbounded, so drop the final sigmoid (bounded=False)
    # and skip the target clipping / bias seeding (those only make sense for the
    # bounded [0, 1] per-region head).
    bounded = not aggregate
    label_clip = TARGET_CLIP if bounded else None
    bias_init = None
    if bounded:
        # Seed the sigmoid output at the (clipped) label mean: logit(mean). This
        # starts the output near the data mean instead of 0.5, so it doesn't have
        # to climb toward the saturating rail to cut loss.
        lo, hi = TARGET_CLIP
        mean = float(train_loader.dataset.y.clamp(lo, hi).mean())
        bias_init = math.log(mean / (1.0 - mean))
        print(f"clipping targets to {TARGET_CLIP}; seeding output bias at "
              f"logit({mean:.4f})={bias_init:.4f}")

    model = HeterogeneityScoreModel(dropout = DROPOUT, ker_size = KER_SIZE,
                                  in_channels = in_channels,
                                  num_filters = num_filters, num_blocks = num_blocks,
                                  pool = pool, bounded = bounded, bias_init = bias_init)

    trainer = Trainer(model, train_loader, val_loader, num_epochs=EPOCHS, lr=LR,
                      weight_decay=WEIGHT_DECAY, grad_clip=GRAD_CLIP, patience=PATIENCE,
                      early_stopping=EARLY_STOPPING, checkpoint_path=checkpoint_path,
                      label_clip=label_clip)
    train_losses, val_losses = trainer.fit()

    plot_loss_curves(train_losses, val_losses, out_dir=OUT_DIR,
                     filename=f"loss_curves{suffix}{arch}{feat}.png")
    print(f"best val pearson: {trainer.best_val_corr:.4f}  (val loss at that epoch tracked separately)")
    print(f"model saved to {checkpoint_path}")


if __name__ == "__main__":
    main()
