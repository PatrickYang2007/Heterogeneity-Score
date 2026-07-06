"""Shared experiment configuration.

Edit WINDOW (and AGGREGATE) here ONCE; the data-prep scripts (widen_windows.py,
aggregate_bins.py) and train.py all import these, so the window size stays in sync
across the whole pipeline instead of being set in three separate files.

Workflow when you change WINDOW:
  1. Set WINDOW (and AGGREGATE) below.
  2. Regenerate the matching data for that size (run once):
       AGGREGATE = False -> widen_windows.py   -> data/{split}_w{WINDOW}.parquet
       AGGREGATE = True  -> aggregate_bins.py   -> data/{split}_agg{WINDOW}.parquet
  3. Train: train.py picks the right files and checkpoint automatically.
"""

# Sequence window width in bp. Set to None to use the original 16 bp regions
# (per-region path only).
WINDOW = 2048

# Summed-bin experiment toggle.
#   False -> per-region score, data/{split}_w{WINDOW}.parquet, sigmoid output.
#   True  -> summed-bin label,  data/{split}_agg{WINDOW}.parquet, linear output.
AGGREGATE = False

# Region-mask input channel (per-region path only). When WINDOW widens each 16 bp
# region into a much larger context window, the label still depends on only those
# central REGION_WIDTH bp, but a pure conv+pool stack is position-blind and the
# attention pool averages the whole window, diluting the labeled region's signal
# ~WINDOW/REGION_WIDTH-fold. That collapses the output to a constant (Pearson nan).
# REGION_MASK adds a 5th input channel that is 1.0 over the central REGION_WIDTH
# positions and 0.0 elsewhere, telling the model which positions the score is
# about so attention can anchor on them while still seeing the full context.
# Ignored for the summed-bin (AGGREGATE) path, which has no single region.
REGION_MASK = True

# Width (bp) of the original labeled region at the center of each window. The raw
# bedgraph regions are 16 bp, and widen_windows.py centers the window on the
# region midpoint, so the region occupies the central REGION_WIDTH positions.
REGION_WIDTH = 16

# Train-only class rebalancing (per-region path). The per-region labels pile up
# at exactly 1.0 (~41% of rows); under MSE that spike dominates the gradient and
# drags predictions toward the high mean (range compression / regression to the
# mean). When BALANCE_SPIKE is True, the TRAIN split thins the
# score>=SPIKE_THRESHOLD pile-up down to SPIKE_KEEP_FRAC of its rows so the model
# optimizes against a flatter score distribution. Only the training data is
# touched -- val/test always keep the real distribution so their metrics stay
# comparable to non-balanced runs. Overridable per run with
# --balance/--no-balance and --cap-frac on train.py.
BALANCE_SPIKE = False
SPIKE_THRESHOLD = 1.0   # rows with score >= this are the spike to thin
SPIKE_KEEP_FRAC = 0.3   # fraction of spike rows to keep (0.3 -> ~41% spike becomes ~17%)

# Inverse-density loss weighting (per-region path). An alternative to spike
# downsampling that keeps every row: each training example is weighted by the
# inverse frequency of its score bin, so the sparse low-score tail contributes
# as much gradient as the dense 1.0 pile-up. This directly counters range
# compression / regression to the mean (best-fit slope < 1) without throwing
# away data. None -> plain MSE; "inv_density" -> weighted. Overridable per run
# with --loss-weighting {none,inv_density}. Ignored for the summed-bin path.
LOSS_WEIGHTING = None

# Which validation metric selects the saved checkpoint and drives early
# stopping. "loss" (min MSE) is the historical default, but under a label that
# piles up at 1.0 the min-MSE model is the MORE mean-regressed one, so it works
# against the reported goal. "pearson" (max) saves the model that best preserves
# rank/spread -- the metric eval_report actually cares about. Overridable per
# run with --monitor {loss,pearson}.
MONITOR = "loss"


# --------------------------------------------------------------------------
# Derived settings. train.py / predict.py / eval_report.py all need the same
# window->pooling and per-region-vs-aggregate->channel rules; deriving them here
# keeps the three entry points from drifting out of sync.
# --------------------------------------------------------------------------

def pool_for_window(window):
    """Per-block max-pool factor for a given window width.

    2 halves the length each block so a wide window grows the receptive field;
    1 is the no-pool path for the original 16 bp inputs (window None/0).
    """
    return 2 if window else 1


def region_mask_enabled(aggregate):
    """Whether the region-mask input channel is used.

    Only the per-region path carries it; the summed-bin (aggregate) label has no
    single region to mark. Gated by the REGION_MASK toggle above.
    """
    return REGION_MASK and not aggregate


def in_channels_for(region_mask):
    """Model input channels: 4 base one-hot, plus 1 for the region mask."""
    return 5 if region_mask else 4
