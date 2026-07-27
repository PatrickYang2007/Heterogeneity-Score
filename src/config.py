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

# Raw-signal target (per-region path). When True, train.py regresses the
# un-clipped specificity-entropy signal from data/{split}_w{WINDOW}_raw.parquet
# (built by attach_raw_label.py) instead of the bedgraph score. That target is
# continuous, heavy-tailed (log1p-transformed) and has no 1.0 pile-up, so the
# model uses a LINEAR head (bounded=False) and every anti-saturation lever
# (TARGET_CLIP, BALANCE_SPIKE, LOSS_WEIGHTING, bias seeding) is forced off --
# they only exist to fight the saturation this target doesn't have. Checkpoint
# and eval must load the same _raw data with bounded=False. Overridable per run
# with --raw-label/--no-raw-label. Ignored for the summed-bin (AGGREGATE) path.
RAW_LABEL = False

# Drop duplicate regions in prepare_data.py. The source bedgraph is annotated,
# not deduplicated: the same 16 bp interval is emitted once per overlapping
# annotation (gene x feature class), so a locus in a region with two overlapping
# genes appears twice with the SAME score. Measured on the current splits, 37-39%
# of rows are exact duplicates of another row (up to 9 copies of one locus).
#
# That silently reweights training: duplication multiplicity anti-correlates with
# the label (multiplicity 1 -> mean score 0.79, multiplicity 5 -> 0.49), so the
# low-score tail is oversampled 3-5x relative to the real per-locus distribution.
# It also makes val/test metrics per-ROW rather than per-LOCUS, which is not the
# quantity being reported. Leave this on unless you specifically want the old
# annotation-weighted behavior.
DEDUP_REGIONS = True

# Run MACS2 bdgpeakcall and keep only regions overlapping a called peak.
#
# OFF by default because on this bedgraph it does not work. bdgpeakcall assumes a
# position-sorted coverage track; this file is sorted by ANNOTATION CLASS first
# (it restarts the genome 12 times -- 3primeUTR, 5primeUTR, CA, ..., pELS -- with
# 5.77M chromosome switches), so MACS2 saw a wildly non-monotonic coordinate
# stream and emitted 46 nonsensical "peaks" spanning up to 203 Mb, overlapping and
# nested. Filtering on those dropped ~48% of the data on an arbitrary,
# chromosome-uneven basis (chr16 kept only 2.2 Mb of 90 Mb; chr2 kept one 203 Mb
# "peak"). prepare_data.py now sorts by position before calling MACS2, which is
# necessary but probably not sufficient -- a 0-1 bounded score is not the signal
# bdgpeakcall was designed for. Verify the peak file looks sane before trusting it.
PEAK_FILTER = False

# Crop the central CROP bp out of each window at load time (None = use the full
# window). This exists so context width can be swept WITHOUT regenerating the
# multi-GB parquet files: the data stays at WINDOW bp and the dataset hands the
# model the central CROP bp.
#
# Why it matters: sequence composition predicts the label best at ~256 bp of
# context and gets monotonically WORSE beyond that (test Pearson of a GC/CpG
# linear model: 128bp 0.590, 256bp 0.605, 512bp 0.602, 1024bp 0.588, 2048bp 0.556;
# adding wide context on top of the central 256 bp is worth +0.005). The label is
# a local property, so the 2048 bp window is ~8x more sequence than the signal
# supports -- it dilutes the attention pool, forces 5 pooling blocks, and gives
# the model 8x more to memorize (these runs overfit by epoch 2). Overridable per
# run with --crop.
CROP = None

# Concatenate the center-position features onto the attention-pooled vector
# before the output head (per-region path only).
#
# The label describes the central REGION_WIDTH bp, but AttentionPool softmaxes
# over the whole window and returns a weighted average, so the labeled region is
# one of WINDOW/pool**num_blocks positions (1 of 64 at 2048 bp with 5 blocks) and
# its contribution is averaged away. REGION_MASK marks where it is, but the head
# still only ever sees the pooled average. When CENTER_POOL is on, the head gets
# [attention_pool(x) ; x[:, :, center]] -- the global context AND the features
# actually over the labeled region. Doubles the head's input width; ignored on
# the summed-bin (AGGREGATE) path, which has no single central region.
CENTER_POOL = False


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


def effective_width(window, crop=None):
    """Sequence length the MODEL actually sees, after any center crop.

    The parquet holds `window` bp per row; --crop hands the model only the
    central `crop` bp of that. Everything downstream that depends on input
    length -- the max blocks that still fit under pooling, the region-mask
    channel's length, predict.py's cropping -- must use this, not `window`.
    """
    if not crop or not window or crop >= window:
        return window
    return crop


def max_blocks_for(length, pool):
    """Deepest block stack whose pooling does not shrink `length` below 1.

    Each pooled block divides the length by `pool`, so `pool**num_blocks` must
    stay <= length. Cropping shortens the input, which lowers this ceiling --
    e.g. 5 blocks need >= 32 bp, but a 256 bp crop leaves only 8 output
    positions for the attention pool to work with.
    """
    if not length or not pool or pool <= 1:
        return None
    n = 0
    while pool ** (n + 1) <= length:
        n += 1
    return n


def check_flags_match_checkpoint(weights, crop=None, center_pool=False):
    """Warn when eval/predict flags disagree with the checkpoint's own name tags.

    --center-pool is self-enforcing: it changes the head's shape, so a mismatch
    raises in load_state_dict. --crop is NOT. Convolutions accept any length, so
    evaluating a model trained on a 256 bp crop with the full 2048 bp window
    silently produces predictions on the wrong context width and quietly wrong
    numbers -- the worst kind of failure.

    train.py already encodes both in the checkpoint filename (_c256, _ctr), so
    compare against that. This is a heuristic on a filename, hence a warning
    rather than an error: a renamed checkpoint should not block a legitimate run.
    """
    import os
    import re

    name = os.path.basename(str(weights))
    m = re.search(r"_c(\d+)", name)
    tagged_crop = int(m.group(1)) if m else None
    tagged_center = "_ctr" in name

    problems = []
    if tagged_crop != (crop or None):
        problems.append(
            f"checkpoint name implies crop={tagged_crop or 'none'} but "
            f"--crop {crop or 'none'} was given")
    if tagged_center != bool(center_pool):
        problems.append(
            f"checkpoint name implies center_pool={tagged_center} but "
            f"--center-pool {'was' if center_pool else 'was not'} given")
    for p in problems:
        print(f"  !! WARNING: {p}")
    if problems:
        print("  !! A --crop mismatch does NOT raise -- conv layers accept any "
              "input length -- so the metrics below would be silently wrong. "
              "Check the flags against how the model was trained.")
    return not problems
