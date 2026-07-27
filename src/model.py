import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np

from prepare_data import encode_indices


def region_mask_channel(length, region_width):
    """A (1, length) tensor that is 1.0 over the central `region_width` positions.

    Marks where the labeled 16 bp region sits inside a widened, region-centered
    window so the model gets explicit position information a translation-invariant
    conv stack otherwise lacks. Shared by GenomicDataset (training/val) and
    predict.py so the extra channel is built identically everywhere.
    """
    mask = torch.zeros(1, length)
    lo = length // 2 - region_width // 2
    mask[0, lo:lo + region_width] = 1.0
    return mask


def downsample_spike(df, threshold, keep_frac, seed=0):
    """Randomly drop rows whose score is >= `threshold` so only `keep_frac` of
    them survive; rows below the threshold are left untouched.

    The per-region labels pile up at exactly 1.0 (~41% of rows). Under MSE that
    spike dominates the gradient and pulls predictions toward the high mean
    (range compression / regression to the mean). Thinning it on the TRAIN split
    rebalances the distribution the model optimizes against. Apply this to train
    only -- val/test must stay on the real distribution or their metrics stop
    being comparable across runs. Seeded so the subsample is reproducible.
    """
    if keep_frac is None or keep_frac >= 1.0:
        return df
    spike = df["score"].to_numpy() >= threshold
    spike_idx = np.flatnonzero(spike)
    n_keep = int(round(len(spike_idx) * keep_frac))
    rng = np.random.default_rng(seed)
    keep_spike = rng.choice(spike_idx, size=n_keep, replace=False)
    keep = ~spike                       # keep every below-threshold row
    keep[keep_spike] = True             # plus the sampled fraction of the spike
    out = df.iloc[keep].reset_index(drop=True)
    print(f"  downsample_spike: score>={threshold}: {len(spike_idx)} -> {n_keep} "
          f"kept (keep_frac={keep_frac}); rows {len(df)} -> {len(out)}")
    return out


def center_crop(x, crop):
    """Keep the central `crop` positions along the last axis of `x`.

    Lets a context-width sweep reuse the existing WINDOW bp parquet instead of
    regenerating multi-GB files per width: the row still stores 2048 bp, the model
    is handed the central `crop` bp. Because widen_windows.py centers every window
    on its region's midpoint, the labeled region stays centered after cropping.
    """
    if not crop or crop >= x.shape[-1]:
        return x
    lo = x.shape[-1] // 2 - crop // 2
    return x[..., lo:lo + crop]


class GenomicDataset(Dataset):
    def __init__(self, parquet_path, region_mask=False, region_width=16,
                 balance=False, cap_threshold=1.0, cap_frac=0.3, cap_seed=0,
                 rc_augment=False, crop=None):
        df = pd.read_parquet(parquet_path, columns=["sequence", "score"])
        # Optional train-only rebalancing: when `balance` is on, thin the
        # score>=cap_threshold spike down to cap_frac of its rows. Off by default,
        # so val/test/eval/predict always see the real, untouched distribution.
        if balance:
            df = downsample_spike(df, cap_threshold, cap_frac, cap_seed)
        # Crop the SEQUENCE STRINGS before encoding, not the encoded tensor after.
        # Cropping afterwards still builds the full-width encoding first, so a
        # --crop 256 run peaked at the same memory as the full 2048 bp one and
        # OOM-killed identically. Slicing here makes a narrow run genuinely ~8x
        # lighter, which is the whole point of being able to sweep context width.
        seq = df["sequence"]
        if crop and crop < len(seq.iat[0]):
            lo = len(seq.iat[0]) // 2 - crop // 2
            seq = seq.str.slice(lo, lo + crop)
        sequences = seq.str.upper().tolist()
        # Drop the frame and the intermediate Series before encoding: each holds a
        # full copy of the sequence data, and the encode step needs headroom.
        del seq
        df = df[["score"]].copy()
        # Store sequences as compact int8 indices (A/C/G/T=0-3, N/other=4) and
        # build the one-hot tensor on the fly in __getitem__. This keeps memory
        # flat as the window grows: a preloaded float32 one-hot would be ~18 GB
        # at 256 bp / ~36 GB at 512 bp for 4.4M rows, blowing the job's RAM,
        # whereas int8 indices are ~1-2 GB.
        self.x = torch.from_numpy(encode_indices(sequences))
        del sequences
        # Belt-and-braces: the strings were already sliced above, so this is a
        # no-op on that path. It still guards the case where a caller hands in
        # pre-encoded rows wider than `crop`.
        self.x = center_crop(self.x, crop)
        self.y = torch.tensor(df["score"].values, dtype=torch.float32)
        # Optional 5th channel marking the central region. It's identical for
        # every row (windows are region-centered), so build it once here and
        # concatenate in __getitem__ rather than rebuilding it per access.
        self.region_mask = region_mask
        self._mask = (region_mask_channel(self.x.shape[1], region_width)
                      if region_mask else None)
        # Reverse-complement augmentation (train only). DNA is double-stranded, so
        # a window and its reverse complement describe the same locus and share the
        # label; showing the model both, chosen at random per access, roughly
        # doubles the effective data and is a strong regularizer against the fast
        # overfitting a from-scratch conv stack shows here. Must be OFF for val/test
        # so their metrics stay deterministic and comparable.
        self.rc_augment = rc_augment

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        idx_seq = self.x[idx].long()                       # (L,) values 0-4
        onehot = F.one_hot(idx_seq.clamp(max=3), num_classes=4).float()
        onehot[idx_seq == 4] = 0.0                          # N -> all-zero vector
        x = onehot.transpose(0, 1)                          # (4, L)
        if self.rc_augment and torch.rand(1).item() < 0.5:
            # Reverse complement = reverse along length AND complement bases.
            # Channel order is A,C,G,T (0,1,2,3), so complementing swaps A<->T
            # (0<->3) and C<->G (1<->2). Applied while still 4-channel; the region
            # mask is symmetric about the center, so it's added afterward unchanged.
            x = torch.flip(x, dims=[1])[[3, 2, 1, 0]]
        if self.region_mask:
            x = torch.cat([x, self._mask], dim=0)           # (5, L)
        return x, self.y[idx]


def make_dataloader(parquet_path, batch_size = 64, shuffle = True, num_workers = 4,
                    region_mask = False, region_width = 16,
                    balance = False, cap_threshold = 1.0, cap_frac = 0.3, cap_seed = 0,
                    rc_augment = False, crop = None):
    ds = GenomicDataset(parquet_path, region_mask=region_mask, region_width=region_width,
                        balance=balance, cap_threshold=cap_threshold,
                        cap_frac=cap_frac, cap_seed=cap_seed, rc_augment=rc_augment,
                        crop=crop)
    return DataLoader(ds, batch_size = batch_size, shuffle = shuffle,
                      num_workers = num_workers, pin_memory = True)


def conv_block(dim, dim_out, ker_size, dropout, pool=2):
    # 'same' padding keeps length fixed through the conv; an optional MaxPool
    # then halves the length. Stacking pooled blocks grows the receptive field
    # geometrically, so a wide input window is actually integrated over long
    # range instead of only the ~13 bp a stack of unpadded k=5 convs would see.
    layers = [
        nn.BatchNorm1d(dim),
        nn.GELU(),
        nn.Conv1d(dim, dim_out, ker_size, padding=ker_size // 2),
        nn.Dropout(dropout),
    ]
    if pool and pool > 1:
        layers.append(nn.MaxPool1d(pool))
    return nn.Sequential(*layers)


class AttentionPool(nn.Module):
    def __init__(self, channels, hidden=32):
        super().__init__()
        self.scores = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        # x shape comes in as (batch, channels, L)
        # but nn.Linear wants (batch, L, channels)
        x = x.transpose(1, 2)
        scores = self.scores(x)
        weights = scores.softmax(dim = 1)
        output = (x * weights).sum(dim = 1)
        return output


class HeterogeneityScoreModel(nn.Module):
    def __init__(self, dropout, ker_size=5, in_channels=4, num_filters=32,
                 num_blocks=3, pool=2, bounded=True, bias_init=None,
                 center_pool=False):
        super().__init__()
        # Stack `num_blocks` conv blocks; channels double each block
        # (num_filters, num_filters*2, num_filters*4, ...). Raise num_blocks for
        # depth or num_filters for width to add capacity without editing layers.
        #
        # pool=1 reproduces the old no-pooling behavior (sensible for 16 bp
        # inputs); pool=2 (default) halves length each block, so the input length
        # must stay > 2**num_blocks (e.g. <=9 blocks for a 2048 bp window).
        #
        # bounded=True applies a final sigmoid, squashing the output to [0, 1] for
        # the per-region score label. Set bounded=False for the summed-bin label
        # (aggregate_bins.py), whose value ranges ~0..#regions_in_bin and so needs
        # a linear (unbounded) output instead.
        self.bounded = bounded
        self.num_blocks = num_blocks
        # Each block with pool>1 divides the sequence length by `pool`, so the
        # input must be at least pool**num_blocks long or a block reduces the
        # length to zero. Stored so forward() can fail fast with a clear message
        # instead of producing silent garbage.
        self._pool_factor = pool if (pool and pool > 1) else 1

        self._blocks = []
        dim = in_channels
        for i in range(num_blocks):
            out = num_filters * (2 ** i)
            block = conv_block(dim, out, ker_size, dropout, pool)
            # Register as block1, block2, ... so a 3-block/32-filter model keeps
            # the same state_dict keys as before (old checkpoints still load).
            setattr(self, f"block{i + 1}", block)
            self._blocks.append(block)
            dim = out

        self.pool = AttentionPool(dim)
        # Center-anchored head. The label describes the central REGION_WIDTH bp,
        # but AttentionPool returns a softmax-weighted AVERAGE over every output
        # position (64 of them at 2048 bp with 5 pool-2 blocks), so the labeled
        # region contributes ~1/64 of the vector the head sees and the region-mask
        # channel can only nudge the weights. With center_pool on, the head also
        # gets the feature column sitting directly over the region:
        # fc([attention_pool(x) ; x[:, :, center]]). That is why the head input
        # doubles to 2*dim.
        self.center_pool = center_pool
        self.fc = nn.Linear(dim * (2 if center_pool else 1), 1)

        # Seed the output bias so a bounded (sigmoid) model starts at the label
        # mean instead of sigmoid(0)=0.5. With a 0 bias the output begins at 0.5,
        # far from the ~0.76 mean, so the optimizer cuts loss by cranking the
        # logit up toward the saturating upper rail where sigmoid'(z)->0; the
        # gradient then vanishes and the model freezes as a constant ~1.0 (zero
        # output variance -> undefined Pearson). bias_init = logit(mean) starts
        # the output on the steep part of the curve. Ignored for the linear
        # (bounded=False) head, whose bias has no logit interpretation.
        if self.bounded and bias_init is not None:
            nn.init.constant_(self.fc.bias, bias_init)

    def forward(self, x):
        if self._pool_factor > 1:
            min_len = self._pool_factor ** self.num_blocks
            if x.shape[-1] < min_len:
                raise ValueError(
                    f"input length {x.shape[-1]} too short for {self.num_blocks} "
                    f"blocks with pool={self._pool_factor}; need length >= {min_len}"
                )
        for block in self._blocks:
            x = block(x)
        pooled = self.pool(x)                      # (batch, dim) global context
        if self.center_pool:
            # The column of features over the window's midpoint, i.e. over the
            # 16 bp region the label actually describes.
            center = x[:, :, x.shape[-1] // 2]     # (batch, dim)
            pooled = torch.cat([pooled, center], dim=1)
        out = self.fc(pooled)
        return torch.sigmoid(out) if self.bounded else out


def load_model(weight_file, device, in_channels=4, num_filters=32, num_blocks=3,
               ker_size=5, dropout=0.3, pool=2, bounded=True, center_pool=False):
    """Build a HeterogeneityScoreModel, load a checkpoint into it, and put it on
    `device` in eval mode.

    Shared by predict.py and eval_report.py so inference always reconstructs the
    architecture the same way. The arch args (in_channels, num_filters,
    num_blocks, ker_size, pool, center_pool) and `bounded` must match how the
    weights were trained, or load_state_dict will fail on a shape/key mismatch.
    """
    model = HeterogeneityScoreModel(dropout=dropout, ker_size=ker_size,
                                    in_channels=in_channels, num_filters=num_filters,
                                    num_blocks=num_blocks, pool=pool, bounded=bounded,
                                    center_pool=center_pool)
    model.load_state_dict(torch.load(weight_file, map_location=device))
    return model.to(device).eval()
