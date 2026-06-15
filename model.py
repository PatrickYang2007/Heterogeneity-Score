import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np

from prepare_data import encode_indices


class GenomicDataset(Dataset):
    def __init__(self, parquet_path):
        df = pd.read_parquet(parquet_path, columns=["sequence", "score"])
        sequences = df["sequence"].str.upper().tolist()
        # Store sequences as compact int8 indices (A/C/G/T=0-3, N/other=4) and
        # build the one-hot tensor on the fly in __getitem__. This keeps memory
        # flat as the window grows: a preloaded float32 one-hot would be ~18 GB
        # at 256 bp / ~36 GB at 512 bp for 4.4M rows, blowing the job's RAM,
        # whereas int8 indices are ~1-2 GB.
        self.x = torch.from_numpy(encode_indices(sequences))
        self.y = torch.tensor(df["score"].values, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        idx_seq = self.x[idx].long()                       # (L,) values 0-4
        onehot = F.one_hot(idx_seq.clamp(max=3), num_classes=4).float()
        onehot[idx_seq == 4] = 0.0                          # N -> all-zero vector
        return onehot.transpose(0, 1), self.y[idx]          # (4, L)


def make_dataloader(parquet_path, batch_size = 64, shuffle = True, num_workers = 4):
    ds = GenomicDataset(parquet_path)
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


class HomogeneityScoreModel(nn.Module):
    def __init__(self, dropout, ker_size=5, in_channels=4, num_filters=32, pool=2):
        super().__init__()
        # pool=1 reproduces the old no-pooling behavior (sensible for 16 bp
        # inputs); pool=2 (default) halves length each block so wider windows
        # are summarized over progressively longer range.
        self.block1 = conv_block(in_channels, num_filters, ker_size, dropout, pool)
        self.block2 = conv_block(num_filters, num_filters * 2, ker_size, dropout, pool)
        self.block3 = conv_block(num_filters * 2, num_filters * 4, ker_size, dropout, pool)
        self.pool = AttentionPool(num_filters * 4)
        self.fc = nn.Linear(num_filters * 4, 1)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.pool(x)
        x = self.fc(x)
        return torch.sigmoid(x)
