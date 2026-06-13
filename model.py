import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np

from prepare_data import one_hot_encode


class GenomicDataset(Dataset):
    def __init__(self, parquet_path):
        df = pd.read_parquet(parquet_path, columns=["sequence", "score"])
        sequences = df["sequence"].str.upper().tolist()
        # one_hot_encode returns (N, L, 4); transpose to (N, 4, L) for Conv1d
        encoded = one_hot_encode(sequences)
        self.x = torch.from_numpy(encoded.transpose(0, 2, 1).astype("float32"))
        self.y = torch.tensor(df["score"].values, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


def make_dataloader(parquet_path, batch_size = 64, shuffle = True, num_workers = 4):
    ds = GenomicDataset(parquet_path)
    return DataLoader(ds, batch_size = batch_size, shuffle = shuffle,
                      num_workers = num_workers, pin_memory = True)


def conv_block(dim, dim_out, ker_size, dropout):
    return nn.Sequential(
        nn.BatchNorm1d(dim),
        nn.GELU(),
        nn.Conv1d(dim, dim_out, ker_size),
        nn.Dropout(dropout),
    )


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
    def __init__(self, dropout, ker_size=5, in_channels=4, num_filters=32):
        super().__init__()
        self.block1 = conv_block(in_channels, num_filters, ker_size, dropout)
        self.block2 = conv_block(num_filters, num_filters * 2, ker_size, dropout)
        self.block3 = conv_block(num_filters * 2, num_filters * 4, ker_size, dropout)
        self.pool = AttentionPool(num_filters * 4)
        self.fc = nn.Linear(num_filters * 4, 1)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.pool(x)
        x = self.fc(x)
        return torch.sigmoid(x)
