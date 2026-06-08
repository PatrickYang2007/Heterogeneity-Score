import torch
import torch.nn as nn
import torch.nn.functional as F
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


def make_dataloader(parquet_path, batch_size=64, shuffle=True, num_workers=4):
    ds = GenomicDataset(parquet_path)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, pin_memory=True)


class homogeneity_score_model(nn.Module):
    def __init__(self):
        super().__init__()


class attention_pool(nn.Module):
    def __init__(self):
        super().__init__()
