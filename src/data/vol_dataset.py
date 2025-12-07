from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .load_prices import train_val_test_split_indices


class TimeSeriesVolDataset(Dataset):
    """
    Sequence-to-one dataset for realized volatility forecasting.

    X_t: (lookback, n_features)
    y_t: scalar target (e.g. rvol_{t + horizon - 1})

    We use a contiguous block of indices [start, end) for each split.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        target_col: str,
        lookback: int = 60,
        horizon: int = 1,
        split: str = "train",
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        indices: Optional[Tuple[int, int]] = None,
    ):
        super().__init__()
        assert split in {"train", "val", "test"}

        self.df = df.reset_index(drop=True)
        self.feature_cols = feature_cols
        self.target_col = target_col
        self.lookback = lookback
        self.horizon = horizon

        if indices is not None:
            self.start_idx, self.end_idx = indices
        else:
            # Fallback to the old ratio logic if no explicit indices provided
            n = len(self.df)
            train_end, val_end = train_val_test_split_indices(
                n, train_ratio=train_ratio, val_ratio=val_ratio
            )
            if split == "train":
                self.start_idx = 0
                self.end_idx = train_end
            elif split == "val":
                self.start_idx = train_end
                self.end_idx = val_end
            else:
                self.start_idx = val_end
                self.end_idx = n

        # Last usable index for constructing (X, y)
        self.max_shift = self.lookback + self.horizon - 1
        if self.end_idx - self.start_idx <= self.max_shift:
            raise ValueError("Not enough data in the selected split.")

    def __len__(self) -> int:
        # e.g. if we have indices [start, end), max index we can use as 't' is end - 1
        # we need lookback + horizon - 1 points ending at t + horizon - 1
        return (self.end_idx - self.start_idx) - self.max_shift

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # global index for the end of the window (inclusive)
        s = self.start_idx + idx
        # X window: [s, s + lookback)
        x_start = s
        x_end = s + self.lookback
        # y index: s + lookback + horizon - 1
        y_idx = s + self.lookback + self.horizon - 1

        x = self.df.loc[x_start:x_end - 1, self.feature_cols].values
        y = self.df.loc[y_idx, self.target_col]

        x = torch.tensor(x, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)
        return x, y
