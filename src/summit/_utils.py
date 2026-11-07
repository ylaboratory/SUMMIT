import numpy as np
import pandas as pd
import torch


def align(values, names, what):
    """A labeled Series is lined up with the row names of X; anything else
    (array, list, Series with a 0..n-1 index) is taken in row order."""
    if isinstance(values, pd.Series) and not values.index.equals(pd.RangeIndex(len(values))):
        if names is None:
            raise ValueError(f"{what} is a labeled Series but X has no row names; "
                             f"pass X as a DataFrame or {what} as an array")
        if not values.index.is_unique or set(values.index) != set(names):
            raise ValueError(f"{what} index must match the row names of X")
        values = values.loc[list(names)]
    if torch.is_tensor(values):
        values = values.detach().cpu().numpy()
    return np.asarray(values)


def row_weights(values, names, n):
    """One nonnegative weight per row, rescaled to mean one."""
    w = align(values, names, "row_weight").astype(float)
    if w.shape != (n,) or not np.isfinite(w).all() or (w < 0).any() or w.sum() <= 0:
        raise ValueError("row_weight must have one finite, nonnegative value per row "
                         "and a positive sum")
    return w * n / w.sum()


def free_gpu(device):
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache()