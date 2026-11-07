# SUMMIT

**S**igned **U**nsupervised **M**atrix factorization with **M**ultipatch **I**n**T**egration
 
---

SUMMIT is a convex NMF fitted with Adam. It splits a data matrix `X` (rows x columns) into

```
X ≈ W @ A @ H = W @ H,    H = A @ X
```

- `W` (rows x k): how much each row belongs to each component. Nonnegative.
- `A` (k x rows): each component as a weighted average of rows. Each row of `A` sums to one.
- `H` (k x columns): the profile of each component over the columns. Can be negative.

The method works on any rows x columns matrix. In our paper, rows are samples and columns are CpG probes.
 
## Quick navigation

- [SUMMIT](#summit)
  - [Quick navigation](#quick-navigation)
  - [Installation](#installation)
  - [Usage](#usage)
    - [Input data](#input-data)
    - [Fitting the model](#fitting-the-model)
    - [New rows and new columns](#new-rows-and-new-columns)
    - [Choosing k with BCV](#choosing-k-with-bcv)
    - [Ensemble for large data](#ensemble-for-large-data)
    - [Save and load](#save-and-load)
  - [Citation](#citation)

## Installation

```bash
git clone https://github.com/ylaboratory/SUMMIT.git
cd summit
uv sync            # or: pip install .
```

Requires Python, numpy, pandas, torch, scipy and scikit-learn. For a GPU, install the CUDA build of torch, then pass `device="cuda"`.

## Usage

### Input data

- `X`: a pandas DataFrame (or numpy array), **rows x columns**, for example samples x probes.
- SUMMIT uses `X` exactly as given. If you want to center it (for example, subtract a reference value per column), do that before fitting.
- `row_weight` (optional): one weight per row, used in the loss. A labeled Series is matched to `X` by row name. An array is taken in row order.

### Fitting the model

```python
from summit import SUMMIT

model = SUMMIT(n_components=50, init="soft_kmeans", device="cuda")
model.fit(X, row_weight=w)

W = model.W        # rows x k
A = model.A        # k x rows
H = model.H        # k x columns, H = A @ X
model.reconstruction_mse_   # weighted MSE of X - W @ H
```

Useful settings:

| Setting | Default | Meaning |
|---|---|---|
| `init` | `"kmeans"` | Start from `"random"`, `"kmeans"` or `"soft_kmeans"` |
| `lr` | `1e-2` | Adam learning rate |
| `max_iter`, `tol` | `5000`, `1e-4` | Stop after this many steps, or when the loss stops improving |
| `chunk_size` | `20000` | Columns moved to the device at a time (lower it if GPU memory is short) |
| `random_state` | `42` | Seed |

### New rows and new columns

**New rows, same columns:** get `W` for new samples.

```python
W_new = model.transform_W(X_new)      # X_new: new rows x training columns
```

**Same rows, new columns:** get `H` for new features by least squares with `W` fixed.

```python
H_new = model.transform_H(X_cols)     # X_cols: training rows x any columns
```

`transform_H` gives the best `H` for the fitted `W`. To use another `W` (for example a consensus `W`), pass `transform_H(X_cols, W=W_other)`.

Both functions take `expand=True`, which adds the new rows to `W` (or the new columns to `H`) inside the model.

### Choosing k with BCV

`bcv` runs bi-cross-validation for several values of k. For each k it does the following:

1. It hides some rows and some columns.
2. It fits SUMMIT on many small column patches.
3. It predicts the hidden block two ways: with each patch's own `W`, and with one consensus `W` built from all the patches.

```python
from summit import bcv

out = bcv(
    X, ks=[10, 20, 30, 50, 75, 100],
    col_ratio=0.02,          # 1% of columns kept per patch, 1% hidden
    n_patches=10,
    row_weight=w,
    init="soft_kmeans", device="cuda",
)

out["summary"][["mse0_mean", "mse_own_mean", "mse_cons_mean", "effective_k"]]
```

By default, half of the rows are hidden at random (`test_size=0.5`). To choose the hidden rows yourself, pass `row_split`:

```python
out = bcv(X, ks=[...], row_split=hidden_mask)            # boolean, True = hidden row
out = bcv(X, ks=[...], row_split=["sample_3", "sample_8"])  # or a list of row names
```

The main outputs:

| Key | One row per | What's in it |
|---|---|---|
| `summary` | k | Mean and standard error of each error, plus consensus statistics |
| `results` | k and patch | `mse0` (baseline, predicting zero), `mse_own`, `mse_cons`, `train_mse_own`, `train_mse_cons` |
| `components` | k and component | Support across patches, redundancy, and how well W and H agree |
| `consensus_W` | k | The consensus `W` as a DataFrame |

If you pass `row_weight`, each error also has a weighted version, ending in `_w`.

**How to read it.** Pick the k where `mse_cons_mean` stops dropping (the elbow). `effective_k` tells you how many components were found again and again across patches.

### Ensemble for large data

`SUMMITEnsemble` does three things:

1. It fits many small SUMMIT models, each on a random share of the rows and the columns.
2. It combines them into one consensus `W` and `A`.
3. It solves `H` for **all** columns with that `W`.

The result works like a normal SUMMIT model.

```python
from summit import SUMMITEnsemble

model = SUMMITEnsemble(
    n_components=50,
    n_patches=10, col_ratio=0.01, row_ratio=0.5,
    init="soft_kmeans", device="cuda",
)
model.fit(X, row_weight=w)              # optional: row_groups=... to pick rows within each group

model.W, model.A, model.H               # H is the least-squares H on all columns
model.consensus_["summary"]             # effective k, silhouette, MSE, run time
model.consensus_["components"]          # one row per component
W_new = model.transform_W(X_new)
```

### Save and load

Models are plain Python objects, so `pickle` works:

```python
import pickle

pickle.dump(model, open("summit_model.pkl", "wb"))
model = pickle.load(open("summit_model.pkl", "rb"))
```

## Citation

Coming soon.
