"""SUMMITEnsemble: SUMMIT fitted as a consensus of (row x column) patches.

 1. Patches: n_patches pairs of
      columns: col_ratio of all columns, disjoint between patches
      rows:    row_ratio of the rows (within each row group, if given)
 2. One SUMMIT per patch (own seed: random_state + patch), weighted loss.
 3. Fill-in: patch rows keep their fitted W; the other rows get W from
    transform_W. Every patch now has a W for every row.
 4. Consensus W (see consensus.build_consensus), unit-length columns.
 5. Consensus A: per component and row, the median of the member
    components' A rows over the patches that contained the row; rows of A
    rescaled to sum to one.
 6. H on ALL columns by weighted least squares with the consensus W.

The result is a SUMMIT (W, A, H, transform_W, transform_H work as usual).
Diagnostics are in consensus_.
"""
import inspect
import time
import warnings

import numpy as np
import pandas as pd
import torch

from ._utils import align, free_gpu, row_weights
from .consensus import build_consensus, draw_col_sets, match_components
from .model import SUMMIT

_BASE_PARAMS = [p for p in inspect.signature(SUMMIT.__init__).parameters
                if p not in ("self", "n_components", "random_state")]


class SUMMITEnsemble(SUMMIT):
    """Parameters (everything else goes to SUMMIT and is used for every patch)

    n_patches : number of patches. Default 10.
    col_ratio : share of all columns in each patch. Default 0.01.
    row_ratio : share of the rows (of each row group) in each patch. Default 0.5.
    replace : False = disjoint column sets.
    min_per_group : fewest rows kept from any row group.
    density_threshold, local_neighborhood_size, min_support, redundant_corr :
        consensus settings (see consensus.build_consensus).
    ridge : relative ridge for the least-squares H.
    transform_max_iter : max_iter for the fill-in transform_W (None = max_iter).
    same_model_seed : True = every patch uses random_state.
    progress : print progress.
    """

    def __init__(self, n_components, *, n_patches=10, col_ratio=0.01, row_ratio=0.5,
                 replace=False, min_per_group=1, density_threshold=None,
                 local_neighborhood_size=0.3, min_support=None, redundant_corr=0.9,
                 ridge=1e-6, transform_max_iter=None, same_model_seed=False, progress=True,
                 **summit_kwargs):
        super().__init__(n_components, **summit_kwargs)
        if not 0 < row_ratio <= 1:
            raise ValueError("row_ratio must be in (0, 1]")
        self.n_patches, self.col_ratio, self.row_ratio = n_patches, col_ratio, row_ratio
        self.replace, self.min_per_group = replace, min_per_group
        self.density_threshold, self.local_neighborhood_size = density_threshold, local_neighborhood_size
        self.min_support, self.redundant_corr, self.ridge = min_support, redundant_corr, ridge
        self.transform_max_iter, self.same_model_seed = transform_max_iter, same_model_seed
        self.progress = progress
        self.consensus_ = None

    @torch.no_grad()
    def fit(self, X, row_weight=None, row_groups=None):
        """X: rows x columns (DataFrame or array).
        row_weight: loss weight per row (as in SUMMIT.fit).
        row_groups: optional group label per row; patch rows are then picked
            within each group (None = all rows as one group)."""
        is_df = isinstance(X, pd.DataFrame)
        row_names = list(X.index) if is_df else None
        Xv = np.asarray(X, dtype=np.float32)
        n, m = Xv.shape
        k = self.n_components
        say = print if self.progress else (lambda *a, **kw: None)
        t_start = time.time()
        seed = self.random_state_ = (int(np.random.randint(2 ** 31 - 1)) if self.random_state is None
                                     else int(self.random_state))
        rng = np.random.default_rng(seed)
        w = np.ones(n) if row_weight is None else row_weights(row_weight, row_names, n)

        # 1. patches
        col_sets = draw_col_sets(np.arange(m), int(round(self.col_ratio * m)), self.n_patches,
                                 self.replace, rng)
        groups = np.zeros(n) if row_groups is None else align(row_groups, row_names, "row_groups")
        if groups.shape != (n,) or pd.isna(groups).any():
            raise ValueError("row_groups must have one nonmissing label per row")
        by_group = [np.flatnonzero(groups == g) for g in pd.unique(groups)]

        def pick():
            if self.row_ratio >= 1:
                return np.arange(n)
            return np.sort(np.concatenate([
                rng.choice(pos, size=min(len(pos), max(self.min_per_group,
                                                       int(round(self.row_ratio * len(pos))))),
                           replace=False) for pos in by_group]))
        row_sets = [pick() for _ in range(self.n_patches)]
        if k > min(len(S) for S in row_sets):
            raise ValueError("k exceeds the rows of the smallest patch; raise row_ratio or lower k")
        say(f"{self.n_patches} patches: {len(col_sets[0])} columns x "
            f"~{np.mean([len(S) for S in row_sets]):.0f} of {n} rows")

        # 2-3. fit each patch, fill in the other rows
        params = {name: getattr(self, name) for name in _BASE_PARAMS}
        W_list, A_list, patch_rows = [], [], []
        for i, (P, S) in enumerate(zip(col_sets, row_sets)):
            out = np.setdiff1d(np.arange(n), S)
            patch_seed = seed if self.same_model_seed else seed + i
            t0 = time.time()
            model = SUMMIT(k, random_state=patch_seed, **params)
            model.fit(Xv[np.ix_(S, P)], row_weight=None if row_weight is None else w[S])
            fit_seconds = time.time() - t0
            if self.transform_max_iter is not None:
                model.max_iter = int(self.transform_max_iter)
            W_full = np.empty((n, k))
            W_full[S] = model.W
            if len(out):
                W_full[out] = model.transform_W(Xv[np.ix_(out, P)])
            A_full = np.full((k, n), np.nan)           # NaN = row not in this patch
            A_full[:, S] = model.A
            W_list.append(W_full)
            A_list.append(A_full)
            patch_rows.append({"patch": i, "model_seed": patch_seed, "n_rows": len(S),
                               "n_cols": len(P), "converged": bool(model.converged_),
                               "n_iter": int(model.n_iter_), "fit_seconds": fit_seconds,
                               "fill_seconds": time.time() - t0 - fit_seconds})
            say(f"patch {i}: {patch_rows[-1]['fit_seconds']:.0f}s fit + "
                f"{patch_rows[-1]['fill_seconds']:.0f}s fill-in, converged {model.converged_}",
                flush=True)
            del model
            free_gpu(self.device)

        # 4. consensus W
        cons = build_consensus(W_list, k, self.density_threshold, self.local_neighborhood_size,
                               self.min_support, self.redundant_corr, random_state=seed)
        cols = cons["columns"]

        # 5. consensus A (rows sum to one)
        A_cons = np.zeros((k, n))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)   # all-NaN columns
            for j in range(k):
                members = cols[cols["component"] == j]
                stacked = np.stack([A_list[r.patch][r.patch_component] for r in members.itertuples()])
                A_cons[j] = np.nan_to_num(np.nanmedian(stacked, axis=0), nan=0.0)
        A_cons = np.clip(A_cons, 0.0, None)
        empty = A_cons.sum(axis=1) <= 0
        if empty.any():
            warnings.warn(f"{int(empty.sum())} consensus A rows are all zero; set to equal weights")
            A_cons[empty] = 1.0
        A_cons /= A_cons.sum(axis=1, keepdims=True)

        # 6. least-squares H on all columns, stored as a SUMMIT
        self.row_names_ = row_names
        self.col_names_ = list(X.columns) if is_df else None
        self.n_features_in_, self._n_train_ = m, n
        self.row_weight_ = None if row_weight is None else self._dev(w)
        self.W_ = self._dev(cons["W"])
        self.A_ = self._dev(A_cons)
        Xh = self._host(Xv)
        self.H_ = self._solve_H(Xh, self.W_, self.ridge)
        self.reconstruction_mse_ = self._mse(Xh)
        patches = pd.DataFrame(patch_rows)
        self.converged_ = bool(patches["converged"].all())
        self.n_iter_ = int(patches["n_iter"].max())
        self.loss_ = self.loss_iterations_ = None

        # diagnostics
        comp = cons["clusters"].copy()
        comp["W_corr_median"] = np.median(
            np.column_stack([match_components(Wp, cons["W"])[1] for Wp in W_list]), axis=1)
        names = np.asarray(row_names) if is_df else np.arange(n)
        col_names = np.asarray(self.col_names_) if is_df else np.arange(m)
        summary = {"k": k, **cons["summary"], "reconstruction_mse": self.reconstruction_mse_,
                   "share_patches_converged": float(patches["converged"].mean()),
                   "total_seconds": time.time() - t_start}
        self.consensus_ = {"summary": pd.Series(summary), "components": comp,
                           "local_density": cols, "patches": patches,
                           "patch_rows": {i: list(names[S]) for i, S in enumerate(row_sets)},
                           "patch_cols": {i: list(col_names[P]) for i, P in enumerate(col_sets)}}
        say(f"done in {summary['total_seconds']:.0f}s: effective k {summary['effective_k']} of {k}, "
            f"silhouette {summary['silhouette']:.3f}, MSE {self.reconstruction_mse_:.4g}")
        return self