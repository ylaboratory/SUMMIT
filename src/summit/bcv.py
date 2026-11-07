"""Consensus bi-cross-validation (BCV) of SUMMIT over column patches.

For every k in ks: how well does each patch's own W predict held-out data,
and how well does one shared (consensus) W do?

Done once:
  1. Row split: hidden rows (random test_size share, or your row_split);
     the rest are training rows.
  2. Columns: one hidden column set (half of col_ratio * n_columns), shared
     by every patch and every k; each patch gets its own kept columns (the
     other half), drawn from the remaining columns.

Blocks of a patch:
                        kept columns    hidden columns
    hidden rows            B               T (target, never used to fit)
    training rows          D               C

For each k:
  3. One SUMMIT per patch on D.
  4. Consensus W from all patch W's (consensus.build_consensus).
  5. Predict T two ways:
       own:  W = transform_W(B),                 H = transform_H(C)
       cons: W = transform_W(B, H=H_cons_kept),  H = transform_H(C, W=W_cons)
     with H_cons_kept = transform_H(D, W=W_cons).

Errors (columns of out["results"]):
    mse0              MSE of predicting zero for T (baseline)
    mse_own, mse_cons hidden-block MSE
    train_mse_own, train_mse_cons   MSE on D
  With row_weight, each also gets a weighted version (suffix _w): the
  weighted mean over rows of each row's MSE. Row weights are also used in
  fitting and in transform_H.

X is used as given. If you centered X with all rows, the hidden rows
contributed a little to the center.
"""
import time

import numpy as np
import pandas as pd
import torch

from ._utils import align, free_gpu, row_weights
from .consensus import build_consensus, draw_col_sets, match_components, row_corr
from .model import SUMMIT


def _mse(name, resid, w):
    sq = np.einsum("ij,ij->i", resid, resid, dtype=np.float64) / resid.shape[1]   # per-row MSE
    out = {name: float(sq.mean())}
    if w is not None:
        out[name + "_w"] = float((w * sq).sum() / w.sum())
    return out


def bcv(X, ks, *, col_ratio=0.1, n_patches=10, replace=False, row_split=None, test_size=0.5,
        row_weight=None, density_threshold=None, local_neighborhood_size=0.3, min_support=None,
        redundant_corr=0.9, ridge=1e-6, transform_max_iter=None, random_state=0,
        model_seed=None, verbose=True, **summit_kwargs):
    """Consensus BCV of SUMMIT for each k in ks.

    X : rows x columns (DataFrame or array).
    col_ratio : share of all columns per patch, kept + hidden together.
    row_split : None = random split with test_size hidden. Or a boolean mask
        (True = hidden row; a labeled Series is lined up with X's index), or a
        list of hidden row names (positions for an array).
    row_weight : one weight per row (fit, transform_H and the _w errors).
    density_threshold : None, one number, or {k: number}.
    model_seed : random_state of every SUMMIT fit (None = random_state).
    **summit_kwargs : other SUMMIT settings (init, max_iter, device, ...).

    Returns a dict: results (one row per k and patch), summary (one row per
    k: mean and SE across patches, plus consensus stats), components (one
    row per k and consensus component), local_density, consensus_W
    ({k: DataFrame}), train_rows, hidden_rows, hidden_cols, patch_cols.
    """
    is_df = isinstance(X, pd.DataFrame)
    rows = X.index if is_df else pd.RangeIndex(np.shape(X)[0])
    cols = X.columns if is_df else pd.RangeIndex(np.shape(X)[1])
    names = list(rows) if is_df else None
    Xv = np.asarray(X, dtype=np.float32)
    n, m = Xv.shape
    ks = sorted({int(k) for k in ks})
    rng = np.random.default_rng(random_state)
    seed = int(random_state if model_seed is None else model_seed)
    say = print if verbose else (lambda *a, **kw: None)

    # 1. row split
    if row_split is None:
        hidden = np.zeros(n, dtype=bool)
        hidden[rng.permutation(n)[:int(round(test_size * n))]] = True
    else:
        split = align(row_split, names, "row_split")
        hidden = split.astype(bool) if split.dtype == bool else np.asarray(rows.isin(split))
    if hidden.shape != (n,):
        raise ValueError("row_split must have one value per row")
    train, hid = np.flatnonzero(~hidden), np.flatnonzero(hidden)
    if not len(train) or not len(hid):
        raise ValueError("the split needs both training and hidden rows")
    if ks[0] < 1 or ks[-1] > len(train):
        raise ValueError(f"every k must be between 1 and the {len(train)} training rows")
    w = None if row_weight is None else row_weights(row_weight, names, n)
    w_tr = None if w is None else row_weights(w[train], None, len(train))
    w_hid = None if w is None else w[hid]

    # 2. columns
    size = int(round(col_ratio * m))
    n_hid = size // 2
    perm = rng.permutation(m)
    hid_cols = np.sort(perm[:n_hid])
    kept_sets = draw_col_sets(perm[n_hid:], size - n_hid, n_patches, replace, rng)
    C = Xv[np.ix_(train, hid_cols)]
    T = Xv[np.ix_(hid, hid_cols)].astype(np.float64)
    base = _mse("mse0", T, w_hid)
    say(f"{len(train)} training / {len(hid)} hidden rows; {n_patches} patches of "
        f"{len(kept_sets[0])} kept columns, {n_hid} hidden columns shared")

    results, components, densities, summaries, consensus_W = [], [], [], [], {}
    for k in ks:
        # 3. one model per patch
        rows_k, W_own, H_own = [], [], []
        for p, P in enumerate(kept_sets):
            D, B = Xv[np.ix_(train, P)], Xv[np.ix_(hid, P)]
            t0 = time.time()
            model = SUMMIT(k, random_state=seed, **summit_kwargs).fit(D, row_weight=w_tr)
            fit_seconds = time.time() - t0
            if transform_max_iter is not None:
                model.max_iter = int(transform_max_iter)
            H_kept, H_hid = model.transform_H(D, ridge=ridge), model.transform_H(C, ridge=ridge)
            rows_k.append({"k": k, "patch": p, **base,
                           **_mse("mse_own", T - model.transform_W(B) @ H_hid, w_hid),
                           **_mse("train_mse_own", D - model.W @ H_kept, w_tr),
                           "converged": model.converged_, "n_iter": model.n_iter_,
                           "fit_seconds": fit_seconds})
            W_own.append(model.W)
            H_own.append((H_kept, H_hid))
            del model
            free_gpu(summit_kwargs.get("device", "cpu"))
            say(f"k={k} patch {p}: own MSE {rows_k[-1]['mse_own']:.4g} "
                f"(baseline {base['mse0']:.4g}), {fit_seconds:.0f}s", flush=True)

        # 4. consensus W
        thr = density_threshold.get(k) if isinstance(density_threshold, dict) else density_threshold
        cons = build_consensus(W_own, k, thr, local_neighborhood_size, min_support,
                               redundant_corr, random_state=int(random_state))
        W_cons = cons["W"]
        consensus_W[k] = pd.DataFrame(W_cons, index=rows[train],
                                      columns=[f"comp_{j}" for j in range(k)])

        # 5. predict with the consensus W (a SUMMIT holding W_cons as its W)
        solver = SUMMIT(k, random_state=seed, **summit_kwargs)
        if transform_max_iter is not None:
            solver.max_iter = int(transform_max_iter)
        solver.W_, solver._n_train_, solver.random_state_ = solver._dev(W_cons), len(train), seed
        solver.row_weight_ = None if w_tr is None else solver._dev(w_tr)
        per_comp = []
        for p, P in enumerate(kept_sets):
            D, B = Xv[np.ix_(train, P)], Xv[np.ix_(hid, P)]
            order, w_corr = match_components(W_own[p], W_cons)
            H_kept_c, H_hid_c = solver.transform_H(D, ridge=ridge), solver.transform_H(C, ridge=ridge)
            h_kept = row_corr(H_own[p][0][order], H_kept_c)
            h_hid = row_corr(H_own[p][1][order], H_hid_c)
            rows_k[p].update({**_mse("mse_cons", T - solver.transform_W(B, H=H_kept_c) @ H_hid_c, w_hid),
                              **_mse("train_mse_cons", D - W_cons @ H_kept_c, w_tr),
                              "W_corr_median": float(np.median(w_corr)),
                              "W_corr_min": float(w_corr.min()),
                              "H_kept_corr_median": float(np.median(h_kept)),
                              "H_hidden_corr_median": float(np.median(h_hid))})
            per_comp.append(pd.DataFrame({"component": np.arange(k), "patch": p,
                                          "W_corr": w_corr, "H_kept_corr": h_kept,
                                          "H_hidden_corr": h_hid}))
        del solver
        free_gpu(summit_kwargs.get("device", "cpu"))

        results.extend(rows_k)
        comp = cons["clusters"].join(pd.concat(per_comp).groupby("component")
                                     [["W_corr", "H_kept_corr", "H_hidden_corr"]]
                                     .median().add_suffix("_median"))
        components.append(comp.reset_index().assign(k=k))
        densities.append(cons["columns"].assign(k=k))
        r = pd.DataFrame(rows_k)
        metrics = [c for c in r if c.startswith(("mse", "train_mse", "W_corr", "H_"))]
        summaries.append({"k": k, **cons["summary"], "share_converged": float(r["converged"].mean()),
                          **{f"{c}_{s}": getattr(r[c], s)() for c in metrics for s in ("mean", "sem")}})
        say(f"k={k}: MSE own {summaries[-1]['mse_own_mean']:.4g}, consensus "
            f"{summaries[-1]['mse_cons_mean']:.4g}, effective k {cons['summary']['effective_k']}",
            flush=True)

    return {"results": pd.DataFrame(results),
            "summary": pd.DataFrame(summaries).set_index("k"),
            "components": pd.concat(components, ignore_index=True),
            "local_density": pd.concat(densities, ignore_index=True),
            "consensus_W": consensus_W,
            "train_rows": list(rows[train]), "hidden_rows": list(rows[hid]),
            "hidden_cols": list(cols[hid_cols]),
            "patch_cols": {p: list(cols[P]) for p, P in enumerate(kept_sets)}}