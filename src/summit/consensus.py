"""Shared by SUMMITEnsemble and bcv: column patches, cNMF-style consensus W
(Kotliar et al., 2019, eLife), and matching components to the consensus."""
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment


def unit_columns(W):
    W = np.asarray(W, dtype=np.float64)
    return W / np.maximum(np.linalg.norm(W, axis=0), 1e-12)


def col_corr(A, B):
    """Pearson correlation between every column of A and every column of B."""
    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)
    A = A - A.mean(axis=0)
    B = B - B.mean(axis=0)
    return (A / (np.linalg.norm(A, axis=0) + 1e-12)).T @ (B / (np.linalg.norm(B, axis=0) + 1e-12))


def row_corr(A, B):
    """Pearson correlation of row i of A with row i of B."""
    return np.diag(col_corr(np.asarray(A).T, np.asarray(B).T)).copy()


def draw_col_sets(pool, size, n_sets, replace, rng):
    """n_sets sorted sets of `size` columns from `pool` (column positions).
    replace=False: disjoint sets (error if they do not fit); True: may overlap."""
    if size < 2:
        raise ValueError(f"a patch would have only {size} columns; increase col_ratio")
    pool = rng.permutation(pool)
    if replace:
        return [np.sort(rng.choice(pool, size=size, replace=False)) for _ in range(n_sets)]
    if size * n_sets > len(pool):
        raise ValueError(f"{n_sets} disjoint patches of {size} columns need {size * n_sets} "
                         f"columns, only {len(pool)} available; use fewer patches, a smaller "
                         "col_ratio, or replace=True")
    return [np.sort(pool[i * size:(i + 1) * size]) for i in range(n_sets)]


def match_components(W_own, W_cons):
    """One-to-one matching (Hungarian on W column correlation).
    Returns (order, corr): own component order[j] matches consensus j."""
    C = col_corr(W_own, W_cons)
    rows, cols = linear_sum_assignment(-C)
    order = np.empty(W_cons.shape[1], dtype=int)
    order[cols] = rows
    return order, C[order, np.arange(W_cons.shape[1])]


def build_consensus(W_list, k, density_threshold=None, local_neighborhood_size=0.3,
                    min_support=None, redundant_corr=0.9, random_state=0, n_init=10):
    """Consensus of the W matrices of many patches (same rows, same order).

    1. Scale each W column to unit length; stack all n_patches * k columns.
    2. Local density of each column: mean distance to its n nearest other
       columns, n = local_neighborhood_size * n_patches (at least 1).
    3. Drop columns with density > density_threshold (None = keep all).
    4. k-means into k clusters on the kept columns.
    5. Consensus column = row-wise median of its cluster, unit length.
    Components are numbered by support (most-supported first).

    A component is `supported` if at least min_support of the patches have a
    member in its cluster (None = local_neighborhood_size), and `redundant` if
    its W correlates above redundant_corr with another consensus component.

    Returns a dict: W (rows x k), columns (one row per patch component:
    local_density, kept, component; -1 = filtered out), clusters (one row per
    consensus component), summary (dict for one k).
    """
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    P = len(W_list)
    V = np.concatenate([unit_columns(W) for W in W_list], axis=1).T     # (P*k) x rows
    patch_id = np.repeat(np.arange(P), k)

    dist = np.sqrt(np.clip(2.0 - 2.0 * (V @ V.T), 0.0, None))         # unit vectors
    np.fill_diagonal(dist, np.inf)
    n_nb = min(max(1, int(round(local_neighborhood_size * P))), len(V) - 1)
    density = np.partition(dist, n_nb - 1, axis=1)[:, :n_nb].mean(axis=1)
    del dist

    kept = (np.ones(len(V), dtype=bool) if density_threshold is None
            else density <= float(density_threshold))
    if kept.sum() < k:
        raise ValueError(f"only {int(kept.sum())} components pass density_threshold="
                         f"{density_threshold}, fewer than k={k}; raise the threshold")
    Vk = V[kept]
    lab = KMeans(n_clusters=k, n_init=n_init, random_state=random_state).fit(Vk).labels_
    silhouette = float(silhouette_score(Vk, lab)) if 1 < k < len(Vk) else np.nan
    W_cons = unit_columns(np.stack([np.median(Vk[lab == c], axis=0) for c in range(k)], axis=1))

    labels = np.full(len(V), -1)
    labels[kept] = lab
    rows = []
    for c in range(k):
        members = labels == c
        per_patch = np.bincount(patch_id[members], minlength=P)
        rows.append({"n_members": int(members.sum()),
                     "support": float((per_patch > 0).mean()),
                     "single_support": float((per_patch == 1).mean()),
                     "mean_local_density": float(density[members].mean())})
    clusters = pd.DataFrame(rows)

    order = clusters.sort_values(["single_support", "n_members"], ascending=False).index.to_numpy()
    remap = np.empty(k, dtype=int)
    remap[order] = np.arange(k)
    W_cons = W_cons[:, order]
    labels = np.where(labels >= 0, remap[np.maximum(labels, 0)], -1)
    clusters = clusters.loc[order].reset_index(drop=True).rename_axis("component")

    if k > 1:
        cc = col_corr(W_cons, W_cons)
        np.fill_diagonal(cc, -np.inf)
        clusters["max_corr_other"] = cc.max(axis=1)
    else:
        clusters["max_corr_other"] = np.nan
    min_support = local_neighborhood_size if min_support is None else min_support
    clusters["supported"] = clusters["support"] >= min_support
    clusters["supported_single"] = clusters["single_support"] >= min_support
    clusters["redundant"] = clusters["max_corr_other"] > redundant_corr

    columns = pd.DataFrame({"patch": patch_id, "patch_component": np.tile(np.arange(k), P),
                            "local_density": density, "kept": kept, "component": labels})
    summary = {"silhouette": silhouette,
               "n_columns_kept": int(kept.sum()), "n_columns_total": len(kept),
               "mean_support": float(clusters["support"].mean()),
               "mean_single_support": float(clusters["single_support"].mean()),
               "effective_k": int(clusters["supported"].sum()),
               "effective_k_single": int(clusters["supported_single"].sum()),
               "n_redundant": int(clusters["redundant"].sum())}
    return {"W": W_cons, "columns": columns, "clusters": clusters, "summary": summary}