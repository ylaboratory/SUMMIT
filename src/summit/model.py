"""SUMMIT: (Signed Unsupervised Matrix factorization with Multipatch InTegration.

X is rows x columns, used as given (center it yourself if you want to).

    X ~= W @ H,   H = A @ X
    W : rows x k        nonnegative
    A : k x rows        nonnegative, each row sums to one
    H : k x columns     signed; each row is a weighted average of rows of X

Fitting only needs the rows x rows Gram matrix K = X X^T / m, built a block
of columns at a time, so X never goes to the device as a whole.

    fit(X, row_weight)    learn W, A and H = A @ X
    transform_W(X_new)    new rows x training columns -> W for the new rows
    transform_H(X_new)    training rows x any columns -> H by weighted
                          least squares with W fixed (own W or a given W)
"""
import warnings

import numpy as np
import pandas as pd
import torch

from ._utils import row_weights


def _check(ok, message):
    if not ok:
        raise ValueError(message)


class SUMMIT:
    """Convex NMF fitted with Adam, with K-means / soft K-means start and row weights.

    init:
        'random'      random factors (+0.2).
        'kmeans'      Lloyd K-means on the rows, computed from K.
        'soft_kmeans' start from that K-means, then soft K-means (MacKay):
                          r[i, j] = softmax_j(-beta * d2(x_i, m_j))
                      beta = init_beta / s, s = median squared distance of rows
                      to their own K-means center. init_beta='auto' picks beta by
                      bisection so the mean responsibility of rows to their own
                      cluster equals init_responsibility (no soft rounds). Then
                          A[:, j] = (1 - a) r[:, j] / sum(r[:, j]) + a / n
                          W       = (1 - a) r + a / k,      a = init_smooth

    Fitting: A = softmax(Z) over rows. Adam updates W and Z; W is clamped at
    zero after each step. The loss is the weighted MSE of X - W A X,
        sum_i d_i ||x_i - (W H)_i||^2 / (n m),
    computed from K only. d = row_weight rescaled to mean one (all ones if
    None). Row weights change the loss only, not the K-means start.
    """

    def __init__(self, n_components, lr=1e-2, max_iter=5000, tol=1e-4, device="cpu",
                 dtype=torch.float32, random_state=42, init="kmeans", verbose=False,
                 check_every=10, kmeans_iter=50, eps=1e-12, init_smooth=0.1,
                 init_beta="auto", init_responsibility=0.8, chunk_size=20000):

        for name, value in (("n_components", n_components), ("max_iter", max_iter),
                            ("check_every", check_every), ("kmeans_iter", kmeans_iter),
                            ("chunk_size", chunk_size)):
            _check(isinstance(value, (int, np.integer)) and value >= 1,
                   f"{name} must be a positive integer")
        _check(lr > 0 and tol >= 0 and eps > 0, "lr and eps must be positive, tol nonnegative")
        _check(init in ("random", "kmeans", "soft_kmeans"),
               "init must be 'random', 'kmeans' or 'soft_kmeans'")
        _check(dtype in (torch.float32, torch.float64), "dtype must be torch.float32 or float64")
        _check(0 < init_smooth < 1 and 0 < init_responsibility < 1,
               "init_smooth and init_responsibility must be strictly between 0 and 1")
        _check(init_beta == "auto" or (not isinstance(init_beta, str) and init_beta > 0),
               "init_beta must be 'auto' or a positive number")

        self.n_components = int(n_components)
        self.lr, self.max_iter, self.tol = float(lr), int(max_iter), float(tol)
        self.device, self.dtype, self.random_state = device, dtype, random_state
        self.init, self.verbose, self.check_every = init, verbose, int(check_every)
        self.kmeans_iter, self.eps, self.init_smooth = int(kmeans_iter), float(eps), float(init_smooth)
        self.init_beta, self.init_responsibility = init_beta, float(init_responsibility)
        self.chunk_size = int(chunk_size)
        self.W_ = self.A_ = self.H_ = self.row_weight_ = None


    def _require_fitted(self):
        if self.W_ is None:
            raise ValueError("Model has not been fitted yet")

    @staticmethod
    def _host(X):
        if torch.is_tensor(X):
            Xh = X.detach()
        else:
            values = np.asarray(X)
            if values.dtype not in (np.float32, np.float64):
                values = values.astype(np.float64)
            Xh = torch.from_numpy(np.ascontiguousarray(values))
        _check(Xh.ndim == 2 and min(Xh.shape) > 0, "X must be a nonempty 2D matrix")
        return Xh

    def _dev(self, a):
        a = a.detach() if torch.is_tensor(a) else torch.as_tensor(np.asarray(a, dtype=float))
        return a.to(device=self.device, dtype=self.dtype)

    def _chunks(self, Xh):
        for start in range(0, Xh.shape[1], self.chunk_size):
            block = Xh[:, start:start + self.chunk_size].to(device=self.device, dtype=self.dtype)
            _check(bool(torch.isfinite(block).all()), "X must be finite")
            yield start, block

    def _gram(self, Xh):
        K = torch.zeros(Xh.shape[0], Xh.shape[0], device=self.device, dtype=torch.float64)
        for _, block in self._chunks(Xh):
            K += (block @ block.T).double()
        K = (K / Xh.shape[1]).to(self.dtype)
        _check(bool(torch.isfinite(K).all()), "Gram matrix overflow; rescale X or use float64")
        return K

    def _initialize(self, K):
        n, k = K.shape[0], self.n_components
        self.init_labels_ = self.init_cluster_sizes_ = None
        self.init_beta_ = self.init_mean_responsibility_ = self.init_min_center_gap_ = None
        generator = torch.Generator(device=self.device).manual_seed(self.random_state_)
        if self.init == "random":
            rand = lambda: torch.rand(n, k, device=self.device, dtype=self.dtype,
                                      generator=generator) + 0.2
            A = rand()
            return rand(), A / A.sum(dim=0, keepdim=True)
        labels, membership, counts = self._kmeans(K, generator)
        if self.init == "soft_kmeans":
            return self._soft_kmeans(K, labels, membership, counts)
        self.init_labels_, self.init_cluster_sizes_ = labels.cpu().numpy(), counts.cpu().numpy()
        return membership + 0.2, (membership + 0.2) / counts[None, :]

    @staticmethod
    def _center_d2(K_diag, C, KC):
        # squared distance of every row to every center m_j = sum_i C[i, j] x_i
        return (K_diag[:, None] - 2 * KC + (C * KC).sum(dim=0)[None, :]).clamp_min(0)

    def _kmeans(self, K, generator):
        n, k = K.shape[0], self.n_components
        A = torch.zeros(n, k, device=self.device, dtype=self.dtype)
        A[torch.randperm(n, device=self.device, generator=generator)[:k],
          torch.arange(k, device=self.device)] = 1
        previous = None
        for _ in range(self.kmeans_iter):
            d2 = self._center_d2(K.diagonal(), A, K @ A)
            labels = d2.argmin(dim=1)
            counts = torch.bincount(labels, minlength=k)
            for empty in torch.where(counts == 0)[0].tolist():   # refill empty clusters
                own = d2.gather(1, labels[:, None]).squeeze(1)
                point = own.masked_fill(counts[labels] <= 1, -1).argmax()
                counts[labels[point]] -= 1
                labels[point] = empty
                counts[empty] += 1
            membership = torch.nn.functional.one_hot(labels, k).to(self.dtype)
            A = membership / counts[None, :]
            if previous is not None and torch.equal(labels, previous):
                break
            previous = labels.clone()
        return labels, membership, counts

    def _soft_kmeans(self, K, own, membership, counts):
        k, K_diag = self.n_components, K.diagonal()
        C = membership / counts.to(self.dtype)[None, :]
        KC = K @ C
        d2 = self._center_d2(K_diag, C, KC)
        scale = d2.gather(1, own[:, None]).median().clamp_min(self.eps).item()
        if self.init_beta == "auto":
            beta = self._auto_beta(d2 / scale, own) / scale
        else:
            beta = float(self.init_beta) / scale
            for _ in range(self.kmeans_iter):
                r = torch.softmax(-beta * d2, dim=1)
                C_new = r / r.sum(dim=0, keepdim=True).clamp_min(self.eps)
                KC_new = K @ C_new
                shift = ((C_new - C) * (KC_new - KC)).sum().clamp_min(0) / (C * KC).sum().clamp_min(self.eps)
                C, KC = C_new, KC_new
                d2 = self._center_d2(K_diag, C, KC)
                if shift.sqrt().item() <= self.tol:
                    break
        r = torch.softmax(-beta * d2, dim=1)
        self.init_beta_ = beta
        self.init_mean_responsibility_ = self._own_responsibility(r, own)

        G = C.T @ KC                                   # did clusters merge?
        gaps = (G.diagonal()[:, None] + G.diagonal()[None, :] - 2 * G).clamp_min(0) / scale
        gaps.fill_diagonal_(float("inf"))
        self.init_min_center_gap_ = gaps.min().item()
        if self.init_min_center_gap_ < 1e-2:
            warnings.warn("Soft K-means merged some clusters; increase init_beta", RuntimeWarning)

        self.init_labels_ = r.argmax(dim=1).cpu().numpy()
        self.init_cluster_sizes_ = r.sum(dim=0).cpu().numpy()
        a = self.init_smooth
        A = (1 - a) * r / r.sum(dim=0, keepdim=True) + a / K.shape[0]
        return (1 - a) * r + a / k, A

    def _own_responsibility(self, r, own):
        # per cluster, mean responsibility of its rows to their own cluster; then mean
        k = self.n_components
        sums = torch.zeros(k, device=r.device, dtype=r.dtype).index_add_(
            0, own, r.gather(1, own[:, None]).squeeze(1))
        return (sums / torch.bincount(own, minlength=k).to(r.dtype)).mean().item()

    def _auto_beta(self, d2_rel, own, n_steps=50):
        target = self.init_responsibility
        _check(target > 1.0 / self.n_components,
               f"init_responsibility must be above 1/k = {1.0 / self.n_components:.4f}")
        score = lambda lb: self._own_responsibility(torch.softmax(-(10.0 ** lb) * d2_rel, dim=1), own)
        lo, hi = -4.0, 6.0
        if score(hi) < target:
            warnings.warn("init_responsibility not reached; using the largest beta tried",
                          RuntimeWarning)
            return 10.0 ** hi
        if score(lo) >= target:
            return 10.0 ** lo
        for _ in range(n_steps):
            mid = 0.5 * (lo + hi)
            lo, hi = (mid, hi) if score(mid) < target else (lo, mid)
        return 10.0 ** hi

    @staticmethod
    def _loss(K, W, A, const, d, exact=False):
        # sum_i d_i [K_ii - 2 (W * KA)_i. + (W (A^T K A) * W)_i.] / n
        #   = mean over rows and columns of d_i (X - W A^T X)^2   (A is rows x k)
        KA = K @ A
        DW = W if d is None else d[:, None] * W
        cross, quad = DW * KA, (A.T @ KA) * (W.T @ DW)
        if exact:
            cross, quad = cross.double(), quad.double()
        return (const - 2 * cross.sum() + quad.sum()) / K.shape[0]

    def _mse(self, Xh):
        """Weighted MSE of X - W_ @ H_ (rows of X = training rows)."""
        d = None if self.row_weight_ is None else self.row_weight_.double()[:, None]
        error = 0.0
        for start, block in self._chunks(Xh):
            sq = (block - self.W_ @ self.H_[:, start:start + block.shape[1]]).double().square()
            error += (sq if d is None else sq * d).sum().item()
        return error / Xh.numel()

    @torch.no_grad()
    def fit(self, X, row_weight=None):
        """X: rows x columns (DataFrame, array or tensor).

        row_weight: one nonnegative weight per row for the loss. An array
        follows row order; a labeled Series is lined up with X's index.
        """
        names = list(X.index) if isinstance(X, pd.DataFrame) else None
        self.col_names_ = list(X.columns) if isinstance(X, pd.DataFrame) else None
        self.row_names_ = names
        Xh = self._host(X)
        n, m = Xh.shape
        _check(self.n_components <= n, "n_components cannot exceed the number of rows")
        self.random_state_ = (int(np.random.randint(2 ** 31 - 1)) if self.random_state is None
                              else int(self.random_state))
        self.n_features_in_, self._n_train_ = m, n
        self.row_weight_ = None if row_weight is None else self._dev(row_weights(row_weight, names, n))
        d = self.row_weight_

        K = self._gram(Xh)
        W, A = self._initialize(K)
        scale = A.sum(dim=0, keepdim=True).clamp_min(self.eps)   # A columns must sum to one
        W, A = W * scale, A / scale

        W = torch.nn.Parameter(W)
        Z = torch.nn.Parameter(torch.log(A.clamp_min(torch.finfo(self.dtype).tiny)))
        optimizer = torch.optim.Adam([W, Z], lr=self.lr)
        diag = K.diagonal().double()
        const = (diag if d is None else d.double() * diag).sum().item()
        mse = lambda: self._loss(K, W, torch.softmax(Z, dim=0), const, d, exact=True).clamp_min(0).item()

        self.loss_, self.loss_iterations_ = [mse()], [0]
        self.n_iter_, self.converged_ = 0, False
        for iteration in range(1, self.max_iter + 1):
            optimizer.zero_grad(set_to_none=True)
            with torch.enable_grad():
                loss = self._loss(K, W, torch.softmax(Z, dim=0), const, d)
                if not torch.isfinite(loss):
                    raise RuntimeError("Nonfinite loss; reduce lr or rescale X")
                loss.backward()
            optimizer.step()
            W.clamp_(min=0)
            self.n_iter_ = iteration
            if iteration % self.check_every == 0 or iteration == self.max_iter:
                previous, value = self.loss_[-1], mse()
                self.loss_.append(value)
                self.loss_iterations_.append(iteration)
                if self.verbose:
                    print(f"Iteration {iteration}: MSE={value:.6g}")
                if self.tol and 0 <= previous - value <= self.tol * max(previous, self.eps):
                    self.converged_ = True
                    break

        self.W_ = W.detach().clone()                              # n x k
        self.A_ = torch.softmax(Z, dim=0).detach().T.contiguous()  # k x n
        del optimizer, Z, K
        self.H_ = torch.cat([self.A_ @ block for _, block in self._chunks(Xh)], dim=1)
        self.reconstruction_mse_ = self._mse(Xh)
        return self

    def fit_transform(self, X, row_weight=None):
        return self.fit(X, row_weight=row_weight).W

    def transform(self, X):
        return self.transform_W(X)

    # ------------------------------------------------------------------ #
    # Transforms
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def transform_W(self, X, H=None, expand=False):
        """New rows x training columns -> nonnegative W (new rows x k).

        Mixed-sign multiplicative projection with H fixed. H: project on this
        (k x columns) matrix instead of the model's H_, e.g. H from a consensus
        W. expand=True appends the new rows to W_ (own H only).
        """
        self._require_fitted()
        _check(H is None or not expand, "expand=True only works with the model's own H")
        H = self.H_ if H is None else self._dev(H)
        Xh = self._host(X)
        n_new, m = Xh.shape
        _check(H.shape == (self.n_components, m), "X columns must match the columns of H")

        generator = torch.Generator(device=self.device).manual_seed(self.random_state_)
        W = torch.rand(n_new, self.n_components, device=self.device, dtype=self.dtype,
                       generator=generator) + 0.2
        B = torch.zeros(n_new, self.n_components, device=self.device, dtype=torch.float64)
        for start, block in self._chunks(Xh):
            B += (block @ H[:, start:start + block.shape[1]].T).double()
        B = (B / m).to(self.dtype)
        Q = H @ H.T / m
        Bp, Bn, Qp, Qn = B.clamp_min(0), (-B).clamp_min(0), Q.clamp_min(0), (-Q).clamp_min(0)

        previous = None
        for iteration in range(1, self.max_iter + 1):
            W *= torch.sqrt((Bp + W @ Qn).clamp_min(self.eps) / (Bn + W @ Qp).clamp_min(self.eps))
            if iteration % self.check_every == 0 or iteration == self.max_iter:
                value = (((W @ Q) * W).double().sum() - 2 * (W * B).double().sum()).item()
                if not np.isfinite(value):
                    raise RuntimeError("Nonfinite projection loss; use float64")
                if previous is not None and self.tol and \
                        0 <= previous - value <= self.tol * max(abs(previous), self.eps):
                    break
                previous = value
        if expand:
            self.W_ = torch.cat([self.W_, W], dim=0)
        return W.cpu().numpy()

    def _solve_H(self, Xh, W, ridge):
        # H = (W^T D W + lam I)^-1 W^T D X,  lam = ridge * trace(W^T D W) / k
        n, k = W.shape
        _check(Xh.shape[0] == n, f"X must have one row per training row ({n}), in training order")
        d = (torch.ones(n, device=self.device, dtype=torch.float64) if self.row_weight_ is None
             else self.row_weight_.double())
        DW = W.double() * d[:, None]
        G = W.double().T @ DW
        G += ridge * max(G.diagonal().sum().item() / k, self.eps) * torch.eye(
            k, device=self.device, dtype=torch.float64)
        L = torch.linalg.cholesky(G)
        return torch.cat([torch.cholesky_solve(DW.T @ block.double(), L).to(self.dtype)
                          for _, block in self._chunks(Xh)], dim=1)

    @torch.no_grad()
    def transform_H(self, X, W=None, ridge=1e-6, expand=False):
        """Training rows x any columns -> H (k x columns) by weighted ridge
        least squares with W fixed:

            H = argmin_H  sum_i d_i ||x_i - w_i H||^2 + lam ||H||^2

        d = the row weights from fit. W: None = the model's W, or any
        (training rows x k) matrix, e.g. a consensus W. X rows must be the
        training rows, in training order. expand=True appends the new columns
        to H_ (own W only). Note: H_ from fit is A @ X, so appended columns
        follow a different rule than the fitted ones.
        """
        self._require_fitted()
        _check(W is None or not expand,
               "expand=True only works with the model's own W")
        _check(ridge > 0, "ridge must be positive")
        W = self.W_[:self._n_train_] if W is None else self._dev(W)
        _check(W.shape == (self._n_train_, self.n_components),
               f"W must be training rows x components = ({self._n_train_}, {self.n_components})")
        H = self._solve_H(self._host(X), W, ridge)
        if expand:
            self.H_ = torch.cat([self.H_, H], dim=1)
        return H.cpu().numpy()

    # ------------------------------------------------------------------ #
    @property
    def W(self):
        self._require_fitted()
        return self.W_.cpu().numpy()

    @property
    def A(self):
        self._require_fitted()
        return self.A_.cpu().numpy()

    @property
    def H(self):
        self._require_fitted()
        return self.H_.cpu().numpy()

    @property
    def row_weight(self):
        """Row weights used in the loss (mean one), or None."""
        return None if self.row_weight_ is None else self.row_weight_.cpu().numpy()