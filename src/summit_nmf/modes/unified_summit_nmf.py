import torch
import numpy as np
import pandas as pd
from copy import deepcopy
from itertools import product
from tqdm import tqdm
from .base_summit_nmf import SummitNMFBase
from summit_nmf.dataloader import get_dataloader, SUMMITNMFDataset
from typing import Iterable, Optional, Union


class UnifiedSummitNMF(SummitNMFBase):
    """Unified Summit NMF implementation that combines multiple NMF methods."""

    def __init__(
        self,
        n_components: int = 25,
        lam: float = 1,
        flexible_label_matrix: bool = False,
        class_imbalance: bool = False,
        batch_size: int = None,
        max_iter: int = 200,
        device: str = None,
        random_state: int = None,
        eps: float = 1e-10,
        tol: float = 1e-4,
        verbose: bool = False,
        preload: bool = False,
        dtype: torch.dtype = torch.float32,
        stratify_labels: Optional[Iterable[Union[str, int]]] = None,
    ):
        device = device if device else torch.device("cpu")
        super().__init__(
            n_components,
            lam,
            flexible_label_matrix,
            class_imbalance,
            batch_size,
            max_iter,
            device,
            random_state,
            eps,
            tol,
            verbose,
            preload,
            dtype,
            stratify_labels,
        )

    def fit(self, X, y, **kwargs):
        W = kwargs.get("W", None)
        H = kwargs.get("H", None)
        Bs = kwargs.get("Bs", None)
        dataset = SUMMITNMFDataset(
            X=X,
            y=y,
            W=W,
            H=H,
            Bs=Bs,
            n_signatures=self.n_components,
            class_imbalance=self.class_imbalance,
            preload=self.preload,
            device=self.device,
            random_state=self.random_state,
            dtype=self.dtype,
        )

        self.ref_Y = dataset.get_ref_Y()
        self.n_covariates = len(self.ref_Y)
        self.cov_names = self.ref_Y.keys()
        self.alpha = self._check_alpha(self.alpha)
        self.batch_size = (
            self.batch_size if self.batch_size is not None else len(dataset)
        )
        dataloader = get_dataloader(
            dataset, batch_size=self.batch_size, stratify_labels=self.stratify_labels
        )

        self.loss_history_ = {
            "reconstruction": [],
            "harmonic": [],
            "total": [],
            **{f"harmonic_{cov_name}": [] for cov_name in self.cov_names},
        }

        iter_ = 0
        progress_bar = (
            tqdm(range(self.max_iter), desc="Training", postfix={"loss": 0.0})
            if self.verbose
            else range(self.max_iter)
        )
        for iter_ in progress_bar:
            for batch_data, batch_idx in dataloader:
                X, W, H, M, Ys, Ls, Bs = batch_data

                # compute L
                L = torch.ones_like(Ls[0], dtype=self.dtype, device=Ls[0].device)
                for l_mat in Ls:
                    L *= l_mat

                # update H
                H = self._update_H(X=X, W=W, H=H, M=M)

                # update Bs
                for i in range(self.n_covariates):
                    YB = self._compute_YB(Ys, Bs, L)
                    Bs[i] = self._update_B(Y=Ys[i], W=W, B=Bs[i], YB=YB, L=L)

                # update Ys
                if self.flexible_label_matrix:
                    for i in range(self.n_covariates):
                        YB = self._compute_YB(Ys, Bs, L)
                        Ys[i] = self._update_Y(Y=Ys[i], W=W, B=Bs[i], YB=YB, L=L)

                # update W
                YB = self._compute_YB(Ys, Bs, L)
                W = self._update_W(X=X, W=W, H=H, YB=YB, M=M, L=L)

                # update to dataset
                dataset.update(indices=batch_idx, W=W, H=H, Bs=Bs, Ys=Ys)

            reconstruction_loss, harmonic_loss, total_loss = self._compute_loss(
                X, Ys, W, H, Bs, M, L
            )
            self.loss_history_["reconstruction"].append(reconstruction_loss)
            self.loss_history_["harmonic"].append(harmonic_loss)
            self.loss_history_["total"].append(total_loss)

            if self.verbose:
                progress_bar.set_postfix({"loss": total_loss})

            iter_ += 1
            if (
                iter_ > 2
                and abs(
                    self.loss_history_["total"][-1] - self.loss_history_["total"][-2]
                )
                < self.tol
            ):
                break

        self.iter = iter_
        self.loss_history_ = pd.DataFrame(self.loss_history_)

        matrices = dataset.get_all_matrices()
        self.W = matrices["W"].cpu().numpy()
        self.H = matrices["H"].cpu().numpy()
        self.M = matrices["M"].cpu().numpy()
        self.Bs = [B.cpu().numpy() for B in matrices["Bs"]]
        self.Ys = [Y.cpu().numpy() for Y in matrices["Ys"]]

        L = torch.ones_like(Ls[0], dtype=self.dtype, device=Ls[0].device)
        for l_mat in Ls:
            L *= l_mat
        self.L = L

    def transform(self, X, y=None, max_iter=None):
        self._check_parameters(["H", "ref_Y"])
        max_iter = self.max_iter if max_iter is None else max_iter

        transform_dataset = SUMMITNMFDataset(
            X=X,
            y=y,
            W=None,
            H=self.H,
            Bs=self.Bs,
            n_signatures=self.n_components,
            class_imbalance=self.class_imbalance,
            preload=self.preload,
            device=self.device,
            random_state=self.random_state,
            dtype=self.dtype,
            ref_Y=self.ref_Y,
        )

        transform_dataloader = get_dataloader(
            transform_dataset,
            batch_size=self.batch_size,
            stratify_labels=None,
        )

        progress_bar = (
            tqdm(range(max_iter), desc="Transforming")
            if self.verbose
            else range(max_iter)
        )
        for iter_ in progress_bar:
            for batch_data, batch_idx in transform_dataloader:
                X, W, H, M, Ys, Ls, Bs = batch_data

                # compute L
                L = torch.ones_like(Ls[0], dtype=self.dtype, device=Ls[0].device)
                for l_mat in Ls:
                    L *= l_mat

                if y is not None:
                    if self.flexible_label_matrix:
                        YB = self._compute_YB(Ys, Bs, L)
                        Ys = [
                            self._update_Y(Y=Y, W=W, B=B, YB=YB, L=L)
                            for Y, B in zip(Ys, Bs)
                        ]

                    YB = self._compute_YB(Ys, Bs, L)
                    W = self._update_W(X=X, W=W, H=H, YB=YB, M=M, L=L)
                else:
                    # if is unsupervised, only update W using X
                    numerator = X @ H.T
                    denominator = W @ H @ H.T
                    W *= numerator / torch.clamp(denominator, min=self.eps)

        matrices = transform_dataset.get_all_matrices()
        if y is not None:
            Ys = [Y.cpu().numpy() for Y in matrices["Ys"]]
            return matrices["W"].cpu().numpy(), Ys
        else:
            return matrices["W"].cpu().numpy()

    def fit_transform(self, X, y):
        """Fit and transform the data using multiple NMF methods."""
        return self.fit(X, y).transform(X, y)

    def reconstruct(self):
        self._check_parameters(["W", "H"])
        return self.W @ self.H

    def get_components(self):
        L = np.ones_like(self.Ls[0])
        for l_mat in self.Ls:
            L *= l_mat

        matrices = {
            "W": deepcopy(self.W),
            "H": deepcopy(self.H),
            "M": deepcopy(self.M),
            "L": deepcopy(L),
            "Bs": deepcopy(self.Bs),
            "Ys": deepcopy(self.Ys),
        }
        return matrices

    def predict(self, X):
        # Ensure the model has been fitted and required parameters are available
        self._check_parameters(["H", "Bs", "ref_Y"])

        # Transform the input data to the latent space
        W_pred = self.transform(X)

        # Generate all combinations and their corresponding vectors
        result = {
            indices: sum(
                self.Bs[matrix_idx][row_idx, :]
                for matrix_idx, row_idx in enumerate(indices)
            )
            for indices in product(*(range(B.shape[0]) for B in self.Bs))
        }

        # Calculate the distance between W_pred and all combinations
        combinations = np.array(list(result.values()))
        D = np.linalg.norm(W_pred[:, np.newaxis, :] - combinations, axis=2)
        predicted_indices = np.argmin(D, axis=1)

        # Map the indices back to their original labels
        predicted_labels = [
            [
                label
                for cov_name, row_idx in zip(self.cov_names, list(result.keys())[idx])
                for label, mapped_idx in self.ref_Y[cov_name].items()
                if mapped_idx == row_idx
            ]
            for idx in predicted_indices
        ]
        predicted_labels = np.array(predicted_labels).T
        predicted_labels = {
            cov_name: labels
            for cov_name, labels in zip(self.cov_names, predicted_labels)
        }
        return predicted_labels

    def _check_parameters(self, parameters) -> None:
        missing_params = [param for param in parameters if not hasattr(self, param)]
        if missing_params:
            raise AttributeError(
                f"The following parameters are missing in the model: {', '.join(missing_params)}. Please fit the model first."
            )

    def _compute_reconstruction_loss(self, X, W, H, M):
        return torch.norm(M * (X - W @ H)).item() ** 2

    def _compute_harmonic_loss(self, Ys, W, Bs, L):
        YB = torch.zeros_like(W, dtype=self.dtype, device=self.device)
        for i in range(self.n_covariates):
            YB += Ys[i] @ Bs[i]
        return torch.norm(L * (W - YB)).item() ** 2

    def _compute_loss(self, X, Ys, W, H, Bs, M, L):
        reconstruction_loss = self._compute_reconstruction_loss(X, W, H, M)
        harmonic_loss = self._compute_harmonic_loss(Ys, W, Bs, L)
        return (
            reconstruction_loss,
            harmonic_loss,
            reconstruction_loss + self.lam * harmonic_loss,
        )

    def _compute_YB(self, Ys, Bs, L):
        YB = torch.zeros_like(L, dtype=self.dtype, device=self.device)
        for i in range(self.n_covariates):
            YB += Ys[i] @ Bs[i]
        return YB

    def _update_W(self, X, W, H, YB, M, L):
        recon_num = (M**2 * X) @ H.T
        recon_den = (M**2 * (W @ H)) @ H.T
        pred_num = L**2 * (YB)
        pred_den = L**2 * W

        numerator = recon_num + self.lam * pred_num
        denominator = recon_den + self.lam * pred_den
        return W * (numerator / torch.clamp(denominator, min=self.eps))

    def _update_H(self, X, W, H, M):
        numerator = W.T @ (M**2 * X)
        denominator = W.T @ (M**2 * (W @ H))
        return H * (numerator / torch.clamp(denominator, min=self.eps))

    def _update_B(self, Y, W, B, YB, L):
        numerator = Y.T @ (L**2 * W)
        denominator = Y.T @ (L**2 * (YB))
        return B * (numerator / torch.clamp(denominator, min=self.eps))

    def _update_Y(self, Y, W, B, YB, L):
        numerator = (L**2 * W) @ B.T
        denominator = (L**2 * (YB)) @ B.T
        return Y * (numerator / torch.clamp(denominator, min=self.eps))
