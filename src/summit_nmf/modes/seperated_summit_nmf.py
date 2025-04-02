import torch
import numpy as np
import pandas as pd

from copy import deepcopy
from tqdm import tqdm
from torch.utils.data import DataLoader
from typing import Iterable, Dict, List, Union, Optional, Tuple

from summit_nmf.modes.base_summit_nmf import SummitNMFBase
from summit_nmf.utils.data_preprocess import DataPreprocessor
from summit_nmf.utils.dataset import preload_dataset, swap_dataset

TensorLike = Union[np.ndarray, torch.Tensor]


class SeparatedSummitNMF(SummitNMFBase):
    def __init__(
        self,
        n_components: int = 25,
        lam: float = 1,
        alpha: Iterable[float] = None,
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
        shuffle: bool = True,
        num_workers: int = 0,
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
            shuffle,
            num_workers,
        )
        self.alpha = alpha

    def fit(self, X: TensorLike, y: Dict[str, Iterable[str | int]], **kwargs):
        W = kwargs.get("W", None)
        H = kwargs.get("H", None)
        Bs = kwargs.get("Bs", None)

        self.dp = DataPreprocessor(
            n_components=self.n_components,
            class_imbalance=self.class_imbalance,
            random_state=self.random_state,
            dtype=self.dtype,
        )
        X, Ys, W, H, Bs, M, Ls = self.dp.fit(X, y, W, H, Bs)
        H = H.to(self.device, non_blocking=True)
        Bs = [B.to(self.device, non_blocking=True) for B in Bs]

        dataset = None
        if self.preload:
            dataset = preload_dataset(X=X, Ys=Ys, W=W, M=M, Ls=Ls, device=self.device)
        else:
            dataset = swap_dataset(X=X, Ys=Ys, W=W, M=M, Ls=Ls, device=self.device)

        self.ref_Y = self.dp.ref_Y
        self.n_covariates = len(self.ref_Y)
        self.cov_names = self.ref_Y.keys()
        self.alpha = self._check_alpha(self.alpha)
        self.batch_size = (
            self.batch_size if self.batch_size is not None else len(dataset)
        )

        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            pin_memory=True,
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
                X_batch, W_batch, M_batch, Ys_batch, Ls_batch = batch_data

                # update H
                H = self._update_H(X=X_batch, W=W_batch, H=H, M=M_batch)

                # update Bs
                Bs = [
                    self._update_B(Y=Y, W=W_batch, B=B, L=L)
                    for Y, B, L in zip(Ys_batch, Bs, Ls_batch)
                ]

                # update Ys
                if self.flexible_label_matrix:
                    Ys_batch = [
                        self._update_Y(Y=Y, W=W_batch, B=B, L=L)
                        for Y, B, L in zip(Ys_batch, Bs, Ls_batch)
                    ]

                # update W
                W_batch = self._update_W(
                    X=X_batch,
                    Ys=Ys_batch,
                    W=W_batch,
                    H=H,
                    Bs=Bs,
                    M=M_batch,
                    Ls=Ls_batch,
                )

                # update to dataset
                dataset.update(indices=batch_idx, W=W_batch, Ys=Ys_batch)

            # loss is computed on the entire dataset, and are on cpu
            # to avoid memory issues
            X, W, M, Ys, Ls = dataset.get_all_matrices()
            H_copy = H.clone().cpu()
            Bs_copy = [B.clone().cpu() for B in Bs]
            reconstruction_loss, harmonic_loss, ind_harmonic_loss, total_loss = (
                self._compute_loss(X, Ys, W, H_copy, Bs_copy, M, Ls)
            )
            self.loss_history_["reconstruction"].append(reconstruction_loss)
            self.loss_history_["harmonic"].append(harmonic_loss)
            for i, loss in enumerate(ind_harmonic_loss):
                self.loss_history_[f"harmonic_{self.n_cov_names[i]}"].append(loss)
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

        X, W, M, Ys, Ls = dataset.get_all_matrices()
        self.W = W.numpy()
        self.H = H.numpy()
        self.M = M.numpy()
        self.Bs = [B.numpy() for B in Bs]
        self.Ys = [Y.numpy() for Y in Ys]
        self.Ls = [L.numpy() for L in Ls]

    def fit_transform(
        self, X: TensorLike, y: Dict[str, Iterable[str | int]]
    ) -> Tuple[np.ndarray, List[np.ndarray]]:
        return self.fit(X, y).transform(X, y)

    def transform(
        self,
        X: TensorLike,
        y: Dict[str, Iterable[str | int]] = None,
        max_iter: Optional[int] = None,
    ):
        self._check_parameters(["H", "ref_Y"])
        max_iter = self.max_iter if max_iter is None else max_iter

        H = torch.tensor(self.H, dtype=self.dtype).to(self.device, non_blocking=True)
        Bs = [
            torch.tensor(B, dtype=self.dtype).to(self.device, non_blocking=True)
            for B in self.Bs
        ]

        if y is None:
            X, W, M = self.dp.transform(X)
            Ys = [
                torch.zeros((X.shape[0], B.shape[0]), dtype=self.dtype) for B in Bs
            ]  # placeholder
            Ls = [
                torch.zeros((X.shape[0], B.shape[1]), dtype=self.dtype) for B in Bs
            ]  # placeholder
        else:
            X, Ys, W, M, Ls = self.dp.transform(X, y)

        dataset = None
        if self.preload:
            dataset = preload_dataset(X=X, Ys=Ys, W=W, M=M, Ls=Ls, device=self.device)
        else:
            dataset = swap_dataset(X=X, Ys=Ys, W=W, M=M, Ls=Ls, device=self.device)

        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            pin_memory=True,
        )

        progress_bar = (
            tqdm(range(max_iter), desc="Transforming")
            if self.verbose
            else range(max_iter)
        )
        for iter_ in progress_bar:
            for batch_data, batch_idx in dataloader:
                X_batch, W_batch, M_batch, Ys_batch, Ls_batch = batch_data

                if y is not None:
                    if self.flexible_label_matrix:
                        Ys_batch = [
                            self._update_Y(Y=Y, W=W, B=B, L=L)
                            for Y, B, L in zip(Ys_batch, Bs, Ls_batch)
                        ]

                    W = self._update_W(
                        X=X_batch,
                        Ys=Ys_batch,
                        W=W_batch,
                        H=H,
                        Bs=Bs,
                        M=M_batch,
                        Ls=Ls_batch,
                    )
                else:
                    # if is unsupervised, only update W using X
                    numerator = X_batch @ H.T
                    denominator = W_batch @ H @ H.T
                    W_batch *= numerator / torch.clamp(denominator, min=self.eps)

                dataset.update(indices=batch_idx, W=W_batch, Ys=Ys_batch)

        X, W, M, Ys, Ls = dataset.get_all_matrices()
        if y is not None:
            Ys = [Y.numpy() for Y in Ys]
            return W.numpy(), Ys
        else:
            return W.numpy()

    def reconstruct(self) -> np.ndarray:
        self._check_parameters(["W", "H"])
        return self.W @ self.H

    def predict(self, X: TensorLike) -> Dict[str, List[str]]:
        self._check_parameters(["H", "Bs", "ref_Y"])

        # Transform the input data to the latent space
        W_pred = self.transform(X)

        # Calculate the distance between W_pred and each Bs[c] separately
        predicted_labels = {}
        for cov_name, B in zip(self.cov_names, self.Bs):
            label_mapping = self.ref_Y[cov_name]
            # Compute the distance between W_pred and each row of B
            D = np.linalg.norm(W_pred[:, np.newaxis, :] - B[np.newaxis, :, :], axis=2)
            predicted_indices = np.argmin(D, axis=1)

            # Map the indices back to their original labels
            labels = np.array([label for label, idx in label_mapping.items()])
            predicted_labels[cov_name] = labels[predicted_indices]

        return predicted_labels

    def get_components(self) -> Dict[str, np.ndarray]:
        matrices = {
            "W": deepcopy(self.W),
            "H": deepcopy(self.H),
            "M": deepcopy(self.M),
            "Bs": deepcopy(self.Bs),
            "Ys": deepcopy(self.Ys),
            "Ls": deepcopy(self.Ls),
        }
        return matrices

    def _check_alpha(self, alpha):
        """Check if alpha is a list of floats."""
        if alpha is None:
            alpha = [1.0 / self.n_covariates]
            return alpha

        if not isinstance(alpha, list):
            raise ValueError("Alpha should be a list of floats.")
        if len(alpha) != self.n_covariates:
            raise ValueError(f"Alpha should have length {self.n_components}.")

        for a in alpha:
            if a < 0:
                raise ValueError("Alpha values should be non-negative.")

        if sum(alpha) != 1:
            raise ValueError("Alpha values should sum to 1.")
        return alpha

    def _check_parameters(self, parameters) -> None:
        missing_params = [param for param in parameters if not hasattr(self, param)]
        if missing_params:
            raise AttributeError(
                f"The following parameters are missing in the model: {', '.join(missing_params)}. Please fit the model first."
            )

    def _compute_reconstruction_loss(self, X, W, H, M):
        return torch.norm(M * (X - W @ H)).item() ** 2

    def _compute_harmonic_loss(self, Ys, W, Bs, Ls):
        total_loss = 0
        ind_loss = [0] * self.n_covariates
        for i in range(self.n_covariates):
            ind_loss[i] = (
                self.alpha[i] * torch.norm(Ls[i] * (W - Ys[i] @ Bs[i])).item() ** 2
            )
            total_loss += ind_loss[i]
        return total_loss, ind_loss

    def _compute_loss(self, X, Ys, W, H, Bs, M, Ls):
        reconstruction_loss = self._compute_reconstruction_loss(X, W, H, M)
        total_pred_loss, ind_pred_loss = self._compute_harmonic_loss(Ys, W, Bs, Ls)
        return (
            reconstruction_loss,
            total_pred_loss,
            ind_pred_loss,
            reconstruction_loss + self.lam * total_pred_loss,
        )

    def _update_W(self, X, Ys, W, H, Bs, M, Ls):
        recon_num = (M**2 * X) @ H.T
        recon_den = (M**2 * (W @ H)) @ H.T

        pred_num, pred_den = 0, 0
        for Y, B, L, alpha in zip(Ys, Bs, Ls, self.alpha):
            pred_num += alpha * L**2 * (Y @ B)
            pred_den += alpha * L**2 * W

        numerator = recon_num + self.lam * pred_num
        denominator = recon_den + self.lam * pred_den
        return W * (numerator / torch.clamp(denominator, min=self.eps))

    def _update_H(self, X, W, H, M):
        numerator = W.T @ (M**2 * X)
        denominator = W.T @ (M**2 * (W @ H))
        return H * (numerator / torch.clamp(denominator, min=self.eps))

    def _update_B(self, Y, W, B, L):
        numerator = Y.T @ (L**2 * W)
        denominator = Y.T @ (L**2 * (Y @ B))
        return B * (numerator / torch.clamp(denominator, min=self.eps))

    def _update_Y(self, Y, W, B, L):
        numerator = (L**2 * W) @ B.T
        denominator = (L**2 * (Y @ B)) @ B.T
        return Y * (numerator / torch.clamp(denominator, min=self.eps))
