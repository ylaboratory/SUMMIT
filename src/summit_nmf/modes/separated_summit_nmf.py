import torch
import numpy as np
import pandas as pd

from copy import deepcopy
from tqdm import tqdm
from torch.utils.data import DataLoader
from typing import Iterable, Dict, List, Union, Optional, Tuple

from summit_nmf.modes.base_summit_nmf import SummitNMFBase
from summit_nmf.utils.data_preprocess import DataPreprocessor
from summit_nmf.utils.dataset import summit_dataset

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
        dtype: torch.dtype = torch.float32,
        shuffle: bool = True,
        num_workers: int = 0,
        save_loss_interval: int = 5,
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
            dtype,
            shuffle,
            num_workers,
            save_loss_interval,
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

        Ys = [Y.to(self.device) for Y in Ys]
        W = W.to(self.device)
        H = H.to(self.device)
        Bs = [B.to(self.device) for B in Bs]
        Ls = [L.to(self.device) for L in Ls]

        dataset = summit_dataset(X=X, M=M)

        self.ref_Y_encoders = self.dp.ref_Y_encoders
        self.Y_label_encoded_order = {
            cov_name: self.ref_Y_encoders[cov_name].classes_
            for cov_name in self.ref_Y_encoders.keys()
        }
        self.n_covariates = len(self.ref_Y_encoders)
        self.cov_names = list(self.ref_Y_encoders.keys())
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

        self.iter_ = 0
        progress_bar = (
            tqdm(range(self.max_iter), desc="Training", postfix={"loss": 0.0})
            if self.verbose
            else range(self.max_iter)
        )

        for iter_ in progress_bar:
            self.iter_ = iter_

            for X_batch, M_batch, batch_idx in dataloader:
                X_batch = X_batch.to(self.device, non_blocking=True)
                Ys_batch = [Y[batch_idx] for Y in Ys]
                W_batch = W[batch_idx]
                Ls_batch = [L[batch_idx] for L in Ls]
                M_batch = M_batch.to(self.device, non_blocking=True)

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
                        self._update_Y(Y=Y, W=W, B=B, L=L)
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

                # update dataset
                W[batch_idx] = W_batch
                if self.flexible_label_matrix:
                    for i in range(self.n_covariates):
                        Ys[i][batch_idx] = Ys_batch[i]

            # compute and save the loss
            if iter_ % self.save_loss_interval == 0:
                X_ref, M_ref = dataset.get_all_matrices()
                W_copy = W.clone().cpu()
                H_copy = H.clone().cpu()
                Bs_copy = [B.clone().cpu() for B in Bs]
                Ys_copy = [Y.clone().cpu() for Y in Ys]
                Ls_copy = [L.clone().cpu() for L in Ls]
                reconstruction_loss, harmonic_loss, ind_harmonic_loss, total_loss = (
                    self._compute_loss(
                        X_ref, Ys_copy, W_copy, H_copy, Bs_copy, M_ref, Ls_copy
                    )
                )
                self.loss_history_["reconstruction"].append(reconstruction_loss)
                self.loss_history_["harmonic"].append(harmonic_loss)
                for i, loss in enumerate(ind_harmonic_loss):
                    self.loss_history_[f"harmonic_{self.cov_names[i]}"].append(loss)
                self.loss_history_["total"].append(total_loss)

                if self.verbose:
                    progress_bar.set_postfix({"loss": total_loss})

                # check for convergence
                if len(self.loss_history_["total"]) > 2:
                    total_diff = abs(
                        self.loss_history_["total"][-1]
                        - self.loss_history_["total"][-2]
                    )
                    if total_diff < self.tol:
                        break

        self.loss_history_ = pd.DataFrame(self.loss_history_)
        self.loss_history_["n_epoch"] = [
            i * self.save_loss_interval for i in range(len(self.loss_history_))
        ]

        _, M = dataset.get_all_matrices()
        self.W = W.cpu().numpy()
        self.H = H.cpu().numpy()
        self.M = M.numpy()
        self.Bs = [B.cpu().numpy() for B in Bs]
        self.Ys = [Y.cpu().numpy() for Y in Ys]
        self.Ls = [L.cpu().numpy() for L in Ls]
        return self

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
        self._check_parameters(["H", "ref_Y_encoders"])
        max_iter = self.max_iter if max_iter is None else max_iter

        H = torch.tensor(self.H, dtype=self.dtype).to(self.device)
        Bs = [torch.tensor(B, dtype=self.dtype).to(self.device) for B in self.Bs]

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
        Ys = [Y.to(self.device) for Y in Ys]
        W = W.to(self.device)
        Ls = [L.to(self.device) for L in Ls]

        dataset = summit_dataset(X=X, M=M)

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
            for X_batch, M_batch, batch_idx in dataloader:
                X_batch = X_batch.to(self.device, non_blocking=True)
                Ys_batch = [Y[batch_idx] for Y in Ys]
                W_batch = W[batch_idx]
                Ls_batch = [L[batch_idx] for L in Ls]
                M_batch = M_batch.to(self.device, non_blocking=True)

                if y is not None:
                    if self.flexible_label_matrix:
                        Ys_batch = [
                            self._update_Y(Y=Y, W=W, B=B, L=L)
                            for Y, B, L in zip(Ys_batch, Bs, Ls_batch)
                        ]

                    W_batch = self._update_W(
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

                # update dataset
                W[batch_idx] = W_batch
                if y is not None:
                    for i in range(self.n_covariates):
                        Ys[i][batch_idx] = Ys_batch[i]

        if y is not None:
            Ys = [Y.cpu().numpy() for Y in Ys]
            return W.cpu().numpy(), Ys
        else:
            return W.cpu().numpy()

    def reconstruct(self) -> np.ndarray:
        self._check_parameters(["W", "H"])
        return self.W @ self.H

    def predict(self, X: TensorLike) -> Dict[str, np.ndarray[str]]:
        self._check_parameters(["H", "Bs", "ref_Y_encoders"])

        # Transform the input data to the latent space
        W_pred = self.transform(X)

        # Calculate the distance between W_pred and each Bs[c] separately
        predicted_labels = {}
        for cov_name, B in zip(self.cov_names, self.Bs):
            label_encoder = self.ref_Y_encoders[cov_name]
            # Compute the distance between W_pred and each row of B
            D = np.linalg.norm(W_pred[:, np.newaxis, :] - B[np.newaxis, :, :], axis=2)
            predicted_indices = np.argmin(D, axis=1)

            # Use the label encoder to inverse transform the indices back to labels
            predicted_labels[cov_name] = label_encoder.inverse_transform(
                predicted_indices
            )

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
            alpha = [1.0 / self.n_covariates] * self.n_covariates
            return alpha

        if not isinstance(alpha, list):
            raise ValueError("Alpha should be a list of floats.")
        if len(alpha) != self.n_covariates:
            raise ValueError(f"Alpha should have length {self.n_components}.")

        for a in alpha:
            if a < 0:
                raise ValueError("Alpha values should be non-negative.")

        if abs(sum(alpha) - 1) > 1e-3:
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

    def __str__(self):
        return (
            "SeparatedSummitNMF(\n"
            f"  n_components={self.n_components},\n"
            f"  lam={self.lam},\n"
            f"  alpha={self.alpha},\n"
            f"  flexible_label_matrix={self.flexible_label_matrix},\n"
            f"  class_imbalance={self.class_imbalance},\n"
            f"  batch_size={self.batch_size},\n"
            f"  max_iter={self.max_iter},\n"
            f"  device={self.device},\n"
            f"  random_state={self.random_state},\n"
            f"  eps={self.eps},\n"
            f"  tol={self.tol},\n"
            f"  verbose={self.verbose},\n"
            f"  dtype={self.dtype},\n"
            f"  shuffle={self.shuffle},\n"
            f"  num_workers={self.num_workers}\n"
            ")"
        )

    def __repr__(self):
        return self.__str__()
