import torch
import numpy as np
import pandas as pd

from copy import deepcopy
from tqdm import tqdm
from torch.utils.data import DataLoader
from typing import Iterable, Dict, List, Union, Tuple, Optional
from itertools import product

from summit_nmf.modes.base_summit_nmf import SummitNMFBase
from summit_nmf.utils.data_preprocess import DataPreprocessor
from summit_nmf.utils.dataset import hybrid_dataset

TensorLike = Union[np.ndarray, torch.Tensor]


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
            dtype,
            shuffle,
            num_workers,
        )

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
        H = H.to(self.device)
        Bs = [B.to(self.device) for B in Bs]

        dataset = hybrid_dataset(X=X, Ys=Ys, W=W, M=M, Ls=Ls, device=self.device)

        self.ref_Y_encoders = self.dp.ref_Y_encoders
        self.Y_label_encoded_order = {
            cov_name: self.ref_Y_encoders[cov_name].classes_
            for cov_name in self.ref_Y_encoders.keys()
        }
        self.n_covariates = len(self.ref_Y_encoders)
        self.cov_names = list(self.ref_Y_encoders.keys())
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
                X_batch = X_batch.to(self.device, non_blocking=True)

                # compute L
                L_batch = self._compute_L(Ls_batch)

                # update H
                H = self._update_H(X=X_batch, W=W_batch, H=H, M=M_batch)

                # update Bs
                for i in range(self.n_covariates):
                    YB = self._compute_YB(Ys_batch, Bs, L_batch)
                    Bs[i] = self._update_B(
                        Y=Ys_batch[i], W=W_batch, B=Bs[i], YB=YB, L=L_batch
                    )

                # update Ys
                if self.flexible_label_matrix:
                    for i in range(self.n_covariates):
                        YB = self._compute_YB(Ys_batch, Bs, L_batch)
                        Ys_batch[i] = self._update_Y(
                            Y=Ys_batch[i], W=W_batch, B=Bs[i], YB=YB, L=L_batch
                        )

                # update W
                YB = self._compute_YB(Ys_batch, Bs, L_batch)
                W_batch = self._update_W(
                    X=X_batch, W=W_batch, H=H, YB=YB, M=M_batch, L=L_batch
                )

                # update to dataset
                dataset.update(indices=batch_idx, W=W_batch, Ys=Ys_batch)

            # loss is computed on the entire dataset, and are on cpu
            # to avoid memory issues
            X, W, M, Ys, Ls = dataset.get_all_matrices()
            L = self._compute_L(Ls)
            H_copy = H.clone().cpu()
            Bs_copy = [B.clone().cpu() for B in Bs]
            reconstruction_loss, harmonic_loss, total_loss = self._compute_loss(
                X, Ys, W, H_copy, Bs_copy, M, L
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

        X, W, M, Ys, Ls = dataset.get_all_matrices()
        self.W = W.numpy()
        self.H = H.cpu().numpy()
        self.M = M.numpy()
        self.Bs = [B.cpu().numpy() for B in Bs]
        self.Ys = [Y.numpy() for Y in Ys]

        L = self._compute_L(Ls)
        self.L = L.cpu().numpy()
        return self

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

        dataset = hybrid_dataset(X=X, Ys=Ys, W=W, M=M, Ls=Ls, device=self.device)

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
                X_batch = X_batch.to(self.device, non_blocking=True)

                # compute L
                L = self._compute_L(Ls_batch)

                if y is not None:
                    if self.flexible_label_matrix:
                        YB = self._compute_YB(Ys_batch, Bs, L)
                        Ys_batch = [
                            self._update_Y(Y=Y, W=W, B=B, YB=YB, L=L)
                            for Y, B in zip(Ys_batch, Bs)
                        ]

                    YB = self._compute_YB(Ys_batch, Bs, L)
                    W_batch = self._update_W(
                        X=X_batch, W=W_batch, H=H, YB=YB, M=M_batch, L=L
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

    def fit_transform(
        self, X: TensorLike, y: Dict[str, Iterable[str]]
    ) -> Tuple[np.ndarray, List[np.ndarray]]:
        """Fit and transform the data using multiple NMF methods."""
        return self.fit(X, y).transform(X, y)

    def reconstruct(self) -> np.ndarray:
        self._check_parameters(["W", "H"])
        return self.W @ self.H

    def predict(self, X: TensorLike) -> Dict[str, List[str]]:
        # Ensure the model has been fitted and required parameters are available
        self._check_parameters(["H", "Bs", "ref_Y_encoders"])

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

        # Map the indices back to their original labels using inverse_transform
        predicted_labels = {
            cov_name: self.ref_Y_encoders[cov_name].inverse_transform(
                [list(result.keys())[idx][i] for idx in predicted_indices]
            )
            for i, cov_name in enumerate(self.cov_names)
        }
        return predicted_labels

    def get_components(self) -> Dict[str, np.ndarray]:
        matrices = {
            "W": deepcopy(self.W),
            "H": deepcopy(self.H),
            "M": deepcopy(self.M),
            "L": deepcopy(self.L),
            "Bs": deepcopy(self.Bs),
            "Ys": deepcopy(self.Ys),
        }
        return matrices

    def _check_parameters(self, parameters) -> None:
        missing_params = [param for param in parameters if not hasattr(self, param)]
        if missing_params:
            raise AttributeError(
                f"The following parameters are missing in the model: {', '.join(missing_params)}. Please fit the model first."
            )

    def _compute_L(self, Ls):
        L = torch.ones_like(Ls[0], dtype=self.dtype, device=Ls[0].device)
        for l_mat in Ls:
            L *= l_mat
        return L

    def _compute_reconstruction_loss(self, X, W, H, M):
        return torch.norm(M * (X - W @ H)).item() ** 2

    def _compute_harmonic_loss(self, Ys, W, Bs, L):
        YB = torch.zeros_like(W, dtype=self.dtype, device=L.device)
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
        YB = torch.zeros_like(L, dtype=self.dtype, device=L.device)
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

    def __str__(self):
        return (
            "UnifiedSummitNMF(\n"
            f"  n_components={self.n_components},\n"
            f"  lam={self.lam},\n"
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
