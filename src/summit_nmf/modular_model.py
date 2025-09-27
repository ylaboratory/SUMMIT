import torch
import numpy as np
import pandas as pd

from copy import deepcopy
from tqdm import tqdm
from typing import Union, Dict, List, Union, Optional
from itertools import product
from sklearn.decomposition._nmf import _initialize_nmf # type: ignore

from summit_nmf.base import SummitNMFBase
from contextlib import nullcontext
from .utils.encoder import FeatureEncoders
from .utils.sampling import (
    generate_epoch_indices,
    get_batch_indices,
    get_num_batches,
    create_joint_labels_from_dummy_matrices,
)


class ModularSummitNMF(SummitNMFBase):
    def __init__(
        self,
        n_components: int,
        alpha: List[Union[int, float]] = [],
        lam: Union[int, float] = 1.0,
        weighted_sampling: bool = False,
        batch_size: Optional[int] = None,
        max_iter: int = 200,
        init: str = "random",
        device: str = "cpu",
        random_state: int = 0,
        eps: float = 1e-10,
        tol: float = 1e-4,
        dtype: torch.dtype = torch.float32,
        save_loss_interval: int = 5,
        verbose: bool = False,
    ):
        super().__init__(
            n_components,
            lam,
            weighted_sampling,
            batch_size,
            max_iter,
            init,
            device,
            random_state,
            eps,
            tol,
            dtype,
            save_loss_interval,
            verbose,
        )
        self.alpha = alpha

    def _init_matrices(self, X, y):
        # set random seed for cpu and gpu
        torch.manual_seed(self.random_state)
        if self.device.type == "cuda":
            torch.cuda.manual_seed(self.random_state)

        # initialize X
        self.X = X.to(device=self.device, dtype=self.dtype)
        self.n_samples, self.n_features = X.shape

        # Initialize W and H with non-negative random values
        W, H = _initialize_nmf(X.cpu().numpy(), self.n_components, init=self.init, random_state=self.random_state)
        self.W = torch.tensor(W, device=self.device, dtype=self.dtype).clamp(min=self.eps)
        self.H = torch.tensor(H, device=self.device, dtype=self.dtype).clamp(min=self.eps)

        # Encode covariates
        self.covariate_keys = y.columns.tolist()
        self.fe: FeatureEncoders = FeatureEncoders(self.covariate_keys)
        Ys = self.fe.fit_transform(y)

        self.Ys: List[torch.Tensor] = [
            torch.tensor(Y, dtype=self.dtype, device=self.device) for Y in Ys
        ]

        # Initialize B matrices
        self.Bs = [
            torch.rand(
                (Y.shape[1], self.n_components), device=self.device, dtype=self.dtype
            )
            for Y in self.Ys
        ]
        self.Bs = [B.clamp(min=self.eps) for B in self.Bs]

        # Initialize auxiliary variables and R
        self.ones = torch.ones(
            (self.n_samples, 1), device=self.device, dtype=self.dtype
        )
        self.Rs = [
            torch.rand((1, self.n_components), device=self.device, dtype=self.dtype)
            for _ in range(len(self.covariate_keys))
        ]

        # Initialize alpha
        if len(self.alpha) == 0:
            self.alpha = [1.0 for _ in range(len(self.covariate_keys))]
        elif len(self.alpha) != len(self.covariate_keys):
            raise ValueError(
                f"Length of alpha must be equal to number of covariates ({len(self.covariate_keys)})."
            )
        elif any(a < 0 for a in self.alpha):
            raise ValueError("All elements in alpha must be non-negative.")
        elif sum(self.alpha) - 1 > 1e-6:
            raise ValueError("Sum of alpha must be 1.")

    def _fit(self, X: torch.Tensor, y: pd.DataFrame):
        self._init_matrices(X, y)
        loss_history: List[List[float]] = []

        # setting progress bar
        pbar = tqdm(total=self.max_iter, desc="Iteration", ncols=100, disable=not self.verbose)

        joint_labels = create_joint_labels_from_dummy_matrices(self.Ys)

        if self.batch_size is None:
            self.batch_size = X.shape[0]

        with torch.no_grad():
            with pbar:
                for i_iter in range(self.max_iter):
                    # Generate indices for the entire epoch
                    epoch_indices = generate_epoch_indices(
                        joint_labels=joint_labels,
                        weighted_sampling=self.weighted_sampling,
                        device=self.device,
                    )

                    # calculate how many batches needed
                    num_batches = get_num_batches(len(epoch_indices), self.batch_size)

                    # iterate over all batches
                    for batch_num in range(num_batches):
                        batch_indices = get_batch_indices(
                            epoch_indices, batch_num, self.batch_size
                        )
                        if len(batch_indices) == 0:
                            break

                        X_batch = self.X[batch_indices, :]
                        Ys_batch = [Y[batch_indices, :] for Y in self.Ys]
                        W_batch = self.W[batch_indices, :]
                        ones_batch = self.ones[batch_indices, :]

                        # update H
                        numerator = 2 * W_batch.T @ X_batch
                        denominator = 2 * W_batch.T @ (W_batch @ self.H)
                        self.H = self.H * (numerator / denominator.clamp(min=self.eps))

                        # update Bs
                        for idx in range(len(self.Bs)):
                            Y_batch = Ys_batch[idx]

                            # update B
                            numerator = (
                                2 * self.lam * self.alpha[idx] * Y_batch.T @ W_batch
                            )
                            denominator = (
                                2
                                * self.lam
                                * self.alpha[idx]
                                * Y_batch.T
                                @ (Y_batch @ self.Bs[idx] + ones_batch @ self.Rs[idx])
                            )
                            self.Bs[idx] *= numerator / denominator.clamp(min=self.eps)

                        # update Rs
                        for idx in range(len(self.Rs)):
                            Y_batch = Ys_batch[idx]

                            # update R
                            numerator = (
                                2 * self.lam * self.alpha[idx] * self.ones.T @ W_batch
                            )
                            denominator = (
                                2
                                * self.lam
                                * self.alpha[idx]
                                * self.ones.T
                                @ (Y_batch @ self.Bs[idx] + ones_batch @ self.Rs[idx])
                            )
                            self.Rs[idx] *= numerator / denominator.clamp(min=self.eps)

                        # update W
                        YBR = torch.zeros_like(W_batch)
                        for a, Y, B, R in zip(self.alpha, Ys_batch, self.Bs, self.Rs):
                            YBR += a * (Y @ B + ones_batch @ R)
                        numerator = 2 * X_batch @ self.H.T + 2 * self.lam * YBR
                        denominator = (
                            2 * W_batch @ (self.H @ self.H.T)
                            + 2 * self.lam * sum(self.alpha) * W_batch
                        )
                        W_batch = W_batch * (
                            numerator / denominator.clamp(min=self.eps)
                        )
                        self.W[batch_indices, :] = W_batch

                    # compute loss
                    if (i_iter % self.save_loss_interval == 0) or (
                        i_iter == self.max_iter - 1
                    ):
                        loss = self._compute_loss()
                        loss_history.append([i_iter] + loss)
                        pbar.set_postfix({"loss": f"{loss[0]:.4f}"})
                    pbar.update()

                    # check convergence
                    if len(loss_history) > 1:
                        if (i_iter > 0) and (
                            abs(loss_history[-1][1] - loss_history[-2][1]) < self.tol
                        ):
                            print(f"Converged at iteration {i_iter}.")
                            break

        self.loss_history = pd.DataFrame(
            loss_history,
            columns=["iteration", "total_loss", "reconstruction_loss"]
            + [f"harmonic_loss_{i}" for i in range(len(self.covariate_keys))],
        )

        # move everything to cpu
        self.X = self.X.cpu()
        self.W = self.W.cpu()
        self.H = self.H.cpu()
        self.Ys = [Y.cpu() for Y in self.Ys]
        self.Bs = [B.cpu() for B in self.Bs]
        self.Rs = [R.cpu() for R in self.Rs]
        self.ones = self.ones.cpu()

    def _transform(self, X: torch.Tensor, max_iter: int) -> torch.Tensor:
        if not self._is_fitted:
            raise RuntimeError("Model is not fitted yet.")
        self._validate_X(X)

        # Check feature dimension
        n_samples, n_feature = X.shape
        if n_feature != self.n_features:
            raise ValueError(
                f"X must have {self.n_features} features, but got {n_feature}."
            )
        X = X.to(device=self.device, dtype=self.dtype)

        # Initialize W and copy H
        W = torch.rand(
            (n_samples, self.n_components), device=self.device, dtype=self.dtype
        )
        W = W.clamp(min=self.eps)
        H = self.H.clone().to(device=self.device, dtype=self.dtype)

        for _ in range(max_iter):
            # Update W
            W *= (X @ H.T) / (W @ (H @ H.T)).clamp(min=self.eps)

        return W.cpu()

    def _compute_loss(self) -> List[float]:
        reconstruction_loss = self._compute_reconstruction_loss()
        harmonic_losses = self._compute_harmonic_loss()
        total_loss = reconstruction_loss + self.lam * sum(
            [a * hl for a, hl in zip(self.alpha, harmonic_losses)]
        )
        return [total_loss, reconstruction_loss] + harmonic_losses

    def _compute_reconstruction_loss(self) -> float:
        reconstruction = self.W @ self.H
        loss = torch.norm(self.X - reconstruction, p="fro") ** 2
        return loss.item()

    def _compute_harmonic_loss(self) -> List[float]:
        diffs = []
        for Y, B, R in zip(self.Ys, self.Bs, self.Rs):
            diff = self.W - Y @ B - self.ones @ R
            diffs.append((diff.norm(p="fro") ** 2).item())
        return diffs

    def _predict(self, X: torch.Tensor, max_iter: int) -> pd.DataFrame:
        if not self._is_fitted:
            raise RuntimeError("Model is not fitted yet.")
        self._validate_X(X)

        W = self.transform(X, max_iter)

        decoded_labels = {}
        for i, cov in enumerate(self.covariate_keys):
            BR = self.Bs[i] + self.Rs[i].expand_as(self.Bs[i])
            # For each sample, find the closest category embedding
            distances = torch.cdist(
                W, BR, p=2
            )  # shape: (n_samples, n_categories)
            min_indices = torch.argmin(distances, dim=1)  # shape: (n_samples,)

            # Decode indices to original labels
            labels = [
                self.fe.encoded_labels[cov][cat_idx]
                for cat_idx in min_indices
            ]
            
            # Remove covariate prefix from labels
            labels = [
                label.replace(f"{cov}_", "")
                for label in labels
            ]
            decoded_labels[cov] = labels

        # Convert to DataFrame with appropriate column names
        pred_df = pd.DataFrame(decoded_labels)
        return pred_df

    def get_label_embeddings(self):
        if not self._is_fitted:
            raise RuntimeError("Model is not fitted yet.")
        label_embeddings: Dict[str, Dict[str, np.ndarray]] = {}

        for i, cov in enumerate(self.covariate_keys):
            label_embeddings[cov] = {}
            for j, category in enumerate(self.fe.encoded_labels[cov]):
                # embedding = (self.Bs[i][j] + self.Rs[i]).cpu().numpy()
                embedding = self.Bs[i][j].cpu().numpy()
                label_embeddings[cov][category.replace(f"{cov}_", "")] = embedding
        return label_embeddings

    def _get_matrices(self, to_numpy: bool):
        if to_numpy:
            return {
                "W": self.W.numpy(),
                "H": self.H.numpy(),
                "Ys": [Y.numpy() for Y in self.Ys],
                "Bs": [B.numpy() for B in self.Bs],
                "Rs": [R.numpy() for R in self.Rs],
                "ones": self.ones.numpy(),
            }
        else:
            return {
                "W": self.W,
                "H": self.H,
                "Ys": self.Ys,
                "Bs": self.Bs,
                "Rs": self.Rs,
                "ones": self.ones,
            }
