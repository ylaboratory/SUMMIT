import torch
import numpy as np
import pandas as pd

from copy import deepcopy
from tqdm import tqdm
from typing import Union, Dict, List, Union, Optional
from itertools import product

from summit_nmf.base import SummitNMFBase
from contextlib import nullcontext
from .utils.encoder import FeatureEncoders
from .utils.sampling import (
    generate_epoch_indices,
    get_batch_indices,
    get_num_batches,
    create_joint_labels_from_dummy_matrices,
)

class UnifiedSummitNMF(SummitNMFBase):
    def __init__(
        self,
        n_components: int,
        lam: Union[int, float] = 1.0,
        weighted_sampling: bool = False,
        batch_size: Optional[int] = None,
        max_iter: int = 200,
        device: str = "cpu",
        random_state: int = 0,
        eps: float = 1e-10,
        tol: float = 1e-4,
        dtype: torch.dtype = torch.float32,
        shuffle: bool = True,
        save_loss_interval: int = 5,
    ):
        super().__init__(
            n_components=n_components,
            lam=lam,
            weighted_sampling=weighted_sampling,
            batch_size=batch_size,
            max_iter=max_iter,
            device=device,
            random_state=random_state,
            eps=eps,
            tol=tol,
            dtype=dtype,
            shuffle=shuffle,
            save_loss_interval=save_loss_interval,
        )

    def _init_matrices(self, X, y):
        
        # initialize X
        self.X = X.to(device=self.device, dtype=self.dtype)
        self.n_samples, self.n_features = X.shape
        
        # Initialize W and H with non-negative random values
        self.W = torch.rand((self.n_samples, self.n_components), device=self.device, dtype=self.dtype)
        self.W = self.W.clamp(min=self.eps)

        self.H = torch.rand((self.n_components, self.n_features), device=self.device, dtype=self.dtype)
        self.H = self.H.clamp(min=self.eps)

        # Encode covariates
        self.covariate_keys = y.columns.tolist()
        self.fe: FeatureEncoders = FeatureEncoders(self.covariate_keys)
        Ys = self.fe.fit_transform(y)

        self.Ys: List[torch.Tensor] = [
            torch.tensor(Y, dtype=self.dtype, device=self.device)
            for Y in Ys
        ]

        # Initialize B matrices 
        self.Bs = [torch.rand((Y.shape[1], self.n_components), device=self.device, dtype=self.dtype) for Y in self.Ys]
        self.Bs = [B.clamp(min=self.eps) for B in self.Bs]

        # Initialize auxiliary variables and R
        self.ones = torch.ones((self.n_samples, 1), device=self.device, dtype=self.dtype)
        self.R = torch.rand((1, self.n_components), device=self.device, dtype=self.dtype)

        # Initialize beta
        self.beta = torch.rand(len(self.covariate_keys), device=self.device, dtype=self.dtype)

    def _fit(self, X: torch.Tensor, y: pd.DataFrame):
        self._init_matrices(X, y)
        loss_history: List[List[float]] = []

        # setting progress bar
        pbar = tqdm(total=self.max_iter, desc="Iteration", ncols=100)

        joint_labels = create_joint_labels_from_dummy_matrices(self.Ys)

        if self.batch_size is None:
            self.batch_size = X.shape[0]

        def _compute_YBR(Ys, Bs, ones, R, beta):
            YBR = ones @ R
            for b, Y, B in zip(beta, Ys, Bs):
                YBR += b * (Y @ B)
            return YBR

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

                        # update H
                        numerator = 2 * W_batch.T @ X_batch
                        denominator = 2 * W_batch.T @ (W_batch @ self.H)
                        self.H = self.H * (numerator / denominator.clamp(min=self.eps))

                        # update Bs
                        for idx in range(len(self.Bs)):
                            YBR = _compute_YBR(Ys_batch, self.Bs, self.ones, self.R, self.beta)
                            Y_batch = Ys_batch[idx]
                            
                            # update B
                            numerator = 2 * self.lam * self.beta[idx] * Y_batch.T @ W_batch
                            denominator = 2 * self.lam * self.beta[idx] * Y_batch.T @ YBR
                            self.Bs[idx] *= (numerator / denominator.clamp(min=self.eps))

                        # update R
                        YBR = _compute_YBR(Ys_batch, self.Bs, self.ones, self.R, self.beta)
                        numerator = 2 * self.lam * self.ones.T @ W_batch
                        denominator = 2 * self.lam * self.ones.T @ YBR
                        self.R *= (numerator / denominator.clamp(min=self.eps))

                        # update beta
                        for idx in range(len(self.beta)):
                            YBR = _compute_YBR(Ys_batch, self.Bs, self.ones, self.R, self.beta)
                            T0 = Ys_batch[idx] @ self.Bs[idx]

                            numerator = 2 * self.lam * torch.trace(T0 @ W_batch.T)
                            denominator = 2 * self.lam * torch.trace(T0 @ YBR.T)
                            self.beta[idx] *= (numerator / denominator.clamp(min=self.eps))
                        
                        # update W
                        YBR = _compute_YBR(Ys_batch, self.Bs, self.ones, self.R, self.beta)
                        numerator = 2 * X_batch @ self.H.T + 2 * self.lam * YBR
                        denominator = 2 * W_batch @ (self.H @ self.H.T) + 2 * self.lam * W_batch
                        W_batch = W_batch * (numerator / denominator.clamp(min=self.eps))
                        self.W[batch_indices, :] = W_batch                        
                    
                    # compute loss
                    if (i_iter % self.save_loss_interval == 0) or (i_iter == self.max_iter - 1):
                        loss = self._compute_loss()
                        loss_history.append([i_iter] + loss)
                        pbar.set_postfix({"loss": f"{loss[0]:.4f}"})
                    pbar.update()

                    # check convergence
                    if len(loss_history) > 1:
                        if (i_iter > 0) and (abs(loss_history[-1][1] - loss_history[-2][1]) < self.tol):
                            print(f"Converged at iteration {i_iter}.")
                            break

        self.loss_history = pd.DataFrame(
            loss_history, columns=["iteration", "total_loss", "reconstruction_loss", "harmonic_loss"]
        )

        # move everything to cpu
        self.X = self.X.cpu()
        self.W = self.W.cpu()
        self.H = self.H.cpu()
        self.Ys = [Y.cpu() for Y in self.Ys]
        self.Bs = [B.cpu() for B in self.Bs]
        self.R = self.R.cpu()
        self.ones = self.ones.cpu()
        self.beta = self.beta.cpu()
    
    def _transform(self, X: torch.Tensor, max_iter: int) -> torch.Tensor:
        if not self._is_fitted:
            raise RuntimeError("Model is not fitted yet.")
        self._validate_X(X)

        # Check feature dimension
        n_samples, n_feature = X.shape
        if n_feature != self.n_features:
            raise ValueError(f"X must have {self.n_features} features, but got {n_feature}.")
        X = X.to(device=self.device, dtype=self.dtype)

        # Initialize W and copy H
        W = torch.rand((n_samples, self.n_components), device=self.device, dtype=self.dtype)
        W = W.clamp(min=self.eps)
        H = self.H.clone().to(device=self.device, dtype=self.dtype)

        for _ in range(max_iter):
            # Update W
            W *= (X @ H.T) / (W @ (H @ H.T)).clamp(min=self.eps)

        return W.cpu()

    def _compute_loss(self) -> List[float]:
        reconstruction_loss = self._compute_reconstruction_loss()
        harmonic_loss = self._compute_harmonic_loss()
        total_loss = reconstruction_loss + self.lam * harmonic_loss
        return [total_loss, reconstruction_loss, harmonic_loss]

    def _compute_reconstruction_loss(self) -> float:
        reconstruction = self.W @ self.H
        loss = torch.norm(self.X - reconstruction, p='fro')**2
        return loss.item()

    def _compute_harmonic_loss(self) -> float:
        diff = torch.zeros_like(self.W)
        diff += self.W
        for beta, Y, B in zip(self.beta, self.Ys, self.Bs):
            diff -= beta * Y @ B
        return (diff.norm(p='fro') ** 2).item()
    
    def _predict(self, X: torch.Tensor, max_iter: int) -> pd.DataFrame:
        if not self._is_fitted:
            raise RuntimeError("Model is not fitted yet.")
        self._validate_X(X)

        W = self.transform(X, max_iter)
        joint_label_embeddings = self._get_joint_label_embeddings()

        # For each sample, find the closest joint label embedding
        distances = torch.cdist(W, joint_label_embeddings, p=2)  # shape: (n_samples, n_combinations)
        min_indices = torch.argmin(distances, dim=1)  # shape: (n_samples,)

        # Decode the combination indices to covariate values
        category_sizes = [Y.shape[1] for Y in self.Ys]
        combinations = list(product(*[range(size) for size in category_sizes]))
        predicted_combos = [combinations[idx] for idx in min_indices.cpu().numpy()]

        # Use encoded_labels to convert indices to original labels
        decoded_labels = []
        for combo in predicted_combos:
            labels = [
                self.fe.encoded_labels[key][cat_idx]
                for key, cat_idx in zip(self.covariate_keys, combo)
            ]
            labels = [label.replace(f"{key}_", "") for label, key in zip(labels, self.covariate_keys)]
            decoded_labels.append(labels)

        # Convert to DataFrame with appropriate column names
        pred_df = pd.DataFrame(decoded_labels, columns=self.covariate_keys)
        return pred_df

    def _get_joint_label_embeddings(self):
        # Get the number of categories for each covariate
        category_sizes = [Y.shape[1] for Y in self.Ys]
        # Generate all possible combinations of category indices
        combinations = list(product(*[range(size) for size in category_sizes]))
        joint_embeddings = torch.zeros((len(combinations), self.n_components), device="cpu", dtype=self.dtype)

        for idx, combo in enumerate(combinations):
            emb = torch.zeros(self.n_components, device="cpu", dtype=self.dtype)
            for i, cat_idx in enumerate(combo):
                emb += self.beta[i] * self.Bs[i][cat_idx, :]
            emb += self.R.squeeze(0)
            joint_embeddings[idx, :] = emb

        return joint_embeddings
    

    def _get_matrices(self, to_numpy: bool):
        if to_numpy:
            return {
                "W": self.W.numpy(),
                "H": self.H.numpy(),
                "Ys": [Y.numpy() for Y in self.Ys],
                "Bs": [B.numpy() for B in self.Bs],
                "R": self.R.numpy(),
                "ones": self.ones.numpy(),
                "beta": self.beta.numpy()
            }
        else:
            return {
                "W": self.W,
                "H": self.H,
                "Ys": self.Ys,
                "Bs": self.Bs,
                "R": self.R,
                "ones": self.ones,
                "beta": self.beta
            }