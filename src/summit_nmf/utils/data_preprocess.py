import torch
import numpy as np
import pandas as pd
from typing import Dict, Iterable, Optional, Any, Union, Tuple
from sklearn.preprocessing import LabelEncoder

TensorLike = Union[np.ndarray, torch.Tensor]


def is_missing(label) -> bool:
    """Check if a label represents a missing value."""
    return (
        (isinstance(label, float) and np.isnan(label))
        or label is None
        or pd.isna(label)
    )


class DataPreprocessor:
    def __init__(
        self,
        n_components: int = None,
        class_imbalance: bool = False,
        random_state: Optional[int] = None,
        dtype: torch.dtype = torch.float32,
    ):
        self.n_components = n_components
        self.class_imbalance = class_imbalance
        self.dtype = dtype

        self.rng = torch.Generator()
        if random_state is not None:
            self.rng.manual_seed(random_state)

        self.ref_Y_encoders = {}  # Store label encoders for each covariate

    def fit(
        self,
        X: TensorLike,
        Y: Dict[str, Iterable[str]],
        W: Optional[TensorLike] = None,
        H: Optional[TensorLike] = None,
        Bs: Optional[Iterable[TensorLike]] = None,
    ):
        # Process X and create mask M
        X, M = self._process_X(X)
        self.n_samples, self.n_features = X.shape

        # create W and H
        median_X = torch.median(X)
        W = (
            torch.as_tensor(W, dtype=self.dtype)
            if W is not None
            else torch.rand(
                self.n_samples, self.n_components, dtype=self.dtype, generator=self.rng
            )
            * median_X
        )
        H = (
            torch.as_tensor(H, dtype=self.dtype)
            if H is not None
            else torch.rand(
                self.n_components, self.n_features, dtype=self.dtype, generator=self.rng
            )
            * median_X
        )

        # Validate shapes
        assert W.shape == (self.n_samples, self.n_components), "W shape mismatch"
        assert H.shape == (self.n_components, self.n_features), "H shape mismatch"

        # Ensure non-negative property
        if torch.any(W < 0):
            raise ValueError("W contains negative values")
        if torch.any(H < 0):
            raise ValueError("H contains negative values")

        # Process Y and create mask L
        self.covariate_order = list(Y.keys())
        Ys, Ls = self._process_Y(Y)

        # create Bs
        if Bs is not None:
            Bs = [torch.as_tensor(B, dtype=self.dtype) for B in Bs]
            assert len(Bs) == len(Ys), "Bs must match number of covariates"
        else:
            Bs = [
                torch.rand(
                    Ys_i.shape[1],
                    self.n_components,
                    dtype=self.dtype,
                    generator=self.rng,
                )
                for Ys_i in Ys
            ]

        # Ensure non-negative property
        for i, B in enumerate(Bs):
            if torch.any(B < 0):
                raise ValueError(f"B[{i}] contains negative values")

        return X, Ys, W, H, Bs, M, Ls

    def transform(
        self,
        X: TensorLike,
        Y: Optional[Dict[str, Iterable[str]]] = None,
    ):
        # Process X and create mask M
        X, M = self._process_X(X)
        n_samples, _ = X.shape

        # create W and H
        median_X = torch.median(X)
        W = torch.rand(n_samples, self.n_components, dtype=self.dtype) * median_X

        if Y is not None:
            Ys, Ls = self._process_test_Y(Y, n_samples)
            return X, Ys, W, M, Ls
        else:
            return X, W, M

    def _process_X(self, X: TensorLike) -> Tuple[torch.Tensor, torch.Tensor]:
        # Convert X to tensor and pin memory if not preloading
        X = torch.as_tensor(X, dtype=self.dtype) if not torch.is_tensor(X) else X

        # Generate M mask from X, and replace NaN values with 0 in X
        M = torch.ones_like(X)
        M[torch.isnan(X)] = 0
        X[torch.isnan(X)] = 0

        if torch.any(X < 0):
            raise ValueError("X contains negative values")
        return X, M

    def _process_Y(
        self, Y: Dict[str, Iterable[Any]]
    ) -> Tuple[list[torch.Tensor], list[torch.Tensor]]:
        Ys = []
        Ls = []

        for cov_name in self.covariate_order:
            labels = Y[cov_name]

            # Use LabelEncoder to normalize labels
            encoder = LabelEncoder()
            valid_labels = [label for label in labels if not is_missing(label)]
            encoder.fit(valid_labels)
            self.ref_Y_encoders[cov_name] = encoder  # Store encoder for reference

            n_labels = len(encoder.classes_)
            Y_mat = torch.zeros(self.n_samples, n_labels, dtype=self.dtype)
            for sample_idx, label in enumerate(labels):
                if not is_missing(label):
                    Y_mat[sample_idx, encoder.transform([label])[0]] = 1

            L = self._create_mask(Y_mat)
            Ys.append(Y_mat)
            Ls.append(L)

        return Ys, Ls

    def _process_test_Y(
        self, Y: Dict[str, Iterable[Any]], n_samples: int
    ) -> Tuple[list[torch.Tensor], list[torch.Tensor]]:
        Ys = []
        Ls = []

        for cov_name in self.covariate_order:
            if cov_name not in Y:
                raise ValueError(f"Covariate '{cov_name}' missing in test set")
            labels = Y[cov_name]

            # Use stored LabelEncoder to transform test labels
            encoder = self.ref_Y_encoders[cov_name]
            valid_labels = [label for label in labels if not is_missing(label)]
            if new_labels := set(valid_labels) - set(encoder.classes_):
                raise ValueError(f"New labels {new_labels} in test set '{cov_name}'")

            n_labels = len(encoder.classes_)
            Y_mat = torch.zeros(n_samples, n_labels, dtype=self.dtype)
            for sample_idx, label in enumerate(labels):
                if not is_missing(label):
                    Y_mat[sample_idx, encoder.transform([label])[0]] = 1

            L = self._create_mask(Y_mat)
            Ys.append(Y_mat)
            Ls.append(L)

        return Ys, Ls

    def _create_mask(self, Y):
        n_sample, n_classes = Y.shape
        L = torch.ones((n_sample, self.n_components), dtype=self.dtype)

        # Compute class weights for imbalance
        if self.class_imbalance:
            n_total_samples = Y.shape[0]
            class_weights = n_total_samples / (
                n_classes * Y.sum(dim=0).clamp(min=1)
            )  # Avoid division by zero

        # Find the samples with no labels
        row_sums = Y.sum(dim=1, keepdim=True)  # Sum across classes per sample
        L[row_sums.squeeze() == 0] = 0  # Zero out rows where no class is assigned

        if self.class_imbalance:
            # Find the weighted sum of class weights for each sample
            sample_weights = (Y * class_weights).sum(
                dim=1, keepdim=True
            )  # Sum over active classes
            L *= sample_weights  # Broadcast multiplication across components

        return L
