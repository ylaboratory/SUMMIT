import torch
import numpy as np
import pandas as pd
from typing import Dict, Iterable, Optional, Any, Union, Tuple

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
            else torch.rand(self.n_samples, self.n_components, dtype=self.dtype)
            * median_X
        )
        H = (
            torch.as_tensor(H, dtype=self.dtype)
            if H is not None
            else torch.rand(self.n_components, self.n_features, dtype=self.dtype)
            * median_X
        )

        # Validate shapes
        assert self.W.shape == (self.n_samples, self.n_components), "W shape mismatch"
        assert self.H.shape == (self.n_components, self.n_features), "H shape mismatch"

        # Ensure non-negative property
        if torch.any(self.W < 0):
            raise ValueError("W contains negative values")
        if torch.any(self.H < 0):
            raise ValueError("H contains negative values")

        # Process Y and create mask L
        self.covariate_order = list(Y.keys())
        Ys, Ls = self._process_Y(Y)

        # create Bs
        if Bs is not None:
            Bs = [torch.as_tensor(B, dtype=self.dtype) for B in Bs]
            assert len(Bs) == len(Ys), "Bs must match number of covariates"
        else:
            self.Bs = [
                torch.rand(Ys_i.shape[1], self.n_components, dtype=self.dtype)
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
        n_samples, n_features = X.shape

        # create W and H
        median_X = torch.median(X)
        W = torch.rand(n_samples, self.n_components, dtype=self.dtype) * median_X

        if Y is not None:
            Ys, Ls = self._process_test_Y(Y)
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

    def _process_Y(
        self, Y: Dict[str, Iterable[Any]]
    ) -> Tuple[list[torch.Tensor], list[torch.Tensor]]:
        Ys = []
        Ls = []
        self.ref_Y = {}

        for cov_name in self.covariate_order:
            labels = Y[cov_name]
            normalized_labels = []
            invalid_labels = []

            # Normalize labels and check for invalid labels
            for label in labels:
                if is_missing(label):
                    continue
                elif isinstance(
                    label, (int, np.integer, torch.Tensor)
                ) and not isinstance(label, bool):
                    if isinstance(label, torch.Tensor):
                        label = label.item()
                    normalized_labels.append(str(int(label)))
                elif isinstance(label, str):
                    normalized_labels.append(label)
                else:
                    invalid_labels.append(label)
            if invalid_labels:
                raise ValueError(f"Invalid labels in '{cov_name}': {invalid_labels}")

            unique_labels = sorted(set(normalized_labels))
            label_map = {label: idx for idx, label in enumerate(unique_labels)}
            n_labels = len(unique_labels)

            # Create Y matrix
            Y_mat = torch.zeros(self.n_samples, n_labels, dtype=self.dtype)
            for sample_idx, label in enumerate(labels):
                if not is_missing(label):
                    if isinstance(label, torch.Tensor):
                        label = label.item()
                    label_str = (
                        str(int(label))
                        if isinstance(label, (int, np.integer))
                        else label
                    )
                    Y_mat[sample_idx, label_map[label_str]] = 1
            L = self._create_mask(Y_mat)

            Ys.append(Y_mat)
            Ls.append(L)
            self.ref_Y[cov_name] = label_map

        return Ys, Ls

    def _process_test_Y(
        self, Y: Dict[str, Iterable[Any]]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        Ys = []
        Ls = []

        for cov_name in self.covariate_order:
            if cov_name not in Y:
                raise ValueError(f"Covariate '{cov_name}' missing in test set")
            labels = Y[cov_name]
            label_map = self.ref_Y[cov_name]
            n_labels = len(label_map)
            ref_labels = set(label_map.keys())
            valid_test_labels = set()
            invalid_labels = []

            # Normalize labels and check for invalid labels
            for label in labels:
                if is_missing(label):
                    continue
                elif isinstance(
                    label, (int, np.integer, torch.Tensor)
                ) and not isinstance(label, bool):
                    if isinstance(label, torch.Tensor):
                        label = label.item()
                    valid_test_labels.add(str(int(label)))
                elif isinstance(label, str):
                    valid_test_labels.add(label)
                else:
                    invalid_labels.append(label)
            if new_labels := valid_test_labels - ref_labels:
                raise ValueError(f"New labels {new_labels} in test set '{cov_name}'")
            if invalid_labels:
                raise ValueError(
                    f"Invalid labels in test set '{cov_name}': {invalid_labels}"
                )

            # Create Y matrix
            Y_mat = torch.zeros(self.n_samples, n_labels, dtype=self.dtype)
            for sample_idx, label in enumerate(labels):
                if not is_missing(label):
                    if isinstance(label, torch.Tensor):
                        label = label.item()
                    label_str = (
                        str(int(label))
                        if isinstance(label, (int, np.integer))
                        else label
                    )
                    if label_str in label_map:
                        Y_mat[sample_idx, label_map[label_str]] = 1
            L = self._create_mask(Y_mat)

            Ys.append(Y_mat)
            Ls.append(L)
        return Ys, Ls
