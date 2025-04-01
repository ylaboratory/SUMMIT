import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader, Sampler
from typing import Dict, Iterable, Optional, Union
from copy import deepcopy

TensorLike = Union[np.ndarray, torch.Tensor]


class StratifiedBatchSampler(Sampler):
    def __init__(
        self,
        labels: Iterable,
        batch_size: int,
        shuffle: bool = True,
        missing_label_value: Optional[Union[int, str]] = None,
    ):
        """
        Args:
            labels (Iterable): 1D array-like of sample labels for stratification.
            batch_size (int): Size of each batch.
            shuffle (bool): Whether to shuffle within strata.
            missing_label_value (Optional): Value to assign to missing labels (e.g., np.nan, None). If None, exclude them.
        """
        self.labels = np.array(labels)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.n_samples = len(self.labels)
        self.missing_label_value = missing_label_value

        # Handle NaN or missing labels
        if self.missing_label_value is not None:
            # Replace missing labels with a placeholder value
            mask = [is_missing(label) for label in self.labels]
            self.labels = np.where(mask, self.missing_label_value, self.labels)
        else:
            # Exclude missing labels entirely
            valid_mask = [not is_missing(label) for label in self.labels]
            self.labels = self.labels[valid_mask]
            self.n_samples = len(self.labels)

        self.unique_labels, self.counts = np.unique(self.labels, return_counts=True)
        self.indices = [
            np.where(self.labels == label)[0] for label in self.unique_labels
        ]

    def __iter__(self):
        all_indices = []
        for idx_list in self.indices:
            if self.shuffle:
                np.random.shuffle(idx_list)
            all_indices.extend(idx_list)

        for i in range(0, self.n_samples, self.batch_size):
            yield all_indices[i : i + self.batch_size]

    def __len__(self):
        return (self.n_samples + self.batch_size - 1) // self.batch_size


def is_missing(label) -> bool:
    """Check if a label represents a missing value."""
    return (
        (isinstance(label, float) and np.isnan(label))
        or label is None
        or pd.isna(label)
    )


class SUMMITNMFDataset(Dataset):
    def __init__(
        self,
        X: TensorLike,
        Y: Dict[str, Iterable[str]],
        W: TensorLike = None,
        H: TensorLike = None,
        Bs: Iterable[TensorLike] = None,
        n_components: int = None,
        class_imbalance: bool = False,
        preload: bool = False,
        device: str = "cpu",
        random_state: Optional[int] = None,
        dtype: torch.dtype = torch.float32,
        ref_Y: Optional[Dict[str, Dict[str, int]]] = None,
    ):
        self.preload = preload
        self.n_components = n_components
        self.device = torch.device(device)
        self.class_imbalance = class_imbalance
        self.dtype = dtype

        if random_state is not None:
            torch.manual_seed(random_state)

        # Convert X to tensor and pin memory if not preloading
        self.X = torch.as_tensor(X, dtype=self.dtype) if not torch.is_tensor(X) else X
        self.n_samples, self.n_features = self.X.shape
        if not self.preload:
            self.X = self.X.pin_memory()  # Optimize CPU-to-GPU transfer

        # Generate M mask from X
        self.M = torch.ones_like(self.X)
        self.M[torch.isnan(self.X)] = 0

        if not self.preload:
            self.M = self.M.pin_memory()

        # Set NaN entities in X to 0
        self.X[torch.isnan(self.X)] = 0

        # Handle Y (required)
        if Y is None:
            raise ValueError("Y (covariates) is required")
        self.Y_dict = Y
        self.covariate_order = list(Y.keys()) if ref_Y is None else list(ref_Y.keys())
        self.Ys = []
        self.Ls = []
        self.ref_Y = {} if ref_Y is None else ref_Y

        if ref_Y is None:
            self._process_Y(Y)
        else:
            self._process_test_Y(Y, ref_Y)

        # Initialize W, H, Bs if not provided
        median_X = torch.median(self.X)
        self.W = (
            torch.as_tensor(W, dtype=self.dtype)
            if W is not None
            else torch.rand(self.n_samples, n_components, dtype=self.dtype) * median_X
        )
        self.H = (
            torch.as_tensor(H, dtype=self.dtype)
            if H is not None
            else torch.rand(n_components, self.n_features, dtype=self.dtype) * median_X
        )
        if not self.preload:
            self.W = self.W.pin_memory()
            self.H = self.H.pin_memory()

        if Bs is not None:
            self.Bs = [torch.as_tensor(B, dtype=self.dtype) for B in Bs]
            assert len(self.Bs) == len(self.Ys), "Bs must match number of covariates"
        else:
            self.Bs = [
                torch.rand(Ys_i.shape[1], n_components, dtype=self.dtype)
                for Ys_i in self.Ys
            ]
        if not self.preload:
            self.Ys = [Y.pin_memory() for Y in self.Ys]
            self.Ls = [L.pin_memory() for L in self.Ls]
            self.Bs = [B.pin_memory() for B in self.Bs]

        # Preload to device if specified
        if self.preload:
            self.X = self.X.to(self.device)
            self.W = self.W.to(self.device)
            self.H = self.H.to(self.device)
            self.M = self.M.to(self.device)
            self.Ys = [Y.to(self.device) for Y in self.Ys]
            self.Ls = [L.to(self.device) for L in self.Ls]
            self.Bs = [B.to(self.device) for B in self.Bs]

        # Validate shapes
        assert self.W.shape == (self.n_samples, n_components), "W shape mismatch"
        assert self.H.shape == (n_components, self.n_features), "H shape mismatch"

        # Ensure non-negative property for W and H
        if torch.any(self.X < 0):
            raise ValueError("X contains negative values. Ensure X is non-negative.")
        if torch.any(self.W < 0):
            raise ValueError("W contains negative values. Ensure W is non-negative.")
        if torch.any(self.H < 0):
            raise ValueError("H contains negative values. Ensure H is non-negative.")
        for i, B in enumerate(self.Bs):
            if torch.any(B < 0):
                raise ValueError(
                    f"B[{i}] contains negative values. Ensure B is non-negative."
                )

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

    def _process_Y(self, Y: Dict[str, Iterable[str]]) -> None:
        """Process Y for training set: create Ys and Ls."""
        self.Ys = []
        self.Ls = []
        self.ref_Y = {}

        for cov_name in self.covariate_order:
            labels = Y[cov_name]
            valid_labels = [
                label
                for label in labels
                if isinstance(label, (str, int)) and not is_missing(label)
            ]
            invalid_labels = [
                label
                for label in labels
                if not isinstance(label, (str, int)) and not is_missing(label)
            ]
            if invalid_labels:
                raise ValueError(
                    f"Invalid labels found in '{cov_name}': {invalid_labels}. Labels must be str or int."
                )

            unique_labels = sorted(set(valid_labels))
            label_map = {label: idx for idx, label in enumerate(unique_labels)}
            n_labels = len(unique_labels)

            Y_mat = torch.zeros(self.n_samples, n_labels, dtype=self.dtype)
            L = torch.ones(self.n_samples, self.n_components, dtype=self.dtype)

            for sample_idx, label in enumerate(labels):
                if isinstance(label, (str, int)):
                    Y_mat[sample_idx, label_map[label]] = 1
            L = self._create_mask(Y_mat)

            self.Ys.append(Y_mat)
            self.Ls.append(L)
            self.ref_Y[cov_name] = label_map

    def _process_test_Y(
        self, Y: Dict[str, Iterable[str]], ref_Y: Dict[str, Dict[str, int]]
    ) -> None:
        """Process Y for test set: ensure consistency with training set."""
        self.Ys = []
        self.Ls = []

        for cov_name in self.covariate_order:
            if cov_name not in Y:
                raise ValueError(f"Covariate '{cov_name}' missing in test set")

            labels = Y[cov_name]
            label_map = ref_Y[cov_name]
            n_labels = len(label_map)
            ref_labels = set(label_map.keys())

            valid_test_labels = set(
                label
                for label in labels
                if isinstance(label, (str, int)) and not is_missing(label)
            )
            new_labels = valid_test_labels - ref_labels
            if new_labels:
                raise ValueError(
                    f"New labels {new_labels} in test set covariate '{cov_name}'"
                )

            invalid_labels = [
                label
                for label in labels
                if not isinstance(label, (str, int)) and not is_missing(label)
            ]
            if invalid_labels:
                raise ValueError(
                    f"Invalid labels found in test set '{cov_name}': {invalid_labels}"
                )

            Y_mat = torch.zeros(self.n_samples, n_labels, dtype=self.dtype)
            L = torch.ones(self.n_samples, self.n_components, dtype=self.dtype)

            for sample_idx, label in enumerate(labels):
                if isinstance(label, (str, int)) and label in label_map:
                    Y_mat[sample_idx, label_map[label]] = 1
            L = self._create_mask(Y_mat)

            self.Ys.append(Y_mat)
            self.Ls.append(L)

    def get_ref_Y(self) -> Dict[str, Dict[str, int]]:
        return deepcopy(self.ref_Y)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        # Minimize device transfers by reusing tensors if possible
        X_sample = self.X[idx]
        W_sample = self.W[idx]
        M_sample = self.M[idx]
        Ys_sample = [Y[idx] for Y in self.Ys]
        Ls_sample = [L[idx] for L in self.Ls]
        Bs_sample = self.Bs  # Reference, not copy
        H_sample = self.H  # Ensure H is included

        if not self.preload:
            # Asynchronous transfer with non_blocking=True
            X_sample = X_sample.to(self.device, non_blocking=True)
            W_sample = W_sample.to(self.device, non_blocking=True)
            M_sample = M_sample.to(self.device, non_blocking=True)
            Ys_sample = [Y.to(self.device, non_blocking=True) for Y in Ys_sample]
            Ls_sample = [L.to(self.device, non_blocking=True) for L in Ls_sample]
            Bs_sample = [B.to(self.device, non_blocking=True) for B in Bs_sample]
            H_sample = H_sample.to(self.device, non_blocking=True)  # Move H to device

        data = [
            X_sample,
            W_sample,
            H_sample,
            M_sample,
            Ys_sample,
            Ls_sample,
            Bs_sample,
        ]
        return data, idx  # Return data dict and idx

    def update(
        self,
        indices,
        W,
        H,
        Ys,
        Bs,
    ) -> None:
        W = torch.as_tensor(W, dtype=self.dtype).to(self.W.device)
        self.W[indices] = W

        H = torch.as_tensor(H, dtype=self.dtype).to(self.H.device)
        self.H = H

        for i, Y in enumerate(Ys):
            Y = torch.as_tensor(Y, dtype=self.dtype).to(self.Ys[i].device)
            self.Ys[i][indices] = Y

        for i, B in enumerate(Bs):
            B = torch.as_tensor(B, dtype=self.dtype).to(self.Bs[i].device)
            self.Bs[i] = B

    def get_all_matrices(self) -> Dict[str, Union[torch.Tensor, list[torch.Tensor]]]:
        """
        Return all matrices stored in the dataset.

        Returns:
            A dictionary containing the full X, W, H, M, Ys, Ls, and Bs matrices.
            - 'X': Tensor of shape (n_samples, n_features)
            - 'W': Tensor of shape (n_samples, n_signatures)
            - 'H': Tensor of shape (n_signatures, n_features)
            - 'M': Tensor of shape (n_samples, n_features)
            - 'Ys': List of tensors, each of shape (n_samples, n_labels) per covariate
            - 'Ls': List of tensors, each of shape (n_samples, n_signatures) per covariate
            - 'Bs': List of tensors, each of shape (n_labels, n_signatures) per covariate
        """
        return {
            "X": self.X,
            "W": self.W,
            "H": self.H,
            "M": self.M,
            "Ys": self.Ys,
            "Ls": self.Ls,
            "Bs": self.Bs,
        }


def get_dataloader(
    dataset, batch_size, stratify_labels=None, num_workers=4, shuffle=True
):
    if stratify_labels is not None:
        sampler = StratifiedBatchSampler(stratify_labels, batch_size, shuffle=shuffle)
        return DataLoader(
            dataset,
            batch_size=None,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=True,  # Optimize data loading
        )
    else:
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True,
        )
