import torch
import numpy as np
import pandas as pd

from abc import ABC, abstractmethod
from typing import Dict, Optional, Union, Self, Any


class SummitNMFBase(ABC):
    """Abstract base class for all SUMMIT-NMF implementations."""

    def __init__(
        self,
        n_components: int,
        lam: Union[int, float],
        weighted_sampling: bool,
        batch_size: Optional[int],
        max_iter: int,
        init: str,
        device: str,
        random_state: int,
        eps: float,
        tol: float,
        dtype: torch.dtype,
        save_loss_interval: int,
        verbose: bool
    ):
        self._validate_init_params(
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
            verbose
        )
        self.n_components = n_components
        self.lam = lam
        self.max_iter = max_iter
        self.weighted_sampling = weighted_sampling
        self.batch_size = batch_size
        self.init = init
        self.device = torch.device(device)
        self.random_state = random_state
        self.eps = eps
        self.tol = tol
        self.dtype = dtype
        self.save_loss_interval = save_loss_interval
        self.verbose = verbose
        self._is_fitted = False

    @abstractmethod
    def _fit(self, X, y):
        """Fit the model to data."""
        pass

    @abstractmethod
    def _transform(self, X, max_iter) -> torch.Tensor:
        """Transform new data using the learned components."""
        pass

    @abstractmethod
    def _predict(self, X, max_iter) -> pd.DataFrame:
        """Predict the output using the learned components."""
        pass

    @abstractmethod
    def _get_matrices(self, to_numpy: bool) -> Dict[str, Any]:
        """Get the learned components."""
        pass

    def fit(self, X: torch.Tensor, y: pd.DataFrame) -> Self:
        self._validate_inputs(X, y)
        self._fit(X, y)
        self._is_fitted = True
        return self

    def transform(
        self,
        X: torch.Tensor,
        max_iter: int = 200,
    ) -> torch.Tensor:
        if not self._is_fitted:
            raise RuntimeError("Model is not fitted yet.")
        self._validate_X(X)

        return self._transform(X, max_iter)

    def fit_transform(
        self, X: torch.Tensor, y: pd.DataFrame, max_iter: int = 200
    ) -> torch.Tensor:
        """Fit the model and return transformed data."""
        return self.fit(X, y).transform(X, max_iter=max_iter)

    def predict(self, X: torch.Tensor, max_iter: int = 200) -> pd.DataFrame:
        if not self._is_fitted:
            raise RuntimeError("Model is not fitted yet.")
        self._validate_X(X)
        return self._predict(X, max_iter=max_iter)

    def get_loss_history(self) -> pd.DataFrame:
        if not self._is_fitted:
            raise RuntimeError("Model is not fitted yet.")
        return self.loss_history.copy()  # type: ignore

    def get_matrices(self, to_numpy: bool = True):
        if not self._is_fitted:
            raise RuntimeError("Model is not fitted yet.")
        return self._get_matrices(to_numpy=to_numpy)

    def _validate_init_params(
        self,
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
        verbose
    ):
        if not isinstance(n_components, int) or n_components <= 0:
            raise ValueError("n_components must be a positive integer.")
        if not isinstance(lam, (int, float)) or lam < 0:
            raise ValueError("lam must be a non-negative float.")
        if not isinstance(weighted_sampling, bool):
            raise ValueError("weighted_sampling must be a boolean.")
        if batch_size is not None and (
            not isinstance(batch_size, int) or batch_size <= 0
        ):
            raise ValueError("batch_size must be a positive integer or None.")
        if not isinstance(max_iter, int) or max_iter <= 0:
            raise ValueError("max_iter must be a positive integer.")
        if init not in ["random", "nndsvd", "nndsvda", "nndsvdar"]:
            raise ValueError("init must be one of 'random', 'nndsvd', 'nndsvda', or 'nndsvdar'.")
        if not isinstance(device, str):
            raise ValueError("device must be a string.")
        if not isinstance(random_state, int):
            raise ValueError("random_state must be an integer.")
        if not isinstance(eps, float) or eps <= 0:
            raise ValueError("eps must be a positive float.")
        if not isinstance(tol, float) or tol < 0:
            raise ValueError("tol must be a non-negative float.")
        if not isinstance(dtype, torch.dtype):
            raise ValueError("dtype must be a torch.dtype.")
        if not isinstance(save_loss_interval, int) or save_loss_interval <= 0:
            raise ValueError("save_loss_interval must be a positive integer.")
        if not isinstance(verbose, bool):
            raise ValueError("verbose must be a boolean.")

    def _validate_X(self, X: torch.Tensor):
        if not isinstance(X, torch.Tensor):
            raise ValueError("X must be a torch.Tensor.")
        if X.dim() != 2:
            raise ValueError("X must be a 2D tensor.")
        if X.size(0) == 0 or X.size(1) == 0:
            raise ValueError("X must not be empty.")
        if X.min() < 0:
            raise ValueError("X must be non-negative.")

    def _validate_inputs(self, X: torch.Tensor, y: pd.DataFrame):
        self._validate_X(X)
        if not isinstance(y, pd.DataFrame):
            raise ValueError("y must be a pandas DataFrame.")
        if X.dim() != 2:
            raise ValueError("X must be a 2D tensor.")
        if X.size(0) == 0 or X.size(1) == 0:
            raise ValueError("X must not be empty.")
        if X.min() < 0:
            raise ValueError("X must be non-negative.")

        # Validate covariate keys
        if not isinstance(y, pd.DataFrame):
            raise ValueError("y must be a pandas DataFrame.")
        if y.empty:
            raise ValueError("y must not be empty.")
