from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union, Iterable, Tuple
import numpy as np
import torch

TensorLike = Union[np.ndarray, torch.Tensor]


class SummitNMFBase(ABC):
    """Abstract base class for all SUMMIT-NMF implementations."""

    def __init__(
        self,
        n_components: int,
        lam: float,
        flexible_label_matrix: bool,
        class_imbalance: bool,
        batch_size: int,
        max_iter: int,
        device: str,
        random_state: int,
        eps: float,
        tol: float,
        verbose: bool,
        preload: bool,
        dtype: torch.dtype,
        stratify_labels: Optional[Iterable[Union[str, int]]],
    ):
        self.n_components = n_components
        self.lam = lam
        self.max_iter = max_iter
        self.flexible_label_matrix = flexible_label_matrix
        self.class_imbalance = class_imbalance
        self.batch_size = batch_size
        self.device = device if device else torch.device("cpu")
        self.random_state = random_state
        self.eps = eps
        self.tol = tol
        self.verbose = verbose
        self.preload = preload
        self.dtype = dtype
        self.stratify_labels = stratify_labels

    @abstractmethod
    def fit(self, X: TensorLike, y: Dict[str, Iterable[str]], **kwargs) -> None:
        """Fit the model to data."""
        pass

    @abstractmethod
    def transform(
        self,
        X: TensorLike,
        y: Optional[Dict[str, Iterable[str]]] = None,
        max_iter: Optional[int] = None,
    ) -> Union[np.ndarray, Tuple[np.ndarray, List[np.ndarray]]]:
        """Transform new data using the learned components."""
        pass

    @abstractmethod
    def fit_transform(self, X: TensorLike, y: Dict[str, Iterable[str]]) -> np.ndarray:
        """Fit the model and return transformed data."""
        pass

    @abstractmethod
    def reconstruct(self) -> np.ndarray:
        """Reconstruct input data from the factorized matrices."""
        pass

    @abstractmethod
    def get_components(self) -> Dict[str, np.ndarray]:
        """Get the learned components."""
        pass

    @abstractmethod
    def predict(self, X: TensorLike) -> Dict[str, List[str]]:
        """Predict the output using the learned components."""
        pass
