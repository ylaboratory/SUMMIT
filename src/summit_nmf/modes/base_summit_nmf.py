from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union, Iterable, Tuple
import numpy as np
import torch

ArrayLike = Union[np.ndarray, torch.Tensor]


class SummitNMFBase(ABC):
    """Abstract base class for all SUMMIT-NMF implementations."""

    def __init__(
        self,
        n_components: int,
        lam: float,
        alpha: Iterable[float],
        class_imbalance: bool,
        batch_size: int,
        max_iter: int,
        device: str,
        random_state: int,
        eps: float,
        tol: float,
        verbose: bool,
        dtype: torch.dtype,
        shuffle: bool,
        num_workers: int,
        save_loss_interval: int,
    ):
        self.n_components = n_components
        self.lam = lam
        self.alpha = alpha
        self.max_iter = max_iter
        self.class_imbalance = class_imbalance
        self.batch_size = batch_size
        self.device = device if device else torch.device("cpu")
        self.random_state = random_state
        self.eps = eps
        self.tol = tol
        self.verbose = verbose
        self.dtype = dtype
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.save_loss_interval = save_loss_interval

    @abstractmethod
    def fit(self, X: ArrayLike, y: Dict[str, Iterable[str]], **kwargs) -> None:
        """Fit the model to data."""
        pass

    @abstractmethod
    def transform(
        self,
        X: ArrayLike,
        y: Optional[Dict[str, Iterable[str]]] = None,
        max_iter: Optional[int] = None,
    ) -> Union[np.ndarray, Tuple[np.ndarray, List[np.ndarray]]]:
        """Transform new data using the learned components."""
        pass

    @abstractmethod
    def fit_transform(self, X: ArrayLike, y: Dict[str, Iterable[str]]) -> np.ndarray:
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
    def predict(self, X: ArrayLike) -> Dict[str, List[str]]:
        """Predict the output using the learned components."""
        pass
