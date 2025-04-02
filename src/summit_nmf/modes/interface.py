import torch
from typing import Iterable, Union

from summit_nmf.modes.seperated_summit_nmf import SeparatedSummitNMF
from summit_nmf.modes.unified_summit_nmf import UnifiedSummitNMF


class SummitNMF:
    @staticmethod
    def create_summit_nmf(
        model_type: str = "seperated",
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
        preload: bool = False,
        dtype: torch.dtype = torch.float32,
        shuffle: bool = True,
        num_workers: int = 0,
    ) -> Union[SeparatedSummitNMF, UnifiedSummitNMF]:
        """
        Factory function to create a SummitNMF model instance.

        Parameters:
            model_type (str): Specifies the type of model to create.
                Must be either "seperated" or "unified". Default is "seperated".
            n_components (int): Number of components for the NMF model. Default is 25.
            lam (float): Regularization parameter. Default is 1.
            alpha (Iterable[float], optional): Only for `model_type` is "seperated". Regularization parameters for each component. Default is None.
            flexible_label_matrix (bool): Whether to use a flexible label matrix. Default is False.
            class_imbalance (bool): Whether to account for class imbalance. Default is False.
            batch_size (int, optional): Batch size for training. Default is None.
            max_iter (int): Maximum number of iterations for training. Default is 200.
            device (str, optional): Device to use for computation (e.g., "cpu" or "cuda"). Default is None.
            random_state (int, optional): Random seed for reproducibility. Default is None.
            eps (float): Small constant to avoid division by zero. Default is 1e-10.
            tol (float): Tolerance for convergence. Default is 1e-4.
            verbose (bool): Whether to print verbose output during training. Default is False.
            preload (bool): Whether to preload data into memory. Default is False.
            dtype (torch.dtype): Data type for computations. Default is torch.float32.
            shuffle (bool): Whether to shuffle data during training. Default is True.
            num_workers (int): Number of worker threads for data loading. Default is 0.

        Returns:
            SeparatedSummitNMF or UnifiedSummitNMF: An instance of the specified SummitNMF model.

        Raises:
            ValueError: If `model_type` is not "seperated" or "unified".
        """

        if model_type == "seperated":
            return SeparatedSummitNMF(
                n_components,
                lam,
                alpha,
                flexible_label_matrix,
                class_imbalance,
                batch_size,
                max_iter,
                device,
                random_state,
                eps,
                tol,
                verbose,
                preload,
                dtype,
                shuffle,
                num_workers,
            )
        elif model_type == "unified":
            return UnifiedSummitNMF(
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
                preload,
                dtype,
                shuffle,
                num_workers,
            )
        else:
            raise ValueError("model_type must be either 'seperated' or 'unified'")
