import torch
from typing import Iterable, Union

from summit_nmf.modes.seperated_summit_nmf import SeparatedSummitNMF
from summit_nmf.modes.unified_summit_nmf import UnifiedSummitNMF
from summit_nmf.modes.fast_seperated_summit_nmf import FastSeparatedSummitNMF
from summit_nmf.modes.fast_unified_summit_nmf import FastUnifiedSummitNMF


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
    ) -> Union[
        SeparatedSummitNMF,
        UnifiedSummitNMF,
        FastSeparatedSummitNMF,
        FastUnifiedSummitNMF,
    ]:
        """
        Factory function to create a Summit NMF model instance based on the specified model type.

        Parameters:
            model_type (str): The type of model to create. Options are:
                - "seperated" or "s": Creates a SeparatedSummitNMF instance.
                - "unified" or "u": Creates a UnifiedSummitNMF instance.
                - "fast_seperated" or "fs": Creates a FastSeparatedSummitNMF instance.
                - "fast_unified" or "fu": Creates a FastUnifiedSummitNMF instance.
            n_components (int): Number of components for the NMF model. Default is 25.
            lam (float): Regularization parameter. Default is 1.
            alpha (Iterable[float], optional): Regularization weights for the separated model. Default is None.
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
            num_workers (int): Number of workers for data loading. Default is 0.

        Returns:
            Union[SeparatedSummitNMF, UnifiedSummitNMF, FastSeparatedSummitNMF, FastUnifiedSummitNMF]:
                An instance of the specified Summit NMF model.

        Raises:
            ValueError: If an invalid `model_type` is provided.
        """

        if model_type in ["seperated", "s"]:
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
        elif model_type in ["unified", "u"]:
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
        elif model_type in ["fast_seperated", "fs"]:
            UserWarning(
                "Using FastSeparatedSummitNMF, please make sure the entire dataset can be loaded into device's memory. The following parameters won't be used: batch_size, preload, shuffle, and num_works."
            )

            return FastSeparatedSummitNMF(
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
        elif model_type in ["fast_unified", "fu"]:
            UserWarning(
                "Using FastSeparatedSummitNMF, please make sure the entire dataset can be loaded into device's memory. The following parameters won't be used: batch_size, preload, shuffle, and num_works."
            )

            return FastUnifiedSummitNMF(
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
            raise ValueError(
                f"Invalid model_type '{model_type}'. Must be 'seperated', 's', 'unified', 'u', 'fast_seperated', or 'fs'."
            )
