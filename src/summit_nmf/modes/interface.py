import torch
from typing import Iterable, Union

from summit_nmf.modes.separated_summit_nmf import StandardSummitNMF
from summit_nmf.modes.fast_separated_summit_nmf import FastSummitNMF


class SummitNMF:
    """
    SummitNMF class provides a factory method to create instances of different 
    SummitNMF modes, such as 'standard' or 'fast'.
    Methods
    -------
    new(
    ) -> Union[StandardSummitNMF, FastSummitNMF]
        Factory method to create an instance of either StandardSummitNMF or 
        FastSummitNMF based on the specified mode.
        Parameters
        ----------
        mode : str, optional
            The mode of the SummitNMF instance. Must be either 'standard' or 'fast'. 
            Defaults to 'standard'.
        n_components : int, optional
            Number of components for the NMF model. Defaults to 25.
        lam : float, optional
            Regularization parameter. Defaults to 1.
        alpha : Iterable[float], optional
            Regularization weights for each component. Defaults to None.
        class_imbalance : bool, optional
            Whether to handle class imbalance. Defaults to False.
        batch_size : int, optional
            Batch size for training. Not used in 'fast' mode. Defaults to None.
        max_iter : int, optional
            Maximum number of iterations for training. Defaults to 200.
        device : str, optional
            Device to use for computation (e.g., 'cpu', 'cuda'). Defaults to None.
        random_state : int, optional
            Random seed for reproducibility. Defaults to None.
        eps : float, optional
            Small constant for numerical stability. Defaults to 1e-10.
        tol : float, optional
            Tolerance for convergence. Defaults to 1e-4.
        verbose : bool, optional
            Whether to print verbose output. Defaults to False.
        dtype : torch.dtype, optional
            Data type for computations. Defaults to torch.float32.
        shuffle : bool, optional
            Whether to shuffle data during training. Not used in 'fast' mode. 
            Defaults to True.
        num_workers : int, optional
            Number of workers for data loading. Not used in 'fast' mode. 
            Defaults to 0.
        save_loss_interval : int, optional
            Interval for saving loss values during training. Defaults to 5.
        Returns
        -------
        Union[StandardSummitNMF, FastSummitNMF]
            An instance of either StandardSummitNMF or FastSummitNMF, depending on 
            the specified mode.
        Raises
        ------
        ValueError
            If an invalid mode is provided.
        Notes
        -----
        - In 'fast' mode, the entire dataset must fit into the device's memory. 
          Parameters such as `batch_size`, `shuffle`, and `num_workers` are ignored.
        - In 'standard' mode, the model supports batch processing and shuffling.
    """

    @staticmethod
    def new (
        mode: str = "standard",
        n_components: int = 25,
        lam: float = 1,
        alpha: Iterable[float] = None,
        class_imbalance: bool = False,
        batch_size: int = None,
        max_iter: int = 200,
        device: str = None,
        random_state: int = None,
        eps: float = 1e-10,
        tol: float = 1e-4,
        verbose: bool = False,
        dtype: torch.dtype = torch.float32,
        shuffle: bool = True,
        num_workers: int = 0,
        save_loss_interval: int = 5,
    ) -> Union[
        StandardSummitNMF,
        FastSummitNMF,
    ]:
        """
        Creates a new instance of either `StandardSummitNMF` or `FastSummitNMF` based on the specified mode.
        Parameters:
            mode (str): The mode of the NMF algorithm to use. Must be either "standard" or "fast". Default is "standard".
            n_components (int): Number of components for the NMF model. Default is 25.
            lam (float): Regularization parameter. Default is 1.
            alpha (Iterable[float], optional): Regularization parameters for each component. Default is None.
            class_imbalance (bool): Whether to account for class imbalance. Default is False.
            batch_size (int, optional): Batch size for processing data. Only used in "standard" mode. Default is None.
            max_iter (int): Maximum number of iterations for the algorithm. Default is 200.
            device (str, optional): Device to run the computation on (e.g., "cpu" or "cuda"). Default is None.
            random_state (int, optional): Seed for random number generation. Default is None.
            eps (float): Small constant to avoid division by zero. Default is 1e-10.
            tol (float): Tolerance for convergence. Default is 1e-4.
            verbose (bool): Whether to print progress messages. Default is False.
            dtype (torch.dtype): Data type for computations. Default is `torch.float32`.
            shuffle (bool): Whether to shuffle data during training. Only used in "standard" mode. Default is True.
            num_workers (int): Number of workers for data loading. Only used in "standard" mode. Default is 0.
            save_loss_interval (int): Interval for saving loss values during training. Default is 5.
        Returns:
            Union[StandardSummitNMF, FastSummitNMF]: An instance of the selected NMF model.
        Raises:
            ValueError: If the specified mode is not "standard" or "fast".
        Notes:
            - In "fast" mode, the entire dataset must fit into the device's memory. Parameters such as `batch_size`, 
              `shuffle`, and `num_workers` are ignored.
        """

        if mode == "standard":
            return StandardSummitNMF(
                n_components=n_components,
                lam=lam,
                alpha=alpha,
                class_imbalance=class_imbalance,
                batch_size=batch_size,
                max_iter=max_iter,
                device=device,
                random_state=random_state,
                eps=eps,
                tol=tol,
                verbose=verbose,
                dtype=dtype,
                shuffle=shuffle,
                num_workers=num_workers,
                save_loss_interval=save_loss_interval,
            )
        
        elif mode == "fast":

            if verbose:
                print(
                    "Using FastSummitNMF, please make sure the entire dataset can be loaded into device's memory.\n"
                    "The following parameters won't be used: batch_size, shuffle, and num_works."
                )

            return FastSummitNMF(
                n_components=n_components,
                lam=lam,
                alpha=alpha,
                class_imbalance=class_imbalance,
                batch_size=batch_size,
                max_iter=max_iter,
                device=device,
                random_state=random_state,
                eps=eps,
                tol=tol,
                verbose=verbose,
                dtype=dtype,
                shuffle=shuffle,
                num_workers=num_workers,
                save_loss_interval=save_loss_interval,
            )
        else:
            raise ValueError(
                f"Invalid mode '{mode}'. Must be 'standard' and 'fast'."
            )
