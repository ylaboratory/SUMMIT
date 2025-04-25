import numpy as np
import pandas as pd
import optuna
import torch

from sklearn.metrics import f1_score
from sklearn.model_selection import KFold
from summit_nmf.modes.separated_summit_nmf import SeparatedSummitNMF
from summit_nmf.modes.fast_separated_summit_nmf import FastSeparatedSummitNMF
from summit_nmf.modes.unified_summit_nmf import UnifiedSummitNMF
from summit_nmf.modes.fast_unified_summit_nmf import FastUnifiedSummitNMF
from typing import Dict, Union


class Optimizer:
    def __init__(
        self,
        model_type: str,
        n_trials: int = 100,
        n_cv_splits: int = 3,
        flexible_label_matrix: bool = False,
        class_imbalance: bool = False,
        batch_size: int = None,
        max_iter: int = 200,
        device: str = None,
        random_state: int = None,
        eps: float = 1e-10,
        tol: float = 1e-4,
        dtype: torch.dtype = torch.float32,
        shuffle: bool = True,
        num_workers: int = 0,
        weight_n_comp: float = 0,
    ):
        self.model_type = model_type
        self.n_trials = n_trials
        self.n_cv_splits = n_cv_splits
        self.flexible_label_matrix = flexible_label_matrix
        self.class_imbalance = class_imbalance
        self.batch_size = batch_size
        self.max_iter = max_iter
        self.device = device
        self.random_state = random_state
        self.eps = eps
        self.tol = tol
        self.dtype = dtype
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.weight_n_comp = weight_n_comp

        # Initialize the appropriate optimizer based on the model type
        if self.model_type in ["unified", "u"]:
            self.model_class = UnifiedSummitNMF

        elif self.model_type in ["separated", "s"]:
            self.model_class = SeparatedSummitNMF

        elif self.model_type in ["fast_unified", "fu"]:
            UserWarning(
                "Using FastSeparatedSummitNMF, please make sure the dataset can be loaded into device's memory. The batch_size is set to the entire dataset."
            )
            self.model_class = FastUnifiedSummitNMF

        elif self.model_type in ["fast_separated", "fs"]:
            UserWarning(
                "Using FastSeparatedSummitNMF, please make sure the dataset can be loaded into device's memory. The batch_size is set to the entire dataset."
            )
            self.model_class = FastSeparatedSummitNMF

        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

    def search(
        self, X, y, n_components_range=(5, 50), lam_range=(1e-2, 1e2)
    ) -> Dict[str, Union[str, int, float]]:
        # y input preprocessing
        if isinstance(y, (np.ndarray, torch.Tensor)):
            y = {f"cov{i}": y[:, i] for i in range(y.shape[1])}
        self.covariates = list(y.keys())

        def objective(trial):
            n_components = trial.suggest_int(
                "n_components", n_components_range[0], n_components_range[1]
            )
            lam = trial.suggest_float("lam", lam_range[0], lam_range[1], log=True)

            num_params = len(y)

            alpha_raw = [
                -np.log(
                    trial.suggest_float(f"unnormalized_alpha_{i}", 1e-10, 1, log=True)
                )
                for i in range(num_params)
            ]
            alpha_sum = sum(alpha_raw)
            alphas = [a / alpha_sum for a in alpha_raw]

            for i in range(num_params):
                trial.set_user_attr(f"alpha_{i}", alphas[i])

            params = {
                "n_components": n_components,
                "lam": lam,
                "alpha": alphas,
                "max_iter": self.max_iter,
                "flexible_label_matrix": self.flexible_label_matrix,
                "class_imbalance": self.class_imbalance,
                "batch_size": self.batch_size,
                "eps": self.eps,
                "device": self.device,
                "dtype": self.dtype,
                "random_state": self.random_state,
                "tol": self.tol,
                "shuffle": self.shuffle,
                "num_workers": self.num_workers,
            }

            kf = KFold(
                n_splits=self.n_cv_splits, shuffle=True, random_state=self.random_state
            )
            harmonic_losses = []
            recon_losses = []
            f1_scores = [0] * len(y)

            for train_index, test_index in kf.split(X):
                X_train, X_test = X[train_index], X[test_index]
                y_train = {key: labels[train_index] for key, labels in y.items()}
                y_test = {key: labels[test_index] for key, labels in y.items()}

                model = self.model_class(**params)
                model.fit(X_train, y_train)

                y_pred = model.predict(X_test)
                score = 0
                for i, key in enumerate(y_pred.keys()):
                    non_nan_indices = np.array(
                        [
                            i
                            for i, val in enumerate(y_test[key])
                            if val is not None and val == val
                        ]
                    )
                    score += f1_score(
                        y_test[key][non_nan_indices],
                        y_pred[key][non_nan_indices],
                        average="weighted",
                    )

                    f1_scores[i] += f1_score(
                        y_test[key][non_nan_indices],
                        y_pred[key][non_nan_indices],
                        average="weighted",
                    )

                harmonic_losses.append(score)

                recon_loss = np.linalg.norm(
                    X_test - model.transform(X_test) @ model.H, ord="fro"
                ) / np.linalg.norm(X_test, ord="fro")
                recon_losses.append(recon_loss)

            f1_scores = [f1 / self.n_cv_splits for f1 in f1_scores]
            for i, key in enumerate(y_pred.keys()):
                trial.set_user_attr(f"f1_score_{key}", f1_scores[i])

            harmonic_mean = np.mean(harmonic_losses)
            recon_mean = np.mean(recon_losses)

            return recon_mean - harmonic_mean + self.weight_n_comp * n_components

        study = optuna.create_study(
            sampler=optuna.samplers.TPESampler(),
            pruner=optuna.pruners.MedianPruner(),
            direction="minimize",
            study_name="multi-covariates optimization (SeparatedSummitNMF)",
        )
        study.optimize(objective, n_trials=self.n_trials)

        best_alphas = []
        for i in range(len(y)):
            best_alphas.append(study.best_params[f"unnormalized_alpha_{i}"])
        best_alphas = [a / sum(best_alphas) for a in best_alphas]

        best_params = {
            "n_components": study.best_params["n_components"],
            "lam": study.best_params["lam"],
            "alpha": best_alphas,
            "flexible_label_matrix": self.flexible_label_matrix,
            "class_imbalance": self.class_imbalance,
            "batch_size": self.batch_size,
            "max_iter": self.max_iter,
            "eps": self.eps,
            "device": self.device,
            "dtype": self.dtype,
            "random_state": self.random_state,
            "tol": self.tol,
            "shuffle": self.shuffle,
            "num_workers": self.num_workers,
        }

        self.best_params = best_params
        self.study_df = study.trials_dataframe().sort_values(by="value", ascending=True)
        self.study = study

        return best_params

    def get_f1_report(self, top_n=20):
        report_data = []
        for cov in self.covariates:
            column = self.study_df[f"user_attrs_f1_score_{cov}"][0:top_n]
            report_data.append(
                {
                    "covariate": cov,
                    "mean": column.mean(),
                    "std": column.std(),
                    "min": column.min(),
                    "max": column.max(),
                    "median": column.median(),
                }
            )
        return pd.DataFrame(report_data)

    def __repr__(self):
        return f"Optimizer(model_class={self.model_class.__name__}, n_trials={self.n_trials}, n_cv_splits={self.n_cv_splits}, flexible_label_matrix={self.flexible_label_matrix}, class_imbalance={self.class_imbalance}, batch_size={self.batch_size}, max_iter={self.max_iter}, device={self.device}, random_state={self.random_state}, eps={self.eps}, tol={self.tol}, dtype={self.dtype}, shuffle={self.shuffle}, num_workers={self.num_workers})"

    def __str__(self):
        return f"Optimizer(model_class={self.model_class.__name__}, n_trials={self.n_trials}, n_cv_splits={self.n_cv_splits}, flexible_label_matrix={self.flexible_label_matrix}, class_imbalance={self.class_imbalance}, batch_size={self.batch_size}, max_iter={self.max_iter}, device={self.device}, random_state={self.random_state}, eps={self.eps}, tol={self.tol}, dtype={self.dtype}, shuffle={self.shuffle}, num_workers={self.num_workers})"
