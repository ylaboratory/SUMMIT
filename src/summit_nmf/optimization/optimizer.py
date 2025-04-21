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
        self.max_iter = max_iter
        self.flexible_label_matrix = flexible_label_matrix
        self.class_imbalance = class_imbalance
        self.batch_size = batch_size
        self.n_trials = n_trials
        self.n_cv_splits = n_cv_splits
        self.random_state = random_state
        self.eps = eps
        self.tol = tol
        self.device = device
        self.dtype = dtype
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.weight_n_comp = weight_n_comp

        params = {
            "max_iter": self.max_iter,
            "flexible_label_matrix": self.flexible_label_matrix,
            "class_imbalance": self.class_imbalance,
            "batch_size": self.batch_size,
            "n_trials": self.n_trials,
            "n_cv_splits": self.n_cv_splits,
            "random_state": self.random_state,
            "eps": self.eps,
            "tol": self.tol,
            "device": self.device,
            "dtype": self.dtype,
            "shuffle": self.shuffle,
            "num_workers": self.num_workers,
            "weight_n_comp": self.weight_n_comp,
        }

        # Initialize the appropriate optimizer based on the model type
        if self.model_type in ["unified", "u"]:
            self.optimizer = UOptimizer(model_class=UnifiedSummitNMF, **params)

        elif self.model_type in ["separated", "s"]:
            self.optimizer = SOptimizer(model_class=SeparatedSummitNMF, **params)

        elif self.model_type in ["fast_unified", "fu"]:
            UserWarning(
                "Using FastSeparatedSummitNMF, please make sure the dataset can be loaded into device's memory. The batch_size is set to the entire dataset."
            )
            self.optimizer = UOptimizer(model_class=FastUnifiedSummitNMF, **params)

        elif self.model_type in ["fast_separated", "fs"]:
            UserWarning(
                "Using FastSeparatedSummitNMF, please make sure the dataset can be loaded into device's memory. The batch_size is set to the entire dataset."
            )
            self.optimizer = SOptimizer(model_class=FastSeparatedSummitNMF, **params)

        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

    def search(
        self, X, y, n_components_range=(5, 50), lam_range=(1e-5, 1e0)
    ) -> Dict[str, Union[str, int, float]]:
        print(
            "Suggestion for the best hyperparameters: please balance the loss between harmonic and reconstruction by adjusting the lambda to have similar range."
        )
        self.covariates = list(y.keys())
        self.best_params = self.optimizer.search(X, y, n_components_range, lam_range)
        self.study_df = self.optimizer.study_df.sort_values(by="value", ascending=True)
        self.study = self.optimizer.study

        return self.best_params
    
    def get_prediction_report(self, top_n=20):
        report_data = []
        for cov in self.covariates:
            column = self.study_df[f"user_attrs_f1_score_{cov}"][0:top_n]
            report_data.append({
                "covariate": cov,
                "mean": column.mean(),
                "std": column.std(),
                "min": column.min(),
                "max": column.max(),
                "median": column.median(),
            })
        return pd.DataFrame(report_data)

class UOptimizer:
    def __init__(
        self,
        model_class,
        n_trials,
        n_cv_splits,
        flexible_label_matrix,
        class_imbalance,
        batch_size,
        max_iter,
        device,
        random_state,
        eps,
        tol,
        dtype,
        shuffle,
        num_workers,
        weight_n_comp,
    ):
        self.model_class = model_class
        self.max_iter = max_iter
        self.flexible_label_matrix = flexible_label_matrix
        self.class_imbalance = class_imbalance
        self.batch_size = batch_size
        self.n_trials = n_trials
        self.n_cv_splits = n_cv_splits
        self.random_state = random_state
        self.eps = eps
        self.tol = tol
        self.device = device
        self.dtype = dtype
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.weight_n_comp = weight_n_comp

    def __repr__(self):
        return f"Optimizer(model_class={self.model_class.__name__}, n_trials={self.n_trials}, n_cv_splits={self.n_cv_splits}, flexible_label_matrix={self.flexible_label_matrix}, class_imbalance={self.class_imbalance}, batch_size={self.batch_size}, max_iter={self.max_iter}, device={self.device}, random_state={self.random_state}, eps={self.eps}, tol={self.tol}, dtype={self.dtype}, shuffle={self.shuffle}, num_workers={self.num_workers})"

    def __str__(self):
        return f"Optimizer(model_class={self.model_class.__name__}, n_trials={self.n_trials}, n_cv_splits={self.n_cv_splits}, flexible_label_matrix={self.flexible_label_matrix}, class_imbalance={self.class_imbalance}, batch_size={self.batch_size}, max_iter={self.max_iter}, device={self.device}, random_state={self.random_state}, eps={self.eps}, tol={self.tol}, dtype={self.dtype}, shuffle={self.shuffle}, num_workers={self.num_workers})"

    def search(self, X, y, n_components_range=(5, 50), lam_range=(1e-5, 1e0)):
        def objective(trial):
            n_components = trial.suggest_int(
                "n_components", n_components_range[0], n_components_range[1]
            )
            lam = trial.suggest_float("lam", lam_range[0], lam_range[1], log=True)

            params = {
                "n_components": n_components,
                "lam": lam,
                "max_iter": self.max_iter,
                "flexible_label_matrix": self.flexible_label_matrix,
                "class_imbalance": self.class_imbalance,
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


            # y_array = np.array(list(y.values())).T  # Convert y dictionary to a 2D array for splitting

            for train_index, test_index in kf.split(X):
                X_train, X_test = X[train_index], X[test_index]
                y_train = {key: labels[train_index] for key, labels in y.items()}
                y_test = {key: labels[test_index] for key, labels in y.items()}

                # null_params = deepcopy(params)
                # null_params["lam"] = 0
                # null_model = self.model_class(**null_params)
                # null_model.fit(X_train, y_train)
                # null_recon = np.linalg.norm(
                #     X_test - null_model.transform(X_test) @ null_model.H, ord="fro"
                # )

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
                    score += lam * f1_score(
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

            harmonic_mean = np.mean(harmonic_losses)
            recon_mean = np.mean(recon_losses)

            f1_scores = [
                f1 / self.n_cv_splits for f1 in f1_scores
            ]
            for i, key in enumerate(y_pred.keys()):
                trial.set_user_attr(f"f1_score_{key}", f1_scores[i])

            return recon_mean - harmonic_mean + self.weight_n_comp * n_components

        study = optuna.create_study(
            sampler=optuna.samplers.TPESampler(),
            pruner=optuna.pruners.MedianPruner(),
            direction="minimize",
            study_name="multi-covariates optimization (UnifiedSummitNMF)",
        )
        study.optimize(objective, n_trials=self.n_trials)

        best_params = {
            "n_components": study.best_params["n_components"],
            "lam": study.best_params["lam"],
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

        self.study = study
        self.study_df = study.trials_dataframe()
        return best_params


class SOptimizer:
    def __init__(
        self,
        model_class,
        n_trials,
        n_cv_splits,
        flexible_label_matrix,
        class_imbalance,
        batch_size,
        max_iter,
        device,
        random_state,
        eps,
        tol,
        dtype,
        shuffle,
        num_workers,
        weight_n_comp,
    ):
        self.model_class = model_class
        self.max_iter = max_iter
        self.flexible_label_matrix = flexible_label_matrix
        self.class_imbalance = class_imbalance
        self.batch_size = batch_size
        self.n_trials = n_trials
        self.n_cv_splits = n_cv_splits
        self.random_state = random_state
        self.eps = eps
        self.tol = tol
        self.device = device
        self.dtype = dtype
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.weight_n_comp = weight_n_comp

    def __repr__(self):
        return f"Optimizer(model_class={self.model_class.__name__}, n_trials={self.n_trials}, n_cv_splits={self.n_cv_splits}, flexible_label_matrix={self.flexible_label_matrix}, class_imbalance={self.class_imbalance}, batch_size={self.batch_size}, max_iter={self.max_iter}, device={self.device}, random_state={self.random_state}, eps={self.eps}, tol={self.tol}, dtype={self.dtype}, shuffle={self.shuffle}, num_workers={self.num_workers})"

    def __str__(self):
        return f"Optimizer(model_class={self.model_class.__name__}, n_trials={self.n_trials}, n_cv_splits={self.n_cv_splits}, flexible_label_matrix={self.flexible_label_matrix}, class_imbalance={self.class_imbalance}, batch_size={self.batch_size}, max_iter={self.max_iter}, device={self.device}, random_state={self.random_state}, eps={self.eps}, tol={self.tol}, dtype={self.dtype}, shuffle={self.shuffle}, num_workers={self.num_workers})"

    def search(self, X, y, n_components_range=(5, 50), lam_range=(1e-3, 1e3)):
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

                # null_params = deepcopy(params)
                # null_params["lam"] = 0
                # null_model = self.model_class(**null_params)
                # null_model.fit(X_train, y_train)
                # null_recon = np.linalg.norm(
                #     X_test - null_model.transform(X_test) @ null_model.H, ord="fro"
                # )

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
                    score += (
                        lam
                        * alphas[i]
                        * f1_score(
                            y_test[key][non_nan_indices],
                            y_pred[key][non_nan_indices],
                            average="weighted",
                        )
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
                # recon_losses.append(null_recon / recon_loss)
                recon_losses.append(recon_loss)

            f1_scores = [
                f1 / self.n_cv_splits for f1 in f1_scores
            ]
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

        self.study = study
        self.study_df = study.trials_dataframe()
        return best_params
