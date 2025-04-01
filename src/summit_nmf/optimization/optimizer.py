import numpy as np
import optuna
import optunahub
import torch

from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold, KFold
from summit_nmf.modes.seperated_summit_nmf import SeparatedSummitNMF
from summit_nmf.modes.unified_summit_nmf import UnifiedSummitNMF
from copy import deepcopy
from typing import Dict, Union

class Optimizer:

    def __init__(
        self,
        model_type,
        max_iter=200,
        flexible_label_matrix=False,
        class_imbalance=False,
        n_trials=50,
        n_cv_splits=3,
        random_state=None,
        eps=1e-10,
        tol=1e-4,
        device=None,
        preload=False,
        dtype=torch.float32
    ):
        self.model_type = model_type
        self.max_iter = max_iter
        self.flexible_label_matrix = flexible_label_matrix
        self.class_imbalance = class_imbalance
        self.n_trials = n_trials
        self.n_cv_splits = n_cv_splits
        self.random_state = random_state
        self.eps = eps
        self.tol = tol
        self.device = device
        self.preload = preload
        self.dtype = dtype

        params = {
            "max_iter": self.max_iter,
            "flexible_label_matrix": self.flexible_label_matrix,
            "class_imbalance": self.class_imbalance,
            "n_trials": self.n_trials,
            "n_cv_splits": self.n_cv_splits,
            "random_state": self.random_state,
            "eps": self.eps,
            "tol": self.tol,
            "device": self.device,
            "preload": self.preload,
            "dtype": self.dtype
        }

        # Initialize the appropriate optimizer based on the model type
        if self.model_type == "Unified":
            self.optimizer = UOptimizer(**params)
        elif self.model_type == "Seperated":
            self.optimizer = SOptimizer(**params)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

    def search(self, X, y, n_components_range=(5, 50), lam_range=(1e-5, 1e+0)) -> Dict[str, Union[str, int, float]]:
        """
        Calls the `search` method of the appropriate optimizer.
        """
        print("Suggestion for the best hyperparameters: please balance the loss between harmonic and reconstruction by adjusting the lambda to have similar range.")
        return self.optimizer.search(X, y, n_components_range, lam_range)

class UOptimizer ():

    def __init__ (
        self,
        max_iter=200,
        flexible_label_matrix=False,
        class_imbalance=False,
        n_trials=100,
        n_cv_splits=3,
        random_state=None,
        eps=1e-10,
        tol=1e-4,
        device=None,
        dtype=torch.float32
    ):

        self.max_iter = max_iter
        self.flexible_label_matrix = flexible_label_matrix
        self.class_imbalance = class_imbalance
        self.n_trials = n_trials
        self.n_cv_splits = n_cv_splits
        self.random_state = random_state
        self.eps = eps
        self.tol = tol
        self.device = device
        self.dtype = dtype

    def search(self, X, y, n_components_range=(5, 50), lam_range=(1e-5, 1e+0)):

        def objective(trial):
            n_components = trial.suggest_int("n_components", n_components_range[0], n_components_range[1])
            lam = trial.suggest_float("lam", lam_range[0], lam_range[1], log=True)


            params = {
                "n_components": n_components,
                "lam": lam,
                "max_iter": self.max_iter,
                "flexible_label_matrix": self.flexible_label_matrix,
                "class_imbalance": self.class_imbalance,
                "eps": self.eps,
                "device": self.device,
                "preload": self.preload,
                "dtype": self.dtype,
                "random_state": self.random_state,
                "tol": self.tol,
            }
            
            kf = KFold(n_splits=self.n_cv_splits, shuffle=True, random_state=self.random_state)
            harmonic_losses = []
            recon_losses = []

            # y_array = np.array(list(y.values())).T  # Convert y dictionary to a 2D array for splitting

            for train_index, test_index in kf.split(X):
                X_train, X_test = X[train_index], X[test_index]
                y_train = {key: labels[train_index] for key, labels in y.items()}
                y_test = {key: labels[test_index] for key, labels in y.items()}

                null_params = deepcopy(params)
                null_params['lam'] = 0
                null_model = UnifiedSummitNMF(**null_params)
                null_model.fit(X_train, y_train)
                null_recon = np.linalg.norm(X_test - null_model.transform(X_test) @ null_model.H, ord="fro")


                model = UnifiedSummitNMF(**params)
                model.fit(X_train, y_train)

                y_pred = model.predict(X_test)
                score = 0
                for key in y_test.keys():
                    score += f1_score(y_test[key], y_pred[key], average="weighted")
                score /= len(y_test)
                harmonic_losses.append(score)

                recon_loss = np.linalg.norm(X_test - model.transform(X_test) @ model.H, ord="fro")
                recon_losses.append(null_recon / recon_loss)


            harmonic_mean = np.mean(harmonic_losses)
            recon_mean = np.mean(recon_losses)

            return harmonic_mean + recon_mean

        sampler = optunahub.load_module("samplers/auto_sampler").AutoSampler()
        study = optuna.create_study(
            sampler=sampler,
            direction="maximize",
            study_name="multi-covariates optimization (UHNMF)",
        )
        study.optimize(objective, n_trials=self.n_trials)
        self.study = study
        self.study_df = study.trials_dataframe()


        best_params = {
            "n_components": self.study.best_params["n_components"],
            "lam": self.study.best_params["lam"],
            "flexible_label_matrix": self.flexible_label_matrix,
            "class_imbalance": self.class_imbalance,
            "max_iter": self.max_iter,
            "eps": self.eps,
            "device": self.device,
            "preload": self.preload,
            "dtype": self.dtype,
            "random_state": self.random_state,
            "tol": self.tol,
        }
        return best_params

class SOptimizer ():

    def __init__ (
        self,
        max_iter=200,
        flexible_label_matrix=False,
        class_imbalance=False,
        n_trials=100,
        n_cv_splits=3,
        random_state=None,
        eps=1e-10,
        tol=1e-4,
        device=None,
        dtype=torch.float32
    ):

        self.max_iter = max_iter
        self.flexible_label_matrix = flexible_label_matrix
        self.class_imbalance = class_imbalance
        self.n_trials = n_trials
        self.n_cv_splits = n_cv_splits
        self.random_state = random_state
        self.eps = eps
        self.tol = tol
        self.device = device
        self.dtype = dtype

    def search(self, X, y, n_components_range=(5, 50), lam_range=(1e-5, 1e+0)):

        def objective(trial):
            n_components = trial.suggest_int("n_components", n_components_range[0], n_components_range[1])
            lam = trial.suggest_float("lam", lam_range[0], lam_range[1], log=True)
            
            num_params = len(y)
            param_names = [f"alpha_{i}" for i in range(num_params)] 

            # Suggest the first (num_params - 1) parameters
            alphas = []
            remaining = 1.0
            for i in range(num_params - 1):
                alpha = trial.suggest_float(param_names[i], 0.0, remaining)
                alphas.append(alpha)
                remaining -= alpha  # Ensure sum ≤ 1

            # Compute the last parameter to ensure sum = 1
            alphas.append(remaining)

            # Store the last parameter manually
            trial.set_user_attr(param_names[-1], alphas[-1])

            params = {
                "n_components": n_components,
                "lam": lam,
                "alpha": alphas,
                "max_iter": self.max_iter,
                "flexible_label_matrix": self.flexible_label_matrix,
                "class_imbalance": self.class_imbalance,
                "eps": self.eps,
                "device": self.device,
                "preload": self.preload,
                "dtype": self.dtype,
                "random_state": self.random_state,
                "tol": self.tol,
            }
            
            kf = KFold(n_splits=self.n_cv_splits, shuffle=True, random_state=self.random_state)
            harmonic_losses = []
            recon_losses = []

            # y_array = np.array(list(y.values())).T  # Convert y dictionary to a 2D array for splitting

            for train_index, test_index in kf.split(X):
                X_train, X_test = X[train_index], X[test_index]
                y_train = {key: labels[train_index] for key, labels in y.items()}
                y_test = {key: labels[test_index] for key, labels in y.items()}

                null_params = deepcopy(params)
                null_params['lam'] = 0
                null_model = SeparatedSummitNMF(**null_params)
                null_model.fit(X_train, y_train)
                null_recon = np.linalg.norm(X_test - null_model.transform(X_test) @ null_model.H, ord="fro")


                model = SeparatedSummitNMF(**params)
                model.fit(X_train, y_train)

                y_pred = model.predict(X_test)
                score = 0
                for key in y_test.keys():
                    score += f1_score(y_test[key], y_pred[key], average="weighted")
                score /= len(y_test)
                harmonic_losses.append(score)

                recon_loss = np.linalg.norm(X_test - model.transform(X_test) @ model.H, ord="fro")
                recon_losses.append(null_recon / recon_loss)


            harmonic_mean = np.mean(harmonic_losses)
            recon_mean = np.mean(recon_losses)

            return harmonic_mean + recon_mean

        sampler = optunahub.load_module("samplers/auto_sampler").AutoSampler()
        study = optuna.create_study(
            sampler=sampler,
            direction="maximize",
            study_name="multi-covariates optimization (SHNMF)",
        )
        study.optimize(objective, n_trials=self.n_trials)
        self.study = study
        self.study_df = study.trials_dataframe()

        alphas = []
        for i in range(len(y)-1):
            alphas.append(self.study.best_params[f"alpha_{i}"])
        alphas.append(1.0 - float(np.sum(alphas, axis=0)))

        best_params = {
            "n_components": self.study.best_params["n_components"],
            "lam": self.study.best_params["lam"],
            "alpha": alphas,
            "flexible_label_matrix": self.flexible_label_matrix,
            "class_imbalance": self.class_imbalance,
            "max_iter": self.max_iter,
            "eps": self.eps,
            "device": self.device,
            "preload": self.preload,
            "dtype": self.dtype,
            "random_state": self.random_state,
            "tol": self.tol,
        }
        return best_params