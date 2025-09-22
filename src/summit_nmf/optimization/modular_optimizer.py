import torch
import optuna
import pandas as pd
import numpy as np

from sklearn.metrics import f1_score
from sklearn.model_selection import KFold
from typing import Optional, Tuple, Dict, Callable
from summit_nmf.modular_model import ModularSummitNMF

class ModularSummitOptimizer:
    def __init__(
        self,
        n_components_range: Tuple[int, int]=(5, 50),
        lam_range: Tuple[float, float]=(1e-0, 1e5),
        n_folds: int = 3,
        weighted_sampling: bool = False,
        batch_size: Optional[int] = None,
        max_iter: int = 200,
        device: str = "cpu",
        random_state: int = 42,
        eps: float = 1e-10,
        tol: float = 1e-4,
        dtype: torch.dtype = torch.float32,
        save_loss_interval: int = 5,
        verbose: bool = False,
        score_func: Optional[Callable] = None,
        n_trials: int = 100
    ):
        self.n_components_range = n_components_range
        self.lam_range = lam_range
        self.n_folds = n_folds
        self.n_trials = n_trials
        self._validate_params()

        if score_func is not None:
            self.score_func = score_func
        else:
            self.score_func = self.compute_score

        # parameters for the model
        self.params = {
            "weighted_sampling": weighted_sampling,
            "batch_size": batch_size,
            "max_iter": max_iter,
            "device": device,
            "random_state": random_state,
            "eps": eps,
            "tol": tol,
            "dtype": dtype,
            "save_loss_interval": save_loss_interval,
            "verbose": verbose
        }

    def search(self, X: torch.Tensor, y: pd.DataFrame) -> Dict:
        self._validate_inputs(X, y)

        covariate_keys = y.columns.tolist()
        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=self.params["random_state"])

        def objective(trial):
            params = self.params.copy()

            # sampling components and lambda
            n_components = trial.suggest_int(
                "n_components", self.n_components_range[0], self.n_components_range[1]
            )
            lam = trial.suggest_float("lam", self.lam_range[0], self.lam_range[1], log=True)
            
            # sampling alphas
            n_alphas = len(covariate_keys)
            alpha_raw = [
                -np.log(
                    trial.suggest_float(f"unnormalized_alpha_{i}", 0, 1)
                )
                for i in range(n_alphas)
            ]
            alpha_sum = sum(alpha_raw)
            alphas = [a / alpha_sum for a in alpha_raw]

            for i in range(n_alphas):
                trial.set_user_attr(f"alpha_{i}", alphas[i])

            params.update({"n_components": n_components, "lam": lam, "alpha": alphas})


            # store scores for each covariate
            cov_scores = np.zeros(len(covariate_keys))

            for train_index, test_index in kf.split(X.numpy()):
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y.iloc[train_index], y.iloc[test_index]

                # fit model
                model = ModularSummitNMF(**params)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                # compute per-covariate scores
                for cov in range(y_test.shape[1]):
                    cov_score = self.score_func(y_test.iloc[:, cov], y_pred.iloc[:, cov])
                    cov_scores[cov] += cov_score

            # average over folds
            cov_scores /= self.n_folds
            # return as tuple → multi-objective
            return tuple(float(s) for s in cov_scores)

        
        # create multi-objective study
        study = optuna.create_study(
            sampler=optuna.samplers.NSGAIISampler(),  # NSGA-II works well for multi-objective
            directions=["maximize"] * len(covariate_keys),
            study_name="multi-covariates optimization (SummitNMF)",
        )
        study.optimize(objective, n_trials=self.n_trials)

        # extract best trial(s) — Pareto front
        pareto_trials = study.best_trials

        # store results
        self.study_df = study.trials_dataframe().sort_values(by="number", ascending=True)
        self.study = study

        # pick one solution from Pareto front (e.g. best balance)
        best_trial = pareto_trials[0]  # or implement custom selection strategy
        best_params = self.params.copy()
        best_params["n_components"] = best_trial.params["n_components"]
        best_params["lam"] = best_trial.params["lam"]

        # retrieve alphas
        n_alphas = len(covariate_keys)
        alphas = [best_trial.user_attrs[f"alpha_{i}"] for i in range(n_alphas)]
        best_params["alpha"] = alphas
        self.best_params = best_params

        return best_params

    def compute_score(self, y_true, y_pred):
        return f1_score(y_true, y_pred, average="weighted")

    def _validate_inputs(self, X: torch.Tensor, y: pd.DataFrame):
        if not isinstance(X, torch.Tensor):
            raise ValueError("X must be a torch.Tensor.")
        if not isinstance(y, pd.DataFrame):
            raise ValueError("y must be a pandas DataFrame.")
        if X.ndim != 2:
            raise ValueError("X must be a 2D tensor.")
        if X.size(0) != y.shape[0]:
            raise ValueError("Number of samples in X and y must match.")
        if X.min() < 0:
            raise ValueError("X must be non-negative.")

    def _validate_params(self):
        if not (isinstance(self.n_components_range, tuple) and len(self.n_components_range) == 2):
            raise ValueError("n_components_range must be a tuple of two integers.")
        if not (isinstance(self.lam_range, tuple) and len(self.lam_range) == 2):
            raise ValueError("lam_range must be a tuple of two floats.")
        if self.n_components_range[0] >= self.n_components_range[1]:
            raise ValueError("In n_components_range, the first value must be less than the second value.")
        if self.lam_range[0] >= self.lam_range[1]:
            raise ValueError("In lam_range, the first value must be less than the second value.")
        if not isinstance(self.n_folds, int) or self.n_folds < 2:
            raise ValueError("n_folds must be an integer greater than 1.")