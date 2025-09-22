import torch
import itertools
import pandas as pd
from summit_nmf import UnifiedSummitNMF
from sklearn.model_selection import KFold


class ModelSelection:
    def __init__(self, model: UnifiedSummitNMF, n_folds: int = 3, random_state: int = 42):
        if model.__class__.__name__ != "UnifiedSummitNMF":
            raise ValueError("Model must be of class 'UnifiedSummitNMF'")

        self.model = model
        self.n_folds = n_folds
        self.random_state = random_state

    def select(self, X: torch.Tensor, y: pd.DataFrame):
        # get all combinations of covariates
        self.covariate_keys = y.columns.tolist()
        all_combinations = []
        for r in range(1, len(self.covariate_keys) + 1):
            all_combinations.extend(itertools.combinations(self.covariate_keys, r))

        # perform k-fold cross-validation
        kf = KFold(n_splits=self.n_folds, random_state=self.random_state, shuffle=True)

        combination_results = []
        for combination in all_combinations:
            # subset y to only include the current combination of covariates
            subset_y = y[list(combination)]
            scores = []
            for train_index, val_index in kf.split(X.numpy()):
                X_train, X_val = X[train_index], X[val_index]
                y_train, y_val = subset_y.iloc[train_index], subset_y.iloc[val_index]

                self.model = self.model.fit(X_train, y_train)
                H = self.model.H
                W_val = self.model.transform(X_val)

                reconsturction_loss = (torch.norm(X_val - torch.matmul(W_val, H), p='fro')**2).item()
                scores.append(reconsturction_loss)
            combination_results.append([combination, sum(scores) / self.n_folds])

        self.results_df = pd.DataFrame(combination_results, columns=['Covariates', 'Avg_Reconstruction_Loss'])
        self.results_df = self.results_df.sort_values(by='Avg_Reconstruction_Loss').reset_index(drop=True)
        return self.results_df