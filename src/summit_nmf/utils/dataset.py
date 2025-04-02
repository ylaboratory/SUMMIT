import torch
from torch.utils.data import Dataset
from typing import List

class hybrid_dataset(Dataset):
    def __init__(
        self,
        X: torch.Tensor,
        Ys: List[torch.Tensor],
        W: torch.Tensor,
        M: torch.Tensor,
        Ls: List[torch.Tensor],
        device: str,
    ):
        self.X = X  # Keep on CPU
        self.Ys = [Y.to(device) for Y in Ys]  # Preload smaller tensors
        self.W = W.to(device)
        self.M = M.to(device)
        self.Ls = [L.to(device) for L in Ls]
        self.device = device

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        X_sample = self.X[idx]  # CPU tensor
        W_sample = self.W[idx]  # GPU tensor
        M_sample = self.M[idx]  # GPU tensor
        Ys_sample = [Y[idx] for Y in self.Ys]  # GPU tensors
        Ls_sample = [L[idx] for L in self.Ls]  # GPU tensors
        return (X_sample, W_sample, M_sample, Ys_sample, Ls_sample), idx

    def update(self, indices, W, Ys):
        self.W[indices] = W.to(self.device)
        for i, Y in enumerate(Ys):
            self.Ys[i][indices] = Y.to(self.Ys[i].device)

    def get_all_matrices(self):
        X_copy = self.X.clone()
        W_copy = self.W.clone().cpu()
        M_copy = self.M.clone().cpu()
        Ys_copy = [Y.clone().cpu() for Y in self.Ys]
        Ls_copy = [L.clone().cpu() for L in self.Ls]

        return X_copy, W_copy, M_copy, Ys_copy, Ls_copy