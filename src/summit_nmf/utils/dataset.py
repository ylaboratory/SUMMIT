import torch
from torch.utils.data import Dataset
from typing import List


class preload_dataset(Dataset):
    def __init__(
        self,
        X: torch.Tensor,
        Ys: List[torch.Tensor],
        W: torch.Tensor,
        M: torch.Tensor,
        Ls: List[torch.Tensor],
        device: str,
    ):
        self.X = X.to(device)
        self.Ys = [Y.to(device) for Y in Ys]
        self.W = W.to(device)
        self.M = M.to(device)
        self.Ls = [L.to(device) for L in Ls]
        self.device = device

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        X_sample = self.X[idx]
        W_sample = self.W[idx]
        M_sample = self.M[idx]
        Ys_sample = [Y[idx] for Y in self.Ys]
        Ls_sample = [L[idx] for L in self.Ls]

        return [X_sample, W_sample, M_sample, Ys_sample, Ls_sample], idx

    def update(
        self,
        indices,
        W: torch.Tensor,
        Ys: List[torch.Tensor],
    ):
        W = torch.as_tensor(W).to(self.W.device)
        self.W[indices] = W

        for i, Y in enumerate(Ys):
            Y = torch.as_tensor(Y).to(self.Ys[i].device)
            self.Ys[i][indices] = Y

    def get_all_matrices(self):
        X_copy = self.X.clone().cpu()
        W_copy = self.W.clone().cpu()
        M_copy = self.M.clone().cpu()
        Ys_copy = [Y.clone().cpu() for Y in self.Ys]
        Ls_copy = [L.clone().cpu() for L in self.Ls]

        return X_copy, W_copy, M_copy, Ys_copy, Ls_copy


class swap_dataset(Dataset):
    def __init__(
        self,
        X: torch.Tensor,
        Ys: List[torch.Tensor],
        W: torch.Tensor,
        M: torch.Tensor,
        Ls: List[torch.Tensor],
        device: str,
    ):
        self.X = X
        self.Ys = Ys
        self.W = W
        self.M = M
        self.Ls = Ls
        self.device = device

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        X_sample = self.X[idx]
        W_sample = self.W[idx]
        M_sample = self.M[idx]
        Ys_sample = [Y[idx] for Y in self.Ys]
        Ls_sample = [L[idx] for L in self.Ls]

        return [X_sample, W_sample, M_sample, Ys_sample, Ls_sample], idx

    def update(
        self,
        indices,
        W: torch.Tensor,
        Ys: List[torch.Tensor],
    ):
        W = torch.as_tensor(W).to(self.W.device)
        self.W[indices] = W

        for i, Y in enumerate(Ys):
            Y = torch.as_tensor(Y).to(self.Ys[i].device)
            self.Ys[i][indices] = Y

    def get_all_matrices(self):
        X_copy = self.X.clone().cpu()
        W_copy = self.W.clone().cpu()
        M_copy = self.M.clone().cpu()
        Ys_copy = [Y.clone().cpu() for Y in self.Ys]
        Ls_copy = [L.clone().cpu() for L in self.Ls]

        return X_copy, W_copy, M_copy, Ys_copy, Ls_copy
