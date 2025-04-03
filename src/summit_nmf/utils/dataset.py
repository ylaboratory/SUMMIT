import torch
from torch.utils.data import Dataset

class summit_dataset(Dataset):
    def __init__(
        self,
        X: torch.Tensor,
        M: torch.Tensor,
    ):
        self.X = X 
        self.M = M

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        X_sample = self.X[idx]  # CPU tensor
        M_sample = self.M[idx]  # GPU tensor

        return X_sample, M_sample, idx

    def get_all_matrices(self):
        X_ref = self.X
        M_ref = self.M

        return X_ref, M_ref